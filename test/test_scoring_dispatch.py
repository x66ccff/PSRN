"""Run with python -m unittest discover -s test -p test_scoring_dispatch.py.

Loads only model code so the scoring tests do not require regressor extras.
"""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

PACKAGE = "_psrn_scoring_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "psrn")]
sys.modules.setdefault(PACKAGE, package)
PSRN = importlib.import_module(PACKAGE + ".model.models").PSRN


def original_scores(net, x, y):
    with torch.no_grad():
        sums = torch.zeros((1, net.out_dim), device=net.device)
        for row in range(x.shape[0]):
            prediction = net.forward(x[row].reshape(1, -1))
            target = y[row]
            if target.dim() > 1:
                target = target.reshape(1)
            diff = prediction - target
            sums += diff ** 2
        scores = (sums / x.shape[0]).reshape(-1)
        scores[~torch.isfinite(scores)] = float("inf")
        return scores


class ScoringDispatchTests(unittest.TestCase):
    def test_budget_boundary_routes_and_exact_scores(self):
        net = PSRN(2, ["Add", "Sub", "Identity"], 1, device="cpu")
        for samples, chunk in ((1, 16), (16, 16), (17, 16), (100, 16),
                               (3, 2), (1, 1), (17, 32)):
            with self.subTest(samples=samples, chunk=chunk):
                x = torch.arange(samples * 2, dtype=torch.float32).reshape(samples, 2) / 7
                y = torch.linspace(-1, 1, samples).reshape(samples, 1)
                expected = original_scores(net, x, y)
                bound = net.estimate_dense_score_bytes(x, y)
                groups = (samples + min(chunk, samples) - 1) // min(chunk, samples)
                # Smallest integer base budget admitting the dense estimate.
                scale = chunk * groups
                budget = (bound * samples + scale - 1) // scale
                with patch.object(net.list[-1], "iter_tiles", wraps=net.list[-1].iter_tiles) as tiles:
                    actual = net.score_mse(x, y, sample_chunk_size=chunk,
                                           workspace_budget_bytes=budget)
                    self.assertEqual(net.last_score_strategy, "dense")
                    tiles.assert_not_called()
                    self.assertTrue(torch.equal(actual, expected))
                    actual = net.score_mse(x, y, tile_size=3, sample_chunk_size=chunk,
                                           workspace_budget_bytes=budget - 1)
                    self.assertEqual(net.last_score_strategy, "tiled")
                    self.assertGreater(tiles.call_count, 0)
                    self.assertTrue(torch.equal(actual, expected))

    def test_small_queries_use_soft_allowance_above_base_budget(self):
        net = PSRN(1, ["Identity"], 1, device="cpu")
        base = 512 * 1024 ** 2
        # Exercise a just-over-base estimate without allocating a huge model.
        with patch.object(net, "estimate_dense_score_bytes", return_value=base + 1):
            for samples, expected_strategy in ((1, "dense"), (16, "tiled"),
                                               (17, "dense"), (100, "dense")):
                x = torch.ones(samples, 1)
                y = torch.zeros(samples)
                actual = net.score_mse(x, y)
                self.assertEqual(net.last_score_strategy, expected_strategy)
                self.assertTrue(torch.equal(actual, original_scores(net, x, y)))

    def test_scalar_promotion_changes_estimate_and_routing(self):
        net = PSRN(2, ["Identity"], 1, device="cpu")
        x = torch.tensor([[.5, 1.25], [-.75, 2.5]], dtype=torch.float32).repeat(8, 1)
        scalar_targets = torch.linspace(-1, 1, 32, dtype=torch.float64)[::2]
        for shape, expected_bound, expected_strategy in (
                ((16,), 32, "dense"), ((16, 1), 48, "tiled"),
                ((16, 1, 1, 1), 48, "tiled")):
            y = scalar_targets.reshape(shape)
            with self.subTest(shape=shape):
                self.assertEqual(net.estimate_dense_score_bytes(x, y), expected_bound)
                expected = original_scores(net, x, y)
                actual = net.score_mse(x, y, workspace_budget_bytes=32)
                self.assertEqual(net.last_score_strategy, expected_strategy)
                self.assertTrue(torch.equal(actual, expected))
                for strategy in ("dense", "tiled"):
                    actual = net.score_mse(x, y, strategy=strategy, tile_size=1)
                    self.assertTrue(torch.equal(actual, expected))

    def test_estimator_observes_module_replacement_and_default_dtype(self):
        net = PSRN(1, ["Identity"], 1, device="cpu")
        x, y = torch.ones(16, 1, dtype=torch.float32), torch.zeros(16, dtype=torch.float32)
        original_bound = net.estimate_dense_score_bytes(x, y)
        self.assertEqual(original_bound, 16)
        net.to(dtype=torch.float64)
        self.assertEqual(net.estimate_dense_score_bytes(x.double(), y.double()), 28)
        net.list[-1] = PSRN(1, ["Add"], 1, device="cpu").list[-1]
        self.assertEqual(net.estimate_dense_score_bytes(x, y), 28)
        actual = net.score_mse(x, y, workspace_budget_bytes=original_bound)
        self.assertEqual(net.last_score_strategy, "tiled")
        self.assertTrue(torch.equal(actual, original_scores(net, x, y)))
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertEqual(net.estimate_dense_score_bytes(x, y), 32)
            actual = net.score_mse(x, y, strategy="dense")
            self.assertEqual(actual.dtype, torch.float64)
            self.assertTrue(torch.equal(actual, original_scores(net, x, y)))
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_forced_tiles_mask_noncontiguous_fp64_and_target_shapes(self):
        operators = ["Add", "SemiSub", "SemiDiv", "Identity", "Sin", "Inv", "Log"]
        pre = PSRN(2, operators, 1, device="cpu")
        mask = torch.arange(pre.out_dim) % 3 != 0
        net = PSRN(2, operators, 2, dr_mask=mask, device="cpu")
        x = torch.tensor([[0., 9., -1., 9.], [2., 9., 3., 9.],
                          [-2., 9., 0.5, 9.]], dtype=torch.float64)[:, ::2]
        for target_dtype in (torch.float32, torch.float64):
            for shape in ((3,), (3, 1), (3, 1, 1), (3, 1, 1, 1)):
                y = torch.tensor([1., 0., -1.], dtype=target_dtype).reshape(shape)
                expected = original_scores(net, x, y)
                for strategy in ("dense", "tiled", "auto"):
                    actual = net.score_mse(x, y, tile_size=11, sample_chunk_size=2,
                                           strategy=strategy)
                    self.assertTrue(torch.equal(actual, expected), (strategy, shape, target_dtype))
                    self.assertTrue(torch.equal(torch.topk(actual, 5, largest=False).indices,
                                                torch.topk(expected, 5, largest=False).indices))
        self.assertTrue(all(layer.offset_tensor is None for layer in net.list
                            if hasattr(layer, "offset_tensor")))

    def test_cpu_pow_remains_dense_even_when_tiles_requested(self):
        net = PSRN(3, ["Pow", "Identity"], 1, device="cpu")
        x = torch.tensor([[0., -2., 0.5], [1.1, 2.3, -0.7]])
        y = torch.zeros(2)
        with patch.object(net.list[-1], "iter_tiles", side_effect=AssertionError("CPU Pow tiled")):
            actual = net.score_mse(x, y, strategy="tiled", workspace_budget_bytes=1)
        self.assertEqual(net.last_score_strategy, "dense")
        self.assertTrue(torch.equal(actual, original_scores(net, x, y)))

    def test_bound_tracks_dtype_and_triangle_storage(self):
        net = PSRN(10, ["Add"], 1, device="cpu")
        x = torch.ones(1, 10)
        y = torch.zeros(1)
        # FP32 scores + input + results + two int32 indices + two gathers.
        expected = 55 * 4 + 10 * 4 + 55 * (4 + 8 + 2 * 4)
        self.assertEqual(net.estimate_dense_score_bytes(x, y), expected)
        self.assertGreater(net.estimate_dense_score_bytes(x.double(), y.double()), expected)

    def test_invalid_dispatch_options_and_scalar_target(self):
        net = PSRN(2, ["Add"], 1, device="cpu")
        x, y = torch.ones(1, 2), torch.zeros(1)
        for options in ({"strategy": "unknown"}, {"workspace_budget_bytes": True},
                        {"workspace_budget_bytes": 0}, {"tile_size": 0},
                        {"sample_chunk_size": False}):
            with self.assertRaises(ValueError):
                net.score_mse(x, y, **options)
        with self.assertRaises(ValueError):
            net.score_mse(x, y[0])


if __name__ == "__main__":
    unittest.main()
