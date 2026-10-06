"""Check dense scratch reduction against the unchanged ordered MSE formula."""
import importlib
import itertools
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

package = types.ModuleType('_psrn_dense_inplace_tests')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'psrn')]
sys.modules[package.__name__] = package
models = importlib.import_module(package.__name__ + '.model.models')
PSRN = models.PSRN
FAMILIES = [
    ['Add', 'Mul', 'Sub', 'Div', 'Identity', 'Sin', 'Cos', 'Exp', 'Log'],
    ['Add', 'Mul', 'SemiSub', 'SemiDiv', 'Identity', 'Sin', 'Cos', 'Exp', 'Log'],
    ['Add', 'Mul', 'Identity', 'Pow2', 'Pow3', 'Neg'],
    ['Pow', 'Identity'],
    ['Identity', 'Sin', 'Cos', 'Exp', 'Log', 'Neg', 'Inv', 'Sign', 'Pow2',
     'Pow3', 'Sigmoid', 'Abs', 'Cosh', 'Tanh', 'Sqrt'],
    ['Add', 'Mul', 'Sub', 'Div', 'SemiSub', 'SemiDiv', 'Pow', 'Identity',
     'Sin', 'Cos', 'Exp', 'Log', 'Neg', 'Inv', 'Sign', 'Pow2', 'Pow3',
     'Sigmoid', 'Abs', 'Cosh', 'Tanh', 'Sqrt'],
]


@torch.no_grad()
def original_scores(net, x, y):
    if y.dim() > 2:
        y = y.reshape(x.shape[0], 1)
    sums = torch.zeros((1, net.out_dim), device=net.device)
    for row in range(x.shape[0]):
        h = net.forward(x[row].reshape(1, -1))
        diff = h - y[row]
        square = diff ** 2
        sums += square
    result = (sums / x.shape[0]).reshape(-1)
    result[~torch.isfinite(result)] = float('inf')
    return result


def integer_view(values):
    return values.contiguous().view(torch.int64 if values.element_size() == 8 else torch.int32)


class DenseInplaceScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_scores_ties_dtype_promotion_masks_and_strides(self):
        for operators, masked in itertools.product(FAMILIES, (False, True)):
            prefix = PSRN(2, operators, 1, device='cpu')
            mask = torch.arange(prefix.out_dim) % 3 != 1 if masked else None
            net = PSRN(2, operators, 2, dr_mask=mask, device='cpu')
            for xd, yd, rank, samples, invalid in itertools.product(
                    (torch.float32, torch.float64), (torch.float32, torch.float64),
                    (1, 2, 4), (1, 10, 17), (False, True)):
                with self.subTest(operators=operators, masked=masked, xd=xd, yd=yd,
                                  rank=rank, samples=samples, invalid=invalid):
                    storage = torch.linspace(.1, 1.7, samples * 8, dtype=xd).reshape(samples * 2, 4)
                    x = storage[::2, ::2]
                    if invalid:
                        x[0, 0] = 0
                        x[0, 1] = -1
                        if samples > 1:
                            x[1, 0], x[2, 0] = float('nan'), float('inf')
                            x[3, 0], x[4, 0] = -float('inf'), 10000
                    y = torch.linspace(-1, 1, samples * 2, dtype=yd)[::2]
                    y = y.reshape((samples,) + (1,) * (rank - 1))
                    x_bits = integer_view(x).clone()
                    y_bits = integer_view(y).clone()
                    expected = original_scores(net, x, y)
                    actual = net.score_mse(x, y, strategy='dense')
                    self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))
                    k = min(5, net.out_dim)
                    self.assertTrue(torch.equal(torch.topk(actual, k, largest=False).indices,
                                                torch.topk(expected, k, largest=False).indices))
                    self.assertTrue(torch.equal(integer_view(x), x_bits))
                    self.assertTrue(torch.equal(integer_view(y), y_bits))
                    self.assertEqual(net.last_score_strategy, 'dense')

    def test_autocast_keeps_original_pow_path(self):
        net = PSRN(2, ['Identity', 'Add', 'Pow2'], 1, device='cpu')
        real_pow = torch.Tensor.pow_
        calls = []

        def observed_pow(values, exponent):
            calls.append(values.shape)
            return real_pow(values, exponent)

        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([[.25, .5], [1., -.75], [.125, 2.]], dtype=dtype)
            y = torch.tensor([.125, -1., .5], dtype=dtype)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                expected = original_scores(net, x, y)
                with patch.object(torch.Tensor, 'pow_', new=observed_pow):
                    actual = net.score_mse(x, y, strategy='dense')
            self.assertEqual(calls, [])
            self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))
        with patch.object(torch.Tensor, 'pow_', new=observed_pow):
            actual = net.score_mse(x, y, strategy='dense')
        self.assertEqual(len(calls), x.shape[0])
        self.assertTrue(torch.equal(integer_view(actual), integer_view(original_scores(net, x, y))))

    def test_legacy_autocast_query_arity_cpu_cuda_and_unknown_metadata(self):
        # All computation stays on CPU. Only device-query metadata is mocked
        # to exercise CUDA and unknown legacy API selection without hardware.
        net = PSRN(2, ['Identity'], 1, device='cpu')
        x = torch.tensor([[.25, -.5], [1.25, 2.]])
        y = torch.tensor([.1, -.2])
        expected = original_scores(net, x, y)
        real_pow = torch.Tensor.pow_
        for device, enabled in (('cpu', False), ('cpu', True),
                                ('cuda', False), ('cuda', True), ('mps', False)):
            calls = []

            def legacy_query(*args):
                if args:
                    raise TypeError('legacy query accepts no arguments')
                return enabled

            def observed_pow(values, exponent):
                calls.append(values.shape)
                return real_pow(values, exponent)

            with self.subTest(device=device, enabled=enabled):
                with patch.object(torch, 'is_autocast_enabled', side_effect=legacy_query) as query, \
                        patch.object(torch, 'is_autocast_cpu_enabled', return_value=enabled) as cpu_query, \
                        patch.object(torch.Tensor, 'device', new=property(lambda _: torch.device(device))), \
                        patch.object(torch.Tensor, 'pow_', new=observed_pow):
                    actual = net.score_mse(x, y, strategy='dense')
                self.assertEqual(query.call_args_list[0].args, (device,))
                if device == 'cpu':
                    self.assertEqual(query.call_count, 1)
                    cpu_query.assert_called_once_with()
                elif device == 'cuda':
                    self.assertEqual(query.call_count, 2)
                    self.assertEqual(query.call_args_list[1].args, ())
                    cpu_query.assert_not_called()
                else:
                    self.assertEqual(query.call_count, 1)
                    cpu_query.assert_not_called()
                expected_pow_calls = x.shape[0] if device != 'mps' and not enabled else 0
                self.assertEqual(len(calls), expected_pow_calls)
                self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))

    def test_modern_autocast_query_avoids_legacy_fallback(self):
        net = PSRN(2, ['Identity'], 1, device='cpu')
        x, y = torch.tensor([[.1, .5]]), torch.tensor([.25])
        with patch.object(torch, 'is_autocast_enabled', wraps=torch.is_autocast_enabled) as query, \
                patch.object(torch, 'is_autocast_cpu_enabled', side_effect=AssertionError('legacy called')):
            actual = net.score_mse(x, y, strategy='dense')
        query.assert_called_once_with('cpu')
        self.assertTrue(torch.equal(integer_view(actual), integer_view(original_scores(net, x, y))))

    def test_dtype_size_modern_metadata_needs_no_allocation(self):
        with patch.object(torch, 'empty', side_effect=AssertionError('modern allocation')):
            for dtype in (torch.float32, torch.float64, torch.bfloat16):
                self.assertEqual(models._dtype_element_size(dtype), dtype.itemsize)

    def test_legacy_dtype_size_auto_bound_route_and_scores(self):
        class LegacyDtype:
            def __init__(self, actual):
                self.actual = actual

        real_empty = torch.empty
        real_result_type = torch.result_type
        real_default_dtype = torch.get_default_dtype
        real_tensor_dtype = torch.Tensor.dtype
        allocations = []

        def legacy_empty(*args, **kwargs):
            self.assertEqual(args, ((),))
            self.assertEqual(kwargs['device'], 'cpu')
            self.assertIsInstance(kwargs['dtype'], LegacyDtype)
            allocations.append(kwargs['dtype'].actual)
            kwargs['dtype'] = kwargs['dtype'].actual
            return real_empty(*args, **kwargs)

        def legacy_tensor_dtype(tensor):
            return LegacyDtype(real_tensor_dtype.__get__(tensor, torch.Tensor))

        net = PSRN(2, ['Add', 'Identity'], 1, device='cpu')
        x = torch.linspace(-1, 1, 32, dtype=torch.float32).reshape(16, 2)
        for shape in ((16,), (16, 1)):
            y = torch.linspace(-.5, .5, 16, dtype=torch.float64).reshape(shape)
            bound = net.estimate_dense_score_bytes(x, y)
            for budget in (bound - 1, bound):
                modern = net.score_mse(x, y, workspace_budget_bytes=budget)
                modern_route = net.last_score_strategy
                with patch.object(torch.Tensor, 'dtype', new=property(legacy_tensor_dtype)), \
                        patch.object(torch, 'result_type', side_effect=lambda *a: LegacyDtype(real_result_type(*a))), \
                        patch.object(torch, 'get_default_dtype', side_effect=lambda: LegacyDtype(real_default_dtype())), \
                        patch.object(torch, 'empty', side_effect=legacy_empty):
                    self.assertEqual(net.estimate_dense_score_bytes(x, y), bound)
                    legacy = net.score_mse(x, y, workspace_budget_bytes=budget)
                self.assertEqual(net.last_score_strategy, modern_route)
                self.assertTrue(torch.equal(integer_view(legacy), integer_view(modern)))
        self.assertEqual(len(allocations), 24)
        self.assertIn(torch.float64, allocations)
        self.assertIn(torch.float32, allocations)

    def test_active_autocast_overrides_auto_and_explicit_tiles(self):
        net = PSRN(2, ['Identity', 'Pow2'], 1, device='cpu')
        x, y = torch.tensor([[1., .5], [.25, 2.]]), torch.tensor([2 ** -12, -.5])
        expected = original_scores(net, x, y)
        for strategy in ('auto', 'dense', 'tiled'):
            with patch.object(torch, 'is_autocast_enabled', return_value=True) as query, \
                    patch.object(net, 'estimate_dense_score_bytes', side_effect=AssertionError('AMP estimate')), \
                    patch.object(net.list[-1], 'iter_tiles', side_effect=AssertionError('AMP tiles')):
                actual = net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
            query.assert_called_once_with('cpu')
            self.assertEqual(net.last_score_strategy, 'dense')
            self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))

    def test_actual_cpu_autocast_preserves_cat_promotion_and_scores(self):
        net = PSRN(2, ['Identity', 'Pow2', 'Add', 'Sin'], 2, device='cpu')
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([[1., .5], [.25, 2.], [-1., 0.]], dtype=dtype)
            for shape in ((3,), (3, 1)):
                y = torch.tensor([2 ** -12, -.5, .25], dtype=dtype).reshape(shape)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    expected = original_scores(net, x, y)
                    for strategy in ('auto', 'dense', 'tiled'):
                        actual = net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
                        self.assertEqual(net.last_score_strategy, 'dense')
                        self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))

    def test_off_autocast_queries_once_and_keeps_original_routes(self):
        net = PSRN(2, ['Identity', 'Add'], 1, device='cpu')
        x, y = torch.ones(17, 2), torch.zeros(17)
        expected = original_scores(net, x, y)
        for strategy, route in (('dense', 'dense'), ('tiled', 'tiled'), ('auto', 'tiled')):
            with patch.object(torch, 'is_autocast_enabled', wraps=torch.is_autocast_enabled) as query:
                actual = net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
            query.assert_called_once_with('cpu')
            self.assertEqual(net.last_score_strategy, route)
            self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))

    def test_half_operator_dtype_counterexample_without_cuda(self):
        # Emulate the mixed output dtypes of CUDA AMP Identity and Pow2.
        # This is dtype arithmetic on CPU, not a vendor CUDA AMP runtime test.
        identity = torch.tensor([[1.]], dtype=torch.float16)
        power = identity.float().pow(2)
        target = torch.tensor([2 ** -12], dtype=torch.float16)
        dense = (torch.cat((identity, power), dim=1) - target).pow(2).float()
        per_operator = torch.cat(((identity - target).pow(2).float(),
                                  (power - target).pow(2).float()), dim=1)
        self.assertEqual(dense[0, 0].item(), 0.9995117783546448)
        self.assertEqual(per_operator[0, 0].item(), 1.)
        self.assertFalse(torch.equal(integer_view(dense), integer_view(per_operator)))

    def test_integer_cat_promotion_counterexample_and_compatibility_routes(self):
        net = PSRN(1, ['Identity', 'Sin'], 1, device='cpu')
        x = torch.tensor([[2 ** 24 + 1]], dtype=torch.int64)
        y = torch.zeros(1, dtype=torch.int64)
        expected = original_scores(net, x, y)
        per_operator_square = x.pow(2).float().reshape(-1)
        self.assertEqual(expected[0].item(), float(2 ** 48))
        self.assertEqual(per_operator_square[0].item(), float(2 ** 48 + 2 ** 25))
        self.assertFalse(torch.equal(integer_view(expected[:1]), integer_view(per_operator_square)))
        for strategy in ('auto', 'dense', 'tiled'):
            actual = net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
            self.assertEqual(net.last_score_strategy, 'dense')
            self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))
        for dtype, operators, samples, target_dtype, rank in itertools.product(
                (torch.int32, torch.int64),
                (['Identity', 'Sin'], ['Add', 'Identity', 'Div', 'Log']),
                (1, 3, 17), (torch.int64, torch.float32, torch.float64), (1, 2)):
            net = PSRN(2, operators, 1, device='cpu')
            storage = torch.arange(samples * 8, dtype=dtype).reshape(samples * 2, 4) % 5 + 1
            x = storage[::2, ::2]
            y = torch.arange(samples, dtype=target_dtype) % 3
            if rank == 2:
                y = y.reshape(samples, 1)
            expected = original_scores(net, x, y)
            for strategy in ('auto', 'dense', 'tiled'):
                with patch.object(net, 'estimate_dense_score_bytes', side_effect=AssertionError('integer estimate')), \
                        patch.object(net.list[-1], 'iter_tiles', side_effect=AssertionError('integer tiles')):
                    actual = net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
                self.assertEqual(net.last_score_strategy, 'dense')
                self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))

    def test_complex_inputs_preserve_original_scoring_error(self):
        net = PSRN(2, ['Identity', 'Abs'], 1, device='cpu')
        x = torch.tensor([[1 + 2j, -1j]])
        y = torch.zeros(1)
        with self.assertRaises(RuntimeError) as reference:
            original_scores(net, x, y)
        for strategy in ('auto', 'dense', 'tiled'):
            with self.assertRaises(RuntimeError) as actual:
                net.score_mse(x, y, strategy=strategy, workspace_budget_bytes=1)
            self.assertEqual(str(actual.exception), str(reference.exception))
            self.assertEqual(net.last_score_strategy, 'dense')

    def test_off_amp_all_operator_float_output_dtypes(self):
        # Check each output separately: cat would hide heterogeneous outputs.
        for dtype, operator in itertools.product(
                (torch.float16, torch.bfloat16, torch.float32, torch.float64), FAMILIES[-1]):
            net = PSRN(2, [operator], 1, device='cpu')
            x = torch.tensor([[.5, 1.25]], dtype=dtype)
            with torch.autocast('cpu', enabled=False):
                self.assertEqual(net.forward(x).dtype, dtype, (dtype, operator))

    def test_no_grad_default_accumulator_dtype_and_three_layer_mask(self):
        operators = FAMILIES[-1]
        prefix = PSRN(2, operators, 2, device='cpu')
        mask = torch.arange(prefix.out_dim) < 3
        net = PSRN(2, operators, 3, dr_mask=mask, device='cpu')
        x = torch.tensor([[.1, .3], [.6, 1.2]], requires_grad=True)
        y = torch.tensor([.5, -.25], dtype=torch.float64, requires_grad=True)
        previous = torch.get_default_dtype()
        try:
            for accumulator_dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(accumulator_dtype)
                actual = net.score_mse(x, y, strategy='dense')
                expected = original_scores(net, x, y)
                self.assertEqual(actual.dtype, accumulator_dtype)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(torch.equal(integer_view(actual), integer_view(expected)))
        finally:
            torch.set_default_dtype(previous)


if __name__ == '__main__':
    unittest.main()
