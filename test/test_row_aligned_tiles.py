"""Exact row-tile geometry and scoring checks; no accelerator performance claim."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

package = types.ModuleType('_psrn_rowtile_tests')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'psrn')]
sys.modules[package.__name__] = package
models = importlib.import_module(package.__name__ + '.model.models')
PSRN, SymbolLayer, DRLayer = models.PSRN, models.SymbolLayer, models.DRLayer
OPS = ['Add', 'Mul', 'Sub', 'Div', 'SemiSub', 'SemiDiv', 'Pow', 'Identity',
       'Sin', 'Cos', 'Exp', 'Log', 'Neg', 'Inv', 'Sign', 'Pow2', 'Pow3',
       'Sigmoid', 'Abs', 'Cosh', 'Tanh', 'Sqrt']


def original_scores(net, x, y):
    with torch.no_grad():
        sums = torch.zeros((1, net.out_dim), device=net.device)
        for row in range(x.shape[0]):
            h = net.forward(x[row].reshape(1, -1))
            target = y[row]
            if target.dim() > 1:
                target = target.reshape(1)
            diff = h - target
            sums += diff ** 2
        scores = (sums / x.shape[0]).reshape(-1)
        scores[~torch.isfinite(scores)] = float('inf')
        return scores


def dense_expressions(net, indices):
    """Recover expressions through the original dense operand tables."""
    tables = {id(layer): layer.get_offset_tensor('cpu') for layer in net.list
              if isinstance(layer, SymbolLayer)}

    def expression(index, position):
        if position < 0:
            return net.current_expr_ls[index]
        layer = net.list[position]
        if isinstance(layer, DRLayer):
            return expression(layer.dr_indices[index].item(), position - 1)
        start = 0
        for op in layer.list:
            if index < start + op.out_dim:
                break
            start += op.out_dim
        left, right = tables[id(layer)][index].tolist()
        if op.is_unary:
            return op.operator.get_expr(expression(left, position - 1))
        return op.operator.get_expr(expression(left, position - 1),
                                    expression(right, position - 1))

    return [expression(index, len(net.list) - 1) for index in indices]


class RowAlignedTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(20261005)

    def assert_prediction_bits(self, actual, expected):
        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
        self.assertTrue(torch.equal(torch.isposinf(actual), torch.isposinf(expected)))
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        finite = torch.isfinite(expected)
        integer_dtype = torch.int32 if actual.dtype == torch.float32 else torch.int64
        self.assertTrue(torch.equal(actual.contiguous().view(integer_dtype)[finite],
                                    expected.contiguous().view(integer_dtype)[finite]))

    def collect(self, layer, x, cap):
        values, next_offset = [], 0
        for offset, tile in layer.iter_tiles(x, cap):
            self.assertEqual(offset, next_offset)
            self.assertLessEqual(tile.shape[1], cap)
            self.assertEqual(tile.shape[0], x.shape[0])
            values.append(tile)
            next_offset += tile.shape[1]
        self.assertEqual(next_offset, layer.out_dim)
        return torch.cat(values, dim=1)

    def test_exhaustive_triangle_row_boundaries_and_fallback(self):
        for width in range(1, 21):
            layer = SymbolLayer(width, ['Add', 'SemiSub'], device='cpu')
            x = torch.arange(2 * width, dtype=torch.float64).reshape(2, width)
            expected = torch.cat([layer(row.reshape(1, -1)) for row in x])
            for cap in {1, 7, 31, max(1, width - 1), width, width + 1,
                        2 * width + 1, 1048576}:
                with self.subTest(width=width, cap=cap):
                    with patch.object(torch, 'searchsorted', wraps=torch.searchsorted) as search:
                        actual = self.collect(layer, x, cap)
                    self.assert_prediction_bits(actual, expected)
                    self.assertEqual(search.call_count == 0, width <= cap)

    def test_all_operators_raw_predictions_reordered_and_strided(self):
        orders = [OPS, list(reversed(OPS)), OPS[::2] + OPS[1::2]]
        for dtype in (torch.float32, torch.float64):
            for width in (1, 3, 8):
                storage = torch.rand(6, width * 2, dtype=dtype) * 4 - 2
                storage[0, 0] = 0
                x = storage[::2, ::2]
                for operators in orders:
                    layer = SymbolLayer(width, operators, device='cpu')
                    expected = torch.cat([layer(row.reshape(1, -1)) for row in x])
                    for cap in {1, 7, 31, width, width + 1, 1048576}:
                        with self.subTest(dtype=dtype, width=width, operators=operators, cap=cap):
                            # CPU Pow historically differs for some gathered
                            # layouts: production scoring retains its dense
                            # override. Its predictions are checked separately.
                            actual = self.collect(layer, x, cap)
                            non_pow = torch.ones(layer.out_dim, dtype=torch.bool)
                            start = 0
                            for op in layer.list:
                                if isinstance(op, models.Pow):
                                    non_pow[start:start + op.out_dim] = False
                                start += op.out_dim
                            self.assert_prediction_bits(actual[:, non_pow], expected[:, non_pow])

    def check_scores(self, net, x, y, caps):
        net.current_expr_ls = ['x{}'.format(i) for i in range(net.n_variables)]
        expected = original_scores(net, x, y)
        k = min(5, net.out_dim)
        selected = torch.topk(expected, k, largest=False).indices
        exprs = dense_expressions(net, selected.tolist())
        for cap in caps:
            actual = net.score_mse(x, y, strategy='tiled', tile_size=cap,
                                   sample_chunk_size=2)
            self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
            self.assertTrue(torch.equal(torch.topk(actual, k, largest=False).indices, selected))
            self.assertEqual([net.get_expr(i) for i in selected.tolist()], exprs)
            has_pow = any(isinstance(op, models.Pow) for op in net.list[-1].list)
            self.assertEqual(net.last_score_strategy, 'dense' if has_pow else 'tiled')

    def test_each_operator_two_three_layers_mask_dtypes_target_shapes(self):
        for operator in OPS:
            for layers in (2, 3):
                prefix = PSRN(2, [operator], layers - 1, device='cpu')
                mask = torch.arange(prefix.out_dim) % 3 != 1
                for masked in (False, True):
                    net = PSRN(2, [operator], layers, dr_mask=mask if masked else None,
                               device='cpu')
                    for dtype in (torch.float32, torch.float64):
                        x = torch.tensor([[0., 9., -1., 9.], [2., 9., 0.5, 9.],
                                          [-0.5, 9., 1.2, 9.]], dtype=dtype)[:, ::2]
                        for target_shape in ((3,), (3, 1)):
                            y = torch.tensor([1., 0., -1.], dtype=torch.float64).reshape(target_shape)
                            with self.subTest(op=operator, layers=layers, mask=masked,
                                              dtype=dtype, target_shape=target_shape):
                                self.check_scores(net, x, y, (7, 31, 1048576))

    def test_combined_reordered_masked_networks_and_nonfinite_inputs(self):
        non_pow = [op for op in OPS if op != 'Pow']
        for operators in (OPS, list(reversed(OPS)), OPS[::2] + OPS[1::2],
                          non_pow, list(reversed(non_pow)), non_pow[::2] + non_pow[1::2]):
            for layers in (2, 3):
                prefix = PSRN(2, operators, layers - 1, device='cpu')
                # Bound the combined three-layer final width without changing
                # the preceding full layer's operator evaluation or ordering.
                mask = torch.arange(prefix.out_dim) < 5
                for dtype in (torch.float32, torch.float64):
                    net = PSRN(2, operators, layers, dr_mask=mask, device='cpu')
                    x = torch.tensor([[0., 9., -1., 9.], [float('inf'), 9., 0.5, 9.],
                                      [float('nan'), 9., -float('inf'), 9.]], dtype=dtype)[:, ::2]
                    y = torch.tensor([1., 0., -1.], dtype=dtype).reshape(3, 1, 1, 1)
                    self.check_scores(net, x, y, (1, 7, 31, 1048576))
                    finite_x = torch.tensor([[0.5, 1.2], [0.8, 1.1], [1.3, 0.7]], dtype=dtype)
                    self.check_scores(net, finite_x, y.reshape(3), (7, 31, 1048576))

    def test_row_tiles_cross_sample_chunk_tail_without_pow_override(self):
        for operators in (['Add', 'Sub', 'SemiDiv', 'Identity'],
                          ['Div', 'SemiSub', 'Mul', 'Cos']):
            net = PSRN(3, operators, 2, device='cpu')
            x = torch.rand(34, 6, dtype=torch.float64)[::2, ::2] * 2 - 1
            y = torch.linspace(-1, 1, 17).reshape(17, 1)
            expected = original_scores(net, x, y)
            width = net.list[-1].in_dim
            for cap in (width - 1, width, width + 1, width * 2 + 1, 1048576):
                actual = net.score_mse(x, y, strategy='tiled', tile_size=cap,
                                       sample_chunk_size=16)
                self.assertEqual(net.last_score_strategy, 'tiled')
                self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
