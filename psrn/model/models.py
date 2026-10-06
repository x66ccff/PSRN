import math
import torch
import torch.nn as nn

from .functions import (
    Identity,
    Sin,
    Cos,
    Exp,
    Log,
    Neg,
    Inv,
    Add,
    Mul,
    Div,
    Sub,
    SemiDiv,
    SemiSub,
)

from .functions import Sign, Pow2, Pow3, Pow, Sigmoid, Abs, Cosh, Tanh, Sqrt


def _dtype_element_size(dtype):
    try:
        return dtype.itemsize
    except AttributeError:
        # Older Torch exposes size on tensors rather than the dtype object.
        return torch.empty((), dtype=dtype, device="cpu").element_size()


# Duplicate Removal Layer
class DRLayer(nn.Module):
    def __init__(self, in_dim, dr_mask, device=None):
        super(DRLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = round(torch.sum(dr_mask).item())
        arange_tensor = torch.arange(len(dr_mask), device=device)
        self.dr_indices = arange_tensor[dr_mask]  # (n,)
        self.dr_mask = dr_mask  # (n,)

        self.dr_indices = self.dr_indices.to(device)
        self.dr_mask = self.dr_mask.to(device)

    def forward(self, x):
        # shape x: (batch_size, in_dim)
        return x[:, self.dr_mask]

    def get_op_and_offset(self, index):
        return self.dr_indices[index].item()


class SymbolLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
        device=None,
    ):
        super(SymbolLayer, self).__init__()

        self.device = device

        self.in_dim = in_dim
        self.n_triu = in_dim * (in_dim + 1) // 2
        self.in_dim_square = in_dim * in_dim
        self.operators = operators

        self.list = nn.ModuleList()
        self.n_binary_U = 0  # undirected * +
        self.n_binary_D = 0  # directed   / -
        self.n_unary = 0

        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary:
                if func.is_directed:
                    self.n_binary_D += 1
                else:
                    self.n_binary_U += 1
            else:
                self.n_unary += 1
            # self.list.append(func)
        # first place Add and Mul (triangled-shaped ops)
        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary and not func.is_directed:
                self.list.append(func)

        # then place Sub and Div (squared-shape ops)
        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary and func.is_directed:
                self.list.append(func)

        # finally unary ops
        for op in operators:
            func = eval(op)(in_dim, device)
            if func.is_unary:
                self.list.append(func)

        self.out_dim = (
            self.n_unary * self.in_dim
            + self.n_binary_U * self.n_triu
            + self.n_binary_D * self.in_dim_square
        )

        self.out_dim_cum_ls = None
        self.init_offset(device)

    def forward(self, x):
        # shape x: (batch_size, in_dim)
        h = []
        for module in self.list:
            h.append(module(x))
        h = torch.cat(h, dim=1)  # shape: (batch_size, out_dim)
        return h

    def iter_tiles(self, x, tile_size):
        """Yield (global offset, values) without a full final-layer output.

        Inputs may contain multiple independent rows. Candidate order is the
        same operator-block/operand order as ``forward`` for a single row.
        """
        if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError("tile inputs must have shape (samples, in_dim)")
        binary_operations = {
            Add: torch.add, Mul: torch.mul,
            Sub: torch.sub, SemiSub: torch.sub,
            Div: torch.div, SemiDiv: torch.div, Pow: torch.pow,
        }
        triangle_starts = None
        if self.n_binary_U and self.in_dim > tile_size:
            rows = torch.arange(self.in_dim + 1, device=x.device, dtype=torch.int64)
            triangle_starts = rows * (2 * self.in_dim - rows + 1) // 2
        offset = 0
        for module in self.list:
            count = module.out_dim
            if module.is_unary:
                # Preserve the original single-row/full-width kernel shape.
                # CPU transcendental vector/tail kernels can round differently
                # if rows are batched or the unary input itself is tiled.
                unary_values = torch.cat(
                    [module(x[row:row + 1]) for row in range(x.shape[0])], dim=0
                )
            elif self.in_dim <= tile_size:
                operation = binary_operations.get(type(module))
                if operation is None:
                    raise NotImplementedError("tiled scoring for {}".format(type(module).__name__))
                row_start = 0
                while row_start < self.in_dim:
                    if module.is_directed:
                        row_end = min(self.in_dim, row_start + tile_size // self.in_dim)
                        start = row_start * self.in_dim
                        # Views of complete rows preserve the original directed
                        # broadcast geometry without candidate-sized gathers.
                        values = operation(
                            x[:, row_start:row_end, None], x[:, None, :]
                        ).reshape(x.shape[0], -1)
                    else:
                        start = row_start * (2 * self.in_dim - row_start + 1) // 2
                        # Find the last complete triangular row within the cap.
                        # All arithmetic stays on the host and uses exact ints.
                        low, high = row_start + 1, self.in_dim + 1
                        while high - low > 1:
                            middle = (low + high) // 2
                            stop = middle * (2 * self.in_dim - middle + 1) // 2
                            if stop - start <= tile_size:
                                low = middle
                            else:
                                high = middle
                        row_end = low
                        indices = torch.triu_indices(
                            row_end - row_start, self.in_dim, offset=row_start,
                            dtype=torch.int64, device=x.device,
                        )
                        indices[0].add_(row_start)
                        values = operation(x[:, indices[0]], x[:, indices[1]])
                    yield offset + start, values
                    del values
                    if not module.is_directed:
                        del indices
                    row_start = row_end
                offset += count
                continue
            for start in range(0, count, tile_size):
                end = min(start + tile_size, count)
                if module.is_unary:
                    values = unary_values[:, start:end]
                else:
                    operation = binary_operations.get(type(module))
                    if operation is None:
                        raise NotImplementedError("tiled scoring for {}".format(type(module).__name__))
                    local = torch.arange(start, end, device=x.device, dtype=torch.int64)
                    if module.is_directed:
                        left = torch.div(local, self.in_dim, rounding_mode="floor")
                        right = local % self.in_dim
                    else:
                        left = torch.searchsorted(triangle_starts, local, right=True) - 1
                        right = left + local - triangle_starts[left]
                    # Match functions.forward, including unprotected division,
                    # powers, and their NaN/Inf behavior.
                    values = operation(x[:, left], x[:, right])
                yield offset + start, values
                del values
                if not module.is_unary:
                    del local, left, right
            if module.is_unary:
                del unary_values
            offset += count

    def init_offset(self, device):
        # Operand indices are reconstructed only for selected expressions.
        # Keep the dense-table builder available for compatibility and audits,
        # but do not materialize its candidate-sized storage at initialization.
        self.offset_tensor = None

    def get_offset_tensor(self, device):
        offset_tensor = torch.zeros((self.out_dim, 2), dtype=torch.int, device=device)
        arange_tensor = torch.arange(self.in_dim, dtype=torch.int, device=device)

        binary_U_tensor = torch.zeros((self.n_triu, 2), dtype=torch.int, device=device)
        binary_D_tensor = torch.zeros(
            (self.in_dim_square, 2), dtype=torch.int, device=device
        )
        unary_tensor = torch.zeros((self.in_dim, 2), dtype=torch.int, device=device)

        unary_tensor[:, 0] = arange_tensor
        unary_tensor[:, 1] = self.in_dim

        start = 0
        for i in range(self.in_dim):
            len_ = self.in_dim - i
            binary_U_tensor[start : start + len_, 0] = i
            binary_U_tensor[start : start + len_, 1] = arange_tensor[i:]
            start += len_

        start = 0
        for i in range(self.in_dim):
            len_ = self.in_dim
            binary_D_tensor[start : start + len_, 0] = i
            binary_D_tensor[start : start + len_, 1] = arange_tensor[0:]
            start += len_

        start = 0
        for func in self.list:
            if not func.is_unary:
                if func.is_directed:
                    t = binary_D_tensor
                else:
                    t = binary_U_tensor
            else:
                t = unary_tensor
            len_ = t.shape[0]

            offset_tensor[start : start + len_ :] = t
            start += len_

        return offset_tensor

    def get_out_dim_cum_ls(self):
        if self.out_dim_cum_ls != None:
            return self.out_dim_cum_ls

        out_dim_ls = []
        for func in self.list:
            if not func.is_unary:
                if func.is_directed:
                    out_dim_ls.append(self.in_dim_square)
                else:
                    out_dim_ls.append(self.n_triu)
            else:
                out_dim_ls.append(self.in_dim)
        self.out_dim_cum_ls = [sum(out_dim_ls[: i + 1]) for i in range(len(out_dim_ls))]
        return self.out_dim_cum_ls

    def get_op_and_offset(self, index):
        if index < 0 or index >= self.out_dim:
            raise IndexError("candidate index out of range")
        out_dim_cum_ls = self.get_out_dim_cum_ls()
        start = 0
        for i, func in enumerate(self.list):
            if index < out_dim_cum_ls[i]:
                break
            start = out_dim_cum_ls[i]
        local = index - start
        if func.is_unary:
            offset = [local, self.in_dim]
        elif func.is_directed:
            offset = [local // self.in_dim, local % self.in_dim]
        else:
            # Invert row-major upper-triangle enumeration using integer
            # arithmetic. The initial root may lie one row past the answer.
            d = self.in_dim
            row = (2 * d + 1 - math.isqrt((2 * d + 1) ** 2 - 8 * local)) // 2
            prefix = row * (2 * d - row + 1) // 2
            if prefix > local:
                row -= 1
                prefix = row * (2 * d - row + 1) // 2
            offset = [row, row + local - prefix]
        return func.operator, offset


class PSRN(nn.Module):
    def __init__(
        self,
        n_variables=1,
        operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
        n_symbol_layers=3,
        dr_mask=None,
        device="cuda",
    ):
        super(PSRN, self).__init__()

        if isinstance(device, str):
            if device == "cuda":
                self.device = torch.device("cuda")
            elif device == "cpu":
                self.device = torch.device("cpu")
            else:
                raise ValueError("device must be cuda or cpu, got {}".format(device))
        self.device = device
        self.n_variables = n_variables
        self.operators = operators
        self.n_symbol_layers = n_symbol_layers

        self.list = nn.ModuleList()

        if dr_mask is None:
            self.use_dr_mask = False
        else:
            self.use_dr_mask = True

        if self.use_dr_mask:
            assert type(dr_mask) == torch.Tensor, "dr_mask must be a tensor"
            assert dr_mask.dim() == 1, "dr_mask should be 1-dim, got {}".format(
                dr_mask.dim()
            )
            dr_mask = dr_mask.to(self.device)

        for i in range(n_symbol_layers):
            if self.use_dr_mask and i == n_symbol_layers - 1:
                self.list.append(
                    DRLayer(self.list[-1].out_dim, dr_mask=dr_mask, device=self.device)
                )

            if i == 0:
                self.list.append(
                    SymbolLayer(n_variables, operators, device=self.device)
                )
            else:
                self.list.append(
                    SymbolLayer(self.list[-1].out_dim, operators, device=self.device)
                )

        self.current_expr_ls = []

        self.out_dim = self.list[-1].out_dim

    def __repr__(self):
        return (
            super().__repr__()
            + "\n"
            + "n_inputs: {}, operators: {}, n_layers: {}".format(
                self.n_variables, self.operators, self.n_symbol_layers
            )
            + "\n dim:"
            + "\n".join(str(layer.out_dim) for layer in self.list)
        )

    def forward(self, x):
        # shape x: (batch_size, n_variables)
        h = x
        for i, layer in enumerate(self.list):
            h = layer(h)
        return h  # shape: (batch_size, out_dim)

    def estimate_dense_score_bytes(self, x, y):
        """Conservative live-tensor bound for scoring one dense prediction row.

        Count the score buffer, layer input/output, operator outputs retained
        until cat, triangular int32 operand indices and two gathered operands,
        and the prediction/error/square at accumulation. This is a dispatch
        estimate, not an allocator/reserved-memory or whole-process limit.
        """
        value_bytes = _dtype_element_size(x.dtype)
        # A zero-dimensional target has scalar promotion rules, unlike a
        # one-element vector. No tensor values or device state are read.
        error_dtype = torch.result_type(x, y[0])
        error_bytes = _dtype_element_size(error_dtype)
        score_bytes = self.out_dim * _dtype_element_size(torch.get_default_dtype())
        peak = 0
        for layer in self.list:
            inputs = layer.in_dim * value_bytes
            outputs = layer.out_dim * value_bytes
            # All operator results coexist with the concatenated output.
            layer_peak = inputs + 2 * outputs
            if isinstance(layer, SymbolLayer):
                largest_triangle = 0
                for op in layer.list:
                    if not op.is_unary and not op.is_directed:
                        largest_triangle = max(largest_triangle, op.out_dim)
                # Only the largest triangular block can set this bound.
                # Inspect current modules rather than cache mutable topology.
                if largest_triangle:
                    layer_peak = max(
                        layer_peak,
                        inputs + outputs + largest_triangle * (8 + 2 * value_bytes),
                    )
            peak = max(peak, layer_peak)
        peak = max(peak, self.out_dim * (value_bytes + 2 * error_bytes))
        return score_bytes + peak

    @torch.no_grad()
    def score_mse(self, x, y, tile_size=1048576, sample_chunk_size=16,
                  strategy="auto", workspace_budget_bytes=512 * 1024 ** 2):
        """Return ordered candidate MSEs using dense rows or final-layer tiles.

        Auto uses the original single-row forward when its estimated live
        tensor storage fits a sample-count-aware allowance. The allowance is
        workspace_budget_bytes * sample_chunk_size / average_chunk_rows,
        where average_chunk_rows counts actual rows per tiled sample group.
        Thus small queries can stay dense when tiling has little batching
        benefit. The budget is a policy base, not a hard workspace cap; the
        global score vector is always retained. Use strategy="tiled" to
        exercise tiles explicitly, or "dense" for a reference evaluation.
        Both paths keep original accumulator dtype, sample order and global
        min/topk ties. CPU Pow, active autocast and nonfloating inputs always
        use dense rows to preserve rounding, final cat dtype promotion and
        the original behavior for unsupported input types.
        last_score_strategy records the selected path without device queries.
        """
        if x.dim() != 2 or x.shape[0] == 0 or x.shape[1] != self.n_variables:
            raise ValueError("x must contain at least one row of n_variables inputs")
        sample_count = x.shape[0]
        target_rank = y.dim()
        if target_rank == 0 or y.shape[0] != sample_count or y.numel() != sample_count:
            raise ValueError("y must contain one scalar target per input row")
        for name, value in (("tile_size", tile_size), ("sample_chunk_size", sample_chunk_size),
                            ("workspace_budget_bytes", workspace_budget_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if strategy not in ("auto", "dense", "tiled"):
            raise ValueError("strategy must be auto, dense, or tiled")
        # Preserve scalar targets for (N,), and vector promotion for every
        # higher rank, without checking or reshaping each target row.
        if target_rank > 2:
            y = y.reshape(sample_count, 1)
        # AMP may promote operator outputs before the original final cat.
        # Preserve that shared prediction dtype as well as AMP pow semantics.
        try:
            autocast_enabled = torch.is_autocast_enabled(x.device.type)
        except TypeError:
            # Torch before 2.4 has separate no-argument CUDA/CPU queries.
            if x.device.type == "cuda":
                autocast_enabled = torch.is_autocast_enabled()
            elif x.device.type == "cpu":
                autocast_enabled = torch.is_autocast_cpu_enabled()
            else:
                # An unknown legacy backend keeps the original square.
                autocast_enabled = True
        cpu_pow = x.device.type == "cpu" and any(
            isinstance(op, Pow) for op in self.list[-1].list
        )
        # Integer operators can retain integer outputs while transcendental
        # operators promote to float. Score only after the shared final cat.
        # Complex inputs likewise retain original forward/scoring behavior.
        use_dense = (autocast_enabled or not x.is_floating_point()
                     or cpu_pow or strategy == "dense")
        if not use_dense and strategy == "auto":
            chunk_rows = min(sample_chunk_size, sample_count)
            sample_groups = (sample_count + chunk_rows - 1) // chunk_rows
            # Compare without floating point rounding at the policy boundary.
            use_dense = self.estimate_dense_score_bytes(x, y) * sample_count <= (
                workspace_budget_bytes * sample_chunk_size * sample_groups
            )
        self.last_score_strategy = "dense" if use_dense else "tiled"
        if use_dense:
            sums = torch.zeros((1, self.out_dim), device=self.device)
            for sample in range(sample_count):
                h = self.forward(x[sample].reshape(1, -1))
                target = y[sample]
                diff = h - target
                if autocast_enabled:
                    sums += diff ** 2
                else:
                    diff.pow_(2)
                    sums += diff
                # Do not carry the previous full row into the next forward.
                del h, diff
            sums /= sample_count
            scores = sums.reshape(-1)
            scores.nan_to_num_(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            return scores
        scores = torch.zeros(self.out_dim, device=self.device)
        final_layer = self.list[-1]
        # Each sample chunk completes all candidate tiles before the next
        # chunk; each candidate still accumulates samples in original order.
        for sample_start in range(0, sample_count, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, sample_count)
            # Retain only this chunk's pre-final features. Directed earlier
            # layers expect a single row, so they must not receive a batch.
            if len(self.list) == 1:
                # Preserve original row strides for unary kernel selection.
                features = x[sample_start:sample_end]
            else:
                features = []
                for sample in range(sample_start, sample_end):
                    h = x[sample].reshape(1, -1)
                    for layer in self.list[:-1]:
                        h = layer(h)
                    features.append(h)
                features = torch.cat(features, dim=0)
            for offset, predictions in final_layer.iter_tiles(features, tile_size):
                sums = scores[offset:offset + predictions.shape[1]]
                for local_sample in range(sample_end - sample_start):
                    # Preserve the original target shape and dtype promotion:
                    # y[i] is scalar for (N,), one-element for (N, 1).
                    target = y[sample_start + local_sample]
                    diff = predictions[local_sample] - target
                    square = diff ** 2
                    sums += square
                # Release consumer references before the generator allocates
                # the next tile; its own references are cleared after yield.
                del predictions, diff, square
        # In-place division avoids a second full candidate vector. Element
        # arithmetic and normalization order are identical to the baseline.
        scores /= sample_count
        # Preserve every finite value and map all nonfinite scores to +Inf,
        # without candidate-sized abs and boolean-indexing intermediates.
        scores.nan_to_num_(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        return scores

    def get_expr(self, index):
        return self._get_expr(index, -1)

    def _get_expr(self, index, layer_idx):

        if len(self.list) + layer_idx < 0:
            return self.current_expr_ls[index]

        layer = self.list[layer_idx]

        if layer._get_name() == "DRLayer":
            new_index = layer.get_op_and_offset(index)
            return self._get_expr(new_index, layer_idx - 1)

        else:
            # SymbolLayer

            func_op, offset = layer.get_op_and_offset(index)

            if func_op.is_unary:
                return func_op.get_expr(self._get_expr(offset[0], layer_idx - 1))
            else:
                return func_op.get_expr(
                    self._get_expr(offset[0], layer_idx - 1),
                    self._get_expr(offset[1], layer_idx - 1),
                )
