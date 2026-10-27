"""Statically tiled TOSA emitter for multi-head self-attention (MHSA).

The emitter lowers scaled dot-product attention to plain TOSA ops with fully
static shapes.  The sequence axis is partitioned into fixed query and
key/value tiles that are chosen so that every intermediate tensor produced by
the emitted schedule fits inside a caller-supplied byte budget (the Coral NPU
scratchpad tile size).

Softmax is computed in streaming form (the "online softmax" recurrence), so
the full ``seq x kv_seq`` score matrix is never materialised:

    for each query tile q:
        for each key/value tile kv:
            s   = q . k^T
            m'  = max(m, rowmax(s))
            p   = exp(s - m')
            l   = l * exp(m - m') + rowsum(p)
            acc = acc * exp(m - m') + p . v
            m   = m'
        out_tile = acc / l

Every loop is unrolled at emission time, so the result contains no control
flow: each tile is a straight-line sequence of TOSA ops whose shapes are known
to the compiler.  That makes the memory footprint a property of the IR rather
than of a runtime scheduler.

Layout contract
---------------
``mhsa_tiled`` consumes packed ``[batch * heads, seq, head_dim]`` tensors.
:func:`emit_wrapper` adapts the usual LiteRT/JAX/PyTorch ``[B, S, H, D]`` (or
``[B, H, S, D]``) function signature to that packed form and back.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8}

#: Guard against pathological unrolling when a tiny tile is forced.
MAX_UNROLLED_TILES = 8192

_TENSOR_RE = re.compile(r"^tensor<([0-9]+(?:x[0-9]+)*)x(bf16|f16|f32|f64)>$")

NEG_MASK_VALUE = {
    "f16": -6.5e4,
    "bf16": -1.0e30,
    "f32": -1.0e30,
    "f64": -1.0e30,
}


class TilePlanError(ValueError):
    """Raised when no tile configuration satisfies the memory budget."""


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def tensor_type(shape: Sequence[int], dtype: str) -> str:
    return "tensor<" + "x".join(str(int(d)) for d in shape) + "x" + dtype + ">"


def parse_tensor_type(text: str) -> tuple[tuple[int, ...], str]:
    """Parse a *static* ``tensor<...>`` type.  Raises on dynamic dimensions."""
    match = _TENSOR_RE.match(text.strip())
    if not match:
        raise ValueError(
            f"expected a static tensor type, got {text!r}; "
            "bind dynamic dimensions first (--seq-len)"
        )
    dims, dtype = match.groups()
    return tuple(int(d) for d in dims.split("x")), dtype


def _fmt_float(value: float) -> str:
    return f"{value:.6e}"


def _fmt_nested(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_nested(v) for v in value) + "]"
    return _fmt_float(float(value))


def _names(prefix: str = "v") -> Iterable[str]:
    index = 0
    while True:
        yield f"%{prefix}{index}"
        index += 1


# ---------------------------------------------------------------------------
# Shapes, tiles, budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionShape:
    """Logical (unpadded) MHSA problem size."""

    batch: int
    heads: int
    seq: int
    kv_seq: int
    head_dim: int
    dtype: str = "f32"
    #: Absolute KV position of query row 0.  Defaults to
    #: ``max(0, kv_seq - seq)``, i.e. "the queries are the newest tokens",
    #: which is correct for both prefill (seq == kv_seq) and decode.
    q_offset: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        for name in ("batch", "heads", "seq", "kv_seq", "head_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.kv_seq < self.seq:
            raise ValueError(
                f"kv_seq ({self.kv_seq}) must be >= seq ({self.seq}); "
                "cross-attention with fewer keys than queries is not supported"
            )
        if (
            self.q_offset is not None
            and not 0 <= self.q_offset <= self.kv_seq - self.seq
        ):
            raise ValueError(
                f"q_offset must be in [0, {self.kv_seq - self.seq}], got {self.q_offset}"
            )

    @property
    def bh(self) -> int:
        return self.batch * self.heads

    @property
    def elem_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def query_offset(self) -> int:
        return (
            self.q_offset
            if self.q_offset is not None
            else max(0, self.kv_seq - self.seq)
        )


@dataclass(frozen=True)
class TilePlan:
    q_tile: int
    kv_tile: int

    def padded_seq(self, shape: AttentionShape) -> int:
        return ceil_div(shape.seq, self.q_tile) * self.q_tile

    def padded_kv(self, shape: AttentionShape) -> int:
        return ceil_div(shape.kv_seq, self.kv_tile) * self.kv_tile

    def tile_counts(self, shape: AttentionShape) -> tuple[int, int]:
        return (
            ceil_div(self.padded_seq(shape), self.q_tile),
            ceil_div(self.padded_kv(shape), self.kv_tile),
        )


@dataclass
class Budget:
    """Estimated live bytes for one unrolled tile step."""

    budget_bytes: int
    safety_factor: float
    reserved_bytes: int
    stages: dict[str, int] = field(default_factory=dict)

    @property
    def peak_bytes(self) -> int:
        return max(self.stages.values(), default=0)

    @property
    def required_bytes(self) -> int:
        return math.ceil(self.peak_bytes * self.safety_factor) + self.reserved_bytes

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.budget_bytes

    def to_dict(self) -> dict:
        return {
            "budget_bytes": self.budget_bytes,
            "safety_factor": self.safety_factor,
            "reserved_bytes": self.reserved_bytes,
            "stages_raw_bytes": dict(self.stages),
            "peak_raw_bytes": self.peak_bytes,
            "required_bytes": self.required_bytes,
            "fits": self.fits,
        }


def estimate_budget(
    shape: AttentionShape,
    plan: TilePlan,
    *,
    causal: bool = False,
    has_mask: bool = False,
    budget_bytes: int = 0,
    safety_factor: float = 1.0,
    reserved_bytes: int = 0,
) -> Budget:
    """Upper bound on simultaneously live bytes inside one unrolled tile.

    The schedule is split into stages and each stage is summed independently,
    which assumes the backend fuses elementwise chains and reuses buffers
    between stages.  This is deliberately conservative but not paranoid: SSA
    values inside a stage are never assumed to alias.
    """

    elems = shape.elem_bytes * shape.bh
    qt, kt, d = plan.q_tile, plan.kv_tile, shape.head_dim

    stages = {
        # q slice -> scaled q (scale constant is materialised as a tile).
        "prep": elems * (3 * qt * d),
        # score computation: scaled q, k tile, score, optional mask tiles.
        "scores": elems
        * (
            qt * d
            + kt * d
            + 2 * qt * kt
            + (2 * qt * kt if causal else 0)
            + (2 * qt * kt if has_mask else 0)
        ),
        # softmax: score, broadcast max, probabilities, running state.
        "softmax": elems * (3 * qt * kt + 4 * qt),
        # value: probabilities, v tile, accumulator, rescaled accumulator, pv.
        "value": elems * (qt * kt + kt * d + 3 * qt * d),
        # normalise by the running softmax denominator.
        "epilogue": elems * (2 * qt * d),
    }
    return Budget(
        budget_bytes=budget_bytes,
        safety_factor=safety_factor,
        reserved_bytes=reserved_bytes,
        stages=stages,
    )


def _plan_key(shape: AttentionShape, plan: TilePlan) -> tuple[int, int, int, int]:
    """Prefer the least padding, then the largest tile (fewest unrolled ops)."""

    padded_work = plan.padded_seq(shape) * plan.padded_kv(shape)
    return (padded_work, -(plan.q_tile * plan.kv_tile), plan.q_tile, plan.kv_tile)


def select_plan(
    shape: AttentionShape,
    *,
    budget_bytes: int,
    causal: bool = False,
    has_mask: bool = False,
    q_tile: int | None = None,
    kv_tile: int | None = None,
    q_candidates: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    kv_candidates: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024),
    safety_factor: float = 1.25,
    reserved_bytes: int = 0,
    max_tiles: int = MAX_UNROLLED_TILES,
) -> tuple[TilePlan, Budget]:
    """Pick tile sizes that fit ``budget_bytes``.

    Explicit ``q_tile``/``kv_tile`` values win, but are still validated against
    the budget.  When only one is given the other is chosen automatically.
    """

    def check(plan: TilePlan) -> tuple[TilePlan, Budget]:
        counts = plan.tile_counts(shape)
        if counts[0] * counts[1] > max_tiles:
            raise TilePlanError(
                f"tile plan q={plan.q_tile} kv={plan.kv_tile} unrolls to "
                f"{counts[0] * counts[1]} steps (limit {max_tiles}); "
                "increase --q-tile/--kv-tile or raise --max-unrolled-tiles"
            )
        budget = estimate_budget(
            shape,
            plan,
            causal=causal,
            has_mask=has_mask,
            budget_bytes=budget_bytes,
            safety_factor=safety_factor,
            reserved_bytes=reserved_bytes,
        )
        if not budget.fits:
            stages = ", ".join(f"{k}={v}" for k, v in budget.stages.items())
            raise TilePlanError(
                f"tile q={plan.q_tile} kv={plan.kv_tile} needs "
                f"{budget.required_bytes} B but the tile budget is "
                f"{budget.budget_bytes} B (peak {budget.peak_bytes} B * "
                f"{safety_factor} + {reserved_bytes} B reserved; {stages})"
            )
        if has_mask and not causal and shape.kv_seq % plan.kv_tile != 0:
            raise TilePlanError(
                f"an additive mask cannot be tiled with kv_tile={plan.kv_tile} "
                f"because kv_seq={shape.kv_seq} is not a multiple of it; "
                "use --causal (padded keys are masked) or pick a divisor"
            )
        return plan, budget

    if q_tile and kv_tile:
        return check(TilePlan(q_tile, kv_tile))

    if q_tile:
        best = None
        for kt in kv_candidates:
            if kt > shape.kv_seq:
                continue
            try:
                plan, budget = check(TilePlan(q_tile, kt))
            except TilePlanError:
                continue
            key = _plan_key(shape, plan)
            if best is None or key < best[0]:
                best = (key, plan, budget)
        if best is not None:
            return best[1], best[2]
        return check(TilePlan(q_tile, kv_candidates[0]))

    if kv_tile:
        best: tuple[tuple[int, int, int, int], TilePlan, Budget] | None = None
        for qt in q_candidates:
            if qt > shape.seq:
                continue
            try:
                plan, budget = check(TilePlan(qt, kv_tile))
            except TilePlanError:
                continue
            key = _plan_key(shape, plan)
            if best is None or key < best[0]:
                best = (key, plan, budget)
        if best is not None:
            return best[1], best[2]
        return check(TilePlan(q_candidates[0], kv_tile))

    best = None
    for qt in q_candidates:
        if qt > shape.seq:
            continue
        for kt in kv_candidates:
            if kt > shape.kv_seq:
                continue
            try:
                plan, budget = check(TilePlan(qt, kt))
            except TilePlanError:
                continue
            key = _plan_key(shape, plan)
            if best is None or key < best[0]:
                best = (key, plan, budget)
    if best is not None:
        return best[1], best[2]

    smallest = TilePlan(min(q_candidates), min(kv_candidates))
    budget = estimate_budget(
        shape,
        smallest,
        causal=causal,
        has_mask=has_mask,
        budget_bytes=budget_bytes,
        safety_factor=safety_factor,
        reserved_bytes=reserved_bytes,
    )
    raise TilePlanError(
        "no tile configuration fits the budget: the smallest plan "
        f"(q={smallest.q_tile}, kv={smallest.kv_tile}) requires "
        f"{budget.required_bytes} B for shape {shape}; raise --budget-bytes, "
        "reduce heads/head_dim, or split the head dimension across functions"
    )


# ---------------------------------------------------------------------------
# IR builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Value:
    name: str
    type: str
    shape: tuple[int, ...]
    dtype: str


class Builder:
    """Tiny SSA builder that keeps result types in sync with TOSA semantics.

    Targets the current TOSA dialect where shape parameters (slice starts,
    reshape targets, tile multiples, padding) are ``!tosa.shape<N>`` SSA
    operands produced by ``tosa.const_shape``, reductions keep their axis, and
    matmul carries explicit zero-point operands.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._names = _names()
        self._zero_points: dict[str, Value] = {}
        self._shift: Value | None = None
        self._shapes: dict[tuple[int, ...], Value] = {}

    # -- plumbing ---------------------------------------------------------
    def comment(self, text: str) -> None:
        self.lines.append(f"  // {text}")

    def _emit(
        self,
        shape: Sequence[int],
        dtype: str,
        op: str,
        operands: Sequence[Value],
        attrs: Sequence[tuple[str, str]] = (),
        suffix: str = "",
    ) -> Value:
        result = Value(
            next(self._names), tensor_type(shape, dtype), tuple(shape), dtype
        )
        attr_text = ""
        if attrs:
            attr_text = " {" + ", ".join(f"{k} = {v}" for k, v in attrs) + "}"
        operand_text = ", ".join(v.name for v in operands)
        operand_types = ", ".join(v.type for v in operands)
        self.lines.append(
            f"  {result.name} = {op} {operand_text}{suffix}{attr_text} : "
            f"({operand_types}) -> {result.type}"
        )
        return result

    def _const(self, literal: str, shape: Sequence[int], dtype: str) -> Value:
        result = Value(
            next(self._names), tensor_type(shape, dtype), tuple(shape), dtype
        )
        self.lines.append(f"  {result.name} = arith.constant {literal} : {result.type}")
        return result

    def shape(self, values: Sequence[int]) -> Value:
        dims = tuple(int(v) for v in values)
        cached = self._shapes.get(dims)
        if cached is not None:
            return cached
        result = Value(next(self._names), f"!tosa.shape<{len(dims)}>", dims, "index")
        self._shapes[dims] = result
        literal = (
            "dense<["
            + ", ".join(str(v) for v in dims)
            + f"]> : tensor<{len(dims)}xindex>"
        )
        self.lines.append(
            f"  {result.name} = tosa.const_shape values({literal}) : () -> {result.type}"
        )
        return result

    def shift(self) -> Value:
        if self._shift is None:
            self._shift = self._const("dense<0>", (1,), "i8")
        return self._shift

    def zero_point(self, dtype: str) -> Value:
        if dtype not in self._zero_points:
            self._zero_points[dtype] = self._const(
                f"dense<{_fmt_float(0.0)}>", (1,), dtype
            )
        return self._zero_points[dtype]

    # -- constants --------------------------------------------------------
    def splat(self, value: float, shape: Sequence[int], dtype: str) -> Value:
        return self._const(f"dense<{_fmt_float(value)}>", shape, dtype)

    def dense2d(self, rows: Sequence[Sequence[float]], dtype: str) -> Value:
        shape = (1, len(rows), len(rows[0]))
        return self._const(
            "dense<" + _fmt_nested([[list(row) for row in rows]]) + ">", shape, dtype
        )

    # -- shape ops --------------------------------------------------------
    def slice(self, value: Value, start: Sequence[int], size: Sequence[int]) -> Value:
        start_t = tuple(int(s) for s in start)
        size_t = tuple(int(s) for s in size)
        for axis, (s, n) in enumerate(zip(start_t, size_t)):
            if s < 0 or n < 1 or s + n > value.shape[axis]:
                raise ValueError(
                    f"slice {start_t} + {size_t} out of bounds for {value.shape} "
                    f"on axis {axis}"
                )
        return self._emit(
            size_t,
            value.dtype,
            "tosa.slice",
            [value, self.shape(start_t), self.shape(size_t)],
        )

    def reshape(self, value: Value, shape: Sequence[int]) -> Value:
        new_shape = tuple(int(d) for d in shape)
        if math.prod(new_shape) != math.prod(value.shape):
            raise ValueError(f"cannot reshape {value.shape} to {new_shape}")
        return self._emit(
            new_shape, value.dtype, "tosa.reshape", [value, self.shape(new_shape)]
        )

    def transpose(self, value: Value, perms: Sequence[int]) -> Value:
        perm = tuple(int(p) for p in perms)
        if sorted(perm) != list(range(len(value.shape))):
            raise ValueError(f"invalid permutation {perm} for {value.shape}")
        return self._emit(
            tuple(value.shape[p] for p in perm),
            value.dtype,
            "tosa.transpose",
            [value],
            suffix=" perms([" + ", ".join(str(p) for p in perm) + "])",
        )

    def tile(self, value: Value, multiples: Sequence[int]) -> Value:
        mult = tuple(int(m) for m in multiples)
        if len(mult) != len(value.shape):
            raise ValueError(f"tile multiples {mult} do not match {value.shape}")
        return self._emit(
            tuple(d * m for d, m in zip(value.shape, mult)),
            value.dtype,
            "tosa.tile",
            [value, self.shape(mult)],
        )

    def pad(self, value: Value, padding: Sequence[int]) -> Value:
        pads = tuple(int(p) for p in padding)
        if len(pads) != 2 * len(value.shape):
            raise ValueError(f"padding {pads} does not match rank of {value.shape}")
        new_shape = tuple(
            value.shape[i] + pads[2 * i] + pads[2 * i + 1]
            for i in range(len(value.shape))
        )
        return self._emit(
            new_shape,
            value.dtype,
            "tosa.pad",
            [value, self.shape(pads), self.zero_point(value.dtype)],
        )

    def concat(self, values: Sequence[Value], axis: int) -> Value:
        if not values:
            raise ValueError("concat requires at least one value")
        axis = int(axis)
        ref = values[0]
        for other in values[1:]:
            if (
                other.shape[:axis] + other.shape[axis + 1 :]
                != ref.shape[:axis] + ref.shape[axis + 1 :]
            ):
                raise ValueError("concat inputs must agree on every non-concat axis")
        shape = list(ref.shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        return self._emit(
            tuple(shape),
            ref.dtype,
            "tosa.concat",
            list(values),
            suffix=f" axis({axis})",
        )

    def cast(self, value: Value, dtype: str) -> Value:
        return self._emit(value.shape, dtype, "tosa.cast", [value])

    # -- arithmetic -------------------------------------------------------
    def _same(self, a: Value, b: Value) -> None:
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(
                f"shape mismatch: {a.shape}{a.dtype} vs {b.shape}{b.dtype}"
            )

    def add(self, a: Value, b: Value) -> Value:
        self._same(a, b)
        return self._emit(a.shape, a.dtype, "tosa.add", [a, b])

    def sub(self, a: Value, b: Value) -> Value:
        self._same(a, b)
        return self._emit(a.shape, a.dtype, "tosa.sub", [a, b])

    def mul(self, a: Value, b: Value) -> Value:
        self._same(a, b)
        return self._emit(a.shape, a.dtype, "tosa.mul", [a, b, self.shift()])

    def maximum(self, a: Value, b: Value) -> Value:
        self._same(a, b)
        return self._emit(a.shape, a.dtype, "tosa.maximum", [a, b])

    def exp(self, value: Value) -> Value:
        return self._emit(value.shape, value.dtype, "tosa.exp", [value])

    def reciprocal(self, value: Value) -> Value:
        return self._emit(value.shape, value.dtype, "tosa.reciprocal", [value])

    def reduce(self, value: Value, axis: int, kind: str) -> Value:
        axis = int(axis)
        shape = list(value.shape)
        shape[axis] = 1
        return self._emit(
            tuple(shape),
            value.dtype,
            f"tosa.reduce_{kind}",
            [value],
            suffix=f" axis({axis})",
        )

    def matmul(self, a: Value, b: Value) -> Value:
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise ValueError("TOSA matmul requires rank-3 operands")
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
            raise ValueError(f"matmul shape mismatch: {a.shape} x {b.shape}")
        zero_point = self.zero_point(a.dtype)
        return self._emit(
            (a.shape[0], a.shape[1], b.shape[2]),
            a.dtype,
            "tosa.matmul",
            [a, b, zero_point, zero_point],
        )


# ---------------------------------------------------------------------------
# Signature analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str

    @property
    def shape(self) -> tuple[int, ...]:
        return parse_tensor_type(self.type)[0]

    @property
    def dtype(self) -> str:
        return parse_tensor_type(self.type)[1]


@dataclass
class AttentionSpec:
    shape: AttentionShape
    layout: str  # "bshd" | "bhsd" | "packed"
    mask_arg: ArgSpec | None = None
    mask_layout: str | None = None  # "b1sd" | "bhsd" | "packed"


def _infer_layout(arg: ArgSpec, requested: str) -> tuple[str, tuple[int, ...]]:
    shape = arg.shape
    if len(shape) == 3:
        return "packed", shape
    if len(shape) != 4:
        raise ValueError(
            f"{arg.name}: expected rank-3 or rank-4 tensor, got {arg.type}"
        )
    if requested == "packed":
        raise ValueError(f"{arg.name}: rank-4 tensor cannot use the packed layout")
    if requested in ("bshd", "bhsd"):
        return requested, shape
    # auto: [B, S, H, D] is the LiteRT/PyTorch convention.  When S and H are
    # equal (small test shapes) prefer it as well.
    return "bshd", shape


def analyze_signature(
    args: Sequence[ArgSpec],
    result_type: str | None,
    *,
    layout: str = "auto",
    dtype: str | None = None,
    q_offset: int | None = None,
) -> AttentionSpec:
    """Map a ``(q, k, v[, mask]) -> out`` signature onto a packed MHSA problem."""

    if len(args) not in (3, 4):
        raise ValueError(
            "attention function must take (query, key, value) with an optional "
            f"additive mask, got {len(args)} arguments"
        )
    q, k, v = args[0], args[1], args[2]
    mask = args[3] if len(args) == 4 else None

    q_layout, q_shape = _infer_layout(q, layout)
    k_layout, k_shape = _infer_layout(k, layout)
    v_layout, v_shape = _infer_layout(v, layout)
    if not (q_layout == k_layout == v_layout):
        raise ValueError("query/key/value must share the same layout")

    data_dtype = dtype or q.dtype
    for arg in (q, k, v):
        if arg.dtype != data_dtype:
            raise ValueError(
                f"{arg.name}: expected dtype {data_dtype}, got {arg.dtype}; "
                "cast the attention inputs to one dtype first"
            )

    if q_layout == "packed":
        bh, seq, head_dim = q_shape
        kv_bh, kv_seq, kv_dim = k_shape
        if kv_bh != bh or kv_dim != head_dim:
            raise ValueError(
                f"packed layout mismatch: q{bh}x{seq}x{head_dim} vs "
                f"k{kv_bh}x{kv_seq}x{kv_dim}"
            )
        batch, heads = 1, bh
    else:
        if q_layout == "bshd":
            batch, seq, heads, head_dim = q_shape
            kv_batch, kv_seq, kv_heads, kv_dim = k_shape
        else:  # bhsd
            batch, heads, seq, head_dim = q_shape
            kv_batch, kv_heads, kv_seq, kv_dim = k_shape
        if (kv_batch, kv_heads, kv_dim) != (batch, heads, head_dim):
            raise ValueError(
                f"layout mismatch: q{q_shape} vs k{k_shape} "
                f"(expected the same batch/head/head_dim as the query)"
            )
        if v_shape != k_shape:
            raise ValueError(f"value shape {v_shape} must match key shape {k_shape}")

    shape = AttentionShape(
        batch=batch,
        heads=heads,
        seq=seq,
        kv_seq=kv_seq,
        head_dim=head_dim,
        dtype=data_dtype,
        q_offset=q_offset,
    )

    mask_layout = None
    if mask is not None:
        mshape = mask.shape
        if mask.dtype not in ("f16", "bf16", "f32", "f64"):
            raise ValueError(
                f"{mask.name}: additive masks must be floating point, got {mask.dtype}"
            )
        if len(mshape) == 3:
            mask_layout = "packed"
            expected = (shape.bh, seq, kv_seq)
        elif len(mshape) == 4 and mshape[1] == 1:
            mask_layout = "b1sd"
            expected = (batch, 1, seq, kv_seq)
        elif len(mshape) == 4:
            mask_layout = "bhsd"
            expected = (batch, heads, seq, kv_seq)
        else:
            expected = ()
        if tuple(mshape) != expected:
            raise ValueError(
                f"{mask.name}: additive mask shape {mshape} does not match the "
                f"attention problem; expected {expected}"
            )

    if result_type is not None and result_type != q.type:
        raise ValueError(
            f"attention result type {result_type} does not match the query type "
            f"{q.type}; the tiled kernel preserves the query shape"
        )

    return AttentionSpec(
        shape=shape, layout=q_layout, mask_arg=mask, mask_layout=mask_layout
    )


# ---------------------------------------------------------------------------
# Kernel emission
# ---------------------------------------------------------------------------


def _causal_rows(
    shape: AttentionShape,
    q_start: int,
    kv_start: int,
    q_tile: int,
    kv_tile: int,
) -> list[list[float]]:
    """Additive causal mask for one tile (0.0 visible, -large masked)."""

    neg = NEG_MASK_VALUE[shape.dtype]
    rows: list[list[float]] = []
    origin = shape.query_offset + q_start
    for r in range(q_tile):
        query_pos = origin + r
        rows.append(
            [0.0 if (kv_start + c) <= query_pos else neg for c in range(kv_tile)]
        )
    return rows


def emit_kernel(
    shape: AttentionShape,
    plan: TilePlan,
    *,
    causal: bool = False,
    mask_shape: Sequence[int] | None = None,
    mask_dtype: str | None = None,
    func_name: str = "mhsa_tiled",
    budget_bytes: int = 0,
    safety_factor: float = 1.0,
    reserved_bytes: int = 0,
) -> str:
    """Emit the packed, statically tiled attention kernel."""

    if mask_shape is not None and mask_dtype is None:
        raise ValueError("mask_dtype is required when mask_shape is given")

    dt = shape.dtype
    bh, seq, kv_seq, head_dim = shape.bh, shape.seq, shape.kv_seq, shape.head_dim
    qt, kt = plan.q_tile, plan.kv_tile
    seq_pad = plan.padded_seq(shape)
    kv_pad = plan.padded_kv(shape)
    q_tiles, kv_tiles = plan.tile_counts(shape)

    budget = estimate_budget(
        shape,
        plan,
        causal=causal,
        has_mask=mask_shape is not None,
        budget_bytes=budget_bytes,
        safety_factor=safety_factor,
        reserved_bytes=reserved_bytes,
    )

    b = Builder()
    q_type = tensor_type([bh, seq, head_dim], dt)
    k_type = tensor_type([bh, kv_seq, head_dim], dt)
    v_type = k_type
    args = [("%q", q_type), ("%k", k_type), ("%v", v_type)]
    if mask_shape is not None:
        args.append(("%mask", tensor_type(mask_shape, mask_dtype)))

    q = Value("%q", q_type, (bh, seq, head_dim), dt)
    k = Value("%k", k_type, (bh, kv_seq, head_dim), dt)
    v = Value("%v", v_type, (bh, kv_seq, head_dim), dt)
    mask = (
        Value(
            "%mask", tensor_type(mask_shape, mask_dtype), tuple(mask_shape), mask_dtype
        )
        if mask_shape is not None
        else None
    )

    b.comment(
        f"tile_q={qt} tile_kv={kt} padded_seq={seq_pad} padded_kv={kv_pad} "
        f"steps={q_tiles * kv_tiles}"
    )

    if seq_pad != seq:
        q = b.pad(q, [0, 0, 0, seq_pad - seq, 0, 0])
    if kv_pad != kv_seq:
        padding = [0, 0, 0, kv_pad - kv_seq, 0, 0]
        k = b.pad(k, padding)
        v = b.pad(v, padding)
    if mask is not None and seq_pad != seq:
        # Rows past the end of the sequence are dropped from the result; their
        # mask values never matter.  The sequence dim is the second-to-last.
        pad = [0] * (2 * len(mask.shape))
        pad[2 * (len(mask.shape) - 2) + 1] = seq_pad - seq
        mask = b.pad(mask, pad)

    scale = 1.0 / math.sqrt(head_dim)
    scale_t = b.tile(b.splat(scale, (1, 1, 1), dt), (bh, qt, head_dim))
    zero_state = b.splat(0.0, (bh, qt, 1), dt)
    epsilon_value = {"f16": 6.0e-5, "bf16": 1.0e-6, "f32": 1.0e-12, "f64": 1.0e-12}[dt]
    epsilon = b.splat(epsilon_value, (bh, qt, 1), dt)

    out_tiles: list[Value] = []
    for q_index in range(q_tiles):
        q_start = q_index * qt
        q_slice = b.slice(q, (0, q_start, 0), (bh, qt, head_dim))
        q_scaled = b.mul(q_slice, scale_t)

        acc: Value | None = None
        row_max: Value | None = None
        row_sum: Value | None = None

        for kv_index in range(kv_tiles):
            kv_start = kv_index * kt
            b.comment(f"q_tile={q_index} kv_tile={kv_index}")

            k_slice = b.slice(k, (0, kv_start, 0), (bh, kt, head_dim))
            k_t = b.transpose(k_slice, (0, 2, 1))
            scores = b.matmul(q_scaled, k_t)

            if mask is not None:
                if len(mask.shape) == 3:
                    m_slice = b.slice(mask, (0, q_start, kv_start), (bh, qt, kt))
                else:
                    m_slice = b.slice(
                        mask,
                        (0, 0, q_start, kv_start),
                        (mask.shape[0], mask.shape[1], qt, kt),
                    )
                    m_slice = b.reshape(
                        m_slice, (mask.shape[0] * mask.shape[1], qt, kt)
                    )
                    if mask.shape[1] == 1:
                        m_slice = b.tile(m_slice, (shape.heads, 1, 1))
                if m_slice.shape != (bh, qt, kt):
                    raise ValueError(
                        f"internal mask tiling produced {m_slice.shape}, "
                        f"expected {(bh, qt, kt)}"
                    )
                if m_slice.dtype != dt:
                    m_slice = b.cast(m_slice, dt)
                scores = b.add(scores, m_slice)

            if causal:
                mask_rows = _causal_rows(shape, q_start, kv_start, qt, kt)
                c_mask = b.dense2d(mask_rows, dt)
                c_mask = b.tile(c_mask, (bh, 1, 1))
                scores = b.add(scores, c_mask)

            tile_max = b.reduce(scores, 2, "max")
            v_slice = b.slice(v, (0, kv_start, 0), (bh, kt, head_dim))

            if acc is None:
                # Clamping the initial maximum to >= 0 is exact (softmax is
                # invariant to a per-row constant) and keeps a fully masked
                # first tile from turning -1e30 + 1e30 into exp(0) == 1.
                row_max = b.maximum(tile_max, zero_state)
                probs = b.exp(b.sub(scores, b.tile(row_max, (1, 1, kt))))
                acc = b.matmul(probs, v_slice)
                row_sum = b.reduce(probs, 2, "sum")
            else:
                assert row_max is not None and row_sum is not None
                new_max = b.maximum(row_max, tile_max)
                alpha = b.exp(b.sub(row_max, new_max))
                probs = b.exp(b.sub(scores, b.tile(new_max, (1, 1, kt))))
                row_sum = b.add(
                    b.mul(row_sum, alpha),
                    b.reduce(probs, 2, "sum"),
                )
                acc = b.add(
                    b.mul(acc, b.tile(alpha, (1, 1, head_dim))),
                    b.matmul(probs, v_slice),
                )
                row_max = new_max

        assert acc is not None and row_sum is not None
        # A fully masked row has denominator 0; keep the reciprocal finite and
        # let the numerator (also 0) define the output as 0.
        normalised = b.mul(
            acc, b.tile(b.reciprocal(b.maximum(row_sum, epsilon)), (1, 1, head_dim))
        )
        out_tiles.append(normalised)

    if len(out_tiles) == 1:
        out = out_tiles[0]
    else:
        out = b.concat(out_tiles, 1)
    if seq_pad != seq:
        out = b.slice(out, (0, 0, 0), (bh, seq, head_dim))

    attrs = [
        ("tile_q", f"{qt} : i64"),
        ("tile_kv", f"{kt} : i64"),
        ("tile_bytes", f"{int(budget_bytes)} : i64"),
        ("peak_bytes", f"{budget.peak_bytes} : i64"),
        ("required_bytes", f"{budget.required_bytes} : i64"),
        ("unrolled_steps", f"{q_tiles * kv_tiles} : i64"),
    ]
    header = (
        f"func.func @{func_name}("
        + ", ".join(f"{name}: {typ}" for name, typ in args)
        + f") -> {q_type} attributes {{"
        + ", ".join(f"{k} = {v}" for k, v in attrs)
        + "} {"
    )

    lines = [header, *b.lines, f"  return {out.name} : {out.type}", "}"]
    return "\n".join(lines) + "\n"


def emit_wrapper(
    signature,
    spec: AttentionSpec,
    plan: TilePlan,
    kernel_name: str,
    *,
    causal: bool = False,
    budget_bytes: int = 0,
    safety_factor: float = 1.0,
    reserved_bytes: int = 0,
) -> str:
    """Emit the kernel plus a function with the original signature.

    The original function is replaced in place; call sites keep working because
    argument and result types are preserved.
    """

    args = signature.args
    q, k, v = args[0], args[1], args[2]
    out_type = signature.result_type or q.type

    b = Builder()
    packed: list[Value] = []
    for arg in (q, k, v):
        value = Value(arg.name, arg.type, arg.shape, arg.dtype)
        shape = value.shape
        if spec.layout == "bshd":
            value = b.transpose(value, (0, 2, 1, 3))
            value = b.reshape(value, (shape[0] * shape[2], shape[1], shape[3]))
        elif spec.layout == "bhsd":
            value = b.reshape(value, (shape[0] * shape[1], shape[2], shape[3]))
        packed.append(value)

    call_args = list(packed)
    if spec.mask_arg is not None:
        call_args.append(
            Value(
                spec.mask_arg.name,
                spec.mask_arg.type,
                spec.mask_arg.shape,
                spec.mask_arg.dtype,
            )
        )
    result_type = tensor_type(
        (spec.shape.bh, spec.shape.seq, spec.shape.head_dim), spec.shape.dtype
    )
    operand_types = ", ".join(val.type for val in call_args)
    b.lines.append(
        f"  %packed_out = func.call @{kernel_name}("
        + ", ".join(val.name for val in call_args)
        + f") : ({operand_types}) -> {result_type}"
    )

    out = Value(
        "%packed_out",
        result_type,
        (spec.shape.bh, spec.shape.seq, spec.shape.head_dim),
        spec.shape.dtype,
    )
    if spec.layout == "bshd":
        out = b.reshape(
            out,
            (spec.shape.batch, spec.shape.heads, spec.shape.seq, spec.shape.head_dim),
        )
        out = b.transpose(out, (0, 2, 1, 3))
    elif spec.layout == "bhsd":
        out = b.reshape(
            out,
            (spec.shape.batch, spec.shape.heads, spec.shape.seq, spec.shape.head_dim),
        )

    if out.type != out_type:
        raise ValueError(f"generated result {out.type} does not match {out_type}")

    header = (
        f"func.func {signature.visibility + ' ' if signature.visibility else ''}"
        f"@{signature.name}(" + ", ".join(f"{arg.name}: {arg.type}" for arg in args)
    )
    if signature.result_type is not None:
        header += f") -> {signature.result_type}"
    else:
        header += ")"
    if signature.header_attrs:
        header += f" {signature.header_attrs}"
    header += " {"

    body = [
        header,
        *b.lines,
        f"  return {out.name} : {out.type}",
        "}",
    ]
    kernel = emit_kernel(
        spec.shape,
        plan,
        causal=causal,
        mask_shape=spec.mask_arg.shape if spec.mask_arg else None,
        mask_dtype=spec.mask_arg.dtype if spec.mask_arg else None,
        func_name=kernel_name,
        budget_bytes=budget_bytes,
        safety_factor=safety_factor,
        reserved_bytes=reserved_bytes,
    )
    return kernel + "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# CLI (standalone emitter)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a statically tiled TOSA MHSA kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--kv-seq", type=int)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--q-offset",
        type=int,
        help="absolute KV position of query row 0 (default: kv_seq - seq)",
    )
    parser.add_argument("--dtype", default="f32", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--q-tile", type=int)
    parser.add_argument("--kv-tile", type=int)
    parser.add_argument("--budget-bytes", type=int, default=256 * 1024)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--reserved-bytes", type=int, default=4096)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--mask", action="store_true", help="take an additive mask argument"
    )
    parser.add_argument("--func-name", default="mhsa_tiled")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="print the tile plan as JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    shape = AttentionShape(
        batch=args.batch,
        heads=args.heads,
        seq=args.seq,
        kv_seq=args.kv_seq or args.seq,
        head_dim=args.head_dim,
        dtype=args.dtype,
        q_offset=args.q_offset,
    )
    try:
        plan, budget = select_plan(
            shape,
            budget_bytes=args.budget_bytes,
            causal=args.causal,
            has_mask=args.mask,
            q_tile=args.q_tile,
            kv_tile=args.kv_tile,
            safety_factor=args.safety_factor,
            reserved_bytes=args.reserved_bytes,
        )
    except TilePlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mask_shape = None
    if args.mask:
        if args.causal:
            print("error: --mask and --causal are mutually exclusive", file=sys.stderr)
            return 2
        mask_shape = (shape.batch, 1, shape.seq, shape.kv_seq)

    text = emit_kernel(
        shape,
        plan,
        causal=args.causal,
        mask_shape=mask_shape,
        mask_dtype=shape.dtype,
        func_name=args.func_name,
        budget_bytes=args.budget_bytes,
        safety_factor=args.safety_factor,
        reserved_bytes=args.reserved_bytes,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "shape": shape.__dict__,
                    "tile_plan": plan.__dict__,
                    "budget": budget.to_dict(),
                },
                indent=2,
            )
        )
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        q_tiles, kv_tiles = plan.tile_counts(shape)
        print(
            f"wrote {args.output} (q_tile={plan.q_tile} kv_tile={plan.kv_tile}, "
            f"{q_tiles}x{kv_tiles} steps, required {budget.required_bytes} B "
            f"of {budget.budget_bytes} B)"
        )
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
