"""Unit tests for the tile planner and TOSA attention emitter."""

from __future__ import annotations

import math

import pytest

from mlir_text import find_function
from tosa_mhsa import (
    NEG_MASK_VALUE,
    ArgSpec,
    AttentionShape,
    TilePlan,
    TilePlanError,
    _causal_rows,
    analyze_signature,
    ceil_div,
    emit_kernel,
    emit_wrapper,
    estimate_budget,
    parse_tensor_type,
    select_plan,
    tensor_type,
)


def make_shape(**overrides) -> AttentionShape:
    values = dict(batch=1, heads=2, seq=8, kv_seq=8, head_dim=4)
    values.update(overrides)
    return AttentionShape(**values)


# ---------------------------------------------------------------------------
# Types and shapes
# ---------------------------------------------------------------------------


def test_ceil_div() -> None:
    assert ceil_div(0, 4) == 0
    assert ceil_div(1, 4) == 1
    assert ceil_div(4, 4) == 1
    assert ceil_div(5, 4) == 2


def test_tensor_type_round_trip() -> None:
    text = tensor_type([1, 8, 2, 4], "f32")
    assert text == "tensor<1x8x2x4xf32>"
    assert parse_tensor_type(text) == ((1, 8, 2, 4), "f32")


@pytest.mark.parametrize("text", ["tensor<?x4xf32>", "tensor<4xf32", "f32", ""])
def test_parse_tensor_type_rejects_non_static(text: str) -> None:
    with pytest.raises(ValueError):
        parse_tensor_type(text)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": "int8"},
        {"batch": 0},
        {"heads": 0},
        {"seq": 0},
        {"kv_seq": 0},
        {"head_dim": 0},
        {"seq": 4, "kv_seq": 3},
        {"seq": 2, "kv_seq": 4, "q_offset": 3},
        {"seq": 2, "kv_seq": 4, "q_offset": -1},
    ],
)
def test_shape_rejects_invalid_values(overrides) -> None:
    with pytest.raises(ValueError):
        make_shape(**overrides)


def test_shape_derived_values() -> None:
    shape = make_shape(batch=2, heads=3, seq=2, kv_seq=8)
    assert shape.bh == 6
    assert shape.elem_bytes == 4
    assert shape.query_offset == 6  # kv_seq - seq
    assert make_shape(seq=2, kv_seq=8, q_offset=3).query_offset == 3


# ---------------------------------------------------------------------------
# Budget model
# ---------------------------------------------------------------------------


def test_budget_stage_arithmetic() -> None:
    shape = make_shape()
    budget = estimate_budget(
        shape,
        TilePlan(4, 4),
        causal=True,
        has_mask=True,
        budget_bytes=1 << 20,
        safety_factor=1.0,
        reserved_bytes=0,
    )
    elems = shape.elem_bytes * shape.bh
    qt = kt = 4
    d = shape.head_dim
    assert budget.stages == {
        "prep": elems * (3 * qt * d),
        "scores": elems
        * (qt * d + kt * d + 2 * qt * kt + 2 * qt * kt + 2 * qt * kt),
        "softmax": elems * (3 * qt * kt + 4 * qt),
        "value": elems * (qt * kt + kt * d + 3 * qt * d),
        "epilogue": elems * (2 * qt * d),
    }
    assert budget.peak_bytes == max(budget.stages.values())
    assert budget.required_bytes == budget.peak_bytes
    assert budget.fits


def test_budget_safety_factor_and_reserve() -> None:
    shape = make_shape()
    budget = estimate_budget(
        shape,
        TilePlan(4, 4),
        budget_bytes=1 << 20,
        safety_factor=1.25,
        reserved_bytes=4096,
    )
    assert budget.required_bytes == math.ceil(budget.peak_bytes * 1.25) + 4096
    assert budget.to_dict()["fits"] is True


def test_budget_does_not_fit() -> None:
    budget = estimate_budget(make_shape(), TilePlan(4, 4), budget_bytes=1)
    assert not budget.fits


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_padding_and_tile_counts() -> None:
    shape = make_shape(seq=5, kv_seq=7)
    plan = TilePlan(2, 4)
    assert plan.padded_seq(shape) == 6
    assert plan.padded_kv(shape) == 8
    assert plan.tile_counts(shape) == (3, 2)


def test_select_plan_prefers_largest_tile_without_padding() -> None:
    plan, budget = select_plan(make_shape(), budget_bytes=1 << 20, causal=True)
    assert plan.q_tile == 8
    assert plan.kv_tile == 8
    assert budget.fits


def test_select_plan_explicit_tiles_are_honoured() -> None:
    plan, _ = select_plan(
        make_shape(), budget_bytes=1 << 20, q_tile=4, kv_tile=8, causal=True
    )
    assert plan == TilePlan(4, 8)


def test_select_plan_explicit_plan_must_fit() -> None:
    with pytest.raises(TilePlanError, match="needs .* budget"):
        select_plan(make_shape(), budget_bytes=1, q_tile=8, kv_tile=8)


def test_select_plan_only_q_tile_chooses_kv() -> None:
    plan, _ = select_plan(make_shape(), budget_bytes=1 << 20, q_tile=2)
    assert plan.q_tile == 2
    assert plan.kv_tile == 8


def test_select_plan_only_kv_tile_chooses_q() -> None:
    plan, _ = select_plan(make_shape(), budget_bytes=1 << 20, kv_tile=8)
    assert plan.kv_tile == 8
    assert plan.q_tile == 8


def test_select_plan_unroll_limit() -> None:
    with pytest.raises(TilePlanError, match="unrolls to"):
        select_plan(
            make_shape(), budget_bytes=1 << 20, q_tile=1, kv_tile=1, max_tiles=4
        )


def test_select_plan_no_configuration_fits() -> None:
    with pytest.raises(TilePlanError, match="no tile configuration fits"):
        select_plan(make_shape(), budget_bytes=1, causal=True)


def test_select_plan_rejects_inexact_additive_mask_tiles() -> None:
    with pytest.raises(TilePlanError, match="cannot be tiled"):
        select_plan(
            make_shape(seq=5, kv_seq=7),
            budget_bytes=1 << 20,
            q_tile=4,
            kv_tile=4,
            causal=False,
            has_mask=True,
        )


# ---------------------------------------------------------------------------
# Kernel emission
# ---------------------------------------------------------------------------


def test_emit_kernel_shape_and_attributes() -> None:
    kernel = emit_kernel(
        make_shape(seq=4, kv_seq=4),
        TilePlan(2, 2),
        causal=True,
        func_name="attn",
        budget_bytes=1 << 20,
    )
    assert "func.func @attn(" in kernel
    assert "tosa.const_shape" in kernel
    assert "scf." not in kernel
    assert "tile_q = 2 : i64" in kernel
    assert "tile_kv = 2 : i64" in kernel
    assert "unrolled_steps = 4 : i64" in kernel
    assert "tile_bytes = 1048576 : i64" in kernel
    # Two matmuls (scores and P·V) per one of the 2x2 unrolled steps.
    assert kernel.count("tosa.matmul") == 8


def test_emit_kernel_pads_and_slices_back() -> None:
    kernel = emit_kernel(make_shape(seq=3, kv_seq=3), TilePlan(2, 2), causal=True)
    assert "tosa.pad" in kernel
    assert "tensor<2x3x4xf32>" in kernel  # result is unpadded
    assert "tosa.slice" in kernel


def test_emit_kernel_causal_mask_is_optional() -> None:
    causal = emit_kernel(make_shape(), TilePlan(4, 4), causal=True)
    plain = emit_kernel(make_shape(), TilePlan(4, 4), causal=False)
    assert NEG_MASK_VALUE["f32"] < 0
    assert "tosa.maximum" in causal
    assert causal != plain
    assert len(causal.splitlines()) > len(plain.splitlines())


def test_emitted_kernel_uses_version_portable_tosa_syntax() -> None:
    """The kernel must parse on both IREE 3.12 and 3.13.

    Four TOSA custom assemblies changed between those releases, and each
    spelling is rejected by the other version:

      tosa.const_shape values(...)      vs  {values = ...}
      tosa.transpose perms([...])       vs  {perms = array<i32: ...>}
      tosa.concat ... axis(N)           vs  {axis = N : i32}
      tosa.reduce_* ... axis(N)         vs  {axis = N : i32}

    3.12 is the version coralnpu-compiler pins, so the emitter uses the generic
    (quoted) form for exactly those ops, which both versions accept.  This test
    fails if a custom spelling is reintroduced.
    """
    kernel = emit_kernel(make_shape(seq=4, kv_seq=4), TilePlan(2, 2), causal=True)

    # The version-sensitive ops are present in this kernel, in generic form.
    assert '"tosa.const_shape"()' in kernel
    assert '"tosa.transpose"(' in kernel
    assert '"tosa.concat"(' in kernel
    assert '"tosa.reduce_max"(' in kernel
    assert '"tosa.reduce_sum"(' in kernel

    # None of the version-specific custom spellings appear anywhere.
    assert "values(dense<" not in kernel
    assert "perms([" not in kernel
    for axis in range(4):
        assert f"axis({axis})" not in kernel


def test_emit_kernel_additive_mask() -> None:
    kernel = emit_kernel(
        make_shape(),
        TilePlan(4, 4),
        causal=False,
        mask_shape=(1, 1, 8, 8),
        mask_dtype="f32",
    )
    assert "%mask" in kernel
    assert "tensor<1x1x8x8xf32>" in kernel
    assert "tosa.tile" in kernel  # broadcast one mask head over all heads


def test_emit_kernel_requires_mask_dtype() -> None:
    with pytest.raises(ValueError, match="mask_dtype"):
        emit_kernel(make_shape(), TilePlan(4, 4), mask_shape=(1, 1, 8, 8))


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_emit_kernel_supports_floating_point_dtypes(dtype: str) -> None:
    shape = make_shape(dtype=dtype)
    kernel = emit_kernel(shape, TilePlan(4, 4), causal=True)
    assert f"x{dtype}>" in kernel
    assert NEG_MASK_VALUE[dtype] < 0


def test_causal_rows_offsets() -> None:
    # Decode: one query at absolute position 3 sees the whole 4-token cache.
    rows = _causal_rows(make_shape(seq=1, kv_seq=4), 0, 0, 1, 4)
    assert rows == [[0.0, 0.0, 0.0, 0.0]]

    # Second KV tile of a 2-query/4-key problem at q_offset=2.
    rows = _causal_rows(make_shape(seq=2, kv_seq=4), 0, 2, 2, 2)
    neg = NEG_MASK_VALUE["f32"]
    assert rows == [[0.0, neg], [0.0, 0.0]]


# ---------------------------------------------------------------------------
# Signature analysis
# ---------------------------------------------------------------------------


def test_analyze_signature_packed() -> None:
    q = ArgSpec("%q", "tensor<2x8x4xf32>")
    k = ArgSpec("%k", "tensor<2x8x4xf32>")
    v = ArgSpec("%v", "tensor<2x8x4xf32>")
    spec = analyze_signature([q, k, v], q.type)
    assert spec.layout == "packed"
    assert (spec.shape.batch, spec.shape.heads) == (1, 2)
    assert (spec.shape.seq, spec.shape.kv_seq, spec.shape.head_dim) == (8, 8, 4)
    assert spec.mask_arg is None


def test_analyze_signature_bshd() -> None:
    q = ArgSpec("%q", "tensor<1x8x2x4xf32>")
    k = ArgSpec("%k", "tensor<1x8x2x4xf32>")
    v = ArgSpec("%v", "tensor<1x8x2x4xf32>")
    spec = analyze_signature([q, k, v], q.type)
    assert spec.layout == "bshd"
    assert (spec.shape.batch, spec.shape.seq, spec.shape.heads) == (1, 8, 2)


def test_analyze_signature_bhsd() -> None:
    q = ArgSpec("%q", "tensor<1x2x8x4xf32>")
    k = ArgSpec("%k", "tensor<1x2x8x4xf32>")
    v = ArgSpec("%v", "tensor<1x2x8x4xf32>")
    spec = analyze_signature([q, k, v], q.type, layout="bhsd")
    assert spec.layout == "bhsd"
    assert (spec.shape.batch, spec.shape.heads, spec.shape.seq) == (1, 2, 8)


@pytest.mark.parametrize(
    "k_type, v_type, message",
    [
        ("tensor<1x8x3x4xf32>", "tensor<1x8x2x4xf32>", "layout mismatch"),
        ("tensor<1x8x2x4xf16>", "tensor<1x8x2x4xf16>", "expected dtype"),
        (
            "tensor<1x8x2x4xf32>",
            "tensor<1x8x2x8xf32>",
            "value shape",
        ),
    ],
)
def test_analyze_signature_rejects_mismatches(
    k_type: str, v_type: str, message: str
) -> None:
    q = ArgSpec("%q", "tensor<1x8x2x4xf32>")
    with pytest.raises(ValueError, match=message):
        analyze_signature([q, ArgSpec("%k", k_type), ArgSpec("%v", v_type)], q.type)


def test_analyze_signature_rejects_wrong_arity() -> None:
    q = ArgSpec("%q", "tensor<2x8x4xf32>")
    with pytest.raises(ValueError, match="must take"):
        analyze_signature([q, q], q.type)


def test_analyze_signature_rejects_result_mismatch() -> None:
    q = ArgSpec("%q", "tensor<2x8x4xf32>")
    with pytest.raises(ValueError, match="does not match the query type"):
        analyze_signature([q, q, q], "tensor<2x8x4xf16>")


@pytest.mark.parametrize(
    "mask_type, layout",
    [
        ("tensor<2x8x8xf32>", "packed"),
        ("tensor<1x1x8x8xf32>", "b1sd"),
        ("tensor<1x2x8x8xf32>", "bhsd"),
    ],
)
def test_analyze_signature_mask_layouts(mask_type: str, layout: str) -> None:
    q = ArgSpec("%q", "tensor<2x8x4xf32>")
    spec = analyze_signature([q, q, q, ArgSpec("%mask", mask_type)], q.type)
    assert spec.mask_layout == layout


def test_analyze_signature_rejects_bad_mask_shape() -> None:
    q = ArgSpec("%q", "tensor<2x8x4xf32>")
    with pytest.raises(ValueError, match="does not match the attention problem"):
        analyze_signature([q, q, q, ArgSpec("%mask", "tensor<2x8x7xf32>")], q.type)


# ---------------------------------------------------------------------------
# Wrapper emission
# ---------------------------------------------------------------------------

BSHD_MODULE = """
module {
  func.func @attention(%q: tensor<1x8x2x4xf32>, %k: tensor<1x8x2x4xf32>,
                       %v: tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32> {
    %0 = tensor.empty() : tensor<1x8x2x4xf32>
    return %0 : tensor<1x8x2x4xf32>
  }
}
"""

BHSD_MODULE = """
module {
  func.func @attention(%q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>,
                       %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
    %0 = tensor.empty() : tensor<1x2x8x4xf32>
    return %0 : tensor<1x2x8x4xf32>
  }
}
"""

PACKED_MODULE = """
module {
  func.func @attention(%q: tensor<2x8x4xf32>, %k: tensor<2x8x4xf32>,
                       %v: tensor<2x8x4xf32>) -> tensor<2x8x4xf32> {
    %0 = tensor.empty() : tensor<2x8x4xf32>
    return %0 : tensor<2x8x4xf32>
  }
}
"""


def _wrap(module: str) -> str:
    fn = find_function(module, "attention")
    assert fn is not None
    spec = analyze_signature(fn.args, fn.result_type)
    return emit_wrapper(fn, spec, TilePlan(2, 2), "mhsa_tiled", causal=True)


def test_emit_wrapper_bshd_round_trip() -> None:
    text = _wrap(BSHD_MODULE)
    assert "func.func @mhsa_tiled(" in text
    assert "func.func @attention(" in text
    assert "func.call @mhsa_tiled" in text
    assert "tosa.transpose" in text
    assert "tensor<1x8x2x4xf32>" in text


def test_emit_wrapper_bhsd_round_trip() -> None:
    text = _wrap(BHSD_MODULE)
    assert "func.call @mhsa_tiled" in text
    assert "tosa.reshape" in text
    assert "tensor<1x2x8x4xf32>" in text


def test_emit_wrapper_packed_is_passthrough() -> None:
    text = _wrap(PACKED_MODULE)
    assert "func.call @mhsa_tiled" in text
    # The kernel itself transposes K; the wrapper must not add a layout
    # transpose when the function is already packed.
    wrapper = text.split("func.func @attention(", 1)[1]
    assert "tosa.transpose" not in wrapper


def test_emit_wrapper_preserves_visibility_and_attributes() -> None:
    module = """
module {
  func.func private @attention(%q: tensor<2x8x4xf32>, %k: tensor<2x8x4xf32>,
                               %v: tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
      attributes {foo = "bar"} {
    %0 = tensor.empty() : tensor<2x8x4xf32>
    return %0 : tensor<2x8x4xf32>
  }
}
"""
    fn = find_function(module, "attention")
    assert fn is not None
    spec = analyze_signature(fn.args, fn.result_type)
    text = emit_wrapper(fn, spec, TilePlan(2, 2), "mhsa_tiled")
    assert "func.func private @attention(" in text
    assert "attributes {foo" in text
