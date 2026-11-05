"""Unit tests for the dependency-free MLIR text helpers."""

from __future__ import annotations

import pytest

from mlir_text import (
    MlirParseError,
    attention_candidates,
    bind_dynamic_dims,
    dynamic_extent_count,
    find_function,
    find_functions,
    match_delimiter,
    replace_function,
    split_top_level,
    tensor_stats,
)

MODULE = """
// a comment with a brace } and a quote "
module {
  func.func @main(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    // nested braces in a string must not confuse the parser
    %0 = "test.op"(%x) {attr = "{not a body}", other = ")"} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }

  func.func private @attention_0(%q: tensor<2x8x4xf32>, %k: tensor<2x8x4xf32>,
                                %v: tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
      attributes {tile_q = 4 : i64} {
    %0 = tensor.empty() : tensor<2x8x4xf32>
    return %0 : tensor<2x8x4xf32>
  }

  func.func @helper(%a: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    return %a : tensor<1x2x3xf32>
  }
}
"""


def test_find_functions() -> None:
    functions = find_functions(MODULE)
    assert [f.name for f in functions] == ["main", "attention_0", "helper"]
    attention = functions[1]
    assert attention.visibility == "private"
    assert len(attention.args) == 3
    assert attention.result_type == "tensor<2x8x4xf32>"
    assert "tile_q" in attention.header_attrs
    assert attention.looks_like_attention()
    assert not functions[0].looks_like_attention()


def test_find_function_by_name() -> None:
    assert find_function(MODULE, "helper") is not None
    assert find_function(MODULE, "missing") is None


def test_function_text_round_trip() -> None:
    function = find_function(MODULE, "helper")
    assert function is not None
    assert function.text.startswith("func.func @helper")
    assert function.text.rstrip().endswith("}")


def test_replace_function_preserves_surroundings() -> None:
    function = find_function(MODULE, "helper")
    assert function is not None
    replacement = "func.func @helper(%a: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {\n  return %a : tensor<1x2x3xf32>\n}\n"
    updated = replace_function(MODULE, function, replacement)
    assert updated.count("func.func @helper") == 1
    assert "return %a : tensor<1x2x3xf32>" in updated
    assert "func.func private @attention_0" in updated


def test_bind_dynamic_dims_only_touches_tensor_types() -> None:
    # The count is the number of dynamic extents in the text, not the number
    # of distinct tensor types: main has an argument, a result, an op operand,
    # an op result, and a return type.
    bound, count = bind_dynamic_dims(MODULE, 16)
    assert count == 5
    assert "tensor<?x4xf32>" not in bound
    assert "tensor<16x4xf32>" in bound
    # The string attribute and comments are untouched.
    assert 'attr = "{not a body}"' in bound


def test_bind_dynamic_dims_rejects_zero() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        bind_dynamic_dims("tensor<?xf32>", 0)


def test_dynamic_extent_count() -> None:
    assert dynamic_extent_count(MODULE) == 5
    assert dynamic_extent_count("tensor<1x2xf32>") == 0


def test_tensor_stats_sorted_by_size() -> None:
    text = "tensor<2x8x4xf32> tensor<2x8x4xf32> tensor<1x2xf32>"
    stats = tensor_stats(text)
    assert stats[0].type == "tensor<2x8x4xf32>"
    assert stats[0].count == 2
    assert stats[0].bytes_per_instance == 2 * 8 * 4 * 4
    assert stats[0].total_bytes == 2 * stats[0].bytes_per_instance


def test_attention_candidates_requires_name_and_rank() -> None:
    functions = find_functions(MODULE)
    candidates = attention_candidates(functions)
    assert [f.name for f in candidates] == ["attention_0"]


def test_attention_candidates_accepts_common_names() -> None:
    module = """
module {
  func.func @sdpa(%q: tensor<2x8x4xf32>, %k: tensor<2x8x4xf32>,
                  %v: tensor<2x8x4xf32>) -> tensor<2x8x4xf32> {
    return %q : tensor<2x8x4xf32>
  }
}
"""
    assert [f.name for f in attention_candidates(find_functions(module))] == ["sdpa"]


def test_match_delimiter_skips_strings_and_comments() -> None:
    text = '{a = "}", b = 1} tail'
    end = match_delimiter(text, 0, "{", "}")
    assert text[: end + 1] == '{a = "}", b = 1}'


def test_match_delimiter_unbalanced() -> None:
    with pytest.raises(MlirParseError):
        match_delimiter("{", 0, "{", "}")


def test_split_top_level() -> None:
    assert split_top_level("a, b<c, d>, e") == ["a", "b<c, d>", "e"]
    assert split_top_level("tensor<1x2xf32>, tensor<3x4xf32>") == [
        "tensor<1x2xf32>",
        "tensor<3x4xf32>",
    ]


def test_find_functions_ignores_malformed_entries() -> None:
    text = "func.func @broken(\nmodule { func.func @ok() { return } }"
    # The malformed one is skipped; the parser still finds the valid one.
    assert [f.name for f in find_functions(text)] == ["ok"]
