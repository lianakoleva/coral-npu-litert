"""CPU-only numerical checks for the attention schedule.

These tests validate the *algorithm* (online softmax over tiles, causal offset,
padding) without requiring IREE.  ``test_integration_iree.py`` runs the emitted
TOSA through IREE and compares against the same reference when the tools are
available.
"""

from __future__ import annotations

import numpy as np
import pytest

from check_attention_numerics import reference_attention


def naive_attention(q, k, v, *, causal: bool, q_offset: int):
    scores = (q / np.sqrt(np.float32(q.shape[-1]))) @ np.swapaxes(k, -1, -2)
    if causal:
        rows = np.arange(q.shape[1])[:, None] + q_offset
        cols = np.arange(k.shape[1])[None, :]
        scores = np.where(cols <= rows, scores, np.float32(-1.0e30))
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v


def streaming_attention(q, k, v, *, q_tile: int, kv_tile: int, causal: bool, q_offset: int):
    """NumPy transcription of the emitted online-softmax schedule."""

    bh, seq, head_dim = q.shape
    kv_seq = k.shape[1]
    scale = np.float32(1.0 / np.sqrt(head_dim))
    out = np.empty_like(q)
    for q_start in range(0, seq, q_tile):
        q_end = min(q_start + q_tile, seq)
        q_block = q[:, q_start:q_end, :] * scale
        acc = None
        row_max = None
        row_sum = None
        for kv_start in range(0, kv_seq, kv_tile):
            kv_end = min(kv_start + kv_tile, kv_seq)
            scores = q_block @ np.swapaxes(k[:, kv_start:kv_end, :], -1, -2)
            if causal:
                rows = np.arange(q_start, q_end)[:, None] + q_offset
                cols = np.arange(kv_start, kv_end)[None, :]
                scores = np.where(cols <= rows, scores, np.float32(-1.0e30))
            tile_max = scores.max(axis=-1, keepdims=True)
            if acc is None:
                row_max = np.maximum(tile_max, np.float32(0.0))
                probs = np.exp(scores - row_max)
                acc = probs @ v[:, kv_start:kv_end, :]
                row_sum = probs.sum(axis=-1, keepdims=True)
            else:
                new_max = np.maximum(row_max, tile_max)
                alpha = np.exp(row_max - new_max)
                probs = np.exp(scores - new_max)
                row_sum = row_sum * alpha + probs.sum(axis=-1, keepdims=True)
                acc = acc * alpha + probs @ v[:, kv_start:kv_end, :]
                row_max = new_max
        out[:, q_start:q_end, :] = acc / np.maximum(row_sum, np.float32(1e-12))
    return out


@pytest.fixture
def tensors():
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((2, 5, 4), dtype=np.float32),
        rng.standard_normal((2, 7, 4), dtype=np.float32),
        rng.standard_normal((2, 7, 4), dtype=np.float32),
    )


@pytest.mark.parametrize(
    "seq, kv_seq, causal, q_offset",
    [
        (5, 5, False, 0),
        (5, 5, True, 0),  # prefill
        (1, 7, True, 6),  # decode
        (3, 7, True, 4),  # partially cached prefill
        (3, 7, False, 0),
    ],
)
def test_reference_matches_naive(seq, kv_seq, causal, q_offset) -> None:
    rng = np.random.default_rng(1)
    q = rng.standard_normal((2, seq, 4), dtype=np.float32)
    k = rng.standard_normal((2, kv_seq, 4), dtype=np.float32)
    v = rng.standard_normal((2, kv_seq, 4), dtype=np.float32)
    actual = reference_attention(q, k, v, causal=causal, q_offset=q_offset)
    expected = naive_attention(q, k, v, causal=causal, q_offset=q_offset)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "seq, kv_seq, q_tile, kv_tile, causal, q_offset",
    [
        (6, 6, 2, 2, False, 0),
        (6, 6, 2, 2, True, 0),
        (5, 7, 2, 4, True, 2),
        (1, 7, 1, 4, True, 6),
        (5, 7, 3, 4, False, 0),
    ],
)
def test_streaming_schedule_matches_reference(
    seq, kv_seq, q_tile, kv_tile, causal, q_offset
) -> None:
    rng = np.random.default_rng(2)
    q = rng.standard_normal((2, seq, 4), dtype=np.float32)
    k = rng.standard_normal((2, kv_seq, 4), dtype=np.float32)
    v = rng.standard_normal((2, kv_seq, 4), dtype=np.float32)
    expected = reference_attention(q, k, v, causal=causal, q_offset=q_offset)
    actual = streaming_attention(
        q,
        k,
        v,
        q_tile=q_tile,
        kv_tile=kv_tile,
        causal=causal,
        q_offset=q_offset,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_causal_mask_is_position_based_not_tile_based() -> None:
    q = np.zeros((1, 3, 1), dtype=np.float32)
    k = np.zeros((1, 7, 1), dtype=np.float32)
    v = np.arange(7, dtype=np.float32).reshape(1, 7, 1)
    # Query rows are absolute positions 4, 5, 6; row i sees keys 0..4+i.
    out = reference_attention(q, k, v, causal=True, q_offset=4)
    expected = np.array([[[2.0], [2.5], [3.0]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)
