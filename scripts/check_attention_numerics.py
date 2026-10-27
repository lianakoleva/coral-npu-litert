#!/usr/bin/env python3
"""Numerically validate the emitted tiled TOSA attention against NumPy.

The standalone kernel is compiled for the host CPU with IREE, executed with
``iree-run-module``, and compared with a reference scaled-dot-product attention
using the same causal mask.  This checks the pieces that are easy to get wrong
in generated IR: tile offsets, padding, the online-softmax recurrence, and the
causal mask alignment.

Requirements: ``iree-base-compiler``, ``iree-base-runtime``, ``numpy``.

    python scripts/npu/check_attention_numerics.py --seq 5 --kv-seq 7 \\
        --q-tile 4 --kv-tile 4 --causal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tosa_mhsa import (
    AttentionShape,
    TilePlan,
    emit_kernel,
    select_plan,
)


def reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
    q_offset: int,
) -> np.ndarray:
    """NumPy MHSA with the same packed layout and mask convention."""

    head_dim = q.shape[-1]
    scores = (q / np.sqrt(np.float32(head_dim))) @ np.swapaxes(k, -1, -2)
    if causal:
        rows = np.arange(q.shape[1])[:, None] + q_offset
        cols = np.arange(k.shape[1])[None, :]
        scores = np.where(cols <= rows, scores, np.float32(-1.0e30))
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the generated tiled TOSA attention numerically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=5)
    parser.add_argument("--kv-seq", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=4)
    parser.add_argument("--q-tile", type=int, default=4)
    parser.add_argument("--kv-tile", type=int, default=4)
    parser.add_argument("--q-offset", type=int)
    parser.add_argument("--causal", dest="causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-run-module", default="iree-run-module")
    parser.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shape = AttentionShape(
        batch=args.batch,
        heads=args.heads,
        seq=args.seq,
        kv_seq=args.kv_seq,
        head_dim=args.head_dim,
        q_offset=args.q_offset,
    )
    plan = TilePlan(args.q_tile, args.kv_tile)
    try:
        plan, _ = select_plan(
            shape,
            budget_bytes=256 * 1024,
            causal=args.causal,
            q_tile=args.q_tile,
            kv_tile=args.kv_tile,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    q = rng.standard_normal((shape.bh, shape.seq, shape.head_dim), dtype=np.float32)
    k = rng.standard_normal((shape.bh, shape.kv_seq, shape.head_dim), dtype=np.float32)
    v = rng.standard_normal((shape.bh, shape.kv_seq, shape.head_dim), dtype=np.float32)
    expected = reference_attention(
        q, k, v, causal=args.causal, q_offset=shape.query_offset
    )

    workdir = Path(tempfile.mkdtemp(prefix="npu-attn-"))
    kernel = workdir / "kernel.mlir"
    vmfb = workdir / "kernel.vmfb"
    kernel.write_text(
        emit_kernel(
            shape,
            plan,
            causal=args.causal,
            func_name="mhsa_tiled",
            budget_bytes=256 * 1024,
        )
    )

    compile_cmd = [
        args.iree_compile,
        str(kernel),
        "--iree-input-type=tosa",
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "-o",
        str(vmfb),
    ]
    print("+ " + " ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)

    q_path, k_path, v_path, out_path = (
        workdir / "q.npy",
        workdir / "k.npy",
        workdir / "v.npy",
        workdir / "out.npy",
    )
    np.save(q_path, q)
    np.save(k_path, k)
    np.save(v_path, v)

    run_cmd = [
        args.iree_run_module,
        f"--module={vmfb}",
        "--function=mhsa_tiled",
        f"--input=@{q_path}",
        f"--input=@{k_path}",
        f"--input=@{v_path}",
        f"--output=@{out_path}",
    ]
    print("+ " + " ".join(run_cmd))
    subprocess.run(run_cmd, check=True)

    actual = np.load(out_path)
    if actual.shape != expected.shape:
        print(f"FAIL: shape {actual.shape}, expected {expected.shape}", file=sys.stderr)
        return 1

    error = np.abs(actual - expected)
    ok = np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    print(
        f"{'PASS' if ok else 'FAIL'}: max|error|={error.max():.3e} "
        f"mean|error|={error.mean():.3e} "
        f"(atol={args.atol}, rtol={args.rtol})"
    )
    print(
        f"  tiles q={plan.q_tile} kv={plan.kv_tile} "
        f"shape {shape.batch}x{shape.heads}x{shape.seq}x{shape.kv_seq}x{shape.head_dim} "
        f"q_offset={shape.query_offset} causal={args.causal}"
    )
    if args.keep:
        print(f"  artifacts in {workdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
