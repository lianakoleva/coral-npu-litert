"""Integration tests that require external IREE (and optionally Coral NPU) tools.

These are skipped automatically when the tools are not on PATH.  They are the
tests that catch TOSA dialect drift and prove the emitted kernel actually
executes, which unit tests cannot do.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import compile_litert
from tosa_mhsa import AttentionShape, TilePlan, emit_kernel

pytestmark = pytest.mark.integration

IREEC = shutil.which("iree-compile")
IREEOPT = shutil.which("iree-opt")
RUNMOD = shutil.which("iree-run-module")


def _has_coralnpu_device(iree_compile: str | None) -> bool:
    if not iree_compile:
        return False
    result = subprocess.run(
        [iree_compile, "--help"], capture_output=True, text=True, check=False
    )
    return "coralnpu" in (result.stdout + result.stderr)


HAS_CORALNPU = _has_coralnpu_device(IREEC)

ATTENTION_MODULE = """
module {
  func.func @attention(%q: tensor<1x8x2x4xf32>, %k: tensor<1x8x2x4xf32>,
                       %v: tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32> {
    %0 = tensor.empty() : tensor<1x8x2x4xf32>
    return %0 : tensor<1x8x2x4xf32>
  }
}
"""


@pytest.mark.skipif(not IREEOPT, reason="iree-opt is not installed")
def test_emitted_kernel_verifies(tmp_path) -> None:
    shape = AttentionShape(batch=1, heads=2, seq=8, kv_seq=8, head_dim=4)
    kernel = emit_kernel(
        shape,
        TilePlan(2, 2),
        causal=True,
        func_name="mhsa_tiled",
        budget_bytes=64 * 1024,
    )
    path = tmp_path / "kernel.mlir"
    path.write_text(kernel)
    subprocess.run(
        [IREEOPT, "--verify-each", str(path), "-o", os.devnull], check=True
    )


@pytest.mark.skipif(not IREEOPT, reason="iree-opt is not installed")
def test_driver_verify_stage_accepts_wrapper(tmp_path) -> None:
    model = tmp_path / "model.mlir"
    model.write_text(ATTENTION_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(out),
                "--stop-after",
                "verify",
                "--attention-func",
                "attention",
                "--q-tile",
                "2",
                "--kv-tile",
                "2",
                "--budget-bytes",
                str(64 * 1024),
                "--iree-opt",
                IREEOPT,
            ]
        )
        == 0
    )
    assert (out / "03_tiled.mlir").exists()


@pytest.mark.skipif(
    not (IREEC and RUNMOD), reason="iree-compile/iree-run-module are not installed"
)
def test_tiled_attention_runs_and_matches_numpy(tmp_path) -> None:
    from check_attention_numerics import main as check_main

    assert (
        check_main(
            [
                "--seq",
                "5",
                "--kv-seq",
                "7",
                "--q-tile",
                "4",
                "--kv-tile",
                "4",
                "--causal",
                "--iree-compile",
                IREEC,
                "--iree-run-module",
                RUNMOD,
            ]
        )
        == 0
    )


def _compile_coralnpu(tmp_path):
    """Compile the tiled kernel for the Coral NPU; return the VMFB path."""
    shape = AttentionShape(batch=1, heads=2, seq=4, kv_seq=4, head_dim=4)
    kernel = emit_kernel(
        shape,
        TilePlan(2, 2),
        causal=True,
        func_name="mhsa_tiled",
        budget_bytes=64 * 1024,
    )
    source = tmp_path / "kernel.mlir"
    vmfb = tmp_path / "kernel.vmfb"
    source.write_text(kernel)

    subprocess.run(
        [
            IREEC,
            str(source),
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            "--iree-hal-target-device=coralnpu",
            "-o",
            str(vmfb),
        ],
        check=True,
    )
    return vmfb


@pytest.mark.skipif(
    not HAS_CORALNPU,
    reason="the installed IREE has no Coral NPU compiler target (use coralnpu-compiler)",
)
def test_coralnpu_compiles_tiled_kernel(tmp_path) -> None:
    """The Coral NPU target lowers the emitted kernel to a non-empty VMFB.

    This half of the pipeline works on any host, including arm64, where the
    simulator runtime itself is x86_64-only.
    """
    vmfb = _compile_coralnpu(tmp_path)
    assert vmfb.exists() and vmfb.stat().st_size > 0


@pytest.mark.skipif(
    not HAS_CORALNPU or not RUNMOD,
    reason="needs iree-compile with the Coral NPU target and iree-run-module "
    "(build both with coralnpu-compiler)",
)
def test_coralnpu_compile_and_simulated_execution(tmp_path) -> None:
    vmfb = _compile_coralnpu(tmp_path)

    # The module carries two HAL devices (host llvm-cpu fallback + coralnpu,
    # matching scripts/coralnpu.json), so both drivers must be provided, in
    # module order.
    result = subprocess.run(
        [
            RUNMOD,
            f"--module={vmfb}",
            "--device=local-task",
            "--device=coralnpu",
            "--function=mhsa_tiled",
            "--input=2x4x4xf32=0",
            "--input=2x4x4xf32=0",
            "--input=2x4x4xf32=0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "EXEC" in result.stdout or "mhsa_tiled" in result.stdout
