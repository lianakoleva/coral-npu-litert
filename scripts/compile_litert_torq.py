#!/usr/bin/env python3
"""Turn LiteRT's serialized NPU partitions into a Torq VMFB.

This is the AOT half of the Coral NPU compiler plugin (see
``litert/vendors/coralnpu/compiler_plugin``).  The plugin writes the ops it
selected into a single TFLite flatbuffer; this driver lowers that flatbuffer to
a module the Torq runtime can execute, and the plugin's dispatch library then
runs it on the NPU.

Pipeline
--------
1. ``tosa-converter-for-tflite`` legalizes the TFLite partitions to TOSA MLIR.
   One subgraph becomes one function, named ``fn_<subgraph index>``.
2. ``scripts/tosa_patch.py`` rewrites the handful of constructs the Torq
   backend cannot take as-is (i1 tensors, integer ``FLOOR_MOD``, dynamic slice
   updates, and 1x1 convolutions whose consumer wants a transposed result), and
   renames ``fn_<i>`` to ``partition_<i>`` so the names match the plugin's call
   infos.
3. ``torq-compile`` compiles the module for the Coral NPU (``--torq-hw``).

Why AOT instead of on-device
----------------------------
The Torq toolchain is an x86-64 host tool, and a 270M-parameter transformer
takes minutes and gigabytes of scratch space to lower.  So the VMFB is built on
a build host and cached; the plugin looks it up by content hash
(``CORALNPU_AOT_CACHE``) and the device only reads the artifact back.

The two stages have different dependencies (the TOSA converter is an Arm wheel,
``torq-compile`` ships with the Torq compiler), so they can be run separately
with ``--stage convert`` / ``--stage compile`` on machines that have only one of
them.  ``--stage all`` (the default) needs both.

Usage
-----
    compile_litert_torq.py partitions.tflite --out-dir build/npu
    compile_litert_torq.py partitions.tflite --out-dir build/npu --stage convert
    compile_litert_torq.py build/npu/02_partitions.mlir --out-dir build/npu \\
        --stage compile
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHER = HERE / "tosa_patch.py"

CONVERTED = "01_tosa.mlir"
PATCHED = "02_partitions.mlir"
VMFB = "partitions.vmfb"


def Run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def Convert(args: argparse.Namespace, src: Path) -> Path:
    """TFLite partitions -> patched TOSA MLIR."""
    converted = args.out_dir / CONVERTED
    patched = args.out_dir / PATCHED
    Run([args.tosa_converter, str(src), "--text", "-o", str(converted)])

    argv = [sys.executable, str(PATCHER), str(converted), str(patched),
            "--rename", "fn_:partition_"]
    if args.embedding_arg:
        # Bring-up only: keep the embedding table out of the module by taking it
        # as an extra function argument.  The production path excludes
        # EMBEDDING_LOOKUP from the partition in the plugin instead.
        argv += ["--embedding-arg", args.embedding_arg]
    Run(argv)
    if not args.keep_intermediate:
        converted.unlink(missing_ok=True)
    return patched


def Compile(args: argparse.Namespace, src: Path) -> Path:
    """Patched TOSA MLIR -> Torq VMFB."""
    vmfb = args.out_dir / VMFB
    argv = [
        args.torq_compile,
        str(src),
        "--iree-input-type=tosa",
        "--iree-hal-target-backends=torq",
        f"--torq-hw={args.torq_hw}",
        # The CSS path rejects modules whose per-dispatch stack request does not
        # fit the default 1 KiB window; --torq-disable-css keeps the lowering on
        # the path the Coral NPU kernels are built for.
        "--torq-disable-css",
        "-o", str(vmfb),
    ]
    for extra in args.torq_flag:
        argv.append(extra)
    Run(argv)
    return vmfb


def Find(tool: str, env: str, override: str | None) -> str:
    if override:
        return override
    from_env = os.environ.get(env)
    if from_env:
        return from_env
    found = shutil.which(tool)
    if found:
        return found
    raise SystemExit(
        f"cannot find {tool}: set --{tool.replace('_', '-')} or {env}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="partitions.tflite (convert stage) or the patched "
                         "MLIR (compile stage)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=("all", "convert", "compile"), default="all")
    ap.add_argument("--tosa-converter", default=None,
                    help="default: $CORALNPU_TOSA_CONVERTER or "
                         "tosa-converter-for-tflite on $PATH")
    ap.add_argument("--torq-compile", default=None,
                    help="default: $CORALNPU_TORQ_COMPILE or torq-compile on $PATH")
    ap.add_argument("--torq-hw", default=os.environ.get("CORALNPU_TORQ_HW", "SL2610"))
    ap.add_argument("--torq-flag", action="append", default=[],
                    help="extra torq-compile flag (repeatable)")
    ap.add_argument("--embedding-arg", default=None,
                    help="bring-up only; see scripts/tosa_patch.py")
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"no such input: {input_path}")

    src = input_path
    if args.stage in ("all", "convert"):
        args.tosa_converter = Find("tosa-converter-for-tflite",
                                   "CORALNPU_TOSA_CONVERTER",
                                   args.tosa_converter)
        src = Convert(args, input_path)
    if args.stage in ("all", "compile"):
        if args.stage == "compile" and args.input.suffix == ".tflite":
            src = args.out_dir / PATCHED
        args.torq_compile = Find("torq-compile", "CORALNPU_TORQ_COMPILE",
                                 args.torq_compile)
        Compile(args, src)


if __name__ == "__main__":
    main()
