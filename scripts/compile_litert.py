#!/usr/bin/env python3
"""Compile a LiteRT LLM graph for the RISC-V Coral NPU with IREE/MLIR.

Pipeline
--------
1. **Ingest** a LiteRT flatbuffer (``.tflite``, or a ``.litertlm`` bundle that
   contains one) with ``iree-import-tflite``, producing TOSA dialect MLIR.
2. **Bind** the single dynamic extent (LLM sequence length) to a static value,
   because TOSA and the Coral NPU codegen want static shapes.
3. **Lower + partition** the multi-head attention function: the imported
   attention body is replaced with a statically tiled TOSA kernel whose query
   and key/value tiles are chosen to fit a fixed per-tile byte budget.
4. **Verify** the result with ``iree-opt --verify-each``.
5. **Compile** with ``iree-compile`` for the Coral NPU RV32 + vector target.

The generated kernel has no dynamic shapes and no control flow: tiling is
unrolled at compile time, so the memory footprint is a property of the IR.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlir_text import (
    FunctionDef,
    attention_candidates,
    bind_dynamic_dims,
    dynamic_extent_count,
    find_function,
    find_functions,
    replace_function,
    tensor_stats,
)
from tosa_mhsa import (
    AttentionShape,
    TilePlan,
    TilePlanError,
    analyze_signature,
    emit_kernel,
    emit_wrapper,
    select_plan,
)

PROGRAM = "compile_litert_npu"

DEFAULTS: dict[str, Any] = {
    "memory": {
        # Fixed scratchpad tile available to a kernel on the NPU.  Set this to
        # the SRAM/TCM window your Coral NPU build exposes to IREE.
        "tile_bytes": 256 * 1024,
        "stack_bytes": 64 * 1024,
        "safety_factor": 1.25,
        "reserved_bytes": 4 * 1024,
        "q_tile_candidates": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "kv_tile_candidates": [8, 16, 32, 64, 128, 256, 512, 1024],
        "max_unrolled_tiles": 8192,
    },
    "attention": {
        "causal": True,
        "layout": "auto",
        "q_tile": None,
        "kv_tile": None,
    },
    "target": {
        "name": "coral-npu",
        "iree_compile": "iree-compile",
        "iree_opt": "iree-opt",
        "iree_import_tflite": "iree-import-tflite",
        # Conservative RV32 + vector baseline.  Replace these with the exact
        # flags your Coral NPU SDK's iree-compile wrapper uses.
        "compile_flags": [
            "--iree-input-type=tosa",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-triple=riscv32-unknown-elf",
            "--iree-llvmcpu-target-cpu=generic-rv32",
            "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl128b,+zve32f",
            "--iree-llvmcpu-target-abi=ilp32d",
            "--iree-llvmcpu-link-embedded=true",
            "--iree-llvmcpu-debug-symbols=false",
            "--iree-llvmcpu-stack-allocation-limit={stack_bytes}",
            "--iree-opt-level=O2",
        ],
    },
}

TOOL_HINT = (
    "Install the IREE tools first:\n"
    "    pip install iree-base-compiler iree-tools-tflite\n"
    "or point the script at an existing build with --iree-compile/--iree-opt/"
    "--iree-import-tflite."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def die(message: str) -> SystemExit:
    raise SystemExit(f"{PROGRAM}: error: {message}")


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def human_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


class _Substitutions(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(
            f"unknown placeholder {{{key}}} in compile_flags; available: "
            + ", ".join(sorted(self))
        )


def run(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex_quote(part) for part in command)
    print(f"    $ {printable}")
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        die(f"command not found: {command[0]}\n{TOOL_HINT}")
    except subprocess.CalledProcessError as exc:
        die(f"command failed with exit code {exc.returncode}: {printable}")


def shlex_quote(part: str) -> str:
    if not part or re.search(r"[^\w@%+=:,./-]", part):
        return "'" + part.replace("'", "'\\''") + "'"
    return part


def resolve_tool(tool: str) -> str:
    candidate = shutil.which(tool)
    if candidate:
        return candidate
    path = Path(tool).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    die(f"required tool not found: {tool}\n{TOOL_HINT}")
    return tool  # unreachable


# ---------------------------------------------------------------------------
# Stage 1: LiteRT -> TOSA
# ---------------------------------------------------------------------------


def extract_tflite(bundle: Path, out_dir: Path) -> Path:
    """Pull the TFLite flatbuffer out of a ``.litertlm`` container."""

    if zipfile.is_zipfile(bundle):
        with zipfile.ZipFile(bundle) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".tflite")
            ]
            if not members:
                die(f"{bundle}: zip bundle contains no .tflite member")
            if len(members) > 1:
                print(
                    f"    note: {len(members)} .tflite members found, using "
                    f"{members[0]!r}"
                )
            target = out_dir / "extracted_model.tflite"
            target.write_bytes(archive.read(members[0]))
            return target

    payload = bundle.read_bytes()
    offsets = [match.start() for match in re.finditer(b"TFL3", payload)]
    if not offsets:
        die(
            f"{bundle}: no TFLite flatbuffer found. Pass the .tflite file "
            "directly with --tflite, or extract it with the LiteRT-LM tooling."
        )
    # The TFLite file identifier sits at bytes 4..8 of the flatbuffer.
    start = offsets[0] - 4
    if start < 0:
        die(f"{bundle}: malformed TFLite payload")
    if len(offsets) > 1:
        print(
            f"    note: found {len(offsets)} 'TFL3' markers in {bundle.name}; "
            "using the first one"
        )
    target = out_dir / "extracted_model.tflite"
    target.write_bytes(payload[start:])
    return target


def ingest(model: Path, raw_path: Path, config: dict, dry_run: bool) -> str:
    if not model.exists():
        die(f"input model not found: {model}")

    suffix = model.suffix.lower()
    if suffix == ".mlir":
        text = model.read_text()
        raw_path.write_text(text)
        print(f"    using MLIR input {model} ({len(text.splitlines())} lines)")
        return text

    tflite = model
    if suffix in {".litertlm", ".litert", ".bin"}:
        tflite = extract_tflite(model, raw_path.parent)
        print(f"    extracted TFLite model -> {tflite}")
    elif suffix != ".tflite":
        print(f"    warning: unexpected extension {suffix!r}, trying it as TFLite")

    importer = (
        config["target"]["iree_import_tflite"]
        if dry_run
        else resolve_tool(config["target"]["iree_import_tflite"])
    )
    run([importer, str(tflite), "-o", str(raw_path)], dry_run=dry_run)
    if dry_run and not raw_path.exists():
        return ""
    return raw_path.read_text()


# ---------------------------------------------------------------------------
# Stage 3: attention selection + replacement
# ---------------------------------------------------------------------------


def _unique_kernel_name(target: FunctionDef, existing: set[str]) -> str:
    base = "mhsa_tiled_" + re.sub(r"[^A-Za-z0-9_]", "_", target.name)
    name = base
    index = 1
    while name in existing:
        name = f"{base}_{index}"
        index += 1
    return name


def lower_attention(
    text: str,
    *,
    attention_func: str | None,
    config: dict,
    allow_untiled: bool,
    force_layout: str | None,
    q_offset: int | None,
) -> tuple[str, dict | None]:
    functions = find_functions(text)
    if not functions:
        die("no func.func definitions found in the imported module")

    target: FunctionDef | None = None
    if attention_func:
        target = find_function(text, attention_func.lstrip("@"))
        if target is None:
            names = ", ".join(sorted(f.name for f in functions))
            die(f"attention function @{attention_func} not found; module has: {names}")
    else:
        candidates = attention_candidates(functions)
        if len(candidates) == 1:
            target = candidates[0]
        elif len(candidates) > 1:
            listing = "\n".join(
                f"  @{f.name}{f.signature[len('func.func '):]}" for f in candidates
            )
            die(
                "several attention candidates found; choose one with "
                "--attention-func:\n" + listing
            )

    if target is None:
        if allow_untiled:
            print(
                "    warning: no attention function found; leaving the imported "
                "attention graph untiled (--allow-untiled-attention)"
            )
            return text, None
        die(
            "no multi-head attention function found. The importer keeps fused "
            "LiteRT attention in the caller, so outline it as a function taking "
            "(query, key, value[, mask]) and returning the attention output, "
            "then pass --attention-func. Use --list-functions to inspect the "
            "module, or --allow-untiled-attention to compile without tiling."
        )

    try:
        spec = analyze_signature(
            target.args,
            target.result_type,
            layout=force_layout or config["attention"]["layout"],
            q_offset=q_offset,
        )
    except ValueError as exc:
        die(f"cannot lower @{target.name}: {exc}")

    causal = bool(config["attention"]["causal"])
    memory = config["memory"]
    try:
        plan, budget = select_plan(
            spec.shape,
            budget_bytes=int(memory["tile_bytes"]),
            causal=causal,
            has_mask=spec.mask_arg is not None,
            q_tile=config["attention"]["q_tile"],
            kv_tile=config["attention"]["kv_tile"],
            q_candidates=[int(v) for v in memory["q_tile_candidates"]],
            kv_candidates=[int(v) for v in memory["kv_tile_candidates"]],
            safety_factor=float(memory["safety_factor"]),
            reserved_bytes=int(memory["reserved_bytes"]),
            max_tiles=int(memory["max_unrolled_tiles"]),
        )
    except TilePlanError as exc:
        die(f"cannot fit @{target.name} into the tile budget: {exc}")

    existing = {function.name for function in functions}
    kernel_name = _unique_kernel_name(target, existing)
    replacement = emit_wrapper(
        target,
        spec,
        plan,
        kernel_name,
        causal=causal,
        budget_bytes=int(memory["tile_bytes"]),
        safety_factor=float(memory["safety_factor"]),
        reserved_bytes=int(memory["reserved_bytes"]),
    )
    lowered = replace_function(text, target, replacement)

    q_tiles, kv_tiles = plan.tile_counts(spec.shape)
    print(f"    function   @{target.name} ({spec.layout})")
    print(
        f"    shape      batch={spec.shape.batch} heads={spec.shape.heads} "
        f"seq={spec.shape.seq} kv_seq={spec.shape.kv_seq} "
        f"head_dim={spec.shape.head_dim} dtype={spec.shape.dtype}"
    )
    print(
        f"    tiles      q={plan.q_tile} kv={plan.kv_tile} "
        f"(padded {plan.padded_seq(spec.shape)}x{plan.padded_kv(spec.shape)}, "
        f"{q_tiles}x{kv_tiles} = {q_tiles * kv_tiles} unrolled steps)"
    )
    print(
        f"    memory     peak {human_bytes(budget.peak_bytes)} x "
        f"{memory['safety_factor']} + {human_bytes(int(memory['reserved_bytes']))} "
        f"= {human_bytes(budget.required_bytes)} "
        f"<= {human_bytes(int(memory['tile_bytes']))}"
    )
    if causal:
        print("    mask       causal (generated per tile)")
    if spec.mask_arg is not None:
        print(
            f"    mask       additive argument {spec.mask_arg.name} ({spec.mask_layout})"
        )

    plan_info = {
        "function": target.name,
        "kernel": kernel_name,
        "layout": spec.layout,
        "shape": {
            "batch": spec.shape.batch,
            "heads": spec.shape.heads,
            "seq": spec.shape.seq,
            "kv_seq": spec.shape.kv_seq,
            "head_dim": spec.shape.head_dim,
            "dtype": spec.shape.dtype,
        },
        "tiles": {
            "q_tile": plan.q_tile,
            "kv_tile": plan.kv_tile,
            "padded_seq": plan.padded_seq(spec.shape),
            "padded_kv": plan.padded_kv(spec.shape),
            "unrolled_steps": q_tiles * kv_tiles,
        },
        "budget": budget.to_dict(),
    }
    if spec.mask_arg is not None:
        plan_info["mask_shape"] = list(spec.mask_arg.shape)
        plan_info["mask_dtype"] = spec.mask_arg.dtype
    return lowered, plan_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Compile a LiteRT LLM graph to Coral NPU via TOSA, with the "
            "multi-head attention sequence axis explicitly partitioned into "
            "fixed-size memory tiles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s model.litertlm --seq-len 128 --budget-bytes 65536\n"
            "  %(prog)s model.tflite --seq-len 512 --attention-func attn_0 \\\n"
            "      --q-tile 16 --kv-tile 64 --stop-after attention\n"
        ),
    )
    parser.add_argument(
        "model", nargs="?", type=Path, help=".tflite, .litertlm, or .mlir input"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("build/npu"), help="artifact directory"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="concrete value for every dynamic tensor extent (LLM sequence length)",
    )
    parser.add_argument(
        "--attention-func",
        help="name of the func.func implementing (q, k, v[, mask]) -> attention output",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "bshd", "bhsd", "packed"],
        help="tensor layout of the attention function (default: config, usually auto)",
    )
    parser.add_argument(
        "--causal",
        dest="causal",
        action="store_true",
        default=None,
        help="generate a causal mask per tile",
    )
    parser.add_argument(
        "--no-causal",
        dest="causal",
        action="store_false",
        help="disable the causal mask",
    )
    parser.add_argument(
        "--q-offset",
        type=int,
        help="absolute KV position of query row 0 (default: kv_seq - seq)",
    )
    parser.add_argument("--q-tile", type=int, help="override the query tile size")
    parser.add_argument("--kv-tile", type=int, help="override the key/value tile size")
    parser.add_argument("--budget-bytes", type=int, help="per-tile memory budget")
    parser.add_argument(
        "--safety-factor", type=float, help="multiplier on the peak estimate"
    )
    parser.add_argument(
        "--stop-after",
        choices=["ingest", "bind", "attention", "verify", "compile"],
        default="compile",
        help="stop after a pipeline stage (default: compile)",
    )
    parser.add_argument(
        "--skip-attention", action="store_true", help="do not tile attention"
    )
    parser.add_argument(
        "--allow-untiled-attention",
        action="store_true",
        help="do not fail when no attention function is found",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="skip iree-opt verification"
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="print discovered functions and exit",
    )
    parser.add_argument(
        "--report-tensors",
        action="store_true",
        help="print the largest tensor types after shape binding",
    )
    parser.add_argument(
        "--print-plan", action="store_true", help="print the tile plan and exit"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="print the merged config and exit"
    )
    parser.add_argument("--config", type=Path, help="JSON file overriding the defaults")
    parser.add_argument("--iree-compile", help="path to iree-compile")
    parser.add_argument("--iree-opt", help="path to iree-opt")
    parser.add_argument("--iree-import-tflite", help="path to iree-import-tflite")
    parser.add_argument(
        "--extra-compile-flag",
        action="append",
        default=[],
        help="append a flag to the final iree-compile invocation (repeatable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="do not execute external tools"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def load_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if args.config:
        if not args.config.exists():
            die(f"config not found: {args.config}")
        deep_merge(config, json.loads(args.config.read_text()))
    if args.budget_bytes is not None:
        config["memory"]["tile_bytes"] = args.budget_bytes
    if args.safety_factor is not None:
        config["memory"]["safety_factor"] = args.safety_factor
    if args.q_tile is not None:
        config["attention"]["q_tile"] = args.q_tile
    if args.kv_tile is not None:
        config["attention"]["kv_tile"] = args.kv_tile
    if args.layout is not None:
        config["attention"]["layout"] = args.layout
    if args.causal is not None:
        config["attention"]["causal"] = args.causal
    for key in ("iree_compile", "iree_opt", "iree_import_tflite"):
        override = getattr(args, key)
        if override:
            config["target"][key] = override
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args)

    if args.print_config:
        print(json.dumps(config, indent=2))
        return 0
    if args.model is None:
        parser.error("a model path is required")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "01_tosa.mlir"
    static_path = out_dir / "02_static.mlir"
    tiled_path = out_dir / "03_tiled.mlir"
    kernel_path = out_dir / "mhsa_tiled.mlir"
    plan_path = out_dir / "tile_plan.json"

    print(f"[1/5] ingest     {args.model} -> {raw_path}")
    text = ingest(args.model, raw_path, config, args.dry_run)
    if args.stop_after == "ingest":
        return 0
    if not text and not raw_path.exists():
        die("ingest produced no MLIR (dry run?)")
    if not text:
        text = raw_path.read_text()

    print("[2/5] bind       static shapes")
    dynamic = dynamic_extent_count(text)
    if dynamic:
        if args.seq_len is None:
            die(
                f"the graph has {dynamic} dynamic tensor extents; pass --seq-len "
                "with the concrete sequence length (TOSA and the NPU need static "
                "shapes)"
            )
        text, replaced = bind_dynamic_dims(text, args.seq_len)
        print(f"    bound {replaced} dynamic extent(s) to {args.seq_len}")
    elif args.seq_len is not None:
        print(f"    note: graph is already static, --seq-len {args.seq_len} unused")
    static_path.write_text(text)

    if args.report_tensors:
        print("    largest tensor types (weights live in read-only memory):")
        for stat in tensor_stats(text)[:6]:
            print(
                f"      {human_bytes(stat.bytes_per_instance):>9}  x{stat.count:<4} "
                f"{stat.type}"
            )

    if args.list_functions:
        print("    functions:")
        for function in find_functions(text):
            marker = "  <- attention" if function.looks_like_attention() else ""
            print(f"      {function.signature}{marker}")
        return 0

    if args.stop_after == "bind":
        return 0

    print("[3/5] attention  lower + partition into fixed tiles")
    if args.skip_attention:
        print("    skipped (--skip-attention)")
        tiled, plan_info = text, None
    else:
        tiled, plan_info = lower_attention(
            text,
            attention_func=args.attention_func,
            config=config,
            allow_untiled=args.allow_untiled_attention,
            force_layout=args.layout,
            q_offset=args.q_offset,
        )
    tiled_path.write_text(tiled)

    if plan_info is not None:
        spec = plan_info["shape"]
        shape = AttentionShape(
            batch=spec["batch"],
            heads=spec["heads"],
            seq=spec["seq"],
            kv_seq=spec["kv_seq"],
            head_dim=spec["head_dim"],
            dtype=spec["dtype"],
        )
        plan = TilePlan(plan_info["tiles"]["q_tile"], plan_info["tiles"]["kv_tile"])
        kernel_path.write_text(
            emit_kernel(
                shape,
                plan,
                causal=bool(config["attention"]["causal"]),
                mask_shape=(
                    tuple(plan_info["mask_shape"])
                    if "mask_shape" in plan_info
                    else None
                ),
                mask_dtype=plan_info.get("mask_dtype"),
                func_name=plan_info["kernel"],
                budget_bytes=int(config["memory"]["tile_bytes"]),
                safety_factor=float(config["memory"]["safety_factor"]),
                reserved_bytes=int(config["memory"]["reserved_bytes"]),
            )
        )
        plan_path.write_text(json.dumps(plan_info, indent=2) + "\n")
        print(f"    kernel     {kernel_path}")
        print(f"    plan       {plan_path}")
        if args.print_plan:
            print(json.dumps(plan_info, indent=2))
    elif args.print_plan:
        print("    no attention tile plan")

    if args.stop_after == "attention":
        return 0

    print("[4/5] verify     iree-opt --verify-each")
    if args.skip_verify:
        print("    skipped (--skip-verify)")
    else:
        iree_opt = (
            config["target"]["iree_opt"]
            if args.dry_run
            else resolve_tool(config["target"]["iree_opt"])
        )
        run(
            [iree_opt, "--verify-each", str(tiled_path), "-o", os.devnull],
            dry_run=args.dry_run,
        )

    if args.stop_after == "verify":
        return 0

    print(f"[5/5] compile    {config['target']['name']}")
    iree_compile = (
        config["target"]["iree_compile"]
        if args.dry_run
        else resolve_tool(config["target"]["iree_compile"])
    )
    substitutions = _Substitutions(
        tile_bytes=int(config["memory"]["tile_bytes"]),
        stack_bytes=int(config["memory"]["stack_bytes"]),
        required_bytes=(
            plan_info["budget"]["required_bytes"]
            if plan_info
            else int(config["memory"]["tile_bytes"])
        ),
    )
    flags = [
        flag.format_map(substitutions) for flag in config["target"]["compile_flags"]
    ]
    vmfb = out_dir / f"{args.model.stem}.vmfb"
    run(
        [
            iree_compile,
            str(tiled_path),
            *flags,
            *args.extra_compile_flag,
            "-o",
            str(vmfb),
        ],
        dry_run=args.dry_run,
    )
    print(f"    output     {vmfb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
