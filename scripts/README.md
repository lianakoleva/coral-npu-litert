# Coral NPU compilation: LiteRT LLM → tiled TOSA

`compile_litert.py` turns a LiteRT LLM flatbuffer into IREE input for the
Coral NPU RISC-V target, replacing the multi-head attention subgraph with
statically shaped TOSA whose sequence axis is explicitly partitioned into
fixed memory tiles.

Three things make this necessary on an NPU:

1. **Static shapes.** LiteRT LLM exports carry a dynamic sequence extent.
   TOSA and the NPU codegen want concrete extents, so the script binds that
   one dimension before lowering.
2. **Bounded scratchpad.** A stock attention lowering materialises the whole
   `seq × kv_seq` score matrix. That does not fit a fixed SRAM/TCM window, so
   the attention kernel is emitted with query/key-value tiling and a streaming
   (online) softmax that never holds a full score matrix.
3. **No runtime scheduler.** Tiling is unrolled at compile time, so the memory
   footprint is a property of the IR rather than of a runtime heuristic. The
   emitted function records its tile sizes and peak byte estimate as
   attributes, and the planner has already checked them against the budget.

The result is plain TOSA (`tosa.matmul`, `tosa.slice`, `tosa.reduce_*`,
`tosa.exp`, …) plus `scf`-free straight-line code, followed by the normal
`iree-compile` RV32 pipeline.

## Requirements

- IREE command line tools:

  ```bash
  pip install iree-base-compiler iree-tools-tflite
  ```

  The generated IR targets the current TOSA dialect where shape parameters
  (`slice` starts, `reshape` targets, `tile` multiples, `padding`) are
  `!tosa.shape` operands produced by `tosa.const_shape`.

  The emitter is written to verify on both IREE 3.12 (the release
  `coralnpu-compiler` pins) and 3.13. Four TOSA custom assemblies changed
  between those releases — `tosa.const_shape`, `tosa.transpose`,
  `tosa.concat`, and `tosa.reduce_*` — and each spelling is rejected by the
  other version, so those ops are emitted in the generic (quoted) form:

  ```mlir
  %0 = "tosa.const_shape"() {values = dense<[2, 2, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = "tosa.transpose"(%v) {perms = array<i32: 0, 2, 1>} : (tensor<2x2x4xf32>) -> tensor<2x4x2xf32>
  ```

  Everything else uses normal custom assembly, which is stable. Prefer the
  IREE release your Coral NPU SDK uses; older revisions (pre-3.12) used
  attributes instead of `!tosa.shape` operands, which this emitter does not
  produce.
- A LiteRT model: `.tflite`, or a `.litertlm` bundle that contains one.
  You can also feed an already-imported `.mlir` file (useful for iterating).

## Quick start

```bash
# 1. See what the importer produced and pick the attention function.
python scripts/compile_litert.py model.litertlm \
    --seq-len 128 --list-functions

# 2. Tile attention and stop before invoking IREE.
python scripts/compile_litert.py model.litertlm \
    --seq-len 128 \
    --budget-bytes 262144 \
    --attention-func attention_0 \
    --stop-after attention

# 3. Full pipeline: import, bind, tile, verify, compile for Coral NPU.
python scripts/compile_litert.py model.litertlm \
    --seq-len 128 \
    --budget-bytes 262144 \
    --attention-func attention_0 \
    --profile coralnpu \
    --iree-compile /path/to/coralnpu/iree-compile
```

`--budget-bytes` is the fixed per-tile memory budget. The default (`256 KiB`)
is a placeholder: set it to the SRAM/TCM window your Coral NPU build exposes
to a kernel.

Useful flags: `--profile {baseline,coralnpu}`, `--q-tile`, `--kv-tile` (force a
plan), `--causal`/`--no-causal`, `--q-offset`, `--print-plan`, `--print-config`,
`--report-tensors`, `--dry-run`,
`--stop-after {ingest,bind,attention,verify,compile}`.

This is an AOT reference pipeline, not yet a LiteRT accelerator. See
[`docs/litert-backend.md`](../docs/litert-backend.md) for the Compiler Plugin
and Dispatch `.so` integration plan.

## Pipeline

| Stage | Output | What happens |
| --- | --- | --- |
| 1. ingest | `01_tosa.mlir` | `iree-import-tflite` converts the flatbuffer to TOSA. `.litertlm` is unpacked first (zip member, or the first `TFL3` marker). |
| 2. bind | `02_static.mlir` | Every dynamic tensor extent is bound to `--seq-len`. The graph must have one dynamic extent, which is the normal LiteRT LLM export. |
| 3. attention | `03_tiled.mlir`, `mhsa_tiled.mlir`, `tile_plan.json` | The attention function body is replaced with a call to a generated, statically tiled TOSA kernel. Signature and layout are preserved. |
| 4. verify | — | `iree-opt --verify-each` parses and verifies every op. |
| 5. compile | `<model>.vmfb` | `iree-compile --iree-input-type=tosa` with the Coral NPU RV32 target flags. |

The scripts never touch the original model. They only write under `--out-dir`
(default `build/npu`).

## Attention contract

The tiling rewriter works on a **function boundary**, which keeps it safe: it
replaces one `func.func` and leaves call sites untouched.

- Signature: `(query, key, value[, additive_mask]) -> attention_output`.
- Layout: `[B, S, H, D]` (default), `[B, H, S, D]`, or packed `[B*H, S, D]`;
  detected from the types, or forced with `--layout`.
- The function must contain only attention (typically the query/key/value
  projections stay with the caller). Auto-detection is by name
  (`attention`, `attn`, `mhsa`, `sdpa`, `flash`) and can be overridden with
  `--attention-func`.
- Default masking is causal. `--no-causal` disables it; a fourth floating-point
  argument is treated as an additive mask of shape `[B, 1, S, S_kv]`,
  `[B, H, S, S_kv]`, or packed `[B*H, S, S_kv]`.
- `--q-offset` sets the absolute KV position of the first query row. The default
  (`kv_seq - seq`) is correct for prefill (`seq == kv_seq`) and for decode
  (`seq == 1`, all cached keys visible).

If the imported graph keeps fused attention inside the main function, the
script stops with an actionable error. Outlining the attention call in the
exporter (or adding a wrapper function) is what makes the substitution
well-defined; a heuristic rewrite of an arbitrary fused body would be unsafe.

## How the tensor sizes are partitioned

For a problem with `BH` packed batch·heads, head dimension `D`, query tile
`QT`, and key/value tile `KT`, the schedule has these live sets:

| Stage | Live bytes (elements × element size) |
| --- | --- |
| prep (scale Q) | `BH · 3·QT·D` |
| scores | `BH · (QT·D + KT·D + 2·QT·KT + causal·2·QT·KT + mask·2·QT·KT)` |
| softmax | `BH · (3·QT·KT + 4·QT)` |
| value | `BH · (QT·KT + KT·D + 3·QT·D)` |
| epilogue | `BH · 2·QT·D` |

`peak = max(stage)` and `required = ceil(peak · safety_factor) + reserved`.
The planner searches the configured candidate tile sizes, rejects plans that
do not satisfy `required ≤ budget`, and prefers the least padding and then the
largest tile. With the defaults (`safety_factor = 1.25`,
`reserved = 4 KiB`) and a 256 KiB budget:

```text
shape   B=1 H=8 S=128 KV=1024 D=64 f32
tiles   q=16 kv=32  (padded 128x1024, 8x32 = 256 unrolled steps)
memory  peak 176 KiB x 1.25 + 4 KiB = 224 KiB <= 256 KiB
```

The softmax recurrence makes key/value tiling exact without recomputation:

```text
m'  = max(m, rowmax(s))
p   = exp(s - m')
l   = l * exp(m - m') + rowsum(p)
acc = acc * exp(m - m') + p · v
out = acc / l
```

The first tile's maximum is clamped to `>= 0` (softmax is invariant to a
per-row constant), and the denominator is floored to keep fully masked rows
finite instead of producing `0/0`.

## Targeting the Coral NPU

There are two target profiles.

**`--profile coralnpu` (recommended).** This is the real Coral NPU IREE HAL
device used by
[`coralnpu-compiler`](https://github.com/google-coral/coralnpu-compiler):

```text
--iree-hal-target-device=local
--iree-hal-local-target-device-backends=llvm-cpu
--iree-llvmcpu-target-cpu=host
--iree-hal-target-device=coralnpu
--iree-opt-level=O2
```

It emits host + Coral NPU executables, so the module carries two HAL devices
and `iree-run-module` needs both drivers in module order:
`iree-run-module --device=local-task --device=coralnpu --module=model.vmfb`
(MPACT simulator by default; put the directory containing
`libcoralnpu_simulator_mpact.so` on `LD_LIBRARY_PATH`), or use the LiteRT
dispatch library.

**`--profile baseline` (default).** A conservative bare-metal RV32 + vector
`llvm-cpu` target. It is useful for host-side experiments and for checking the
generated TOSA without a Coral NPU build, but it does **not** produce a module
the Coral NPU HAL can dispatch:

```text
--iree-input-type=tosa
--iree-hal-target-device=local
--iree-hal-local-target-device-backends=llvm-cpu
--iree-llvmcpu-target-triple=riscv32-unknown-elf
--iree-llvmcpu-target-cpu=generic-rv32
--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl128b,+zve32f
--iree-llvmcpu-target-abi=ilp32d
--iree-llvmcpu-link-embedded=true
--iree-llvmcpu-debug-symbols=false
--iree-llvmcpu-stack-allocation-limit={stack_bytes}
--iree-opt-level=O2
```

To pin the exact ISA/linking flags and memory budget for your SDK, copy the
example config and edit it:

```bash
cp scripts/coralnpu.json npu.json
# edit npu.json: target.iree_compile, target.compile_flags, memory.tile_bytes
python scripts/compile_litert.py model.tflite --seq-len 128 --config npu.json
```

`{tile_bytes}`, `{stack_bytes}`, and `{required_bytes}` are substituted in
compile flags.

## Artifacts

- `tile_plan.json` — chosen tiles, padded extents, stage-by-stage byte
  estimates, and the raw IR function attributes (`tile_q`, `tile_kv`,
  `required_bytes`, `unrolled_steps`).
- `mhsa_tiled.mlir` — the standalone kernel, useful for `iree-run-module`
  experiments or for linking into another pipeline.
- `03_tiled.mlir` — the full module with the attention function replaced.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The unit tests need only Python and NumPy. They cover the planner arithmetic,
the emitted IR shape/partitioning, dynamic-extent binding, function splicing,
the driver pipeline through the attention stage, and the online-softmax
recurrence against a NumPy reference.

With IREE on `PATH`, the integration tests additionally run
`iree-opt --verify-each` on the emitted TOSA and execute the tiled kernel with
`iree-run-module`, comparing it to NumPy. If the installed `iree-compile`
exposes `--iree-hal-target-device=coralnpu`, the suite also compiles and runs
the kernel on the MPACT simulator; otherwise that case is skipped. The simulator
dylib (`libcoralnpu_simulator_mpact.so`) must be on `LD_LIBRARY_PATH`, and the
module's two devices mean `iree-run-module` gets both `--device=` flags.

```bash
pip install iree-base-compiler iree-base-runtime iree-tools-tflite
python -m pytest tests/test_integration_iree.py
```

The suite is green on IREE **3.12.0** (the version `coralnpu-compiler` pins)
and on **3.13.0**; CI runs both (see `.github/workflows/tests.yml`). The
emitted kernel is identical for both — see the note on generic op syntax under
[Requirements](#requirements).

`--verify-each` is what catches TOSA dialect drift; the numerical run is what
catches tile-offset, padding, and causal-mask mistakes.

## Limits

- Only attention is tiled. Everything else (projections, MLP, embedding
  lookup) is handed to IREE's own scheduling; `--report-tensors` lists the
  largest types so you can see where the remaining pressure is.
- Tiling is unrolled: IR size grows with the number of tiles. The default cap
  is 8192 steps (`memory.max_unrolled_tiles`); raise it or increase the budget
  for very long contexts.
- The emitted kernel is floating point (`f32`, `f16`, `bf16`). Quantized
  attention is not synthesised; quantized projections stay as imported.
- Padded shapes are sliced back, and padded keys are masked causally. With
  `--no-causal`, `kv_seq` must be a multiple of `kv_tile`; an additive mask
  additionally requires exact KV tiles.
- `.litertlm` extraction is best effort. If the container format changes,
  pass the `.tflite` file directly.
- This is not yet a LiteRT dispatch backend. The scripts produce a `.vmfb` you
  can run with `iree-run-module`; the compiler-plugin and dispatch-library work
  is tracked in [`docs/litert-backend.md`](../docs/litert-backend.md).
