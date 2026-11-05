# Coral NPU as a LiteRT backend

This document describes the LiteRT accelerator integration for Coral NPU: a
**Compiler Plugin** shared library and a **Dispatch** shared library, both
implemented in `litert/vendors/coralnpu/`.

The AOT scripts in `scripts/` solve the hard model-level problem (static-shape,
memory-bounded attention). The vendor libraries turn that work into
`CompiledModel` with `HardwareAccelerator.NPU`.

## Status

| Piece | State |
| --- | --- |
| TOSA attention tiling and budget planner | Implemented in `scripts/tosa_mhsa.py` |
| LiteRT flatbuffer ingest + dynamic-shape binding | Implemented in `scripts/compile_litert.py` / `scripts/mlir_text.py` |
| Coral NPU IREE target profile | `--profile coralnpu`, `scripts/coralnpu.json` |
| Python test suite | `tests/` — 117 passed, 1 skipped on IREE 3.12/3.13; **119 passed with `coralnpu-compiler`** |
| `libLiteRtCompilerPlugin_CoralNpu.so` | **Implemented**; whole-graph partition, AOT compile |
| `libLiteRtDispatch_CoralNpu.so` | **Implemented**; ABI, buffer registry, 3 executors |
| Vendor ABI tests | **5 tests, pass on macOS/arm64, the board (g++ 13), and aarch64 Linux (clang 19)** |
| Real invocation through the shipped `.so` | **Verified on the board and on the MPACT simulator** (`tosa.abs`, outputs match) |
| `--device=coralnpu` (MPACT simulator) run | **Verified**: `coralnpu-compiler` built; tiled attention is 24/24 dispatches on the NPU |
| ATS registration and CI | Not implemented |

Neither library links `libLiteRt.so`. They speak only the versioned C interface
tables in `litert/vendors/c/`, which is what lets them be built and shipped
separately from a LiteRT release.

## The two required shared libraries

LiteRT does **not** require forking `libLiteRt.so`. It loads vendor libraries:

```text
libLiteRtCompilerPlugin_CoralNpu.so   # host-side AOT compile
libLiteRtDispatch_CoralNpu.so         # target-side execution
```

Both are built by `litert/vendors/coralnpu/CMakeLists.txt` and
`litert/vendors/coralnpu/BUILD.bazel`.

### Compiler Plugin

Source: `compiler_plugin/coralnpu_compiler_plugin.cc`.

The library exports exactly one symbol, `LiteRtCompilerPluginQueryInterface`,
which returns the `LiteRtCompilerPluginInterface_V1` table.

Policy:

1. **Partition.** `LiteRtCompilerPluginPartition` claims every op of the
   subgraph as partition 0. Coral NPU wants the whole graph: IREE does its own
   host/NPU partitioning inside the compiled module, so splitting earlier would
   only fragment the analysis. Anything the toolchain cannot lower surfaces as
   an IREE diagnostic naming the op, rather than a silent partial offload.
2. **Serialize.** `compile()` calls
   `context->serialize_model_with_signatures()` with one signature key per
   partition (`partition_0`, `partition_1`, ...), producing a single TFLite
   model whose subgraphs are the partitions.
3. **Compile.** The serialized model goes to the `CompileBackend` seam
   (`compiler_plugin/coralnpu_compile_backend.h`). The production backend
   (`aot_compile_backend.cc`) writes it to a scratch directory and runs
   `scripts/compile_litert.py`, which owns the TOSA conversion, the attention
   tiling, and the shape binding.
4. **Return.** One VMFB plus one call info per partition. All partitions live
   in the single module; `partition_N` is the exported function name.

Compilation is configured through the environment, so no ABI-visible struct
grows:

| Variable | Meaning |
| --- | --- |
| `CORALNPU_AOT_DRIVER` | path to `scripts/compile_litert.py` (**required unless `CORALNPU_AOT_CACHE` holds the artifact**) |
| `CORALNPU_PYTHON` | interpreter (default `python3`) |
| `CORALNPU_AOT_PROFILE` | `compile_litert --profile` (default `coralnpu`) |
| `CORALNPU_AOT_SEQ_LEN` | `--seq-len`, for graphs with a dynamic sequence extent |
| `CORALNPU_AOT_EXTRA_FLAGS` | extra whitespace-split driver flags |
| `CORALNPU_AOT_KEEP_TEMP` | keep the scratch directory |

Because the driver is a subprocess, the plugin has no build-time dependency on
Python, IREE, or the Coral NPU toolchain. A missing driver is a clear runtime
error, not a link error.

### Dispatch

Source: `dispatch/coralnpu_dispatch.cc`.

The library exports exactly one symbol, `LiteRtDispatchQueryInterface`, which
returns the `LiteRtDispatchInterface_V1` table. `coralnpu_dispatch.cc` owns the
ABI, the tensor-buffer registry, invocation contexts, and argument validation;
execution is behind the `DispatchExecutor` seam in
`dispatch/coralnpu_executor.h`.

Three executors are provided; exactly one is linked:

| Executor | Build flag | Behaviour |
| --- | --- | --- |
| `coralnpu_executor_unavailable.cc` | default | Validates everything, then returns `kLiteRtStatusErrorUnsupported` with a build hint. Lets the ABI be conformance-tested anywhere. |
| `coralnpu_executor_command.cc` | `-DCORALNPU_WITH_COMMAND_EXECUTOR=ON` | Stages inputs as `.npy`, runs `iree-run-module` as a subprocess, copies outputs back. **No IREE at build time.** |
| `coralnpu_executor_iree.cc` | `-DCORALNPU_WITH_IREE_RUNTIME=ON` | **Not implemented.** In-process IREE linking; the build option fails with an explanation instead of shipping untested code. |

The command executor is the recommended bring-up path: it follows whatever IREE
runtime the device ships, and `--device=torq` selects the Coral NPU. Its cost is
one process per invocation plus tensor files, which is fine for correctness
work and not intended for latency-critical deployment.

Command executor configuration:

| Variable | Meaning |
| --- | --- |
| `CORALNPU_RUN_MODULE` | path to `iree-run-module` (default: `$CORALNPU_IREE_BIN/iree-run-module`, then `$PATH`) |
| `CORALNPU_DEVICE` | `torq` for the Coral NPU; `local-task`/`local-sync` for CPU (default `local-task`) |
| `CORALNPU_RUN_MODULE_EXTRA_FLAGS` | extra whitespace-split flags |
| `CORALNPU_DISPATCH_KEEP_TEMP` | keep the scratch directory |

Requirements currently advertise only `kLiteRtTensorBufferTypeHostMemory`; the
executor reads and writes those buffers through
`runtime_context->get_tensor_buffer_host_memory`. Adding a DMA-BUF path means
adding the type here and using `lock_tensor_buffer` in the executor.

For embedded/Linux-IoT builds, either library can be statically registered
(`LiteRtStaticLinkedDispatchGetApi`) instead of loaded with `dlopen`.

## What maps from this repository

| Current code | Backend role |
| --- | --- |
| `scripts/compile_litert.py` | the compiler plugin's `CompileBackend` |
| `scripts/mlir_text.py` | function discovery/splicing |
| `scripts/tosa_mhsa.py` | TOSA attention tiling + budget planner |
| `scripts/coralnpu.json`, `--profile coralnpu` | target flags for `iree-compile` |
| `scripts/check_attention_numerics.py` | numerical test seed |
| `tests/` | unit, IR-verification, numerical, integration coverage |

Long term, `tosa_mhsa.py` should become an MLIR pass so the pipeline does not
manipulate MLIR as text. The text-based path is acceptable for the AOT
reference implementation and for the first vertical slice.

## Target selection

The baseline profile is a generic RV32 `llvm-cpu` build and is **not** the
Coral NPU HAL device:

```text
--iree-llvmcpu-target-triple=riscv32-unknown-elf
--iree-llvmcpu-target-cpu=generic-rv32
```

The real target, matching `coralnpu-compiler`'s own examples, is:

```text
--iree-hal-target-device=local
--iree-hal-local-target-device-backends=llvm-cpu
--iree-llvmcpu-target-cpu=host
--iree-hal-target-device=coralnpu
```

Use:

```bash
python scripts/compile_litert.py model.tflite --seq-len 128 \
  --profile coralnpu --budget-bytes 262144
```

or copy `scripts/coralnpu.json` and edit the tool paths/flags.

## Sequence-length policy

`LiteRtCompilerPluginPartition` receives a `LiteRT subgraph` of flatbuffer ops,
not a `func.func`, while the attention rewrite is function-based. The plugin
therefore claims the whole graph (option 1 below) and lets the driver handle
function-level work:

1. **Whole graph as one partition** — implemented.
2. **Attention composite as its own partition** — recognize `odml.mha`/SDPA in
   the partitioner, then tile inside the compiler.
3. **MLIR pass with fine-grained partitioning** — best long-term shape; enables
   CPU fallback for unsupported ops.

LiteRT LLM exports have a dynamic sequence extent and the NPU needs static
shapes, so images must pick a policy:

- `CORALNPU_AOT_SEQ_LEN` (one AOT-compiled module per sequence bucket), or
- separate prefill and decode entry points.

`--seq-len`, `--q-offset`, and `--causal` map directly onto these options.

## Layout

```text
litert/vendors/coralnpu/
  common/
    coralnpu_common.h            # element sizes, TensorByteSize, logging macros
    coralnpu_subprocess.h        # scratch dirs, fork/exec, file I/O
  compiler_plugin/
    coralnpu_compiler_plugin.cc  # LiteRtCompilerPlugin* C ABI
    coralnpu_compile_backend.h   # CompileBackend seam
    aot_compile_backend.cc       # drives scripts/compile_litert.py
  dispatch/
    coralnpu_dispatch.cc         # LiteRtDispatch* C ABI + buffer registry
    coralnpu_executor.h          # DispatchExecutor seam
    coralnpu_executor_unavailable.cc
    coralnpu_executor_command.cc
  tests/
  BUILD.bazel
  CMakeLists.txt
  README.md
```

Build artifacts:

```text
libLiteRtCompilerPlugin_CoralNpu.so
libLiteRtDispatch_CoralNpu.so
```

## Building

```bash
cmake -S litert/vendors/coralnpu -B build/coralnpu \
    -DLITERT_SOURCE_ROOT=/path/to/LiteRT \
    -DCORALNPU_WITH_COMMAND_EXECUTOR=ON
cmake --build build/coralnpu
ctest --test-dir build/coralnpu --output-on-failure
```

`LITERT_SOURCE_ROOT` only needs the C headers (`litert/c`,
`litert/vendors/c`) plus `tflite/c` and `litert/build_common/build_config.h`.
Nothing here requires a full LiteRT build, and the same command works on the
board with its native g++.

Build prerequisites:

- LiteRT source tree for `litert/vendors/c/*.h` and `litert/c/*.h`.
- For the MPACT simulator target: a `coralnpu-compiler` build (see below).
- No IREE at build time for the default or command-executor configurations.

The dispatch library must not pull in `libIREECompiler`; keep compiler and
runtime dependencies separate so device images stay small.

## Testing

### Python (no external compiler needed for the first three)

```bash
python -m pytest tests/test_tosa_mhsa.py tests/test_mlir_text.py \
  tests/test_compile_litert.py tests/test_attention_math.py
```

```bash
pip install iree-base-compiler iree-base-runtime iree-tools-tflite
python -m pytest tests/test_integration_iree.py
```

- `iree-opt --verify-each` catches TOSA dialect drift.
- `iree-run-module` on the host compares the emitted kernel against NumPy.
- The Coral NPU case is skipped unless the installed `iree-compile` exposes
  `--iree-hal-target-device=coralnpu`.

IREE 3.12 and 3.13 changed the custom assembly of four TOSA ops
(`const_shape`, `transpose`, `concat`, `reduce_*`), and neither spelling is
accepted by the other release. 3.12 is what `coralnpu-compiler` pins, so the
emitter uses the **generic** form for exactly those ops; the emitted kernel is
byte-identical for both and the suite is green on both (`117 passed,
1 skipped`). CI runs the integration job against both versions.

`tests/test_tosa_mhsa.py::test_emitted_kernel_uses_version_portable_tosa_syntax`
guards this statically, so a single compiler version on the test machine cannot
hide a regression.

### Vendor ABI tests (C++, no gtest, no absl)

```bash
ctest --test-dir build/coralnpu --output-on-failure
```

| Test | What it pins down |
| --- | --- |
| `coralnpu_dispatch_abi` | Interface negotiation, version rejection, buffer registry, requirements (including int4 rounding), attach/detach, invoke preconditions, null arguments |
| `coralnpu_compile_backend` | Driver argument construction, scratch handling, exit-code propagation, missing VMFB, `partition_N` call infos |
| `coralnpu_compiler_plugin_metadata` | Interface table layout, SoC gate, transformations, null-result handling |
| `coralnpu_command_executor` | Full staging path with a stub tool: `.npy` encoding, flag plumbing, output copy, size-mismatch refusal, missing tool |
| `coralnpu_dlopen_smoke` | The **built `.so`** loaded with `dlopen`, table entries non-null, ABI driven for real |

The tests are plain executables that print the first failure and exit non-zero,
so they run under CTest, under a Bazel `sh_test`, or by hand over `adb shell`.

### Real invocation through the shipped library

`coralnpu_dlopen_smoke_test` takes an optional VMFB and, when given one,
performs a real invocation:

```bash
CORALNPU_RUN_MODULE=/path/to/iree-run-module \
  ./build/coralnpu_dlopen_smoke_test \
  ./build/libLiteRtDispatch_CoralNpu.so /path/to/module.vmfb
```

Verified on the board (aarch64, IREE 3.12 runtime):

```text
EXEC @main
real invocation: 6 outputs match tosa.abs
```

### Still to add

- **LiteRT ATS**: register a `coralnpu` backend in `litert_device_common.bzl`
  and run `--backend=npu` with `--dispatch_dir`/`--plugin_dir`, plus
  `--compile_mode` on the host. ATS compares against the CPU backend and
  reports latency.
- **`run_model`**: `--accelerator=npu --dispatch_library_dir=...`, assert
  `model.is_fully_accelerated()`.
- **Negative tests**: unsupported op → CPU fallback; no attention function →
  actionable error; budget too small → `TilePlanError`; dynamic sequence length
  without a policy → explicit failure.
- **Performance/memory regression**: assert `required_bytes <= tile_bytes`,
  record latency and dispatch counts.

## Board bring-up

Target: Synaptics Astra `Coralboard – Google I/O Edition`.

### The board's Torq NPU is the Coral NPU

`Synaptics Torq` is the first production implementation of the Google Coral NPU
ML core, so `torq://` **is** the on-device access path to the Coral NPU. The
base device tree has a single NPU node, `synpu@f7600000` (`syna,npu`); there is
no separate "Coral" node to enable.

Observed inventory:

| Item | Value |
| --- | --- |
| SoC / kernel | Astra SL2619, Linux 6.12.62, aarch64 |
| CPU / RAM | 2× Cortex-A55, 1.9 GB |
| Toolchain | g++ 13.3, cmake 3.28, python 3.12, root shell |
| System IREE devices | `local-sync`, `local-task`, **`torq`** |
| NPU smoke test | `synap_cli -m .../mobilenetv2/npu/model.synap` → ~35.6 ms |

### Version skew is the current blocker for `--device=torq`

The board's system IREE runtime exposes `torq://` but predates the VM bytecode
feature bits that current compilers emit:

```text
required module features [Ch] are not available in this runtime configuration;
have [EXT_F32|EXT_F64]
```

Both 3.12 and 3.13 compilers produce modules this runtime rejects. The
workaround used for the CPU path is to install a matching runtime next to it:

```bash
adb push <venv>/iree/_runtime_libs /home/root/iree312
# 3.12 runtime devices: local-sync, local-task, vulkan   (no torq)
```

That is why the verified real invocation above runs `--device=local-task`: the
runtime that accepts the module is a CPU-only build. `torq` comes back once a
compiler matching the board's runtime exists, which is exactly what building
`coralnpu-compiler` provides.

### Building `coralnpu-compiler`

**Built and verified.** It needs Bazel 9.1.0, clang 19, lld 19, a JRE (MPACT's
decoder genrule), the IREE and LLVM submodules, and roughly **30 GB** of scratch
space. The build here ran in an Ubuntu 25.04 **aarch64** container (colima,
6 CPU / 10 GiB) with clang-19/lld-19 and Bazel 9.1.0 via bazelisk.

The repo pins IREE at `2b05c5db…`, which is exactly the commit
`iree-base-compiler==3.12.0` is built from.

The recipe below is from `coralnpu-compiler`'s README (`Bazel 9.1.0` comes from
its `.bazelversion`):

```bash
# Prerequisites (Debian/Ubuntu): git, bash>=4, bazel 9.1.0, clang-19, lld-19,
# cmake>=3.26, python3.13, shfmt.
cd ~/coralnpu-compiler
git submodule update --init --depth=1 -j "$(nproc)" -- .
# Skip IREE's duplicate llvm-project checkout (~2.4 GB).
git -C third_party/iree submodule update --init --depth=1 -j "$(nproc)" -- \
    . ":(exclude)third_party/llvm-project"
./scripts/patch-third_party.sh --restore-first all

# Compiler + runtime (the runtime registers the `coralnpu` HAL driver, i.e.
# MPACT simulation), plus the simulator dylib the driver dlopen()s.
bazel build --config=dev \
    @iree_core//tools:iree-compile \
    @iree_core//tools:iree-run-module \
    @coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so
```

> **arm64 hosts.** Upstream gates the simulator runtime on `linux_x86_64`
> (`runtime/sim/BUILD.bazel`, and the comment in `CMakeLists.txt`), so an
> unpatched tree only builds `iree-run-module` on x86_64. The code itself is
> portable C++ (no x86 intrinsics), and the MPACT dylib is compiled from source,
> so deleting the `select()` around `simulator.cpp` / `simulation` in
> `runtime/sim/BUILD.bazel` is enough. That local patch is what produced the
> aarch64 binaries verified below; it is not upstreamed. The shape of the edit
> (same for `simulator`'s `srcs`/`hdrs`):
>
> ```diff
> -    srcs = select({
> -        ":linux_x86_64": ["simulator_backend.c", ...],
> -        "//conditions:default": [],
> -    }),
> -    hdrs = select({
> -        ":linux_x86_64": ["simulator_backend.h", ...],
> -        "//conditions:default": [],
> -    }),
> -    deps = select({
> -        ":linux_x86_64": [":simulator", ...],
> -        "//conditions:default": [],
> -    }),
> +    srcs = ["simulator_backend.c", ...],
> +    hdrs = ["simulator_backend.h", ...],
> +    deps = [":simulator", ...],
> ```
>
> `iree-compile` takes ~3.3 h at `--jobs=4` on 6 vCPUs; the runtime adds a few
> minutes because most of LLVM is cached.

> **arm64 gotcha.** The pinned RV32 toolchain ships x86_64 binaries, which
> Bazel runs through `qemu-x86_64` when building `//crt:libcoralnpu_crt`. The
> container needs the x86_64 loader and runtime libraries available at
> `/lib64/ld-linux-x86-64.so.2` and `/usr/lib/x86_64-linux-gnu`
> (`apt-get download libc6:amd64 libstdc++6:amd64 libgcc-s1:amd64 libzstd1:amd64`
> then `dpkg-deb -x`).

Verified results:

1. **Compilation.** `iree-compile --iree-hal-target-device=coralnpu` lowers both
   a `tosa.abs` module and the emitted tiled attention kernel. With
   `--coralnpu-roofline-speedup-threshold=0
   --coralnpu-dump-affinity-profile-format=pretty`, the attention kernel reports
   **24 dispatches, 100% affinity `coralnpu`**.
2. **Simulation.** The MPACT simulator runs the modules on the host:

   ```bash
   REPO=~/coralnpu-compiler
   SIM=$(cd "$REPO" && bazel cquery --output=files \
       @coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so)
   RUNTIME=$(cd "$REPO" && bazel cquery --output=files \
       @iree_core//tools:iree-run-module)

   # Pure-NPU module:
   LD_LIBRARY_PATH="$(dirname "$REPO/$SIM")" \
     "$REPO/$RUNTIME" --module=abs_coralnpu.vmfb \
     --device=coralnpu --function=main --input=2x3xf32=1.0
   # EXEC @main
   # 2x3xf32=[1 1 1][1 1 1]
   ```

   Modules built with `scripts/coralnpu.json` carry two HAL devices (host
   llvm-cpu fallback first, then `coralnpu`), so pass both drivers **in module
   order**: `--device=local-task --device=coralnpu`.
3. **Shipped dispatch library, end to end.** No code change was needed:

   ```bash
   CORALNPU_RUN_MODULE="$REPO/$RUNTIME" \
   CORALNPU_DEVICE=coralnpu \
   LD_LIBRARY_PATH="$(dirname "$REPO/$SIM")" \
     ./build/coralnpu/coralnpu_dlopen_smoke_test \
     ./build/coralnpu/libLiteRtDispatch_CoralNpu.so abs_coralnpu.vmfb
   # EXEC @main
   # real invocation: 6 outputs match tosa.abs
   ```

   The driver name is `coralnpu` (`runtime/driver/registration/driver_module.c`),
   so the command executor reaches the simulator through the same
   `--device=` switch it uses for `torq` and `local-task`.
4. **Tests.** With `iree-compile`, `iree-opt`, and `iree-run-module` from this
   build on `PATH`, the whole Python suite passes in the container:
   **119 passed, 0 skipped** (the Coral NPU simulated-execution test included).
5. On the board, a compiler matched to its runtime is what makes `--device=torq`
   accept the module; until then the CPU path above is the verified one.

## Milestones

- [x] Test suite for the TOSA pipeline.
- [x] Coral NPU target profile and config.
- [x] Validate emitted TOSA and numerics with IREE on the host.
- [x] `libLiteRtCompilerPlugin_CoralNpu.so` with whole-graph partitioning.
- [x] `libLiteRtDispatch_CoralNpu.so` with the ABI, buffer registry, and three
      executors.
- [x] Vendor ABI tests passing on host and on the board; real invocation
      verified on the board.
- [x] Build `coralnpu-compiler` and run the tiled model on `--device=coralnpu`
      (MPACT simulator).
- [ ] In-process IREE executor (`coralnpu_executor_iree.cc`) against a built
      IREE runtime + Coral NPU HAL driver.
- [ ] Port the tiling path to an MLIR pass or a C++ helper.
- [ ] Register the backend with ATS and add it to CI.
- [ ] Decide the dynamic-sequence and prefill/decode policy.

## Next steps

1. Keep the `coralnpu` simulator path green: rebuild `coralnpu-compiler` when
   bumping the IREE pin, and run
   `python -m pytest tests/test_integration_iree.py -k coralnpu -vv` with the
   built tools on `PATH` and the simulator directory on `LD_LIBRARY_PATH`.
2. Try the board with a matching `coralnpu-compiler` runtime (`--device=torq`)
   now that the build recipe is known, then implement
   `coralnpu_executor_iree.cc` and add a `CORALNPU_WITH_IREE_RUNTIME` CI job
   once a prebuilt IREE runtime with the `coralnpu` driver is available.
3. Register the backend in `litert_device_common.bzl` and run ATS with
   `--backend=npu`, asserting `model.is_fully_accelerated()`.

Keep the AOT driver as the reference implementation and use its artifacts to
debug the plugin; do not delete it when the `.so` targets land.
