# LiteRT Coral NPU vendor backend

The two LiteRT shared libraries that make Coral NPU a supported `CompiledModel`
accelerator.

See [`docs/litert-backend.md`](../../../docs/litert-backend.md) for target
flags, partitioning strategy, board bring-up, and the full test matrix.

## Artifacts

```text
libLiteRtCompilerPlugin_CoralNpu.so   # partition + compile
libLiteRtDispatch_CoralNpu.so         # execute compiled VMFB on Coral NPU
```

Neither library is a fork of LiteRT, and neither links `libLiteRt.so`. They
implement the vendor ABI from `litert/vendors/c/` and are loaded by the stock
LiteRT runtime.

## Layout

```text
common/
  coralnpu_common.h            # element sizes, TensorByteSize, log macros
  coralnpu_subprocess.h        # scratch dirs, fork/exec, file I/O
compiler_plugin/
  coralnpu_compiler_plugin.cc  # LiteRtCompilerPlugin* entry points
  coralnpu_compile_backend.h   # CompileBackend seam
  aot_compile_backend.cc       # drives scripts/compile_litert.py
dispatch/
  coralnpu_dispatch.cc         # LiteRtDispatchQueryInterface + V1 table
  coralnpu_executor.h          # DispatchExecutor seam
  coralnpu_executor_unavailable.cc
  coralnpu_executor_command.cc
tests/
BUILD.bazel
CMakeLists.txt
```

## Building

```bash
cmake -S litert/vendors/coralnpu -B build/coralnpu \
    -DLITERT_SOURCE_ROOT=/path/to/LiteRT \
    -DCORALNPU_WITH_COMMAND_EXECUTOR=ON
cmake --build build/coralnpu
ctest --test-dir build/coralnpu --output-on-failure
```

`LITERT_SOURCE_ROOT` only needs the C headers, `tflite/c`, and
`litert/build_common/build_config.h`; a full LiteRT build is not required. The
same commands work on an aarch64 board with its native g++.

Executor selection (exactly one):

| Flag | Executor |
| --- | --- |
| *(none)* | unavailable — validates everything, then reports how to build a working one |
| `-DCORALNPU_WITH_COMMAND_EXECUTOR=ON` | `iree-run-module` subprocess; no IREE at build time |
| `-DCORALNPU_WITH_IREE_RUNTIME=ON` | not implemented; fails with an explanation |

## Compiler plugin contract

Partition claims every op of the subgraph as partition 0 (Coral NPU wants the
whole graph; IREE does its own host/NPU split inside the module). `compile()`
serializes the partitions with one signature key each (`partition_0`,
`partition_1`, ...), then hands the flatbuffer to the `CompileBackend`, which by
default runs `scripts/compile_litert.py`. One VMFB comes back, with one exported
function per partition.

Configuration is environment-based so the ABI stays fixed:

```text
CORALNPU_AOT_DRIVER        path to scripts/compile_litert.py   (required)
CORALNPU_PYTHON            interpreter                         (default python3)
CORALNPU_AOT_PROFILE       compile_litert --profile             (default coralnpu)
CORALNPU_AOT_SEQ_LEN       --seq-len for dynamic sequence extents
CORALNPU_AOT_EXTRA_FLAGS   extra whitespace-split flags
CORALNPU_AOT_KEEP_TEMP     keep the scratch directory
```

## Dispatch contract

Exports a single C symbol, `LiteRtDispatchQueryInterface`, returning a
`LiteRtDispatchInterface_V1` table. The invocation context owns the embedded
VMFB bytes and the entry-point name; `LiteRtDispatchInvoke` resolves the
registered buffers and calls the linked executor.

Requirements currently advertise `kLiteRtTensorBufferTypeHostMemory` only. The
executor reaches the buffers through
`runtime_context->get_tensor_buffer_host_memory`.

Command executor configuration:

```text
CORALNPU_RUN_MODULE               path to iree-run-module
CORALNPU_IREE_BIN                 directory containing it
CORALNPU_DEVICE                   torq (NPU) / local-task / local-sync
CORALNPU_RUN_MODULE_EXTRA_FLAGS   extra whitespace-split flags
CORALNPU_DISPATCH_KEEP_TEMP       keep the scratch directory
```

## Status

Both `.so` targets are implemented and built on macOS/arm64, on an aarch64
Coralboard, and on aarch64 Linux with clang 19. Five ABI tests pass on all
three, including one that `dlopen()`s the built `.so` and drives the interface
table for real; with a VMFB it performs an actual invocation through
`iree-run-module` (`tosa.abs`, outputs match).

The command executor has also been verified against a real `coralnpu-compiler`
build: `CORALNPU_DEVICE=coralnpu` drives the MPACT simulator and the shipped
library returns the expected `tosa.abs` outputs. Not done: the in-process IREE
executor (`coralnpu_executor_iree.cc`).
