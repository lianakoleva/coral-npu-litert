# Coral NPU LiteRT dispatch library

`libLiteRtDispatch_CoralNpu.so`: the target-side half of the Coral NPU backend.

## What is here

| File | Role |
| --- | --- |
| `coralnpu_dispatch.cc` | Implements `LiteRtDispatchQueryInterface`, the `LiteRtDispatchInterface_V1` table, the tensor-buffer registry, invocation contexts, and argument validation |
| `coralnpu_executor.h` | `DispatchExecutor` seam + `TensorArg` |
| `coralnpu_executor_unavailable.cc` | Default: validates everything, returns `Unsupported` with a build hint |
| `coralnpu_executor_command.cc` | Stages `.npy` inputs, runs `iree-run-module`, copies outputs back |
| `coralnpu_executor_iree.cc` | Not implemented; the build option fails with an explanation |

Exactly one executor is linked; `CMakeLists.txt` selects it and rejects
conflicting flags.

## Behaviour

- `LiteRtDispatchQueryInterface` rejects runtimes older than 1.0 with
  `kLiteRtStatusErrorUnsupportedRuntimeVersion`.
- Only `kLiteRtDispatchExecutableTypeMlModel` is accepted: the compiler plugin
  emits VMFBs, not DSP libraries or JIT handles.
- Buffer requirements advertise `kLiteRtTensorBufferTypeHostMemory`. Sub-byte
  element types round up to whole bytes; dynamic dimensions are rejected
  because the compiler binds all shapes ahead of time.
- The tensor type from `get_{input,output}_requirements` is recorded on the
  invocation context and passed to the executor, which needs it to describe
  tensors to a backend.
- `invoke` refuses to run unless every input and output is attached, rather
  than executing partially.

## Command executor

Propagates nothing implicitly: everything is configured through the
environment.

```text
CORALNPU_RUN_MODULE               path to iree-run-module
CORALNPU_IREE_BIN                 directory containing it
CORALNPU_DEVICE                   torq (Coral NPU) / local-task / local-sync
CORALNPU_RUN_MODULE_EXTRA_FLAGS   extra whitespace-split flags
CORALNPU_DISPATCH_KEEP_TEMP       keep the scratch directory
```

Inputs become `.npy` files (so the tool sees shape and dtype, never guessed
bytes); outputs are read back as raw `.bin` and byte-count checked against the
tensor type before copying, so a short output is an error rather than a
truncated tensor.

Packed 4-bit, fp8, and bfloat16 are rejected with an explicit `Unsupported` —
numpy cannot represent them, and passing their bytes through as `int8` would
produce a module that runs and returns wrong numbers.

## Tests

```bash
ctest --test-dir build/coralnpu --output-on-failure
```

`coralnpu_dispatch_abi` covers the interface table and validation logic;
`coralnpu_command_executor` covers staging and the copy-back path with a stub
tool; `coralnpu_dlopen_smoke` loads the built `.so` with `dlopen()` and can run
a real VMFB:

```bash
CORALNPU_RUN_MODULE=/path/to/iree-run-module \
  ./build/coralnpu_dlopen_smoke_test \
  ./build/libLiteRtDispatch_CoralNpu.so /path/to/module.vmfb
```

Verified on an aarch64 Coralboard: `real invocation: 6 outputs match tosa.abs`.

## Non-goals

- Do not link `libIREECompiler` into this library.
- Do not implement a CPU fallback; LiteRT keeps unsupported ops on CPU.
- Do not link `libLiteRt.so`; only the versioned C interface tables are used.
