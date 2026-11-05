# Coral NPU LiteRT compiler plugin

`libLiteRtCompilerPlugin_CoralNpu.so`: the build-time half of the Coral NPU
backend.

## What is here

| File | Role |
| --- | --- |
| `coralnpu_compiler_plugin.cc` | `LiteRtCompilerPluginQueryInterface`, the `LiteRtCompilerPluginInterface_V1` table, partitioning, serialization, result accessors |
| `coralnpu_compile_backend.h` | `CompileBackend` seam (`CompileRequest` → `CompileArtifact`) |
| `aot_compile_backend.cc` | Production backend: writes the partitions out and runs `scripts/compile_litert.py` |

## Pipeline

1. **Partition.** `LiteRtCompilerPluginPartition` claims every op of the
   subgraph as partition 0. Coral NPU wants the whole graph, because IREE does
   its own host/NPU split inside the compiled module; splitting earlier only
   fragments the analysis. Ops the toolchain cannot lower surface as an IREE
   diagnostic naming the op, instead of a silent partial offload.
2. **Serialize.** `compile()` calls
   `LiteRtCompilerPluginCompile`'s
   `context->serialize_model_with_signatures()` with one signature key per
   partition (`partition_0`, `partition_1`, ...), producing a single TFLite
   flatbuffer whose subgraphs are the partitions.
3. **Compile.** The flatbuffer goes to `GetCompileBackend()->Compile()`. The AOT
   backend writes it into a scratch directory and runs
   `scripts/compile_litert.py`, which owns the TOSA conversion, the
   tiled-attention rewrites, and the dynamic-shape binding.
4. **Return.** One VMFB plus one call info per partition. Every partition lives
   in the same module, and `partition_N` is the exported function name the
   dispatch library calls.

The driver runs as a subprocess, so this library has no build-time dependency
on Python, IREE, or the Coral NPU toolchain. A missing driver is a clear runtime
error rather than a link error.

## Configuration

```text
CORALNPU_AOT_DRIVER        path to scripts/compile_litert.py   (required)
CORALNPU_PYTHON            interpreter                         (default python3)
CORALNPU_AOT_PROFILE       compile_litert --profile             (default coralnpu)
CORALNPU_AOT_SEQ_LEN       --seq-len for dynamic sequence extents
CORALNPU_AOT_EXTRA_FLAGS   extra whitespace-split flags
CORALNPU_AOT_KEEP_TEMP     keep the scratch directory
```

`CORALNPU_AOT_EXTRA_FLAGS=--attention-func=attn_0` pins the attention function
for LLM graphs; generic partitions pass through with
`--allow-untiled-attention`.

## Ownership note

`serialize_model_with_signatures` returns memory owned by an
`OwningBufferRef<uint8_t>` whose default allocator is `Newlocator`, i.e. the
bytes come from `new uint8_t[]`. The plugin releases them with `delete[]` in an
exception-safe guard. Using `free()` there corrupts the heap.

## Tests

```bash
ctest --test-dir build/coralnpu --output-on-failure
```

`coralnpu_compiler_plugin_metadata` covers the interface table, SoC gate,
transformations, and null-result handling. `coralnpu_compile_backend` covers
argument construction, scratch handling, exit-code propagation, and the
`partition_N` call infos.

The graph-walking paths need a `LiteRtCompilerContext` and are not yet covered
by a host test; that requires a LiteRT runtime build and is tracked in
`docs/litert-backend.md`.

## Non-goals

- Do not link the dispatch runtime into this library.
- Do not reimplement LiteRT ingest; use the standard TOSA converter.
