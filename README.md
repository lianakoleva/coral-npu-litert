## Coral NPU with LiteRT

`scripts/compile_litert.py` compiles the LiteRT graph ahead of time for the
Coral NPU's RISC-V core. It imports the flatbuffer to TOSA, binds the dynamic
sequence length, replaces the multi-head attention function with a statically
tiled TOSA kernel sized to a fixed per-tile memory budget, verifies the result,
and hands it to `iree-compile`:

```bash
python scripts/npu/compile_litert.py model.litertlm \
  --seq-len 128 --budget-bytes 262144
```

See [`scripts/README.md`](scripts/README.md) for the attention
contract, the tile/memory model, and the Coral NPU target flags.
