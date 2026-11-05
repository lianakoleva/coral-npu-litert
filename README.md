## Coral NPU with LiteRT

`scripts/compile_litert.py` compiles a LiteRT graph ahead of time for the
Coral NPU's RISC-V core. It imports the flatbuffer to TOSA, binds the dynamic
sequence length, replaces the multi-head attention function with a statically
tiled TOSA kernel sized to a fixed per-tile memory budget, verifies the result,
and hands it to `iree-compile`:

```bash
python scripts/compile_litert.py model.litertlm \
  --seq-len 128 --budget-bytes 262144 --profile coralnpu
```

`--profile coralnpu` targets the real Coral NPU IREE HAL device
(`--iree-hal-target-device=coralnpu`). The default `baseline` profile is a
generic RV32 `llvm-cpu` fallback for host-side experiments.

This repository currently contains a working **AOT reference pipeline**. Making
Coral NPU a supported LiteRT backend requires a Compiler Plugin `.so` and a
Dispatch `.so`; see [`docs/litert-backend.md`](docs/litert-backend.md) for the
architecture, ABI checklist, and milestones.

See [`scripts/README.md`](scripts/README.md) for the attention contract, the
tile/memory model, and the Coral NPU target flags. Run the tests with:

```bash
pip install -r requirements-dev.txt
python -m pytest
```
