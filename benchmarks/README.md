# Gemma 3 270M on the Coral NPU

`bench_gemma3.py` measures the same Gemma 3 270M model three ways on an Astra
SL2619 board (2× Cortex-A55, 1.9 GB RAM, Coral NPU):

| backend | runtime | where the math runs |
| --- | --- | --- |
| `torq` | pre-compiled Torq VMFB via the `torq` IREE device | **Coral NPU** |
| `litert-cpu` | `.litertlm` through LiteRT-LM, XNNPACK | CPU |
| `litert-npu` | `.litertlm` through LiteRT-LM with the Coral NPU dispatch library | NPU when the compiler plugin can AOT-compile, CPU otherwise |

## Results

Board firmware, 2 CPU threads, greedy sampling (`temperature=0`), 1 warmup.
Prompt: *"Explain in three sentences why a small language model can run on an
NPU even though its weights are much larger than the NPU's local memory."*
(~34 prompt tokens.)

| backend | TTFT | decode tok/s | e2e tok/s | tokens | config |
| --- | ---: | ---: | ---: | ---: | --- |
| `torq` (NPU) | **2901 ms** | **10.25** | **7.00** | 62 | 3 runs, 64 max |
| `litert-cpu` | 3942 ms | 3.25 | 2.72 | 61 | 3 runs, 64 max |
| `litert-npu` | 5188 ms | 2.89 | 2.01 | 32 | 1 run, 32 max¹ |

¹ The NPU row is the LiteRT-LM CPU fallback. The compiler plugin loads and
partitions the graph correctly (see below), but the on-device AOT toolchain is
not installed, so LiteRT-LM logs `CORALNPU_AOT_DRIVER is not set`, applies zero
plugins, and executes on XNNPACK. The numbers are therefore CPU numbers and
should be compared with `litert-cpu`, not with `torq`.

**Takeaway:** the Torq NPU path decodes **3.2× faster** than LiteRT-LM on the
CPU and reaches the first token **1.4× sooner**. The 270M weights stream from
DDR through the NPU; they never need to fit in the NPU's local memory.

## Why the Torq path works on this board

The Torq model ships as a pre-compiled `transformer.vmfb`. Nothing is compiled
on the board at run time, so the 2× A55 cores only tokenize, sample, and drive
the NPU.

The board's kernel driver predates the Torq hardware-compatibility ioctl, so
the runtime needs a tiny `LD_PRELOAD` shim that reports hardware id 0
(`benchmarks/torq_hwcompat.c`):

```bash
cc -shared -fPIC -O2 -o libtorq_hwcompat.so benchmarks/torq_hwcompat.c -ldl
LD_PRELOAD=$PWD/libtorq_hwcompat.so \
  /home/root/torq-examples/.venv/bin/python benchmarks/bench_gemma3.py \
    --backend torq \
    --model /home/root/torq-examples/models/Synaptics/gemma-3-270-GPTQ-int4-it-torq/transformer.vmfb \
    --torq-device-allocator cpu \
    --runs 3 --max-tokens 64
```

The prebuilt model has a **256-token KV cache** (33 tokens of warmup). Prompts
longer than ~220 tokens leave no room to generate, so the SayLess assistant
selects the one relevant tool schema instead of sending all ten.

## LiteRT-LM NPU status

The Coral NPU compiler plugin (`litert/vendors/coralnpu/compiler_plugin`) now
exports the full `LiteRtCompilerPlugin` ABI, not just
`LiteRtCompilerPluginQueryInterface`. Without those symbols the LiteRT loader
fails with `undefined symbol: LiteRtGetCompilerPluginVersion` (and then
`LiteRtCreateCompilerPlugin`) and silently runs on the CPU.

With the symbols exported, the loader resolves the plugin, applies it, and
partitions every subgraph:

```
Loaded plugin at: .../libLiteRtCompilerPlugin_CoralNpu.so
Resolved plugin api at: .../libLiteRtCompilerPlugin_CoralNpu.so
Partitioned subgraph<0>, selected 1537 ops ... resulted in 1 partitions
```

Compilation then needs the AOT driver, because the TOSA lowering, attention
tiling, and `iree-compile` live in `scripts/compile_litert.py`:

```bash
export CORALNPU_AOT_DRIVER=$PWD/scripts/compile_litert.py
export CORALNPU_PYTHON=/path/to/venv/bin/python   # has iree-base-compiler
export CORALNPU_AOT_SEQ_LEN=256
python benchmarks/bench_gemma3.py --backend litert-npu ...
```

On a host with `iree-base-compiler`, `iree-tools-tflite`, and the Coral NPU
toolchain, the plugin produces a VMFB and the dispatch library runs it on the
NPU. On the board itself the toolchain is absent, so the backend falls back to
CPU. `bench_gemma3.py` reports the requested backend in `meta`; check the
LiteRT-LM log for `0 compiler plugins were applied successfully` to detect the
fallback.

## Running the benchmark

Each backend needs its own Python environment:

```bash
# Torq NPU (torq-examples venv already has torq + iree.runtime)
LD_PRELOAD=.../libtorq_hwcompat.so \
  /home/root/torq-examples/.venv/bin/python benchmarks/bench_gemma3.py \
    --backend torq --model .../transformer.vmfb \
    --torq-device-allocator cpu --runs 3 --max-tokens 64

# LiteRT-LM CPU (venv with `pip install litert-lm`)
/home/root/litert-venv/bin/python benchmarks/bench_gemma3.py \
  --backend litert-cpu --model /home/root/models/gemma3-270m-it-q8.litertlm \
  --threads 2 --runs 3 --max-tokens 64 \
  --tokenizer .../gemma-3-270-GPTQ-int4-it-torq/tokenizer.json

# LiteRT-LM NPU (adds the vendor dispatch/compiler plugins)
/home/root/litert-venv/bin/python benchmarks/bench_gemma3.py \
  --backend litert-npu --model /home/root/models/gemma3-270m-it-q8.litertlm \
  --dispatch-lib-dir /data/local/tmp/boardtest/build \
  --threads 2 --runs 3 --max-tokens 64
```

`--json-out` writes raw samples plus a summary. Metrics:

* **TTFT** — submit to first non-empty streamed chunk (prefill + first token).
* **decode tok/s** — `(tokens - 1) / (total - TTFT)`; excludes prefill.
* **e2e tok/s** — `tokens / total`; includes prefill.

LiteRT-LM runs use a fresh conversation per sample so the KV cache is cold and
prefill is measured. Torq runs call `runner.reset()` between samples for the
same reason.
