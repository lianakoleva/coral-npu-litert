#!/usr/bin/env python3
"""Benchmark Gemma 3 270M on the Coralboard: Torq NPU vs LiteRT-LM.

Two very different ways to execute the same 270M model on an Astra board:

* ``torq`` — a pre-compiled Torq VMFB executed by the ``torq`` IREE device.
  Weights stream from DDR through the NPU, so no compiler runs on the board.
* ``litert-cpu`` — a ``.litertlm`` bundle through LiteRT-LM on the CPU
  (XNNPACK).  This is the baseline the assistant used before the NPU path.
* ``litert-npu`` — the same bundle through LiteRT-LM with a vendor dispatch
  library.  LiteRT-LM only offloads subgraphs the compiler plugin can
  partition; if the plugin is missing or fails, it runs on the CPU.

Run each backend with the Python environment that has its runtime::

    # NPU (torq-examples venv, which already has torq + iree.runtime)
    /home/root/torq-examples/.venv/bin/python benchmarks/bench_gemma3.py \\
      --backend torq \\
      --model /home/root/torq-examples/models/Synaptics/gemma-3-270-GPTQ-int4-it-torq/transformer.vmfb

    # LiteRT-LM (a venv with ``pip install litert-lm``)
    /home/root/litert-venv/bin/python benchmarks/bench_gemma3.py \\
      --backend litert-cpu --model /home/root/models/gemma3-270m-it-q8.litertlm

Metrics per run:

* **TTFT** — wall time from submit to the first non-empty streamed chunk
  (prompt prefill plus the first decoded token).
* **decode tok/s** — ``(tokens - 1) / (total - TTFT)``; excludes prefill.
* **e2e tok/s** — ``tokens / total``; includes prefill.

Results are printed as a table and can be written as JSON with ``--json-out``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _message_text(message: Any) -> str:
    """Extract plain text from a LiteRT-LM streamed Message."""
    data = message.to_json() if hasattr(message, "to_json") else dict(message)
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _load_tokenizer(path: str | Path | None):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415

        return Tokenizer.from_file(str(path))
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"warning: could not load tokenizer {path}: {exc}", file=sys.stderr)
        return None


def _count_tokens(text: str, tokenizer, fallback: int | None = None) -> int:
    if fallback:
        return fallback
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    return len(text.split())


def _sample(
    ttft_s: float | None,
    total_s: float,
    tokens: int,
    text: str,
    tokenizer,
) -> dict[str, Any]:
    tokens = _count_tokens(text, tokenizer, fallback=tokens)
    ttft = ttft_s if ttft_s is not None else total_s
    decode_s = max(total_s - ttft, 1e-9)
    return {
        "ttft_ms": round(ttft * 1e3, 1),
        "total_ms": round(total_s * 1e3, 1),
        "tokens": tokens,
        "decode_tok_s": round(max(tokens - 1, 0) / decode_s, 2),
        "e2e_tok_s": round(tokens / max(total_s, 1e-9), 2),
        "text": text.strip(),
    }


def _run_torq(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = Path(args.torq_examples_dir).expanduser().resolve()
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))

    from gemma3.src.runner import Gemma3Static  # noqa: PLC0415

    started = time.perf_counter()
    runner = Gemma3Static(
        args.model,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_tokens,
        instruct_model=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        runtime_flags=[f"--torq_device_allocator={args.torq_device_allocator}"],
        device_io=False,
    )
    load_s = time.perf_counter() - started

    tokenizer = _load_tokenizer(args.tokenizer or Path(args.model).parent / "tokenizer.json")
    print(
        f"torq: max_seq_len={runner.max_seq_len} "
        f"device_allocator={args.torq_device_allocator} load={load_s:.1f}s",
        flush=True,
    )

    for _ in range(args.warmup):
        runner.run(args.prompt)
    runner.reset()

    samples: list[dict[str, Any]] = []
    for index in range(args.runs):
        runner.reset()
        started = time.perf_counter()
        ttft_s: float | None = None
        parts: list[str] = []
        for chunk in runner.run_stream(args.prompt):
            if ttft_s is None and chunk:
                ttft_s = time.perf_counter() - started
            parts.append(chunk)
        total_s = time.perf_counter() - started
        sample = _sample(
            ttft_s, total_s, runner.generated_tokens, "".join(parts), tokenizer
        )
        samples.append(sample)
        print(
            f"  run {index + 1}: ttft={sample['ttft_ms']:.0f} ms "
            f"decode={sample['decode_tok_s']:.2f} tok/s "
            f"e2e={sample['e2e_tok_s']:.2f} tok/s "
            f"tokens={sample['tokens']}",
            flush=True,
        )
    return samples, {"load_s": round(load_s, 3), "max_seq_len": runner.max_seq_len}


def _run_litert(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import litert_lm  # noqa: PLC0415

    if args.backend == "litert-npu":
        if not args.dispatch_lib_dir:
            raise SystemExit("--dispatch-lib-dir is required for --backend litert-npu")
        backend = litert_lm.Backend.NPU(
            litert_dispatch_lib_dir=str(Path(args.dispatch_lib_dir).expanduser())
        )
        backend_label = "npu"
    else:
        backend = litert_lm.Backend.CPU(thread_count=args.threads)
        backend_label = "cpu"

    started = time.perf_counter()
    engine = litert_lm.Engine(args.model, backend=backend)
    load_s = time.perf_counter() - started

    tokenizer = _load_tokenizer(args.tokenizer)
    print(
        f"litert-lm: requested={backend_label} load={load_s:.1f}s "
        f"threads={args.threads}",
        flush=True,
    )

    sampler = litert_lm.SamplerConfig(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )

    def one_run() -> dict[str, Any]:
        # A fresh conversation per run measures prefill instead of reusing a
        # warm KV cache.
        conversation = engine.create_conversation(
            sampler_config=sampler, max_output_tokens=args.max_tokens
        )
        started = time.perf_counter()
        ttft_s: float | None = None
        parts: list[str] = []
        with conversation:
            for message in conversation.send_message_async(args.prompt):
                text = _message_text(message)
                if ttft_s is None and text:
                    ttft_s = time.perf_counter() - started
                parts.append(text)
        total_s = time.perf_counter() - started
        return _sample(ttft_s, total_s, 0, "".join(parts), tokenizer)

    for _ in range(args.warmup):
        one_run()

    samples: list[dict[str, Any]] = []
    for index in range(args.runs):
        sample = one_run()
        samples.append(sample)
        print(
            f"  run {index + 1}: ttft={sample['ttft_ms']:.0f} ms "
            f"decode={sample['decode_tok_s']:.2f} tok/s "
            f"e2e={sample['e2e_tok_s']:.2f} tok/s "
            f"tokens={sample['tokens']}",
            flush=True,
        )
    return samples, {"load_s": round(load_s, 3), "requested_backend": backend_label}


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    return {
        "ttft_ms_median": round(statistics.median(s["ttft_ms"] for s in samples), 1),
        "decode_tok_s_median": round(
            statistics.median(s["decode_tok_s"] for s in samples), 2
        ),
        "e2e_tok_s_median": round(
            statistics.median(s["e2e_tok_s"] for s in samples), 2
        ),
        "tokens_median": int(statistics.median(s["tokens"] for s in samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 3 270M on the Coralboard NPU vs LiteRT-LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("torq", "litert-cpu", "litert-npu"),
        default="torq",
    )
    parser.add_argument("--model", required=True, help="transformer.vmfb or .litertlm")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in three sentences why a small language model can run on "
            "an NPU even though its weights are much larger than the NPU's "
            "local memory."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--tokenizer",
        help="tokenizer.json used to count LiteRT output tokens "
        "(defaults to the Torq model's tokenizer for --backend torq)",
    )
    parser.add_argument("--json-out", help="write raw samples and summary as JSON")
    parser.add_argument("--label", help="label for this run in the JSON output")

    torq = parser.add_argument_group("torq backend")
    torq.add_argument(
        "--torq-examples-dir",
        default="/home/root/torq-examples",
        help="checkout providing gemma3/src/runner.py",
    )
    torq.add_argument(
        "--torq-device-allocator",
        default="cpu",
        help="Torq device allocator (cpu works with older kernel drivers)",
    )
    torq.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="override the model's KV cache length",
    )

    litert = parser.add_argument_group("LiteRT-LM backend")
    litert.add_argument("--threads", type=int, default=2)
    litert.add_argument(
        "--dispatch-lib-dir",
        help="directory with libLiteRtDispatch_*.so (litert-npu)",
    )

    args = parser.parse_args()
    if args.backend == "torq":
        samples, meta = _run_torq(args)
    else:
        samples, meta = _run_litert(args)

    summary = _summarize(samples)
    result = {
        "label": args.label or args.backend,
        "backend": args.backend,
        "model": str(Path(args.model).expanduser()),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "meta": meta,
        "samples": samples,
        "summary": summary,
    }

    print()
    print(
        f"{result['label']}: "
        f"TTFT {summary['ttft_ms_median']:.0f} ms, "
        f"decode {summary['decode_tok_s_median']:.2f} tok/s, "
        f"e2e {summary['e2e_tok_s_median']:.2f} tok/s "
        f"({summary['tokens_median']} tokens)"
    )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
