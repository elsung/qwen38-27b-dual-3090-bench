#!/usr/bin/env python3
"""
Detailed throughput matrix benchmark.

Measures prefill + decode speeds across:
  - prompt sizes:    small (~64t), medium (~1k t), large (~8k t), xlarge (~32k t)
  - max_tokens gen:  short (128), long (1024)
  - concurrency:     1, 2, 4

Writes one JSONL record per (model, config) cell with:
  - prompt_tokens, completion_tokens
  - prompt_per_second, predicted_per_second (decode t/s)
  - wall_s, concurrency
  - finish_reason

Usage:
  matrix_bench.py --label huihui --model-path /path/to.gguf \
    --host http://127.0.0.1:8091 --port 8091 \
    --prompt-sizes 64,1024,8192,32768 \
    --max-tokens-list 128,1024 \
    --concurrencies 1,2,4 \
    --runs 3 --out matrix_huihui.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:8091"
_DEFAULT_MODEL_CACHE = {}


def _resolve_default_model(host):
    """Probe /v1/models once and cache the first model id."""
    if host in _DEFAULT_MODEL_CACHE:
        return _DEFAULT_MODEL_CACHE[host]
    try:
        with urllib.request.urlopen(f"{host}/v1/models", timeout=10) as r:
            data = json.loads(r.read())
        mid = data.get("data", [{}])[0].get("id")
    except Exception:
        mid = None
    _DEFAULT_MODEL_CACHE[host] = mid
    return mid


def post_chat(host, messages, *, max_tokens=1024, temperature=0.6, model_path=None,
              timeout=600):
    """Single chat completion. Returns parsed JSON or raises."""
    body = {
        "model": model_path or _resolve_default_model(host) or "x",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build_prompt(target_tokens):
    """Build a synthetic prompt of approximately target_tokens tokens.

    Uses a single repeated sentence and binary search-ish sizing via repetition.
    Actual token count depends on the model's tokenizer; the harness records
    the *actual* prompt_tokens from the server response.
    """
    # Qwen3 tokenizer averages ~4 chars/token for English; pad with a fixed
    # marker so the server can identify it for debugging.
    sentence = ("The quick brown fox jumps over the lazy dog. " * 50).strip()
    chars_per_token = 4.0
    target_chars = int(target_tokens * chars_per_token)
    n = max(1, target_chars // len(sentence))
    body = ("Benchmark payload. " + (sentence + " ") * n)[:target_chars]
    return body


def run_one(host, model_path, prompt, *, max_tokens, temperature=0.6, label=""):
    """Run a single chat completion, return timings dict."""
    t0 = time.time()
    try:
        resp = post_chat(host, [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature,
                         model_path=model_path)
    except Exception as e:
        return {"label": label, "error": str(e)}
    wall = time.time() - t0
    choice = resp["choices"][0]
    timings = resp.get("timings", {})
    usage = resp.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    # llama-server returns timings.{prompt_per_second,predicted_per_second};
    # vLLM does not, so compute from wall time as a fallback (decode t/s only,
    # since we don't have separate prefill timing).
    prefill_tps = timings.get("prompt_per_second", 0)
    decode_tps = timings.get("predicted_per_second", 0)
    if (not decode_tps or decode_tps == 0) and wall > 0.001 and ct > 0:
        decode_tps = ct / wall
    return {
        "label": label,
        "wall_s": round(wall, 3),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "prompt_per_s": round(prefill_tps, 1),
        "predicted_per_s": round(decode_tps, 1),
        "prompt_ms": timings.get("prompt_ms", 0),
        "predicted_ms": timings.get("predicted_ms", 0),
        "finish_reason": choice.get("finish_reason"),
    }


def run_concurrent(host, model_path, prompt, *, max_tokens, n, temperature=0.6, label=""):
    """Run n concurrent chat completions. Returns aggregate stats."""
    results = [None] * n
    barrier = threading.Barrier(n)
    def worker(i):
        barrier.wait()
        results[i] = run_one(host, model_path, prompt,
                             max_tokens=max_tokens, temperature=temperature,
                             label=f"{label}_c{i}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    # aggregate
    ok = [r for r in results if r and "error" not in r]
    if not ok:
        return {"label": label, "concurrency": n, "error": "all requests failed"}
    return {
        "label": label,
        "concurrency": n,
        "n_ok": len(ok),
        "n_total": n,
        "aggregate_decode_tps": round(sum(r["predicted_per_s"] for r in ok), 1),
        "aggregate_prompt_tps": round(sum(r["prompt_per_s"] for r in ok), 1),
        "mean_decode_tps": round(statistics.mean(r["predicted_per_s"] for r in ok), 1),
        "mean_prompt_tps": round(statistics.mean(r["prompt_per_s"] for r in ok), 1),
        "mean_wall_s": round(statistics.mean(r["wall_s"] for r in ok), 2),
        "p50_decode_tps": round(statistics.median(r["predicted_per_s"] for r in ok), 1),
        "samples": [r["predicted_per_s"] for r in ok],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-path", default=None,
                    help="override model field sent to API")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--prompt-sizes", default="64,1024,8192,32768",
                    help="comma-separated target token counts")
    ap.add_argument("--max-tokens-list", default="128,1024",
                    help="comma-separated max_tokens values")
    ap.add_argument("--concurrencies", default="1,2,4",
                    help="comma-separated concurrency levels")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-concurrent", action="store_true",
                    help="also run concurrent cells (slower)")
    args = ap.parse_args()

    prompt_sizes = [int(x) for x in args.prompt_sizes.split(",")]
    max_tokens_list = [int(x) for x in args.max_tokens_list.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    out = open(args.out, "a")
    print(f"[{args.label}] matrix benchmark starting", flush=True)
    print(f"  host={args.host}  prompt_sizes={prompt_sizes}  "
          f"max_tokens={max_tokens_list}  concurrencies={concurrencies}  "
          f"runs={args.runs}  concurrent={args.include_concurrent}",
          flush=True)

    for ps in prompt_sizes:
        prompt = build_prompt(ps)
        for mt in max_tokens_list:
            for run in range(args.runs):
                r = run_one(args.host, args.model_path, prompt,
                             max_tokens=mt, temperature=args.temperature,
                             label=f"{args.label}_p{ps}_m{mt}_r{run}")
                if "error" in r:
                    print(f"  ERROR p{ps} m{mt} r{run}: {r['error']}", file=sys.stderr)
                else:
                    print(f"  p{ps:6d}t m{mt:5d}  r{run}: "
                          f"prefill={r['prompt_per_s']:7.1f} t/s  "
                          f"decode={r['predicted_per_s']:6.1f} t/s  "
                          f"({r['completion_tokens']}t gen, {r['wall_s']:.1f}s)",
                          flush=True)
                out.write(json.dumps({
                    "label": args.label,
                    "prompt_target": ps,
                    "max_tokens": mt,
                    "run": run,
                    "concurrency": 1,
                    **{k: v for k, v in r.items() if k != "label"},
                }) + "\n")
            out.flush()

    if args.include_concurrent:
        # Only run concurrent cells at one prompt size (medium) and one max_tokens (short)
        # to avoid combinatorial blowup
        ps = 1024
        mt = 128
        prompt = build_prompt(ps)
        for n in concurrencies:
            if n <= 1: continue
            for run in range(args.runs):
                agg = run_concurrent(args.host, args.model_path, prompt,
                                     max_tokens=mt, n=n,
                                     temperature=args.temperature,
                                     label=f"{args.label}_p{ps}_m{mt}_c{n}_r{run}")
                if "error" in agg:
                    print(f"  c{n} r{run}: {agg['error']}", file=sys.stderr)
                else:
                    print(f"  c{n}  r{run}: aggregate_decode={agg['aggregate_decode_tps']:.1f} t/s  "
                          f"mean_decode={agg['mean_decode_tps']:.1f} t/s  "
                          f"mean_wall={agg['mean_wall_s']:.2f}s",
                          flush=True)
                out.write(json.dumps({
                    "label": args.label,
                    "prompt_target": ps,
                    "max_tokens": mt,
                    "run": run,
                    "concurrency": n,
                    **{k: v for k, v in agg.items() if k != "label"},
                }) + "\n")
            out.flush()

    out.close()
    print(f"[{args.label}] matrix benchmark complete. Results in {args.out}", flush=True)


if __name__ == "__main__":
    main()