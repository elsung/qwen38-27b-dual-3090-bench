#!/usr/bin/env python3
"""
Combined throughput matrix + GSM8K accuracy benchmark.

Same matrix as matrix_bench.py (prompt sizes x max_tokens x concurrency),
PLUS a GSM8K accuracy run with the CoD system prompt.

Usage:
  matrix_and_accuracy.py --label huihui --model-path /path/to.gguf \
    --accuracy-data bench/gsm8k_10.jsonl --host-r /runs 3 \
    --out matrix_with_acc.jsonl
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

# Reuse the prompt builder from matrix_bench
sys.path.insert(0, '.')
from matrix_bench import post_chat, build_prompt, run_one, run_concurrent

# GSM8K answer extraction (same logic as gsm8k_harness.py)
def gsm8k_gold(answer_field):
    if "####" in answer_field:
        tail = answer_field.split("####")[-1].strip()
        return tail.replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer_field)
    return nums[-1].replace(",", "") if nums else ""

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def extract_answer(text):
    s = text.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    m = re.search(r"\\boxed\s*\{\s*([^}]+?)\s*\}", s)
    if m: return _normalize(m.group(1))
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", s)
    if m: return _normalize(m.group(1))
    m = re.search(r"(?:answer\s*(?:is|:)\s*|=\s*)(-?\d[\d,]*\.?\d*)", s, flags=re.IGNORECASE)
    if m: return _normalize(m.group(1))
    nums = _NUM_RE.findall(s[-200:])
    return _normalize(nums[-1]) if nums else ""

def _normalize(num):
    return num.replace(",", "").rstrip(".").rstrip(".")

def is_correct(model_text, gold_field):
    pred = extract_answer(model_text)
    gold = gsm8k_gold(gold_field)
    try: return abs(float(pred) - float(gold)) < 1e-6, pred, gold
    except ValueError: return pred == gold, pred, gold


def run_gsm8k(host, model_path, problems, *, runs, max_tokens=1024,
              temperature=0.6, system_prompt=None):
    """Run each GSM8K problem `runs` times, return accuracy stats."""
    n = len(problems)
    n_total = n * runs
    correct = 0
    total_decode_ms = 0
    total_prompt_ms = 0
    total_completion = 0
    total_prompt = 0
    cells = []
    for run_idx in range(runs):
        for i, p in enumerate(problems):
            messages = []
            if system_prompt:
                messages.append({"role":"system","content": system_prompt})
            messages.append({"role":"user","content": p["question"]})
            try:
                resp = post_chat(host, messages, max_tokens=max_tokens,
                                  temperature=temperature, model_path=model_path,
                                  timeout=600)
            except Exception as e:
                cells.append({"run": run_idx, "q": i, "error": str(e)})
                continue
            choice = resp["choices"][0]
            text = choice["message"]["content"]
            timings = resp.get("timings", {})
            usage = resp.get("usage", {})
            ok, pred, gold = is_correct(text, p["answer"])
            if ok: correct += 1
            total_decode_ms += timings.get("predicted_ms", 0)
            total_prompt_ms += timings.get("prompt_ms", 0)
            total_completion += usage.get("completion_tokens", 0)
            total_prompt += usage.get("prompt_tokens", 0)
            cells.append({"run": run_idx, "q": i, "correct": ok, "pred": pred,
                          "gold": gold, "decode_tps": round(timings.get("predicted_per_second",0),1),
                          "completion_tokens": usage.get("completion_tokens",0),
                          "prompt_tokens": usage.get("prompt_tokens",0)})
    denom = max(1e-6, total_decode_ms/1000)
    avg_tps = total_completion/denom
    return {
        "accuracy": round(correct/n_total, 3),
        "mean_decode_tps": round(avg_tps, 2),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "n_correct": correct,
        "n_total": n_total,
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--host", default="http://127.0.0.1:8091")
    ap.add_argument("--prompt-sizes", default="64,1024,8192,32768")
    ap.add_argument("--max-tokens-list", default="128,1024")
    ap.add_argument("--concurrencies", default="1,2,4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--include-concurrent", action="store_true")
    ap.add_argument("--accuracy-data", default=None,
                    help="path to gsm8k_*.jsonl for accuracy test")
    ap.add_argument("--accuracy-runs", type=int, default=3)
    ap.add_argument("--accuracy-max-tokens", type=int, default=1024)
    ap.add_argument("--system-prompt", default=None,
                    help="e.g. the CoD prompt for accuracy testing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = open(args.out, "a")
    print(f"[{args.label}] combined bench + accuracy starting", flush=True)

    # === Phase 1: throughput matrix (same as matrix_bench.py) ===
    prompt_sizes = [int(x) for x in args.prompt_sizes.split(",")]
    max_tokens_list = [int(x) for x in args.max_tokens_list.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    print(f"[{args.label}] phase 1: throughput matrix", flush=True)
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
                          f"decode={r['predicted_per_s']:6.1f} t/s",
                          flush=True)
                out.write(json.dumps({
                    "type": "speed",
                    "label": args.label,
                    "prompt_target": ps,
                    "max_tokens": mt,
                    "run": run,
                    "concurrency": 1,
                    **{k: v for k, v in r.items() if k != "label"},
                }) + "\n")
            out.flush()

    if args.include_concurrent:
        ps = 1024
        mt = 128
        prompt = build_prompt(ps)
        for n in concurrencies:
            if n <= 1: continue
            for run in range(args.runs):
                agg = run_concurrent(args.host, args.model_path, prompt,
                                     max_tokens=mt, n=n, temperature=args.temperature,
                                     label=f"{args.label}_p{ps}_m{mt}_c{n}_r{run}")
                if "error" in agg:
                    print(f"  c{n} r{run}: {agg['error']}", file=sys.stderr)
                else:
                    print(f"  c{n}  r{run}: agg_decode={agg['aggregate_decode_tps']:.1f} t/s  "
                          f"mean={agg['mean_decode_tps']:.1f} t/s  "
                          f"wall={agg['mean_wall_s']:.2f}s",
                          flush=True)
                out.write(json.dumps({
                    "type": "speed",
                    "label": args.label,
                    "prompt_target": ps,
                    "max_tokens": mt,
                    "run": run,
                    "concurrency": n,
                    **{k: v for k, v in agg.items() if k != "label"},
                }) + "\n")
            out.flush()

    # === Phase 2: accuracy (if --accuracy-data given) ===
    if args.accuracy_data:
        problems = []
        with open(args.accuracy_data) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                problems.append({"question": obj["question"], "answer": obj["answer"]})
                if len(problems) >= 10: break
        print(f"[{args.label}] phase 2: GSM8K accuracy x {args.accuracy_runs}", flush=True)
        sysprompt = args.system_prompt or "Think step by step, but write each step in at most 5 words. Be extremely concise."
        result = run_gsm8k(args.host, args.model_path, problems,
                            runs=args.accuracy_runs,
                            max_tokens=args.accuracy_max_tokens,
                            temperature=args.temperature,
                            system_prompt=sysprompt)
        print(f"[{args.label}] GSM8K accuracy: {result['accuracy']:.2f} "
              f"({result['n_correct']}/{result['n_total']} correct, "
              f"avg decode={result['mean_decode_tps']:.1f} t/s)", flush=True)
        out.write(json.dumps({
            "type": "accuracy",
            "label": args.label,
            "system_prompt": sysprompt,
            "accuracy": result["accuracy"],
            "n_correct": result["n_correct"],
            "n_total": result["n_total"],
            "mean_decode_tps": result["mean_decode_tps"],
            "total_prompt_tokens": result["total_prompt_tokens"],
            "total_completion_tokens": result["total_completion_tokens"],
            "cells": result["cells"],
        }) + "\n")
        out.flush()

    out.close()
    print(f"[{args.label}] combined benchmark complete. Results in {args.out}", flush=True)


if __name__ == "__main__":
    main()