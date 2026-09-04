#!/usr/bin/env python3
"""
GSM8K eval harness for the reasoning-efficiency sweep on Qwen3.8-27B GGUFs
(huihui abliterated or unsloth base) served by llama-server.

Measures for each (config, problem) pair:
  - correctness (vs GSM8K gold "#### N")
  - decode t/s (predicted_per_second)
  - think length (chars) - uses reasoning_content field when available
  - finish_reason

Usage:
    gsm8k_harness.py --label xhigh --reasoning-effort xhigh --runs 3
    gsm8k_harness.py --label nowait --logit-bias-file nowait_tokens.json
    gsm8k_harness.py --label cod --system-prompt "Think in 5-word drafts..."
    gsm8k_harness.py --label unsloth-cod --model-path /media/.../Q4_K_XL.gguf \\
                     --system-prompt "Think in 5-word drafts..."

The harness uses the chat completions endpoint with extra_body for llama-server
specifics (reasoning_effort, reasoning_budget, logit_bias).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

PORT = int(os.environ.get("LLAMA_PORT", "8091"))
BASE = f"http://127.0.0.1:{PORT}"
DEFAULT_MODEL = "$MODELS_DIR/models/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"


# ---------- answer extraction ----------

def gsm8k_gold(answer_field):
    """GSM8K answers end with '#### N'."""
    if "####" in answer_field:
        tail = answer_field.split("####")[-1].strip()
        return tail.replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer_field)
    return nums[-1].replace(",", "") if nums else ""


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    """Extract a final numeric answer from model output."""
    s = text.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", s, flags=re.DOTALL)
    m = re.search(r"\\boxed\s*\{\s*([^}]+?)\s*\}", s)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", s)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"(?:answer\s*(?:is|:)\s*|=\s*)(-?\d[\d,]*\.?\d*)", s, flags=re.IGNORECASE)
    if m:
        return _normalize(m.group(1))
    nums = _NUM_RE.findall(s[-200:])
    if nums:
        return _normalize(nums[-1])
    return ""


def _normalize(num):
    n = num.replace(",", "").rstrip(".")
    if n.endswith("."):
        n = n[:-1]
    return n


def is_correct(model_text, gold_field):
    pred = extract_answer(model_text)
    gold = gsm8k_gold(gold_field)
    try:
        ok = abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        ok = pred == gold
    return ok, pred, gold


# ---------- HTTP ----------

def post_chat(messages, *, max_tokens=1024, temperature=0.0, stop=None,
              reasoning_effort=None, reasoning_budget=None, logit_bias=None,
              model_path=None):
    body = {
        "model": model_path or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if stop is not None:
        body["stop"] = stop
    extra = {}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    if reasoning_budget is not None:
        extra["reasoning_budget"] = reasoning_budget
    if logit_bias is not None:
        extra["logit_bias"] = logit_bias
    if extra:
        body["extra_body"] = extra
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


# ---------- problem loading ----------

def load_problems(path, n):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"question": obj["question"], "answer": obj["answer"]})
            if len(items) >= n:
                break
    return items


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/reasoning-sweep/data/gsm8k_10.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="/tmp/reasoning-sweep/results.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--reasoning-budget", type=int, default=None)
    ap.add_argument("--logit-bias-file", default=None)
    ap.add_argument("--model-path", default=None,
                    help="override the model field sent to the chat-completions API "
                         "(defaults to huihui abliterated UD-Q4_K_XL)")
    ap.add_argument("--port", type=int, default=None,
                    help="override llama-server port (env LLAMA_PORT also works)")
    args = ap.parse_args()

    # Allow --port to override the module-level BASE
    global BASE
    if args.port is not None:
        BASE = f"http://127.0.0.1:{args.port}"

    probs = load_problems(args.data, args.n)
    print(f"[{args.label}] loaded {len(probs)} problems x {args.runs} runs on port {args.port or PORT}", flush=True)

    logit_bias = None
    if args.logit_bias_file:
        with open(args.logit_bias_file) as f:
            logit_bias = json.load(f)
        print(f"[{args.label}] logit_bias has {len(logit_bias)} entries", flush=True)

    if args.model_path:
        print(f"[{args.label}] model: {args.model_path}", flush=True)

    out = open(args.out, "a")
    for run_idx in range(args.runs):
        correct = 0
        total_decode_ms = 0
        total_completion_tokens = 0
        think_lens = []
        think_tokens_total = 0

        for i, p in enumerate(probs):
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": p["question"]})

            t0 = time.time()
            try:
                resp = post_chat(
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_budget=args.reasoning_budget,
                    logit_bias=logit_bias,
                    model_path=args.model_path,
                )
            except Exception as e:
                print(f"  [{args.label}] q{i} ERROR: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0

            choice = resp["choices"][0]
            text = choice["message"]["content"]
            timings = resp.get("timings", {})
            usage = resp.get("usage", {})

            ok, pred, gold = is_correct(text, p["answer"])
            if ok:
                correct += 1

            total_decode_ms += timings.get("predicted_ms", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            # Think length: prefer reasoning_content field
            rc = choice["message"].get("reasoning_content")
            if rc is not None:
                think_len = len(rc)
                think_tokens = usage.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                )
            else:
                think_len = 0
                think_tokens = 0
                m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
                if m:
                    think_len = len(m.group(1))
            think_lens.append(think_len)
            think_tokens_total += think_tokens

            record = {
                "label": args.label,
                "run": run_idx,
                "q": i,
                "wall_s": round(wall, 2),
                "correct": ok,
                "pred": pred,
                "gold": gold,
                "model_path": args.model_path,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "predicted_ms": timings.get("predicted_ms", 0),
                "prompt_ms": timings.get("prompt_ms", 0),
                "predicted_per_s": timings.get("predicted_per_second"),
                "prompt_per_s": timings.get("prompt_per_second"),
                "think_chars": think_len,
                "think_tokens": think_tokens,
                "finish_reason": choice.get("finish_reason"),
            }
            out.write(json.dumps(record) + "\n")
            print(f"  [{args.label}] run{run_idx} q{i} {'OK' if ok else 'XX'} "
                  f"pred={pred} gold={gold} "
                  f"gen={usage.get('completion_tokens',0)}t "
                  f"@ {timings.get('predicted_per_second',0):.1f}t/s "
                  f"think_chars={think_len} "
                  f"fr={choice.get('finish_reason')}", flush=True)

        denom = max(1e-6, total_decode_ms / 1000)
        avg_decode_tps = total_completion_tokens / denom
        run_acc = correct / len(probs) if probs else 0
        summary = {
            "label": args.label,
            "run": run_idx,
            "accuracy": round(run_acc, 3),
            "mean_decode_tps": round(avg_decode_tps, 2),
            "mean_think_chars": int(sum(think_lens) / max(1, len(think_lens))),
            "total_think_tokens": think_tokens_total,
            "total_completion_tokens": total_completion_tokens,
            "model_path": args.model_path,
        }
        out.write(json.dumps({"__summary__": summary}) + "\n")
        out.flush()
        print(f"[{args.label}] run{run_idx} SUMMARY: acc={run_acc:.2f} "
              f"@ {avg_decode_tps:.2f} t/s mean_think_chars={summary['mean_think_chars']} "
              f"total_think_tokens={think_tokens_total}", flush=True)

    out.close()


if __name__ == "__main__":
    main()