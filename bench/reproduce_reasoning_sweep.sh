#!/usr/bin/env bash
#
# Reproduce the reasoning-effort sweep from REASONING_EFFORT_SWEEP_2026-09-04.md.
#
# Prerequisites:
#   - huihui Qwen3.8-27B abliterated GGUF launched on port 8091 (start-huihui-27b-abliterated.sh)
#   - python3 in PATH; no extra packages needed (urllib only)
#
# Outputs:
#   - results.jsonl (appended to; safe to delete first to start fresh)
#   - log_<config>.txt per config
#
# Total runtime: ~50 min on dual RTX 3090 / Ryzen 7 5800XT
set -euo pipefail

cd "$(dirname "$0")"
DATA="${DATA:-gsm8k_10.jsonl}"
N="${N:-10}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

# Re-derive the NoWait token list if it's missing (needs the llama-server on 8091)
if [ ! -f nowait_tokens.json ]; then
    echo "[setup] deriving nowait_tokens.json from llama-server tokenizer"
    python3 - <<'PY'
import json, urllib.request, sys
KEYWORDS = ["wait","alternatively","hmm","but","however","alternative",
            "another","check","double-check","oh","maybe","verify","other",
            "again","now","ah","any"]
def tokenize(t):
    r = urllib.request.Request("http://127.0.0.1:8091/tokenize",
        data=json.dumps({"content":t}).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r,timeout=30).read())["tokens"]
def detok(ids):
    r = urllib.request.Request("http://127.0.0.1:8091/detokenize",
        data=json.dumps({"tokens":ids}).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r,timeout=30).read())["content"]

VARIANTS = set()
for kw in KEYWORDS:
    for w in ["", " "]:
        for cased in [kw, kw.capitalize(), kw.upper()]:
            for s in ["",".",",","!","?",";",":"," —"," -","\n"," "]:
                VARIANTS.add(f"{w}{cased}{s}")

candidates = {}
for v in VARIANTS:
    for tid in tokenize(v):
        candidates.setdefault(tid, set()).add(v)

def first_word_ok(text):
    t = text.strip().lower().lstrip(".,!?:;—-")
    return any(t.startswith(kw+" ") or t==kw or t.startswith(kw+".") or
               t.startswith(kw+",") or t.startswith(kw+"!") or
               t.startswith(kw+"?") or t.startswith(kw+";") or
               t.startswith(kw+":") for kw in KEYWORDS)

keep = {}
for tid in candidates:
    try: dec = detok([tid])
    except: continue
    if first_word_ok(dec):
        keep[tid] = -100.0
for kw in KEYWORDS:
    for tid in tokenize(kw):
        try: dec = detok([tid]).strip().lower()
        except: continue
        if dec == kw:
            keep[tid] = -100.0

with open("nowait_tokens.json","w") as f:
    json.dump({str(k):v for k,v in keep.items()}, f, indent=2)
print(f"[setup] wrote nowait_tokens.json ({len(keep)} tokens)", file=sys.stderr)
PY
fi

# Run all configs in sequence. Use a single results.jsonl; truncate first if requested.
if [ "${RESET:-0}" = "1" ] && [ -f results.jsonl ]; then
    echo "[reset] truncating results.jsonl"
    : > results.jsonl
fi

run() {
    local label="$1"; shift
    echo "==================== ${label} ===================="
    python3 gsm8k_harness.py --label "${label}" --max-tokens "${MAX_TOKENS}" \
        --runs "${RUNS}" --n "${N}" --data "${DATA}" \
        --out results.jsonl "$@" 2>&1 | tee "log_${label}.txt"
}

# temperature=0 sweeps (no observable effect expected, but documented)
run xhigh       --reasoning-effort xhigh
run medium      --reasoning-effort medium
run low         --reasoning-effort low

# reasoning-budget sweep (no effect expected on this template)
for B in 0 256 1024 4096 16384; do
    run "budget-${B}" --reasoning-budget "${B}"
done

# NoWait at T=0 (null result expected)
run nowait --logit-bias-file nowait_tokens.json

# temperature=0.6 sweeps (real signal expected)
run baseline-t06 --temperature 0.6
run nowait-t06   --temperature 0.6 --logit-bias-file nowait_tokens.json
run cod          --temperature 0.6 --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise."
run tokbudget    --temperature 0.6 --system-prompt "Think step by step using at most 200 tokens of reasoning, then give the final answer."

echo "==================== DONE ===================="
echo "Aggregate with: python3 -c 'see REASONING_EFFORT_SWEEP_2026-09-04.md'"