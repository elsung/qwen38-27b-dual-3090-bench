#!/usr/bin/env bash
# 3-way comparison: huihui abliterated UD-Q4_K_XL (Unsloth Dynamic v2.x)
# vs unsloth Qwen3.8-27B UD-Q4_K_XL v3.0 vs unsloth UD-Q4_K_M v3.0,
# all on the same GSM8K 10 problems, all with CoD prompt at T=0.6.
#
# Run order:
#   1. huihui baseline (no CoD) - confirms prior sweep number
#   2. huihui + CoD            - confirms CoD works on huihui at T=0.6
#   3. unsloth UD-Q4_K_XL + CoD - the v3.0 same-size competitor
#   4. unsloth UD-Q4_K_M  + CoD - the v3.0 smaller competitor
# After each, swap llama-server. ~25 min total (each bench is ~5 min incl load).
#
# All output appended to compare_3way.jsonl. Each summary line is one record.
set -euo pipefail

cd "$(dirname "$0")"
HARNESS="gsm8k_harness.py"
DATA="gsm8k_10.jsonl"
RESULTS="compare_3way.jsonl"
N=10
RUNS=3
MAX_TOKENS=4096
TEMP=0.6
COD_PROMPT="Think step by step, but write each step in at most 5 words. Be extremely concise."

HUIHUI_PATH="$MODELS_DIR/models/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"
UNSLOTH_XL="$MODELS_DIR/models/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"
UNSLOTH_M="$MODELS_DIR/models/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"

# Reset results file
: > "$RESULTS"

wait_for_server() {
    # Wait until /health returns {"status":"ok"} AND a models query returns
    # a model path matching what we expect. Avoids the trap where the old
    # server's lingering port replies briefly before the new one binds.
    local label="$1"
    local expected_model_substring="$2"  # match against model.id returned by /v1/models
    local i
    for i in $(seq 1 90); do
      sleep 2
      if curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1; then
        local model=$(curl -sf http://127.0.0.1:8091/v1/models 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
        if [ -n "$model" ] && [[ "$model" == *"$expected_model_substring"* ]]; then
          echo "  server up after ${i}*2s with model: $model"
          return 0
        fi
      fi
    done
    echo "  TIMEOUT waiting for server with model '$expected_model_substring'"
    return 1
}

bench() {
    local label="$1" launcher="$2" expected_substr="$3"; shift 3
    echo "=========================================="
    echo "  $label  (model expected: $expected_substr)"
    echo "=========================================="
    bash "$launcher" > /tmp/launcher_out.txt 2>&1
    # The launcher script backgrounds llama-server with nohup and exits; the
    # wait_for_server() polls for the new server to actually be serving.
    wait_for_server "$label" "$expected_substr"
    # Bench
    python3 "$HARNESS" \
        --label "$label" --n "$N" --runs "$RUNS" \
        --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
        --out "$RESULTS" --data "$DATA" \
        "$@" 2>&1 | tee "log_${label}.txt"
    # Kill server
    for PF in /tmp/llama-huihui.pid /tmp/llama-unsloth.pid; do
      if [ -f "$PF" ]; then
        OPID=$(cat "$PF" 2>/dev/null || echo "")
        if [ -n "$OPID" ] && kill -0 "$OPID" 2>/dev/null; then
          kill "$OPID" 2>/dev/null || true
        fi
        rm -f "$PF"
      fi
    done
    # Wait for VRAM to release + port to free
    sleep 8
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
}

# 1. huihui baseline (no CoD)
bench "huihui-baseline" \
    $DESKTOP/start-huihui-27b-abliterated.sh \
    "Huihui-Qwen3.8-27B-abliterated" \
    --model-path "$HUIHUI_PATH"

# 2. huihui + CoD
bench "huihui-cod" \
    $DESKTOP/start-huihui-27b-abliterated.sh \
    "Huihui-Qwen3.8-27B-abliterated" \
    --model-path "$HUIHUI_PATH" \
    --system-prompt "$COD_PROMPT"

# 3. unsloth UD-Q4_K_XL v3.0 + CoD
bench "unsloth-xl-cod" \
    $DESKTOP/start-unsloth-qwen38-27b.sh \
    "Q4_K_XL" \
    --model-path "$UNSLOTH_XL" \
    --system-prompt "$COD_PROMPT"

# 4. unsloth UD-Q4_K_M v3.0 + CoD
bench "unsloth-m-cod" \
    $DESKTOP/start-unsloth-qwen38-27b.sh \
    "Q4_K_M" \
    --model-path "$UNSLOTH_M" \
    --system-prompt "$COD_PROMPT"

# Restore huihui as production
bash $DESKTOP/start-huihui-27b-abliterated.sh > /tmp/restore.log 2>&1

echo "=========================================="
echo "  ALL DONE.  results: $RESULTS"
echo "=========================================="