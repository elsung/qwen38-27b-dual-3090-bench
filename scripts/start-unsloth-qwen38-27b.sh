#!/usr/bin/env bash
# Launch unsloth Qwen3.8-27B-GGUF v3.0 (base, unabliterated) on the same
# port as the huihui launcher so the gsm8k harness can target either model.
# Same recipe as start-huihui-27b-abliterated.sh; only -m and the LLM_REASONING_STYLE
# semantic change.
#
# Usage:
#   LLAMA_MODEL=UD-Q4_K_XL bash start-unsloth-qwen38-27b.sh
#   LLAMA_MODEL=UD-Q4_K_M bash start-unsloth-qwen38-27b.sh
set -euo pipefail

CONTAINER_NAME=vllm-flashnext-fg
PID_FILE=/tmp/llama-unsloth.pid
LOG=/tmp/llama-unsloth.log
LLAMA=$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server
PORT="${LLAMA_UNSLOTH_PORT:-8091}"
MODEL_VARIANT="${LLAMA_MODEL:-UD-Q4_K_XL}"
REASONING_STYLE="${LLAMA_REASONING_STYLE:-baseline}"

case "$MODEL_VARIANT" in
  UD-Q4_K_XL) MODEL="$MODELS_DIR/models/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf" ;;
  UD-Q4_K_M)  MODEL="$MODELS_DIR/models/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf" ;;
  *) echo "ERROR: LLAMA_MODEL='$MODEL_VARIANT' must be one of UD-Q4_K_XL | UD-Q4_K_M" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model file not found: $MODEL" >&2
  exit 2
fi

declare -A REASONING_PROMPTS=(
  [cod]="Think step by step, but write each step in at most 5 words. Be extremely concise."
  [tokbudget]="Think step by step using at most 200 tokens of reasoning, then give the final answer."
  [baseline]=""
  [none]=""
)
case "$REASONING_STYLE" in
  cod|tokbudget|baseline|none) ;;
  *) echo "ERROR: LLAMA_REASONING_STYLE='$REASONING_STYLE' is not one of: cod | tokbudget | baseline | none" >&2; exit 2 ;;
esac
SYSTEM_PROMPT="${REASONING_PROMPTS[$REASONING_STYLE]}"
echo "$SYSTEM_PROMPT" > /tmp/llama-unsloth-system-prompt.txt

echo "[$(date +%H:%M:%S)] stopping any running vLLM container on $PORT..."
sg docker -c "docker rm -f $CONTAINER_NAME" 2>/dev/null || true

# Stop any prior llama-unsloth OR llama-huihui on this port
for PF in /tmp/llama-unsloth.pid /tmp/llama-huihui.pid; do
  if [ -f "$PF" ]; then
    OPID=$(cat "$PF" 2>/dev/null || echo "")
    if [ -n "$OPID" ] && kill -0 "$OPID" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] killing prior llama pid $OPID (from $PF)"
      kill "$OPID" 2>/dev/null || true
      sleep 2
    fi
    rm -f "$PF"
  fi
done

echo "[$(date +%H:%M:%S)] starting llama-server for unsloth Qwen3.8-27B ($MODEL_VARIANT) on $PORT (reasoning-style=$REASONING_STYLE)..."
nohup "$LLAMA" \
    -m "$MODEL" \
    -ngl 99 \
    -fa on \
    -c 32768 \
    --split-mode tensor \
    --tensor-split 1,1 \
    --parallel 1 \
    --jinja \
    --host 0.0.0.0 \
    --port "$PORT" \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    -ctk q8_0 \
    -ctv q8_0 \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "[$(date +%H:%M:%S)] llama-server pid $PID, waiting for /health..."

for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] unsloth Qwen3.8-27B ($MODEL_VARIANT) is up on port $PORT"
    echo "Model: $MODEL"
    echo "Reasoning-style: $REASONING_STYLE"
    if [ -n "$SYSTEM_PROMPT" ]; then
      echo "System prompt (inject as first system message):"
      echo "  -->  $SYSTEM_PROMPT"
      echo "Stored at: /tmp/llama-unsloth-system-prompt.txt"
    fi
    exit 0
  fi
  sleep 2
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1