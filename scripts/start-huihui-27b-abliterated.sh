#!/usr/bin/env bash
# Start the huihui-ai Qwen3.8-27B abliterated GGUF as a llama-server for SillyTavern / OMP.
# This is the FAST option: ~80 t/s decode empty at 262K, ~60 t/s at 70K populated
# on dual RTX 3090 with the Inovello `flashnext-2x3090` branch (b10753 / 9bd97fe54).
# Native model max context is 262,144 (262K); use the full window so SillyTavern
# can hold long conversations / character cards / lorebooks without truncation.
# IMPORTANT: keep KV cache at q8_0 (not q4_0). q4_0 KV measurably degrades the
# model's reasoning accuracy on this dense 27B; q8_0 is the highest-precision
# KV that still fits ~262K on dual 3090 with FP16 weights on GPU.
#
# REASONING-STYLE CONFIGURATION (2026-09-04):
#   The llama-server b10753 build does NOT support server-side system-prompt
#   injection. We tried patching the chat template (minja rejected the patch
#   on character-escape edge cases; subsequent debugging showed
#   --chat-template-kwargs is a dead-flag in this build — it's parsed but
#   never passed to the renderer). CoD therefore stays CLIENT-SIDE:
#
#   **SillyTavern users**: switch to the "Local Huihui Qwen3.8-27B Abliterated
#   (Tensor-Split + CoD)" connection profile at
#   $ST_DIR/data/default-user/OpenAI Settings/
#   That profile has the CoD system prompt pre-set (use_sysprompt: true).
#   The other profile "Local Huihui Qwen3.8-27B Abliterated (Tensor-Split)"
#   has use_sysprompt: false and stays in baseline behavior.
#
#   **OMP / curl / direct API clients**: include a
#   {"role":"system","content":"Think step by step, but write each step
#   in at most 5 words. Be extremely concise."} message as the first entry
#   in messages[] (or pass --system-prompt in OMP).
#
#   The CoD prompt improves GSM8K accuracy from 23% → 77% on this model
#   (see UNSLOTH_V3_COMPARISON_2026-09-04.md and
#    REASONING_EFFORT_SWEEP_2026-09-04.md).
#
# This launcher only spins up the server. No patches applied.
#
# Stops any running vLLM Flash-Next container first, since both bind port 8091.
set -euo pipefail

CONTAINER_NAME=vllm-flashnext-fg
PID_FILE=/tmp/llama-huihui.pid
LOG=/tmp/llama-huihui.log
MODEL=$MODELS_DIR/models/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf
LLAMA=$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server
PORT="${LLAMA_HUIHUI_PORT:-8091}"

cat <<'EOF'
================================================================
  Huihui llama-server launcher (2026-09-04 final)

  Reasoning-style is CLIENT-SIDE. The server does not inject
  CoD / tokbudget / baseline automatically because
  --chat-template-kwargs is a dead-flag in b10753.

  RECOMMENDED CoD prompt (paste into your client's system role):
    Think step by step, but write each step in at most 5 words.
    Be extremely concise.

  ST users: switch to the "Local Huihui ... (Tensor-Split + CoD)"
  connection profile (pre-injects this prompt).

  OMP / curl users: send this prompt as the first system message,
  or pass --system-prompt in OMP.

  See REASONING_EFFORT_SWEEP_2026-09-04.md for benchmark details.
================================================================
EOF

echo "[$(date +%H:%M:%S)] stopping any running vLLM container on $PORT..."
sg docker -c "docker rm -f $CONTAINER_NAME" 2>/dev/null || true

# Stop any prior llama-huihui process bound to this port
if [ -f "$PID_FILE" ]; then
    OLDPID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] killing prior llama-huihui pid $OLDPID"
        kill "$OLDPID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

echo "[$(date +%H:%M:%S)] starting llama-server for huihui on $PORT (ctx=262144, q8_0 KV)..."
nohup "$LLAMA" \
    -m "$MODEL" \
    -ngl 99 \
    -fa on \
    -c 262144 \
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

for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] huihui abliterated is up on port $PORT"
        echo "Context: 262144 tokens (full native); decode ~80 t/s empty, ~60 t/s at 70K populated (q8_0 KV on b10753)"
        echo "SillyTavern: pick 'Local Huihui Qwen3.8-27B Abliterated (Tensor-Split)' or '... + CoD' from the API dropdown"
        echo "OMP / Tailscale clients: point at http://$TAILSCALE_IP:8091/v1 (or LAN IP); pass CoD prompt as first system message"
        exit 0
    fi
    sleep 2
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1