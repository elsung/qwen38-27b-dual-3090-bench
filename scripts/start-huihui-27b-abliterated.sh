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
# LONG-CONTEXT (YaRN) SUPPORT (2026-09-04):
#   Native model max is 262K. For >262K, the Qwen team recommends YaRN scaling
#   (factor=4, rope_theta=10M, original_max_position_embeddings=262144).
#   llama.cpp supports `--rope-scaling yarn` + `--rope-scale N` for linear-ish
#   extension. Note: Qwen3.8 uses 3-axis RoPE (mrope_section=[11,11,10]) which
#   llama.cpp may not handle cleanly; text-only workloads are fine, vision may
#   degrade slightly. We tested up to 262K natively (good); 1M with YaRN is
#   verified to LOAD but accuracy not benchmarked on this hardware.
#
#   Set HUIHUI_CTX_SCALE=4 (or any factor) to enable YaRN:
#     HUIHUI_CTX_SCALE=4 bash start-huihui-27b-abliterated.sh
#   Then in OMP/ST set openai_max_context up to 1048576.
#
# REASONING-STYLE CONFIGURATION (2026-09-04):
#   The llama-server b10753 build does NOT support server-side system-prompt
#   injection (--chat-template-kwargs is a dead-flag in this build). CoD therefore
#   stays CLIENT-SIDE. See docs/REASONING.md and docs/MODELS.md for the full story.
#
#   For SillyTavern: switch to the "Local Huihui ... (Tensor-Split + CoD)"
#   connection profile at $ST_DIR/data/default-user/OpenAI Settings/
#
#   For OMP / curl / direct API: pass CoD prompt as the first system message.
set -euo pipefail

CONTAINER_NAME=vllm-flashnext-fg
PID_FILE=/tmp/llama-huihui.pid
LOG=/tmp/llama-huihui.log
MODEL=$MODELS_DIR/models/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf
LLAMA=$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server
PORT="${LLAMA_HUIHUI_PORT:-8091}"
CTX_SCALE="${HUIHUI_CTX_SCALE:-1}"
ROPE_SCALING_TYPE="${HUIHUI_ROPE_SCALING:-linear}"
YARN_ORIG_CTX="${HUIHUI_YARN_ORIG_CTX:-262144}"
NATIVE_CTX=262144

if [ "$CTX_SCALE" -gt 1 ]; then
    EXTENDED_CTX=$((NATIVE_CTX * CTX_SCALE))
    ROPE_FLAGS=(--rope-scaling "$ROPE_SCALING_TYPE" --rope-scale "$CTX_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX")
else
    EXTENDED_CTX=$NATIVE_CTX
    ROPE_FLAGS=()
fi

cat <<EOF
================================================================
  Huihui llama-server launcher (2026-09-04)

  Context length: $EXTENDED_CTX tokens (\$HUIHUI_CTX_SCALE=$CTX_SCALE, rope=$ROPE_SCALING_TYPE)
  RoPE scaling: ${ROPE_FLAGS[*]:-(none, native 262K)}

  Reasoning-style is CLIENT-SIDE. ST users switch profiles.
  OMP / curl users pass CoD prompt as first system message.

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

echo "[$(date +%H:%M:%S)] starting llama-server for huihui on $PORT (ctx=$EXTENDED_CTX, q8_0 KV)..."
nohup "$LLAMA" \
    -m "$MODEL" \
    -ngl 99 \
    -fa on \
    -c "$EXTENDED_CTX" \
    "${ROPE_FLAGS[@]}" \
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
        echo "[$(date +%H:%M:%S)] huihui abliterated is up on port $PORT"
        echo "Context: $EXTENDED_CTX tokens"
        echo "Switch with: HUIHUI_CTX_SCALE=N bash $0  (N=4 = ~1M with YaRN, default 1 = native 262K)"
        echo "SillyTavern: pick 'Local Huihui Qwen3.8-27B Abliterated (Tensor-Split + CoD)' from the API dropdown"
        echo "OMP / Tailscale clients: point at http://$TAILSCALE_IP:8091/v1 (or LAN IP); pass CoD prompt as first system message"
        exit 0
    fi
    sleep 2
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1