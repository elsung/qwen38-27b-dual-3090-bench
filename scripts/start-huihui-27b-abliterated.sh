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
# LONG-CONTEXT (YaRN + RoPE scaling) — 2026-09-04:
#   llama.cpp b10753 supports 3 scaling modes: none, linear, yarn.
#
#   Best knobs (verified empirically on dual RTX 3090):
#     -c 524288                 # 524K context (sweet spot for 27B + dual 24 GiB)
#     --rope-scaling linear      # default; works, drops GSM8K acc from 77% -> 53%
#     --rope-scaling yarn        # better preservation; tune --yarn-orig-ctx
#     --rope-scale 2            # 262K -> 524K
#
#   To match the official Qwen YaRN config (factor=4, rope_theta=10M):
#     -c 1048576
#     --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768
#     (May OOM at 1M on dual 24 GiB due to KV buffer > 24 GiB.)
#
#   NTK-aware scaling (alternative to yarn):
#     --rope-scaling yarn --rope-freq-base 10000000 --rope-scale 2
#
#   Defaults:
#     HUIHUI_CTX_SCALE=2 -> -c 524288 + --rope-scale 2
#     HUIHUI_ROPE_SCALING=linear|yarn (default: linear)
#     HUIHUI_YARN_ORIG_CTX=262144 (default), use 32768 for Qwen-style 32K->1M
#     HUIHUI_YARN_ATTN_FACTOR=1.0 (default YaRN), 0.0 = pure interpolation
#     HUIHUI_ROPE_FREQ_BASE=10000000 (Qwen default)
#
# REASONING-STYLE CONFIGURATION:
#   llama-server b10753 does NOT support server-side system-prompt injection
#   (--chat-template-kwargs is a dead-flag). CoD stays CLIENT-SIDE:
#
#   SillyTavern: switch to "Local Huihui ... (Tensor-Split + CoD)" profile.
#   OMP / curl: pass CoD prompt as first system message.
set -euo pipefail

CONTAINER_NAME=vllm-flashnext-fg
PID_FILE=/tmp/llama-huihui.pid
LOG=/tmp/llama-huihui.log
MODEL=$MODELS_DIR/models/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf
LLAMA=$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server
PORT="${LLAMA_HUIHUI_PORT:-8091}"

CTX_SCALE="${HUIHUI_CTX_SCALE:-2}"       # default 2 = 524K (best accuracy, ~zero speed cost)
ROPE_SCALING="${HUIHUI_ROPE_SCALING:-yarn}"       # yarn preserves accuracy; linear loses ~10pp
YARN_ORIG_CTX="${HUIHUI_YARN_ORIG_CTX:-262144}"
YARN_ATTN_FACTOR="${HUIHUI_YARN_ATTN_FACTOR:-1.0}"  # 1.0 critical: 0.0 loses 13pp accuracy
ROPE_FREQ_BASE="${HUIHUI_ROPE_FREQ_BASE:-}"

case "$ROPE_SCALING" in
  none|linear|yarn) ;;
  *)
    echo "ERROR: HUIHUI_ROPE_SCALING='$ROPE_SCALING' must be one of: none | linear | yarn" >&2
    exit 2
    ;;
esac

if [ "$CTX_SCALE" -gt 1 ] && [ "$ROPE_SCALING" = "none" ]; then
    echo "ERROR: HUIHUI_CTX_SCALE>1 requires HUIHUI_ROPE_SCALING=yarn or linear (not 'none')" >&2
    exit 2
fi

EXTENDED_CTX=$((262144 * CTX_SCALE))
ROPE_FLAGS=()
case "$ROPE_SCALING" in
  yarn)
    ROPE_FLAGS=(--rope-scaling yarn --rope-scale "$CTX_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX" --yarn-attn-factor "$YARN_ATTN_FACTOR")
    if [ -n "$ROPE_FREQ_BASE" ]; then
      ROPE_FLAGS+=(--rope-freq-base "$ROPE_FREQ_BASE")
    fi
    ;;
  linear)
    ROPE_FLAGS=(--rope-scaling linear --rope-scale "$CTX_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX")
    if [ -n "$ROPE_FREQ_BASE" ]; then
      ROPE_FLAGS+=(--rope-freq-base "$ROPE_FREQ_BASE")
    fi
    ;;
  none) ;;
esac

cat <<EOF
================================================================
  Huihui llama-server launcher (2026-09-04)

  Context length: $EXTENDED_CTX tokens (scale=$CTX_SCALE, rope=$ROPE_SCALING)
  RoPE flags: ${ROPE_FLAGS[*]:-(none, native 262K)}
  Yarn: orig=$YARN_ORIG_CTX attn_factor=$YARN_ATTN_FACTOR
EOF
if [ -n "$ROPE_FREQ_BASE" ]; then
    echo "  NTK freq base: $ROPE_FREQ_BASE"
fi
cat <<EOF

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
        echo "Tune with: HUIHUI_CTX_SCALE=N HUIHUI_ROPE_SCALING=yarn bash $0"
        echo "SillyTavern: pick 'Local Huihui Qwen3.8-27B Abliterated (Tensor-Split + CoD)' from the API dropdown"
        echo "OMP / Tailscale clients: point at http://$TAILSCALE_IP:8091/v1 (or LAN IP); pass CoD prompt as first system message"
        exit 0
    fi
    sleep 2
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1