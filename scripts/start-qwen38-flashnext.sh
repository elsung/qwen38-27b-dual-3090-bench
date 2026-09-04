#!/usr/bin/env bash
# Start the Qwen3.8 Flash-Next FP8 vLLM container (Qwen4 architecture preview, 176B/6B MoE).
# ~60.86 t/s single-user with MTP3 + PLE CPU offload. Stops any running huihui llama-server first.
#
# Context length knobs (env vars):
#   LLAMA_HUIHUI_PORT     default 8091 (shared with huihui)
#   VLLM_CONTEXT          16384 | 65536 | 262144 | 1048576   (default 16384)
#                         - 16384 = conservative, fastest KV cache
#                         - 65536 (64K) = good for long roleplay / docs
#                         - 262144 (262K) = native max, 18 GiB KV cache per GPU
#                         - 1048576 (1M) = requires YaRN scaling (auto-enabled at >262K)
#
# YaRN config for 1M context (from Qwen3.8-27B model README):
#   factor=4.0, rope_theta=10M, original_max_position_embeddings=262144
#   Plus mrope_section=[11,11,10] for Qwen3.8's 3-axis RoPE
#
# Usage:
#   VLLM_CONTEXT=65536    bash scripts/start-qwen38-flashnext.sh   # 64K context
#   VLLM_CONTEXT=1048576  bash scripts/start-qwen38-flashnext.sh   # 1M with YaRN
set -eo pipefail

PID_FILE=/tmp/vllm-flashnext.pid
CONTAINER=vllm-flashnext-fg
LOG=/tmp/vllm-flashnext.log
FP8_DIR="${MODELS_DIR:-$HOME}/models/Qwen3.8-27B-FP8"
PORT="${VLLM_FLASHNEXT_PORT:-8091}"
CTX_LEN="${VLLM_CONTEXT:-16384}"

# Sanity check: Qwen3.8-27B native is 262K; >262K needs YaRN scaling.
NEEDS_YARN=0
if [ "$CTX_LEN" -gt 262144 ]; then
    NEEDS_YARN=1
fi

if [ ! -d "$FP8_DIR" ]; then
    echo "ERROR: FP8 model directory not found: $FP8_DIR" >&2
    echo "Set MODELS_DIR or edit this script." >&2
    exit 2
fi

echo "[$(date +%H:%M:%S)] stopping any running huihui llama-server on $PORT..."
if [ -f "$PID_FILE" ]; then
    OLDPID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] killing prior llama-server pid $OLDPID"
        kill "$OLDPID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Build the extra vLLM args
HF_OVERRIDES=""
VLLM_EXTRA=""
if [ "$NEEDS_YARN" -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] enabling YaRN for context > 262K (target=$CTX_LEN)"
    HF_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
    VLLM_EXTRA="--hf-overrides '$HF_OVERRIDES'"
fi

echo "[$(date +%H:%M:%S)] starting vLLM Flash-Next container on $PORT (FP8=$FP8_DIR, ctx=$CTX_LEN)..."
sg docker -c "docker run -d --rm --gpus all --ipc=host --network=host \
    -v $FP8_DIR:/models/fp8:ro \
    --name $CONTAINER \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm/vllm-openai:qwen38-flash-next \
    /models/fp8 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.80 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 4096 \
      --max-model-len $CTX_LEN \
      --no-enable-flashinfer-autotune \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
      --port $PORT \
      $VLLM_EXTRA" > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] waiting for vLLM /v1/models..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        MODEL=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        echo "[$(date +%H:%M:%S)] Flash-Next is up on port $PORT (model: $MODEL)"
        echo "Context length: $CTX_LEN tokens ($([ $NEEDS_YARN -eq 1 ] && echo YaRN-enabled || echo native))"
        echo ""
        echo "In SillyTavern, pick 'Local Qwen3.8-Flash-Next (FP8 vLLM)' from the API dropdown"
        echo "Set temperature 1.0, top_p 0.95 (thinking mode)."
        exit 0
    fi
    sleep 5
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1