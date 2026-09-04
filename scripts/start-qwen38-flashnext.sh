#!/usr/bin/env bash
# Start the Qwen3.8 Flash-Next FP8 vLLM container (Qwen4 architecture preview, 176B/6B MoE).
# ~60.86 t/s single-user with MTP3 + PLE CPU offload. Stops any running huihui llama-server first.
set -euo pipefail

PID_FILE=/tmp/vllm-flashnext.pid
CONTAINER=vllm-flashnext-fg
LOG=/tmp/vllm-flashnext.log
FP8_DIR=$MODELS_DIR/models/Qwen3.8-27B-FP8
PORT="${VLLM_FLASHNEXT_PORT:-8091}"

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

echo "[$(date +%H:%M:%S)] starting vLLM Flash-Next container on $PORT..."
sg docker -c "docker run -d --rm --gpus all --ipc=host --network=host \
    -v $FP8_DIR:/models/fp8:ro \
    --name $CONTAINER \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    vllm/vllm-openai:qwen38-flash-next \
    /models/fp8 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.80 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 4096 \
      --max-model-len 16384 \
      --no-enable-flashinfer-autotune \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
      --port $PORT" > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] waiting for vLLM /v1/models..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Flash-Next is up on port $PORT"
        echo ""
        echo "In SillyTavern, pick 'Local Qwen3.8-Flash-Next (FP8 vLLM)' from the API dropdown"
        echo "Set temperature 1.0, top_p 0.95 (thinking mode)."
        exit 0
    fi
    sleep 5
done
echo "[$(date +%H:%M:%S)] timed out; check $LOG"
exit 1
