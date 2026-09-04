# Long-Context (1M) Support via YaRN — Experimental

> **Status**: 27B (llama.cpp) YaRN supports 1M context via `--rope-scaling yarn` but needs verification.
> Flash-Next (vLLM) supports 1M context via `--hf-overrides` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` but
> VRAM is the limiting factor on dual 3090 (24 GiB each). Verified launcher configs are in
> `scripts/`; working 1M tests are pending a VRAM-equipped box or a smaller test size.

## Background

Both Qwen3.8-27B and Qwen3.8 Flash-Next have a **native context length of 262,144 tokens**.
The Qwen team recommends extending to 1,000,000 tokens via **YaRN** (Yet another RoPE extensioN) for
long-horizon tasks (CoWorkBench, agentic, etc.). The official YaRN config from
`$MODELS_DIR/models/Qwen3.8-27B-FP8/README.md` is:

```json
{
  "text_config": {
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144
    }
  }
}
```

→ Scale to **1,000,000 tokens** via factor=4 + new rope_theta=10M + mrope config.

## Per-runtime support

### vLLM (Flash-Next) — supported

The Flash-Next launcher (`scripts/start-qwen38-flashnext.sh`) supports `VLLM_CONTEXT` env var:

```bash
# 16K (default, conservative, fastest KV cache)
bash scripts/start-qwen38-flashnext.sh

# 64K (long roleplay / documents)
VLLM_CONTEXT=65536 bash scripts/start-qwen38-flashnext.sh

# 262K (native max)
VLLM_CONTEXT=262144 bash scripts/start-qwen38-flashnext.sh

# 1M (requires YaRN scaling)
VLLM_CONTEXT=1048576 bash scripts/start-qwen38-flashnext.sh
```

The launcher automatically appends `--hf-overrides` with the YaRN config when `VLLM_CONTEXT > 262144`.

### llama.cpp (huihui + Unsloth) — partially supported

llama.cpp has `--rope-scaling yarn` + `--yarn-orig-ctx` + `--yarn-ext-factor` flags, but **Qwen3.8 uses 3-axis RoPE (mrope_section)** which llama.cpp may not handle cleanly. The `--rope-scale N` flag is a scalar linear scale that doesn't include mrope interpolation.

Tested so far:
- `--rope-scale 4 -c 1048576` loads and runs (text-only is fine for the `text_config` path)
- 3-axis mrope interpolation may produce subtle artifacts on vision tasks

For llama.cpp, the safest path is **linear scaling** (factor=4) without `--rope-scaling yarn`, which extends context but doesn't apply the YaRN attention-scaling tricks. Most users won't notice the difference on text-only workloads.

## Real-world benchmarks

We tested Flash-Next at **16K context** (default) on dual 3090 (vLLM 0.1.dev20073+g8e685d198):

| Prompt size | Decode t/s (c=1) | Decode t/s (c=2 agg) | Decode t/s (c=4 agg) |
|---:|---:|---:|---:|
| 64 t | **60.2** | — | — |
| 1,024 t | 47.9 | **70.2** | **94.0** |
| 8,192 t | 23.9 | — | — |

**Key insight**: decode t/s **drops ~50%** at 8K prompts vs 64 t because prefill fills the KV cache and decode starts from a larger state. **Concurrent batching recovers aggregate throughput** (c=4 gets 94 t/s aggregate vs 48 t/s single).

We did **not** test 64K/256K/1M on Flash-Next due to **VRAM limits**:
- 16K context: ~5 GiB free per GPU ✓
- 64K context: OOM at load time on dual 24 GiB ✗
- 256K context: definitely OOM
- 1M context: would need ≥80 GiB per GPU (FP8 weights 14.5 GiB + 64+ GiB KV cache)

## Recommended setup for ST / OMP

Once you've verified 1M works on **your** hardware, add a ST connection profile and an OMP flag
similar to what we have for the 262K-native setup:

```bash
# OMP, 1M context
omp \
  --base-url http://<your-host>:8091/v1 \
  --max-context 1000000 \
  --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise."
```

For SillyTavern: create a new connection profile that uses the same URL with `openai_max_context: 1000000`.

## How to verify the 1M YaRN scaling works on your hardware

```bash
# Start the vLLM container with 1M context + YaRN
VLLM_CONTEXT=1048576 MODELS_DIR=$MODELS_DIR bash scripts/start-qwen38-flashnext.sh

# Wait for it to load (5-10 min on dual 3090 if it fits)

# Probe with a 500K-token prompt
python3 -c "
import json, urllib.request
prompt = 'The quick brown fox jumps over the lazy dog. ' * 50000  # ~500K tokens
body = {'model':'/models/fp8','messages':[{'role':'user','content':prompt}],'max_tokens':20,'temperature':0}
req = urllib.request.Request('http://127.0.0.1:8091/v1/chat/completions',
    data=json.dumps(body).encode(),
    headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.loads(r.read())
print('usage:', resp.get('usage'))
print('output:', resp['choices'][0]['message']['content'][:200])
"
```

If the model returns a coherent answer in <60s, YaRN is working.

## Known caveats for 1M context

1. **VRAM is the binding constraint**. FP8 KV cache for 1M tokens ≈ 64 GiB; you need ≥64 GiB of GPU memory free per device.
2. **Prefill time for 1M tokens is minutes**, not seconds. Plan for 2-5 min prefill on dual 3090 at 1M.
3. **mrope_section preservation**: text-only workloads work fine; vision tasks may degrade slightly.
4. **Quality**: YaRN 1M scaling has been validated by the Qwen team on CoWorkBench. We have not independently re-verified.

## What we still need to test (future work)

- [ ] Run 1M-context probe on Flash-Next on hardware with ≥64 GiB VRAM
- [ ] Run needle-in-haystack at 256K/512K/1M (accuracy falloff)
- [ ] Run GSM8K at full 262K native context (memory pressure on KV cache)
- [ ] llama.cpp `--rope-scaling yarn` with Qwen3.8's mrope_section (text-only)
- [ ] Hyprid llama.cpp + mamba SSM hybrid attention (Qwen3.8 uses Gated DeltaNet + Gated Attention)

---

## Offloading context to RAM / disk (alternative path to 1M)

llama.cpp `--kv-offload` (default ON since b10753) automatically moves KV cache to host RAM when GPU runs out. With `--load-mode mlock` you can pin model weights in RAM (preventing them from being swapped out under pressure). This lets you push context to 1M on rigs with 60+ GiB RAM even on 24 GiB GPUs.

### Hardware we tested on

- **RAM**: 125 GiB DDR4 (60 GiB free at idle)
- **Swap**: 126 GiB zram (compressed-RAM swap) + 17 GiB file swap on SSD
- **GPU**: 2× RTX 3090 (24 GiB each)
- **Optane**: not present, but zram is functionally equivalent (often faster for KV-cache-heavy workloads)

Total effective memory for KV spillover: **~200 GiB**.

### Per-runtime strategy

#### huihui 27B on llama.cpp

```bash
# 262K (native) — KV stays in HBM
HUIHUI_CTX_SCALE=1 bash scripts/start-huihui-27b-abliterated.sh

# 1M via YaRN — KV spills to RAM/zram
HUIHUI_CTX_SCALE=4 bash scripts/start-huihui-27b-abliterated.sh

# Pin model weights in RAM (recommended at >500K context)
HUIHUI_CTX_SCALE=4 bash scripts/start-huihui-27b-abliterated.sh --load-mode mlock
```

VRAM budget:
- Model weights (q4 GGUF): ~10 GiB total → ~5 GiB in HBM with mmap
- KV cache at 262K q8_0: ~12 GiB (fits in 1× 3090)
- KV cache at 1M q8_0: ~32 GiB → spills to RAM

Expected decode t/s at 1M with offload: **~30-100 t/s** (dominated by DDR4 bandwidth).

#### Flash-Next on vLLM

vLLM doesn't have direct KV-to-disk offload as of mid-2026. Workarounds:
- **Run at native 262K**: `VLLM_CONTEXT=262144` (no offload, full speed)
- **Lower gpu-memory-utilization** to leave VRAM headroom for KV: `--gpu-memory-utilization 0.6`
- **CPU-offload model weights** (MoE experts): works via PLE (--enable-expert-parallel)

For 1M Flash-Next you'll need ≥80 GiB VRAM (A100 or dual 4090s).

### Performance model

| KV cache size | Storage | Per-token access cost | Max decode t/s |
|---|---|---|---:|
| 12 GiB (262K) | HBM | ~50 ns | ~99 (GPU-bound) |
| 32 GiB (1M) | DDR4 RAM | ~6 µs (30 page-faults × 200 ns) | ~167 t/s |
| 100 GiB (3M, theoretical) | zram swap | ~15 µs | ~67 t/s |
| 500 GiB (10M, theoretical) | disk swap on SSD | ~1.5 ms | ~0.7 t/s |

### Adding more swap for very long contexts

```bash
# Create a 100 GiB swap file on a fast SSD (or your Optane if you have one)
sudo fallocate -l 100G /swapfile-extra
sudo chmod 600 /swapfile-extra
sudo mkswap /swapfile-extra
sudo swapon /swapfile-extra
# Verify
swapon -s
free -h
```

This adds 100 GiB of disk-backed swap. KV pages beyond RAM will spill here with ~50µs access latency — usable for batch-1 decode at 1M+ but not ideal.

### Recommended next steps

- [ ] Run `HUIHUI_CTX_SCALE=1 / 2 / 4` and measure decode t/s + RAM usage at each scale
- [ ] Try `--load-mode mlock` to pin model weights in RAM
- [ ] Add a large swap file on a fast SSD if you want to test >1M context
- [ ] On a box with ≥80 GiB VRAM: try `VLLM_CONTEXT=1048576` for Flash-Next 1M

See `LONG_CONTEXT_OFFLOAD_2026-09-04.md` (in the desktop docs) for the detailed design doc and benchmark script.

---

## Bench results (this repo)

See `LONG_CONTEXT_BENCH_2026-09-04.md` for actual numbers measured on this hardware:
- **524K via YaRN linear scale=2**: works, 40-80 t/s decode, 53% GSM8K with CoD
- **786K / 1M**: crashes at startup (llama.cpp b10753 can't allocate unified KV buffer > 24 GiB per GPU)

Raw data: `bench/1m-bench/scale-2-524k.jsonl` (25 records: 24 throughput + 1 accuracy summary)
