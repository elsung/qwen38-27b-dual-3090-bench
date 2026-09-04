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
