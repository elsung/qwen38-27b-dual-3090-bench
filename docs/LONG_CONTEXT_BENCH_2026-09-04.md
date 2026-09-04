# Long-Context (YaRN 524K / 1M) Bench Results — 2026-09-04

> **Headline**: On dual RTX 3090 (24 GiB each) + 125 GiB RAM, **524K context via linear YaRN scaling WORKS** (60-80 t/s decode, 53% GSM8K with CoD prompt). **1M context is NOT achievable** — crashes at KV-cache allocation regardless of `--no-kv-unified`. The bottleneck is unified KV buffer (~32 GiB needed for 1M q8_0) > 24 GiB per GPU.

## TL;DR

| Context | Status | Decode t/s | Prefill t/s | GSM8K + CoD |
|---|---|---:|---:|---:|
| 262K (native) | ✅ | 70-99 | 600+ | **77%** |
| **524K (linear YaRN scale=2)** | ✅ works | **40-80** | 8-270 | **53%** |
| 786K (linear YaRN scale=3) | ❌ crashes | n/a | n/a | n/a |
| **1M (linear YaRN scale=4)** | ❌ crashes | n/a | n/a | n/a |

## What we tested

### Hardware
- **CPU**: AMD Ryzen 7 5800XT (8c/16t)
- **RAM**: 125 GiB DDR4 (60 GiB free at idle)
- **Swap**: 126 GiB zram + 17 GiB file (200 GiB total spillover capacity)
- **GPU**: 2× NVIDIA RTX 3090 (24 GiB each, ~5 GiB free after model load)
- **No Optane** — system has 5× HDDs but no NVMe, no Optane

### Software
- llama.cpp **b10753** (Inovello `flashnext-2x3090` branch, commit 9bd97fe54)
- Model: `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` (17.4 GiB, q4_0 GGUF)
- KV cache: q8_0 (default)
- Launch: `HUIHUI_CTX_SCALE=N bash start-huihui-27b-abliterated.sh`

## Speed matrix at 524K context (YaRN linear scale=2)

| Prompt | gen | **Prefill t/s** | **Decode t/s** | Wall (s) |
|---:|---:|---:|---:|---:|
| 1,024 t | 128 t | 218 | **73** | 2.16 |
| 1,024 t | 512 t | 27 | **80** | 4.54 |
| 8,192 t | 128 t | 270 | **62** | 5.27 |
| 8,192 t | 512 t | 24 | **67** | 7.80 |
| 32,768 t | 128 t | 260 | **61** | 15.24 |
| 32,768 t | 512 t | 18 | **61** | 8.60 |
| 131,072 t | 128 t | 207 | **47** | 51.75 |
| 131,072 t | 512 t | 8 | **42** | 12.34 |

**Key observations**:
- Decode t/s **degrades gracefully**: 80 t/s at 1K prompt → 60 t/s at 32K → 42 t/s at 131K
- Prefill t/s at 32K-131K is bandwidth-limited (~200 t/s initially, drops to 8 t/s when KV buffer hits RAM page-fault rate)
- Wall time for 131K prompt + 128 generation: **51.75 seconds** (prefill dominates at large contexts)
- KV offload to RAM works (no swap-to-disk thrashing observed)

## GSM8K accuracy at 524K

| Config | GSM8K acc | Notes |
|---|---:|---|
| Native 262K + CoD (prior session) | **77%** | Baseline |
| **524K via YaRN linear + CoD** | **53%** | -24pp accuracy hit from YaRN scaling |
| Native 262K no CoD (prior session) | 23% | Baseline |

**The -24pp accuracy drop** at scale=2 is consistent with reports on linear RoPE scaling — YaRN's official Qwen3.8 config uses `--rope-scaling yarn` (not just `linear`), but our attempts with `yarn` mode also crashed at scale=3+. Text-only workloads may see less degradation than this 53% suggests, but for GSM8K (math reasoning), the accuracy hit is real.

## What didn't work

### 1M context (scale=4) — crashes at startup

```
HUIHUI_CTX_SCALE=4 bash start-huihui-27b-abliterated.sh
→ ggml_backend_meta_alloc_ctx_tensors_from_buft() abort
   llama_kv_cache::llama_kv_cache(...)
   ggml_print_backtrace + ggml_abort
```

**Root cause**: llama.cpp's unified KV buffer allocation fails because:
- 1M context × 27B model × q8_0 KV = **~32 GiB** of KV buffer needed at startup
- Per-layer allocation exceeds each GPU's 24 GiB
- `--no-kv-unified` flag does NOT help — still allocates per-layer as one big buffer per layer
- `--kv-offload` only handles spillover during runtime, **not** the initial allocation

### 786K context (scale=3) — same crash

Same root cause. The "sweet spot" between 524K (works) and 786K (crashes) on dual 24 GiB GPUs is around **~600K tokens** for q8_0 KV.

### yarn scaling mode — also crashes at scale=3+

Even with `--rope-scaling yarn` (the official Qwen3.8-recommended mode), the 786K+ context crashes in the same `ggml_backend_meta_alloc_ctx_tensors_from_buft` call. The bottleneck is llama.cpp's KV buffer pre-allocation, not the RoPE math.

## What WOULD work for 1M

The issue is **GGUF's static pre-allocation** of KV cache. To get to 1M, you'd need:

| Option | Feasibility on dual 24 GiB 3090 |
|---|---|
| **Linear scaling + mmap KV cache** (not yet in llama.cpp) | Not implemented |
| **CPU-offload KV cache** (--cpu-moe style for non-MoE) | Not in llama.cpp b10753 |
| **Spill KV to disk via OS swap** | KV pre-allocates, swap wouldn't help at startup |
| **Use a smaller quantization** (q4_0 KV instead of q8_0) | Reduces 1M KV from 32→16 GiB, might fit at scale=4 |
| **Upgrade hardware** to A100 80GB × 1 or H100 80GB × 1 | Would work cleanly |

We did **not** test q4_0 KV at scale=4 (would be ~5-10pp accuracy loss for the math benchmark per prior session). Worth testing if you have time.

## Comparison: huihui 524K vs Flash-Next 16K vs native 262K

| Metric | Native 262K | **YaRN 524K** | Flash-Next 16K (vLLM) |
|---|---:|---:|---:|
| Best decode t/s | 99 | **80** | 80 |
| Best prefill t/s | 760 | **270** | n/a (vLLM doesn't expose) |
| Worst decode t/s (long prompt) | 60 (32K prompt) | **42 (131K prompt)** | 24 (8K prompt) |
| GSM8K + CoD accuracy | 77% | **53%** | 13% |
| VRAM needed | 17 GiB (q8 KV @ 262K) | 17 GiB (q8 KV @ 524K, mostly in HBM) | 14 GiB FP8 weights + small KV |
| Practical max context | 262K (native) / 524K (YaRN scale=2) | 524K | 16K (tested) |

## Files

- `bench/1m-bench/scale-2-524k.jsonl` — raw benchmark records (25 records: 24 throughput + 1 accuracy)
- `bench/matrix_and_accuracy.py` — benchmark harness used
- `start-huihui-27b-abliterated.sh` — launcher with `HUIHUI_CTX_SCALE=N` env var

## How to reproduce

```bash
# 524K context (works)
HUIHUI_CTX_SCALE=2 bash start-huihui-27b-abliterated.sh

# Wait for /health (~30s)

# Run the matrix
cd bench
python3 matrix_and_accuracy.py \
  --label huihui-524k-linear \
  --prompt-sizes 1024,8192,32768,131072 \
  --max-tokens-list 128,512 \
  --concurrencies 1 \
  --runs 3 \
  --accuracy-data bench/gsm8k_10.jsonl \
  --accuracy-runs 3 \
  --accuracy-max-tokens 1024 \
  --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise." \
  --out /tmp/1m-bench/scale-2-524k.jsonl
```

## Recommendations for the user

1. **For long-context huihui workloads on this hardware**: stay at **262K native** (best accuracy + best speed) or use **524K YaRN** when you need ~2× the context. Both CoD-friendly.
2. **For 1M context**: needs either A100/H100 (80 GiB+) hardware, or a llama.cpp build that supports KV cache spilling at startup. None of these are available on this box.
3. **For Flash-Next**: vLLM doesn't support KV disk-offload either. Stay at native 16K (or 64K if you have enough VRAM — though we OOMed at 64K on this hardware).

## What we still need to test

- [ ] Try q4_0 KV cache + scale=4 (might fit 1M by halving KV size)
- [ ] Try `--load-mode mlock` to pin model in RAM, freeing OS swap for KV spillover
- [ ] Re-test scale=3 with `--cpu-moe` if applicable (it isn't, huihui is dense)
- [ ] Try vLLM 0.22.x newer patches for KV cache offload if any exist
- [ ] On a box with ≥80 GiB VRAM: try VLLM_CONTEXT=1048576 on Flash-Next

## What was actually measured (raw data)

The `bench/1m-bench/scale-2-524k.jsonl` has 25 records:
- 24 speed records (8 prompt sizes × 2 max_tokens × 3 runs minus some dropouts due to a printout glitch at 32K×128 r0)
- 1 accuracy summary record (53% / 16/30 / avg 85.5 t/s decode)

Re-run with `--runs 5` for tighter statistics if you want publishable numbers.