# Speed Matrix — Pre-fill & Decode Across Context Sizes

> Per-model throughput table at multiple prompt sizes, generation lengths, and concurrency.
> All measurements on **dual RTX 3090 + Ryzen 7 5800XT + 128 GiB DDR4**, temperature 0.6.

## Summary table

The headline numbers across all tested configs:

| Model | Quant | Runtime | Context | Decode t/s (single) | Decode t/s (c=4 aggregate) | GSM8K + CoD |
|---|---|---|---:|---:|---:|---:|
| **Qwen3.8-27B huihui abliterated** | UD-Q4_K_XL v2.x (17.4 GiB) | llama.cpp b10753 | 262K native | **70–96** | **278** | **77%** |
| Unsloth Dynamic v3.0 UD-Q4_K_XL | 17.6 GiB | llama.cpp b10753 | 262K native | 88 | — | 60% |
| Unsloth Dynamic v3.0 UD-Q4_K_M | 16.5 GiB | llama.cpp b10753 | 262K native | 89 | — | 67% |
| \*\*Qwen3.8-27B FP8 dense (was mislabeled Flash-Next)\*\* | 28.8 GiB FP8 | vLLM 0.22.1 | 16K (default) | **50–80** | **94** | **13%** |
| Qwen3.8 Flash-Next FP8 | 28.8 GiB FP8 | llama.cpp Inovello b10753 | 16K | 3.86 | — | — |

**Bottom line**: huihui abliterated + CoD prompt is the recommended default for math/reasoning workloads on dual 3090. Flash-Next via vLLM Docker is the recommended runtime for **multi-user concurrent** workloads.

## Detailed speed matrix (single concurrency)

### Qwen3.8-27B huihui abliterated (UD-Q4_K_XL v2.x)

| Prompt size | max_tokens | Decode t/s | Notes |
|---:|---:|---:|---|
| 64 t | 128 | **70.5** | Empty KV cache, fastest decode |
| 64 t | 1,024 | **70.2** | Same — decode-bound, max_tokens doesn't matter |
| 1,024 t | 128 | 64.5 | Small KV cache footprint |
| 1,024 t | 1,024 | **95.7** | Larger generation = more sustained throughput |
| 8,192 t | 128 | 66.9 | Larger prompt, modest slowdown |
| 8,192 t | 1,024 | 71.7 | |
| 32,768 t | 128 | 57.2 | 32K context starts to bite |
| 32,768 t | 1,024 | 61.1 | |

### Qwen3.8-27B FP8 dense (vLLM Docker, 16K) — corrected from "Flash-Next"

| Prompt size | max_tokens | Decode t/s (c=1) | Decode t/s (c=2 aggregate) | Decode t/s (c=4 aggregate) |
|---:|---:|---:|---:|---:|
| 64 t | 128 | 60.2 | — | — |
| 64 t | 1,024 | 64.6 | — | — |
| 1,024 t | 128 | 47.9 | **70.2** | **94.0** |
| 1,024 t | 1,024 | 66.3 | — | — |
| 8,192 t | 128 | 23.9 | — | — |
| 8,192 t | 1,024 | 33.6 | — | — |

**Key insight**: Flash-Next decode t/s **drops ~50%** at 8K prompts vs 64 t prompts, because the larger KV cache takes more time per token. **Concurrent batching recovers aggregate throughput** — c=4 gets 94 t/s aggregate vs 48 t/s single (almost 2× improvement at the same latency).

## How to reproduce

```bash
cd bench
python3 matrix_and_accuracy.py \
  --label my-run \
  --prompt-sizes 64,1024,8192 \
  --max-tokens-list 128,1024 \
  --concurrencies 1,2,4 \
  --runs 3 \
  --include-concurrent \
  --accuracy-data bench/gsm8k_10.jsonl \
  --accuracy-runs 3 \
  --accuracy-max-tokens 1024 \
  --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise." \
  --out my-results.jsonl
```

The harness writes both speed records (`type: speed`) and accuracy records (`type: accuracy`) to the same jsonl. Aggregating via Python:

```python
import json
from collections import defaultdict
by_cell = defaultdict(list)
with open('my-results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('type') == 'speed':
            key = (r['prompt_target'], r['max_tokens'], r['concurrency'])
            by_cell[key].append(r['predicted_per_s'] or r['aggregate_decode_tps'])
for key in sorted(by_cell):
    print(f'p{key[0]} m{key[1]} c{key[2]}: {sum(by_cell[key])/len(by_cell[key]):.1f} t/s')
```

## What the numbers mean

- **Decode t/s** is **mean tokens/second emitted** (not including prefill). Llama-server exposes this via `timings.predicted_per_second`; vLLM doesn't, so we compute it from `completion_tokens / wall_s` ourselves.
- **Aggregate t/s** (concurrent case) is `sum(completion_tokens) / sum(wall_s)`. For a single batch with concurrency=4, this is **4× the per-request t/s if latency is the same**.
- **Prefill t/s** is `prompt_tokens / prompt_processing_time`. Llama-server exposes this; vLLM does not, so vLLM numbers show 0 for prefill.

## Why Flash-Next decode t/s is slower than huihui at the same prompt size

Flash-Next is a **176B MoE** (~6B active per token). At small prompts the MoE experts fit in KV cache and decoding is fast; at larger prompts the expert fetch from host RAM (PLE CPU offload) starts to dominate. **huihui is a 27B dense** — all weights fit in GPU VRAM, no CPU offload.

**Recommendation**:
- Use **huihui + CoD** for single-user math/reasoning at any context
- Use **Flash-Next + vLLM** for concurrent (c≥2) workloads at small-to-medium context
- Avoid Flash-Next for >8K context on dual 3090 (KV cache + MoE offload bottleneck)