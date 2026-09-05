# Qwen3.8-Flash-Next 176B — the "Part 2" recipe (GPU LRU expert cache + MTP)

Status: **recipe + launcher ready, not yet benchmarked on our box** (machine was
needed for huihui inference). Launcher: `~/Desktop/start-qwen38-flashnext-2x3090.sh`.

## Background: why we were stuck at ~4 t/s

Our earlier attempts streamed expert weights from host RAM in bulk
(`--n-cpu-moe 40 --moe-expert-cache 16`, best: 4.04 t/s). That approach is
DDR4-bandwidth-bound: 6B active params at Q4 ≈ 3+ GB read per token from a
dual-channel DDR4 host.

The fix (r/LocalLLaMA [1w5vjp6] → [1w6ozbj], author Inovello — same person whose
`flashnext-2x3090` branch we already run at commit `9bd97fe54`, build 10753):

1. **All 48 expert layers pinned in host RAM** (`-ot ffn_(gate|up|down)_exps\.weight=CUDA_Host`),
   PLE/n-gram tables on CPU (`per_layer_token_embd\.weight=CPU`).
2. **GPU-resident LRU expert cache** (`--moe-expert-cache N`): experts repeat across
   consecutive tokens, so hot experts live in VRAM. 84-92% hit rate on code/prose.
3. **MTP on top** (`--spec-type draft-mtp -md mtp-...-shared-Q8_0.gguf`): 50-58%
   draft acceptance on reasoning text, 94% on code emission.
4. Load-time fix (PR #28223): host-destination tensors read directly from file —
   8.5 min → 2 min load. Already in our commit.

OP's measured numbers (2x3090 PCIe 3.0, **dual Xeon E5-2696 v4, 188 GB DDR4-2133
LRDIMM** — 8 memory channels vs our 2):

| Config | decode (thinking) | decode (code, no think) | @131K depth | prefill 26K | load |
|---|---|---|---|---|---|
| Q6, layers parked (≈ our old recipe) | 17 | - | 12 | ~350 (ub2048) | 13 min |
| Q6 + cache 135 | 25-29 | 24 | 17 | 138 | 8.5 min |
| **Q4 + cache 188 + ngram** | 32-35 | 37 | 18-20 | 180-195 | 2 min |
| **Q4 + cache 150 + MTP** | **37-41** | **49** | 14-16 | 180-195 | 2 min |

Our adaptation (in the launcher): `-c 131072` (we have 125 GB RAM vs OP's 188;
host tensors need ~73 GB experts + 28 GB PLE), `-t 16 -tb 16` (Ryzen 8c/16t vs
dual Xeon), MTP draft head Q8_0 downloaded (2.79 GB, checksum-verified).
Expect **less** than OP's numbers — 2 vs 8 DDR4 channels hurts the 8-15% cache-miss
traffic; how much is exactly what we need to measure.

## Launcher knobs

```
FN_CTX=131072  FN_CACHE=150  FN_SPEC=draft-mtp   # default
FN_SPEC=ngram  FN_CACHE=188                      # deep-context sessions (OP: better at 131K)
FN_PORT=8091                                      # one server at a time (RAM)
```

Gotchas carried from OP (verify on our box): q8_0 KV hurts at depth (-18% @131K)
→ keep f16; `--load-mode none` no gain; >2 cache uploads/step saturates PCIe;
MTP helps only with the cache taking verify batches (mainline PR #28243).

## Old-recipe baseline (measured here, prior session)

| Config | Decode |
|---|---|
| `--n-cpu-moe 40` no cache | 2.68 t/s |
| `--n-cpu-moe 40 --moe-expert-cache 16` | 4.04 t/s (cache hit 10.9%) |

Also observed today: relaunching the old recipe cold → 7-token prefill took 87 s
(page-fault storm) before I killed it. The Part-2 recipe's load fix avoids this.

Threads: [Part 1](https://www.reddit.com/r/LocalLLaMA/comments/1w5vjp6/) ·
[Part 2](https://www.reddit.com/r/LocalLLaMA/comments/1w6ozbj/) ·
Branch: `Inovello/llama.cpp@flashnext-2x3090` (local: `$WORKSPACE/inovello-flashnext`,
build `build-3090`, verified `9bd97fe54` = branch head as of 2026-09-04).
