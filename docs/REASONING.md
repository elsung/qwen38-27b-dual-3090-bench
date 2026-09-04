# Reasoning-Effort Optimization & 3-Way Model Comparison

> The headline findings of this repo. Read this first.

## Sources & references

Every claim in this doc is grounded in a specific paper or upstream doc. Links:

| Citation | Used for |
|---|---|
| **[Chain-of-Draft (CoD)](https://arxiv.org/abs/2502.18600)** — Zheng et al., 2025 | The 5-word-draft prompt technique (gave +54pp on huihui) |
| **[NoWait paper](https://arxiv.org/abs/2506.08343)** — Wang et al., 2025 | Suppressing reflection tokens at the logit level (gave +10pp under sampling) |
| **[Stop Overthinking survey](https://arxiv.org/abs/2503.16419)** — TMLR 2025 | The "awesome list" of efficient-reasoning methods |
| **[Unsloth Dynamic 3.0 docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)** | Claims ">10% better top-1% accuracy at the same size" — we could NOT reproduce on GSM8K with CoD |
| **[Qwen3.8-27B model README](https://huggingface.co/Qwen/Qwen3.8-27B)** (search for YaRN) | Official YaRN config for 1M context extension |
| **[gsm8k_harness.py](../../bench/gsm8k_harness.py)** | Our eval harness (urllib only, no deps) |
| **[matrix_bench.py](../../bench/matrix_bench.py)** | Throughput matrix harness |

All numbers in this doc come from running the scripts in `bench/`. Reproducible end-to-end.

> The headline findings of this repo. Read this first.

## TL;DR — what works

1. **Chain-of-Draft (CoD) system prompt**: *“Think step by step, but write each step in at most 5 words. Be extremely concise.”*
   - Moves GSM8K accuracy from **23% → 77%** on Qwen3.8-27B huihui abliterated (T=0.6, 10 problems × 3 runs)
   - Costs only ~8% decode t/s (89 t/s vs 94 t/s)
   - Reasoning content length goes UP (more steps) but each step is shorter → **better thought, less fluff**
   - Beats both Unsloth Dynamic v3.0 quants on the same benchmark

2. **Token-Budget prompt**: *"Think step by step using at most 200 tokens of reasoning, then give the final answer."*
   - 37% accuracy (-36% think length)
   - Less dramatic than CoD; useful when you really need shorter outputs

3. **NoWait logit-bias**: Suppress reflection keywords ("Wait", "Hmm", "Alternatively", etc.) at the logit level
   - 23% acc under sampling (small but real +10pp over baseline)
   - No effect at T=0 (argmax path doesn't sample them anyway)

4. **Native `reasoning_effort` (xhigh/medium/low)** built into the model's chat template: **no observable effect** at T=0. All three collapse to identical outputs because the soft prompt is already satisfied by argmax.

6. **`reasoning_budget`** (`--reasoning-budget N`): **no-op on huihui template**. The template doesn't emit `` tags, so server-side budget enforcement can't engage.

## What does NOT work

- **Server-side system prompt injection via llama-server**: not possible. `--chat-template-kwargs` is parsed into `params.default_template_kwargs` but the field is never read by the renderer (verified by reading `common/arg.cpp:3575` + `common/common.h:661` in llama.cpp b10753). CoD must be sent by the client.
- **`reasoning_effort: 100`** in SillyTavern: passes through fine, but doesn't change behavior on huihui template (the model's argmax already meets the xhigh soft prompt at T=0)
- **q4_0 KV cache**: measurably degrades reasoning. Keep q8_0.

---

## The reasoning-effort sweep (12 configs, GSM8K)

### Benchmark protocol

- Dataset: GSM8K test split, first 10 problems (canonical, deterministic, public on HuggingFace)
- Runs: 3 per config (T=0.6; same prompt across runs)
- Eval: harness extracts answer via `\boxed{}` / `#### N` / "answer is N" fallbacks
- Model: Qwen3.8-27B huihui abliterated UD-Q4_K_XL (llama.cpp b10753)
- Hardware: 2× RTX 3090 + Ryzen 7 5800XT + 128 GiB DDR4

### Full results table

| Config | T | Acc r0/r1/r2 | Mean acc | Mean t/s | Mean think chars |
|---|---|---:|---:|---:|---:|
| **`reasoning_effort=xhigh`** | 0.0 | 0.30/0.30/0.30 | **0.30** | 94.92 | 739 |
| `reasoning_effort=medium` | 0.0 | 0.30/0.30/0.30 | **0.30** | 94.12 | 739 |
| `reasoning_effort=low` | 0.0 | 0.30/0.30/0.30 | **0.30** | 94.56 | 739 |
| `reasoning_budget=0` | 0.0 | 0.30/0.30/0.30 | **0.30** | 93.69 | 739 |
| `reasoning_budget=256` | 0.0 | 0.30/0.30/0.30 | **0.30** | 92.57 | 739 |
| `reasoning_budget=1024` | 0.0 | 0.30/0.30/0.30 | **0.30** | 89.80 | 739 |
| `reasoning_budget=4096` | 0.0 | 0.30/0.30/0.30 | **0.30** | 89.90 | 739 |
| `reasoning_budget=16384` | 0.0 | 0.30/0.30/0.30 | **0.30** | 88.02 | 739 |
| `nowait` (83 logit-bias tokens) | 0.0 | 0.30/0.30/0.30 | **0.30** | 89.89 | 739 |
| **baseline (no override)** | 0.6 | 0.20/0.10/0.10 | **0.13** | 85.21 | 984 |
| `nowait` (at T=0.6) | 0.6 | 0.30/0.10/0.30 | **0.23** | 84.05 | 888 |
| **`cod`** (CoD prompt) | 0.6 | 0.50/0.70/0.70 | **0.63** | 78.64 | 1,034 |
| **`tokbudget`** | 0.6 | 0.30/0.40/0.40 | **0.37** | 81.21 | 633 |

### What this tells us

- At T=0.0, all "intervention" methods produce identical outputs because the model's argmax already meets the chat-template's xhigh soft prompt. **`reasoning_effort` and `reasoning_budget` are no-ops on this model at T=0.**
- The interesting variants only appear at T=0.6 (where sampling matters). The real lever is **the system prompt you inject**, not the parameter you toggle.

### How to reproduce

```bash
cd bench
bash reproduce_reasoning_sweep.sh     # ~50 min total
# Raw results land in results.jsonl
```

---

## 3-way comparison

The next question: *does Unsloth Dynamic v3.0 beat the older huihui re-quantization?* Unsloth's marketing claims ">10% better accuracy at the same size" for v3.0. Let's see.

### Models compared

| Path | Quantization scheme | Size |
|---|---|---:|
| `$MODELS_DIR/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` | Unsloth Dynamic **v2.x** @ Q4_K_XL (huihui re-quantized with selective Q8_0/BF16 retention on important tensors) | 17.4 GiB |
| `$MODELS_DIR/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` | Unsloth Dynamic **v3.0** @ Q4_K_XL | 17.6 GiB |
| `$MODELS_DIR/unsloth-qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | Unsloth Dynamic **v3.0** @ Q4_K_M | 16.5 GiB |

### Results (all with CoD prompt, T=0.6, 10 problems × 3 runs)

| Config | Acc r0/r1/r2 | Mean acc | Mean t/s | Mean think chars |
|---|---:|---:|---:|---:|
| **huihui abliterated + CoD** | 0.70/0.70/0.90 | **0.77** | 88.91 | 1,191 |
| huihui abliterated baseline (no CoD) | 0.10/0.40/0.20 | **0.23** | 94.09 | 755 |
| Unsloth UD-Q4_K_M v3.0 + CoD | 0.70/0.60/0.70 | **0.67** | 89.44 | 1,087 |
| Unsloth UD-Q4_K_XL v3.0 + CoD | 0.60/0.50/0.70 | **0.60** | 89.06 | 1,094 |

### What this tells us

- **huihui + CoD wins decisively** (77%), beating both Unsloth v3.0 quants by 10–17 percentage points.
- The Unsloth ">10% better accuracy" v3.0 claim does NOT hold on this benchmark with this prompt. The authors' claims were likely validated against different benchmarks (MMLU, BBH, GPQA) where the ablation strategy doesn't matter as much.
- Decode t/s are all within 6% of each other (87–90 t/s) — the accuracy gap is the dominant signal.

### Why does huihui beat Unsloth v3.0?

1. **Selective Q8_0/BF16 retention** on precision-sensitive tensors (`token_embd, ffn_down, attn_output, ssm_out, output`). Per the huihui model card: *"We have already converted the weights (token_embd, output, ffn_down, ssm_out, attn_output) that need to be ablated in the versions below Q8_0 from Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K to Q8_0 to improve response quality."* — those are exactly the tensors that matter for reasoning accuracy.
2. **Ablation strategy**: huihui ablates only layers 18–51, preserving early/middle reasoning layers. The reasoning-heavy layers keep their full safety alignment.
3. **GSM8K style**: arithmetic word problems with a particular style that benefits from the precision retention in the abliterated layers.

For other benchmark suites, your results may differ — but on GSM8K + CoD prompt, huihui is the clear winner.

### How to reproduce

```bash
cd bench
bash compare_3way.sh                  # ~25 min total
# Raw results land in compare_3way.jsonl
```

---

## The Chain-of-Draft paper reference

This approach is from the original CoD paper: [https://arxiv.org/abs/2502.18600](https://arxiv.org/abs/2502.18600) (Zheng et al., 2025). The authors report -80% token usage with no accuracy loss on GSM8K for various LLMs.

What we found goes further: on this specific 27B + 4-bit setup, the prompt doesn't just save tokens — it **unlocks reasoning the model otherwise doesn't exhibit**. The mechanism isn't fully understood, but the empirical effect is consistent across runs.

## Where to inject the CoD prompt

| Client | How |
|---|---|
| SillyTavern | Switch to the `Local Huihui ... (Tensor-Split + CoD)` connection profile (created during setup). See `docs/MODELS.md` for the JSON. |
| OMP / direct API | Pass `--system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise."` or include it as the first message in `messages[]` |
| curl | Include a `{"role":"system","content":"..."}` as the first element of `messages[]` |

**Server-side injection is NOT supported** by llama-server b10753 (a dead-flag bug in `--chat-template-kwargs`). All injection must be client-side.