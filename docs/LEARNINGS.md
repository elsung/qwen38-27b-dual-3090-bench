# LEARNINGS — Distilled Best Practices

> A condensed list of what we learned across several weeks of benchmarking. Top-down: most impactful first.

## The Big 3 (worth doing before anything else)

### 1. Inject the Chain-of-Draft prompt for reasoning tasks

**"Think step by step, but write each step in at most 5 words. Be extremely concise."**

- Moves GSM8K from 23% → 77% on Qwen3.8-27B huihui abliterated
- Cost: -8% decode t/s
- No model changes, no fine-tuning, no extra VRAM
- See [REASONING.md](REASONING.md)

### 2. Use tensor-split 1,1 on dual 24-GiB GPUs, q8_0 KV, draft-MTP n_max=3

The combination of:
- `--split-mode tensor --tensor-split 1,1` (use both GPUs, no P2P needed)
- `-ctk q8_0 -ctv q8_0` (KV cache precision)
- `--spec-type draft-mtp --spec-draft-n-max 3` (Multi-Token Prediction draft head)

gets you the best decode t/s for dense 27B models on dual 3090s (~99 t/s empty / ~60 t/s populated @70K).

**Do NOT use q4_0 KV** — measurably degrades reasoning accuracy.

### 3. Match the runtime to the workload

| Workload | Runtime |
|---|---|
| Flash-Next (any batch>1, MoE) | **vLLM Docker** with `--speculative-method qwen3_5_mtp` |
| Flash-Next (CPU-MoE fallback) | llama.cpp Inovello with `--n-cpu-moe 40 --moe-expert-cache 16` |
| Qwen3.8-27B (any) | llama.cpp Inovello tensor-split + draft-MTP |

vLLM is 15× faster than llama.cpp for Flash-Next. llama.cpp is the only viable option for Qwen3.8-27B (vLLM has no MTP for it in our build).

## Top 8 gotchas

1. **`reasoning_effort=xhigh/medium/low` doesn't change behavior at T=0** on the Qwen3.8 chat template. Use the CoD prompt instead.
2. **`reasoning_budget=N` is a no-op** on templates that don't emit `` tags. Use the Token-Budget prompt instead.
3. **`--chat-template-kwargs` is a dead-flag in llama-server b10753** — parsed but never reaches the renderer. Inject CoD client-side.
4. **Per-tensor allocator OOMs on qwen4exp models** under llama.cpp unless you CPU-offload experts. Use `--n-cpu-moe 40` early.
5. **Draft-MTP requires the MTP head in the same GGUF** as the main model. The `--spec-type draft-mtp` flag won't work if the head isn't present.
6. **Unsloth Dynamic v2.x huihui beats Unsloth Dynamic v3.0** on GSM8K + CoD. v3.0's marketing claims don't hold on this benchmark.
7. **PCIe tuning has no effect on GPU-compute-bound or RAM-bandwidth-bound workloads.** Don't chase it.
8. **Thinking exhausts client max_tokens → "empty stop"** on agentic/tool-calling
   clients (OMP): hard turns burn 2-14K tokens in `<think>`, so `content=""`,
   no tool_calls, `finish_reason=length` → client retry cap. Fix per-request with
   `"chat_template_kwargs": {"enable_thinking": false}` in the body (works on
   b10753 even though the CLI flag is dead) or max_tokens ≥ 8192. Verified 0/8
   empty over an 8-turn agentic loop; 3/8 empty at max_tokens=900. Also: the
   server 500s if history contains a malformed tool_call — fresh conversation
   is the only cure.

## Top 5 use-case decisions

1. **SillyTavern roleplay** → huihui baseline (no CoD), temp 1.0, top_p 0.95
2. **SillyTavern math/agentic chat** → huihui + CoD (the CoD profile), temp 0.85, top_p 0.95
3. **OMP coding/agentic** → huihui + CoD via `--system-prompt`
4. **Multi-user concurrent** (>=2 batches) → Flash-Next vLLM
5. **Long-context roleplay** (>=70K tokens) → huihui only; Flash-Next tested only to 16K

## What we measured but don't recommend

- **NoWait logit-bias on reflection tokens**: small win (+10pp) under sampling, no win at T=0. Skip unless you're sampling.
- **Token-Budget prompt**: less effective than CoD on this model. Use CoD unless you specifically need shorter outputs.
- **`reasoning_budget=0`**: no-op. If you want "immediate answer, no thinking", use `enable_thinking=false` in the chat template kwargs (when supported).

## What we didn't measure (and probably should)

- GLM-5.3-Flash (waits on upstream PR #27752)
- 50+ problem GSM8K for proper statistical significance (we used N=10)
- MATH-500 (harder math benchmark, likely a bigger spread between methods)
- HumanEval / MBPP for code accuracy
- Multi-GPU scaling (we only have 2× 3090; 4× might show different bottlenecks)

## Reproducing any of this

```bash
# Install
bash scripts/setup_paths.sh   # sets $MODELS_DIR etc.

# Reasoning-effort sweep (12 configs, ~50 min)
cd bench && bash reproduce_reasoning_sweep.sh

# 3-way model comparison (~25 min)
cd bench && bash compare_3way.sh

# Re-run a single config
python3 bench/gsm8k_harness.py --label my-run --reasoning-effort xhigh --runs 3
```

All numbers in this repo come from these scripts.