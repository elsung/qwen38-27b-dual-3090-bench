# Per-Model Launch Recipes

Each section gives the verified launcher invocation + measured speeds for a single model. All recipes assume the toolchain from [SETUP.md](SETUP.md).
## Sampler settings reference (thinking vs instruct mode)
Per the [Qwen3.8-27B model README](https://huggingface.co/Qwen/Qwen3.8-27B), the recommended samplers are:
| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty | best for |
|---|---:|---:|---:|---:|---:|---:|---|
| **Thinking** (math/agentic) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 | GSM8K-style problems, code, agentic |
| **Instruct** (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 | Chat, Q&A, roleplay |
Use **thinking mode + CoD prompt** for math/agentic (SilentlyTavern "use_sysprompt: true" + the CoD prompt we ship). Use **instruct mode + no CoD** for prose/roleplay.
---
## Qwen3.8-27B huihui-ai abliterated (UD-Q4_K_XL)
**The recommended default for math / agentic / roleplay / OMP.**
```bash
# Launcher: scripts/start-huihui-27b-abliterated.sh
# Underlying invocation:
$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server \
  -m $MODELS_DIR/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf \
  -ngl 99 \
  -fa on \
  -c 262144 \
  --split-mode tensor --tensor-split 1,1 \
  --parallel 1 --jinja \
  --host 0.0.0.0 --port 8091 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ctk q8_0 -ctv q8_0
```
**Measured (dual 3090, Ryzen 7 5800XT, 128 GiB DDR4)**:
| State | Mean decode t/s |
|---|---:|
| Empty KV cache | ~88–99 |
| Populated @ 16K tokens | ~99 |
| Populated @ 70K tokens | ~60 |
**Why this quant?** The huihui re-quantization keeps `token_embd, ffn_down, attn_output, ssm_out, output` at Q8_0/BF16 (the precision-sensitive tensors). For math reasoning, that beats both Unsloth Dynamic v3.0 quant schemes at the same file size. See [REASONING.md](REASONING.md#3-way-comparison).
**Do not use q4_0 KV cache** — measurably degrades reasoning accuracy.
### SillyTavern integration (recommended)
Switch to the `Local Huihui Qwen3.8-27B Abliterated (Tensor-Split + CoD)` connection profile at `$ST_DIR/data/default-user/OpenAI Settings/`. Profile contents:
```json
{
  "chat_completion_source": "custom",
  "custom_url": "http://127.0.0.1:8091/v1",
  "openai_model": "huihui-qwen38-27b-abliterated",
  "temperature": 0.85,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.01,
  "repetition_penalty": 1.1,
  "openai_max_context": 32768,
  "openai_max_tokens": 2048,
  "use_sysprompt": true,
  "system_prompt": "Think step by step, but write each step in at most 5 words. Be extremely concise.",
  "reasoning_effort": 100,
  "stream_openai": true,
  "show_thoughts": true,
  "include_reasoning": true
}
For prose / roleplay (no CoD), use the original profile without `use_sysprompt: true` (the model's default behavior is fine for verbose creative writing).
## Qwen3.8-27B FP8 dense via vLLM

> **Correction 2026-09-04**: previously mislabeled "Flash-Next". The 28.75 GiB FP8 checkpoint is the dense 27B. The real Flash-Next is a 176B MoE (124 GiB GGUF) that runs at ~4 t/s on llama.cpp only.
**The recommended runtime for Flash-Next.** vLLM's expert parallelism + Paged Attention is 15× faster than llama.cpp on this hardware.
# Launcher: scripts/start-qwen38-flashnext.sh
# Uses vllm-openai Docker image, FP8 model, MTP spec decoding
docker run -d --rm --gpus all --network host \
  --name vllm-flashnext \
  -v $MODELS_DIR:/models:ro \
  -e VLLM_USE_V1=1 \
  vllm/vllm-openai:latest \
    --model /models/qwen38-flash-next/Qwen3.8-Flash-Next-FP8 \
    --served-model-name /models/fp8 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --speculative-model /models/qwen38-flash-next/Qwen3.8-Flash-Next-FP8 \
    --speculative-method qwen3_5_mtp \
    --num-speculative-tokens 3 \
    --port 8091
**Measured**: **60.86 t/s** mean decode at batch=1 (dense 27B FP8 — NOT Flash-Next), prompt<8K. Sweet spot is `--max-num-batched-tokens 4096`; bumping to 8192 doesn't help at batch=1.
**Do not use llama.cpp** for this model on dual 3090 — the per-tensor allocator OOMs at load time unless you CPU-offload experts. With `--n-cpu-moe 40 --moe-expert-cache 16` you get ~4 t/s (15× slower than vLLM).
## Qwen3.8-27B (Unsloth Dynamic v3.0)
For users who want the official Unsloth quant (better than v2.x in the authors' benchmarks, but our GSM8K bench shows huihui abliterated beats it).
# Launcher: scripts/start-unsloth-qwen38-27b.sh
# env: LLAMA_MODEL=UD-Q4_K_XL or UD-Q4_K_M
LLAMA_MODEL=UD-Q4_K_XL bash scripts/start-unsloth-qwen38-27b.sh
Both quants are functionally identical to the huihui launcher except for `-m` and `--tensor-split` (we use 32768 ctx instead of 262144 because the unsloth build is smaller / not used in long-context tests).
## GLM-5.3-Flash (UD-IQ3_XXS) — not yet loadable
**Status**: GGUF shards on disk (~120 GiB), upstream llama.cpp PR #27752 adds `glm5next` architecture support (open, mergeable, 2,395 lines). Expected ~30 min to merge + build + bench.
The recommended recipe (once available) is identical to the Flash-Next llama.cpp recipe:
  -m $MODELS_DIR/glm53-flash-udiq3/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  -ngl 99 -fa on -c 8192 \
  --n-cpu-moe 40 --moe-expert-cache 16 \
  -ctk q8_0 -ctv q8_0 \
  --md <path-to-mtp-draft>  # required for MTP
Skip until PR #27752 lands.
## Quick reference: which model when
| Workload | Best choice | Why |
|---|---|---|
| SillyTavern roleplay (no math) | huihui baseline (no CoD) | CoD's 5-word drafts clip verbose prose |
| SillyTavern math / agentic chat | huihui + CoD (ST profile) | +50pp GSM8K accuracy |
| OMP coding / reasoning / agentic | huihui + CoD (pass `--system-prompt`) | Same +50pp boost |
| Multi-user concurrent (>=2 batches) | Flash-Next vLLM | Best at batch > 1 |
| 262K context roleplay | huihui only | Flash-Next tested only to 16K |
| VRAM-constrained single-GPU | huihui UD-Q4_K_XL on one card | Single-GPU flash-attention works |
## SillyTavern + OMP profiles (per model + reasoning style)
We ship two SillyTavern connection profiles pre-configured for these models. Both inject the
Chain-of-Draft system prompt via `use_sysprompt: true`. Both use the **official Qwen3.8 thinking-mode
samplers** (T=1.0, top_p=0.95, top_k=20, no presence penalty).
| Profile | URL | Model field | OpenAI max context |
|---|---|---|---:|
| `Local Huihui ... (Tensor-Split + CoD)` | `http://127.0.0.1:8091/v1` | `huihui-qwen38-27b-abliterated` | 32,768 (huihui native 262K supported, just bump) |
| `Local Qwen3.8-27B FP8 Dense (vLLM + CoD)` — *renamed from mislabeled 'Flash-Next'* | `http://127.0.0.1:8091/v1` | `/models/fp8` | 16,384 default; bump to 262K or 1M via launcher |
Both profiles live at `$ST_DIR/data/default-user/OpenAI Settings/`:
- `Local Huihui Qwen3.8-27B Abliterated (Tensor-Split + CoD).json`
- `Local Qwen3.8-Flash-Next (FP8 vLLM + CoD).json`
The CoD prompt injected in both:
> Think step by step, but write each step in at most 5 words. Be extremely concise.
For OMP on a Tailscale-connected device:
# huihui (recommended for math/agentic)
omp --base-url http://$TAILSCALE_IP:8091/v1 \
  --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise."
# Flash-Next with 1M context (launcher must have VLLM_CONTEXT=1048576 set)
  --max-context 1000000 \
For curl / direct API: include `{"role":"system","content":"..."}` as the first message in `messages[]`.
