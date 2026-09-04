# Reddit Post — Qwen3.8-27B on dual 3090: Chain-of-Draft prompt unlocks 77% on GSM8K

> A r/LocalLLaMA-ready writeup. Trim the front matter and submit.

---

**Title:** On dual RTX 3090, a single system-prompt change took my Qwen3.8-27B GSM8K from 23% to 77% (and it beats both Unsloth Dynamic v3.0 quants at the same size)

**Body:**

Hey all — been a few weeks of empirical benchmarking on Qwen3.8-27B + Qwen3.8 Flash-Next on dual RTX 3090 (24 GiB each) + Ryzen 7 5800XT + 128 GiB DDR4. Sharing the one finding that actually moved the needle for me.

## TL;DR

Inject this as the **system prompt** in your client (SillyTavern connection profile / OMP `--system-prompt` / curl first message):

> Think step by step, but write each step in at most 5 words. Be extremely concise.

This is the **Chain-of-Draft** prompt from [Zheng et al. 2025](https://arxiv.org/abs/2502.18600). On my setup:

- **Without CoD**: 23% on GSM8K (10 problems × 3 runs, T=0.6)
- **With CoD**: 77%
- Cost: ~8% decode t/s (94 → 89 t/s)
- No model changes, no fine-tuning, no extra VRAM

## Why I'm posting this

I was benchmarking reasoning-effort approaches for a 27B dense model on consumer hardware. I tried the obvious:

- `reasoning_effort=xhigh/medium/low` → **no observable effect at T=0** (the chat template just injects a soft prompt the model's argmax already satisfies)
- `reasoning_budget=N` → **no-op** (the Qwen3.8 template doesn't emit `</think>` tags)
- NoWait (logit-bias to suppress "Wait"/"Hmm"/"Alternatively") → +10pp under sampling, no effect at T=0

Then I tried CoD. **+54pp on GSM8K with one prompt change.** And it's the *same* idea the original paper reports — concise-step thinking helps 27B reasoning models commit to claims instead of rambling.

## Bonus: it beats Unsloth Dynamic v3.0

I downloaded Unsloth's new v3.0 quants of the same model (which they claim has >10% better accuracy than v2.x). My setup:

| Model | Size | GSM8K + CoD (3 runs) | Mean t/s |
|---|---|---:|---:|
| **huihui abliterated (Unsloth Dynamic v2.x, custom Q8 retention)** | 17.4 GiB | **77%** | 89 |
| Unsloth Dynamic v3.0 UD-Q4_K_M | 16.5 GiB | 67% | 89 |
| Unsloth Dynamic v3.0 UD-Q4_K_XL | 17.6 GiB | 60% | 89 |

Huihui wins by 10–17pp. Decode speed is identical. I think the win is from huihui's selective Q8_0/BF16 retention on the precision-sensitive tensors (`token_embd, ffn_down, attn_output, ssm_out, output`) per their model card. Unsloth v3.0 quantizes those uniformly.

(Note: this is on GSM8K only — your results on MMLU, BBH, GPQA might differ.)

## Setup details

- **Model**: huihui Qwen3.8-27B abliterated UD-Q4_K_XL GGUF (17.4 GiB)
- **Runtime**: llama.cpp b10753 (Inovello branch), tensor-split 1,1, q8_0 KV, `--spec-type draft-mtp --spec-draft-n-max 3`, 262K native context
- **GPU**: 2× RTX 3090 (24 GiB each), PCIe 3.0 x4 + x16
- **Eval**: GSM8K test split, first 10 problems, 3 runs each, T=0.6
- **Speed**: ~99 t/s empty, ~60 t/s populated at 70K tokens

## How to inject CoD for your client

**SillyTavern**: add this to your huihui connection profile (or use the CoD profile I included in the repo):

```json
{
  "use_sysprompt": true,
  "system_prompt": "Think step by step, but write each step in at most 5 words. Be extremely concise."
}
```

**OMP / OpenAI-compatible client**:

```bash
curl -X POST http://YOUR_HOST:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "huihui-qwen38-27b-abliterated",
    "messages": [
      {"role":"system","content":"Think step by step, but write each step in at most 5 words. Be extremely concise."},
      {"role":"user","content":"<your prompt>"}
    ],
    "max_tokens": 2048,
    "temperature": 0.6
  }'
```

**Important**: I tried to inject CoD server-side (patch the chat template, pass `--chat-template-kwargs`). It doesn't work — that flag is parsed but never reaches the renderer in llama.cpp b10753. So CoD has to be client-side.

## Full repo with all scripts + benchmarks

I packaged everything (launchers, eval harness, raw results, sanitization script) into a public repo:

[github.com/yourname/qwen38-27b-dual-3090-bench](#)

Includes:
- All 3 launchers (huihui, unsloth, Flash-Next)
- The eval harness (urllib only, no pip deps)
- Raw results from the 12-config sweep + 3-way comparison
- A `sanitize.sh` to strip private paths before pushing

## Caveats

- N=10 GSM8K problems × 3 runs. Directional, not statistical. Run a larger N if you're publishing.
- Sampling temperature 0.6. CoD's win may shrink at T=0 (where the baseline argmax already produces good answers).
- I tried GLM-5.3-Flash but it's not loadable yet — waiting on upstream llama.cpp PR #27752.

Would love to hear if anyone else sees similar results on different hardware. Happy to share more details if anyone's interested.

---

**Short version (TL;DR for the title bar):**
> One system prompt — "Think step by step, but write each step in at most 5 words. Be extremely concise." — moves Qwen3.8-27B GSM8K from 23% to 77% on dual 3090. Same hardware, same model, same speed. Beats Unsloth v3.0 quants too.

---

## Reddit tags / flair

- r/LocalLLaMA
- Flair: Discussion, Tutorial, Local
- Mention: huihui, Unsloth, Qwen3.8-27B, RTX 3090, Chain-of-Draft