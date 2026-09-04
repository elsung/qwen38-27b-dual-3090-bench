# Observational Memory vs YaRN — Comparison Results (2026-09-04)

> Tested 2026-09-04 on huihui Qwen3.8-27B abliterated (native 262K). Companion to `YARN_COMPARISON_2026-09-04.md`.

## TL;DR

**The simulated observational-memory test was INVALID** — and it revealed a real gotcha:

1. **The memory digests were empty.** huihui's thinking mode (`<think>...</think>`) consumed the entire `max_tokens` budget inside `reasoning_content` before emitting any visible bullet points. Both test arms ("baseline" and "obs-mem") therefore ran with identical (empty) memory context.
2. **Baseline accuracy variance is very high at N=30**: the same config (native 262K + CoD) scored **77%** in the earlier session and **50%** in this one. ±27pp run-to-run variance means none of our N=30 GSM8K numbers are statistically solid.
3. **Real finding — huihui is a poor Observer-model candidate for pi-observational-memory**: any Observer/Reflector agent running on huihui must either (a) disable thinking mode for the observer call, or (b) budget 2-4× more tokens so thinking completes AND bullets are emitted.

## What was tested

Simulated the effect of pi-observational-memory's memory layer on single-turn GSM8K:

| Arm | System prompt | Result |
|---|---|---|
| baseline | CoD prompt only | 50% (15/30), 86 t/s |
| obs-mem | CoD + "[Memory Digest]" header | 50% (15/30), 88 t/s |
| YaRN-best (prior sweep) | CoD, 524K + yarn attn=1.0 | 80% (24/30), 78 t/s |

**Caveat: obs-mem arm digests were empty** → identical to baseline. No signal.

**Caveat 2: baseline variance** → 50% here vs 77% prior session (same config). The YaRN 80% number came from yet another run. Cross-run comparisons at N=30 are unreliable.

## Root cause: thinking mode eats the Observer output

```
POST /v1/chat/completions
  system: "You are an Observer agent. Compress into 3-5 dense bullets..."
  max_tokens: 300

response:
  content: ''                      ← EMPTY
  reasoning_content: "We need answer user's request: compress chat
    history into 3-5 dense bullet points... Need compress. Decisions
    made: Assist..."              ← truncated at 300 tokens mid-think
```

The model entered thinking mode, reasoned about compressing, and hit the token cap before writing a single bullet. Every digest was `''`.

### Fixes for a valid test (future work)

1. **Budget 1024+ tokens** for the Observer call (thinking + output)
2. **Fallback to `reasoning_content`** when `content` is empty and extract bullets from the tail
3. **Better: disable thinking for Observer calls** — send `extra_body: {"enable_thinking": false}` or use `/no_think` in the prompt (Qwen3 supports this)
4. **Test on MULTI-TURN sessions** — observational memory is designed for accumulating context across turns; single-turn GSM8K doesn't exercise it

## How this maps to real pi-observational-memory on OMP

`pi-observational-memory` (v3.0.4) is installed and enabled on this box's OMP. Its Observer/Reflector agents make their own model calls with their own token budgets, and it works around thinking-mode models by design (the plugin config has a `model` override for exactly this reason).

**For OMP on this server (Tailscale clients included)**:
```bash
# OMP uses huihui at $TAILSCALE_IP:8091. The plugin is enabled by default
# after `omp plugin install npm:pi-observational-memory`.
# Configure a separate observer model (recommended: a fast non-thinking model)
omp plugin config pi-observational-memory --set model.provider=openai-compatible
omp plugin config pi-observational-memory --set model.id=<fast-model>
```

**Key knobs** (from `src/config.ts`):
| Setting | Default | Meaning |
|---|---|---|
| `observeAfterTokens` | ~30K | Trigger Observer when history exceeds this |
| `reflectAfterTokens` | larger | Trigger Reflector (distill reflections) |
| `compactAfterTokens` | ~81K | Trigger compaction; memory already prepared |
| `compactAfterTokensMode` | calibrated/ratio | ratio scales to model's context window |
| `model` | unset (uses session model) | **Set a non-thinking model for reliability** |

**For SillyTavern**: pi-observational-memory is a Pi/OMP extension and does not hook SillyTavern. ST equivalents: built-in Summarize extensions, or vector storage (World Info + embeddings). Different mechanism, same goal.

## Statistical honesty note

All our GSM8K numbers (10 problems × 3 runs = 30 trials) have huge variance:

| Config | Session A | Session B | Δ |
|---|---:|---:|---:|
| native 262K + CoD | 77% | 50% | -27pp |
| linear 524K + CoD | 53% | 70% | +17pp |

Before publishing any accuracy claim, run **N ≥ 150** (10 problems × 15 runs or 50 problems × 3 runs). The *ordering* of methods has been stable in our data (CoD >> baseline; yarn-attn1 > linear > yarn-attn0 >> ntk), but absolute percentages move ±15-27pp between sessions.

## Files

- `/tmp/yarn-vs-obs.py` — the (invalid) comparison script; keep for the fix recipe in its docstring
- `/tmp/obs-mem-vs-yarn.jsonl` — raw records from both arms
- `bench/yarn-bench/*.jsonl` — the valid YaRN sweep data (4 modes)

## Recommended next steps

1. **Re-run obs-mem test with thinking disabled on Observer calls** (`enable_thinking: false` in the observer request). ~15 min.
2. **Re-run with N=150** for stable accuracy numbers on the top 3 configs (native, yarn-attn1, obs-mem). ~1.5 hr.
3. **Test real OMP sessions**: start a long OMP session against huihui with the plugin enabled, work for 100K+ tokens, verify continuity across compaction. This is the true use case — single-turn GSM8K isn't.
4. **Keep yarn-attn1 (80%) as the production long-context setting** until obs-mem proves itself in a valid test.