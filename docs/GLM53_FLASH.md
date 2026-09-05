# GLM-5.3-Flash (ox-alpha) on 2x RTX 3090 — setup status

Status: **model on disk, runtime built, launcher written — NOT yet run**
(machine reserved for huihui inference). Launcher: `~/Desktop/start-glm53-flash.sh`.

## What it is

320B MoE, **18B active**, multimodal, hybrid sparse + linear attention
(Z.ai). Claims to rival Claude Opus 4.8 on coding/agentic (Terminal-Bench 2.1:
84.3, DeepSWE 63.4 per unsloth). Max context 1,048,576.

- Docs: <https://unsloth.ai/docs/models/glm-5.3-flash>
- Speed post: <https://www.reddit.com/r/unsloth/comments/1w736ww/> (3.3x faster
  inference via their llama.cpp PR #27754: faster decode path + MTP)

## Local inventory (2026-09-04)

| Item | State |
|---|---|
| GGUF | `$WORKSPACE/models/glm53-flash-udiq3/UD-IQ3_XXS/` 4 shards, 120.36 GB total (arch `glm5next`) |
| Runtime | unsloth fork `glm5next/upstream` @ `629b505`, built at `$WORKSPACE/unsloth-glm53/build-3090` (sm86, CUDA) |
| MTP | built into fork via `--spec-type draft-mtp` — **no separate draft file exists** (checked HF repo) |
| Mainline llama.cpp | **cannot load it** — no `glm5next` arch in b96806d96 |

## Why mainline + our usual stack won't work

- Arch `glm5next` is unsloth-fork-only until PR #27754 lands.
- The fork has **no `--moe-expert-cache`** (that's Inovello's PR). Expert offload
  options: `-ot ...=CUDA_Host` regex or `--n-cpu-moe N`.
- Unsloth's 3.3x numbers (1x B200, UD-IQ1_S): tg32 62.8 → 63.1 t/s; **@64K ctx
  20.7 → 49.0 t/s**; with MTP n=2 @4K prompt: 58.6 → **86.5 t/s**. n=3+ slower.

## Memory budget on this box (125 GB RAM + 48 GB VRAM)

IQ3_XXS = 120.4 GB total; unsloth's table says 3-bit needs 128-150 GB
(RAM+VRAM) — we have ~173 GB, so it *should* fit with experts on host and
attention/dense on GPU, but it's tight (system + KV + compute buffers on top).
Fallback if OOM: `GLM_CTX=32768`, then a smaller quant (UD-IQ2_XXS 101.8 GB,
76.3% top-1 vs IQ3's 81.6% — needs ~100 GB free disk we don't currently have).

## Expected performance here

18B active (3x Flash-Next's 6B) with only dual-channel DDR4 for the bulk of
weights → expert-heavy reads are bandwidth-bound. The fork's decode path + MTP
n=2 is the lever; unsloth's own guidance is the speedup grows with context.
Realistically this lands in "usable for patient/agentic work", not chat-speed —
measure before judging.

## Launcher knobs

```
GLM_CTX=65536 GLM_MTP=1 GLM_N_MAX=2     # defaults
GLM_MTP=0                                # A/B without MTP, or if draft errors
GLM_NCMOE=48                             # use --n-cpu-moe 48 instead of -ot regex
GLM_PORT=8091                            # one server at a time (RAM)
```
