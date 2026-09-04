# Concurrency & Real-World Usability (OMP / SillyTavern)

> Measured 2026-09-04 on the production config: huihui UD-Q4_K_XL, **YaRN 524K** (scale=2, attn_factor=1.0), q8_0 KV, draft-MTP 3, `--parallel 4` (4 slots × 131K ctx each).

## TL;DR

| Setup | Per-session context | True aggregate | Per-stream speed | 128-token turn |
|---|---:|---:|---:|---:|
| **1 OMP session** (parallel=1) | **524K** | — | **~60-80 t/s** | ~2s |
| 2 concurrent (parallel=2) | 262K | 73 t/s | 36 t/s | ~3.5s |
| **4 concurrent (parallel=4)** | **131K** | **118 t/s** | **29 t/s** | ~4.3s |
| 4 concurrent, queued (parallel=1) | 524K (serialized) | ~70 t/s | 70 t/s each, but waits | 2s + queue |

**The trade-off**: context window divides across slots. `--parallel 4` with `-c 524288` gives each session **131K** — enough for any single OMP session or ST chat, but not 500K each. To get 262K per session at c=2, just run `HUIHUI_PARALLEL=2`.

**Recommendation**: for 1-2 OMP instances + ST, run `HUIHUI_PARALLEL=2` (262K each, 36 t/s under full load). For solo work, keep the default `HUIHUI_PARALLEL=1` (full 524K, ~60-80 t/s).

## How context divides (verified empirically)

Server log with `--parallel 4 -c 524288`:
```
n_slots = 4, n_ctx_slot = 131072
```

`-c` sets the **total KV pool**; each slot gets `ctx ÷ parallel`. The 524K pool is the hardware ceiling (786K+ crashes on dual 24 GiB — see `LONG_CONTEXT_BENCH_2026-09-04.md`).

| HUIHUI_PARALLEL | Per-session ctx | Use case |
|---:|---:|---|
| 1 (default) | 524K | Solo OMP / one long ST chat |
| 2 | 262K | OMP + ST simultaneously, or 2 OMP |
| 4 | 131K | 4 light sessions (still huge by ST standards) |

## Measured speeds at parallel=4 (yarn 524K)

Single-stream (slot solo):
| Prompt | Decode | Prefill (cold) |
|---|---:|---:|
| 1K | 70-93 t/s | 550-760 t/s |
| 8K | 62-71 t/s | 760 t/s |
| 32K | 58-68 t/s | 742 t/s |

Concurrent waves (4 × 1K-prompt, 128-token generations):
| Concurrency | Wall for wave | True aggregate | Per stream |
|---|---:|---:|---:|
| c=2 | 3.5s | 73 t/s | 36 t/s |
| c=4 | 4.3s | 118 t/s | 29 t/s |

Prefill under concurrency: cold 32K prefill still runs ~740 t/s on an idle slot; concurrent prefills share the GPU and stretch proportionally.

## What to expect on OMP (realistic, not peak)

Peak decode is 80-92 t/s. But an OMP coding turn is not pure decode:

- **Thinking tokens count as generation** — a "400-token reply" is often 600-900 total tokens with reasoning
- **Prefill grows each turn** — llama-server's LCP prefix cache absorbs most of it (we observed `cached_tokens` > 90% on repeated context), so incremental prefill stays fast
- **Tool calls serialize** — bash/file round-trips dominate wall time, not the GPU

**Realistic per-turn latency on this box (solo, parallel=1):**
| Turn type | Tokens (incl. thinking) | Wall |
|---|---:|---:|
| Quick edit + short reply | ~300 | **3-5s** |
| Typical coding turn | ~600-800 | **8-12s** |
| Deep reasoning turn | ~1500 | **20-25s** |
| Long-context turn (100K ctx cached) | + prefill ~2s incremental | +2s |

With 4 concurrent OMP instances (parallel=4): multiply decode portions by ~2.5× (29 vs 74 t/s effective) — a typical turn lands at **20-30s**.

## SillyTavern

ST is a single stream — use `HUIHUI_PARALLEL=1` or 2. The CoD profile is pre-wired (`openai_max_context: 262144`). Roleplay turns (~200-400 tokens) render at **~60-80 t/s** — effectively instant streaming. Prefill of a large character card + lorebook (~10-30K) takes 15-40s cold, then the prefix cache makes follow-ups incremental.

## Observational-memory gotchas (see `OBS_MEM_VS_YARN_2026-09-04.md`)

1. **huihui can't be the Observer model as-is** — thinking mode consumes the observer's token budget in `reasoning_content` and emits empty bullets. Fix: configure the plugin with a **non-thinking model** for memory work, or budget 1024+ tokens and fall back to the reasoning tail.
2. **Plugin is Pi/OMP-only** — no SillyTavern hook. ST users: built-in Summarize extensions or World Info.
3. **Not a showstopper**: YaRN 524K already delivers what obs-mem was trying to buy (long context at native accuracy).

## Launcher quick reference

```bash
bash scripts/start-huihui-27b-abliterated.sh                          # solo: 524K yarn
HUIHUI_PARALLEL=2 bash scripts/start-huihui-27b-abliterated.sh       # 2 sessions: 262K each
HUIHUI_PARALLEL=4 bash scripts/start-huihui-27b-abliterated.sh       # 4 sessions: 131K each
HUIHUI_CTX_SCALE=1 bash scripts/start-huihui-27b-abliterated.sh      # native 262K, max speed
```

OMP (from any Tailscale device):
```bash
omp --base-url http://$TAILSCALE_IP:8091/v1 \
    --system-prompt "Think step by step, but write each step in at most 5 words. Be extremely concise."
```

## Raw data

`bench/concurrency/par4.jsonl` — 24 throughput records + 1 accuracy (GSM8K 50% @ N=10 single run; variance is high at small N, see statistical note in `OBS_MEM_VS_YARN_2026-09-04.md`).