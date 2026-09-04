# scripts/ — Launchers & Utilities

These are the production launchers for the three supported runtime/model combos. All paths are env-var driven — see [docs/SETUP.md](../docs/SETUP.md).

## `start-huihui-27b-abliterated.sh`

Launches the **Qwen3.8-27B huihui abliterated** GGUF via llama.cpp on `$MODELS_DIR/huihui-qwen38-27b-abliterated-gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf`.

- Tensor-split 1,1 (both 3090s)
- q8_0 KV cache
- Draft-MTP n_max=3 (Multi-Token Prediction draft head)
- 262K native context

Stops any vLLM container on port 8091 first.

```bash
# Optional env: $LLAMA_HUIHUI_PORT (default 8091)
bash scripts/start-huihui-27b-abliterated.sh
```

The launcher prints a banner explaining that CoD reasoning-style injection is **client-side**. ST users switch to the CoD connection profile; OMP users pass `--system-prompt`.

## `start-unsloth-qwen38-27b.sh`

Launches the **Unsloth Dynamic v3.0** quants of Qwen3.8-27B via llama.cpp. Set `LLAMA_MODEL` to pick which quant:

```bash
LLAMA_MODEL=UD-Q4_K_XL bash scripts/start-unsloth-qwen38-27b.sh
LLAMA_MODEL=UD-Q4_K_M  bash scripts/start-unsloth-qwen38-27b.sh
```

Same recipe as the huihui launcher, just a different `-m`. Use 32768 ctx instead of 262K (the unsloth build is for shorter-context tests).

## `start-qwen38-flashnext.sh`

Launches the **Qwen3.8 Flash-Next FP8** model via vLLM Docker on port 8091.

```bash
bash scripts/start-qwen38-flashnext.sh
```

This stops the llama.cpp server first (since both bind port 8091). vLLM is ~15× faster than llama.cpp for Flash-Next.

The Flash-Next FP8 checkpoint should live at `$MODELS_DIR/qwen38-flash-next/Qwen3.8-Flash-Next-FP8/`. Adjust paths in the script if your layout differs.

## `sanitize.sh`

**Run before publishing.** Strips all private paths, IPs, and usernames from the repo:

```bash
bash scripts/sanitize.sh /path/to/llm-bench-public/
```

Idempotent. The redaction list is in the script header — add your own patterns there if needed.