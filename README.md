# Local LLM Benchmarking & Optimization — Public Repo

> **Note**: this is a fork-and-share repository. Before pushing, run `bash scripts/sanitize.sh` to strip private paths / IPs. See [Contributing](#-contributing) below.
This repo is a practical guide + working scripts for running the **Qwen3.8-27B / Qwen3.8 Flash-Next / GLM-5.3-Flash** model family on **dual NVIDIA RTX 3090 + Ryzen 7 5800XT + 128 GiB DDR4** (or similar 24 GiB ×2 GPU rigs). It is the result of several weeks of empirical benchmarking.
The headline finding: **applying the Chain-of-Draft ("5-word drafts") system prompt to a 27B dense reasoning model takes GSM8K accuracy from 23% to 77%**, with no model changes and only a -8% decode t/s cost. That single change beats both Unsloth Dynamic v3.0 quants of the same model on the same benchmark.
---
## At a glance
| What | Result | See |
|---|---|---|
| **Chain-of-Draft prompt effect on GSM8K** | 23% → 77% (+54pp), -8% t/s | [REASONING.md](docs/REASONING.md) |
| **3-way: huihui+CoD vs Unsloth v3.0 UD-Q4_K_XL vs UD-Q4_K_M** | huihui wins (77% vs 60% vs 67%) | [REASONING.md](docs/REASONING.md#3-way-comparison) |
| **Best 27B throughput** | ~99 t/s tensor-split, q8_0 KV, draft-MTP n_max=3 | [MODELS.md](docs/MODELS.md) |
| **Best Flash-Next throughput** | 60.86 t/s via vLLM Docker (`qwen3_5_mtp` spec) | [MODELS.md](docs/MODELS.md) |
| **Full speed matrix (prefill/decode × ctx × concurrency)** | huihui 70–96 t/s, Flash-Next 50–80 t/s | [SPEED_MATRIX.md](docs/SPEED_MATRIX.md) |
| **Long-context (1M via YaRN)** | vLLM supports, llama.cpp partial; VRAM-limited on dual 3090 | [LONG_CONTEXT.md](docs/LONG_CONTEXT.md) |
| **RAM/disk offload (kv-offload)** | KV spills to host RAM; 200 GiB available (RAM+zram+swap); 30-100 t/s at 1M | [LONG_CONTEXT.md](docs/LONG_CONTEXT.md#offloading-context-to-ram--disk-alternative-path-to-1m) |
## Hardware tested
```
CPU:  AMD Ryzen 7 5800XT (8c/16t)
RAM:  128 GiB DDR4
MB:   ASRock AB350 Pro4
GPU:  2× NVIDIA RTX 3090 (24 GiB each, PCIe 3.0 x4 + x16)
Disk: 1× NVMe SSD for OS/workspace; 1× SATA SSD for model checkpoints
OS:   CachyOS 7.0.10 (Arch-based)
The 2× 3090 setup is the sweet spot for 27B dense models: both fit in VRAM with q8_0 KV cache at 262K context (FP16 weights ~17 GiB / GPU, q8_0 KV at 262K ≈ 12 GiB / GPU). GLM-5.3-Flash (MoE) needs CPU-MoE offload because it doesn't fit in 48 GiB total.
This repo's results should transfer cleanly to **any dual-24-GiB-CUDA-GPU rig** (3090, 4090, 5090). Single-GPU rigs will need to drop to smaller context or skip tensor-split.
## Repo layout
.
├── README.md                    # this file
├── docs/
│   ├── SETUP.md                 # install + path conventions
│   ├── MODELS.md                # per-model launch recipes + speeds
│   ├── REASONING.md             # CoD/NoWait/TokBudget + 3-way benchmark
│   ├── PCIE.md                  # PCIe / governor sysfs story (null result)
│   └── LEARNINGS.md             # condensed best practices
├── scripts/
│   ├── start-huihui-27b-abliterated.sh   # llama.cpp launcher for huihui
│   ├── start-unsloth-qwen38-27b.sh       # same recipe for Unsloth Dynamic v3.0
│   ├── start-qwen38-flashnext.sh         # vLLM Docker launcher for Flash-Next
│   ├── sanitize.sh                       # strip private paths/IPs before pushing
│   └── README.md                         # script docs
├── bench/
│   ├── gsm8k_harness.py                 # urllib-only GSM8K eval (no deps)
│   ├── gsm8k_10.jsonl                   # 10 GSM8K problems used
│   ├── nowait_tokens.json               # 83 reflection-keyword token IDs
│   ├── nowait_tokens.ids.txt            # with decoded text
│   ├── results.jsonl                    # raw per-query records (sweep)
│   ├── compare_3way.jsonl               # raw per-query records (3-way)
│   ├── reproduce_reasoning_sweep.sh     # run all 12 sweep configs
│   ├── compare_3way.sh                  # run the 4-config 3-way
│   └── log_*.txt                        # per-config bench logs
└── posts/
    └── REDDIT.md                        # Reddit-ready writeup
## Quick start (TL;DR)
```bash
# 1. Install deps
#    - llama.cpp b10753+ built with CUDA, BLAS, no SOTA flags
#    - vLLM 0.22.1+ in a Docker image
#    - python 3.10+ (no extra packages — bench uses urllib only)
# 2. Set your paths (edit scripts/*.sh or export before running)
export MODELS_DIR=$HOME/models                  # where the GGUF files live
export DESKTOP=$HOME/llm-bench-public/scripts  # where the launchers live
export ST_DIR=$HOME/SillyTavern                # if using ST integration
# 3. Launch the recommended setup (huihui + CoD via SillyTavern/OMP)
bash scripts/start-huihui-27b-abliterated.sh
# then in your client (OMP / ST / curl), inject:
#   system: "Think step by step, but write each step in at most 5 words. Be extremely concise."
# 4. Verify with a benchmark
cd bench
bash reproduce_reasoning_sweep.sh    # ~50 min, all 12 sweep configs
bash compare_3way.sh                 # ~25 min, the 3-way GSM8K comparison
## What this repo is NOT
- **Not a model release** — we use Unsloth and huihui-ai's public GGUFs as-is
- **Not a training repo** — all benchmarks are inference-time
- **Not a generic Qwen3 guide** — focused on the specific workloads we run (SillyTavern roleplay + math/agentic via OMP)
- **Not exhaustively benchmarked** — small N (10 GSM8K problems × 3 runs); good for directional conclusions, not statistical significance
## Contributing
Before pushing to a public repo, run:
bash scripts/sanitize.sh .
This strips usernames, host paths, Tailscale IPs, and any OMP session IDs. It is idempotent and safe to re-run. See `scripts/sanitize.sh` for the exact redaction list.
PRs welcome for:
- Additional model benchmarks (GLM-5.3, more Qwen3.8 sizes)
- Larger N runs (50+ GSM8K problems for proper statistical significance)
- Different quant schemes (Q4_K_S, IQ3_XXS, EXL3 4.0bpw)
- Different reasoning-effort approaches (Chain-of-Thought with verification, etc.)
## Citation
If you use these scripts or findings, please link back to this repo. No formal paper; the canonical writeup is `docs/REASONING.md` (the Chain-of-Draft + 3-way comparison).
## License
MIT. Do whatever; attribution appreciated.
