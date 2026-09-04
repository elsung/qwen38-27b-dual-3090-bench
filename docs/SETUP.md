# Setup & Path Conventions

This doc explains how to install the toolchain and lay out files so the scripts in `scripts/` work out-of-the-box.

## Required toolchain

| Tool | Min version | Notes |
|---|---|---|
| llama.cpp | b10753+ (Inovello branch preferred for Flash-Next) | Build with `-DGGML_CUDA=ON -DGGML_NATIVE=OFF` |
| vLLM | 0.22.1+ | Use the official `vllm/vllm-openai` Docker image |
| Python | 3.10+ | The eval harness uses only `urllib` (no pip deps) |
| curl | any | For cold-start sanity checks |
| jq | any | For filtering llama-server responses |

## Path conventions

The launchers reference these env-var paths. Set them before running:

```bash
export MODELS_DIR=$HOME/models                  # GGUF files live here
export DESKTOP=$HOME/llm-bench-public/scripts  # launchers + bench scripts
export ST_DIR=$HOME/SillyTavern                # only if using SillyTavern integration
export WORKSPACE=$HOME/AI/LLMs                 # llama.cpp + vLLM source trees
export RESULTS_DIR=$HOME/llm-bench-public/docs # where benchmark markdown lives
```

## Suggested file layout under `$MODELS_DIR`

```
$MODELS_DIR/
├── qwen38-27b-exl3-4bpw/                  # EXL3 quant for tensor-parallel on 2x3090
├── Qwen3.8-27B-FP8/                       # official FP8 weights for vLLM (~29 GiB)
├── huihui-qwen38-27b-abliterated-gguf/
│   └── Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf   # 17.4 GiB
├── unsloth-qwen38-27b-gguf/                # Unsloth Dynamic v3.0 quants
│   ├── Qwen3.8-27B-UD-Q4_K_XL.gguf        # 17.6 GiB
│   └── Qwen3.8-27B-UD-Q4_K_M.gguf         # 16.5 GiB
├── qwen38-flash-next/
│   ├── Qwen3.8-Flash-Next-FP8/             # ~30 GiB, used by vLLM
│   └── UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf  # llama.cpp
├── glm53-flash-udiq3/                     # GLM-5.3-Flash GGUF (~120 GiB)
└── (any others you want to try)
```

## llama.cpp build (Inovello branch)

```bash
git clone https://github.com/Inovello/llama.cpp.git -b flashnext-2x3090 $WORKSPACE/inovello-flashnext
cd $WORKSPACE/inovello-flashnext
cmake -B build-3090 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89" \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-3090 -j$(nproc)
# binary at: $WORKSPACE/inovello-flashnext/build-3090/bin/llama-server
```

## vLLM Docker (for Flash-Next FP8)

```bash
docker pull vllm/vllm-openai:latest
# Use the launcher in scripts/start-qwen38-flashnext.sh for the standard invocation
```

## SillyTavern integration (optional)

Install ST normally, then create two OpenAI Settings connection profiles (one for huihui base, one for huihui+CoD). Profiles are stored at `$ST_DIR/data/default-user/OpenAI Settings/`. See `docs/MODELS.md` for the full JSON.

## Verifying the install

```bash
# 1. llama.cpp binary works
$WORKSPACE/inovello-flashnext/build-3090/bin/llama-server --version
# expected: version: 0.3.0-dev (build 10753, commit <hash>)

# 2. vLLM Docker works
docker run --rm --gpus all vllm/vllm-openai:latest --version

# 3. Eval harness works (no install needed)
python3 bench/gsm8k_harness.py --help

# 4. Sanity: can the scripts see the GGUF?
ls $MODELS_DIR/huihui-qwen38-27b-abliterated-gguf/*.gguf
```

## Disk space

Plan for **~300 GiB** of model checkpoints for the full Qwen3.8 + Flash-Next + GLM-5.3 stack. The 3090+3090 setup is VRAM-bound, not disk-bound, so a single SATA SSD is fine.

## Network

The repo expects:
- Internet access to pull models from HuggingFace (CDN routes ~25 MB/s; expect 30+ min per 17 GiB GGUF)
- For OMP / Tailscale integration, the box must be reachable on `0.0.0.0:8091` from your client network

## Common pitfalls

- **GGUF file byte-count mismatch**: HF's HF_TOKEN-gated downloads are faster but require auth; without a token, fall back to authenticated HuggingFace Hub workers
- **Per-tensor allocator OOM** on qwen4exp models: use `--n-cpu-moe N` to offload experts to CPU RAM
- **Draft-MTP requires the MTP head** in the same GGUF; the `--spec-type draft-mtp` flag won't work if it's not present
- **q4_0 KV** measurably degrades reasoning quality — keep q8_0