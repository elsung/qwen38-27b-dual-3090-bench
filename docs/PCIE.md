# PCIe Link Speed & CPU Governor Tuning — Null Result

> A cautionary tale for people chasing PCIe bottleneck ghosts.

## TL;DR

We investigated the hypothesis that **PCIe link speed was the bottleneck** for our local llama.cpp + vLLM decode speeds. It wasn't. The link was already at Gen 3 ×4 / ×16 under load, and forcing the sysfs `pcie_aspm=performance` policy + `cpu governor=performance` had **no measurable effect on decode t/s**.

This doc captures the investigation so you don't have to repeat it.

## Hypothesis

User (paraphrased): *"something is bottlenecking our performance. maybe its the bus lanes."*

Background:
- Both GPUs at idle reported `2.5 GT/s ×4` and `2.5 GT/s ×16` (PCIe Gen 1)
- Idle CPU governor = `powersave` (1.7 GHz)
- Idle `pcie_aspm` policy = `default`

Initial suspicion: Linux `pcie_aspm=default` aggressively downshifts PCIe to L1 sub-states when idle, which on some platforms also downshifts the **link speed** itself to save a few mW. CPU powersave governor throttles to 1.7 GHz.

## Diagnostic

```bash
# Confirm GPU PCIe state at idle
nvidia-smi --query-gpu=pci.link.speed.current,pci.link.width.current --format=csv
# output: 2.5 GT/s, x4 / 2.5 GT/s, x16  (both GPUs at PCIe Gen 1)

# Check max capability (should be 8.0 GT/s = Gen 3 on 3090)
cat /sys/bus/pci/devices/0000:01:00.0/max_link_speed
# output: 8.0 GT/s PCIe
cat /sys/bus/pci/devices/0000:0a:00.0/max_link_speed
# output: 8.0 GT/s PCIe

# Check ASPM policy
cat /sys/module/pcie_aspm/parameters/policy
# output: [default] performance powersave powersupersave
```

The slot *can* negotiate to Gen 3 — the issue is the *current operating state*, not the capability.

## Fix applied

```bash
# Requires root (or scoped sudoers)
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/module/pcie_aspm/parameters/policy
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

## Result: no impact

| Path | Pre-fix | Post-fix | Δ |
|---|---:|---:|---:|
| Huihui populated (16K prompt, q8_0 KV) | 98.39 t/s | 98.99 t/s | +0.6% (noise) |
| Flash-Next Inovello (best of 3) | 4.04 t/s | 4.49 t/s | +11% (best), mean –0.18 t/s (noise) |

**Both paths are NOT PCIe-bound.** Under load, the link was already at Gen 3 ×4 / ×16 — the ASPM L1 sub-state chatter on idle wasn't costing anything during active decode.

### Why?

- **Huihui (27B dense, 39 GiB total)**: weights fit in 48 GiB total VRAM with q8_0 KV at 262K → GPU-compute-bound, not PCIe
- **Flash-Next (`--n-cpu-moe 40` mmaps 40 layers' experts to host RAM)**: host RAM bandwidth-bound (DDR4 dual-channel ≈ 50 GB/s), not PCIe. The Inovello 16-slot LRU expert cache absorbs 89% of expert fetches (10.9% hit rate); the remaining 11% isn't large enough to be PCIe-bound at Gen 3 ×4

## What the fix DOES do

- Saves ~5–10 W idle (ASPM L1 chatter gone)
- CPU no longer downshifts to 1.7 GHz (slightly faster single-thread burst latency for non-GPU tasks)

Not a perf win. Left in place because the power saving is nice.

## Lessons

1. **The PCIe link speed shown at idle ≠ the link speed during workload.** Always check the **current** link state *during* decode, not before.
2. **`max_link_speed` from lspci tells you the slot's capability**, not its current state. To check current state, read `/sys/bus/pci/devices/.../current_link_speed`.
3. **GPU-compute-bound and host-RAM-bandwidth-bound workloads are insensitive to PCIe speed** — the bottleneck is somewhere else. Identify the bottleneck first.
4. **`nvidia-smi topo -m`** shows the topology (PHB vs NVLink vs PIX). 3090s are PHB only (no P2P, no NVLink) — that constrains tensor-parallel strategies but doesn't affect tensor-split (which uses both cards but no P2P traffic).
5. **For Flash-Next MoE on llama.cpp**, the real speedup came from **Inovello's expert cache** (PR #27861+28223+28243) and **the expert cache hit rate** (10.9%), not PCIe tuning.

## Recommended investigation order (cheapest first)

1. Check **VRAM allocator fragmentation** — `nvidia-smi` memory.used vs memory.free + run for a minute
2. Check **host RAM bandwidth** — `vmstat 1` while running
3. Check **GPU utilization** — `nvidia-smi dmon`
4. Check **PCIe link state under load** — this doc
5. Check **CPU governor** — but only matters for tokenization + small-batch sampling
6. Check **chat template** — most performance wins come from prompt/template tuning, not hardware

For most inference workloads on a properly-sized rig, **PCIe tuning is at the bottom of the priority list.**