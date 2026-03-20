# CUDA Optimisation Log

## Baseline

No optimisations. Every kernel launch followed by `cudaDeviceSynchronize()`.
Loss computed entirely on CPU with two PCIe round-trips per batch.

| Epoch | Loss | Train Acc | Time |
|-------|------|-----------|------|
| 1 | 0.1845 | 94.60% | 12.43s |
| 2 | 0.0557 | 98.28% | 11.69s |
| 3 | 0.0376 | 98.84% | 12.11s |
| 4 | 0.0283 | 99.12% | 12.05s |
| 5 | 0.0229 | 99.27% | 12.01s |

**Total: 60.29s &nbsp;|&nbsp; Test accuracy: 98.81%**

---

## Optimisation 1 — Remove `cudaDeviceSynchronize()` after every kernel

Every kernel launch was followed by a sync, forcing the CPU to stall until
the GPU finished before queuing the next operation. The GPU was left idle
between every single kernel.

**Fix:** removed all per-kernel syncs. Kernels in the default stream execute
in order automatically — no CPU intervention needed. The only sync that
matters is when results are read back to the CPU, which already happens
implicitly via `cudaMemcpy(DeviceToHost)` inside `download()`.

| Epoch | Loss | Train Acc | Time |
|-------|------|-----------|------|
| 1 | 0.2054 | 93.82% | 11.42s |
| 2 | 0.0577 | 98.20% | 10.40s |
| 3 | 0.0385 | 98.80% | 10.55s |
| 4 | 0.0298 | 99.08% | 12.53s |
| 5 | 0.0239 | 99.18% | 10.81s |

**Total: 55.72s &nbsp;|&nbsp; Test accuracy: 98.92% &nbsp;|&nbsp; Saved: ~4.6s**

---

## Optimisation 2 — Fused `softmax_cross_entropy` CUDA kernel

Every batch, the loss function was:
1. Downloading all logits from GPU → CPU (64 × 10 = 640 floats over PCIe)
2. Computing softmax + cross-entropy on CPU
3. Uploading the gradient back GPU → CPU (640 floats over PCIe)

That's ~9,370 PCIe round-trips across 5 epochs.

**Fix:** `softmax_cross_entropy_kernel` — one thread per batch item. Each
thread computes its own softmax, accumulates the loss scalar on-device via
`atomicAdd`, and writes its gradient row directly in VRAM. The loss function
now only transfers targets up (640 floats) and downloads 1 float (the scalar
loss) per batch.

| Epoch | Loss | Train Acc | Time |
|-------|------|-----------|------|
| 1 | 0.1792 | 94.70% | 9.89s |
| 2 | 0.0551 | 98.32% | 10.24s |
| 3 | 0.0373 | 98.86% | 10.51s |
| 4 | 0.0276 | 99.13% | 9.98s |
| 5 | 0.0219 | 99.30% | 10.07s |

**Total: 50.69s &nbsp;|&nbsp; Test accuracy: 98.89% &nbsp;|&nbsp; Saved: ~5.0s**

---

## Summary

| Version | Total Time | vs Baseline |
|---------|-----------|-------------|
| Baseline | 60.29s | — |
| + Remove syncs | 55.72s | -4.6s |
| + Loss kernel | 50.69s | **-9.6s (-16%)** |

Test accuracy consistent throughout (~98.9%) — all gains are pure speed, no accuracy trade-off.
