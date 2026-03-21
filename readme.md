# Raw ML

## goals

build a neural network completely from scratch in C++. no PyTorch, no TensorFlow, no shortcuts

1. understand OOP in C++ — abstract classes, smart pointers, constructors/destructors
2. understand the Eigen library for linear algebra
3. implement a full nn framework with custom Tensor, Layer, Backend classes
4. train on XOR, MNIST (done), CIFAR-10 (todo)
5. implement the same thing on GPU with CUDA and compare runtimes
6. optimise the CUDA backend — cuDNN, streams, mixed precision etc

---

## what's been built

**core abstractions**

- `Tensor` — n-dim float array, doesn't know where its memory lives, delegates everything to a backend
- `Backend` (abstract) — owns all compute primitives: matmul, relu, im2col, bias_add, adam_update, softmax etc. two impls: `EigenBackend` (CPU) and `CudaBackend` (GPU)
- `Layer` (abstract) — forward/backward/update. holds a backend ref, never touches raw ptrs directly

**layers**

`Dense`, `Conv2D`, `MaxPool2D`, `GlobalAvgPool`, `Flatten`, `ReLU`, `Sigmoid`, `Sequential`

**training**

Adam optimiser, Xavier weight init, softmax cross-entropy loss. fused loss+gradient kernel on CUDA

**current network (MNIST)**

```
Conv(1→16, 3×3) → ReLU → MaxPool(2×2) → Conv(16→32, 3×3) → ReLU → MaxPool(2×2)
→ Flatten → Dense(1568→128) → ReLU → Dense(128→10)
```

~99.2% train accuracy, ~98.7% test accuracy in 5 epochs

---

## architecture

the backend abstraction is the whole point. a `Tensor` allocated with `EigenBackend` lives in RAM. same `Tensor` with `CudaBackend` lives in VRAM. layer code never touches raw ptrs — it just calls `backend_->matmul(...)`, `backend_->bias_add(...)` etc. swapping backends is one line

layer logic is totally backend-agnostic. the backend is where all the interesting implementation differences live

more detail in [architecture_md.md](architecture_md.md)

---

## CUDA optimisations

| version | total time (5 epochs) | vs baseline |
|---|---|---|
| baseline (sync after every kernel) | 60.3s | — |
| remove per-kernel `cudaDeviceSynchronize()` | 55.7s | -4.6s |
| fused softmax_cross_entropy kernel | 50.7s | -9.6s |
| cuBLAS for matmul | 45.9s | -14.4s (-24%) |

full notes in [cuda_optimisations.md](cuda_optimisations.md)

---

## building

```bash
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

run with Eigen (CPU):
```bash
./raw_ml eigen
```

run with CUDA (GPU):
```bash
./raw_ml cuda
```

---

## todo

- [ ] CIFAR-10 (needs data loader + deeper network prob)
- [ ] cuDNN for convolutions — replaces im2col+cuBLAS with nvidia's optimised conv kernels (Winograd etc.)
- [ ] CUDA streams — overlap CPU data loading with GPU compute
- [ ] mixed precision (BF16/FP16) — 2x tensor core throughput on Blackwell
- [ ] batch normalisation layer
