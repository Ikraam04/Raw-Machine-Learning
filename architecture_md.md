# Neural Network Framework - Architecture

## overview

modular nn framework built from scratch in C++. the whole point is that you can swap the compute backend (CPU vs GPU) without touching any of the layer / training code. basically mimics how pytorch abstracts device placement, but way simpler obv

## high-level architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                   │
│         (Sequential, training loops, loss)          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                   Layer Layer                       │
│     (Dense, Conv2D, MaxPool2D, ReLU, Flatten...)    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  Tensor Layer                       │
│         (n-dim array, delegates everything)         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Backend Layer                       │
│         (EigenBackend CPU / CudaBackend GPU)        │
└─────────────────────────────────────────────────────┘
```

all classes live in `namespace nn`

---

## core components

### 1. Backend Interface (`core/backend_interface.h`)

abstract base class that defines every compute primitive. if it touches data, it goes through here

```cpp
class Backend {
    // memory
    virtual float* allocate(size_t size) = 0;
    virtual void deallocate(float* ptr) = 0;
    virtual void upload(float* dst, const float* src, size_t n) = 0;
    virtual void download(float* dst, const float* src, size_t n) = 0;

    // basic ops
    virtual void matmul(...) = 0;
    virtual void add(...) = 0;
    virtual void scale(...) = 0;
    virtual void transpose(...) = 0;
    virtual void relu(...) = 0;
    virtual void relu_backward(...) = 0;

    // conv
    virtual void im2col(...) = 0;
    virtual void col2im(...) = 0;

    // layer primitives (so layers never dereference raw ptrs)
    virtual void bias_add(...) = 0;       // add bias to every row
    virtual void sum_rows(...) = 0;       // reduce along batch dim (for grad_bias)
    virtual void adam_update(...) = 0;    // full adam step in one kernel
    virtual void softmax_forward(...) = 0;
    virtual void softmax_backward(...) = 0;
    virtual void permute_nhwc_nchw(...) = 0;
    virtual void permute_nchw_nhwc(...) = 0;
    virtual void maxpool_forward(...) = 0;
    virtual void maxpool_backward(...) = 0;
    virtual void global_avg_pool_forward(...) = 0;
    virtual void global_avg_pool_backward(...) = 0;

    // loss (fused kernel on cuda)
    virtual float softmax_cross_entropy_loss_and_gradient(...) = 0;
};
```

pure virtual (= 0) so you cant instantiate Backend directly. polymorphism means layers just call `backend_->whatever()` and dont care which impl runs

---

### 2. EigenBackend (`backends/eigen_backend.h/cpp`)

CPU impl. uses Eigen for matmul/transpose (its very good at those), plain loops for everything else

```cpp
class EigenBackend : public Backend {
    float* allocate(size_t size) override {
        return new float[size];
    }
    void matmul(...) override {
        // wraps ptrs with Eigen::Map (zero copy) then does mat_A * mat_B
    }
    void bias_add(float* data, const float* bias, int rows, int cols) override {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i * cols + j] += bias[j];
    }
    // etc.
};
```

---

### 3. CudaBackend (`backends/cuda_backend.h/cu`)

GPU impl. `allocate` calls `cudaMalloc` so all tensor data lives in VRAM. layer code never sees the difference bc it only ever calls backend methods

```cpp
class CudaBackend : public Backend {
    float* allocate(size_t size) override {
        float* ptr;
        cudaMalloc(&ptr, size * sizeof(float));
        return ptr;  // device pointer — CPU cannot dereference this
    }
    void matmul(...) override {
        // cublasSgemm — uses tensor cores on Blackwell, much faster than naive kernel
    }
    void bias_add(...) override {
        bias_add_kernel<<<...>>>(data, bias, rows, cols);
        // no cudaDeviceSynchronize here — kernels queue in order automatically
    }
};
```

key things abt the cuda backend:
- **no `cudaDeviceSynchronize()` after kernels** — they queue in the default stream and execute in order. syncing after every kernel was the original bottleneck (cost ~4.6s/run)
- **cuBLAS for matmul** — `cublasSgemm` with the column-major transpose trick. way faster than handwritten kernel on tensor-core hardware
- **fused softmax_cross_entropy kernel** — saves ~9370 PCIe roundtrips across 5 epochs
- handle is created lazily on first matmul call

---

### 4. Tensor (`core/tensor.h/cpp`)

n-dim float array. doesnt know or care where its memory is

```cpp
class Tensor {
    std::vector<size_t> shape_;
    size_t size_;
    float* data_;                          // cpu ptr or device ptr, depends on backend
    std::shared_ptr<Backend> backend_;

public:
    Tensor(const std::vector<size_t>& shape, std::shared_ptr<Backend> backend);
    // constructor calls backend_->allocate(), destructor calls backend_->deallocate()
    // RAII — no manual memory management needed
};
```

**why raw `float*` instead of `std::vector<float>`?**
`std::vector` is CPU only. GPU memory comes from `cudaMalloc` and lives at a device address the CPU cant touch. raw ptr works for both, backend handles what it actually means

**why `shared_ptr<Backend>`?**
multiple tensors share one backend instance. automatic cleanup, no manual delete, enables polymorphism

---

### 5. Layer Base (`layers/layer.h`)

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_parameters(float lr) = 0;
    virtual ~Layer() = default;
protected:
    std::shared_ptr<Backend> backend_;
};
```

every layer gets a backend ref at construction. it uses it for all compute — never calls `data()[i]` directly (that would segfault on cuda bc the ptr is in VRAM)

---

### 6. Dense (`layers/dense.h/cpp`)

fully connected layer. y = Wx + b

- weights: `(input_dim × output_dim)`, Xavier init
- forward: matmul + bias_add via backend
- backward: grad_weights = inputᵀ × grad_out, grad_bias = sum_rows(grad_out), grad_input = grad_out × weightsᵀ
- update: adam step via `backend_->adam_update()`

**why cache input during forward?**
backward needs it to compute grad_weights (chain rule). cheaper to just store it than recompute

---

### 7. Conv2D (`layers/conv2d.h/cpp`)

2D convolution via im2col + matmul

forward:
1. `im2col` — unrolls input patches into a matrix
2. `matmul(col, weights)` — one big matmul does all the conv
3. `bias_add`
4. `permute_nhwc_nchw` — layout conversion for the rest of the network

backward: grad_weights = colᵀ × grad_nhwc, grad_col = grad_nhwc × weightsᵀ, then `col2im`

im2col is the standard trick — trades memory for the ability to use a fast generic matmul instead of a slow nested loop conv

---

### 8. Activations (`layers/activation.h/cpp`)

non-linearities, no learnable params

- **ReLU:** `f(x) = max(0, x)`. backward masks grad where input ≤ 0. caches input
- **Sigmoid:** `f(x) = 1/(1+e^-x)`. backward uses output (not input) bc derivative is `σ(x)(1-σ(x))`
- **Softmax:** normalised exp over class dim. mostly used fused with cross-entropy loss now

without non-linearities, stacking layers is just matrix multiplication — multiple linear layers collapse to one. non-linearity is what lets the network learn complex stuff

---

### 9. Sequential (`layers/sequential.h/cpp`)

just a vector of layers with forward/backward/update wired up

```cpp
// forward: iterate in order
for (auto& layer : layers_)
    output = layer->forward(output);

// backward: iterate in REVERSE (chain rule — gradients flow backwards)
for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
    grad = (*it)->backward(grad);
```

stores layers as `shared_ptr<Layer>` so you can mix Dense, Conv2D, ReLU etc in the same vector

---

### 10. Loss (`loss/loss.h/cpp`)

not a layer, just functions. computes scalar loss + gradient to kick off backprop

- **MSE** — regression, measures squared distance
- **cross entropy** — classification, assumes softmax output
- **softmax_cross_entropy** (fused) — numerically stable, does softmax + loss + gradient in one pass. on cuda this is a single kernel with atomicAdd for the loss scalar — avoids downloading logits to cpu every batch

---

## how it all fits together

```cpp
auto backend = std::make_shared<CudaBackend>();  // swap to EigenBackend for CPU

Sequential net;
net.add(std::make_shared<Conv2D>(1, 16, 3, 3, 1, 1, 1, 1, backend));
net.add(std::make_shared<ReLU>(backend));
net.add(std::make_shared<MaxPool2D>(2, 2, backend));
// ... etc

// training loop
Tensor output = net.forward(batch);
auto [loss, grad] = softmax_cross_entropy(output, labels);
net.backward(grad);
net.update_parameters(lr);
```

swapping `CudaBackend` ↔ `EigenBackend` is literally one line. all the layer logic is identical

---

## current capabilities

- full forward + backward prop
- Conv2D, MaxPool2D, GlobalAvgPool, Dense, ReLU, Sigmoid, Flatten
- Adam optimiser, Xavier init
- MNIST CNN: ~99.2% train / ~98.7% test accuracy in 5 epochs (~46s on RTX 5080)
- both CPU (Eigen) and GPU (CUDA) backends fully working
- cuBLAS matmul, fused softmax_cross_entropy kernel

see [cuda_optimisations.md](cuda_optimisations.md) for the optimisation history

---

## key takeaways

1. backend interface = the whole design. everything else falls out of it
2. virtual dispatch lets you treat cpu/gpu identically at the layer level
3. `shared_ptr` everywhere means you basically cant leak memory even if you try
4. layers must never dereference `data()` directly — always go through backend methods (otherwise CUDA segfaults immediately)
5. im2col converts conv into matmul so you get cuBLAS for free
