# Neural Network Framework - Architecture Documentation

## Overview

This is a modular neural network framework built from scratch in C++ with support for pluggable computational backends. The framework enables training neural networks on CPU (via Eigen) with the ability to seamlessly swap to GPU (via CUDA) without changing higher-level code LA LA.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                    │
│            (Sequential, Training Loops)              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                   Layer Layer                        │
│          (Dense, ReLU, Sigmoid, etc.)               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  Tensor Layer                        │
│         (High-level matrix operations)              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Backend Layer                        │
│         (EigenBackend, CudaBackend)                 │
└─────────────────────────────────────────────────────┘
```

## Namespace

All classes live in `namespace nn` to prevent name collisions with other libraries.

```cpp
namespace nn {
    // All framework classes
}
```

---

## Core Components

### 1. Backend Interface (`backend/backend_interface.h`)

**What it is:** Abstract base class defining all computational operations.

```cpp
class Backend {
    virtual float* allocate(size_t size) = 0;
    virtual void deallocate(float* ptr) = 0;
    virtual void matmul(...) = 0;
    virtual void add(...) = 0;
    virtual void relu(...) = 0;
    // ... all primitive operations
};
```

**Purpose:**
- Defines the contract: any backend MUST implement these methods
- Pure virtual functions (= 0) means you cannot instantiate Backend directly
- Enables polymorphism: swap CPU/GPU backends transparently

**Key Design Decision:**
- **Separation of concerns:** Algorithm logic (what to compute) is separate from execution (how/where to compute)
- **Extensibility:** Add new backends (CUDA, OpenCL, etc.) without changing higher-level code
- **Testability:** Can create mock backends for unit testing

---

### 2. EigenBackend (`backend/eigen_backend.h/cpp`)

**What it is:** Concrete CPU implementation using the Eigen library.

```cpp
class EigenBackend : public Backend {
    float* allocate(size_t size) override {
        return new float[size];  // Allocates CPU memory
    }
    
    void matmul(...) override {
        // Uses Eigen's optimized matrix multiplication
        Eigen::Map<...> mat_A(A, rows, cols);
        Eigen::Map<...> mat_B(B, rows, cols);
        mat_result = mat_A * mat_B;
    }
};
```

**Purpose:**
- Implements all Backend operations for CPU execution
- Leverages Eigen library for optimized linear algebra

**Why Eigen?**
- Well-tested, mature library
- Header-only (easy to integrate)
- Optimized for modern CPUs (SIMD, cache-friendly)
- Provides good baseline performance before GPU optimization

---

### 3. Tensor (`core/tensor.h/cpp`)

**What it is:** Multi-dimensional array abstraction with backend support.

```cpp
class Tensor {
private:
    std::vector<size_t> shape_;           // Dimensions: [rows, cols]
    size_t size_;                          // Total number of elements
    float* data_;                          // Raw data pointer (CPU or GPU)
    std::shared_ptr<Backend> backend_;    // Polymorphic backend pointer
    
public:
    Tensor(const std::vector<size_t>& shape, std::shared_ptr<Backend> backend);
    
    Tensor matmul(const Tensor& other) const {
        backend_->matmul(...);  // Virtual dispatch to EigenBackend or CudaBackend
    }
};
```

**Purpose:**
- User-friendly interface for matrix/tensor operations
- Hides backend complexity from users
- Manages memory lifecycle automatically
- Provides high-level mathematical operations

**Key Design Decisions:**

**Why `std::shared_ptr<Backend>`?**
- Multiple tensors can share the same backend instance
- Automatic memory management (no manual delete needed)
- Enables polymorphism: can point to EigenBackend or future CudaBackend

**Why raw `float*` instead of `std::vector<float>`?**
- GPU memory cannot be a `std::vector` - it's allocated with `cudaMalloc`
- Raw pointer can point to either CPU or GPU memory
- Backend handles allocation/deallocation specifics

**Copy vs Move Semantics:**
```cpp
Tensor a = b;              // Copy: allocates new memory, copies all data
Tensor a = std::move(b);   // Move: transfers ownership, no data copy
```

**Memory Management:**
- Constructor allocates via `backend_->allocate()`
- Destructor deallocates via `backend_->deallocate()`
- RAII ensures no memory leaks

---

### 4. Layer Base Class (`layers/layer.h`)

**What it is:** Abstract interface for all neural network layers.

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_parameters(float learning_rate) = 0;
    virtual ~Layer() = default;
    
protected:
    std::shared_ptr<Backend> backend_;
};
```

**Purpose:**
- Defines what every layer must implement: forward pass, backward pass, parameter updates
- Enables building networks from interchangeable components
- Provides polymorphic interface for Sequential network

**Why Abstract?**
- Dense, Conv, ReLU, etc. all behave differently
- But they all follow the same three-step interface
- Sequential network can treat all layers uniformly

---

### 5. Dense Layer (`layers/dense.h/cpp`)

**What it is:** Fully-connected layer with learnable parameters.

```cpp
class Dense : public Layer {
private:
    size_t input_dim_, output_dim_;
    
    // Learnable parameters
    Tensor weights_;       // (input_dim x output_dim)
    Tensor bias_;          // (1 x output_dim)
    
    // Gradient accumulators
    Tensor grad_weights_;  
    Tensor grad_bias_;
    
    // Cached for backpropagation
    Tensor input_cache_;
    
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = input;  // Save for backward pass
        return input.matmul(weights_) + bias_;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Compute gradients via chain rule
        grad_weights_ = input_cache_.transpose().matmul(grad_output);
        grad_bias_ = sum(grad_output, axis=0);
        return grad_output.matmul(weights_.transpose());
    }
    
    void update_parameters(float lr) override {
        weights_ -= lr * grad_weights_;
        bias_ -= lr * grad_bias_;
    }
};
```

**Purpose:**
- Implements affine transformation: `y = Wx + b`
- Learns optimal W and b through gradient descent

**Key Design Decisions:**

**Why cache input during forward?**
- Backward pass needs input to compute `grad_W = input^T * grad_output`
- More efficient to store once than recompute

**Why separate `grad_weights_` from `weights_`?**
- Accumulate gradients across mini-batches
- Enables advanced optimizers (momentum, Adam) that need gradient history
- Separates gradient computation from parameter updates

**Weight Initialization:**
- Uses Xavier/Glorot initialization: `U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))`
- Helps with gradient flow in deep networks

---

### 6. Activation Layers (`layers/activation.h/cpp`)

**What it is:** Element-wise non-linear transformations.

#### ReLU (Rectified Linear Unit)

```cpp
class ReLU : public Layer {
private:
    Tensor input_cache_;  // Cache input for computing derivative
    
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = input;
        return input.relu();  // max(0, x)
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // ReLU derivative: 1 if x > 0, else 0
        return grad_output.multiply(input_cache_.relu_derivative());
    }
    
    void update_parameters(float lr) override {
        // No learnable parameters
    }
};
```

#### Sigmoid

```cpp
class Sigmoid : public Layer {
private:
    Tensor output_cache_;  // Cache OUTPUT for derivative
    
public:
    Tensor forward(const Tensor& input) override {
        Tensor output = input.sigmoid();
        output_cache_ = output;  // Save output, not input!
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Sigmoid derivative: σ(x) * (1 - σ(x))
        return grad_output.multiply(output_cache_.sigmoid_derivative());
    }
};
```

**Purpose:**
- Add non-linearity to the network (essential for learning complex functions)
- No learnable parameters (stateless transformations)

**Why different caching strategies?**
- **ReLU:** Derivative depends on **input** (`x > 0 ? 1 : 0`)
- **Sigmoid:** Derivative depends on **output** (`σ(x) * (1 - σ(x))`)
- Cache what you need for efficient backward computation

**Why non-linearity matters:**
- Without activation functions, stacking layers is just matrix multiplication
- Multiple linear layers = one linear layer (useless)
- Non-linearity enables learning complex, non-linear decision boundaries

---

### 7. Sequential Network (`network/sequential.h/cpp`)

**What it is:** Container that chains layers into a complete network.

```cpp
class Sequential {
private:
    std::vector<std::shared_ptr<Layer>> layers_;
    
public:
    void add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }
    
    Tensor forward(const Tensor& input) {
        Tensor output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }
    
    void backward(const Tensor& grad_output) {
        Tensor grad = grad_output;
        // Iterate in REVERSE order
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }
    
    void update_parameters(float lr) {
        for (auto& layer : layers_) {
            layer->update_parameters(lr);
        }
    }
};
```

**Purpose:**
- Composable architecture: build networks by stacking layers
- Handles forward/backward passes automatically
- Treats all layer types uniformly via polymorphism

**Key Design Decisions:**

**Why `std::shared_ptr<Layer>`?**
- Polymorphism: store Dense, ReLU, Sigmoid in same vector
- Shared ownership: layers could theoretically be shared between networks
- Automatic cleanup when Sequential is destroyed

**Why backward in reverse order?**
- Chain rule: gradients flow backwards through computation graph
- Output layer gradients computed first, then propagated backwards
- Each layer needs gradient from next layer to compute its own

---

## How Components Interact

### Creating a Network

```cpp
// 1. Create backend
auto backend = std::make_shared<EigenBackend>();

// 2. Create network
Sequential network;

// 3. Add layers
network.add(std::make_shared<Dense>(2, 4, backend));   // Input: 2, Hidden: 4
network.add(std::make_shared<ReLU>(backend));           // Activation
network.add(std::make_shared<Dense>(4, 1, backend));   // Hidden: 4, Output: 1
network.add(std::make_shared<Sigmoid>(backend));        // Output activation
```

**What's happening:**
1. EigenBackend object created (shared across all tensors)
2. Dense layer creates Tensors for weights/bias, storing backend pointer
3. Layers stored in Sequential as `shared_ptr<Layer>` (polymorphic)

### Forward Pass Flow

```cpp
Tensor input({1, 2}, backend);
input.set_data({0.5, 0.3});

Tensor output = network.forward(input);
```

**Call chain:**
```
network.forward(input)
  ├─> layers_[0]->forward(input)           // Dense layer
  │     ├─> input.matmul(weights_)
  │     │     └─> backend_->matmul(...)    // Virtual call → EigenBackend::matmul
  │     └─> result.add(bias_)
  │
  ├─> layers_[1]->forward(...)              // ReLU layer
  │     └─> input.relu()
  │           └─> backend_->relu(...)       // EigenBackend::relu
  │
  ├─> layers_[2]->forward(...)              // Dense layer
  └─> layers_[3]->forward(...)              // Sigmoid layer
```

### Backward Pass Flow

```cpp
Tensor grad_output({1, 1}, backend);
grad_output.set_data({1.0});  // Loss gradient

network.backward(grad_output);
```

**Call chain (REVERSE order):**
```
network.backward(grad_output)
  ├─> layers_[3]->backward(grad)    // Sigmoid (last layer)
  │     └─> Returns grad w.r.t. its input
  │
  ├─> layers_[2]->backward(grad)    // Dense
  │     ├─> Computes grad_weights, grad_bias
  │     └─> Returns grad w.r.t. its input
  │
  ├─> layers_[1]->backward(grad)    // ReLU
  └─> layers_[0]->backward(grad)    // Dense (first layer)
```

### Parameter Update Flow

```cpp
network.update_parameters(0.01);  // learning_rate = 0.01
```

**Call chain:**
```
network.update_parameters(0.01)
  ├─> layers_[0]->update_parameters(0.01)
  │     └─> weights_ -= 0.01 * grad_weights_
  │     └─> bias_ -= 0.01 * grad_bias_
  │
  ├─> layers_[1]->update_parameters(0.01)  // No-op (ReLU has no params)
  │
  ├─> layers_[2]->update_parameters(0.01)
  └─> layers_[3]->update_parameters(0.01)  // No-op (Sigmoid has no params)
```

### Memory Lifecycle

```
1. User creates Tensor
   ↓
2. Tensor constructor calls backend_->allocate(size)
   ↓
3. EigenBackend::allocate() returns new float[size] (CPU memory)
   ↓
4. Tensor stores the pointer in data_
   ↓
5. User calls tensor.matmul(other)
   ↓
6. Tensor calls backend_->matmul(data_, other.data_, ...)
   ↓
7. EigenBackend::matmul wraps pointers with Eigen::Map (zero-copy)
   ↓
8. Eigen performs optimized matrix multiplication
   ↓
9. Result stored in newly allocated Tensor
   ↓
10. When Tensor destructor called, calls backend_->deallocate(data_)
    ↓
11. EigenBackend::deallocate() calls delete[] ptr
```

---

## Design Patterns

### 1. Strategy Pattern (Backend)
- **Interface:** Backend defines algorithm interface
- **Strategies:** EigenBackend (CPU), CudaBackend (GPU)
- **Context:** Tensor uses backend without knowing which implementation
- **Benefit:** Swap algorithms at runtime without changing client code

### 2. Template Method (Layer)
- **Base class:** Layer defines skeleton (forward/backward/update)
- **Subclasses:** Dense, ReLU, etc. implement specific behavior
- **Benefit:** Consistent interface, customizable behavior

### 3. Composite Pattern (Sequential)
- **Component:** Layer interface
- **Leaf:** Dense, ReLU, Sigmoid
- **Composite:** Sequential (contains multiple layers)
- **Benefit:** Treat individual layers and compositions uniformly

### 4. RAII (Resource Acquisition Is Initialization)
- **Acquisition:** Tensor constructor allocates memory
- **Release:** Tensor destructor deallocates memory
- **Benefit:** Exception-safe, automatic cleanup, no manual memory management

---

## Design Principles

### Separation of Concerns
- **Backend:** How to compute (CPU vs GPU)
- **Tensor:** What data structure (shape, operations)
- **Layer:** What neural network operation (dense, activation)
- **Sequential:** How to compose layers

### Single Responsibility
- Backend: Only handles low-level computation
- Tensor: Only handles data storage and high-level operations
- Layer: Only handles forward/backward for one operation
- Sequential: Only handles composition

### Open/Closed Principle
- Open for extension: Add new backends, layers without modifying existing code
- Closed for modification: Existing components don't change when extending

### Dependency Inversion
- High-level modules (Tensor, Layer) depend on abstractions (Backend interface)
- Low-level modules (EigenBackend) implement abstractions
- Neither depends on the other's concrete implementation

---

## Why This Architecture?

### Modularity
- Each component has one clear responsibility
- Components are loosely coupled
- Easy to understand and modify individual parts

### Extensibility
Add new functionality without modifying existing code:
- **New backend?** Implement Backend interface → works with all existing code
- **New layer type?** Inherit from Layer → works with Sequential
- **New network architecture?** Use Sequential or create new composer

### Testability
Each component can be tested independently:
- Test backend operations in isolation
- Test tensor operations with mock backend
- Test layers with known inputs/gradients
- Test networks with simple datasets (XOR)

### Performance
- Backend abstraction allows swapping CPU/GPU without code changes
- Virtual function overhead is negligible (~1 pointer dereference)
- Actual computation (matmul, etc.) dominates runtime by orders of magnitude
- Eigen provides SIMD optimizations on CPU
- Future CUDA backend will leverage GPU parallelism

### Maintainability
- Clear separation of concerns makes debugging easier
- Each class has well-defined responsibility
- Data flow is explicit and traceable
- Modern C++ features (smart pointers) prevent memory leaks

---

## Current Capabilities

### What Works Now
- ✅ Complete forward and backward propagation
- ✅ Gradient descent optimization
- ✅ Dense (fully-connected) layers
- ✅ ReLU and Sigmoid activations
- ✅ Sequential network composition
- ✅ CPU execution via Eigen backend
- ✅ Trains and solves XOR problem (non-linear classification)
- ✅ Proper memory management (no leaks)

### Training Example (XOR)

```cpp
auto backend = std::make_shared<EigenBackend>();

Sequential network;
network.add(std::make_shared<Dense>(2, 4, backend));
network.add(std::make_shared<Sigmoid>(backend));
network.add(std::make_shared<Dense>(4, 1, backend));
network.add(std::make_shared<Sigmoid>(backend));

// Training loop
for (int epoch = 0; epoch < 10000; ++epoch) {
    for (auto& [input, target] : training_data) {
        Tensor output = network.forward(input);
        Tensor loss_grad = compute_gradient(output, target);
        network.backward(loss_grad);
        network.update_parameters(learning_rate);
    }
}

// Result: Network learns XOR with ~100% accuracy
```

---

## Future Extensions

### 1. CUDA Backend (Next Step!)

```cpp
class CudaBackend : public Backend {
    float* allocate(size_t size) override {
        float* ptr;
        cudaMalloc(&ptr, size * sizeof(float));  // GPU memory
        return ptr;
    }
    
    void matmul(...) override {
        // Launch CUDA kernel
        dim3 grid(...), block(...);
        matmul_kernel<<<grid, block>>>(...);
        cudaDeviceSynchronize();
    }
    
    void deallocate(float* ptr) override {
        cudaFree(ptr);
    }
};
```

**Impact:** Everything else stays the same! Just swap backend:
```cpp
auto backend = std::make_shared<CudaBackend>();  // Was EigenBackend
// All existing code works on GPU now!
```

### 2. Additional Layer Types
- **Convolutional layers:** For computer vision
- **Pooling layers:** MaxPool, AvgPool
- **Dropout:** Regularization
- **Batch Normalization:** Training stability
- **LSTM/GRU:** Sequence modeling

### 3. Advanced Optimizers
```cpp
class Optimizer {
    virtual void step(std::vector<Tensor*> parameters) = 0;
};

class SGD : public Optimizer { /* momentum */ };
class Adam : public Optimizer { /* adaptive learning rates */ };
class RMSprop : public Optimizer { /* ... */ };
```

### 4. Loss Functions
```cpp
class Loss {
    virtual Tensor forward(const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor backward() = 0;
};

class MSELoss : public Loss { /* Mean squared error */ };
class CrossEntropyLoss : public Loss { /* Classification */ };
```

### 5. Data Pipeline
- Mini-batch creation
- Data shuffling
- Data augmentation (for images)
- Parallel data loading

### 6. Model Serialization
- Save trained weights to disk
- Load pre-trained models
- Transfer learning support

### 7. Real Datasets
- **MNIST:** Handwritten digit recognition (28x28 grayscale images)
- **CIFAR-10:** Object classification (32x32 color images)
- **ImageNet:** Large-scale image classification

---

## Performance Considerations

### Current Performance (CPU)
- Eigen uses SIMD instructions (SSE, AVX)
- Cache-friendly memory access patterns
- Optimized BLAS routines for matrix operations
- Good for prototyping and small models

### Expected Performance (GPU with CUDA)
- 10-100x speedup for large matrix operations
- Batch processing is crucial (leverage parallelism)
- Memory transfers (CPU ↔ GPU) can be bottleneck
- Need proper kernel optimization (memory coalescing, shared memory)

### Optimization Strategies (Future)
- **Kernel fusion:** Combine multiple operations into one kernel
- **Memory pooling:** Reuse allocated memory
- **Mixed precision:** FP16 for speed, FP32 for stability
- **Graph optimization:** Eliminate redundant operations
- **Asynchronous execution:** Overlap computation and memory transfers

---

## Project Structure

```
neural_net/
├── include/
│   ├── backend/
│   │   ├── backend_interface.h    # Abstract backend interface
│   │   └── eigen_backend.h        # CPU implementation
│   ├── core/
│   │   └── tensor.h               # Tensor abstraction
│   ├── layers/
│   │   ├── layer.h                # Abstract layer interface
│   │   ├── dense.h                # Fully-connected layer
│   │   └── activation.h           # ReLU, Sigmoid
│   └── network/
│       └── sequential.h           # Network container
├── src/
│   ├── backend/
│   │   └── eigen_backend.cpp
│   ├── core/
│   │   └── tensor.cpp
│   ├── layers/
│   │   ├── dense.cpp
│   │   └── activation.cpp
│   └── network/
│       └── sequential.cpp
└── tests/
    ├── test_eigen_backend.cpp     # Backend unit tests
    ├── test_tensor.cpp            # Tensor unit tests
    ├── test_dense.cpp             # Layer unit tests
    └── test_xor.cpp               # End-to-end training test
```

---

## Key Takeaways

1. **Abstraction enables flexibility:** Backend interface allows CPU/GPU swap without changing user code
2. **Polymorphism is powerful:** Virtual functions enable treating different backends/layers uniformly
3. **Smart pointers prevent bugs:** `shared_ptr` handles memory automatically, prevents leaks
4. **RAII ensures safety:** Resources (memory) tied to object lifetime
5. **Separation of concerns:** Clear boundaries between components makes code maintainable
6. **Design for extension:** Adding new functionality doesn't require modifying existing code

This architecture mirrors production frameworks (PyTorch, TensorFlow) in design philosophy, just simplified for learning. The patterns and principles scale to real-world deep learning systems.

---

## References & Learning Resources

- **Eigen Documentation:** https://eigen.tuxfamily.org/
- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/
- **Neural Networks and Deep Learning:** http://neuralnetworksanddeeplearning.com/
- **C++ Smart Pointers:** https://en.cppreference.com/w/cpp/memory
- **Design Patterns (GoF):** "Design Patterns: Elements of Reusable Object-Oriented Software"
