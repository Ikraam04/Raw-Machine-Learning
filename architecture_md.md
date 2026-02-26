# Neural Network Framework - Architecture Documentation

## Overview

This is a modular neural network framework built from scratch in C++ with support for pluggable computational backends. The framework enables training neural networks on CPU (via Eigen) with the ability to seamlessly swap to GPU (via CUDA) without changing higher-level code LA LA.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                   │
│         (Sequential, Training Loops, Loss eval)     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                   Layer Layer                       │
│          (Dense, ReLU, Sigmoid, etc.)               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  Tensor Layer                       │
│         (High-level matrix operations)              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Backend Layer                       │
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
- implement memmory managment functions as well as matrix operations and activation functions
- Pure virtual functions (= 0) means you cannot instantiate Backend directly
- Enables polymorphism: swap CPU/GPU backends transparently


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
- Implements affine transformation: **$y = Wx + b$**
- Learns optimal W and b through gradient descent

**Key Design Decisions:**

**Why cache input during forward?**
- Backward pass needs input to compute **$grad_W = input^T * grad_{output}$**
- More efficient to just store
**Why separate `grad_weights_` from `weights_`?**
- Accumulate gradients across mini-batches
- Helps when with optimizers
- Separates gradient computation from parameter updates

**Weight Initialization:**
- Uses Xavier/Glorot initialization: $U(-\sqrt{(6/(n_{in} + n_{out}))}, \sqrt{(6/(n_{in} + n_{out})))}$


---

### 6. Activation Layers (`layers/activation.h/cpp`)

**What it is and why:** just non linear functions to add non linearity - helps with finding more complex patterns
#### ReLU (Rectified Linear Unit)
 $$f(x) = max(0,x)$$
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
$$f(x) = \frac{1}{1+e^{-x}}$$
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

### Softmax

 $$softmax(z_i) = \frac{e^{z_i}}{\sum_j{e^{z_j}}}$$

```cpp
class Softmax : public Layer {
private:
    Tensor output_cache_;
    
public:
    Tensor forward(const Tensor& input) override {
        // For each row (batch sample):
        //   1. Find max for numerical stability
        //   2. Compute exp(x - max) for each element
        //   3. Normalize: softmax[i] = exp(x[i]) / sum(exp(x))
        
        Tensor output = compute_softmax(input);  // Simplified
        output_cache_ = output;
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Softmax Jacobian: 
        // grad[i] = sum_j (softmax[i] * (d_ij - softmax[j]) * grad_out[j])
        // More complex than element-wise due to cross-dependencies
        
        return compute_softmax_gradient(output_cache_, grad_output);
    }
};
```
*full implementation in activation.cpp*

**Purpose:**
- Add non-linearity to the network (essential for learning complex functions)
- No learnable parameters (stateless transformations)

**Why different caching strategies?**
- **ReLU:** Derivative depends on **input** (`x > 0 ? 1 : 0`)
- **Sigmoid:** Derivative depends on **output** (`σ(x) * (1 - σ(x))`)
- **softmax** dont even ask LOL

**Why non-linearity matters:**
- Without activation functions, stacking layers is just matrix multiplication
- Multiple linear layers = one linear layer (useless)
- Non-linearity enables learning complex, non-linear decision boundaries

---

### 7. Loss
**what it is:** Functions that calculate how "wrong" the network is, not exactly structual (layers) but just a function

#### Mean Squared Error (MSE)
$$ Loss = \frac{1}{BD}\sum_{b=1}^{B}\sum_{c=1}^{C}(\hat{y}_{b,c} - y_{b,c})$$

```cpp
float mse_loss(const Tensor& predictions,
               const std::vector<float>& targets,
               size_t batch_size,
               size_t output_dim) 
               {
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; ++b) { // for each sample in batch
        for (size_t c = 0; c < output_dim; ++c) { // for each output dimension
            size_t idx = b * output_dim + c; // index in flat vector
            float pred = predictions.data()[idx]; 
            float target = targets[idx];
            total_loss += (pred - target) * (pred - target); // MSE: (pred - target)^2 and then accumulate
        }
    }
    
    return total_loss / (batch_size * output_dim);  // average over batch and output dimensions
}
```
- assumes input and output are just continious values, literally measures the square distance
- Used for regression

#### cross entropy loss
$$Loss = -z\sum_{c=1}^{C}y_{b,c}log(\hat{y}_{b,c}) $$

```cpp
float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes) {
    const float epsilon = 1e-7f;  // Prevent log(0)
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; ++b) { // for each sample in batch
        for (size_t c = 0; c < num_classes; ++c) { // for each class in output
            size_t idx = b * num_classes + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            
            // Only compute loss for true class (target = 1)
            if (target > 0.5f) {
                // Clamp prediction to [epsilon, 1 - epsilon]
                pred = std::max(epsilon, std::min(1.0f - epsilon, pred));
                total_loss += -std::log(pred);
            }
        }
    }
    
    return total_loss / batch_size;  // Average over batch
}
```
 - mainly used for classification 
 - assumes input (logit) is a valid probability distrubution (after softmax or sigmoid)


### 8. Sequential Network (`network/sequential.h/cpp`)

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

## Current Capabilities

### What Works Now
- Complete forward and backward propagation
- Gradient descent optimization
- Dense (fully-connected) layers
- ReLU and Sigmoid activations
- Sequential network composition
- CPU execution via Eigen backend
- Trains and solves XOR problem (non-linear classification)
- Trains and solves MNIST

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



## Key Takeaways

1. **Abstraction enables flexibility:** Backend interface allows CPU/GPU swap without changing user code
2. **Polymorphism is powerful:** Virtual functions enable treating different backends/layers uniformly
3. **Smart pointers prevent bugs:** `shared_ptr` handles memory automatically, prevents leaks
4. **RAII ensures safety:** Resources (memory) tied to object lifetime
5. **Separation of concerns:** Clear boundaries between components makes code maintainable
6. **Design for extension:** Adding new functionality doesn't require modifying existing code

This architecture mirrors production frameworks (PyTorch, TensorFlow) in design philosophy, just simplified for learning. The patterns and principles scale to real-world deep learning systems.

---

