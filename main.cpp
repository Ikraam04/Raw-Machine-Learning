#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>

#include "core/tensor.h"
#include "backends/eigen_backend.h"
#include "backends/cuda_backend.h"
#include "layers/sequential.h"
#include "layers/conv2d.h"
#include "layers/maxpool2d.h"
#include "layers/flatten.h"
#include "layers/dense.h"
#include "layers/activation.h"
#include "loss/loss.h"
#include "data/mnist_loader.h"

// MNIST CNN architecture:
//   Conv2D(1→16, 3x3, pad=1) → ReLU → MaxPool(2)    → 14x14
//   Conv2D(16→32, 3x3, pad=1) → ReLU → MaxPool(2)   → 7x7
//   Flatten → Dense(32*7*7→128) → ReLU → Dense(128→10)
//   Softmax cross-entropy loss (fused)

int main() {

    std::shared_ptr<nn::Backend> backend = std::make_shared<nn::CudaBackend>();
    std::cout << "Backend: CUDA (GPU)\n";

    // --- hyperparameters ---
    const size_t batch_size   = 64;
    const size_t num_epochs   = 5;
    const float  learning_rate = 0.001f;
    const size_t num_classes  = 10;

    // --- load MNIST ---
    std::string data_dir = "../MNIST_Data/";
    nn::MNISTDataset train_data = nn::load_mnist(
        data_dir + "train-images.idx3-ubyte",
        data_dir + "train-labels.idx1-ubyte");
    nn::MNISTDataset test_data = nn::load_mnist(
        data_dir + "t10k-images.idx3-ubyte",
        data_dir + "t10k-labels.idx1-ubyte");

    std::cout << "Train samples: " << train_data.size() << "\n";
    std::cout << "Test samples:  " << test_data.size() << "\n\n";

    // --- build network ---
    nn::Sequential net;

    // conv block 1: 1×28×28 → 16×14×14
    net.add(std::make_shared<nn::Conv2D>(1, 16, 3, 3, 1, 1, 1, 1, backend));
    net.add(std::make_shared<nn::ReLU>(backend));
    net.add(std::make_shared<nn::MaxPool2D>(2, backend));

    // conv block 2: 16×14×14 → 32×7×7
    net.add(std::make_shared<nn::Conv2D>(16, 32, 3, 3, 1, 1, 1, 1, backend));
    net.add(std::make_shared<nn::ReLU>(backend));
    net.add(std::make_shared<nn::MaxPool2D>(2, backend));

    // classifier: 32*7*7 → 128 → 10
    net.add(std::make_shared<nn::Flatten>(backend));
    net.add(std::make_shared<nn::Dense>(32 * 7 * 7, 128, backend));
    net.add(std::make_shared<nn::ReLU>(backend));
    net.add(std::make_shared<nn::Dense>(128, num_classes, backend));

    std::cout << "Network: Conv(1→16) → ReLU → Pool → Conv(16→32) → ReLU → Pool → "
              << "Flatten → Dense(1568→128) → ReLU → Dense(128→10)\n\n";

    // --- training loop ---
    size_t num_train = train_data.size();
    size_t num_batches = num_train / batch_size;

    // shuffle indices
    std::vector<size_t> indices(num_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    double total_time = 0.0;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::steady_clock::now();

        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        size_t epoch_correct = 0;

        for (size_t b = 0; b < num_batches; ++b) {
            // --- assemble batch ---
            // images as NCHW: {batch_size, 1, 28, 28}
            std::vector<float> batch_images(batch_size * 1 * 28 * 28);
            std::vector<float> batch_targets(batch_size * num_classes, 0.0f);

            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = indices[b * batch_size + i];
                // copy image (already flattened 784 floats, treat as 1×28×28)
                std::copy(train_data.images[idx].begin(),
                          train_data.images[idx].end(),
                          batch_images.begin() + i * 784);
                // one-hot target
                batch_targets[i * num_classes + train_data.labels[idx]] = 1.0f;
            }

            nn::Tensor input({batch_size, 1, 28, 28}, backend);
            input.set_data(batch_images);

            // --- forward ---
            nn::Tensor logits = net.forward(input);

            // --- loss + gradient (fused softmax cross-entropy) ---
            nn::Tensor grad({batch_size, num_classes}, backend);
            float loss = nn::softmax_cross_entropy_loss_and_gradient(
                logits, batch_targets, grad, batch_size, num_classes);

            epoch_loss += loss;

            // --- accuracy (on this batch) ---
            std::vector<float> logits_host(batch_size * num_classes);
            backend->download(logits_host.data(), logits.data(),
                              batch_size * num_classes);

            for (size_t i = 0; i < batch_size; ++i) {
                uint8_t pred = nn::predict_class(
                    std::vector<float>(logits_host.begin() + i * num_classes,
                                       logits_host.begin() + (i + 1) * num_classes));
                size_t idx = indices[b * batch_size + i];
                if (pred == train_data.labels[idx]) {
                    ++epoch_correct;
                }
            }

            // --- backward + update ---
            net.backward(grad);
            net.update_parameters(learning_rate);

            if ((b + 1) % 100 == 0) {
                std::cout << "  epoch " << (epoch + 1) << " batch " << (b + 1)
                          << "/" << num_batches
                          << "  loss=" << (epoch_loss / (b + 1)) << "\n";
            }
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        total_time += epoch_sec;
        
        float avg_loss = epoch_loss / num_batches;
        float train_acc = 100.0f * epoch_correct / (num_batches * batch_size);

        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                  << "  loss=" << avg_loss
                  << "  train_acc=" << train_acc << "%"
                  << "  time=" << epoch_sec << "s\n"
                  << "\n";
    }

    std::cout << "Total training time: " << total_time << "s\n";

    // --- test evaluation ---
    std::cout << "\nEvaluating on test set...\n";

    size_t test_correct = 0;
    size_t test_batches = test_data.size() / batch_size;

    for (size_t b = 0; b < test_batches; ++b) {
        std::vector<float> batch_images(batch_size * 784);

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = b * batch_size + i;
            std::copy(test_data.images[idx].begin(),
                      test_data.images[idx].end(),
                      batch_images.begin() + i * 784);
        }

        nn::Tensor input({batch_size, 1, 28, 28}, backend);
        input.set_data(batch_images);

        nn::Tensor logits = net.forward(input);

        std::vector<float> logits_host(batch_size * num_classes);
        backend->download(logits_host.data(), logits.data(),
                          batch_size * num_classes);

        for (size_t i = 0; i < batch_size; ++i) {
            uint8_t pred = nn::predict_class(
                std::vector<float>(logits_host.begin() + i * num_classes,
                                   logits_host.begin() + (i + 1) * num_classes));
            size_t idx = b * batch_size + i;
            if (pred == test_data.labels[idx]) {
                ++test_correct;
            }
        }
    }

    float test_acc = 100.0f * test_correct / (test_batches * batch_size);
    std::cout << "Test accuracy: " << test_acc << "% ("
              << test_correct << "/" << (test_batches * batch_size) << ")\n";

    return 0;
}
