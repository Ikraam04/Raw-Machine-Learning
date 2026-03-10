#include "layers/sequential.h"
#include "layers/conv2d.h"
#include "layers/maxpool2d.h"
#include "layers/globalavgpool.h"
#include "layers/dense.h"
#include "layers/activation.h"
#include "backends/eigen_backend.h"
#include "data/mnist_loader.h"
#include "loss/loss.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>

using namespace nn;
using BackendPtr = std::shared_ptr<Backend>;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string sep(char c = '-', int n = 60) { return std::string(n, c); }

// simple text progress bar:  [=====>    ] 450/1875
static void print_progress(size_t done, size_t total, float loss) {
    const int bar_width = 30;
    float pct = (float)done / total;
    int filled = (int)(pct * bar_width);

    std::cout << "\r  [";
    for (int i = 0; i < bar_width; ++i)
        std::cout << (i < filled ? '=' : (i == filled ? '>' : ' '));
    std::cout << "] "
              << std::setw(4) << done << "/" << total
              << "  loss: " << std::fixed << std::setprecision(4) << loss
              << "  " << std::flush;
}

// evaluate accuracy on up to max_samples examples using batched forward passes
// input tensors are 4D: {batch, 1, 28, 28}
static float evaluate_accuracy(Sequential& network,
                                const MNISTDataset& dataset,
                                BackendPtr backend,
                                size_t max_samples = 10000,
                                size_t batch_size  = 100) {
    size_t correct     = 0;
    size_t num_samples = std::min(max_samples, dataset.size());

    for (size_t i = 0; i < num_samples; i += batch_size) {
        size_t cur = std::min(batch_size, num_samples - i);

        Tensor batch_input({cur, 1, 28, 28}, backend);
        for (size_t b = 0; b < cur; ++b) {
            for (size_t j = 0; j < 784; ++j) {
                batch_input.data()[b * 784 + j] = dataset.images[i + b][j];
            }
        }

        Tensor output = network.forward(batch_input);

        for (size_t b = 0; b < cur; ++b) {
            std::vector<float> out_vec(10);
            for (size_t j = 0; j < 10; ++j) {
                out_vec[j] = output.data()[b * 10 + j];
            }
            if (predict_class(out_vec) == dataset.labels[i + b]) ++correct;
        }
    }

    return 100.0f * correct / num_samples;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << sep('=') << "\n";
    std::cout << "  MNIST digit classification — CNN\n";
    std::cout << sep('=') << "\n\n";

    try {
        // ── data loading ──────────────────────────────────────────────────────
        std::cout << "Loading data...\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        auto train_data = load_mnist("../MNIST_Data/train-images.idx3-ubyte",
                                     "../MNIST_Data/train-labels.idx1-ubyte",
                                     true);
        auto test_data  = load_mnist("../MNIST_Data/t10k-images.idx3-ubyte",
                                     "../MNIST_Data/t10k-labels.idx1-ubyte",
                                     true);
        auto t1 = std::chrono::high_resolution_clock::now();
        double load_s = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "  train : " << train_data.size() << " images\n";
        std::cout << "  test  : " << test_data.size()  << " images\n";
        std::cout << "  format: 28x28, 1 channel, normalised [0,1]\n";
        std::cout << "  loaded in " << std::fixed << std::setprecision(2)
                  << load_s << "s\n\n";

        // ── network ───────────────────────────────────────────────────────────
        auto backend = std::make_shared<EigenBackend>();

        // architecture constants — keep them in one place so the Dense sizes
        // below stay correct if you change the conv/pool params
        const int  C1 = 32, C2 = 64;   // filter counts
        const int  H0 = 28, W0 = 28;   // input spatial size
        const int  H2 = H0/2, W2 = W0/2;  // after pool1 (conv1 keeps size with pad=1)
        const int  H4 = H2/2, W4 = W2/2;  // after pool2

        Sequential network;

        // block 1
        network.add(std::make_shared<Conv2D>(1,  C1, 3, 3, 1, 1, 1, 1, backend));
        network.add(std::make_shared<ReLU>(backend));
        network.add(std::make_shared<MaxPool2D>(2, backend));

        // block 2
        network.add(std::make_shared<Conv2D>(C1, C2, 3, 3, 1, 1, 1, 1, backend));
        network.add(std::make_shared<ReLU>(backend));
        network.add(std::make_shared<MaxPool2D>(2, backend));

        // global average pool: {batch,64,7,7} -> {batch,64}  (no Flatten needed)
        network.add(std::make_shared<GlobalAvgPool>(backend));

        // classifier: just one dense layer directly from C2 features to 10 classes
        network.add(std::make_shared<Dense>(C2, 10, backend));
        network.add(std::make_shared<Softmax>(backend));

        // parameter counts
        size_t p_conv1 = (size_t)(1  * C1 * 3 * 3 + C1);
        size_t p_conv2 = (size_t)(C1 * C2 * 3 * 3 + C2);
        size_t p_dense = (size_t)(C2 * 10 + 10);
        size_t p_total = p_conv1 + p_conv2 + p_dense;

        std::cout << "Architecture:\n";
        std::cout << "  input              1  x 28 x 28\n";
        std::cout << "  Conv2D (3x3 p=1)   " << std::setw(2) << 1
                  << " -> " << std::setw(2) << C1
                  << "   output: " << C1 << " x " << H0 << " x " << W0
                  << "   params: " << p_conv1 << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  MaxPool2D (2x2)         output: "
                  << C1 << " x " << H2 << " x " << W2 << "\n";
        std::cout << "  Conv2D (3x3 p=1)   " << std::setw(2) << C1
                  << " -> " << std::setw(2) << C2
                  << "   output: " << C2 << " x " << H2 << " x " << W2
                  << "   params: " << p_conv2 << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  MaxPool2D (2x2)         output: "
                  << C2 << " x " << H4 << " x " << W4 << "\n";
        std::cout << "  GlobalAvgPool           output: " << C2 << "\n";
        std::cout << "  Dense              " << std::setw(4) << C2
                  << " -> 10      params: " << p_dense << "\n";
        std::cout << "  Softmax\n";
        std::cout << "  " << sep() << "\n";
        std::cout << "  total trainable params: " << p_total << "\n\n";

        // ── hyperparameters ───────────────────────────────────────────────────
        const int    epochs      = 5;        // Adam + GAP converges fast
        const float  lr          = 0.001f;  // Adam default
        const size_t batch_size  = 32;
        const size_t n_train     = train_data.size();
        const size_t num_batches = (n_train + batch_size - 1) / batch_size;

        std::cout << "Hyperparameters:\n";
        std::cout << "  optimiser   : Adam (beta1=0.9, beta2=0.999, eps=1e-8)\n";
        std::cout << "  epochs      : " << epochs     << "\n";
        std::cout << "  batch size  : " << batch_size  << "\n";
        std::cout << "  learning rate: " << lr          << "\n";
        std::cout << "  batches/epoch: " << num_batches << "\n\n";

        // ── training ──────────────────────────────────────────────────────────
        std::cout << sep('=') << "\n";
        std::cout << "  Training\n";
        std::cout << sep('=') << "\n\n";

        std::vector<size_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());

        float best_test_acc = 0.0f;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto ep_start = std::chrono::high_resolution_clock::now();

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";

            std::shuffle(indices.begin(), indices.end(), gen);

            float epoch_loss   = 0.0f;
            size_t n_processed = 0;

            for (size_t bi = 0; bi < num_batches; ++bi) {
                size_t bs  = std::min(batch_size, n_train - bi * batch_size);

                // build 4D batch: {bs, 1, 28, 28}
                Tensor batch_input({bs, 1, 28, 28}, backend);
                std::vector<float> targets(bs * 10);

                for (size_t b = 0; b < bs; ++b) {
                    size_t idx = indices[bi * batch_size + b];
                    for (size_t j = 0; j < 784; ++j) {
                        batch_input.data()[b * 784 + j] = train_data.images[idx][j];
                    }
                    auto onehot = label_to_onehot(train_data.labels[idx], 10);
                    for (size_t j = 0; j < 10; ++j) {
                        targets[b * 10 + j] = onehot[j];
                    }
                }

                // forward
                Tensor output = network.forward(batch_input);

                // loss + gradient
                Tensor grad({bs, 10}, backend);
                float  loss = cross_entropy_loss(output, targets, bs, 10);
                cross_entropy_gradient(output, targets, grad, bs, 10);

                epoch_loss  += loss * (float)bs;
                n_processed += bs;

                // backward + update
                network.backward(grad);
                network.update_parameters(lr);

                // progress bar every 50 batches or at the end
                if ((bi + 1) % 50 == 0 || bi == num_batches - 1) {
                    print_progress(bi + 1, num_batches, epoch_loss / n_processed);
                }
            }
            std::cout << "\n";  // end progress bar line

            auto ep_end = std::chrono::high_resolution_clock::now();
            double ep_s = std::chrono::duration<double>(ep_end - ep_start).count();

            float avg_loss  = epoch_loss / n_processed;
            float train_acc = evaluate_accuracy(network, train_data, backend, 10000, 100);
            float test_acc  = evaluate_accuracy(network, test_data,  backend, 10000, 100);
            best_test_acc   = std::max(best_test_acc, test_acc);

            std::cout << "  avg loss : " << std::fixed << std::setprecision(4) << avg_loss  << "\n";
            std::cout << "  train acc: " << std::fixed << std::setprecision(2) << train_acc << "%\n";
            std::cout << "  test  acc: " << std::fixed << std::setprecision(2) << test_acc  << "%";
            if (test_acc == best_test_acc) std::cout << "  <-- best";
            std::cout << "\n";
            std::cout << "  time     : " << std::fixed << std::setprecision(1) << ep_s << "s\n\n";
        }

        // ── final metrics ─────────────────────────────────────────────────────
        std::cout << sep('=') << "\n";
        std::cout << "  Results\n";
        std::cout << sep('=') << "\n\n";

        float final_acc = evaluate_accuracy(network, test_data, backend, 10000, 100);
        std::cout << "  final test accuracy : " << std::fixed << std::setprecision(2)
                  << final_acc << "%\n";
        std::cout << "  best  test accuracy : " << std::fixed << std::setprecision(2)
                  << best_test_acc << "%\n\n";

        // sample predictions table
        std::cout << "  Sample predictions (first 20 test images):\n\n";
        std::cout << "  img | true | pred | confidence\n";
        std::cout << "  ----|------|------|------------\n";

        for (size_t i = 0; i < 20; ++i) {
            Tensor input({1, 1, 28, 28}, backend);
            input.set_data(test_data.images[i]);

            Tensor output  = network.forward(input);
            std::vector<float> out_vec(output.data(), output.data() + 10);
            uint8_t pred   = predict_class(out_vec);
            float   conf   = out_vec[pred];
            bool    correct = (pred == test_data.labels[i]);

            std::cout << "  " << std::setw(3) << i
                      << " | " << std::setw(4) << (int)test_data.labels[i]
                      << " | " << std::setw(4) << (int)pred
                      << " | " << std::fixed << std::setprecision(3) << conf
                      << (correct ? "  ok" : "  WRONG") << "\n";
        }

        std::cout << "\ndone.\n\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
