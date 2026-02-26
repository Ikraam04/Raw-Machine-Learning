#include "sequential.h"
#include "dense.h"
#include "activation.h"
#include "eigen_backend.h"
#include "mnist_loader.h"
#include "loss.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <chrono>

using namespace nn;
using BackendPtr = std::shared_ptr<Backend>;// alias
/**
 * Evaluate accuracy on a dataset (using batching for speed) 
 */
float evaluate_accuracy(Sequential& network, 
                       const MNISTDataset& dataset,
                       BackendPtr backend,
                       size_t max_samples = 10000,
                       size_t batch_size = 100) {
    size_t correct = 0;
    size_t num_samples = std::min(max_samples, dataset.size());
    
    for (size_t i = 0; i < num_samples; i += batch_size) {
        size_t current_batch = std::min(batch_size, num_samples - i);
        
        // Create batch
        Tensor batch_input({current_batch, 784}, backend);
        for (size_t b = 0; b < current_batch; ++b) {
            for (size_t j = 0; j < 784; ++j) {
                batch_input.data()[b * 784 + j] = dataset.images[i + b][j];
            }
        }
        
        // Forward pass
        Tensor output = network.forward(batch_input);
        
        // Check predictions
        for (size_t b = 0; b < current_batch; ++b) {
            std::vector<float> output_vec(10);
            for (size_t j = 0; j < 10; ++j) {
                output_vec[j] = output.data()[b * 10 + j];
            }
            
            uint8_t predicted = predict_class(output_vec);
            if (predicted == dataset.labels[i + b]) {
                correct++;
            }
        }
    }
    
    return 100.0f * correct / num_samples;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  MNIST Digit Classification           \n";
    std::cout << "  (With Mini-Batch Training)           \n";
    std::cout << "========================================\n\n";
    
    try {
        // ============================================================
        // Load MNIST Dataset
        // ============================================================
        
        std::cout << "Loading training data...\n";
        auto train_data = load_mnist("../MNIST_Data/train-images.idx3-ubyte",
                                    "../MNIST_Data/train-labels.idx1-ubyte",
                                    true);  // normalize to [0, 1]
        
        std::cout << "Loading test data...\n";
        auto test_data = load_mnist("../MNIST_Data/t10k-images.idx3-ubyte",
                                   "../MNIST_Data/t10k-labels.idx1-ubyte",
                                   true);
        
        // ============================================================
        // Create Network
        // ============================================================
        
        auto backend = std::make_shared<EigenBackend>();
        
        Sequential network;
        network.add(std::make_shared<Dense>(784, 128, backend));  // Input: 28×28=784
        network.add(std::make_shared<ReLU>(backend));
        network.add(std::make_shared<Dense>(128, 64, backend));
        network.add(std::make_shared<ReLU>(backend));
        network.add(std::make_shared<Dense>(64, 10, backend));    // Output: 10 classes
        network.add(std::make_shared<Softmax>(backend));          // Softmax for probabilities!
        
        std::cout << "Network Architecture:\n";
        std::cout << "  Input(784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax\n";
        std::cout << "  Total parameters: ~100K\n\n";
        
        // ============================================================
        // Training Configuration
        // ============================================================
        
        const int epochs = 10;
        const float learning_rate = 0.05f;  // Higher learning rate works better with softmax
        const size_t batch_size = 32;       // Mini-batch size
        const size_t train_samples = 60000;
        const size_t num_batches = (train_samples + batch_size - 1) / batch_size;
        
        std::cout << "Training Configuration:\n";
        std::cout << "  Epochs: " << epochs << "\n";
        std::cout << "  Learning rate: " << learning_rate << "\n";
        std::cout << "  Batch size: " << batch_size << "\n";
        std::cout << "  Training samples: " << train_samples << "\n";
        std::cout << "  Batches per epoch: " << num_batches << "\n\n";
        
        // ============================================================
        // Training Loop
        // ============================================================
        
        std::cout << "Starting training...\n\n";
        
        // For shuffling
        std::vector<size_t> indices(train_samples);
        for (size_t i = 0; i < train_samples; ++i) indices[i] = i;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "=== Epoch " << (epoch + 1) << "/" << epochs << " ===\n";
            
            // Shuffle training data
            std::shuffle(indices.begin(), indices.end(), gen);
            
            float epoch_loss = 0.0f;
            size_t samples_processed = 0;
            
            // Process data in mini-batches
            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                size_t batch_start = batch_idx * batch_size;
                size_t current_batch_size = std::min(batch_size, train_samples - batch_start);
                
                // ============================================================
                // Prepare Batch
                // ============================================================
                
                Tensor batch_input({current_batch_size, 784}, backend);
                std::vector<float> batch_targets(current_batch_size * 10);
                
                for (size_t b = 0; b < current_batch_size; ++b) {
                    size_t idx = indices[batch_start + b];
                    
                    // Copy image data
                    for (size_t j = 0; j < 784; ++j) {
                        batch_input.data()[b * 784 + j] = train_data.images[idx][j];
                    }
                    
                    // Copy target (one-hot)
                    auto target = label_to_onehot(train_data.labels[idx], 10);
                    for (size_t j = 0; j < 10; ++j) {
                        batch_targets[b * 10 + j] = target[j];
                    }
                }
                
                // ============================================================
                // Forward Pass
                // ============================================================
                
                Tensor output = network.forward(batch_input);
                
                // ============================================================
                // Compute Loss and Gradient (Cross-Entropy)
                // ============================================================
                
                Tensor grad_output({current_batch_size, 10}, backend);
                float loss = cross_entropy_loss(output, batch_targets, current_batch_size, 10);
                cross_entropy_gradient(output, batch_targets, grad_output, current_batch_size, 10);
                
                epoch_loss += loss * current_batch_size;  // Accumulate total loss
                
                // ============================================================
                // Backward Pass
                // ============================================================
                
                network.backward(grad_output);
                
                // ============================================================
                // Update Parameters (once per batch!)
                // ============================================================
                
                network.update_parameters(learning_rate);
                
                samples_processed += current_batch_size;
                
                // Progress update every ~10,000 samples
                if ((batch_idx + 1) % 300 == 0 || batch_idx == num_batches - 1) {
                    float avg_loss = epoch_loss / samples_processed;
                    std::cout << "  Batch: " << std::setw(4) << (batch_idx + 1) << "/" << num_batches
                              << " | Samples: " << std::setw(5) << samples_processed 
                              << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss;
                    std::cout << "\n";
                }
            }
            
            // ============================================================
            // Epoch Summary
            // ============================================================
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            std::cout << "  Epoch time: " << duration.count() << " seconds\n";
            
            // Evaluate accuracy
            std::cout << "  Evaluating...\n";
            float train_acc = evaluate_accuracy(network, train_data, backend, 10000, 100);
            float test_acc = evaluate_accuracy(network, test_data, backend, 10000, 100);
            
            std::cout << "  Train Accuracy: " << std::fixed << std::setprecision(2) << train_acc << "%\n";
            std::cout << "  Test Accuracy: " << std::fixed << std::setprecision(2) << test_acc << "%\n\n";
        }
        
        // ============================================================
        // Final Evaluation
        // ============================================================
        
        std::cout << "========================================\n";
        std::cout << "  Training Complete!                   \n";
        std::cout << "========================================\n\n";
        
        std::cout << "Final Test Accuracy: ";
        float final_acc = evaluate_accuracy(network, test_data, backend, 10000, 100);
        std::cout << std::fixed << std::setprecision(2) << final_acc << "%\n\n";
        
        // Show some example predictions
        std::cout << "Sample Predictions:\n";
        std::cout << "Image | True | Predicted | Confidence\n";
        std::cout << "------|------|-----------|------------\n";
        
        for (size_t i = 0; i < 20; ++i) {
            Tensor input({1, 784}, backend);
            input.set_data(test_data.images[i]);
            
            Tensor output = network.forward(input);
            std::vector<float> output_vec(output.data(), output.data() + 10);
            uint8_t predicted = predict_class(output_vec);
            float confidence = output_vec[predicted];
            
            std::cout << std::setw(5) << i << " | " 
                      << std::setw(4) << static_cast<int>(test_data.labels[i]) << " | "
                      << std::setw(9) << static_cast<int>(predicted) << " | "
                      << std::fixed << std::setprecision(3) << confidence;
            
            if (predicted == test_data.labels[i]) {
                std::cout << " ✓\n";
            } else {
                std::cout << " ✗\n";
            }
        }
        
        std::cout << "\n🎉 MNIST training successful!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}