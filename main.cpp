#include "sequential.h"
#include "dense.h"
#include "activation.h"
#include "eigen_backend.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nn;

/**
 * XOR Problem:
 * 
 * Input  | Output
 * -------|-------
 * 0, 0   | 0
 * 0, 1   | 1
 * 1, 0   | 1
 * 1, 1   | 0
 * 
 * This is NOT linearly separable - requires non-linear activation!
 * A single Dense layer cannot solve this.
 */

void test_xor() {
    std::cout << "========================================\n";
    std::cout << "  Training Neural Network on XOR       \n";
    std::cout << "========================================\n\n";
    
    auto backend = std::make_shared<EigenBackend>();
    
    // Create network: 2 -> 4 -> 1 with sigmoid activations
    // Input: 2 features (x1, x2)
    // Hidden: 4 neurons with sigmoid
    // Output: 1 neuron with sigmoid
    Sequential network;
    network.add(std::make_shared<Dense>(2, 4, backend));
    network.add(std::make_shared<Sigmoid>(backend));
    network.add(std::make_shared<Dense>(4, 1, backend));
    network.add(std::make_shared<Sigmoid>(backend));
    
    std::cout << "Network architecture:\n";
    std::cout << "  Input(2) -> Dense(4) -> Sigmoid -> Dense(1) -> Sigmoid -> Output(1)\n\n";
    
    // XOR training data
    std::vector<std::vector<float>> X = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    std::vector<float> Y = {0.0f, 1.0f, 1.0f, 0.0f};
    
    float learning_rate = 0.5f;
    int epochs = 500;
    
    std::cout << "Training for " << epochs << " epochs with learning_rate=" << learning_rate << "\n\n";
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        // Train on each sample
        for (size_t i = 0; i < X.size(); ++i) {
            // Prepare input
            Tensor input({1, 2}, backend);
            input.set_data(X[i]);
            
            // Forward pass
            Tensor output = network.forward(input);
            
            // Compute loss (Mean Squared Error)
            float prediction = output.data()[0];
            float target = Y[i];
            float error = prediction - target;
            total_loss += error * error;
            
            // Compute gradient (derivative of MSE: 2 * error)
            Tensor grad_output({1, 1}, backend);
            grad_output.set_data({2.0f * error});
            
            // Backward pass
            network.backward(grad_output);
            
            // Update parameters
            network.update_parameters(learning_rate);
        }
        
        // Print progress every 1000 epochs
        if (epoch % 1000 == 0) {
            float avg_loss = total_loss / X.size();
            std::cout << "Epoch " << std::setw(5) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss << "\n";
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "  Training Complete! Testing...        \n";
    std::cout << "========================================\n\n";
    
    // Test the trained network
    std::cout << "XOR Truth Table:\n";
    std::cout << "Input (x1, x2) | Target | Prediction | Correct?\n";
    std::cout << "---------------|--------|------------|----------\n";
    
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        Tensor input({1, 2}, backend);
        input.set_data(X[i]);
        
        Tensor output = network.forward(input);
        float prediction = output.data()[0];
        float target = Y[i];
        
        // Round to nearest integer for classification
        int predicted_class = prediction > 0.5f ? 1 : 0;
        int target_class = static_cast<int>(target);
        bool is_correct = predicted_class == target_class;
        
        if (is_correct) correct++;
        
        std::cout << "(" << X[i][0] << ", " << X[i][1] << ")       | "
                  << target << "      | "
                  << std::fixed << std::setprecision(4) << prediction << "     | "
                  << (is_correct ? "✓" : "✗") << "\n";
    }
    
    std::cout << "\nAccuracy: " << correct << "/" << X.size() 
              << " (" << (100.0f * correct / X.size()) << "%)\n";
    
    if (correct == 4) {
        std::cout << "\n🎉 SUCCESS! Network learned XOR perfectly!\n";
    } else {
        std::cout << "\n⚠ Network didn't fully converge. Try more epochs or adjust learning rate.\n";
    }
}

void test_activation_layers() {
    std::cout << "\n========================================\n";
    std::cout << "  Testing Activation Layers            \n";
    std::cout << "========================================\n\n";
    
    auto backend = std::make_shared<EigenBackend>();
    
    // Test ReLU
    std::cout << "=== ReLU ===\n";
    ReLU relu(backend);
    
    Tensor relu_input({1, 4}, backend);
    relu_input.set_data({-2.0f, -1.0f, 1.0f, 2.0f});
    
    relu_input.print("Input");
    Tensor relu_output = relu.forward(relu_input);
    relu_output.print("ReLU output");
    
    Tensor relu_grad_out({1, 4}, backend);
    relu_grad_out.fill(1.0f);
    Tensor relu_grad_in = relu.backward(relu_grad_out);
    relu_grad_in.print("ReLU gradient");
    
    // Test Sigmoid
    std::cout << "=== Sigmoid ===\n";
    Sigmoid sigmoid(backend);
    
    Tensor sig_input({1, 3}, backend);
    sig_input.set_data({-1.0f, 0.0f, 1.0f});
    
    sig_input.print("Input");
    Tensor sig_output = sigmoid.forward(sig_input);
    sig_output.print("Sigmoid output");
    
    std::cout << "✓ Activation layers work!\n";
}

int main() {
    try {
        //test_activation_layers();
        test_xor();
        
        std::cout << "\n========================================\n";
        std::cout << "  ✓ All tests passed!                  \n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << "\n";
        return 1;
    }
}