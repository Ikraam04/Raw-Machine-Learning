#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace nn {

// Helper: Read 32-bit big-endian integer from file
static uint32_t read_int32(std::ifstream& file) {
    uint32_t value = 0;
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    
    // Convert from big-endian to host byte order
    value = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    return value;
}

MNISTDataset load_mnist(const std::string& images_path, 
                        const std::string& labels_path,
                        bool normalize) {
    MNISTDataset dataset;
    
    // ============================================================
    // Load Images
    // ============================================================
    
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file.is_open()) {
        throw std::runtime_error("Could not open images file: " + images_path);
    }
    
    // Read header
    uint32_t magic_number = read_int32(images_file);
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in images file. Expected 2051, got " + 
                                std::to_string(magic_number));
    }
    
    uint32_t num_images = read_int32(images_file);
    uint32_t num_rows = read_int32(images_file);
    uint32_t num_cols = read_int32(images_file);
    
    std::cout << "Loading MNIST images...\n";
    std::cout << "  Number of images: " << num_images << "\n";
    std::cout << "  Image size: " << num_rows << "x" << num_cols << "\n";
    
    if (num_rows != 28 || num_cols != 28) {
        throw std::runtime_error("Expected 28x28 images");
    }
    
    // Read pixel data
    dataset.images.resize(num_images);
    const size_t image_size = num_rows * num_cols;
    
    for (size_t i = 0; i < num_images; ++i) {
        dataset.images[i].resize(image_size);
        
        // Read raw bytes
        std::vector<uint8_t> pixels(image_size);
        images_file.read(reinterpret_cast<char*>(pixels.data()), image_size);
        
        // Convert to float and optionally normalize
        for (size_t j = 0; j < image_size; ++j) {
            if (normalize) {
                // Scale from [0, 255] to [0, 1]
                dataset.images[i][j] = pixels[j] / 255.0f;
            } else {
                dataset.images[i][j] = static_cast<float>(pixels[j]);
            }
        }
        
        // Progress indicator
        if ((i + 1) % 10000 == 0) {
            std::cout << "  Loaded " << (i + 1) << " images...\n";
        }
    }
    
    images_file.close();
    
    // ============================================================
    // Load Labels
    // ============================================================
    
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open()) {
        throw std::runtime_error("Could not open labels file: " + labels_path);
    }
    
    // Read header
    magic_number = read_int32(labels_file);
    if (magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in labels file. Expected 2049, got " + 
                                std::to_string(magic_number));
    }
    
    uint32_t num_labels = read_int32(labels_file);
    
    if (num_labels != num_images) {
        throw std::runtime_error("Number of labels doesn't match number of images");
    }
    
    std::cout << "Loading MNIST labels...\n";
    std::cout << "  Number of labels: " << num_labels << "\n";
    
    // Read label data
    dataset.labels.resize(num_labels);
    labels_file.read(reinterpret_cast<char*>(dataset.labels.data()), num_labels);
    
    labels_file.close();
    
    std::cout << "✓ MNIST dataset loaded successfully!\n\n";
    
    return dataset;
}

std::vector<float> label_to_onehot(uint8_t label, size_t num_classes) {
    if (label >= num_classes) {
        throw std::runtime_error("Label " + std::to_string(label) + 
                                " exceeds num_classes " + std::to_string(num_classes));
    }
    
    std::vector<float> onehot(num_classes, 0.0f);
    onehot[label] = 1.0f;
    return onehot;
}

uint8_t predict_class(const std::vector<float>& output) {
    if (output.empty()) {
        throw std::runtime_error("Cannot predict class from empty output");
    }
    
    // Find index of maximum value
    auto max_it = std::max_element(output.begin(), output.end());
    return static_cast<uint8_t>(std::distance(output.begin(), max_it));
}

} // namespace nn