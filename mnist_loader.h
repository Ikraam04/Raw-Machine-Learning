#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace nn {

/**
 * Simple structure to hold MNIST dataset
 */
struct MNISTDataset {
    std::vector<std::vector<float>> images;  // Each image is 784 floats (28x28)
    std::vector<uint8_t> labels;              // Each label is 0-9
    
    size_t size() const { return images.size(); }
};

/**
 * Load MNIST dataset from binary files.
 * 
 * Files can be downloaded from: http://yann.lecun.com/exdb/mnist/
 * 
 * @param images_path: Path to images file (e.g., "data/train-images-idx3-ubyte")
 * @param labels_path: Path to labels file (e.g., "data/train-labels-idx1-ubyte")
 * @param normalize: If true, scales pixel values from [0, 255] to [0, 1]
 * @return: MNISTDataset containing images and labels
 */
MNISTDataset load_mnist(const std::string& images_path, 
                        const std::string& labels_path,
                        bool normalize = true);

/**
 * Helper: Convert label (0-9) to one-hot encoding
 * Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
 */
std::vector<float> label_to_onehot(uint8_t label, size_t num_classes = 10);

/**
 * Helper: Convert network output to predicted class
 * Returns index of maximum value
 */
uint8_t predict_class(const std::vector<float>& output);

} // namespace nn