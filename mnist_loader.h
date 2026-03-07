#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace nn {

struct MNISTDataset {
    std::vector<std::vector<float>> images;  // each image is 784 floats (28x28 flattened)
    std::vector<uint8_t> labels;             // 0-9

    size_t size() const { return images.size(); }
};

// loads MNIST from the binary idx file format
// normalize=true scales pixels from [0, 255] to [0, 1]
MNISTDataset load_mnist(const std::string& images_path,
                        const std::string& labels_path,
                        bool normalize = true);

// converts label (0-9) to one-hot vector
std::vector<float> label_to_onehot(uint8_t label, size_t num_classes = 10);

// argmax of the output vector = predicted class
uint8_t predict_class(const std::vector<float>& output);

} // namespace nn