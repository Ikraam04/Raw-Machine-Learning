#include "core/tensor.h"
#include "backends/eigen_backend.h"
#include "backends/cuda_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

using namespace nn;

// checks two vectors are equal within a tolerance, prints PASS or FAIL
static void expect(const std::string& test, const std::vector<float>& got,
                   const std::vector<float>& expected, float tol = 1e-4f) {
    if (got.size() != expected.size()) {
        std::cout << "  FAIL [" << test << "] size mismatch\n";
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > tol) {
            std::cout << "  FAIL [" << test << "] at index " << i
                      << ": got " << got[i] << ", expected " << expected[i] << "\n";
            return;
        }
    }
    std::cout << "  PASS [" << test << "]\n";
}

// helper: download tensor to a std::vector for inspection
static std::vector<float> to_vec(const Tensor& t) {
    std::vector<float> out(t.size());
    t.backend()->download(out.data(), t.data(), t.size());
    return out;
}

void test_backend(Tensor::BackendPtr backend, const std::string& name) {
    std::cout << "\n=== " << name << " ===\n";

    // memory - fill
    Tensor t_fill({4}, backend);
    t_fill.fill(7.0f);
    expect("fill", to_vec(t_fill), {7, 7, 7, 7});

    //memory - upload (set_data)
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    Tensor t_up({2, 3}, backend);
    t_up.set_data(src);
    expect("upload", to_vec(t_up), src);

    //  memory - device-to-device copy (copy constructor)
    Tensor t_copy = t_up;
    expect("copy", to_vec(t_copy), src);

    // change original, verify copy is independent
    t_up.fill(0.0f);
    expect("copy independence", to_vec(t_copy), src);

    // add
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {10, 20, 30, 40};
    Tensor a({4}, backend); a.set_data(a_data);
    Tensor b({4}, backend); b.set_data(b_data);
    Tensor r_add = a.add(b);
    expect("add", to_vec(r_add), {11, 22, 33, 44});

    // multiply (element-wise)
    Tensor r_mul = a.multiply(b);
    expect("multiply", to_vec(r_mul), {10, 40, 90, 160});

    // scale
    a.multiply_(2.0f);   // in-place scale by 2
    expect("scale", to_vec(a), {2, 4, 6, 8});

    // add_ (in-place)
    Tensor c({4}, backend); c.set_data({1, 1, 1, 1});
    a.add_(c);
    expect("add_ (in-place)", to_vec(a), {3, 5, 7, 9});

    //relu
    Tensor t_relu({6}, backend); t_relu.set_data({-2, -1, 0, 1, 2, 3});
    Tensor r_relu = t_relu.relu();
    expect("relu", to_vec(r_relu), {0, 0, 0, 1, 2, 3});

    // relu_derivative
    Tensor r_relu_d = t_relu.relu_derivative();
    expect("relu_derivative", to_vec(r_relu_d), {0, 0, 0, 1, 1, 1});

    //sigmoid
    Tensor t_sig({3}, backend); t_sig.set_data({0.0f, 2.0f, -2.0f});
    Tensor r_sig = t_sig.sigmoid();
    expect("sigmoid", to_vec(r_sig), {0.5f, 0.8808f, 0.1192f});

    // sigmoid_derivative (input is already sigmoid output)
    // s*(1-s): 0.5*(0.5)=0.25, 0.8808*(0.1192)=0.1050, 0.1192*(0.8808)=0.1050
    Tensor r_sig_d = r_sig.sigmoid_derivative();
    expect("sigmoid_derivative", to_vec(r_sig_d), {0.25f, 0.1050f, 0.1050f});

    //  matmul
    // A (2x3) * B (3x2) = result (2x2)
    // A = [[1,2,3],[4,5,6]]
    // B = [[7,8],[9,10],[11,12]]
    // result[0][0] = 1*7 + 2*9  + 3*11 = 7+18+33  = 58
    // result[0][1] = 1*8 + 2*10 + 3*12 = 8+20+36  = 64
    // result[1][0] = 4*7 + 5*9  + 6*11 = 28+45+66 = 139
    // result[1][1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    Tensor mat_a({2, 3}, backend); mat_a.set_data({1,2,3,4,5,6});
    Tensor mat_b({3, 2}, backend); mat_b.set_data({7,8,9,10,11,12});
    Tensor mat_r = mat_a.matmul(mat_b);
    expect("matmul", to_vec(mat_r), {58, 64, 139, 154});

    // transpose 
    // A (2x3):          A^T (3x2):
    // [[1, 2, 3],       [[1, 4],
    //  [4, 5, 6]]        [2, 5],
    //                    [3, 6]]
    // flat row-major: 1,2,3,4,5,6  ->  1,4,2,5,3,6
    Tensor t_in({2, 3}, backend); t_in.set_data({1,2,3,4,5,6});
    Tensor t_out = t_in.transpose();
    expect("transpose", to_vec(t_out), {1,4,2,5,3,6});

    // im2col
    // input: 1 image, 1 channel, 3x3
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    // kernel 2x2, no padding, stride 1 -> out_h=2, out_w=2
    // 4 output positions, each receptive field has 4 values (1ch * 2*2)
    //
    // col matrix (4 rows x 4 cols):
    // position (0,0): top-left     2x2 patch -> [1,2,4,5]
    // position (0,1): top-right    2x2 patch -> [2,3,5,6]
    // position (1,0): bottom-left  2x2 patch -> [4,5,7,8]
    // position (1,1): bottom-right 2x2 patch -> [5,6,8,9]
    {
        int batch=1, in_ch=1, h=3, w=3;
        int kh=2, kw=2, out_h=2, out_w=2;
        int pad_h=0, pad_w=0, str_h=1, str_w=1;
        int col_size = batch * out_h * out_w * in_ch * kh * kw;  // 4*4 = 16

        Tensor im({1,1,3,3}, backend); im.set_data({1,2,3,4,5,6,7,8,9});
        Tensor col({(size_t)col_size}, backend);

        backend->im2col(im.data(), col.data(),
                        batch, in_ch, h, w,
                        kh, kw, out_h, out_w,
                        pad_h, pad_w, str_h, str_w);

        expect("im2col", to_vec(col), {1,2,4,5,
                                       2,3,5,6,
                                       4,5,7,8,
                                       5,6,8,9});

        // col2im
        // reverse of the above: scatter col back into a 3x3 input
        // each pixel accumulates from every patch it appeared in:
        // pixel (0,0) -> only patch (0,0) -> val 1
        // pixel (0,1) -> patches (0,0),(0,1) -> 2+2=4
        // pixel (0,2) -> only patch (0,1) -> 3
        // pixel (1,0) -> patches (0,0),(1,0) -> 4+4=8
        // pixel (1,1) -> all 4 patches -> 5+5+5+5=20
        // pixel (1,2) -> patches (0,1),(1,1) -> 6+6=12
        // pixel (2,0) -> only patch (1,0) -> 7
        // pixel (2,1) -> patches (1,0),(1,1) -> 8+8=16
        // pixel (2,2) -> only patch (1,1) -> 9
        Tensor im_out({1,1,3,3}, backend);
        im_out.fill(0.0f);  // must be zeroed before col2im
        backend->col2im(col.data(), im_out.data(),
                        batch, in_ch, h, w,
                        kh, kw, out_h, out_w,
                        pad_h, pad_w, str_h, str_w);
        expect("col2im", to_vec(im_out), {1,4,3, 8,20,12, 7,16,9});
    }
}

int main() {
    try {
        auto eigen = std::make_shared<EigenBackend>();
        test_backend(eigen, "Eigen (CPU)");

        auto cuda = std::make_shared<CudaBackend>();
        test_backend(cuda, "CUDA (GPU)");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
