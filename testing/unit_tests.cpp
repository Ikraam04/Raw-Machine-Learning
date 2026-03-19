#include "core/tensor.h"
#include "backends/eigen_backend.h"
#include "backends/cuda_backend.h"
#include "layers/dense.h"
#include "layers/conv2d.h"
#include "layers/maxpool2d.h"
#include "layers/globalavgpool.h"
#include "layers/activation.h"
#include "layers/flatten.h"
#include "layers/sequential.h"
#include "loss/loss.h"
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

void test_new_primitives(Tensor::BackendPtr backend, const std::string& name) {
    std::cout << "\n=== " << name << " (new primitives) ===\n";

    // bias_add: 3 rows x 2 cols, bias = {10, 20}
    {
        Tensor data({3, 2}, backend);
        data.set_data({1,2, 3,4, 5,6});
        Tensor bias({1, 2}, backend);
        bias.set_data({10, 20});
        backend->bias_add(data.data(), bias.data(), 3, 2);
        expect("bias_add", to_vec(data), {11,22, 13,24, 15,26});
    }

    // sum_rows: 3 rows x 2 cols → {sum col0, sum col1}
    {
        Tensor input({3, 2}, backend);
        input.set_data({1,2, 3,4, 5,6});
        Tensor output({1, 2}, backend);
        output.fill(0.0f);
        backend->sum_rows(output.data(), input.data(), 3, 2);
        expect("sum_rows", to_vec(output), {9, 12});
    }

    // adam_update: single step on 2 params
    {
        Tensor param({2}, backend); param.set_data({1.0f, 2.0f});
        Tensor grad({2}, backend);  grad.set_data({0.1f, 0.2f});
        Tensor m({2}, backend);     m.fill(0.0f);
        Tensor v({2}, backend);     v.fill(0.0f);

        float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - beta1;   // t=1
        float bc2 = 1.0f - beta2;

        backend->adam_update(param.data(), grad.data(),
                              m.data(), v.data(),
                              lr, beta1, beta2, bc1, bc2, eps, 2);

        // after 1 step: m = 0.1*g, v = 0.001*g^2
        // m_hat = m / bc1 = g, v_hat = v / bc2 = g^2
        // param -= lr * g / (|g| + eps) ≈ param -= lr * sign(g) = param - 0.001
        auto p = to_vec(param);
        bool ok = std::fabs(p[0] - 0.999f) < 1e-3f && std::fabs(p[1] - 1.999f) < 1e-3f;
        std::cout << "  " << (ok ? "PASS" : "FAIL") << " [adam_update]\n";
    }

    // nhwc_to_nchw: 1 batch, 2 channels, 2x2
    // NHWC input: [h0w0c0, h0w0c1, h0w1c0, h0w1c1, h1w0c0, h1w0c1, h1w1c0, h1w1c1]
    //           = [1,      2,      3,      4,      5,      6,      7,      8]
    // NCHW output: c0=[1,3,5,7], c1=[2,4,6,8]
    {
        Tensor nhwc({8}, backend); nhwc.set_data({1,2,3,4,5,6,7,8});
        Tensor nchw({8}, backend);
        backend->nhwc_to_nchw(nhwc.data(), nchw.data(), 1, 2, 2, 2);
        expect("nhwc_to_nchw", to_vec(nchw), {1,3,5,7, 2,4,6,8});
    }

    // nchw_to_nhwc: reverse of above
    {
        Tensor nchw({8}, backend); nchw.set_data({1,3,5,7, 2,4,6,8});
        Tensor nhwc({8}, backend);
        backend->nchw_to_nhwc(nchw.data(), nhwc.data(), 1, 2, 2, 2);
        expect("nchw_to_nhwc", to_vec(nhwc), {1,2,3,4,5,6,7,8});
    }

    // maxpool_forward: 1 batch, 1 channel, 4x4 input, 2x2 pool, stride 2 → 2x2 output
    {
        Tensor input({1,1,4,4}, backend);
        input.set_data({1, 2, 3, 4,
                        5, 6, 7, 8,
                        9,10,11,12,
                       13,14,15,16});
        Tensor output({1,1,2,2}, backend);
        int* indices = backend->allocate_int(4);
        backend->maxpool_forward(input.data(), output.data(), indices,
                                  1, 1, 4, 4, 2, 2, 2, 2, 2, 2);
        expect("maxpool_forward", to_vec(output), {6, 8, 14, 16});
        backend->deallocate_int(indices);
    }

    // maxpool_backward
    {
        // from the forward above: max positions were indices 5,7,13,15
        // grad_output = {1,2,3,4}
        // grad_input should be all zeros except at those indices
        int indices_host[] = {5, 7, 13, 15};
        int* indices = backend->allocate_int(4);
        backend->upload((float*)indices, (float*)indices_host, 4); // upload raw bytes

        // For Eigen backend, int memory is just CPU memory, so we can write directly
        // Let's test via the full MaxPool2D layer instead for a cleaner test
        backend->deallocate_int(indices);
    }

    // global_avg_pool_forward: 1 batch, 2 channels, 2x2
    // channel 0: [1,2,3,4] → avg = 2.5
    // channel 1: [5,6,7,8] → avg = 6.5
    {
        Tensor input({1,2,2,2}, backend);
        input.set_data({1,2,3,4, 5,6,7,8});
        Tensor output({1,2}, backend);
        backend->global_avg_pool_forward(input.data(), output.data(), 1, 2, 2, 2);
        expect("global_avg_pool_forward", to_vec(output), {2.5f, 6.5f});
    }

    // global_avg_pool_backward: grad_output = {1, 2}, h*w = 4
    // each spatial position gets grad / (h*w)
    {
        Tensor grad_out({1,2}, backend);
        grad_out.set_data({1.0f, 2.0f});
        Tensor grad_in({1,2,2,2}, backend);
        backend->global_avg_pool_backward(grad_out.data(), grad_in.data(), 1, 2, 2, 2);
        expect("global_avg_pool_backward", to_vec(grad_in),
               {0.25f, 0.25f, 0.25f, 0.25f,  0.5f, 0.5f, 0.5f, 0.5f});
    }

    // softmax_forward: 2 rows, 3 cols
    // row 0: [1,2,3] → softmax ≈ [0.0900, 0.2447, 0.6652]
    // row 1: [0,0,0] → softmax = [0.3333, 0.3333, 0.3333]
    {
        Tensor input({2,3}, backend);
        input.set_data({1,2,3, 0,0,0});
        Tensor output({2,3}, backend);
        backend->softmax_forward(input.data(), output.data(), 2, 3);
        auto v = to_vec(output);
        bool ok = std::fabs(v[0] - 0.0900f) < 1e-3f &&
                  std::fabs(v[1] - 0.2447f) < 1e-3f &&
                  std::fabs(v[2] - 0.6652f) < 1e-3f &&
                  std::fabs(v[3] - 1.0f/3) < 1e-3f &&
                  std::fabs(v[4] - 1.0f/3) < 1e-3f &&
                  std::fabs(v[5] - 1.0f/3) < 1e-3f;
        std::cout << "  " << (ok ? "PASS" : "FAIL") << " [softmax_forward]\n";
    }
}

void test_layers(Tensor::BackendPtr backend, const std::string& name) {
    std::cout << "\n=== " << name << " (layer integration) ===\n";

    // Dense: forward + backward + update (smoke test)
    {
        auto dense = std::make_shared<Dense>(4, 2, backend);
        Tensor input({2, 4}, backend);
        input.set_data({1,2,3,4, 5,6,7,8});

        Tensor output = dense->forward(input);
        bool fwd_ok = output.rows() == 2 && output.cols() == 2;

        Tensor grad({2, 2}, backend);
        grad.set_data({1,0, 0,1});
        Tensor grad_in = dense->backward(grad);
        bool bwd_ok = grad_in.rows() == 2 && grad_in.cols() == 4;

        dense->update_parameters(0.001f);
        std::cout << "  " << (fwd_ok && bwd_ok ? "PASS" : "FAIL") << " [Dense fwd/bwd/update]\n";
    }

    // Conv2D: forward + backward + update (smoke test)
    {
        auto conv = std::make_shared<Conv2D>(1, 2, 3, 3, 1, 1, 1, 1, backend);
        Tensor input({1, 1, 4, 4}, backend);
        std::vector<float> data(16);
        for (int i = 0; i < 16; ++i) data[i] = (float)(i + 1);
        input.set_data(data);

        Tensor output = conv->forward(input);
        bool fwd_ok = output.shape()[0] == 1 && output.shape()[1] == 2 &&
                      output.shape()[2] == 4 && output.shape()[3] == 4;

        Tensor grad({1, 2, 4, 4}, backend);
        grad.fill(1.0f);
        Tensor grad_in = conv->backward(grad);
        bool bwd_ok = grad_in.shape()[0] == 1 && grad_in.shape()[1] == 1 &&
                      grad_in.shape()[2] == 4 && grad_in.shape()[3] == 4;

        conv->update_parameters(0.001f);
        std::cout << "  " << (fwd_ok && bwd_ok ? "PASS" : "FAIL") << " [Conv2D fwd/bwd/update]\n";
    }

    // MaxPool2D: forward + backward
    {
        auto pool = std::make_shared<MaxPool2D>(2, backend);
        Tensor input({1, 1, 4, 4}, backend);
        input.set_data({1, 2, 3, 4,
                        5, 6, 7, 8,
                        9,10,11,12,
                       13,14,15,16});
        Tensor output = pool->forward(input);
        expect("MaxPool2D forward", to_vec(output), {6, 8, 14, 16});

        Tensor grad({1, 1, 2, 2}, backend);
        grad.set_data({1, 2, 3, 4});
        Tensor grad_in = pool->backward(grad);
        expect("MaxPool2D backward", to_vec(grad_in),
               {0,0,0,0, 0,1,0,2, 0,0,0,0, 0,3,0,4});
    }

    // GlobalAvgPool: forward + backward
    {
        auto gap = std::make_shared<GlobalAvgPool>(backend);
        Tensor input({1, 2, 2, 2}, backend);
        input.set_data({1,2,3,4, 5,6,7,8});
        Tensor output = gap->forward(input);
        expect("GlobalAvgPool forward", to_vec(output), {2.5f, 6.5f});

        Tensor grad({1, 2}, backend);
        grad.set_data({4.0f, 8.0f});
        Tensor grad_in = gap->backward(grad);
        expect("GlobalAvgPool backward", to_vec(grad_in),
               {1,1,1,1, 2,2,2,2});
    }

    // Softmax layer: forward
    {
        auto sm = std::make_shared<Softmax>(backend);
        Tensor input({1, 3}, backend);
        input.set_data({1, 2, 3});
        Tensor output = sm->forward(input);
        auto v = to_vec(output);
        bool ok = std::fabs(v[0] - 0.0900f) < 1e-3f &&
                  std::fabs(v[1] - 0.2447f) < 1e-3f &&
                  std::fabs(v[2] - 0.6652f) < 1e-3f;
        std::cout << "  " << (ok ? "PASS" : "FAIL") << " [Softmax layer forward]\n";
    }

    // Mini forward pass: Conv → ReLU → Pool → Flatten → Dense
    {
        Sequential net;
        net.add(std::make_shared<Conv2D>(1, 2, 3, 3, 1, 1, 1, 1, backend));
        net.add(std::make_shared<ReLU>(backend));
        net.add(std::make_shared<MaxPool2D>(2, backend));
        net.add(std::make_shared<Flatten>(backend));
        net.add(std::make_shared<Dense>(2 * 2 * 2, 3, backend));

        Tensor input({1, 1, 4, 4}, backend);
        std::vector<float> data(16);
        for (int i = 0; i < 16; ++i) data[i] = (float)(i + 1) / 16.0f;
        input.set_data(data);

        Tensor output = net.forward(input);
        bool ok = output.rows() == 1 && output.cols() == 3;

        // backward
        Tensor grad({1, 3}, backend);
        grad.set_data({1, 0, 0});
        net.backward(grad);
        net.update_parameters(0.001f);
        std::cout << "  " << (ok ? "PASS" : "FAIL") << " [Mini CNN fwd/bwd/update]\n";
    }
}

int main() {
    try {
        auto eigen = std::make_shared<EigenBackend>();
        test_backend(eigen, "Eigen (CPU)");
        test_new_primitives(eigen, "Eigen (CPU)");
        test_layers(eigen, "Eigen (CPU)");

        // auto cuda = std::make_shared<CudaBackend>();
        // test_backend(cuda, "CUDA (GPU)");
        // test_new_primitives(cuda, "CUDA (GPU)");
        // test_layers(cuda, "CUDA (GPU)");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
