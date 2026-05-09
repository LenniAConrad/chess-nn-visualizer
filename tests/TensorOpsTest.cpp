#include <cmath>
#include <cstdint>
#include <vector>

#include "TestMain.h"
#include "nn/ActivationSnapshot.h"
#include "nn/Tensor.h"
#include "nn/ops/Activations.h"
#include "nn/ops/Attention.h"
#include "nn/ops/BatchNorm.h"
#include "nn/ops/Conv2d.h"
#include "nn/ops/LayerNorm.h"
#include "nn/ops/MatMul.h"

using namespace cnnv::nn;
using namespace cnnv::nn::ops;

namespace {

bool approx(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

}  // namespace

TEST(tensor_basic_alloc_and_index) {
    Tensor<float, 3> t({2, 3, 4});
    CHECK_EQ(t.size(), 24u);
    CHECK_EQ(t.dim(0), 2u);
    CHECK_EQ(t.dim(1), 3u);
    CHECK_EQ(t.dim(2), 4u);

    t.fill(0.0f);
    t.at(1, 2, 3) = 7.5f;
    CHECK(approx(t.at(1, 2, 3), 7.5f));

    // Strides are row-major: index (1,2,3) flattens to 1*12 + 2*4 + 3 = 23.
    CHECK(approx(t.data()[23], 7.5f));
    CHECK(approx(t.data()[0], 0.0f));
}

TEST(matmul_4x3_times_3x5) {
    // A = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    // B = identity-like for first 3 cols, then non-trivial.
    std::vector<float> A = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    };
    // 3x5 matrix.
    std::vector<float> B = {
        1, 0, 0, 1, 2,
        0, 1, 0, 1, 0,
        0, 0, 1, 1, 1,
    };
    std::vector<float> C(4 * 5, 0.0f);
    matmul(A.data(), B.data(), C.data(), 4, 3, 5);

    // Row 0: [1, 2, 3, 1+2+3, 2+3] = [1, 2, 3, 6, 5]
    CHECK(approx(C[0], 1.0f));
    CHECK(approx(C[1], 2.0f));
    CHECK(approx(C[2], 3.0f));
    CHECK(approx(C[3], 6.0f));
    CHECK(approx(C[4], 5.0f));
    // Row 3: [10, 11, 12, 33, 32]
    CHECK(approx(C[15], 10.0f));
    CHECK(approx(C[16], 11.0f));
    CHECK(approx(C[17], 12.0f));
    CHECK(approx(C[18], 33.0f));
    CHECK(approx(C[19], 32.0f));
}

TEST(matmul_bias_adds_per_column) {
    std::vector<float> A = {1, 1, 1, 1};  // 2x2
    std::vector<float> B = {1, 2, 3, 4};  // 2x2
    std::vector<float> bias = {10, 20};
    std::vector<float> C(4, 0.0f);
    matmul_bias(A.data(), B.data(), bias.data(), C.data(), 2, 2, 2);
    // Row sums of B columns: col0 = 1+3=4, col1 = 2+4=6.
    CHECK(approx(C[0], 14.0f));
    CHECK(approx(C[1], 26.0f));
    CHECK(approx(C[2], 14.0f));
    CHECK(approx(C[3], 26.0f));
}

TEST(conv2d_identity_1x1) {
    // 1x1 conv with weight = 1 should pass input through unchanged.
    std::vector<float> input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };  // [1, 3, 3]
    std::vector<float> weight = {1.0f};  // [1, 1, 1, 1]
    std::vector<float> bias = {0.0f};
    std::vector<float> out(9, 0.0f);
    conv2d(input.data(), weight.data(), bias.data(), out.data(),
           /*Ci=*/1, /*Hin=*/3, /*Win=*/3,
           /*Co=*/1, /*Kh=*/1, /*Kw=*/1,
           /*sH=*/1, /*sW=*/1, /*pH=*/0, /*pW=*/0);
    for (std::size_t i = 0; i < 9; ++i) {
        CHECK(approx(out[i], input[i]));
    }
}

TEST(conv2d_3x3_sum_kernel_padded) {
    // A 3x3 sum kernel applied to a 3x3 input with padding 1 produces the
    // sum of each 3x3 neighborhood (zeros outside).
    std::vector<float> input = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    };
    std::vector<float> weight(9, 1.0f);
    std::vector<float> bias = {0.0f};
    std::vector<float> out(9, 0.0f);
    conv2d(input.data(), weight.data(), bias.data(), out.data(),
           1, 3, 3, 1, 3, 3, 1, 1, 1, 1);
    // Corner: only 4 neighbours in-bounds.
    CHECK(approx(out[0], 4.0f));
    CHECK(approx(out[2], 4.0f));
    CHECK(approx(out[6], 4.0f));
    CHECK(approx(out[8], 4.0f));
    // Edge midpoint: 6 neighbours in-bounds.
    CHECK(approx(out[1], 6.0f));
    // Center: all 9.
    CHECK(approx(out[4], 9.0f));
}

TEST(activations_relu_and_clipped_relu) {
    std::vector<float> in = {-1.0f, 0.0f, 0.5f, 2.0f};
    std::vector<float> out(4);
    relu(in.data(), out.data(), 4);
    CHECK(approx(out[0], 0.0f));
    CHECK(approx(out[1], 0.0f));
    CHECK(approx(out[2], 0.5f));
    CHECK(approx(out[3], 2.0f));

    clipped_relu(in.data(), out.data(), 4, 1.0f);
    CHECK(approx(out[0], 0.0f));
    CHECK(approx(out[2], 0.5f));
    CHECK(approx(out[3], 1.0f));
}

TEST(activations_clipped_relu_i16) {
    std::vector<std::int16_t> in = {-100, 0, 50, 130, 4096};
    std::vector<std::int8_t> out(in.size());
    clipped_relu_i16_to_i8(in.data(), out.data(), in.size(), 0);
    CHECK_EQ(static_cast<int>(out[0]), 0);
    CHECK_EQ(static_cast<int>(out[1]), 0);
    CHECK_EQ(static_cast<int>(out[2]), 50);
    CHECK_EQ(static_cast<int>(out[3]), 127);
    CHECK_EQ(static_cast<int>(out[4]), 127);
}

TEST(activations_softmax_uniform_and_skewed) {
    std::vector<float> v = {1.0f, 1.0f, 1.0f, 1.0f};
    softmax(v.data(), v.size());
    for (float x : v) CHECK(approx(x, 0.25f));

    std::vector<float> peak = {0.0f, 0.0f, 100.0f, 0.0f};
    softmax(peak.data(), peak.size());
    CHECK(approx(peak[2], 1.0f, 1e-4f));
    float sum = 0.0f;
    for (float x : peak) sum += x;
    CHECK(approx(sum, 1.0f, 1e-5f));
}

TEST(layer_norm_zero_mean_unit_var_per_row) {
    std::vector<float> x = {
        1, 2, 3, 4,
        10, 10, 10, 10,
    };
    std::vector<float> out(8);
    layer_norm(x.data(), nullptr, nullptr, out.data(), 2, 4);

    float mean = 0.0f;
    for (std::size_t i = 0; i < 4; ++i) mean += out[i];
    CHECK(approx(mean / 4.0f, 0.0f, 1e-5f));
    float var = 0.0f;
    for (std::size_t i = 0; i < 4; ++i) var += out[i] * out[i];
    CHECK(approx(var / 4.0f, 1.0f, 1e-3f));

    // Constant row: numerator is zero, output should be ~0.
    for (std::size_t i = 4; i < 8; ++i) CHECK(approx(out[i], 0.0f, 1e-4f));
}

TEST(batch_norm_2d_passthrough_when_unit_params) {
    std::vector<float> x = {1, 2, 3, 4};  // 1 channel, 2x2.
    std::vector<float> out(4);
    float mean = 0.0f;
    float var = 1.0f;
    float scale = 1.0f;
    float bias = 0.0f;
    batch_norm_2d(x.data(), out.data(), &mean, &var, &scale, &bias, 1, 2, 2,
                  0.0f);
    for (std::size_t i = 0; i < 4; ++i) CHECK(approx(out[i], x[i]));
}

TEST(attention_uniform_when_q_zero) {
    // Q == 0 → all logits zero → uniform softmax → output = mean of V along
    // tokens.
    std::size_t T = 3, H = 1, D = 2;
    std::vector<float> Q(T * H * D, 0.0f);
    std::vector<float> K(T * H * D, 1.0f);
    std::vector<float> V = {1, 2, 3, 4, 5, 6};  // (T=3, H=1, D=2)
    std::vector<float> out(T * H * D, 0.0f);
    std::vector<float> w(H * T * T, 0.0f);
    scaled_dot_product_attention(Q.data(), K.data(), V.data(), out.data(),
                                 w.data(), T, H, D);
    // Each token's output should equal the column-mean of V.
    float mean_d0 = (1 + 3 + 5) / 3.0f;
    float mean_d1 = (2 + 4 + 6) / 3.0f;
    for (std::size_t t = 0; t < T; ++t) {
        CHECK(approx(out[t * D + 0], mean_d0, 1e-5f));
        CHECK(approx(out[t * D + 1], mean_d1, 1e-5f));
    }
    // Weight rows should each sum to ~1.
    for (std::size_t t = 0; t < T; ++t) {
        float s = 0.0f;
        for (std::size_t j = 0; j < T; ++j) s += w[t * T + j];
        CHECK(approx(s, 1.0f, 1e-5f));
    }
}

TEST(activation_snapshot_store_and_read) {
    ActivationSnapshot snap;
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    snap.store("nnue.fc1.relu", {2, 3}, data.data());
    CHECK(snap.has("nnue.fc1.relu"));
    CHECK_EQ(snap.size("nnue.fc1.relu"), 6u);
    CHECK_EQ(snap.shape("nnue.fc1.relu").size(), 2u);
    CHECK_EQ(snap.shape("nnue.fc1.relu")[0], 2u);
    CHECK_EQ(snap.shape("nnue.fc1.relu")[1], 3u);
    CHECK(approx(snap.data("nnue.fc1.relu")[0], 1.0f));
    CHECK(approx(snap.data("nnue.fc1.relu")[5], 6.0f));

    float* dst = snap.allocate("cnn.block0.relu", {1, 4});
    dst[0] = 9.0f;
    CHECK(approx(snap.data("cnn.block0.relu")[0], 9.0f));
    CHECK(approx(snap.data("cnn.block0.relu")[3], 0.0f));

    snap.clear();
    CHECK(!snap.has("nnue.fc1.relu"));
}
