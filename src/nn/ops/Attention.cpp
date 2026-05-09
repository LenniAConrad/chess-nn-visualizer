#include "Attention.h"

#include <cmath>
#include <vector>

#include "Activations.h"
#include "CudaOps.h"

namespace cnnv::nn::ops {

namespace {

inline std::size_t qkv_index(std::size_t t, std::size_t h, std::size_t d,
                             std::size_t heads, std::size_t head_dim) {
    return t * heads * head_dim + h * head_dim + d;
}

}  // namespace

void scaled_dot_product_attention(const float* Q, const float* K,
                                  const float* V, float* out, float* weights,
                                  std::size_t tokens, std::size_t heads,
                                  std::size_t head_dim) {
    if (cuda_backend::try_scaled_dot_product_attention(
            Q, K, V, out, weights, tokens, heads, head_dim)) {
        return;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> local_weights;
    float* w = weights;
    if (!w) {
        local_weights.assign(heads * tokens * tokens, 0.0f);
        w = local_weights.data();
    }

    for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t i = 0; i < tokens; ++i) {
            float* row = w + h * tokens * tokens + i * tokens;
            for (std::size_t j = 0; j < tokens; ++j) {
                float dot = 0.0f;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += Q[qkv_index(i, h, d, heads, head_dim)] *
                           K[qkv_index(j, h, d, heads, head_dim)];
                }
                row[j] = dot * scale;
            }
            softmax(row, tokens);
        }

        for (std::size_t i = 0; i < tokens; ++i) {
            const float* row = w + h * tokens * tokens + i * tokens;
            for (std::size_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (std::size_t j = 0; j < tokens; ++j) {
                    acc += row[j] * V[qkv_index(j, h, d, heads, head_dim)];
                }
                out[qkv_index(i, h, d, heads, head_dim)] = acc;
            }
        }
    }
}

}  // namespace cnnv::nn::ops
