#include "LayerNorm.h"

#include <cmath>

namespace cnnv::nn::ops {

void layer_norm(const float* x, const float* gamma, const float* beta,
                float* out, std::size_t tokens, std::size_t dim, float eps) {
    for (std::size_t t = 0; t < tokens; ++t) {
        const float* row = x + t * dim;
        float* out_row = out + t * dim;

        float mean = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) mean += row[i];
        mean /= static_cast<float>(dim);

        float var = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            const float d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(dim);

        const float inv_std = 1.0f / std::sqrt(var + eps);
        for (std::size_t i = 0; i < dim; ++i) {
            float v = (row[i] - mean) * inv_std;
            if (gamma) v *= gamma[i];
            if (beta) v += beta[i];
            out_row[i] = v;
        }
    }
}

}  // namespace cnnv::nn::ops
