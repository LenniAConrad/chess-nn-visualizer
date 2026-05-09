#include "BatchNorm.h"

#include <cmath>

namespace cnnv::nn::ops {

void batch_norm_2d(const float* x, float* out, const float* mean,
                   const float* var, const float* scale, const float* bias,
                   std::size_t channels, std::size_t H, std::size_t W,
                   float eps) {
    const std::size_t plane = H * W;
    for (std::size_t c = 0; c < channels; ++c) {
        const float inv_std = 1.0f / std::sqrt(var[c] + eps);
        const float s = scale ? scale[c] : 1.0f;
        const float b = bias ? bias[c] : 0.0f;
        const float m = mean[c];
        const float* in_c = x + c * plane;
        float* out_c = out + c * plane;
        for (std::size_t i = 0; i < plane; ++i) {
            out_c[i] = s * (in_c[i] - m) * inv_std + b;
        }
    }
}

}  // namespace cnnv::nn::ops
