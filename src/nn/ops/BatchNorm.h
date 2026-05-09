#pragma once

/**
 * @file BatchNorm.h
 * @brief Inference-time 2D batch-normalization primitive.
 */

#include <cstddef>

namespace cnnv::nn::ops {

/**
 * @brief Applies inference-time batch normalization to a single NCHW image.
 * @param x Input tensor `[channels, H, W]`.
 * @param out Output tensor `[channels, H, W]`.
 * @param mean Per-channel running means.
 * @param var Per-channel running variances.
 * @param scale Per-channel gamma values.
 * @param bias Per-channel beta values.
 * @param channels Number of channels.
 * @param H Spatial height.
 * @param W Spatial width.
 * @param eps Numerical stabilizer added to variance.
 *
 * The formula is `y = scale[c] * (x - mean[c]) / sqrt(var[c] + eps) + bias[c]`.
 */
void batch_norm_2d(const float* x, float* out,
                   const float* mean, const float* var,
                   const float* scale, const float* bias,
                   std::size_t channels, std::size_t H, std::size_t W,
                   float eps = 1e-5f);

}  // namespace cnnv::nn::ops
