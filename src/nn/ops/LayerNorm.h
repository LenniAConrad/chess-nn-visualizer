#pragma once

/**
 * @file LayerNorm.h
 * @brief Layer normalization primitive for token tensors.
 */

#include <cstddef>

namespace cnnv::nn::ops {

/**
 * @brief Applies layer normalization over the last dimension.
 * @param x Input tensor `[tokens, dim]`.
 * @param gamma Optional per-feature scale, or null to use 1.
 * @param beta Optional per-feature bias, or null to use 0.
 * @param out Output tensor `[tokens, dim]`.
 * @param tokens Number of rows.
 * @param dim Width of each row.
 * @param eps Numerical stabilizer added to variance.
 *
 * Formula per row: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
 */
void layer_norm(const float* x, const float* gamma, const float* beta,
                float* out, std::size_t tokens, std::size_t dim,
                float eps = 1e-5f);

}  // namespace cnnv::nn::ops
