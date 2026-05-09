#pragma once

/**
 * @file Attention.h
 * @brief Multi-head scaled dot-product attention primitive.
 */

#include <cstddef>

namespace cnnv::nn::ops {

/**
 * @brief Computes scaled dot-product multi-head attention.
 * @param Q Query tensor shaped `[tokens, heads, head_dim]`.
 * @param K Key tensor shaped `[tokens, heads, head_dim]`.
 * @param V Value tensor shaped `[tokens, heads, head_dim]`.
 * @param out Output tensor shaped `[tokens, heads, head_dim]`.
 * @param weights Optional attention weights shaped `[heads, tokens, tokens]`.
 * Pass null when the caller does not need the heatmap.
 * @param tokens Number of sequence tokens.
 * @param heads Number of attention heads.
 * @param head_dim Width of each head.
 *
 * Layout is token-major, then head, then head dimension. For each output:
 * `out[t,h,d] = sum_t2 weights[h,t,t2] * V[t2,h,d]`.
 */
void scaled_dot_product_attention(const float* Q, const float* K,
                                  const float* V, float* out, float* weights,
                                  std::size_t tokens, std::size_t heads,
                                  std::size_t head_dim);

}  // namespace cnnv::nn::ops
