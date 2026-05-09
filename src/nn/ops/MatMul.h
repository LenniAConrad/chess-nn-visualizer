#pragma once

/**
 * @file MatMul.h
 * @brief Matrix multiplication primitives with optional bias.
 */

#include <cstddef>

namespace cnnv::nn::ops {

/**
 * @brief Computes `C = A * B`.
 * @param A Left matrix shaped `[M, K]`.
 * @param B Right matrix shaped `[K, N]`.
 * @param C Output matrix shaped `[M, N]`.
 * @param M Output rows.
 * @param K Shared dimension.
 * @param N Output columns.
 */
void matmul(const float* A, const float* B, float* C,
            std::size_t M, std::size_t K, std::size_t N);

/**
 * @brief Computes `C = A * B + bias`.
 * @param bias Per-column bias of length `N`.
 */
void matmul_bias(const float* A, const float* B, const float* bias, float* C,
                 std::size_t M, std::size_t K, std::size_t N);

}  // namespace cnnv::nn::ops
