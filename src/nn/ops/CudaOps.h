#pragma once

/**
 * @file CudaOps.h
 * @brief Optional CUDA acceleration hooks for tensor operations.
 */

#include <cstddef>

namespace cnnv::nn::ops::cuda_backend {

/**
 * @brief Reports whether CUDA kernels are compiled and usable at runtime.
 */
bool available();

/**
 * @brief Human-readable CUDA availability or failure status.
 */
const char* status();

/**
 * @brief Attempts GPU matrix multiplication.
 * @return True if CUDA executed the operation; false when the CPU fallback
 * should run.
 */
bool try_matmul(const float* A, const float* B, float* C,
                std::size_t M, std::size_t K, std::size_t N);

/**
 * @brief Attempts GPU matrix multiplication with row bias.
 * @return True if CUDA executed the operation; false for CPU fallback.
 */
bool try_matmul_bias(const float* A, const float* B, const float* bias,
                     float* C, std::size_t M, std::size_t K, std::size_t N);

/**
 * @brief Attempts GPU 2D convolution with the same layout as `ops::conv2d`.
 * @return True if CUDA executed the operation; false for CPU fallback.
 */
bool try_conv2d(const float* input, const float* weight, const float* bias,
                float* output,
                std::size_t channelsIn, std::size_t Hin, std::size_t Win,
                std::size_t channelsOut, std::size_t Kh, std::size_t Kw,
                std::size_t strideH, std::size_t strideW,
                std::size_t padH, std::size_t padW);

/**
 * @brief Attempts GPU scaled dot-product attention.
 * @return True if CUDA executed the operation; false for CPU fallback.
 */
bool try_scaled_dot_product_attention(const float* Q, const float* K,
                                      const float* V, float* out,
                                      float* weights,
                                      std::size_t tokens, std::size_t heads,
                                      std::size_t headDim);

}  // namespace cnnv::nn::ops::cuda_backend
