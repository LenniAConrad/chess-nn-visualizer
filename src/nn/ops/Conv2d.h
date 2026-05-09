#pragma once

/**
 * @file Conv2d.h
 * @brief CPU 2D convolution primitive and shape helper.
 */

#include <cstddef>

namespace cnnv::nn::ops {

/**
 * @brief Computes 2D convolution on a single NCHW tensor.
 * @param input Input tensor `[channelsIn, Hin, Win]`.
 * @param weight Kernel tensor `[channelsOut, channelsIn, Kh, Kw]`.
 * @param bias Optional per-output-channel bias, or null.
 * @param output Output tensor `[channelsOut, Hout, Wout]`.
 * @param channelsIn Number of input channels.
 * @param Hin Input height.
 * @param Win Input width.
 * @param channelsOut Number of output channels.
 * @param Kh Kernel height.
 * @param Kw Kernel width.
 * @param strideH Vertical stride.
 * @param strideW Horizontal stride.
 * @param padH Vertical zero-padding.
 * @param padW Horizontal zero-padding.
 *
 * There is no batch dimension; callers loop over batches if needed.
 */
void conv2d(const float* input, const float* weight, const float* bias,
            float* output,
            std::size_t channelsIn, std::size_t Hin, std::size_t Win,
            std::size_t channelsOut, std::size_t Kh, std::size_t Kw,
            std::size_t strideH, std::size_t strideW,
            std::size_t padH, std::size_t padW);

/**
 * @brief Computes one output spatial dimension for convolution.
 * @param in Input dimension.
 * @param k Kernel dimension.
 * @param stride Stride along the dimension.
 * @param pad Padding on each side.
 * @return `(in + 2 * pad - k) / stride + 1`.
 */
constexpr std::size_t conv2d_out_dim(std::size_t in, std::size_t k,
                                     std::size_t stride, std::size_t pad) {
    return (in + 2 * pad - k) / stride + 1;
}

}  // namespace cnnv::nn::ops
