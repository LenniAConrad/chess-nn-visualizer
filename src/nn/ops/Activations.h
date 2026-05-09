#pragma once

/**
 * @file Activations.h
 * @brief Scalar activation functions used by native NN forward passes.
 */

#include <cstddef>
#include <cstdint>

namespace cnnv::nn::ops {

/**
 * @brief Applies ReLU in-place.
 */
void relu(float* x, std::size_t n);

/**
 * @brief Applies ReLU from one buffer to another.
 */
void relu(const float* in, float* out, std::size_t n);

/**
 * @brief Applies Stockfish-style clipped ReLU to int16 input.
 * @param in Signed 16-bit source values.
 * @param out Signed 8-bit destination values.
 * @param n Number of elements.
 * @param shift Right shift applied before clamping.
 *
 * Outputs are clamped to the range [0, 127].
 */
void clipped_relu_i16_to_i8(const std::int16_t* in, std::int8_t* out,
                            std::size_t n, int shift);

/**
 * @brief Applies clipped ReLU in float space.
 * @param max_val Upper clamp value; lower clamp is zero.
 */
void clipped_relu(const float* in, float* out, std::size_t n,
                  float max_val = 1.0f);

/**
 * @brief Applies sigmoid elementwise.
 */
void sigmoid(const float* in, float* out, std::size_t n);

/**
 * @brief Applies hyperbolic tangent elementwise.
 */
void tanh_act(const float* in, float* out, std::size_t n);

/**
 * @brief Applies tanh-approximate GELU elementwise.
 */
void gelu(const float* in, float* out, std::size_t n);

/**
 * @brief Applies numerically stable softmax in-place.
 */
void softmax(float* x, std::size_t n);

/**
 * @brief Applies numerically stable softmax from one buffer to another.
 */
void softmax(const float* in, float* out, std::size_t n);

}  // namespace cnnv::nn::ops
