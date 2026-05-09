#pragma once

/**
 * @file Lc0CnnLoader.h
 * @brief Loader for CRTK LC0J flat-binary CNN weights.
 */

#include "nn/lc0_cnn/Lc0CnnWeights.h"

#include <cstdint>
#include <string>

namespace cnnv::nn::lc0_cnn {

/**
 * @brief Reads chess-rtk LC0J flat-binary CNN weights.
 *
 * The header starts with magic `LC0J`, version 1, core dimensions, residual
 * block count, value head dimensions, and policy map length. It is followed by
 * convolution layers, optional squeeze-and-excitation units, dense value-head
 * layers, and a compressed policy map. All fields are little-endian and length
 * prefixed as needed.
 */
class Loader {
   public:
    /**
     * @brief Loads and validates LC0 CNN weights from disk.
     */
    static Weights load(const std::string& path);

    /** @brief Supported file-format version. */
    static constexpr std::uint32_t kVersion = 1;
};

}  // namespace cnnv::nn::lc0_cnn
