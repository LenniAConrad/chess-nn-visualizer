#pragma once

/**
 * @file NnueLoader.h
 * @brief Loader for the project's compact CRTK float NNUE format.
 */

#include "nn/nnue/NnueWeights.h"

#include <cstdint>
#include <string>

namespace cnnv::nn::nnue {

/**
 * @brief Reads and validates CRTK simple float NNUE weight files.
 *
 * File format:
 * - bytes 0..3: magic `NNUE`
 * - uint32: version, currently 1
 * - uint32: feature count, expected `64 * 10 * 64`
 * - uint32: hidden size
 * - float32: output scale
 * - length-prefixed float32 feature bias
 * - length-prefixed float32 feature weights
 * - length-prefixed float32 output weights
 * - float32 output bias
 *
 * All multi-byte values are little-endian. Invalid files throw
 * `std::runtime_error`.
 */
class Loader {
   public:
    /**
     * @brief Loads NNUE weights from disk.
     */
    static Weights load(const std::string& path);

    /**
     * @brief Builds a zero-initialized fallback network.
     * @param hiddenSize Hidden width to allocate.
     * @return Valid weights object that evaluates to neutral output.
     */
    static Weights makeFallback(int hiddenSize = 1);

    /** @brief Supported file-format version. */
    static constexpr std::uint32_t kVersion = 1;
};

}  // namespace cnnv::nn::nnue
