#pragma once

/**
 * @file NnueWeights.h
 * @brief Parsed NNUE weight container.
 */

#include <cstddef>
#include <vector>

namespace cnnv::nn::nnue {

/**
 * @brief Parsed CRTK NNUE weights.
 *
 * Mirrors the chess-rtk `Network.Weights` record: a feature transformer
 * `(FEATURE_COUNT x hiddenSize)` in feature-major order plus one output layer
 * mapping concatenated `[us, them]` accumulators to a scalar score.
 *
 * Forward pass:
 * - `accumulator[p][i] = featureBias[i] + sum_active featureWeights[f * H + i]`
 * - `raw = outputBias + dot(outputWeights[0:H], relu(us))`
 * - `raw += dot(outputWeights[H:2H], relu(them))`
 * - `centipawns = raw * outputScale`
 */
struct Weights {
    /** @brief Number of hidden units per perspective accumulator. */
    int hiddenSize = 0;

    /** @brief Feature-transformer bias, length `hiddenSize`. */
    std::vector<float> featureBias;

    /**
     * @brief Feature-transformer weights in feature-major layout.
     *
     * `featureWeights[f * hiddenSize + i]` is the weight of hidden unit `i`
     * for sparse feature `f`.
     */
    std::vector<float> featureWeights;

    /**
     * @brief Output-layer weights, length `hiddenSize * 2`.
     *
     * First half applies to side-to-move features, second half to opponent
     * features.
     */
    std::vector<float> outputWeights;

    /** @brief Scalar output bias before `outputScale`. */
    float outputBias  = 0.0f;

    /** @brief Scale converting raw output to centipawn-like units. */
    float outputScale = 1.0f;

    /** @brief True when no usable network shape has been loaded. */
    bool empty() const noexcept { return hiddenSize == 0; }

    /** @brief Total number of scalar parameters. */
    std::size_t parameterCount() const noexcept {
        return featureBias.size() + featureWeights.size() +
               outputWeights.size() + 1;
    }
};

}  // namespace cnnv::nn::nnue
