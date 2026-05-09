#pragma once

/**
 * @file NnueAccumulator.h
 * @brief Half-KP feature-transformer accumulators for NNUE evaluation.
 */

#include <cstddef>
#include <vector>

namespace cnnv::chess {
class Position;
}

namespace cnnv::nn::nnue {

class Weights;

/**
 * @brief Maintains white- and black-perspective NNUE accumulator vectors.
 *
 * Each accumulator entry equals the feature-transformer bias plus the sum of
 * active half-KP feature weights for that perspective. The visualizer normally
 * rebuilds accumulators from scratch with `refresh()`; incremental add/remove
 * methods are kept for future performance work.
 */
class Accumulator {
   public:
    /**
     * @brief Binds the accumulator to immutable NNUE weights.
     */
    explicit Accumulator(const Weights& weights);

    /**
     * @brief Rebuilds both perspective accumulators from a position.
     */
    void refresh(const cnnv::chess::Position& pos);

    /**
     * @brief Resets both accumulators to feature-transformer bias.
     */
    void reset();

    /**
     * @brief Adds or subtracts one active feature from one perspective.
     * @param whitePerspective True for white-perspective accumulator, false
     * for black-perspective accumulator.
     * @param feature Feature index from `FeatureEncoder`.
     * @param sign Usually `1.0f` for add or `-1.0f` for remove.
     */
    void addFeature(bool whitePerspective, int feature, float sign = 1.0f);

    /**
     * @brief Removes one feature from one perspective accumulator.
     */
    void removeFeature(bool whitePerspective, int feature) {
        addFeature(whitePerspective, feature, -1.0f);
    }

    /**
     * @brief Returns a pointer to one perspective accumulator.
     */
    const float* values(bool whitePerspective) const noexcept {
        return whitePerspective ? m_white.data() : m_black.data();
    }

    /**
     * @brief Number of hidden units in each accumulator.
     */
    std::size_t hiddenSize() const noexcept { return m_white.size(); }

   private:
    const Weights& m_weights;
    std::vector<float> m_white;
    std::vector<float> m_black;
};

}  // namespace cnnv::nn::nnue
