#pragma once

/**
 * @file NnueNetwork.h
 * @brief Native half-KP NNUE evaluator and published snapshot keys.
 */

#include "nn/INetwork.h"
#include "nn/nnue/NnueWeights.h"

#include <memory>
#include <string>

namespace cnnv::nn::nnue {

/**
 * @brief Stable activation keys published by the NNUE evaluator.
 *
 * Keeping keys centralized prevents drift between the network, tests, and
 * visualization panels.
 */
namespace snapshot_keys {
constexpr const char* kAccumulatorWhite = "nnue.accumulator.white";
constexpr const char* kAccumulatorBlack = "nnue.accumulator.black";
constexpr const char* kAccumulatorClippedUs   = "nnue.accumulator.clipped.us";
constexpr const char* kAccumulatorClippedThem = "nnue.accumulator.clipped.them";
constexpr const char* kValueCentipawns = "nnue.value.centipawns";
constexpr const char* kFeatureActiveWhite = "nnue.feature_active.white";
constexpr const char* kFeatureActiveBlack = "nnue.feature_active.black";
constexpr const char* kFeatureBias = "nnue.feature_bias";
constexpr const char* kFeatureWeightsActiveWhite = "nnue.feature_weights.active.white";
constexpr const char* kFeatureWeightsActiveBlack = "nnue.feature_weights.active.black";
constexpr const char* kOutputWeightsUs = "nnue.output_weights.us";
constexpr const char* kOutputWeightsThem = "nnue.output_weights.them";
constexpr const char* kOutputContributionUs = "nnue.output_contribution.us";
constexpr const char* kOutputContributionThem = "nnue.output_contribution.them";
constexpr const char* kOutputAffine = "nnue.output_affine";
constexpr const char* kPieceContributionWhite = "nnue.piece_contribution.white";
constexpr const char* kPieceContributionBlack = "nnue.piece_contribution.black";
}  // namespace snapshot_keys

/**
 * @brief Native float32 NNUE evaluator for the CRTK half-KP model format.
 */
class Network : public INetwork {
   public:
    /**
     * @brief Creates an evaluator with fallback zero weights.
     */
    Network();

    /**
     * @brief Creates an evaluator from already parsed weights.
     */
    explicit Network(Weights weights);

    /**
     * @brief Loads CRTK NNUE weights from disk.
     */
    void load(const std::string& path) override;

    /**
     * @brief Evaluates a position and publishes NNUE intermediate tensors.
     */
    void evaluate(const cnnv::chess::Position& pos,
                  ActivationSnapshot& out) const override;

    /**
     * @brief Display name for the architecture selector.
     */
    std::string name() const override { return "NNUE (half-KP, CRTK fp32)"; }

    /**
     * @brief Hidden accumulator width.
     */
    int hiddenSize() const noexcept { return m_weights.hiddenSize; }

    /**
     * @brief True when non-empty weights are available.
     */
    bool isLoaded() const noexcept { return !m_weights.empty(); }

   private:
    Weights m_weights;
};

}  // namespace cnnv::nn::nnue
