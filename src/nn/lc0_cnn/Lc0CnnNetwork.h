#pragma once

/**
 * @file Lc0CnnNetwork.h
 * @brief Native LC0 CNN evaluator and published snapshot keys.
 */

#include "nn/INetwork.h"
#include "nn/lc0_cnn/Lc0CnnWeights.h"

#include <string>

namespace cnnv::nn::lc0_cnn {

/**
 * @brief Stable activation keys published by the LC0 CNN evaluator.
 */
namespace snapshot_keys {
constexpr const char* kInputPlanes      = "cnn.input.planes";       // [112, 8, 8]
constexpr const char* kStemRelu         = "cnn.stem.relu";          // [Ct, 8, 8]
constexpr const char* kPolicyHidden     = "cnn.policy.stem.relu";   // [Ct, 8, 8]
constexpr const char* kPolicyPlanes     = "cnn.policy.planes";      // [Cp, 8, 8]
constexpr const char* kPolicyLogits     = "cnn.policy.logits";      // [P]
constexpr const char* kValueConv        = "cnn.value.conv.relu";    // [Cv, 8, 8]
constexpr const char* kValueFc1         = "cnn.value.fc1.relu";     // [F]
constexpr const char* kValueLogits      = "cnn.value.logits";       // [3]
constexpr const char* kValueWdl         = "cnn.value.wdl";          // [3]
constexpr const char* kValueScalar      = "cnn.value.scalar";       // [1]
constexpr const char* kFinalActivation  = "cnn.final.relu_mean";    // [64]
inline std::string blockReluKey(int idx) {
    return "cnn.block" + std::to_string(idx) + ".relu";              // [Ct, 8, 8]
}
}  // namespace snapshot_keys

/**
 * @brief CRTK LC0 CNN ResNet evaluator.
 *
 * The implementation operates entirely in float32 because the visualizer does
 * not need quantized engine throughput. It publishes input planes, trunk
 * activations, policy logits, WDL values, and summary heatmaps.
 */
class Network : public INetwork {
   public:
    Network() = default;

    /**
     * @brief Creates an evaluator from already parsed weights.
     */
    explicit Network(Weights weights);

    /**
     * @brief Loads LC0J weights from disk.
     */
    void load(const std::string& path) override;

    /**
     * @brief Evaluates a position and fills CNN activation tensors.
     */
    void evaluate(const cnnv::chess::Position& pos,
                  ActivationSnapshot& out) const override;

    /**
     * @brief Display name for the architecture selector.
     */
    std::string name() const override;

    /**
     * @brief True when non-empty CNN weights are available.
     */
    bool isLoaded() const noexcept { return !m_weights.empty(); }

    /**
     * @brief Read-only access to parsed weights for tests and diagnostics.
     */
    const Weights& weights() const noexcept { return m_weights; }

   private:
    Weights m_weights;
};

}  // namespace cnnv::nn::lc0_cnn
