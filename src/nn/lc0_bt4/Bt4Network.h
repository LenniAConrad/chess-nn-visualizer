#pragma once

/**
 * @file Bt4Network.h
 * @brief Native deterministic BT4-style token transformer for visualization.
 */

#include "nn/INetwork.h"
#include "nn/lc0_bt4/Bt4RealWeights.h"

#include <string>

namespace cnnv::nn::lc0_bt4 {

/**
 * @brief Stable activation keys published by the BT4-style evaluator.
 */
namespace snapshot_keys {
constexpr const char* kInputPlanes = "bt4.input.planes";          // [112, 8, 8]
constexpr const char* kTokenFeatures = "bt4.input.tokens";        // [64, 176]
constexpr const char* kEmbedding = "bt4.embedding.mish";          // [64, D]
constexpr const char* kFinalTokens = "bt4.final.tokens";          // [64, D]
constexpr const char* kFinalTokenMagnitude = "bt4.final.token_rms";  // [64]
constexpr const char* kBoardSalience = "bt4.board.value_salience"; // [64], positive board-aware salience
constexpr const char* kPolicyLogits = "bt4.policy.logits";        // [1858]
constexpr const char* kValueLogits = "bt4.value.logits";          // [3]
constexpr const char* kValueWdl = "bt4.value.wdl";                // [3]
constexpr const char* kValueScalar = "bt4.value.scalar";          // [1]

std::string blockLn1Key(int idx);
std::string blockAttentionKey(int idx);     // [heads, 64, 64]
std::string blockAttentionOutKey(int idx);  // [64, D]
std::string blockLn2Key(int idx);
std::string blockFfnKey(int idx);
std::string blockOutputKey(int idx);
}  // namespace snapshot_keys

/**
 * @brief Native BT4-style token transformer evaluator for visualization.
 *
 * The public LC0 BT4 protobuf is not parsed at runtime. Instead, this class
 * validates a small `.bin` metadata file and produces deterministic
 * transformer activations from the same 112-plane board tensor so the UI can
 * inspect tokenization, attention, FFN layers, policy, and value heads without
 * Java or subprocesses.
 */
class Network : public INetwork {
   public:
    /** @brief Number of LC0 input planes consumed by the tokenizer. */
    static constexpr int kInputChannels = 112;

    /** @brief Number of board tokens, one per square. */
    static constexpr int kTokens = 64;

    /** @brief Positional channels appended to each token. */
    static constexpr int kPosChannels = 64;

    /** @brief Width before projection: board planes plus positional channels. */
    static constexpr int kTokenWidth = kInputChannels + kPosChannels;

    /** @brief Transformer model dimension. */
    static constexpr int kModelDim = 64;

    /** @brief Transformer blocks in the deterministic synthetic fallback. */
    static constexpr int kSyntheticBlocks = 15;

    /** @brief Number of attention heads per block. */
    static constexpr int kHeads = 8;

    /** @brief Width per attention head. */
    static constexpr int kHeadDim = kModelDim / kHeads;

    /** @brief LC0 compressed policy size. */
    static constexpr int kPolicySize = 1858;

    Network() = default;

    /**
     * @brief Loads and validates BT4 visual-model metadata from a `.bin` file.
     */
    void load(const std::string& path) override;

    /**
     * @brief Evaluates a position and publishes BT4-style token activations.
     */
    void evaluate(const cnnv::chess::Position& pos,
                  ActivationSnapshot& out) const override;

    /**
     * @brief Display name for the architecture selector.
     */
    std::string name() const override {
        return m_realLoaded ? "BT4 transformer (real weights)"
                            : "BT4 native token transformer";
    }

    /**
     * @brief Always true: real weights when loaded, deterministic built-in
     * weights otherwise.
     */
    bool isLoaded() const noexcept { return true; }

    /** @brief True when real BT4J weights are loaded (vs the synthetic path). */
    bool hasRealWeights() const noexcept { return m_realLoaded; }

   private:
    bool m_realLoaded = false;
    Bt4RealWeights m_realWeights;
};

}  // namespace cnnv::nn::lc0_bt4
