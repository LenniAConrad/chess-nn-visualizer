#pragma once

/**
 * @file Bt4RealWeights.h
 * @brief In-memory mirror of the CRTK BT4J v2 weight records.
 *
 * These structs mirror the nested `Network.Weights` records in chess-rtk's
 * `chess.nn.lc0.bt4.Network`. They hold already-decoded float32 tensors in the
 * same row-major dense-layer order as the Java reference so the native forward
 * pass can run the BT4 CPU path without a JVM. Optional components (preproc,
 * gates, embedding FFN, smolgen) are represented by empty/zero-sized members
 * gated by boolean flags on the architecture.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace cnnv::nn::lc0_bt4 {

/**
 * @brief Activation functions supported by BT4 attention-body networks.
 *
 * Mirrors `Network.Activation` in chess-rtk.
 */
enum class Bt4Activation : std::uint8_t {
    None = 0,
    Relu = 1,
    Mish = 2,
    Swish = 3,
    Tanh = 4,
};

/**
 * @brief Dense layer with row-major weights `[outDim, inDim]` plus a bias.
 *
 * Mirrors `Network.Dense`. `out[o] = bias[o] + sum_i weights[o*inDim + i] *
 * x[i]`.
 */
struct Dense {
    int inDim = 0;
    int outDim = 0;
    std::vector<float> weights;  ///< row-major `[outDim * inDim]`
    std::vector<float> bias;     ///< `[outDim]`
};

/**
 * @brief Two-layer feed-forward block with an activation between dense layers.
 *
 * Mirrors `Network.Ffn`.
 */
struct Ffn {
    Dense dense1;
    Dense dense2;
};

/**
 * @brief Per-attention smolgen bias generator.
 *
 * Mirrors `Network.Smolgen`. Produces a per-head `[tokens, tokens]` additive
 * bias on the attention scores from the encoder-block input embeddings.
 */
struct Smolgen {
    Dense compress;
    Dense dense1;
    std::vector<float> ln1Gamma;
    std::vector<float> ln1Beta;
    Dense dense2;
    std::vector<float> ln2Gamma;
    std::vector<float> ln2Beta;
};

/**
 * @brief Multi-head attention weights with an optional smolgen generator.
 *
 * Mirrors `Network.Attention`. `hasSmolgen` distinguishes a populated `smolgen`
 * member from an absent one.
 */
struct Attention {
    int heads = 0;
    Dense query;
    Dense key;
    Dense value;
    Dense out;
    bool hasSmolgen = false;
    Smolgen smolgen;
};

/**
 * @brief One transformer encoder block.
 *
 * Mirrors `Network.EncoderBlock`.
 */
struct EncoderBlock {
    Attention attention;
    Dense ffnIn;
    Dense ffnOut;
    std::vector<float> ln1Gamma;
    std::vector<float> ln1Beta;
    std::vector<float> ln2Gamma;
    std::vector<float> ln2Beta;
    Bt4Activation activation = Bt4Activation::Mish;
    float alpha = 1.0f;
};

/**
 * @brief BT4 input embedding stack.
 *
 * Mirrors `Network.InputStack`. Optional components are gated by the boolean
 * flags so an absent record is represented by `false` plus an empty member.
 */
struct InputStack {
    bool hasPreproc = false;
    Dense preproc;

    Dense embedding;

    bool hasEmbLn = false;  ///< PE_DENSE only
    std::vector<float> embLnGamma;
    std::vector<float> embLnBeta;

    bool hasGates = false;
    std::vector<float> multGate;
    std::vector<float> addGate;

    bool hasEmbFfn = false;
    Ffn embFfn;
    std::vector<float> embFfnLnGamma;
    std::vector<float> embFfnLnBeta;
};

/**
 * @brief Attention policy head.
 *
 * Mirrors `Network.PolicyHead`. The policy-only encoder blocks never carry
 * smolgen weights.
 */
struct PolicyHead {
    Dense embedding;
    std::vector<EncoderBlock> encoders;
    Dense query;
    Dense key;
    std::vector<float> promotionWeights;  ///< `[4 * query.outDim]`
    Bt4Activation activation = Bt4Activation::None;
};

/**
 * @brief WDL value head.
 *
 * Mirrors `Network.ValueHead`. `fc2` outputs three logits ordered win, draw,
 * loss.
 */
struct ValueHead {
    Dense embedding;
    Dense fc1;
    Dense fc2;
    Bt4Activation activation = Bt4Activation::None;
};

/**
 * @brief BT4 architecture metadata.
 *
 * Mirrors `Architecture`. `inputEmbedding`/`inputFormat` are kept as the raw
 * enum names from the file; only the bits the forward pass needs are decoded.
 */
struct Arch {
    std::string name;
    std::string inputFormat;     ///< enum name, e.g. "BT4_CANONICAL_112"
    std::string inputEmbedding;  ///< enum name, e.g. "PE_DENSE", "PE_MAP", "NONE"
    int inputChannels = 0;
    int tokens = 0;
    int embeddingSize = 0;
    int encoderLayers = 0;
    int attentionHeads = 0;
    int policySize = 0;
    float layerNormEpsilon = 1.0e-3f;
    int ffnHiddenSize = 0;
    int smolgenHiddenChannels = 0;
    int smolgenHiddenSize = 0;
    int smolgenPerHeadDim = 0;
    int smolgenGlobalSize = 0;
    Bt4Activation defaultActivation = Bt4Activation::Mish;
    Bt4Activation smolgenActivation = Bt4Activation::Swish;
    Bt4Activation ffnActivation = Bt4Activation::Mish;
    bool hasInputPreproc = false;
    bool hasInputEmbFfn = false;
    bool hasInputGates = false;
    bool hasSmolgen = false;

    /** @brief True when the input format requests LC0 canonicalization. */
    bool isCanonical() const { return inputFormat == "BT4_CANONICAL_112"; }
};

/**
 * @brief Complete parsed BT4 weight bundle.
 *
 * Mirrors `Network.Weights`. `smolgenW` is the shared global smolgen projection
 * stored row-major as `[tokens*tokens, smolgenPerHeadDim]`: each of the
 * `tokens*tokens` output positions owns a contiguous `smolgenPerHeadDim` row
 * (matching `wRow = o * perHead` in the Java forward pass). Empty when
 * `arch.hasSmolgen` is false.
 */
struct Bt4RealWeights {
    Arch arch;
    InputStack input;
    std::vector<EncoderBlock> encoders;
    std::vector<float> smolgenW;
    PolicyHead policyHead;
    ValueHead valueHead;
};

}  // namespace cnnv::nn::lc0_bt4
