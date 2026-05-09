#pragma once

/**
 * @file Lc0CnnWeights.h
 * @brief Parsed LC0 CNN layer and model weight containers.
 */

#include <cstddef>
#include <vector>

namespace cnnv::nn::lc0_cnn {

/**
 * @brief One convolution layer in NCHW kernel layout.
 *
 * Weights are `[outChannels, inChannels, kernel, kernel]` and bias length is
 * `outChannels`. Kernel size 1 uses the same row-major formula.
 */
struct ConvLayer {
    int inChannels = 0;
    int outChannels = 0;
    int kernel = 0;
    std::vector<float> weights;
    std::vector<float> bias;
};

/**
 * @brief Fully connected layer with row-major weights `[outDim, inDim]`.
 */
struct DenseLayer {
    int inDim = 0;
    int outDim = 0;
    std::vector<float> weights;
    std::vector<float> bias;
};

/**
 * @brief Optional squeeze-and-excitation unit on a residual block.
 *
 * The hidden FC produces `2 * channels` values split as gamma logits and extra
 * beta values. Residual output is `ReLU(sigmoid(gamma) * z + residual + beta)`.
 */
struct SeUnit {
    int channels = 0;
    int hidden = 0;
    std::vector<float> w1;  // [hidden, channels]
    std::vector<float> b1;  // [hidden]
    std::vector<float> w2;  // [2*channels, hidden]
    std::vector<float> b2;  // [2*channels]
    bool present = false;
};

/**
 * @brief One LC0 residual block: conv1, conv2, and optional SE unit.
 */
struct ResidualBlock {
    ConvLayer conv1;
    ConvLayer conv2;
    SeUnit se;
};

/**
 * @brief Parsed CRTK LC0J flat-format model weights.
 */
struct Weights {
    int inputChannels = 0;
    int trunkChannels = 0;
    int policyChannels = 0;
    int valueChannels = 0;
    int valueHidden = 0;
    int wdlOutputs = 0;
    std::size_t parameterCount = 0;

    ConvLayer inputLayer;
    std::vector<ResidualBlock> blocks;

    ConvLayer policyStem;
    ConvLayer policyOutput;

    ConvLayer valueConv;
    DenseLayer valueFc1;
    DenseLayer valueFc2;

    /**
     * @brief Maps compressed move logits to raw policy-plane indices.
     *
     * Negative entries are treated as out-of-range and zeroed.
     */
    std::vector<int> policyMap;

    /** @brief True when no usable trunk has been loaded. */
    bool empty() const noexcept { return trunkChannels == 0; }
};

}  // namespace cnnv::nn::lc0_cnn
