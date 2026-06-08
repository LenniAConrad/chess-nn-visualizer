#include "nn/lc0_cnn/Lc0CnnNetwork.h"

#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_cnn/Lc0CnnEncoder.h"
#include "nn/lc0_cnn/Lc0CnnLoader.h"
#include "nn/lc0_cnn/PolicyEncoder.h"
#include "nn/ops/Activations.h"
#include "nn/ops/Conv2d.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnnv::nn::lc0_cnn {

namespace chess = cnnv::chess;
namespace ops = cnnv::nn::ops;

namespace {

constexpr int kSquare = 64;

void runConv(const ConvLayer& c, const float* in, float* out) {
    const std::size_t pad = (c.kernel - 1) / 2;
    ops::conv2d(in, c.weights.data(), nullptr, out,
                static_cast<std::size_t>(c.inChannels), 8, 8,
                static_cast<std::size_t>(c.outChannels),
                static_cast<std::size_t>(c.kernel),
                static_cast<std::size_t>(c.kernel),
                /*sH=*/1, /*sW=*/1, pad, pad);
}

void addBiasReLU(float* tensor, const float* bias, int channels) {
    for (int ch = 0; ch < channels; ++ch) {
        const float b = bias[ch];
        float* row = tensor + ch * kSquare;
        for (int i = 0; i < kSquare; ++i) {
            float v = row[i] + b;
            row[i] = v > 0.0f ? v : 0.0f;
        }
    }
}

void addBias(float* tensor, const float* bias, int channels) {
    for (int ch = 0; ch < channels; ++ch) {
        const float b = bias[ch];
        float* row = tensor + ch * kSquare;
        for (int i = 0; i < kSquare; ++i) row[i] += b;
    }
}

// Adds a residual to (conv_out + bias), ReLU, write to dest.
void addResidualReLU(const float* convOut, const float* bias,
                     const float* residual, float* dest, int channels) {
    for (int ch = 0; ch < channels; ++ch) {
        const float b = bias[ch];
        const float* co = convOut + ch * kSquare;
        const float* re = residual + ch * kSquare;
        float* dst = dest + ch * kSquare;
        for (int i = 0; i < kSquare; ++i) {
            float v = co[i] + b + re[i];
            dst[i] = v > 0.0f ? v : 0.0f;
        }
    }
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void sePoolWithBias(const float* convOut, const float* bias, int channels,
                    float* pooledOut) {
    constexpr float invSquares = 1.0f / static_cast<float>(kSquare);
    for (int ch = 0; ch < channels; ++ch) {
        float sum = 0.0f;
        const float* row = convOut + ch * kSquare;
        for (int i = 0; i < kSquare; ++i) sum += row[i];
        pooledOut[ch] = sum * invSquares + bias[ch];
    }
}

void denseRelu(const float* in, const float* w, const float* b, int inDim,
               int outDim, float* out) {
    for (int o = 0; o < outDim; ++o) {
        float acc = b[o];
        const float* wb = w + static_cast<std::size_t>(o) * inDim;
        for (int i = 0; i < inDim; ++i) acc += wb[i] * in[i];
        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

void dense(const float* in, const float* w, const float* b, int inDim,
           int outDim, float* out) {
    for (int o = 0; o < outDim; ++o) {
        float acc = b[o];
        const float* wb = w + static_cast<std::size_t>(o) * inDim;
        for (int i = 0; i < inDim; ++i) acc += wb[i] * in[i];
        out[o] = acc;
    }
}

// gates layout: [gamma_logit[channels], beta_extra[channels]]
void seCombine(const float* convOut, const float* bias, const float* residual,
               float* dest, int channels, const float* gates) {
    for (int ch = 0; ch < channels; ++ch) {
        const float gamma = sigmoid(gates[ch]);
        const float betaExtra = gates[ch + channels];
        const float b = bias[ch];
        const float* co = convOut + ch * kSquare;
        const float* re = residual + ch * kSquare;
        float* dst = dest + ch * kSquare;
        for (int i = 0; i < kSquare; ++i) {
            float z = co[i] + b;
            float v = gamma * z + re[i] + betaExtra;
            dst[i] = v > 0.0f ? v : 0.0f;
        }
    }
}

std::vector<float> mapPolicy(const float* planes, std::size_t planesLen,
                             const std::vector<int>& policyMap) {
    std::vector<float> out(policyMap.size(), 0.0f);
    for (std::size_t i = 0; i < policyMap.size(); ++i) {
        const int idx = policyMap[i];
        if (idx >= 0 && static_cast<std::size_t>(idx) < planesLen) {
            out[i] = planes[idx];
        }
    }
    return out;
}

}  // namespace

Network::Network(Weights weights) : m_weights(std::move(weights)) {}

void Network::load(const std::string& path) {
    m_weights = Loader::load(path);
    m_policyReverseBuilt = false;
}

void Network::ensurePolicyReverse() const {
    if (m_policyReverseBuilt) return;
    m_policyReverse.assign(kRawPolicySize, -1);
    for (std::size_t i = 0; i < m_weights.policyMap.size(); ++i) {
        const int raw = m_weights.policyMap[i];
        if (raw >= 0 && raw < kRawPolicySize) {
            m_policyReverse[static_cast<std::size_t>(raw)] = static_cast<int>(i);
        }
    }
    m_policyReverseBuilt = true;
}

int Network::policyIndexForMove(const cnnv::chess::Position& pos,
                                cnnv::chess::Move move) const {
    if (m_weights.policyMap.empty()) return -1;
    ensurePolicyReverse();
    const int raw = rawPolicyIndex(pos, move);
    if (raw < 0 || raw >= static_cast<int>(m_policyReverse.size())) return -1;
    return m_policyReverse[static_cast<std::size_t>(raw)];
}

std::string Network::name() const {
    if (m_weights.empty()) return "LC0 CNN (no weights)";
    return "LC0 CNN " + std::to_string(m_weights.blocks.size()) + "x" +
           std::to_string(m_weights.trunkChannels);
}

void Network::evaluate(const chess::Position& pos,
                       ActivationSnapshot& out) const {
    if (m_weights.empty()) {
        throw std::runtime_error(
            "lc0_cnn::Network::evaluate: weights not loaded");
    }
    const Weights& w = m_weights;

    auto input = Encoder::encode(pos);
    out.store(snapshot_keys::kInputPlanes,
              {static_cast<std::size_t>(Encoder::kPlanes), 8, 8},
              input.data());

    const std::size_t trunkSize =
        static_cast<std::size_t>(w.trunkChannels) * kSquare;
    std::vector<float> current(trunkSize, 0.0f);
    std::vector<float> next(trunkSize, 0.0f);
    std::vector<float> tmp(trunkSize, 0.0f);
    std::vector<float> scratch(trunkSize, 0.0f);

    runConv(w.inputLayer, input.data(), current.data());
    addBiasReLU(current.data(), w.inputLayer.bias.data(), w.trunkChannels);

    out.store(snapshot_keys::kStemRelu,
              {static_cast<std::size_t>(w.trunkChannels), 8, 8},
              current.data());

    std::vector<float> sePooled, seHidden, seGates;

    for (std::size_t bi = 0; bi < w.blocks.size(); ++bi) {
        const ResidualBlock& block = w.blocks[bi];

        runConv(block.conv1, current.data(), tmp.data());
        addBiasReLU(tmp.data(), block.conv1.bias.data(), w.trunkChannels);

        runConv(block.conv2, tmp.data(), scratch.data());

        if (block.se.present) {
            const SeUnit& se = block.se;
            sePooled.assign(se.channels, 0.0f);
            seHidden.assign(se.hidden, 0.0f);
            seGates.assign(static_cast<std::size_t>(se.channels) * 2, 0.0f);
            sePoolWithBias(scratch.data(), block.conv2.bias.data(),
                           se.channels, sePooled.data());
            denseRelu(sePooled.data(), se.w1.data(), se.b1.data(),
                      se.channels, se.hidden, seHidden.data());
            dense(seHidden.data(), se.w2.data(), se.b2.data(), se.hidden,
                  se.channels * 2, seGates.data());
            seCombine(scratch.data(), block.conv2.bias.data(), current.data(),
                      next.data(), se.channels, seGates.data());
        } else {
            addResidualReLU(scratch.data(), block.conv2.bias.data(),
                            current.data(), next.data(), w.trunkChannels);
        }

        std::swap(current, next);

        out.store(snapshot_keys::blockReluKey(static_cast<int>(bi)),
                  {static_cast<std::size_t>(w.trunkChannels), 8, 8},
                  current.data());
    }

    // Per-square mean activation (final block) for the BoardView overlay.
    {
        std::vector<float> meanPerSquare(kSquare, 0.0f);
        for (int ch = 0; ch < w.trunkChannels; ++ch) {
            const float* row = current.data() + ch * kSquare;
            for (int i = 0; i < kSquare; ++i) meanPerSquare[i] += row[i];
        }
        const float inv = 1.0f / static_cast<float>(w.trunkChannels);
        for (int i = 0; i < kSquare; ++i) meanPerSquare[i] *= inv;
        out.store(snapshot_keys::kFinalActivation, {64}, meanPerSquare.data());
    }

    // Policy head.
    std::vector<float> policyHidden(trunkSize, 0.0f);
    runConv(w.policyStem, current.data(), policyHidden.data());
    addBiasReLU(policyHidden.data(), w.policyStem.bias.data(), w.trunkChannels);
    out.store(snapshot_keys::kPolicyHidden,
              {static_cast<std::size_t>(w.trunkChannels), 8, 8},
              policyHidden.data());

    std::vector<float> policyPlanes(
        static_cast<std::size_t>(w.policyChannels) * kSquare, 0.0f);
    runConv(w.policyOutput, policyHidden.data(), policyPlanes.data());
    addBias(policyPlanes.data(), w.policyOutput.bias.data(), w.policyChannels);
    out.store(snapshot_keys::kPolicyPlanes,
              {static_cast<std::size_t>(w.policyChannels), 8, 8},
              policyPlanes.data());

    auto policy = mapPolicy(policyPlanes.data(), policyPlanes.size(),
                            w.policyMap);
    out.store(snapshot_keys::kPolicyLogits, {policy.size()}, policy.data());

    // Value head.
    std::vector<float> valueInput(
        static_cast<std::size_t>(w.valueChannels) * kSquare, 0.0f);
    runConv(w.valueConv, current.data(), valueInput.data());
    addBiasReLU(valueInput.data(), w.valueConv.bias.data(), w.valueChannels);
    out.store(snapshot_keys::kValueConv,
              {static_cast<std::size_t>(w.valueChannels), 8, 8},
              valueInput.data());

    const int fc1In = w.valueFc1.inDim;
    if (fc1In != static_cast<int>(valueInput.size())) {
        throw std::runtime_error(
            "lc0_cnn::Network: value conv output (" +
            std::to_string(valueInput.size()) +
            ") does not match valueFc1.inDim (" + std::to_string(fc1In) + ")");
    }
    std::vector<float> fc1Out(w.valueFc1.outDim, 0.0f);
    denseRelu(valueInput.data(), w.valueFc1.weights.data(),
              w.valueFc1.bias.data(), w.valueFc1.inDim, w.valueFc1.outDim,
              fc1Out.data());
    out.store(snapshot_keys::kValueFc1, {fc1Out.size()}, fc1Out.data());

    std::vector<float> logits(w.valueFc2.outDim, 0.0f);
    dense(fc1Out.data(), w.valueFc2.weights.data(), w.valueFc2.bias.data(),
          w.valueFc2.inDim, w.valueFc2.outDim, logits.data());
    out.store(snapshot_keys::kValueLogits, {logits.size()}, logits.data());

    std::vector<float> wdl = logits;
    ops::softmax(wdl.data(), wdl.size());
    out.store(snapshot_keys::kValueWdl, {wdl.size()}, wdl.data());

    if (wdl.size() == 3) {
        const float scalar = wdl[0] - wdl[2];
        out.store(snapshot_keys::kValueScalar, {1}, &scalar);
    }
}

}  // namespace cnnv::nn::lc0_cnn
