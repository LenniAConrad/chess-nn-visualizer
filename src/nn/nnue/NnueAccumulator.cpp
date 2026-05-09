#include "nn/nnue/NnueAccumulator.h"

#include "nn/nnue/NnueFeatureEncoder.h"
#include "nn/nnue/NnueWeights.h"

namespace cnnv::nn::nnue {

namespace chess = cnnv::chess;

Accumulator::Accumulator(const Weights& weights)
    : m_weights(weights),
      m_white(weights.hiddenSize, 0.0f),
      m_black(weights.hiddenSize, 0.0f) {
    reset();
}

void Accumulator::reset() {
    const auto& bias = m_weights.featureBias;
    for (std::size_t i = 0; i < m_white.size(); ++i) {
        m_white[i] = bias[i];
        m_black[i] = bias[i];
    }
}

void Accumulator::addFeature(bool whitePerspective, int feature, float sign) {
    auto& target = whitePerspective ? m_white : m_black;
    const std::size_t hidden = target.size();
    const std::size_t base = static_cast<std::size_t>(feature) * hidden;
    const float* w = m_weights.featureWeights.data() + base;
    for (std::size_t i = 0; i < hidden; ++i) {
        target[i] += sign * w[i];
    }
}

void Accumulator::refresh(const chess::Position& pos) {
    reset();
    auto white = FeatureEncoder::activeFeatures(pos, /*whitePerspective=*/true);
    auto black = FeatureEncoder::activeFeatures(pos, /*whitePerspective=*/false);
    for (int f : white) addFeature(true, f);
    for (int f : black) addFeature(false, f);
}

}  // namespace cnnv::nn::nnue
