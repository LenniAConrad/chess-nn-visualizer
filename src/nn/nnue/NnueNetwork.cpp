#include "nn/nnue/NnueNetwork.h"

#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/nnue/NnueAccumulator.h"
#include "nn/nnue/NnueFeatureEncoder.h"
#include "nn/nnue/NnueLoader.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cnnv::nn::nnue {

namespace chess = cnnv::chess;

namespace {

float clipped_relu(float v) noexcept {
    if (v <= 0.0f) return 0.0f;
    if (v >= 1.0f) return 1.0f;
    return v;
}

}  // namespace

Network::Network() : m_weights(Loader::makeFallback()) {}

Network::Network(Weights weights) : m_weights(std::move(weights)) {}

void Network::load(const std::string& path) {
    m_weights = Loader::load(path);
}

void Network::evaluate(const chess::Position& pos,
                       ActivationSnapshot& out) const {
    if (m_weights.empty()) {
        throw std::runtime_error("nnue::Network::evaluate: weights not loaded");
    }

    const std::size_t H = static_cast<std::size_t>(m_weights.hiddenSize);
    Accumulator acc(m_weights);
    acc.refresh(pos);

    const float* whiteAcc = acc.values(true);
    const float* blackAcc = acc.values(false);

    float* dstW = out.allocate(snapshot_keys::kAccumulatorWhite, {H});
    float* dstB = out.allocate(snapshot_keys::kAccumulatorBlack, {H});
    std::copy_n(whiteAcc, H, dstW);
    std::copy_n(blackAcc, H, dstB);

    const bool whiteToMove = (pos.sideToMove() == chess::Color::White);
    const float* us   = whiteToMove ? whiteAcc : blackAcc;
    const float* them = whiteToMove ? blackAcc : whiteAcc;

    float* clipUs   = out.allocate(snapshot_keys::kAccumulatorClippedUs, {H});
    float* clipThem = out.allocate(snapshot_keys::kAccumulatorClippedThem, {H});

    float raw = m_weights.outputBias;
    for (std::size_t i = 0; i < H; ++i) {
        clipUs[i]   = clipped_relu(us[i]);
        clipThem[i] = clipped_relu(them[i]);
        raw += m_weights.outputWeights[i] * clipUs[i];
        raw += m_weights.outputWeights[H + i] * clipThem[i];
    }
    const float centipawns = raw * m_weights.outputScale;

    out.allocate(snapshot_keys::kValueCentipawns, {1})[0] = centipawns;
    out.store(snapshot_keys::kFeatureBias, {H}, m_weights.featureBias.data());
    out.store(snapshot_keys::kOutputWeightsUs, {H}, m_weights.outputWeights.data());
    out.store(snapshot_keys::kOutputWeightsThem, {H},
              m_weights.outputWeights.data() + H);

    std::vector<float> outContribUs(H);
    std::vector<float> outContribThem(H);
    for (std::size_t i = 0; i < H; ++i) {
        outContribUs[i] =
            m_weights.outputWeights[i] * clipUs[i] * m_weights.outputScale;
        outContribThem[i] =
            m_weights.outputWeights[H + i] * clipThem[i] * m_weights.outputScale;
    }
    out.store(snapshot_keys::kOutputContributionUs, {H}, outContribUs.data());
    out.store(snapshot_keys::kOutputContributionThem, {H}, outContribThem.data());
    const float affine[] = {
        m_weights.outputBias,
        m_weights.outputScale,
        raw,
        centipawns,
    };
    out.store(snapshot_keys::kOutputAffine, {4}, affine);

    // Per-square activation overlay derived from the |feature contribution|
    // of each non-king piece on the board to the side-to-move accumulator.
    // Useful for the BoardView heatmap; intensities are normalised to [0, 1]
    // by the consumer (BoardView clamps anyway).
    float* contribW = out.allocate(snapshot_keys::kPieceContributionWhite, {64});
    float* contribB = out.allocate(snapshot_keys::kPieceContributionBlack, {64});

    auto contributeFor = [&](bool whitePerspective, float* dst) {
        const int king = FeatureEncoder::perspectiveKingSquare(pos, whitePerspective);
        for (int sq = 0; sq < 64; ++sq) {
            chess::Piece p = pos.pieceAt(sq);
            const int plane = FeatureEncoder::piecePlane(p, whitePerspective);
            if (plane < 0) {
                dst[sq] = 0.0f;
                continue;
            }
            const int oriented = FeatureEncoder::orientSquare(sq, whitePerspective);
            const int feature = FeatureEncoder::encodeFeature(king, plane, oriented);
            const std::size_t base =
                static_cast<std::size_t>(feature) * H;
            float mag = 0.0f;
            for (std::size_t i = 0; i < H; ++i) {
                const float w = m_weights.featureWeights[base + i];
                mag += w * w;
            }
            dst[sq] = std::sqrt(mag);
        }
    };
    contributeFor(true, contribW);
    contributeFor(false, contribB);

    // Active feature index lists. Stored as float buffers keyed by the
    // feature index count via shape — keeps the snapshot type-uniform.
    auto storeFeatures = [&](bool whitePerspective, const char* key,
                             const char* weightsKey) {
        std::vector<int> features =
            FeatureEncoder::activeFeatures(pos, whitePerspective);
        std::vector<float> as_floats(features.begin(), features.end());
        out.store(key, {as_floats.size()}, as_floats.data());
        std::vector<float> activeWeights(features.size() * H);
        for (std::size_t row = 0; row < features.size(); ++row) {
            const std::size_t base =
                static_cast<std::size_t>(features[row]) * H;
            std::copy_n(m_weights.featureWeights.data() + base, H,
                        activeWeights.data() + row * H);
        }
        out.store(weightsKey, {features.size(), H}, activeWeights.data());
    };
    storeFeatures(true, snapshot_keys::kFeatureActiveWhite,
                  snapshot_keys::kFeatureWeightsActiveWhite);
    storeFeatures(false, snapshot_keys::kFeatureActiveBlack,
                  snapshot_keys::kFeatureWeightsActiveBlack);
}

}  // namespace cnnv::nn::nnue
