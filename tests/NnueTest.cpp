#include "TestMain.h"

#include "BinaryTestWriter.h"
#include "chess/Fen.h"
#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/nnue/NnueAccumulator.h"
#include "nn/nnue/NnueFeatureEncoder.h"
#include "nn/nnue/NnueLoader.h"
#include "nn/nnue/NnueNetwork.h"
#include "nn/nnue/NnueWeights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace cnnv::chess;
using namespace cnnv::nn::nnue;

namespace {

namespace bin = cnnv_test::bin;

bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

std::string tempPath(const char* suffix) {
    static int counter = 0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/cnnv_nnue_%d_%s.bin", counter++, suffix);
    return buf;
}

// Builds a minimal CRTK-format NNUE file at `path` with the given weights and
// hidden size. Returns true on success.
bool writeCrtkNnue(const std::string& path, int hiddenSize,
                   const std::vector<float>& featureBias,
                   const std::vector<float>& featureWeights,
                   const std::vector<float>& outputWeights, float outputBias,
                   float outputScale) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os.write("NNUE", 4);
    bin::writeU32(os, 1);
    bin::writeU32(os, FeatureEncoder::kFeatureCount);
    bin::writeU32(os, static_cast<std::uint32_t>(hiddenSize));
    bin::writeF32(os, outputScale);
    bin::writeFloatArrayU32(os, featureBias);
    bin::writeFloatArrayU32(os, featureWeights);
    bin::writeFloatArrayU32(os, outputWeights);
    bin::writeF32(os, outputBias);
    return os.good();
}

}  // namespace

TEST(nnue_feature_encoder_startpos_active_count) {
    Position pos;
    pos.setStartpos();
    auto white = FeatureEncoder::activeFeatures(pos, true);
    auto black = FeatureEncoder::activeFeatures(pos, false);
    // 16 pieces per side, 2 of which are kings (not emitted) → 30 features.
    CHECK_EQ(white.size(), static_cast<std::size_t>(30));
    CHECK_EQ(black.size(), static_cast<std::size_t>(30));
}

TEST(nnue_feature_encoder_indices_in_range) {
    Position pos;
    pos.setStartpos();
    auto features = FeatureEncoder::activeFeatures(pos, true);
    for (int f : features) {
        CHECK(f >= 0);
        CHECK(f < FeatureEncoder::kFeatureCount);
    }
    // No duplicate features.
    std::set<int> uniq(features.begin(), features.end());
    CHECK_EQ(uniq.size(), features.size());
}

TEST(nnue_feature_encoder_orient_square) {
    // White perspective: identity.
    CHECK_EQ(FeatureEncoder::orientSquare(0,  true), 0);
    CHECK_EQ(FeatureEncoder::orientSquare(63, true), 63);
    // Black perspective: mirror rank (square ^ 56). a1 ↔ a8, h8 ↔ h1.
    CHECK_EQ(FeatureEncoder::orientSquare(0,  false), 56);
    CHECK_EQ(FeatureEncoder::orientSquare(63, false), 7);
}

TEST(nnue_feature_encoder_piece_planes) {
    Piece wp{Color::White, PieceType::Pawn};
    Piece bn{Color::Black, PieceType::Knight};
    Piece wk{Color::White, PieceType::King};
    CHECK_EQ(FeatureEncoder::piecePlane(wp, true),  FeatureEncoder::kOwnPawn);
    CHECK_EQ(FeatureEncoder::piecePlane(wp, false), FeatureEncoder::kEnemyPawn);
    CHECK_EQ(FeatureEncoder::piecePlane(bn, true),  FeatureEncoder::kEnemyKnight);
    CHECK_EQ(FeatureEncoder::piecePlane(bn, false), FeatureEncoder::kOwnKnight);
    CHECK_EQ(FeatureEncoder::piecePlane(wk, true),  -1);
    CHECK_EQ(FeatureEncoder::piecePlane(kNoPiece, true), -1);
}

TEST(nnue_loader_roundtrip_synthetic_h2) {
    const int H = 2;
    std::vector<float> featureBias = {0.5f, -0.25f};
    // Feature-major: feature f, hidden i → featureWeights[f * H + i].
    // Keep feature 0 nonzero, all others zero, so we can predict the result.
    std::vector<float> featureWeights(
        static_cast<std::size_t>(FeatureEncoder::kFeatureCount) * H, 0.0f);
    featureWeights[0 * H + 0] = 1.0f;
    featureWeights[0 * H + 1] = 0.5f;
    std::vector<float> outputWeights = {2.0f, 3.0f, -1.0f, 0.5f};  // [us, them]
    const float outputBias = 0.1f;
    const float outputScale = 100.0f;

    std::string path = tempPath("synthetic_h2");
    CHECK(writeCrtkNnue(path, H, featureBias, featureWeights, outputWeights,
                        outputBias, outputScale));

    Weights w = Loader::load(path);
    CHECK_EQ(w.hiddenSize, H);
    CHECK(approx(w.featureBias[0], 0.5f));
    CHECK(approx(w.featureBias[1], -0.25f));
    CHECK(approx(w.outputBias, 0.1f));
    CHECK(approx(w.outputScale, 100.0f));
    CHECK_EQ(w.featureWeights.size(),
             static_cast<std::size_t>(FeatureEncoder::kFeatureCount) * H);
    CHECK(approx(w.featureWeights[0], 1.0f));
    CHECK(approx(w.featureWeights[1], 0.5f));

    std::remove(path.c_str());
}

TEST(nnue_loader_rejects_bad_magic) {
    std::string path = tempPath("bad_magic");
    {
        std::ofstream os(path, std::ios::binary);
        os.write("XXXX", 4);
        for (int i = 0; i < 16; ++i) os.put(0);
    }
    bool threw = false;
    try {
        Loader::load(path);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}

TEST(nnue_accumulator_refresh_uses_bias_when_empty_board) {
    // Synthetic small network with H=4. With kings only and zero feature
    // weights, the accumulator should equal the feature bias on both sides.
    const int H = 4;
    Weights w;
    w.hiddenSize = H;
    w.featureBias = {1.0f, 2.0f, 3.0f, 4.0f};
    w.featureWeights.assign(
        static_cast<std::size_t>(FeatureEncoder::kFeatureCount) * H, 0.0f);
    w.outputWeights.assign(static_cast<std::size_t>(H) * 2, 0.0f);
    w.outputBias = 0.0f;
    w.outputScale = 1.0f;

    Position pos;
    pos.clear();
    pos.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    pos.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    pos.setSideToMove(Color::White);

    Accumulator acc(w);
    acc.refresh(pos);
    for (std::size_t i = 0; i < static_cast<std::size_t>(H); ++i) {
        CHECK(approx(acc.values(true)[i],  w.featureBias[i]));
        CHECK(approx(acc.values(false)[i], w.featureBias[i]));
    }
}

TEST(nnue_network_evaluate_populates_snapshot) {
    // Build an in-memory weights set where: featureBias = 0, every active
    // feature contributes +0.1 to hidden 0, output uses unit weights, scale
    // 1.0, bias 0. So at startpos the side-to-move accumulator has each unit
    // = 30 * 0.1 = 3.0 → clipped to 1.0; output = H*1*1 + H*1*1 = 2*H.
    const int H = 4;
    Weights w;
    w.hiddenSize = H;
    w.featureBias.assign(static_cast<std::size_t>(H), 0.0f);
    w.featureWeights.assign(
        static_cast<std::size_t>(FeatureEncoder::kFeatureCount) * H, 0.1f);
    w.outputWeights.assign(static_cast<std::size_t>(H) * 2, 1.0f);
    w.outputBias = 0.0f;
    w.outputScale = 1.0f;

    Network net(std::move(w));
    Position pos;
    pos.setStartpos();
    cnnv::nn::ActivationSnapshot snap;
    net.evaluate(pos, snap);

    CHECK(snap.has(snapshot_keys::kAccumulatorWhite));
    CHECK(snap.has(snapshot_keys::kAccumulatorBlack));
    CHECK(snap.has(snapshot_keys::kValueCentipawns));
    CHECK_EQ(snap.size(snapshot_keys::kAccumulatorWhite), static_cast<std::size_t>(H));

    // Each white-perspective accumulator entry = 30 active features × 0.1 = 3.0.
    for (std::size_t i = 0; i < static_cast<std::size_t>(H); ++i) {
        CHECK(approx(snap.data(snapshot_keys::kAccumulatorWhite)[i], 3.0f));
        CHECK(approx(snap.data(snapshot_keys::kAccumulatorBlack)[i], 3.0f));
    }

    // Clipped ReLU caps at 1.0, output weights all 1.0 → centipawns = 2*H.
    CHECK(approx(snap.data(snapshot_keys::kValueCentipawns)[0],
                 static_cast<float>(2 * H)));

    // Active-feature lists exposed as float buffers of length 30.
    CHECK_EQ(snap.size(snapshot_keys::kFeatureActiveWhite),
             static_cast<std::size_t>(30));
    CHECK_EQ(snap.size(snapshot_keys::kFeatureActiveBlack),
             static_cast<std::size_t>(30));
}

TEST(nnue_network_fallback_evaluate_zero_centipawns) {
    Network net;  // default-constructs the fallback (1 hidden unit, all zero).
    Position pos;
    pos.setStartpos();
    cnnv::nn::ActivationSnapshot snap;
    net.evaluate(pos, snap);
    CHECK(approx(snap.data(snapshot_keys::kValueCentipawns)[0], 0.0f));
}
