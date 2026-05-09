#include "TestMain.h"

#include "chess/Fen.h"
#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_cnn/Lc0CnnEncoder.h"
#include "nn/lc0_cnn/Lc0CnnLoader.h"
#include "nn/lc0_cnn/Lc0CnnNetwork.h"
#include "nn/lc0_cnn/Lc0CnnWeights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace cnnv::chess;
using namespace cnnv::nn::lc0_cnn;

namespace {

bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

std::string tempPath(const char* suffix) {
    static int counter = 0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/cnnv_lc0cnn_%d_%s.bin", counter++, suffix);
    return buf;
}

void writeU32(std::ofstream& os, std::uint32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void writeI32(std::ofstream& os, std::int32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void writeFloatArray(std::ofstream& os, const std::vector<float>& v) {
    writeU32(os, static_cast<std::uint32_t>(v.size()));
    if (!v.empty()) {
        os.write(reinterpret_cast<const char*>(v.data()),
                 static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
}
void writeConv(std::ofstream& os, int outC, int inC, int kernel,
               const std::vector<float>& weights,
               const std::vector<float>& bias) {
    writeU32(os, static_cast<std::uint32_t>(outC));
    writeU32(os, static_cast<std::uint32_t>(inC));
    writeU32(os, static_cast<std::uint32_t>(kernel));
    writeFloatArray(os, weights);
    writeFloatArray(os, bias);
}
void writeDense(std::ofstream& os, int outDim, int inDim,
                const std::vector<float>& weights,
                const std::vector<float>& bias) {
    writeU32(os, static_cast<std::uint32_t>(outDim));
    writeU32(os, static_cast<std::uint32_t>(inDim));
    writeFloatArray(os, weights);
    writeFloatArray(os, bias);
}

// Builds a tiny zero-weight LC0J network with no residual blocks. Useful as
// a structural smoke test for the loader and forward path.
std::string writeZeroNetwork() {
    const int Cin = Encoder::kPlanes;
    const int Ct = 2;
    const int Cp = 2;
    const int Cv = 2;
    const int Vh = 4;
    const int K = 3;

    std::vector<float> stemW(static_cast<std::size_t>(Ct) * Cin * K * K, 0.0f);
    std::vector<float> stemB(Ct, 0.0f);
    std::vector<float> polyStemW(static_cast<std::size_t>(Ct) * Ct, 0.0f);
    std::vector<float> polyStemB(Ct, 0.0f);
    std::vector<float> polyOutW(static_cast<std::size_t>(Cp) * Ct, 0.0f);
    std::vector<float> polyOutB(Cp, 0.0f);
    std::vector<float> vConvW(static_cast<std::size_t>(Cv) * Ct, 0.0f);
    std::vector<float> vConvB(Cv, 0.0f);
    std::vector<float> fc1W(static_cast<std::size_t>(Vh) * (Cv * 64), 0.0f);
    std::vector<float> fc1B(Vh, 0.0f);
    std::vector<float> fc2W(static_cast<std::size_t>(3) * Vh, 0.0f);
    std::vector<float> fc2B = {1.0f, 0.0f, 0.0f};  // bias makes "win" > "draw" > "loss"

    std::vector<int> policyMap = {0, 1, -1, 2, 5, 7};

    std::string path = tempPath("zero");
    std::ofstream os(path, std::ios::binary);
    os.write("LC0J", 4);
    writeU32(os, 1);              // version
    writeU32(os, Cin);            // inputChannels
    writeU32(os, Ct);             // trunkChannels
    writeU32(os, 0);              // residualBlocks
    writeU32(os, Cp);             // policyChannels
    writeU32(os, Cv);             // valueChannels
    writeU32(os, Vh);             // valueHidden
    writeU32(os, static_cast<std::uint32_t>(policyMap.size()));  // policyMapLength
    writeU32(os, 3);              // wdlOutputs

    writeConv(os, Ct, Cin, K, stemW, stemB);
    writeConv(os, Ct, Ct, 1, polyStemW, polyStemB);
    writeConv(os, Cp, Ct, 1, polyOutW, polyOutB);
    writeConv(os, Cv, Ct, 1, vConvW, vConvB);
    writeDense(os, Vh, Cv * 64, fc1W, fc1B);
    writeDense(os, 3, Vh, fc2W, fc2B);
    writeU32(os, static_cast<std::uint32_t>(policyMap.size()));
    for (int v : policyMap) writeI32(os, v);
    return path;
}

}  // namespace

TEST(lc0cnn_encoder_startpos_pawn_rank2) {
    Position pos;
    pos.setStartpos();
    auto planes = Encoder::encode(pos);
    // Plane 0 = our pawns (history block 0 plane 0). At startpos with white
    // to move that is rank 2 (squares a2..h2 = bits 8..15).
    for (int sq = 8; sq < 16; ++sq) {
        CHECK(approx(planes[0 * 64 + sq], 1.0f));
    }
    // Plane 5 = our king on e1 = bit 4.
    CHECK(approx(planes[5 * 64 + 4], 1.0f));
    // Plane 11 = their king on e8 = bit 60.
    CHECK(approx(planes[11 * 64 + 60], 1.0f));
    // Plane 108 = side-to-move-is-black: zero at startpos.
    CHECK(approx(planes[108 * 64 + 0], 0.0f));
    // Plane 111 = edge plane: all ones.
    for (int sq = 0; sq < 64; ++sq) {
        CHECK(approx(planes[111 * 64 + sq], 1.0f));
    }
    // Plane 110 reserved zeros.
    for (int sq = 0; sq < 64; ++sq) {
        CHECK(approx(planes[110 * 64 + sq], 0.0f));
    }
}

TEST(lc0cnn_encoder_castling_planes_at_startpos) {
    Position pos;
    pos.setStartpos();
    auto planes = Encoder::encode(pos);
    // 104 we-Q, 105 we-K, 106 they-Q, 107 they-K. All four available at startpos.
    for (int p = 104; p < 108; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            CHECK(approx(planes[p * 64 + sq], 1.0f));
        }
    }
}

TEST(lc0cnn_encoder_black_to_move_mirror) {
    // Make a black-to-move position by flipping side-to-move on startpos.
    // After the side-to-move flip, "our" pieces (black) should appear on the
    // first 6 perspective planes, rank-mirrored to rank 1/2.
    Position pos;
    pos.setStartpos();
    pos.setSideToMove(Color::Black);
    auto planes = Encoder::encode(pos);

    // Plane 108 = stm-is-black, all ones.
    for (int sq = 0; sq < 64; ++sq) {
        CHECK(approx(planes[108 * 64 + sq], 1.0f));
    }
    // Our pawns (plane 0) should be rank 2 from black's perspective, which
    // after the rank-mirror corresponds to original rank 7 mirrored → rank 2.
    // That is, bits 8..15 (a2..h2) should be set.
    for (int sq = 8; sq < 16; ++sq) {
        CHECK(approx(planes[0 * 64 + sq], 1.0f));
    }
    // Our king (plane 5) was black king on e8 = bit 60; mirrored bit 4 (e1).
    CHECK(approx(planes[5 * 64 + 4], 1.0f));
    // Their king (plane 11) was white king on e1 = bit 4; mirrored bit 60.
    CHECK(approx(planes[11 * 64 + 60], 1.0f));
}

TEST(lc0cnn_loader_zero_network_roundtrip) {
    std::string path = writeZeroNetwork();
    Weights w = Loader::load(path);
    CHECK_EQ(w.inputChannels, Encoder::kPlanes);
    CHECK_EQ(w.trunkChannels, 2);
    CHECK_EQ(w.policyChannels, 2);
    CHECK_EQ(w.valueChannels, 2);
    CHECK_EQ(w.valueHidden, 4);
    CHECK_EQ(w.wdlOutputs, 3);
    CHECK_EQ(w.blocks.size(), static_cast<std::size_t>(0));
    CHECK_EQ(w.inputLayer.kernel, 3);
    CHECK_EQ(w.policyStem.kernel, 1);
    CHECK_EQ(w.valueFc1.inDim, 2 * 64);
    CHECK_EQ(w.valueFc1.outDim, 4);
    CHECK_EQ(w.valueFc2.outDim, 3);
    CHECK_EQ(w.policyMap.size(), static_cast<std::size_t>(6));
    std::remove(path.c_str());
}

TEST(lc0cnn_loader_rejects_bad_magic) {
    std::string path = tempPath("bad");
    {
        std::ofstream os(path, std::ios::binary);
        os.write("XXXX", 4);
        for (int i = 0; i < 32; ++i) os.put(0);
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

TEST(lc0cnn_network_evaluate_zero_weights) {
    std::string path = writeZeroNetwork();
    Network net;
    net.load(path);
    CHECK(net.isLoaded());

    Position pos;
    pos.setStartpos();
    cnnv::nn::ActivationSnapshot snap;
    net.evaluate(pos, snap);

    // All weights are 0; only the FC2 bias [1,0,0] survives.
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kInputPlanes));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kStemRelu));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kPolicyHidden));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kPolicyPlanes));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kPolicyLogits));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueConv));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueFc1));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueLogits));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueScalar));
    CHECK(snap.has(cnnv::nn::lc0_cnn::snapshot_keys::kFinalActivation));
    CHECK_EQ(snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kPolicyHidden),
             static_cast<std::size_t>(2 * 64));
    CHECK_EQ(snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kPolicyPlanes),
             static_cast<std::size_t>(2 * 64));
    CHECK_EQ(snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kValueConv),
             static_cast<std::size_t>(2 * 64));
    CHECK_EQ(snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kValueFc1),
             static_cast<std::size_t>(4));
    CHECK_EQ(snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kValueLogits),
             static_cast<std::size_t>(3));

    const float* wdl = snap.data(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl);
    // softmax([1, 0, 0]) ≈ [0.576, 0.212, 0.212]
    const float e0 = std::exp(1.0f);
    const float total = e0 + 2.0f;
    CHECK(approx(wdl[0], e0 / total, 1e-3f));
    CHECK(approx(wdl[1], 1.0f / total, 1e-3f));
    CHECK(approx(wdl[2], 1.0f / total, 1e-3f));
    CHECK(approx(snap.data(cnnv::nn::lc0_cnn::snapshot_keys::kValueScalar)[0],
                 wdl[0] - wdl[2], 1e-3f));

    // Stem after ReLU is all zero (zero weights and zero biases).
    const float* stem = snap.data(cnnv::nn::lc0_cnn::snapshot_keys::kStemRelu);
    for (std::size_t i = 0; i < snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kStemRelu); ++i) {
        CHECK(approx(stem[i], 0.0f));
    }

    std::remove(path.c_str());
}
