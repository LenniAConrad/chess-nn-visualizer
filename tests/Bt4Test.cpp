#include "TestMain.h"

#include "BinaryTestWriter.h"
#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace bin = cnnv_test::bin;

constexpr int kBt4InputChannels = 112;
constexpr int kBt4Tokens = 64;
constexpr int kBt4D = 8;
constexpr int kBt4Heads = 1;
constexpr int kBt4PolicySize = 1858;
constexpr int kBt4FfnHidden = 16;

std::string bt4TempPath() {
    static int counter = 0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/cnnv_bt4_visual_%d.bin",
                  counter++);
    return buf;
}

std::string runtimeBt4Path() {
    for (const char* path : {
             "models/lc0-bt4-tiny-96x4x4h.bin",
             "../models/lc0-bt4-tiny-96x4x4h.bin",
             "../../models/lc0-bt4-tiny-96x4x4h.bin",
         }) {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) return path;
    }
    return {};
}

void writeDense(std::ofstream& os, int inDim, int outDim,
                bool corruptWeights = false) {
    bin::writeI32(os, inDim);
    bin::writeI32(os, outDim);
    const std::size_t weights =
        corruptWeights ? 0U : static_cast<std::size_t>(inDim) * outDim;
    bin::writeFloatArrayI32(os, std::vector<float>(weights, 0.0f));
    bin::writeFloatArrayI32(
        os, std::vector<float>(static_cast<std::size_t>(outDim), 0.0f));
}

void writeDenseWithRawWeightCount(std::ofstream& os,
                                  int inDim,
                                  int outDim,
                                  std::int32_t weightCount) {
    bin::writeI32(os, inDim);
    bin::writeI32(os, outDim);
    bin::writeI32(os, weightCount);
}

/**
 * @brief Schreibt den gemeinsamen Minimal-Header fuer echte BT4J-Testdateien.
 *
 * Die Loader-Tests wollen immer nur ein bestimmtes Feld zerlegen. Der Rest
 * bleibt langweilig gleich, damit man beim Lesen sofort sieht, wo der kaputte
 * Kram sitzt.
 */
void writeBt4JHeader(std::ofstream& os,
                     const char* name,
                     std::int32_t encoderLayers) {
    bin::writeU32(os, 0x4A345442u);
    bin::writeU32(os, 2);

    bin::writeStringI32(os, name);
    bin::writeStringI32(os, "BT4_CANONICAL_112");
    bin::writeStringI32(os, "PE_LEGACY");
    bin::writeI32(os, kBt4InputChannels);
    bin::writeI32(os, kBt4Tokens);
    bin::writeI32(os, kBt4D);
    bin::writeI32(os, encoderLayers);
    bin::writeI32(os, kBt4Heads);
    bin::writeI32(os, kBt4PolicySize);
    bin::writeF32(os, 1.0e-3f);
    bin::writeI32(os, kBt4FfnHidden);
    bin::writeI32(os, 0);
    bin::writeI32(os, 0);
    bin::writeI32(os, 0);
    bin::writeI32(os, 0);
    bin::writeStringI32(os, "MISH");
    bin::writeStringI32(os, "SWISH");
    bin::writeStringI32(os, "MISH");
    bin::writeU8(os, 0);
    bin::writeU8(os, 0);
    bin::writeU8(os, 0);
    bin::writeU8(os, 0);
}

std::string writeBt4VisualModel() {
    std::string path = bt4TempPath();
    std::ofstream os(path, std::ios::binary);
    os.write("BT4V", 4);
    bin::writeU32(os, 1);          // version
    bin::writeU32(os, 112);        // input channels
    bin::writeU32(os, 64);         // tokens
    bin::writeU32(os, 176);        // token width
    bin::writeU32(os, 64);         // model dim
    bin::writeU32(os, 15);         // blocks
    bin::writeU32(os, 8);          // heads
    bin::writeU32(os, 1858);       // policy size
    bin::writeU32(os, 0x43504E56); // deterministic visual seed
    return path;
}

std::string writeMalformedBt4RealModel(bool oversizedArray = false) {
    constexpr int kPolicyD = 8;
    constexpr int kValueD = 4;
    constexpr int kValueHidden = 8;

    std::string path = bt4TempPath();
    std::ofstream os(path, std::ios::binary);
    writeBt4JHeader(os, "bt4-malformed-test", 0);

    if (oversizedArray) {
        // Kaputtes Feld: riesiger Count, aber keine Payload. Der Loader muss
        // vorher raus, nicht erst nach einer sinnlosen Allokation.
        writeDenseWithRawWeightCount(os, kBt4InputChannels, kBt4D,
                                     std::numeric_limits<std::int32_t>::max());
        return path;
    }

    // Kaputtes Feld: Metadata sagt [8,112], die Payload sagt 0 floats. Genau
    // dieser Mismatch soll beim Laden auffallen, nicht spaeter im Evaluator.
    writeDense(os, kBt4InputChannels, kBt4D, true);
    bin::writeI32(os, 0);  // Encoder-Block-Count.

    writeDense(os, kBt4D, kPolicyD);
    bin::writeI32(os, 0);  // Policy-Encoder-Block-Count.
    writeDense(os, kPolicyD, kPolicyD);
    writeDense(os, kPolicyD, kPolicyD);
    bin::writeFloatArrayI32(os, std::vector<float>(4U * kPolicyD, 0.0f));
    bin::writeStringI32(os, "MISH");

    writeDense(os, kBt4D, kValueD);
    writeDense(os, kBt4Tokens * kValueD, kValueHidden);
    writeDense(os, kValueHidden, 3);
    bin::writeStringI32(os, "MISH");
    return path;
}

std::string writeBt4RealModelWithHugeEncoderCount() {
    std::string path = bt4TempPath();
    std::ofstream os(path, std::ios::binary);
    writeBt4JHeader(os, "bt4-huge-count-test",
                    std::numeric_limits<std::int32_t>::max());
    writeDense(os, kBt4InputChannels, kBt4D);
    bin::writeI32(os, std::numeric_limits<std::int32_t>::max());
    return path;
}

}  // namespace

TEST(bt4_accepts_canonical_visual_bin_metadata) {
    cnnv::nn::lc0_bt4::Network net;
    const std::string path = writeBt4VisualModel();
    net.load(path);
    std::remove(path.c_str());
}

TEST(bt4_native_evaluator_populates_tensor_snapshots) {
    cnnv::chess::Position pos;
    pos.setStartpos();

    cnnv::nn::lc0_bt4::Network net;
    cnnv::nn::ActivationSnapshot snap;
    net.evaluate(pos, snap);

    namespace keys = cnnv::nn::lc0_bt4::snapshot_keys;
    using Net = cnnv::nn::lc0_bt4::Network;

    CHECK_EQ(snap.size(keys::kTokenFeatures),
             static_cast<std::size_t>(Net::kTokens * Net::kTokenWidth));
    CHECK_EQ(snap.size(keys::kEmbedding),
             static_cast<std::size_t>(Net::kTokens * Net::kModelDim));
    CHECK_EQ(snap.size(keys::blockAttentionKey(0)),
             static_cast<std::size_t>(Net::kHeads * Net::kTokens * Net::kTokens));
    CHECK_EQ(snap.size(keys::blockOutputKey(Net::kSyntheticBlocks - 1)),
             static_cast<std::size_t>(Net::kTokens * Net::kModelDim));
    CHECK_EQ(snap.size(keys::kFinalTokenMagnitude),
             static_cast<std::size_t>(Net::kTokens));
    CHECK_EQ(snap.size(keys::kBoardSalience),
             static_cast<std::size_t>(Net::kTokens));
    CHECK_EQ(snap.size(keys::kPolicyLogits),
             static_cast<std::size_t>(Net::kPolicySize));
    CHECK_EQ(snap.size(keys::kValueWdl), static_cast<std::size_t>(3));
    CHECK_EQ(snap.size(keys::kValueScalar), static_cast<std::size_t>(1));

    const float* wdl = snap.data(keys::kValueWdl);
    const float sum = wdl[0] + wdl[1] + wdl[2];
    CHECK(std::fabs(sum - 1.0f) < 1e-4f);

    const float* salience = snap.data(keys::kBoardSalience);
    float maxSalience = 0.0f;
    float occupiedSum = 0.0f;
    float emptySum = 0.0f;
    int occupiedCount = 0;
    int emptyCount = 0;
    for (int sq = 0; sq < 64; ++sq) {
        CHECK(salience[sq] >= -1e-6f);
        CHECK(salience[sq] <= 1.0001f);
        maxSalience = std::max(maxSalience, salience[sq]);
        if (pos.pieceAt(sq).isNone()) {
            emptySum += salience[sq];
            ++emptyCount;
        } else {
            occupiedSum += salience[sq];
            ++occupiedCount;
        }
    }
    CHECK(maxSalience > 0.9f);
    CHECK(occupiedCount > 0 && emptyCount > 0);
    CHECK(occupiedSum / static_cast<float>(occupiedCount) >
          emptySum / static_cast<float>(emptyCount));
}

TEST(bt4_generated_runtime_model_loads_and_evaluates) {
    const std::string path = runtimeBt4Path();
    if (path.empty()) return;

    cnnv::nn::lc0_bt4::Network net;
    net.load(path);
    CHECK(net.hasRealWeights());

    cnnv::chess::Position pos;
    pos.setStartpos();
    cnnv::nn::ActivationSnapshot snap;
    net.evaluate(pos, snap);

    namespace keys = cnnv::nn::lc0_bt4::snapshot_keys;
    CHECK_EQ(snap.size(keys::kInputPlanes), static_cast<std::size_t>(112 * 64));
    CHECK_EQ(snap.size(keys::kEmbedding), static_cast<std::size_t>(64 * 96));
    CHECK_EQ(snap.size(keys::blockAttentionKey(0)),
             static_cast<std::size_t>(4 * 64 * 64));
    CHECK_EQ(snap.size(keys::blockOutputKey(3)), static_cast<std::size_t>(64 * 96));
    CHECK(!snap.has(keys::blockOutputKey(4)));
    CHECK_EQ(snap.size(keys::kPolicyLogits), static_cast<std::size_t>(1858));
    CHECK_EQ(snap.size(keys::kValueWdl), static_cast<std::size_t>(3));

    const float* wdl = snap.data(keys::kValueWdl);
    const float sum = wdl[0] + wdl[1] + wdl[2];
    CHECK(std::isfinite(sum));
    CHECK(std::fabs(sum - 1.0f) < 1e-4f);
}

TEST(bt4_real_loader_rejects_dense_length_mismatch) {
    cnnv::nn::lc0_bt4::Network net;
    const std::string path = writeMalformedBt4RealModel();
    bool rejected = false;
    try {
        net.load(path);
    } catch (const std::exception& e) {
        rejected = std::string(e.what()).find("length mismatch") !=
                   std::string::npos;
    }
    std::remove(path.c_str());
    CHECK(rejected);
}

TEST(bt4_real_loader_rejects_oversized_float_array_before_allocating) {
    cnnv::nn::lc0_bt4::Network net;
    const std::string path = writeMalformedBt4RealModel(/*oversizedArray=*/true);
    bool rejected = false;
    try {
        net.load(path);
    } catch (const std::exception& e) {
        rejected = std::string(e.what()).find("exceeds file payload") !=
                   std::string::npos;
    }
    std::remove(path.c_str());
    CHECK(rejected);
}

TEST(bt4_real_loader_rejects_huge_encoder_count_before_reserving) {
    cnnv::nn::lc0_bt4::Network net;
    const std::string path = writeBt4RealModelWithHugeEncoderCount();
    bool rejected = false;
    try {
        net.load(path);
    } catch (const std::exception& e) {
        rejected = std::string(e.what()).find("encoder block count too large") !=
                   std::string::npos;
    }
    std::remove(path.c_str());
    CHECK(rejected);
}
