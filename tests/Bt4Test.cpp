#include "TestMain.h"

#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

namespace {

std::string bt4TempPath() {
    static int counter = 0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/cnnv_bt4_visual_%d.bin",
                  counter++);
    return buf;
}

void writeU32(std::ofstream& os, std::uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string writeBt4VisualModel() {
    std::string path = bt4TempPath();
    std::ofstream os(path, std::ios::binary);
    os.write("BT4V", 4);
    writeU32(os, 1);          // version
    writeU32(os, 112);        // input channels
    writeU32(os, 64);         // tokens
    writeU32(os, 176);        // token width
    writeU32(os, 64);         // model dim
    writeU32(os, 15);         // blocks
    writeU32(os, 8);          // heads
    writeU32(os, 1858);       // policy size
    writeU32(os, 0x43504E56); // deterministic visual seed
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
    CHECK_EQ(snap.size(keys::blockOutputKey(Net::kBlocks - 1)),
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
