#include "TestMain.h"

#include "chess/Fen.h"
#include "nn/ActivationSnapshot.h"
#include "nn/nnue/NnueNetwork.h"

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace cnnv::chess;
using namespace cnnv::nn::nnue;

namespace {

std::string firstExisting(std::initializer_list<const char*> paths) {
    for (const char* path : paths) {
        std::ifstream in(path);
        if (in.good()) {
            return path;
        }
    }
    std::ostringstream os;
    os << "none of the candidate paths exist:";
    for (const char* path : paths) {
        os << ' ' << path;
    }
    throw std::runtime_error(os.str());
}

std::string extractStringField(const std::string& line,
                               const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    std::size_t pos = line.find(needle);
    CHECK(pos != std::string::npos);
    pos += needle.size();

    std::string value;
    bool escaped = false;
    for (; pos < line.size(); ++pos) {
        const char ch = line[pos];
        if (escaped) {
            value.push_back(ch);
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else if (ch == '"') {
            return value;
        } else {
            value.push_back(ch);
        }
    }
    throw std::runtime_error("unterminated JSON string field: " + key);
}

float extractNumberField(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    std::size_t pos = line.find(needle);
    CHECK(pos != std::string::npos);
    pos += needle.size();

    const std::size_t end = line.find_first_of(",}", pos);
    CHECK(end != std::string::npos);
    return std::stof(line.substr(pos, end - pos));
}

}  // namespace

TEST(nnue_matches_crtk_reference_values) {
    const std::string modelPath = firstExisting({
        "models/nnue-halfkp-demo.bin",
        "../models/nnue-halfkp-demo.bin",
        "../../models/nnue-halfkp-demo.bin",
    });
    const std::string refsPath = firstExisting({
        "tests/data/nnue_ref.jsonl",
        "../tests/data/nnue_ref.jsonl",
        "../../tests/data/nnue_ref.jsonl",
    });

    Network net;
    net.load(modelPath);

    std::ifstream refs(refsPath);
    CHECK(refs.good());

    int checked = 0;
    std::string line;
    while (std::getline(refs, line)) {
        if (line.empty()) {
            continue;
        }

        const std::string fen = extractStringField(line, "fen");
        const float expected = extractNumberField(line, "centipawns");

        auto pos = Fen::parse(fen);
        CHECK(pos.has_value());

        cnnv::nn::ActivationSnapshot snapshot;
        net.evaluate(*pos, snapshot);
        CHECK(snapshot.has(snapshot_keys::kValueCentipawns));
        const float actual =
            snapshot.data(snapshot_keys::kValueCentipawns)[0];

        if (std::fabs(actual - expected) > 2.0f) {
            std::ostringstream os;
            os << "NNUE mismatch for " << fen << ": expected " << expected
               << " cp, got " << actual << " cp";
            throw cnnv_test::CheckFailure(os.str());
        }
        ++checked;
    }

    CHECK_EQ(checked, 20);
}
