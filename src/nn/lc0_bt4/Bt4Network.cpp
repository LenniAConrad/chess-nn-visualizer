#include "nn/lc0_bt4/Bt4Network.h"

#include "chess/MoveGenerator.h"
#include "chess/MoveList.h"
#include "chess/Position.h"
#include "io/WeightFileReader.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4BinLoader.h"
#include "nn/lc0_bt4/Bt4RealForward.h"
#include "nn/lc0_cnn/Lc0CnnEncoder.h"
#include "nn/ops/Activations.h"
#include "nn/ops/Attention.h"
#include "nn/ops/LayerNorm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnnv::nn::lc0_bt4 {

namespace chess = cnnv::chess;
namespace ops = cnnv::nn::ops;

namespace snapshot_keys {
std::string blockLn1Key(int idx) {
    return "bt4.block" + std::to_string(idx) + ".ln1";
}
std::string blockAttentionKey(int idx) {
    return "bt4.block" + std::to_string(idx) + ".attention";
}
std::string blockAttentionOutKey(int idx) {
    return "bt4.block" + std::to_string(idx) + ".attention_out";
}
std::string blockLn2Key(int idx) {
    return "bt4.block" + std::to_string(idx) + ".ln2";
}
std::string blockFfnKey(int idx) {
    return "bt4.block" + std::to_string(idx) + ".ffn";
}
std::string blockOutputKey(int idx) {
    return "bt4.block" + std::to_string(idx) + ".output";
}
}  // namespace snapshot_keys

namespace {

constexpr int kSquare = Network::kTokens;
constexpr int kD = Network::kModelDim;
constexpr int kHeads = Network::kHeads;
constexpr int kHeadDim = Network::kHeadDim;

std::uint32_t mixBits(std::uint32_t x) noexcept {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hashWeight(int a, int b, int c, float scale) noexcept {
    const std::uint32_t h = mixBits(
        static_cast<std::uint32_t>(a * 73856093u) ^
        static_cast<std::uint32_t>(b * 19349663u) ^
        static_cast<std::uint32_t>(c * 83492791u));
    const float u = static_cast<float>(h & 0xffffu) / 32767.5f - 1.0f;
    return u * scale;
}

float mish(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return x * std::exp(x);
    return x * std::tanh(std::log1p(std::exp(x)));
}

std::vector<float> toTokenFeatures(const std::vector<float>& planes) {
    std::vector<float> tokens(
        static_cast<std::size_t>(Network::kTokens) * Network::kTokenWidth,
        0.0f);
    for (int token = 0; token < Network::kTokens; ++token) {
        float* row = tokens.data() +
            static_cast<std::size_t>(token) * Network::kTokenWidth;
        for (int ch = 0; ch < Network::kInputChannels; ++ch) {
            row[ch] = planes[static_cast<std::size_t>(ch) * kSquare + token];
        }
        row[Network::kInputChannels + token] = 1.0f;
    }
    return tokens;
}

std::vector<float> projectInput(const std::vector<float>& tokens) {
    std::vector<float> out(static_cast<std::size_t>(kSquare) * kD, 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = tokens.data() +
            static_cast<std::size_t>(token) * Network::kTokenWidth;
        float* dst = out.data() + static_cast<std::size_t>(token) * kD;
        for (int d = 0; d < kD; ++d) {
            float sum = hashWeight(token, d, 11, 0.18f);
            for (int i = 0; i < Network::kTokenWidth; ++i) {
                if (row[i] == 0.0f) continue;
                sum += row[i] * hashWeight(i, d, 23, 0.11f);
            }
            dst[d] = mish(sum);
        }
    }
    return out;
}

std::size_t qkvIndex(int token, int head, int dim) noexcept {
    return (static_cast<std::size_t>(token) * kHeads + head) * kHeadDim + dim;
}

void buildQkv(const std::vector<float>& input, int block,
              std::vector<float>& q, std::vector<float>& k,
              std::vector<float>& v) {
    q.assign(static_cast<std::size_t>(kSquare) * kD, 0.0f);
    k.assign(q.size(), 0.0f);
    v.assign(q.size(), 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = input.data() + static_cast<std::size_t>(token) * kD;
        for (int head = 0; head < kHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const int flat = head * kHeadDim + d;
                const float a = row[(flat * 13 + block * 7) % kD];
                const float b = row[(flat * 5 + head * 11 + 17) % kD];
                const float c = row[(flat * 19 + token + block) % kD];
                q[qkvIndex(token, head, d)] =
                    std::tanh(0.72f * a + 0.26f * b +
                              hashWeight(flat, block, 31, 0.14f));
                k[qkvIndex(token, head, d)] =
                    std::tanh(0.66f * b - 0.21f * c +
                              hashWeight(flat, block, 37, 0.14f));
                v[qkvIndex(token, head, d)] =
                    mish(0.58f * c + 0.28f * a +
                         hashWeight(flat, block, 41, 0.10f));
            }
        }
    }
}

void projectAttention(const std::vector<float>& attended, int block,
                      std::vector<float>& out) {
    out.assign(attended.size(), 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = attended.data() + static_cast<std::size_t>(token) * kD;
        float* dst = out.data() + static_cast<std::size_t>(token) * kD;
        for (int d = 0; d < kD; ++d) {
            const float a = row[d];
            const float b = row[(d * 3 + block * 5 + 7) % kD];
            const float c = row[(d * 11 + 13) % kD];
            dst[d] = 0.62f * a + 0.25f * b + 0.13f * c;
        }
    }
}

void runFfn(const std::vector<float>& input, int block, std::vector<float>& out) {
    out.assign(input.size(), 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = input.data() + static_cast<std::size_t>(token) * kD;
        float* dst = out.data() + static_cast<std::size_t>(token) * kD;
        for (int d = 0; d < kD; ++d) {
            const float a = row[d];
            const float b = row[(d * 7 + block * 3 + 5) % kD];
            const float c = row[(d * 17 + token + 29) % kD];
            const float h1 = mish(0.78f * a + 0.24f * b +
                                  hashWeight(d, block, 53, 0.10f));
            const float h2 = mish(0.40f * c - 0.18f * a);
            dst[d] = 0.72f * h1 + 0.28f * h2;
        }
    }
}

void addScaled(const std::vector<float>& residual, std::vector<float>& value,
               float scale) {
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] = residual[i] + value[i] * scale;
    }
}

std::vector<float> layerNorm(const std::vector<float>& input) {
    std::vector<float> out(input.size(), 0.0f);
    ops::layer_norm(input.data(), nullptr, nullptr, out.data(), kSquare, kD,
                    1e-6f);
    return out;
}

std::vector<float> tokenRms(const std::vector<float>& tokens) {
    std::vector<float> out(kSquare, 0.0f);
    const float inv = 1.0f / static_cast<float>(kD);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = tokens.data() + static_cast<std::size_t>(token) * kD;
        float sum = 0.0f;
        for (int d = 0; d < kD; ++d) sum += row[d] * row[d];
        out[static_cast<std::size_t>(token)] = std::sqrt(sum * inv);
    }
    return out;
}

std::vector<float> normalizePositive(std::vector<float> values) {
    float maxValue = 0.0f;
    for (float v : values) maxValue = std::max(maxValue, v);
    if (maxValue <= 1e-6f) return values;
    for (float& v : values) v = std::clamp(v / maxValue, 0.0f, 1.0f);
    return values;
}

int tokenToBoardSquare(int token, chess::Color stm) {
    return stm == chess::Color::Black ? (token ^ 56) : token;
}

int boardSquareToToken(int square, chess::Color stm) {
    return stm == chess::Color::Black ? (square ^ 56) : square;
}

float pieceImportance(chess::Piece p, chess::Color stm) {
    if (p.isNone()) return 0.0f;
    if (p.type == chess::PieceType::King) {
        return p.color == stm ? 0.62f : 0.50f;
    }
    const float material =
        static_cast<float>(chess::centipawnValue(p.type)) / 900.0f;
    const float sideBoost = p.color == stm ? 0.10f : 0.0f;
    return 0.22f + 0.44f * material + sideBoost;
}

std::vector<float> finalAttentionReceived(const std::vector<float>& attention) {
    std::vector<float> received(kSquare, 0.0f);
    if (attention.size() != static_cast<std::size_t>(kHeads * kSquare * kSquare)) {
        return received;
    }
    const float inv = 1.0f / static_cast<float>(kHeads * kSquare);
    for (int head = 0; head < kHeads; ++head) {
        for (int from = 0; from < kSquare; ++from) {
            for (int to = 0; to < kSquare; ++to) {
                const std::size_t idx =
                    (static_cast<std::size_t>(head) * kSquare + from) * kSquare + to;
                received[static_cast<std::size_t>(to)] += attention[idx] * inv;
            }
        }
    }
    return normalizePositive(std::move(received));
}

bool queenLikeOrKnight(int from, int to) {
    const int df = (to & 7) - (from & 7);
    const int dr = (to >> 3) - (from >> 3);
    if (df == 0 || dr == 0 || std::abs(df) == std::abs(dr)) return true;
    const int adf = std::abs(df);
    const int adr = std::abs(dr);
    return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
}

std::vector<float> policyLogits(const std::vector<float>& tokens) {
    std::vector<float> out;
    out.reserve(Network::kPolicySize);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kD));
    for (int from = 0; from < kSquare &&
                       static_cast<int>(out.size()) < Network::kPolicySize;
         ++from) {
        const float* a = tokens.data() + static_cast<std::size_t>(from) * kD;
        for (int to = 0; to < kSquare &&
                         static_cast<int>(out.size()) < Network::kPolicySize;
             ++to) {
            if (from == to || !queenLikeOrKnight(from, to)) continue;
            const float* b = tokens.data() + static_cast<std::size_t>(to) * kD;
            float dot = 0.0f;
            for (int d = 0; d < kD; ++d) dot += a[d] * b[d];
            out.push_back(dot * inv);
        }
    }
    while (static_cast<int>(out.size()) < Network::kPolicySize) {
        const int i = static_cast<int>(out.size());
        out.push_back(hashWeight(i, 97, 101, 0.05f));
    }
    return out;
}

std::vector<float> valueLogits(const std::vector<float>& tokens) {
    std::vector<float> pooled(kD, 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const float* row = tokens.data() + static_cast<std::size_t>(token) * kD;
        for (int d = 0; d < kD; ++d) pooled[d] += row[d];
    }
    for (float& v : pooled) v /= static_cast<float>(kSquare);

    std::vector<float> logits(3, 0.0f);
    for (int o = 0; o < 3; ++o) {
        float sum = hashWeight(o, 131, 17, 0.06f);
        for (int d = 0; d < kD; ++d) {
            sum += pooled[d] * hashWeight(d, o, 137, 0.18f);
        }
        logits[static_cast<std::size_t>(o)] = sum;
    }
    return logits;
}

std::vector<float> boardValueSalience(const std::vector<float>& tokens,
                                      const std::vector<float>& finalAttention,
                                      const chess::Position& pos) {
    std::vector<float> valueSignal(kSquare, 0.0f);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kD));
    for (int token = 0; token < kSquare; ++token) {
        const float* row = tokens.data() + static_cast<std::size_t>(token) * kD;
        float sum = 0.0f;
        for (int d = 0; d < kD; ++d) {
            const float winW = hashWeight(d, 0, 137, 0.18f);
            const float lossW = hashWeight(d, 2, 137, 0.18f);
            sum += row[d] * (winW - lossW);
        }
        valueSignal[static_cast<std::size_t>(token)] = std::fabs(sum * inv);
    }
    valueSignal = normalizePositive(std::move(valueSignal));
    std::vector<float> energy = normalizePositive(tokenRms(tokens));
    std::vector<float> attention = finalAttentionReceived(finalAttention);

    std::vector<float> board(kSquare, 0.0f);
    const chess::Color stm = pos.sideToMove();
    for (int token = 0; token < kSquare; ++token) {
        const int sq = tokenToBoardSquare(token, stm);
        const chess::Piece piece = pos.pieceAt(sq);
        const int file = sq & 7;
        const int rank = sq >> 3;
        const float center =
            1.0f - (std::abs(file * 2 - 7) + std::abs(rank * 2 - 7)) / 14.0f;
        board[static_cast<std::size_t>(token)] =
            pieceImportance(piece, stm) + 0.05f * std::clamp(center, 0.0f, 1.0f);
    }

    chess::Position work = pos;
    chess::MoveList moves;
    chess::MoveGenerator::generateLegal(work, moves);
    for (chess::Move move : moves) {
        const int fromSq = chess::squareIndex(move.from());
        const int toSq = chess::squareIndex(move.to());
        const int fromToken = boardSquareToToken(fromSq, stm);
        const int toToken = boardSquareToToken(toSq, stm);
        chess::Piece captured = pos.pieceAt(toSq);
        chess::Piece moving = pos.pieceAt(fromSq);
        if (captured.isNone() &&
            moving.type == chess::PieceType::Pawn &&
            pos.epSquare() != chess::Square::None &&
            toSq == chess::squareIndex(pos.epSquare()) &&
            chess::fileOf(move.from()) != chess::fileOf(move.to())) {
            captured = chess::Piece{chess::other(moving.color), chess::PieceType::Pawn};
        }
        board[static_cast<std::size_t>(fromToken)] += 0.045f;
        board[static_cast<std::size_t>(toToken)] += captured.isNone()
            ? 0.055f
            : 0.20f + static_cast<float>(chess::centipawnValue(captured.type)) / 1800.0f;
        if (move.promotion() != chess::Move::Promotion::None) {
            board[static_cast<std::size_t>(toToken)] += 0.22f;
        }
    }
    board = normalizePositive(std::move(board));

    std::vector<float> out(kSquare, 0.0f);
    for (int token = 0; token < kSquare; ++token) {
        const std::size_t i = static_cast<std::size_t>(token);
        const float model = 0.52f * valueSignal[i] +
                            0.30f * attention[i] +
                            0.18f * energy[i];
        const float boardAware = board[i];
        out[i] = 0.62f * boardAware + 0.38f * model;
        if (boardAware < 0.08f && model < 0.45f) {
            out[i] *= 0.30f;
        }
    }
    return normalizePositive(std::move(out));
}

}  // namespace

void Network::load(const std::string& path) {
    // Peek the 4-byte magic to choose the real (BT4J) vs synthetic (BT4V) path.
    char magic[4] = {0, 0, 0, 0};
    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Bt4Network: cannot open " + path);
        }
        probe.read(magic, 4);
    }
    if (std::memcmp(magic, "BT4J", 4) == 0) {
        std::string error;
        if (!loadBt4Real(path, m_realWeights, error)) {
            m_realLoaded = false;
            throw std::runtime_error("Bt4Network: " + error);
        }
        m_realLoaded = true;
        return;
    }

    m_realLoaded = false;
    cnnv::io::WeightFileReader r;
    r.open(path, "BT4V");

    const std::uint32_t version = r.read_u32();
    if (version != 1) {
        throw std::runtime_error(
            "Bt4Network: unsupported visual model version " +
            std::to_string(version));
    }

    const std::uint32_t inputChannels = r.read_u32();
    const std::uint32_t tokens = r.read_u32();
    const std::uint32_t tokenWidth = r.read_u32();
    const std::uint32_t modelDim = r.read_u32();
    const std::uint32_t blocks = r.read_u32();
    const std::uint32_t heads = r.read_u32();
    const std::uint32_t policySize = r.read_u32();
    (void)r.read_u32();  // deterministic seed reserved for future variants

    if (inputChannels != static_cast<std::uint32_t>(kInputChannels) ||
        tokens != static_cast<std::uint32_t>(kTokens) ||
        tokenWidth != static_cast<std::uint32_t>(kTokenWidth) ||
        modelDim != static_cast<std::uint32_t>(kModelDim) ||
        blocks != static_cast<std::uint32_t>(kSyntheticBlocks) ||
        heads != static_cast<std::uint32_t>(kHeads) ||
        policySize != static_cast<std::uint32_t>(kPolicySize)) {
        throw std::runtime_error(
            "Bt4Network: visual model dimensions do not match this build");
    }

    if (r.tell() != r.size()) {
        throw std::runtime_error("Bt4Network: trailing bytes after payload");
    }
}

void Network::evaluate(const chess::Position& pos,
                       ActivationSnapshot& out) const {
    if (m_realLoaded) {
        evaluateBt4Real(m_realWeights, pos, out);
        return;
    }
    std::vector<float> planes = lc0_cnn::Encoder::encode(pos);
    out.store(snapshot_keys::kInputPlanes, {kInputChannels, 8, 8},
              planes.data());

    std::vector<float> features = toTokenFeatures(planes);
    out.store(snapshot_keys::kTokenFeatures, {kTokens, kTokenWidth},
              features.data());

    std::vector<float> flow = projectInput(features);
    out.store(snapshot_keys::kEmbedding, {kTokens, kModelDim}, flow.data());

    std::vector<float> q, k, v, attended, attentionWeights, attnOut, ffn;
    for (int block = 0; block < kSyntheticBlocks; ++block) {
        std::vector<float> ln1 = layerNorm(flow);
        out.store(snapshot_keys::blockLn1Key(block), {kTokens, kModelDim},
                  ln1.data());

        buildQkv(ln1, block, q, k, v);
        attended.assign(static_cast<std::size_t>(kTokens) * kModelDim, 0.0f);
        attentionWeights.assign(
            static_cast<std::size_t>(kHeads) * kTokens * kTokens, 0.0f);
        ops::scaled_dot_product_attention(
            q.data(), k.data(), v.data(), attended.data(),
            attentionWeights.data(), kTokens, kHeads, kHeadDim);
        out.store(snapshot_keys::blockAttentionKey(block),
                  {kHeads, kTokens, kTokens}, attentionWeights.data());

        projectAttention(attended, block, attnOut);
        out.store(snapshot_keys::blockAttentionOutKey(block),
                  {kTokens, kModelDim}, attnOut.data());
        addScaled(flow, attnOut, 0.72f);

        std::vector<float> ln2 = layerNorm(attnOut);
        out.store(snapshot_keys::blockLn2Key(block), {kTokens, kModelDim},
                  ln2.data());

        runFfn(ln2, block, ffn);
        out.store(snapshot_keys::blockFfnKey(block), {kTokens, kModelDim},
                  ffn.data());
        addScaled(ln2, ffn, 0.58f);
        flow = layerNorm(ffn);

        out.store(snapshot_keys::blockOutputKey(block), {kTokens, kModelDim},
                  flow.data());
    }

    out.store(snapshot_keys::kFinalTokens, {kTokens, kModelDim}, flow.data());

    std::vector<float> rms = tokenRms(flow);
    out.store(snapshot_keys::kFinalTokenMagnitude, {kTokens}, rms.data());

    std::vector<float> salience = boardValueSalience(flow, attentionWeights, pos);
    out.store(snapshot_keys::kBoardSalience, {kTokens}, salience.data());

    std::vector<float> policy = policyLogits(flow);
    out.store(snapshot_keys::kPolicyLogits, {policy.size()}, policy.data());

    std::vector<float> logits = valueLogits(flow);
    out.store(snapshot_keys::kValueLogits, {3}, logits.data());

    std::vector<float> wdl = logits;
    ops::softmax(wdl.data(), wdl.size());
    out.store(snapshot_keys::kValueWdl, {3}, wdl.data());

    const float scalar = wdl[0] - wdl[2];
    out.store(snapshot_keys::kValueScalar, {1}, &scalar);
}

}  // namespace cnnv::nn::lc0_bt4
