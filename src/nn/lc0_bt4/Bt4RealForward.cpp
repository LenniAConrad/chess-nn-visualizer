#include "nn/lc0_bt4/Bt4RealForward.h"

#include "chess/Bitboard.h"
#include "chess/Piece.h"
#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"
#include "nn/ops/Activations.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cnnv::nn::lc0_bt4 {

namespace chess = cnnv::chess;

namespace {

// ---------------------------------------------------------------------------
// Canonical transform bit flags (Encoder.java).
// ---------------------------------------------------------------------------
constexpr int kFlipTransform = 1;       // file mirror (a<->h)
constexpr int kMirrorTransform = 2;     // rank mirror (1<->8)
constexpr int kTransposeTransform = 4;  // a1-h8 diagonal transpose

constexpr int kPlanesPerBoard = 13;
constexpr int kHistory = 8;
constexpr int kAuxBase = kPlanesPerBoard * kHistory;  // 104
constexpr int kInputChannels = 112;
constexpr int kTokens = 64;
// LC0 routes the first 12 input channels of each token through the PE_DENSE
// preproc dense (Network.PREPROC_CHANNELS_PER_TOKEN).
constexpr int kPreprocChannelsPerToken = 12;

// Internal attention-policy layout (PolicyEncoder.java).
constexpr int kFromToPolicySize = 64 * 64;
constexpr int kInternalPolicyPlanes = 67;
constexpr int kInternalPolicySize = kInternalPolicyPlanes * 64;

// ---------------------------------------------------------------------------
// Bit-twiddling helpers (Encoder.java).
// ---------------------------------------------------------------------------
std::uint64_t mirrorRanks(std::uint64_t bits) { return __builtin_bswap64(bits); }

std::uint64_t flipFiles(std::uint64_t v) {
    v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
    v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return v;
}

std::uint64_t transpose(std::uint64_t v) {
    v = ((v & 0xAA00AA00AA00AA00ULL) >> 9) | ((v & 0x0055005500550055ULL) << 9) |
        (v & 0x55AA55AA55AA55AAULL);
    v = ((v & 0xCCCC0000CCCC0000ULL) >> 18) |
        ((v & 0x0000333300003333ULL) << 18) | (v & 0x3333CCCC3333CCCCULL);
    v = ((v & 0xF0F0F0F000000000ULL) >> 36) |
        ((v & 0x000000000F0F0F0FULL) << 36) | (v & 0x0F0F0F0FF0F0F0F0ULL);
    return v;
}

std::uint64_t transformBits(std::uint64_t bits, int transform) {
    std::uint64_t out = bits;
    if (transform & kFlipTransform) out = flipFiles(out);
    if (transform & kMirrorTransform) out = mirrorRanks(out);
    if (transform & kTransposeTransform) out = transpose(out);
    return out;
}

// ---------------------------------------------------------------------------
// Board encoding (Encoder.java, BT4_CANONICAL_112).
// ---------------------------------------------------------------------------

// LC0 piece order WP,WN,WB,WR,WQ,WK,BP,...,BK; matches Lc0CnnEncoder.
void collectPieceBitboards(const chess::Position& pos,
                           std::array<std::uint64_t, 12>& out) {
    using PT = chess::PieceType;
    static constexpr std::array<PT, 6> order = {PT::Pawn, PT::Knight, PT::Bishop,
                                                PT::Rook, PT::Queen, PT::King};
    for (int i = 0; i < 6; ++i) {
        out[i] = pos.pieceBitboard(chess::Color::White, order[i]);
        out[i + 6] = pos.pieceBitboard(chess::Color::Black, order[i]);
    }
}

// Side-to-move perspective: identity for white, swap+rank-mirror for black so
// "our" pieces sit nearest rank 1 (toSidePerspective in Encoder.java).
void toSidePerspective(std::array<std::uint64_t, 12>& bits, bool weAreBlack) {
    if (!weAreBlack) return;
    std::array<std::uint64_t, 12> swapped{};
    for (int i = 0; i < 6; ++i) {
        swapped[i] = mirrorRanks(bits[6 + i]);
        swapped[i + 6] = mirrorRanks(bits[i]);
    }
    bits = swapped;
}

void addBits(std::vector<float>& planes, int channel, std::uint64_t bits) {
    const int base = channel * kTokens;
    while (bits) {
        const int sq = __builtin_ctzll(bits);
        planes[static_cast<std::size_t>(base + sq)] = 1.0f;
        bits &= bits - 1;
    }
}

void fillConstant(std::vector<float>& planes, int channel, float value) {
    const int base = channel * kTokens;
    for (int i = 0; i < kTokens; ++i) {
        planes[static_cast<std::size_t>(base + i)] = value;
    }
}

// Converts an en-passant square to a perspective plane bit (squareBit in
// Encoder.java). Returns 0 when there is no en-passant target.
std::uint64_t epSquareBit(chess::Square sq, bool weAreBlack) {
    if (sq == chess::Square::None) return 0ULL;
    int file = chess::fileOf(sq);
    int rank = chess::rankOf(sq);
    if (weAreBlack) rank = 7 - rank;
    return std::uint64_t{1} << ((rank << 3) | file);
}

// Chooses LC0's canonical spatial transform (chooseCanonicalTransform). Disabled
// when any castling right exists. `perspective` holds side-to-move bitboards.
int chooseCanonicalTransform(const chess::Position& pos,
                             const std::array<std::uint64_t, 12>& perspective) {
    if (pos.castlingRights() != 0) return 0;

    std::uint64_t ourKing = perspective[5];
    int transform = 0;
    if ((ourKing & 0x0F0F0F0F0F0F0F0FULL) != 0ULL) {
        transform |= kFlipTransform;
        ourKing = flipFiles(ourKing);
    }
    const std::uint64_t pawns = perspective[0] | perspective[6];
    if (pawns != 0ULL) return transform;
    if ((ourKing & 0xFFFFFFFF00000000ULL) != 0ULL) {
        transform |= kMirrorTransform;
        ourKing = mirrorRanks(ourKing);
    }
    if ((ourKing & 0xE0C08000ULL) != 0ULL) {
        return transform | kTransposeTransform;
    }
    if ((ourKing & 0x10204080ULL) == 0ULL) {
        return transform;
    }
    // Tie-break (compareTransposing in Encoder.java).
    std::array<std::uint64_t, 7> tests{};
    std::uint64_t allUnion = 0ULL;
    for (std::uint64_t b : perspective) allUnion |= b;
    std::uint64_t ownUnion = 0ULL;
    for (int i = 0; i < 6; ++i) ownUnion |= perspective[i];
    tests[0] = allUnion;
    tests[1] = ownUnion;
    tests[2] = perspective[5] | perspective[11];
    tests[3] = perspective[4] | perspective[10];
    tests[4] = perspective[3] | perspective[9];
    tests[5] = perspective[1] | perspective[7];
    tests[6] = perspective[2] | perspective[8];
    for (std::uint64_t test : tests) {
        const std::uint64_t value = transformBits(test, transform);
        const std::uint64_t alternative = transpose(value);
        if (value != alternative) {
            return value > alternative ? (transform | kTransposeTransform)
                                       : transform;
        }
    }
    return transform;
}

// Builds the BT4_CANONICAL_112 planes (channel-major [112][64]) and reports the
// canonical transform. Mirrors encodeHistory + writeAuxPlanes in Encoder.java
// for a single-position history (history slots reuse the current board).
std::vector<float> encodeCanonical112(const chess::Position& pos,
                                      int& transformOut, bool canonical) {
    const bool weAreBlack = pos.sideToMove() == chess::Color::Black;

    std::array<std::uint64_t, 12> perspective{};
    collectPieceBitboards(pos, perspective);
    toSidePerspective(perspective, weAreBlack);

    const int transform =
        canonical ? chooseCanonicalTransform(pos, perspective) : 0;
    transformOut = transform;

    std::vector<float> planes(
        static_cast<std::size_t>(kInputChannels) * kTokens, 0.0f);

    // 8 history slots; the single-position history repeats the current board.
    for (int slot = 0; slot < kHistory; ++slot) {
        const int base = slot * kPlanesPerBoard;
        for (int piece = 0; piece < 12; ++piece) {
            addBits(planes, base + piece,
                    transformBits(perspective[static_cast<std::size_t>(piece)],
                                  transform));
        }
        // Plane base+12 (repetition) stays zero.
    }

    // Castling rook-location planes (BT4_CANONICAL_112 has castlingPlane()).
    // The rook squares live on rank 1 (own) / rank 8 (their) in side-to-move
    // perspective. LC0 stores them as rook-location bits; for standard chess we
    // place rook bits on the corner files matching the rights.
    const std::uint8_t cr = pos.castlingRights();
    const bool whiteCastleK = (cr & chess::WhiteKing) != 0;
    const bool whiteCastleQ = (cr & chess::WhiteQueen) != 0;
    const bool blackCastleK = (cr & chess::BlackKing) != 0;
    const bool blackCastleQ = (cr & chess::BlackQueen) != 0;

    const bool ourK = weAreBlack ? blackCastleK : whiteCastleK;
    const bool ourQ = weAreBlack ? blackCastleQ : whiteCastleQ;
    const bool theirK = weAreBlack ? whiteCastleK : blackCastleK;
    const bool theirQ = weAreBlack ? whiteCastleQ : blackCastleQ;

    // Queenside rook plane (AUX_BASE): our a1 (sq 0), their a8 (sq 56).
    {
        std::uint64_t bits = 0ULL;
        if (ourQ) bits |= std::uint64_t{1} << 0;
        if (theirQ) bits |= std::uint64_t{1} << 56;
        addBits(planes, kAuxBase, transformBits(bits, transform));
    }
    // Kingside rook plane (AUX_BASE+1): our h1 (sq 7), their h8 (sq 63).
    {
        std::uint64_t bits = 0ULL;
        if (ourK) bits |= std::uint64_t{1} << 7;
        if (theirK) bits |= std::uint64_t{1} << 63;
        addBits(planes, kAuxBase + 1, transformBits(bits, transform));
    }

    // En-passant target plane (BT4_CANONICAL_112 has enPassantPlane()).
    const std::uint64_t ep = epSquareBit(pos.epSquare(), weAreBlack);
    addBits(planes, kAuxBase + 4, transformBits(ep, transform));

    // Rule-50 as hectoplies, plus the constant edge/ones plane.
    fillConstant(planes, kAuxBase + 5,
                 static_cast<float>(pos.halfmoveClock()) / 100.0f);
    fillConstant(planes, kAuxBase + 7, 1.0f);

    return planes;
}

// Channel-major [channels][tokens] -> token-major [tokens][channels].
std::vector<float> toTokenMajor(const std::vector<float>& planes, int channels,
                                int tokens) {
    std::vector<float> out(planes.size());
    for (int token = 0; token < tokens; ++token) {
        const int outBase = token * channels;
        for (int channel = 0; channel < channels; ++channel) {
            out[static_cast<std::size_t>(outBase + channel)] =
                planes[static_cast<std::size_t>(channel * tokens + token)];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dense / activation / normalization primitives (Network.java).
// ---------------------------------------------------------------------------

float applyActivation(Bt4Activation act, float x) {
    switch (act) {
        case Bt4Activation::None:
            return x;
        case Bt4Activation::Relu:
            return x > 0.0f ? x : 0.0f;
        case Bt4Activation::Mish:
            return ops::mish(x);
        case Bt4Activation::Swish:
            return ops::swish(x);
        case Bt4Activation::Tanh:
            return std::tanh(x);
    }
    return x;
}

void activate(std::vector<float>& values, Bt4Activation act) {
    if (act == Bt4Activation::None) return;
    for (float& v : values) v = applyActivation(act, v);
}

// out[t] = W * x[t] + b over `tokens` rows; W row-major [outDim, inDim].
std::vector<float> denseTokens(const std::vector<float>& input, int tokens,
                               const Dense& layer) {
    const int inDim = layer.inDim;
    const int outDim = layer.outDim;
    std::vector<float> out(static_cast<std::size_t>(tokens) * outDim);
    for (int token = 0; token < tokens; ++token) {
        const int inBase = token * inDim;
        const int outBase = token * outDim;
        for (int o = 0; o < outDim; ++o) {
            float sum = layer.bias[static_cast<std::size_t>(o)];
            const int wBase = o * inDim;
            for (int i = 0; i < inDim; ++i) {
                sum += layer.weights[static_cast<std::size_t>(wBase + i)] *
                       input[static_cast<std::size_t>(inBase + i)];
            }
            out[static_cast<std::size_t>(outBase + o)] = sum;
        }
    }
    return out;
}

// out = W * x + b for a single vector.
std::vector<float> denseVector(const std::vector<float>& input,
                               const Dense& layer) {
    const int inDim = layer.inDim;
    const int outDim = layer.outDim;
    std::vector<float> out(static_cast<std::size_t>(outDim));
    for (int o = 0; o < outDim; ++o) {
        float sum = layer.bias[static_cast<std::size_t>(o)];
        const int wBase = o * inDim;
        for (int i = 0; i < inDim; ++i) {
            sum += layer.weights[static_cast<std::size_t>(wBase + i)] *
                   input[static_cast<std::size_t>(i)];
        }
        out[static_cast<std::size_t>(o)] = sum;
    }
    return out;
}

void layerNormInPlace(std::vector<float>& values, int tokens, int dim,
                      const std::vector<float>& gamma,
                      const std::vector<float>& beta, float eps) {
    for (int token = 0; token < tokens; ++token) {
        const int base = token * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean += values[static_cast<std::size_t>(base + i)];
        }
        mean /= static_cast<float>(dim);
        float variance = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float centered =
                values[static_cast<std::size_t>(base + i)] - mean;
            variance += centered * centered;
        }
        const float invStd = static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(variance) /
                                static_cast<double>(dim) +
                            static_cast<double>(eps)));
        for (int i = 0; i < dim; ++i) {
            const std::size_t idx = static_cast<std::size_t>(base + i);
            values[idx] = (values[idx] - mean) * invStd *
                              gamma[static_cast<std::size_t>(i)] +
                          beta[static_cast<std::size_t>(i)];
        }
    }
}

void scale(std::vector<float>& values, float factor) {
    for (float& v : values) v *= factor;
}

void addInPlace(std::vector<float>& dest, const std::vector<float>& source) {
    for (std::size_t i = 0; i < dest.size(); ++i) dest[i] += source[i];
}

void softmaxInPlace(float* values, int n) {
    float maxV = values[0];
    for (int i = 1; i < n; ++i) {
        if (values[i] > maxV) maxV = values[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = std::exp(values[i] - maxV);
        values[i] = e;
        sum += e;
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; ++i) values[i] /= sum;
    }
}

// LC0 DeepNet residual scale (2N)^-0.25 (encoderAlpha in Network.java).
float encoderAlpha(int encoderLayers) {
    const int n = encoderLayers > 1 ? encoderLayers : 1;
    return static_cast<float>(
        std::pow(2.0 * static_cast<double>(n), -0.25));
}

// ---------------------------------------------------------------------------
// Smolgen (computeSmolgenBias in Network.java).
// ---------------------------------------------------------------------------

// Produces a flat [heads, tokens, tokens] additive attention bias from the
// encoder-block input embeddings.
std::vector<float> computeSmolgenBias(const std::vector<float>& input,
                                      int tokens, int heads,
                                      const Smolgen& smolgen,
                                      const std::vector<float>& smolgenW,
                                      Bt4Activation activation, float eps) {
    std::vector<float> compressed = denseTokens(input, tokens, smolgen.compress);
    std::vector<float> mid = denseVector(compressed, smolgen.dense1);
    activate(mid, activation);
    layerNormInPlace(mid, 1, static_cast<int>(mid.size()), smolgen.ln1Gamma,
                     smolgen.ln1Beta, eps);
    std::vector<float> gen = denseVector(mid, smolgen.dense2);
    activate(gen, activation);
    layerNormInPlace(gen, 1, static_cast<int>(gen.size()), smolgen.ln2Gamma,
                     smolgen.ln2Beta, eps);

    const int perHead = static_cast<int>(gen.size()) / heads;
    const int outSize = tokens * tokens;
    std::vector<float> bias(static_cast<std::size_t>(heads) * outSize);
    for (int h = 0; h < heads; ++h) {
        const int genBase = h * perHead;
        const int outBase = h * outSize;
        for (int o = 0; o < outSize; ++o) {
            float sum = 0.0f;
            const int wRow = o * perHead;
            for (int d = 0; d < perHead; ++d) {
                sum += gen[static_cast<std::size_t>(genBase + d)] *
                       smolgenW[static_cast<std::size_t>(wRow + d)];
            }
            bias[static_cast<std::size_t>(outBase + o)] = sum;
        }
    }
    return bias;
}

// ---------------------------------------------------------------------------
// Attention (attention(...) in Network.java).
// ---------------------------------------------------------------------------

// Runs multi-head self-attention. When `capturePerHead` is non-null the
// post-softmax scores are written there as [heads, tokens, tokens].
std::vector<float> runAttention(const std::vector<float>& input, int tokens,
                                const Attention& attn,
                                const std::vector<float>* smolgenW,
                                Bt4Activation smolgenActivation, float eps,
                                std::vector<float>* capturePerHead) {
    std::vector<float> q = denseTokens(input, tokens, attn.query);
    std::vector<float> k = denseTokens(input, tokens, attn.key);
    std::vector<float> v = denseTokens(input, tokens, attn.value);
    const int dModel = attn.query.outDim;
    const int heads = attn.heads;
    const int depth = dModel / heads;

    std::vector<float> smolgenBias;
    bool hasBias = false;
    if (attn.hasSmolgen && smolgenW != nullptr && !smolgenW->empty()) {
        smolgenBias = computeSmolgenBias(input, tokens, heads, attn.smolgen,
                                         *smolgenW, smolgenActivation, eps);
        hasBias = true;
    }

    if (capturePerHead != nullptr) {
        capturePerHead->assign(
            static_cast<std::size_t>(heads) * tokens * tokens, 0.0f);
    }

    std::vector<float> combined(static_cast<std::size_t>(tokens) * dModel, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(tokens));
    const float invScale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(depth)));

    for (int head = 0; head < heads; ++head) {
        const int headOffset = head * depth;
        const int biasHeadOffset = head * tokens * tokens;
        for (int queryToken = 0; queryToken < tokens; ++queryToken) {
            const int qBase = queryToken * dModel + headOffset;
            const int biasRow = biasHeadOffset + queryToken * tokens;
            for (int keyToken = 0; keyToken < tokens; ++keyToken) {
                const int kBase = keyToken * dModel + headOffset;
                float sum = 0.0f;
                for (int d = 0; d < depth; ++d) {
                    sum += q[static_cast<std::size_t>(qBase + d)] *
                           k[static_cast<std::size_t>(kBase + d)];
                }
                float score = sum * invScale;
                if (hasBias) {
                    score += smolgenBias[static_cast<std::size_t>(biasRow +
                                                                  keyToken)];
                }
                scores[static_cast<std::size_t>(keyToken)] = score;
            }
            softmaxInPlace(scores.data(), tokens);
            if (capturePerHead != nullptr) {
                for (int keyToken = 0; keyToken < tokens; ++keyToken) {
                    (*capturePerHead)[static_cast<std::size_t>(biasRow +
                                                               keyToken)] =
                        scores[static_cast<std::size_t>(keyToken)];
                }
            }
            const int outBase = queryToken * dModel + headOffset;
            for (int d = 0; d < depth; ++d) {
                float sum = 0.0f;
                for (int keyToken = 0; keyToken < tokens; ++keyToken) {
                    const int vBase = keyToken * dModel + headOffset;
                    sum += scores[static_cast<std::size_t>(keyToken)] *
                           v[static_cast<std::size_t>(vBase + d)];
                }
                combined[static_cast<std::size_t>(outBase + d)] = sum;
            }
        }
    }
    return denseTokens(combined, tokens, attn.out);
}

// ---------------------------------------------------------------------------
// Policy gather map (PolicyEncoder.java).
// ---------------------------------------------------------------------------

bool isQueenLikeOrKnight(int from, int to) {
    const int df = (to & 7) - (from & 7);
    const int dr = (to >> 3) - (from >> 3);
    if (df == 0 || dr == 0 || std::abs(df) == std::abs(dr)) return true;
    static constexpr int kKnight[8][2] = {{1, 2},   {2, 1},   {2, -1},
                                          {1, -2},  {-1, -2}, {-2, -1},
                                          {-2, 1},  {-1, 2}};
    for (const auto& d : kKnight) {
        if (d[0] == df && d[1] == dr) return true;
    }
    return false;
}

std::vector<int> buildCompressedByInternal() {
    std::vector<int> map(static_cast<std::size_t>(kInternalPolicySize), -1);
    int next = 0;
    for (int from = 0; from < 64; ++from) {
        for (int to = 0; to < 64; ++to) {
            if (from != to && isQueenLikeOrKnight(from, to)) {
                map[static_cast<std::size_t>(from * 64 + to)] = next++;
            }
        }
    }
    for (int fromFile = 0; fromFile < 8; ++fromFile) {
        const int minTo = fromFile - 1 > 0 ? fromFile - 1 : 0;
        const int maxTo = fromFile + 1 < 7 ? fromFile + 1 : 7;
        for (int toFile = minTo; toFile <= maxTo; ++toFile) {
            for (int promo = 0; promo < 3; ++promo) {
                map[static_cast<std::size_t>(kFromToPolicySize + fromFile * 24 +
                                             toFile * 3 + promo)] = next++;
            }
        }
    }
    return map;
}

}  // namespace

void evaluateBt4Real(const Bt4RealWeights& w, const chess::Position& pos,
                     cnnv::nn::ActivationSnapshot& out) {
    const Arch& arch = w.arch;
    const int tokens = arch.tokens;
    const int embedding = arch.embeddingSize;
    const float eps = arch.layerNormEpsilon;

    // ---- Input encoding ----------------------------------------------------
    // The encoder always builds BT4_CANONICAL_112 planes; canonicalization is
    // gated inside chooseCanonicalTransform (it returns 0 when castling rights
    // exist or the format is non-canonical).
    int transform = 0;
    std::vector<float> planes = encodeCanonical112(pos, transform, arch.isCanonical());
    out.store(snapshot_keys::kInputPlanes,
              {static_cast<std::size_t>(kInputChannels), 8, 8}, planes.data());
    {
        const float tf = static_cast<float>(transform);
        out.store("bt4.input.transform", {1}, &tf);
    }

    std::vector<float> perToken = toTokenMajor(planes, kInputChannels, tokens);
    out.store(snapshot_keys::kTokenFeatures,
              {static_cast<std::size_t>(tokens),
               static_cast<std::size_t>(kInputChannels)},
              perToken.data());

    // ---- Input embedding stack --------------------------------------------
    const InputStack& stack = w.input;
    std::vector<float> flow;
    if (arch.inputEmbedding == "PE_MAP") {
        // Append a 64-way square one-hot to each token.
        const int width = kInputChannels + tokens;
        std::vector<float> mapped(static_cast<std::size_t>(tokens) * width,
                                  0.0f);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < kInputChannels; ++c) {
                mapped[static_cast<std::size_t>(t * width + c)] =
                    perToken[static_cast<std::size_t>(t * kInputChannels + c)];
            }
            mapped[static_cast<std::size_t>(t * width + kInputChannels + t)] =
                1.0f;
        }
        flow = denseTokens(mapped, tokens, stack.embedding);
    } else if (arch.inputEmbedding == "PE_DENSE") {
        // PE_DENSE: route first 12 channels of every token through the preproc
        // dense, then concatenate the per-token result after the base channels.
        const int outPerTok = stack.preproc.outDim / tokens;
        const int outWidth = kInputChannels + outPerTok;
        std::vector<float> sliced(
            static_cast<std::size_t>(tokens) * kPreprocChannelsPerToken, 0.0f);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < kPreprocChannelsPerToken; ++c) {
                sliced[static_cast<std::size_t>(t * kPreprocChannelsPerToken +
                                                c)] =
                    perToken[static_cast<std::size_t>(t * kInputChannels + c)];
            }
        }
        std::vector<float> processed = denseVector(sliced, stack.preproc);
        std::vector<float> concatenated(
            static_cast<std::size_t>(tokens) * outWidth, 0.0f);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < kInputChannels; ++c) {
                concatenated[static_cast<std::size_t>(t * outWidth + c)] =
                    perToken[static_cast<std::size_t>(t * kInputChannels + c)];
            }
            for (int c = 0; c < outPerTok; ++c) {
                concatenated[static_cast<std::size_t>(t * outWidth +
                                                      kInputChannels + c)] =
                    processed[static_cast<std::size_t>(t * outPerTok + c)];
            }
        }
        flow = denseTokens(concatenated, tokens, stack.embedding);
    } else {
        flow = denseTokens(perToken, tokens, stack.embedding);
    }
    activate(flow, arch.defaultActivation);

    if (arch.inputEmbedding == "PE_DENSE" && stack.hasEmbLn) {
        layerNormInPlace(flow, tokens, embedding, stack.embLnGamma,
                         stack.embLnBeta, eps);
    }
    if (stack.hasGates) {
        for (std::size_t i = 0; i < flow.size(); ++i) {
            flow[i] *= stack.multGate[i];
        }
        for (std::size_t i = 0; i < flow.size(); ++i) {
            flow[i] += stack.addGate[i];
        }
    }

    const float alpha = encoderAlpha(arch.encoderLayers);
    if (stack.hasEmbFfn) {
        std::vector<float> ffnIn = denseTokens(flow, tokens, stack.embFfn.dense1);
        activate(ffnIn, arch.ffnActivation);
        std::vector<float> ffnOut =
            denseTokens(ffnIn, tokens, stack.embFfn.dense2);
        if (alpha != 1.0f) scale(ffnOut, alpha);
        addInPlace(ffnOut, flow);
        layerNormInPlace(ffnOut, tokens, embedding, stack.embFfnLnGamma,
                         stack.embFfnLnBeta, eps);
        flow = std::move(ffnOut);
    }

    out.store(snapshot_keys::kEmbedding,
              {static_cast<std::size_t>(tokens),
               static_cast<std::size_t>(embedding)},
              flow.data());

    // ---- Encoder blocks ----------------------------------------------------
    std::vector<float> perHeadAttention;
    for (std::size_t blockIndex = 0; blockIndex < w.encoders.size();
         ++blockIndex) {
        const EncoderBlock& block = w.encoders[blockIndex];
        const int idx = static_cast<int>(blockIndex);
        const int heads = block.attention.heads;

        // LN1 reference snapshot: the view shows a per-block LN1 tensor. The
        // Java body applies LN after the residual add (post-norm); we expose the
        // input that feeds attention as the "ln1" snapshot for the view.
        out.store(snapshot_keys::blockLn1Key(idx),
                  {static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(embedding)},
                  flow.data());

        std::vector<float> attended = runAttention(
            flow, tokens, block.attention,
            arch.hasSmolgen ? &w.smolgenW : nullptr, arch.smolgenActivation,
            eps, &perHeadAttention);
        out.store(snapshot_keys::blockAttentionKey(idx),
                  {static_cast<std::size_t>(heads),
                   static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(tokens)},
                  perHeadAttention.data());

        if (alpha != 1.0f) scale(attended, alpha);
        addInPlace(attended, flow);
        layerNormInPlace(attended, tokens, embedding, block.ln1Gamma,
                         block.ln1Beta, eps);
        out.store(snapshot_keys::blockAttentionOutKey(idx),
                  {static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(embedding)},
                  attended.data());

        std::vector<float> hidden = denseTokens(attended, tokens, block.ffnIn);
        activate(hidden, block.activation);
        std::vector<float> ffnOut = denseTokens(hidden, tokens, block.ffnOut);
        if (alpha != 1.0f) scale(ffnOut, alpha);
        addInPlace(ffnOut, attended);
        layerNormInPlace(ffnOut, tokens, embedding, block.ln2Gamma,
                         block.ln2Beta, eps);

        // LN2 reference snapshot mirrors the post-FFN normalized tensor.
        out.store(snapshot_keys::blockLn2Key(idx),
                  {static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(embedding)},
                  ffnOut.data());
        out.store(snapshot_keys::blockFfnKey(idx),
                  {static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(embedding)},
                  ffnOut.data());

        flow = std::move(ffnOut);
        out.store(snapshot_keys::blockOutputKey(idx),
                  {static_cast<std::size_t>(tokens),
                   static_cast<std::size_t>(embedding)},
                  flow.data());
    }

    out.store(snapshot_keys::kFinalTokens,
              {static_cast<std::size_t>(tokens),
               static_cast<std::size_t>(embedding)},
              flow.data());

    // ---- Derived per-square signals ---------------------------------------
    {
        std::vector<float> rms(static_cast<std::size_t>(tokens), 0.0f);
        for (int t = 0; t < tokens; ++t) {
            double sum = 0.0;
            const int base = t * embedding;
            for (int d = 0; d < embedding; ++d) {
                const double val =
                    static_cast<double>(flow[static_cast<std::size_t>(base + d)]);
                sum += val * val;
            }
            rms[static_cast<std::size_t>(t)] = static_cast<float>(
                std::sqrt(sum / static_cast<double>(embedding)));
        }
        out.store(snapshot_keys::kFinalTokenMagnitude,
                  {static_cast<std::size_t>(tokens)}, rms.data());
    }

    // ---- Value head --------------------------------------------------------
    std::vector<float> valueFlow = denseTokens(flow, tokens, w.valueHead.embedding);
    activate(valueFlow, w.valueHead.activation);
    std::vector<float> valueHidden = denseVector(valueFlow, w.valueHead.fc1);
    activate(valueHidden, w.valueHead.activation);
    std::vector<float> valueLogits = denseVector(valueHidden, w.valueHead.fc2);
    out.store(snapshot_keys::kValueLogits, {3}, valueLogits.data());

    std::vector<float> wdl = valueLogits;
    softmaxInPlace(wdl.data(), static_cast<int>(wdl.size()));
    out.store(snapshot_keys::kValueWdl, {3}, wdl.data());

    const float scalar = wdl[0] - wdl[2];
    out.store(snapshot_keys::kValueScalar, {1}, &scalar);

    // Board value salience: mean absolute final-token activation per square,
    // weighted positive. Provides the view a per-square scalar signal.
    {
        std::vector<float> salience(static_cast<std::size_t>(tokens), 0.0f);
        for (int t = 0; t < tokens; ++t) {
            double sum = 0.0;
            const int base = t * embedding;
            for (int d = 0; d < embedding; ++d) {
                sum += std::fabs(static_cast<double>(
                    flow[static_cast<std::size_t>(base + d)]));
            }
            salience[static_cast<std::size_t>(t)] =
                static_cast<float>(sum / static_cast<double>(embedding));
        }
        out.store(snapshot_keys::kBoardSalience,
                  {static_cast<std::size_t>(tokens)}, salience.data());
    }

    // ---- Policy head -------------------------------------------------------
    const PolicyHead& head = w.policyHead;
    std::vector<float> policyFlow = denseTokens(flow, tokens, head.embedding);
    activate(policyFlow, head.activation);
    const int policyEmbedding = head.embedding.outDim;
    for (const EncoderBlock& block : head.encoders) {
        std::vector<float> attended =
            runAttention(policyFlow, tokens, block.attention, nullptr,
                         arch.smolgenActivation, eps, nullptr);
        if (block.alpha != 1.0f) scale(attended, block.alpha);
        addInPlace(attended, policyFlow);
        layerNormInPlace(attended, tokens, policyEmbedding, block.ln1Gamma,
                         block.ln1Beta, eps);
        std::vector<float> hidden = denseTokens(attended, tokens, block.ffnIn);
        activate(hidden, block.activation);
        std::vector<float> ffnOut = denseTokens(hidden, tokens, block.ffnOut);
        if (block.alpha != 1.0f) scale(ffnOut, block.alpha);
        addInPlace(ffnOut, attended);
        layerNormInPlace(ffnOut, tokens, policyEmbedding, block.ln2Gamma,
                         block.ln2Beta, eps);
        policyFlow = std::move(ffnOut);
    }

    std::vector<float> q = denseTokens(policyFlow, tokens, head.query);
    std::vector<float> k = denseTokens(policyFlow, tokens, head.key);
    const int policyDModel = head.query.outDim;
    const float invScale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(policyDModel)));

    std::vector<float> internal(static_cast<std::size_t>(kInternalPolicySize),
                                0.0f);
    for (int from = 0; from < 64; ++from) {
        const int qBase = from * policyDModel;
        const int outBase = from * 64;
        for (int to = 0; to < 64; ++to) {
            const int kBase = to * policyDModel;
            float sum = 0.0f;
            for (int d = 0; d < policyDModel; ++d) {
                sum += q[static_cast<std::size_t>(qBase + d)] *
                       k[static_cast<std::size_t>(kBase + d)];
            }
            internal[static_cast<std::size_t>(outBase + to)] = sum * invScale;
        }
    }

    // Underpromotion planes (addUnderpromotionLogits in Network.java).
    auto promotionProjection = [&](int token, int output) {
        float sum = 0.0f;
        const int keyBase = token * policyDModel;
        const int wBase = output * policyDModel;
        for (int d = 0; d < policyDModel; ++d) {
            sum += k[static_cast<std::size_t>(keyBase + d)] *
                   head.promotionWeights[static_cast<std::size_t>(wBase + d)];
        }
        return sum;
    };
    for (int fromFile = 0; fromFile < 8; ++fromFile) {
        const int minTo = fromFile - 1 > 0 ? fromFile - 1 : 0;
        const int maxTo = fromFile + 1 < 7 ? fromFile + 1 : 7;
        for (int toFile = minTo; toFile <= maxTo; ++toFile) {
            const int from = 48 + fromFile;
            const int to = 56 + toFile;
            const float base = internal[static_cast<std::size_t>(from * 64 + to)];
            const float queen = promotionProjection(to, 3);
            for (int promo = 0; promo < 3; ++promo) {
                const int internalIndex =
                    kFromToPolicySize + fromFile * 24 + toFile * 3 + promo;
                internal[static_cast<std::size_t>(internalIndex)] =
                    base + queen + promotionProjection(to, promo);
            }
        }
    }

    static const std::vector<int> compressedByInternal =
        buildCompressedByInternal();
    std::vector<float> policy(static_cast<std::size_t>(arch.policySize), 0.0f);
    for (std::size_t i = 0; i < compressedByInternal.size(); ++i) {
        const int mapped = compressedByInternal[i];
        if (mapped >= 0 &&
            mapped < static_cast<int>(policy.size())) {
            policy[static_cast<std::size_t>(mapped)] = internal[i];
        }
    }
    out.store(snapshot_keys::kPolicyLogits, {policy.size()}, policy.data());
}

}  // namespace cnnv::nn::lc0_bt4
