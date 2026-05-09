#include "nn/lc0_cnn/Lc0CnnEncoder.h"

#include "chess/Bitboard.h"
#include "chess/Piece.h"
#include "chess/Position.h"

#include <array>
#include <cstdint>

namespace cnnv::nn::lc0_cnn {

namespace chess = cnnv::chess;

namespace {

// Vertical rank-flip: rank 1 ↔ rank 8. Equivalent to Java's
// Long.reverseBytes (bytes are 8-bit ranks in our convention).
constexpr std::uint64_t mirror_ranks(std::uint64_t bits) noexcept {
    return __builtin_bswap64(bits);
}

void writeBits(float* planes, int channel, std::uint64_t bits) noexcept {
    float* base = planes + channel * Encoder::kSquares;
    while (bits) {
        const int sq = chess::lsb(static_cast<chess::Bitboard>(bits));
        base[sq] = 1.0f;
        bits &= bits - 1;
    }
}

void fillConstant(float* planes, int channel, float value) noexcept {
    float* base = planes + channel * Encoder::kSquares;
    for (int i = 0; i < Encoder::kSquares; ++i) base[i] = value;
}

void fillOnes(float* planes, int channel) noexcept {
    fillConstant(planes, channel, 1.0f);
}

// Fills the 12 per-piece bitboards in LC0 order: WP,WN,WB,WR,WQ,WK,BP,...,BK.
void collectPieceBitboards(const chess::Position& pos,
                           std::array<std::uint64_t, 12>& out) {
    using PT = chess::PieceType;
    static constexpr std::array<PT, 6> kOrder = {PT::Pawn, PT::Knight, PT::Bishop,
                                                 PT::Rook, PT::Queen, PT::King};
    for (int i = 0; i < 6; ++i) {
        out[i]     = pos.pieceBitboard(chess::Color::White, kOrder[i]);
        out[i + 6] = pos.pieceBitboard(chess::Color::Black, kOrder[i]);
    }
}

// Produces bitboards from the side-to-move perspective. White-to-move:
// identity. Black-to-move: swap white↔black AND rank-mirror each board so
// "our" pieces still appear closest to rank 1.
void toSideToMovePerspective(std::array<std::uint64_t, 12>& bits,
                             bool weAreBlack) {
    if (!weAreBlack) return;
    std::array<std::uint64_t, 12> swapped{};
    for (int i = 0; i < 6; ++i) {
        swapped[i]     = mirror_ranks(bits[6 + i]);
        swapped[6 + i] = mirror_ranks(bits[i]);
    }
    bits = swapped;
}

}  // namespace

std::vector<float> Encoder::encode(const chess::Position& pos) {
    std::vector<float> planes(kInputSize, 0.0f);
    const bool weAreBlack = pos.sideToMove() == chess::Color::Black;

    std::array<std::uint64_t, 12> piece{};
    collectPieceBitboards(pos, piece);
    toSideToMovePerspective(piece, weAreBlack);

    // 8 history blocks, repeated.
    for (int h = 0; h < kHistoryBlocks; ++h) {
        const int base = h * kPlanesPerBoard;
        for (int p = 0; p < 12; ++p) {
            writeBits(planes.data(), base + p, piece[p]);
        }
        // Plane base+12 (repetition) stays zero.
    }

    const std::uint8_t cr = pos.castlingRights();
    const bool whiteCastleK = cr & chess::WhiteKing;
    const bool whiteCastleQ = cr & chess::WhiteQueen;
    const bool blackCastleK = cr & chess::BlackKing;
    const bool blackCastleQ = cr & chess::BlackQueen;

    const bool weCastleQ   = weAreBlack ? blackCastleQ : whiteCastleQ;
    const bool weCastleK   = weAreBlack ? blackCastleK : whiteCastleK;
    const bool theyCastleQ = weAreBlack ? whiteCastleQ : blackCastleQ;
    const bool theyCastleK = weAreBlack ? whiteCastleK : blackCastleK;

    if (weCastleQ)   fillOnes(planes.data(), kAuxBase + 0);
    if (weCastleK)   fillOnes(planes.data(), kAuxBase + 1);
    if (theyCastleQ) fillOnes(planes.data(), kAuxBase + 2);
    if (theyCastleK) fillOnes(planes.data(), kAuxBase + 3);

    if (weAreBlack) fillOnes(planes.data(), kAuxBase + 4);

    fillConstant(planes.data(), kAuxBase + 5,
                 static_cast<float>(pos.halfmoveClock()));

    // Plane kAuxBase + 6 stays zero.
    fillOnes(planes.data(), kAuxBase + 7);

    return planes;
}

}  // namespace cnnv::nn::lc0_cnn
