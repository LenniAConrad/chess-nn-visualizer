#include "nn/lc0_cnn/PolicyEncoder.h"

#include "chess/Bitboard.h"

#include <array>
#include <cstdlib>

namespace cnnv::nn::lc0_cnn {

namespace {

using cnnv::chess::Move;

// Knight deltas in (file, rank), clockwise from NNE — must match the order the
// network was trained with.
constexpr std::array<std::array<int, 2>, 8> kKnightDeltas = {{
    {{1, 2}}, {{2, 1}}, {{2, -1}}, {{1, -2}},
    {{-1, -2}}, {{-2, -1}}, {{-2, 1}}, {{-1, 2}},
}};

// Underpromotion directions: forward, forward-left, forward-right.
constexpr std::array<std::array<int, 2>, 3> kUnderpromoDirs = {{
    {{0, 1}}, {{-1, 1}}, {{1, 1}},
}};

int knightIndex(int dFile, int dRank) {
    for (int i = 0; i < 8; ++i) {
        if (kKnightDeltas[static_cast<std::size_t>(i)][0] == dFile &&
            kKnightDeltas[static_cast<std::size_t>(i)][1] == dRank) {
            return i;
        }
    }
    return -1;
}

int planeIndex(int dirIndex, int distance) {
    if (dirIndex < 0 || distance < 1 || distance > 7) return -1;
    return dirIndex * 7 + (distance - 1);
}

int diagonalDirectionIndex(int dFile, int dRank) {
    if (dFile > 0 && dRank > 0) return 4;
    if (dFile < 0 && dRank > 0) return 5;
    if (dFile > 0 && dRank < 0) return 6;
    if (dFile < 0 && dRank < 0) return 7;
    return -1;
}

// Direction order (queen-like): N, S, E, W, NE, NW, SE, SW.
int slidePlaneIndex(int dFile, int dRank) {
    if (dFile == 0 && dRank != 0) {
        return planeIndex(dRank > 0 ? 0 : 1, std::abs(dRank));
    }
    if (dRank == 0 && dFile != 0) {
        return planeIndex(dFile > 0 ? 2 : 3, std::abs(dFile));
    }
    if (std::abs(dFile) == std::abs(dRank) && dFile != 0) {
        return planeIndex(diagonalDirectionIndex(dFile, dRank), std::abs(dFile));
    }
    return -1;
}

int underpromoPieceIndex(Move::Promotion p) {
    switch (p) {
        case Move::Promotion::Knight: return 0;
        case Move::Promotion::Bishop: return 1;
        case Move::Promotion::Rook:   return 2;
        default:                      return -1;
    }
}

int underpromoDirectionIndex(int dFile, int dRank) {
    for (int i = 0; i < 3; ++i) {
        if (kUnderpromoDirs[static_cast<std::size_t>(i)][0] == dFile &&
            kUnderpromoDirs[static_cast<std::size_t>(i)][1] == dRank) {
            return i;
        }
    }
    return -1;
}

}  // namespace

int rawPolicyIndex(const cnnv::chess::Position& pos, cnnv::chess::Move move) {
    if (move.isNone()) return -1;

    const int fromIdx = cnnv::chess::squareIndex(move.from());
    const int toIdx = cnnv::chess::squareIndex(move.to());
    int fromFile = fromIdx & 7;
    int fromRank = fromIdx >> 3;
    int toFile = toIdx & 7;
    int toRank = toIdx >> 3;

    if (pos.sideToMove() == cnnv::chess::Color::Black) {
        fromRank = 7 - fromRank;
        toRank = 7 - toRank;
    }

    const int dFile = toFile - fromFile;
    const int dRank = toRank - fromRank;
    const int fromSquare = (fromRank << 3) | fromFile;

    const Move::Promotion promo = move.promotion();
    if (promo != Move::Promotion::None && promo != Move::Promotion::Queen) {
        const int promoIndex = underpromoPieceIndex(promo);
        const int dirIndex = underpromoDirectionIndex(dFile, dRank);
        if (promoIndex < 0 || dirIndex < 0) return -1;
        const int plane = 64 + promoIndex * 3 + dirIndex;
        return plane * 64 + fromSquare;
    }

    const int kn = knightIndex(dFile, dRank);
    if (kn >= 0) return (56 + kn) * 64 + fromSquare;

    const int slide = slidePlaneIndex(dFile, dRank);
    if (slide >= 0) return slide * 64 + fromSquare;

    return -1;
}

}  // namespace cnnv::nn::lc0_cnn
