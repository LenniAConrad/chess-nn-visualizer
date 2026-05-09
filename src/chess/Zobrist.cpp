#include "chess/Zobrist.h"

#include <array>

namespace cnnv::chess {

namespace {

// SplitMix64 — a small public-domain PRNG with excellent distribution and a
// trivial implementation. Used here only to fill the Zobrist key tables once
// at first call, with a fixed seed so the keys are stable across runs.
struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

struct Keys {
    std::array<std::array<std::uint64_t, 64>, 12> pieceSquare{};
    std::array<std::uint64_t, 16> castling{};
    std::array<std::uint64_t, 8>  epFile{};
    std::uint64_t blackToMove = 0;
};

const Keys& keys() {
    static const Keys k = [] {
        Keys k;
        SplitMix64 r{0xC0FFEE12345678ULL};
        for (auto& row : k.pieceSquare)
            for (auto& v : row) v = r.next();
        for (auto& v : k.castling) v = r.next();
        for (auto& v : k.epFile) v = r.next();
        k.blackToMove = r.next();
        return k;
    }();
    return k;
}

}  // namespace

std::uint64_t computeZobristHash(const Position& pos) noexcept {
    const Keys& k = keys();
    std::uint64_t h = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = pos.pieceAt(sq);
        if (p.isNone()) continue;
        h ^= k.pieceSquare[pieceIndex(p)][sq];
    }
    h ^= k.castling[pos.castlingRights() & 0xF];
    if (pos.epSquare() != Square::None) {
        h ^= k.epFile[fileOf(pos.epSquare())];
    }
    if (pos.sideToMove() == Color::Black) {
        h ^= k.blackToMove;
    }
    return h;
}

}  // namespace cnnv::chess
