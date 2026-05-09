#pragma once

/**
 * @file SlidingAttacks.h
 * @brief Header-only attack generation helpers for sliding and leaper pieces.
 */

#include "chess/Bitboard.h"
#include "chess/Piece.h"

namespace cnnv::chess {

/**
 * @brief Classical ray-walk attack generation.
 *
 * Magic bitboards are unnecessary for this visualizer because it evaluates a
 * small number of positions per frame. The header-only implementation favors
 * clarity and deterministic tests over maximum engine throughput.
 */
namespace sliding {

/**
 * @brief Ray direction as file/rank deltas.
 */
struct Direction {
    int df;
    int dr;
};

/**
 * @brief Diagonal ray directions used by bishops and queens.
 */
constexpr Direction kBishopDirs[4] = {
    { 1,  1}, {-1,  1}, { 1, -1}, {-1, -1},
};

/**
 * @brief Orthogonal ray directions used by rooks and queens.
 */
constexpr Direction kRookDirs[4] = {
    { 1,  0}, {-1,  0}, { 0,  1}, { 0, -1},
};

/**
 * @brief Walks rays until the board edge or first occupied square.
 * @param fromIdx Source square index.
 * @param occupied Occupancy bitboard that blocks sliding pieces.
 * @param dirs Direction table.
 * @param nDirs Number of directions in `dirs`.
 * @return Attack bitboard from `fromIdx`.
 */
inline Bitboard rayAttacks(int fromIdx, Bitboard occupied, const Direction* dirs, int nDirs) noexcept {
    Bitboard attacks = 0;
    int sf = fromIdx & 7;
    int sr = fromIdx >> 3;
    for (int d = 0; d < nDirs; ++d) {
        int f = sf + dirs[d].df;
        int r = sr + dirs[d].dr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int idx = r * 8 + f;
            attacks |= (Bitboard{1} << idx);
            if (occupied & (Bitboard{1} << idx)) break;
            f += dirs[d].df;
            r += dirs[d].dr;
        }
    }
    return attacks;
}

/**
 * @brief Bishop attacks from a square.
 */
inline Bitboard bishopAttacks(int fromIdx, Bitboard occupied) noexcept {
    return rayAttacks(fromIdx, occupied, kBishopDirs, 4);
}

/**
 * @brief Rook attacks from a square.
 */
inline Bitboard rookAttacks(int fromIdx, Bitboard occupied) noexcept {
    return rayAttacks(fromIdx, occupied, kRookDirs, 4);
}

/**
 * @brief Queen attacks from a square.
 */
inline Bitboard queenAttacks(int fromIdx, Bitboard occupied) noexcept {
    return bishopAttacks(fromIdx, occupied) | rookAttacks(fromIdx, occupied);
}

/**
 * @brief Knight attacks from a square.
 *
 * The lookup table is initialized once on first call.
 */
inline Bitboard knightAttacks(int fromIdx) noexcept {
    static const auto table = [] {
        std::array<Bitboard, 64> t{};
        constexpr int dx[8] = { 1,  2,  2,  1, -1, -2, -2, -1};
        constexpr int dy[8] = { 2,  1, -1, -2, -2, -1,  1,  2};
        for (int s = 0; s < 64; ++s) {
            int sf = s & 7, sr = s >> 3;
            Bitboard a = 0;
            for (int k = 0; k < 8; ++k) {
                int f = sf + dx[k], r = sr + dy[k];
                if (f >= 0 && f < 8 && r >= 0 && r < 8) {
                    a |= (Bitboard{1} << (r * 8 + f));
                }
            }
            t[s] = a;
        }
        return t;
    }();
    return table[fromIdx];
}

/**
 * @brief King single-step attacks from a square.
 *
 * The lookup table is initialized once on first call.
 */
inline Bitboard kingAttacks(int fromIdx) noexcept {
    static const auto table = [] {
        std::array<Bitboard, 64> t{};
        for (int s = 0; s < 64; ++s) {
            int sf = s & 7, sr = s >> 3;
            Bitboard a = 0;
            for (int df = -1; df <= 1; ++df) {
                for (int dr = -1; dr <= 1; ++dr) {
                    if (df == 0 && dr == 0) continue;
                    int f = sf + df, r = sr + dr;
                    if (f >= 0 && f < 8 && r >= 0 && r < 8) {
                        a |= (Bitboard{1} << (r * 8 + f));
                    }
                }
            }
            t[s] = a;
        }
        return t;
    }();
    return table[fromIdx];
}

/**
 * @brief Pawn capture attacks from a square for one color.
 * @param fromIdx Source square index.
 * @param c Pawn color.
 * @return Diagonal capture targets. Push moves are intentionally excluded.
 */
inline Bitboard pawnAttacks(int fromIdx, Color c) noexcept {
    static const auto white = [] {
        std::array<Bitboard, 64> t{};
        for (int s = 0; s < 64; ++s) {
            int sf = s & 7, sr = s >> 3;
            Bitboard a = 0;
            if (sr < 7) {
                if (sf > 0) a |= (Bitboard{1} << (s + 7));
                if (sf < 7) a |= (Bitboard{1} << (s + 9));
            }
            t[s] = a;
        }
        return t;
    }();
    static const auto black = [] {
        std::array<Bitboard, 64> t{};
        for (int s = 0; s < 64; ++s) {
            int sf = s & 7, sr = s >> 3;
            Bitboard a = 0;
            if (sr > 0) {
                if (sf > 0) a |= (Bitboard{1} << (s - 9));
                if (sf < 7) a |= (Bitboard{1} << (s - 7));
            }
            t[s] = a;
        }
        return t;
    }();
    return c == Color::White ? white[fromIdx] : black[fromIdx];
}

}  // namespace sliding
}  // namespace cnnv::chess
