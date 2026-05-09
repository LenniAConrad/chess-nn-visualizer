#pragma once

/**
 * @file Bitboard.h
 * @brief Core square indexing and 64-bit bitboard helpers for chess logic.
 */

#include <array>
#include <cstdint>

namespace cnnv::chess {

/**
 * @brief 64-bit bitboard with one bit per chess square.
 *
 * Squares are rank-major: A1 is bit 0, H1 is bit 7, A8 is bit 56, and H8 is
 * bit 63. The same convention is used by chess-rtk and most C++ chess engines,
 * which keeps numerical match testing direct.
 */
using Bitboard = std::uint64_t;

/**
 * @brief Strongly typed square identifiers for the 64 board squares.
 *
 * Naming uses file letter plus rank digit. `Square::None` is a sentinel used
 * for "no square" states such as missing en-passant targets.
 */
enum class Square : int {
    A1 =  0, B1, C1, D1, E1, F1, G1, H1,
    A2 =  8, B2, C2, D2, E2, F2, G2, H2,
    A3 = 16, B3, C3, D3, E3, F3, G3, H3,
    A4 = 24, B4, C4, D4, E4, F4, G4, H4,
    A5 = 32, B5, C5, D5, E5, F5, G5, H5,
    A6 = 40, B6, C6, D6, E6, F6, G6, H6,
    A7 = 48, B7, C7, D7, E7, F7, G7, H7,
    A8 = 56, B8, C8, D8, E8, F8, G8, H8,
    None = 64,
};

/**
 * @brief Converts a square enum to its 0..64 integer index.
 * @param sq Square to convert.
 * @return Rank-major square index, or 64 for `Square::None`.
 */
constexpr int squareIndex(Square sq) noexcept {
    return static_cast<int>(sq);
}

/**
 * @brief Builds a square from file and rank coordinates.
 * @param file File coordinate in the range 0..7, where 0 is file A.
 * @param rank Rank coordinate in the range 0..7, where 0 is rank 1.
 * @return Square at the requested coordinate.
 */
constexpr Square makeSquare(int file, int rank) noexcept {
    return static_cast<Square>(rank * 8 + file);
}

/**
 * @brief Extracts the file coordinate from a square.
 * @param sq Square to inspect.
 * @return File coordinate in the range 0..7.
 */
constexpr int fileOf(Square sq) noexcept {
    return squareIndex(sq) & 7;
}

/**
 * @brief Extracts the rank coordinate from a square.
 * @param sq Square to inspect.
 * @return Rank coordinate in the range 0..7.
 */
constexpr int rankOf(Square sq) noexcept {
    return squareIndex(sq) >> 3;
}

/**
 * @brief Bit mask containing every square on file A.
 */
constexpr Bitboard kFileA = 0x0101010101010101ULL;

/**
 * @brief Per-file masks indexed by file coordinate 0..7.
 */
constexpr std::array<Bitboard, 8> kFileMasks = {
    kFileA,
    kFileA << 1,
    kFileA << 2,
    kFileA << 3,
    kFileA << 4,
    kFileA << 5,
    kFileA << 6,
    kFileA << 7,
};

/**
 * @brief Bit mask containing every square on rank 1.
 */
constexpr Bitboard kRank1 = 0x00000000000000FFULL;

/**
 * @brief Per-rank masks indexed by rank coordinate 0..7.
 */
constexpr std::array<Bitboard, 8> kRankMasks = {
    kRank1,
    kRank1 << 8,
    kRank1 << 16,
    kRank1 << 24,
    kRank1 << 32,
    kRank1 << 40,
    kRank1 << 48,
    kRank1 << 56,
};

/**
 * @brief Builds a bitboard with only the supplied square set.
 * @param sq Square to set.
 * @return Bitboard containing a single bit for `sq`.
 */
constexpr Bitboard squareBit(Square sq) noexcept {
    return Bitboard{1} << squareIndex(sq);
}

/**
 * @brief Builds a bitboard with one bit set by raw square index.
 * @param idx Square index in the range 0..63.
 * @return Bitboard containing a single bit for `idx`.
 */
constexpr Bitboard squareBit(int idx) noexcept {
    return Bitboard{1} << idx;
}

/**
 * @brief Sets one square bit in-place.
 * @param bb Bitboard to mutate.
 * @param idx Square index to set.
 */
constexpr void setBit(Bitboard& bb, int idx) noexcept {
    bb |= (Bitboard{1} << idx);
}

/**
 * @brief Clears one square bit in-place.
 * @param bb Bitboard to mutate.
 * @param idx Square index to clear.
 */
constexpr void clearBit(Bitboard& bb, int idx) noexcept {
    bb &= ~(Bitboard{1} << idx);
}

/**
 * @brief Tests whether a square bit is set.
 * @param bb Bitboard to inspect.
 * @param idx Square index to test.
 * @return True when the bit for `idx` is set.
 */
constexpr bool testBit(Bitboard bb, int idx) noexcept {
    return (bb >> idx) & Bitboard{1};
}

/**
 * @brief Counts the number of set bits.
 * @param bb Bitboard to count.
 * @return Population count.
 */
constexpr int popcount(Bitboard bb) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#else
    int n = 0;
    while (bb) { bb &= bb - 1; ++n; }
    return n;
#endif
}

/**
 * @brief Finds the index of the least-significant set bit.
 * @param bb Bitboard to inspect. Must be non-zero.
 * @return Square index of the least-significant set bit.
 *
 * @warning Undefined when `bb == 0`; callers must guard before calling.
 */
constexpr int lsb(Bitboard bb) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb);
#else
    int n = 0;
    while (!(bb & 1)) { bb >>= 1; ++n; }
    return n;
#endif
}

/**
 * @brief Clears and returns the least-significant set bit.
 * @param bb Bitboard to mutate. Must be non-zero.
 * @return Index of the bit that was removed.
 */
constexpr int popLsb(Bitboard& bb) noexcept {
    int i = lsb(bb);
    bb &= bb - 1;
    return i;
}

}  // namespace cnnv::chess
