#include "TestMain.h"
#include "chess/Bitboard.h"
#include "chess/SlidingAttacks.h"

using namespace cnnv::chess;
using namespace cnnv::chess::sliding;

TEST(sliding_bishop_empty_board) {
    Bitboard atk = bishopAttacks(squareIndex(Square::D4), 0);
    // Bishop on D4 with empty board attacks 13 squares: A1, B2, C3, E5, F6,
    // G7, H8, A7, B6, C5, E3, F2, G1.
    CHECK_EQ(popcount(atk), 13);
    CHECK(testBit(atk, squareIndex(Square::H8)));
    CHECK(testBit(atk, squareIndex(Square::A1)));
    CHECK(testBit(atk, squareIndex(Square::G1)));
    CHECK(testBit(atk, squareIndex(Square::A7)));
}

TEST(sliding_bishop_blocker) {
    // Block on F6 from D4: H8 and G7 should not be reached.
    Bitboard occ = squareBit(Square::F6);
    Bitboard atk = bishopAttacks(squareIndex(Square::D4), occ);
    CHECK(testBit(atk, squareIndex(Square::E5)));
    CHECK(testBit(atk, squareIndex(Square::F6)));   // capture is included
    CHECK(!testBit(atk, squareIndex(Square::G7)));
    CHECK(!testBit(atk, squareIndex(Square::H8)));
}

TEST(sliding_rook_empty_board) {
    Bitboard atk = rookAttacks(squareIndex(Square::A1), 0);
    // Rook on A1 with empty board attacks all of file A (7 squares) and rank
    // 1 (7 squares) = 14.
    CHECK_EQ(popcount(atk), 14);
    CHECK(testBit(atk, squareIndex(Square::A8)));
    CHECK(testBit(atk, squareIndex(Square::H1)));
}

TEST(sliding_knight_corner) {
    Bitboard atk = knightAttacks(squareIndex(Square::A1));
    // Knight on A1 attacks B3 and C2 — 2 squares.
    CHECK_EQ(popcount(atk), 2);
    CHECK(testBit(atk, squareIndex(Square::B3)));
    CHECK(testBit(atk, squareIndex(Square::C2)));
}

TEST(sliding_king_center) {
    Bitboard atk = kingAttacks(squareIndex(Square::E4));
    CHECK_EQ(popcount(atk), 8);
}

TEST(sliding_pawn_attacks_white) {
    Bitboard atk = pawnAttacks(squareIndex(Square::E4), Color::White);
    CHECK(testBit(atk, squareIndex(Square::D5)));
    CHECK(testBit(atk, squareIndex(Square::F5)));
    CHECK_EQ(popcount(atk), 2);
}

TEST(sliding_pawn_attacks_black) {
    Bitboard atk = pawnAttacks(squareIndex(Square::E5), Color::Black);
    CHECK(testBit(atk, squareIndex(Square::D4)));
    CHECK(testBit(atk, squareIndex(Square::F4)));
    CHECK_EQ(popcount(atk), 2);
}
