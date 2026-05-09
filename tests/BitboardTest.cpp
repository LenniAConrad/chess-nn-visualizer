#include "TestMain.h"
#include "chess/Bitboard.h"

using namespace cnnv::chess;

TEST(bitboard_square_indices) {
    CHECK_EQ(squareIndex(Square::A1), 0);
    CHECK_EQ(squareIndex(Square::H1), 7);
    CHECK_EQ(squareIndex(Square::A8), 56);
    CHECK_EQ(squareIndex(Square::H8), 63);
}

TEST(bitboard_file_rank_helpers) {
    CHECK_EQ(fileOf(Square::E4), 4);
    CHECK_EQ(rankOf(Square::E4), 3);
    CHECK_EQ(makeSquare(4, 3), Square::E4);
}

TEST(bitboard_set_clear_test) {
    Bitboard bb = 0;
    setBit(bb, squareIndex(Square::E4));
    CHECK(testBit(bb, squareIndex(Square::E4)));
    CHECK(!testBit(bb, squareIndex(Square::E5)));
    clearBit(bb, squareIndex(Square::E4));
    CHECK_EQ(bb, Bitboard{0});
}

TEST(bitboard_popcount) {
    CHECK_EQ(popcount(Bitboard{0xFFFF}), 16);
    CHECK_EQ(popcount(Bitboard{0}), 0);
    CHECK_EQ(popcount(~Bitboard{0}), 64);
}

TEST(bitboard_lsb) {
    CHECK_EQ(lsb(Bitboard{0x80}), 7);
    CHECK_EQ(lsb(Bitboard{1}), 0);
    Bitboard bb = 0b10110;
    CHECK_EQ(popLsb(bb), 1);
    CHECK_EQ(popLsb(bb), 2);
    CHECK_EQ(popLsb(bb), 4);
    CHECK_EQ(bb, Bitboard{0});
}

TEST(bitboard_file_rank_masks) {
    CHECK_EQ(kFileMasks[0], Bitboard{0x0101010101010101ULL});
    CHECK_EQ(kRankMasks[0], Bitboard{0x00000000000000FFULL});
    CHECK_EQ(popcount(kFileMasks[3]), 8);
    CHECK_EQ(popcount(kRankMasks[7]), 8);
}
