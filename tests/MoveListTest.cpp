#include "TestMain.h"
#include "chess/MoveList.h"

using namespace cnnv::chess;

TEST(movelist_push_and_iterate) {
    MoveList ml;
    CHECK(ml.empty());
    ml.push(Move(Square::E2, Square::E4));
    ml.push(Move(Square::G1, Square::F3));
    CHECK_EQ(ml.size(), std::size_t{2});
    CHECK_EQ(ml[0].from(), Square::E2);
    CHECK_EQ(ml[1].to(),   Square::F3);

    int n = 0;
    for ([[maybe_unused]] Move m : ml) ++n;
    CHECK_EQ(n, 2);
}

TEST(movelist_clear) {
    MoveList ml;
    ml.push(Move(Square::E2, Square::E4));
    ml.clear();
    CHECK(ml.empty());
}

TEST(movelist_capacity_holds_218) {
    // The known worst case for legal move count in any chess position is 218.
    MoveList ml;
    for (int i = 0; i < 218; ++i) {
        ml.push(Move(Square::A1, Square::A2));
    }
    CHECK_EQ(ml.size(), std::size_t{218});
}

TEST(movelist_swap_erase) {
    MoveList ml;
    ml.push(Move(Square::A1, Square::A2));
    ml.push(Move(Square::B1, Square::B2));
    ml.push(Move(Square::C1, Square::C2));
    ml.swapErase(0);  // removes A1A2; replaces it with C1C2.
    CHECK_EQ(ml.size(), std::size_t{2});
    CHECK_EQ(ml[0].from(), Square::C1);
    CHECK_EQ(ml[1].from(), Square::B1);
}
