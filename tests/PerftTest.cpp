#include "TestMain.h"
#include "chess/Fen.h"
#include "chess/MoveGenerator.h"
#include "chess/Perft.h"

using namespace cnnv::chess;

TEST(perft_startpos_depth_1) {
    Position p;
    p.setStartpos();
    CHECK_EQ(perft(p, 1), std::uint64_t{20});
}

TEST(perft_startpos_depth_2) {
    Position p;
    p.setStartpos();
    CHECK_EQ(perft(p, 2), std::uint64_t{400});
}

TEST(perft_startpos_depth_3) {
    Position p;
    p.setStartpos();
    CHECK_EQ(perft(p, 3), std::uint64_t{8902});
}

TEST(perft_startpos_depth_4) {
    Position p;
    p.setStartpos();
    CHECK_EQ(perft(p, 4), std::uint64_t{197281});
}

TEST(perft_kiwipete_depth_1) {
    auto opt = Fen::parse(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 1), std::uint64_t{48});
}

TEST(perft_kiwipete_depth_2) {
    auto opt = Fen::parse(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 2), std::uint64_t{2039});
}

TEST(perft_kiwipete_depth_3) {
    auto opt = Fen::parse(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 3), std::uint64_t{97862});
}

TEST(perft_position3_depth_4) {
    // Endgame position from the standard perft suite. Exercises EP, sliding
    // attacks across a sparse board.
    auto opt = Fen::parse("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 4), std::uint64_t{43238});
}

TEST(perft_position4_depth_3) {
    // Includes promotion + check.
    auto opt = Fen::parse(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 3), std::uint64_t{9467});
}

TEST(perft_position5_depth_3) {
    // Standard perft position 5: tactics + promotions in tight position.
    auto opt = Fen::parse(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 1), std::uint64_t{44});
    CHECK_EQ(perft(p, 2), std::uint64_t{1486});
    CHECK_EQ(perft(p, 3), std::uint64_t{62379});
}

TEST(perft_position6_depth_3) {
    // Standard perft position 6: middlegame with full piece set.
    auto opt = Fen::parse(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    CHECK(opt.has_value());
    Position p = *opt;
    CHECK_EQ(perft(p, 1), std::uint64_t{46});
    CHECK_EQ(perft(p, 2), std::uint64_t{2079});
    CHECK_EQ(perft(p, 3), std::uint64_t{89890});
}
