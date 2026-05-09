#include "TestMain.h"
#include "chess/Fen.h"

using namespace cnnv::chess;

TEST(fen_round_trip_startpos) {
    auto opt = Fen::parse(Fen::kStartpos);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(Fen::kStartpos));
}

TEST(fen_round_trip_kiwipete) {
    const char* fen =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(fen));
}

TEST(fen_round_trip_endgame) {
    const char* fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(fen));
}

TEST(fen_round_trip_en_passant) {
    const char* fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(fen));
}

TEST(fen_round_trip_promotion_ready) {
    const char* fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(fen));
}

TEST(fen_round_trip_partial_castling) {
    const char* fen = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(*opt), std::string(fen));
}

TEST(fen_no_castling_dash) {
    const char* fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
    auto opt = Fen::parse(fen);
    CHECK(opt.has_value());
    CHECK_EQ(opt->castlingRights(), 0);
}

TEST(fen_invalid_returns_nullopt) {
    CHECK(!Fen::parse("not a fen").has_value());
    CHECK(!Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/8 w KQkq - 0 1").has_value());
    CHECK(!Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/8/PPPPPPPP x KQkq - 0 1").has_value());
}
