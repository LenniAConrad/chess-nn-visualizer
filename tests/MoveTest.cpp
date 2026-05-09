#include "TestMain.h"
#include "chess/Move.h"

using namespace cnnv::chess;

TEST(move_encode_decode) {
    Move m(Square::E2, Square::E4);
    CHECK_EQ(m.from(), Square::E2);
    CHECK_EQ(m.to(),   Square::E4);
    CHECK_EQ(m.promotion(), Move::Promotion::None);
    CHECK(!m.isNone());
}

TEST(move_promotion_round_trip) {
    Move m(Square::A7, Square::A8, Move::Promotion::Queen);
    CHECK_EQ(m.from(), Square::A7);
    CHECK_EQ(m.to(),   Square::A8);
    CHECK_EQ(m.promotion(), Move::Promotion::Queen);
}

TEST(move_uci_round_trip_normal) {
    Move m(Square::E2, Square::E4);
    CHECK_EQ(m.toUci(), std::string("e2e4"));
    Move parsed = Move::parseUci("e2e4");
    CHECK_EQ(parsed, m);
}

TEST(move_uci_round_trip_promotion) {
    Move m(Square::A7, Square::A8, Move::Promotion::Queen);
    CHECK_EQ(m.toUci(), std::string("a7a8q"));
    CHECK_EQ(Move::parseUci("a7a8q"), m);
    CHECK_EQ(Move::parseUci("a7a8r").promotion(), Move::Promotion::Rook);
    CHECK_EQ(Move::parseUci("a7a8b").promotion(), Move::Promotion::Bishop);
    CHECK_EQ(Move::parseUci("a7a8n").promotion(), Move::Promotion::Knight);
}

TEST(move_uci_round_trip_castle_format) {
    Move m(Square::E1, Square::G1);
    CHECK_EQ(m.toUci(), std::string("e1g1"));
    CHECK_EQ(Move::parseUci("e1g1"), m);
}

TEST(move_uci_invalid_returns_none) {
    CHECK(Move::parseUci("").isNone());
    CHECK(Move::parseUci("e2").isNone());
    CHECK(Move::parseUci("z9z9").isNone());
    CHECK(Move::parseUci("e2e4x").isNone());
}

TEST(move_none_sentinel) {
    Move none = Move::none();
    CHECK(none.isNone());
    CHECK_EQ(none.toUci(), std::string());
}
