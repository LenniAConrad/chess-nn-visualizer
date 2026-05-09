#include "TestMain.h"
#include "chess/Fen.h"
#include "chess/Position.h"

using namespace cnnv::chess;

TEST(position_startpos_state) {
    Position p;
    p.setStartpos();
    CHECK_EQ(p.sideToMove(), Color::White);
    CHECK_EQ(p.castlingRights(), AnyCastling);
    CHECK_EQ(p.epSquare(), Square::None);
    CHECK_EQ(p.halfmoveClock(), 0);
    CHECK_EQ(p.fullmoveNumber(), 1);
    CHECK_EQ(p.pieceAt(Square::E1).type, PieceType::King);
    CHECK_EQ(p.pieceAt(Square::E1).color, Color::White);
    CHECK_EQ(p.pieceAt(Square::E8).type, PieceType::King);
    CHECK_EQ(p.pieceAt(Square::E8).color, Color::Black);
}

TEST(position_make_unmake_returns_to_startpos) {
    Position p;
    p.setStartpos();
    std::string before = Fen::format(p);
    p.make(Move(Square::E2, Square::E4));
    p.make(Move(Square::E7, Square::E5));
    p.make(Move(Square::G1, Square::F3));
    p.make(Move(Square::B8, Square::C6));
    p.unmake();
    p.unmake();
    p.unmake();
    p.unmake();
    CHECK_EQ(Fen::format(p), before);
}

TEST(position_castling_rook_actually_moves) {
    auto opt = Fen::parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    p.make(Move(Square::E1, Square::G1));  // O-O
    CHECK_EQ(p.pieceAt(Square::G1).type, PieceType::King);
    CHECK_EQ(p.pieceAt(Square::F1).type, PieceType::Rook);
    CHECK(p.pieceAt(Square::H1).isNone());
    CHECK(p.pieceAt(Square::E1).isNone());
    p.unmake();
    CHECK_EQ(p.pieceAt(Square::E1).type, PieceType::King);
    CHECK_EQ(p.pieceAt(Square::H1).type, PieceType::Rook);
    CHECK(p.pieceAt(Square::F1).isNone());
    CHECK(p.pieceAt(Square::G1).isNone());
}

TEST(position_in_check_detection) {
    auto opt = Fen::parse("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
    CHECK(opt.has_value());
    CHECK(opt->inCheck());
}

TEST(position_en_passant_capture) {
    auto opt = Fen::parse("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    p.make(Move(Square::E5, Square::D6));
    CHECK_EQ(p.pieceAt(Square::D6).type, PieceType::Pawn);
    CHECK(p.pieceAt(Square::D5).isNone());
    p.unmake();
    CHECK_EQ(p.pieceAt(Square::D5).type, PieceType::Pawn);
    CHECK_EQ(p.pieceAt(Square::E5).type, PieceType::Pawn);
}

TEST(position_promotion_round_trip) {
    auto opt = Fen::parse("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    CHECK(opt.has_value());
    Position p = *opt;
    p.make(Move(Square::A7, Square::A8, Move::Promotion::Queen));
    CHECK_EQ(p.pieceAt(Square::A8).type, PieceType::Queen);
    CHECK_EQ(p.pieceAt(Square::A8).color, Color::White);
    p.unmake();
    CHECK_EQ(p.pieceAt(Square::A7).type, PieceType::Pawn);
    CHECK(p.pieceAt(Square::A8).isNone());
}
