#include "TestMain.h"
#include "chess/Fen.h"
#include "chess/San.h"

using namespace cnnv::chess;

TEST(san_pawn_push) {
    Position p;
    p.setStartpos();
    Move m(Square::E2, Square::E4);
    CHECK_EQ(San::toSan(p, m), std::string("e4"));
    CHECK_EQ(San::parse(p, "e4"), m);
}

TEST(san_knight_move_no_disambig) {
    Position p;
    p.setStartpos();
    Move m(Square::G1, Square::F3);
    CHECK_EQ(San::toSan(p, m), std::string("Nf3"));
    CHECK_EQ(San::parse(p, "Nf3"), m);
}

TEST(san_knight_disambig_file) {
    auto opt = Fen::parse("4k3/8/8/8/8/N1N5/8/4K3 w - - 0 1");
    CHECK(opt.has_value());
    // Both knights can reach b5; disambiguation by file.
    Move ma(Square::A3, Square::B5);
    Move mc(Square::C3, Square::B5);
    CHECK_EQ(San::toSan(*opt, ma), std::string("Nab5"));
    CHECK_EQ(San::toSan(*opt, mc), std::string("Ncb5"));
    CHECK_EQ(San::parse(*opt, "Nab5"), ma);
    CHECK_EQ(San::parse(*opt, "Ncb5"), mc);
}

TEST(san_capture_with_x) {
    auto opt = Fen::parse(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    CHECK(opt.has_value());
    Move m(Square::E4, Square::D5);
    CHECK_EQ(San::toSan(*opt, m), std::string("exd5"));
    CHECK_EQ(San::parse(*opt, "exd5"), m);
}

TEST(san_castling) {
    auto opt = Fen::parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    CHECK(opt.has_value());
    Move ks(Square::E1, Square::G1);
    Move qs(Square::E1, Square::C1);
    CHECK_EQ(San::toSan(*opt, ks), std::string("O-O"));
    CHECK_EQ(San::toSan(*opt, qs), std::string("O-O-O"));
    CHECK_EQ(San::parse(*opt, "O-O"),  ks);
    CHECK_EQ(San::parse(*opt, "O-O-O"), qs);
    // Lowercase 'o' / numeric variants should be tolerated.
    CHECK_EQ(San::parse(*opt, "0-0"),   ks);
    CHECK_EQ(San::parse(*opt, "0-0-0"), qs);
}

TEST(san_promotion) {
    auto opt = Fen::parse("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    CHECK(opt.has_value());
    Move mq(Square::A7, Square::A8, Move::Promotion::Queen);
    CHECK_EQ(San::toSan(*opt, mq), std::string("a8=Q+"));
    CHECK_EQ(San::parse(*opt, "a8=Q"), mq);
    CHECK_EQ(San::parse(*opt, "a8=Q+"), mq);
}

TEST(san_check_and_mate_suffix) {
    // Fool's mate: 1.f3 e5 2.g4 Qh4#
    Position p;
    p.setStartpos();
    p.make(San::parse(p, "f3"));
    p.make(San::parse(p, "e5"));
    p.make(San::parse(p, "g4"));
    Move mate = San::parse(p, "Qh4");  // accepts unannotated mate
    CHECK(!mate.isNone());
    CHECK_EQ(San::toSan(p, mate), std::string("Qh4#"));
}

TEST(san_round_trip_fools_mate_full_game) {
    Position p;
    p.setStartpos();
    const char* moves[] = {"f3", "e5", "g4", "Qh4#"};
    const char* expected[] = {"f3", "e5", "g4", "Qh4#"};
    for (std::size_t i = 0; i < sizeof(moves) / sizeof(moves[0]); ++i) {
        Move m = San::parse(p, moves[i]);
        CHECK(!m.isNone());
        CHECK_EQ(San::toSan(p, m), std::string(expected[i]));
        p.make(m);
    }
    CHECK(p.isCheckmate());
}

TEST(position_status_helpers) {
    // Fool's-mate end position: "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    auto opt = Fen::parse(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    CHECK(opt.has_value());
    CHECK(opt->isCheckmate());
    CHECK(!opt->isStalemate());
    CHECK(!opt->isInsufficientMaterial());
    CHECK(!opt->isFiftyMoveDraw());
}

TEST(position_stalemate_detection) {
    auto opt = Fen::parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    CHECK(opt.has_value());
    CHECK(opt->isStalemate());
    CHECK(!opt->isCheckmate());
}

TEST(position_insufficient_material) {
    // K vs K
    {
        auto opt = Fen::parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        CHECK(opt.has_value());
        CHECK(opt->isInsufficientMaterial());
    }
    // K+B vs K
    {
        auto opt = Fen::parse("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
        CHECK(opt.has_value());
        CHECK(opt->isInsufficientMaterial());
    }
    // K+R vs K — sufficient
    {
        auto opt = Fen::parse("4k3/8/8/8/8/8/8/4KR2 w - - 0 1");
        CHECK(opt.has_value());
        CHECK(!opt->isInsufficientMaterial());
    }
}

TEST(position_zobrist_make_unmake_symmetric) {
    Position p;
    p.setStartpos();
    auto h0 = p.hash();
    p.make(Move(Square::E2, Square::E4));
    auto h1 = p.hash();
    p.make(Move(Square::E7, Square::E5));
    auto h2 = p.hash();
    p.unmake();
    CHECK_EQ(p.hash(), h1);
    p.unmake();
    CHECK_EQ(p.hash(), h0);
    CHECK(h0 != h1);
    CHECK(h1 != h2);
}

TEST(position_zobrist_transposition) {
    // Two different knight tours that return to startpos. Hash compares the
    // board + side + castling + EP — not the move counters — so both should
    // collapse onto the starting hash. (We can't use pawn-move transpositions
    // because the EP square depends on which double-push happened last.)
    Position a, b;
    a.setStartpos();
    b.setStartpos();
    auto h0 = a.hash();
    a.make(Move(Square::G1, Square::F3));
    a.make(Move(Square::G8, Square::F6));
    a.make(Move(Square::F3, Square::G1));
    a.make(Move(Square::F6, Square::G8));
    b.make(Move(Square::B1, Square::C3));
    b.make(Move(Square::B8, Square::C6));
    b.make(Move(Square::C3, Square::B1));
    b.make(Move(Square::C6, Square::B8));
    CHECK_EQ(a.hash(), h0);
    CHECK_EQ(b.hash(), h0);
    CHECK_EQ(a.hash(), b.hash());
}
