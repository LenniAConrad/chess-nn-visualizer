#include "TestMain.h"
#include "chess/Piece.h"

using namespace cnnv::chess;

TEST(piece_fen_round_trip_white) {
    const char chars[] = {'K', 'Q', 'R', 'B', 'N', 'P'};
    for (char c : chars) {
        Piece p = pieceFromFenChar(c);
        CHECK_EQ(p.color, Color::White);
        CHECK_EQ(pieceToFenChar(p), c);
    }
}

TEST(piece_fen_round_trip_black) {
    const char chars[] = {'k', 'q', 'r', 'b', 'n', 'p'};
    for (char c : chars) {
        Piece p = pieceFromFenChar(c);
        CHECK_EQ(p.color, Color::Black);
        CHECK_EQ(pieceToFenChar(p), c);
    }
}

TEST(piece_unknown_char_yields_none) {
    CHECK(pieceFromFenChar('.').isNone());
    CHECK(pieceFromFenChar('X').isNone());
    CHECK(pieceFromFenChar(' ').isNone());
}

TEST(piece_index_unique) {
    bool seen[12] = {};
    for (Color c : {Color::White, Color::Black}) {
        for (PieceType t : {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                            PieceType::Rook, PieceType::Queen, PieceType::King}) {
            int idx = pieceIndex(Piece{c, t});
            CHECK(idx >= 0 && idx < 12);
            CHECK(!seen[idx]);
            seen[idx] = true;
        }
    }
    CHECK_EQ(pieceIndex(kNoPiece), 12);
}

TEST(piece_centipawn_values) {
    CHECK_EQ(centipawnValue(PieceType::Pawn),   100);
    CHECK_EQ(centipawnValue(PieceType::Knight), 300);
    CHECK_EQ(centipawnValue(PieceType::Bishop), 300);
    CHECK_EQ(centipawnValue(PieceType::Rook),   500);
    CHECK_EQ(centipawnValue(PieceType::Queen),  900);
    CHECK_EQ(centipawnValue(PieceType::King),   0);
}

TEST(piece_other_color) {
    CHECK_EQ(other(Color::White), Color::Black);
    CHECK_EQ(other(Color::Black), Color::White);
}
