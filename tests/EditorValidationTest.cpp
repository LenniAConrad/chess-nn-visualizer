#include "TestMain.h"
#include "chess/Fen.h"
#include "chess/PositionEditor.h"

using namespace cnnv::chess;

namespace {

bool hasIssue(const EditorValidation& v, const char* needle) {
    for (const auto& s : v.issues) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

TEST(editor_builds_startpos) {
    PositionEditor ed;
    ed.resetToStartpos();
    Position p = ed.build();
    auto opt = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(opt.has_value());
    CHECK_EQ(Fen::format(p), Fen::format(*opt));
}

TEST(editor_validation_passes_on_legal_endgame) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::D1, Piece{Color::White, PieceType::Queen});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.setSideToMove(Color::White);
    ed.setHalfmoveClock(0);
    ed.setFullmoveNumber(1);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(v.legal);
    CHECK_EQ(v.issues.size(), std::size_t{0});
}

TEST(editor_validation_rejects_two_white_kings) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E2, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "White must have exactly one king"));
}

TEST(editor_validation_rejects_no_black_king) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "Black must have exactly one king"));
}

TEST(editor_validation_rejects_pawn_on_back_rank) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.placePiece(Square::A1, Piece{Color::White, PieceType::Pawn});
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "rank 1 or rank 8"));
}

TEST(editor_validation_rejects_side_not_to_move_in_check) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.placePiece(Square::E2, Piece{Color::White, PieceType::Rook});
    ed.setSideToMove(Color::White);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "Black is in check"));
}

TEST(editor_validation_rejects_castling_without_pieces_on_home_squares) {
    PositionEditor ed;
    ed.placePiece(Square::A1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.setCastlingRight(Color::White, PositionEditor::CastleSide::Kingside, true);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "White king-side"));
}

TEST(editor_validation_rejects_inconsistent_ep_square) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.setSideToMove(Color::White);
    // ep target on rank 3 with white to move would mean white just played —
    // but it's white's turn, contradiction.
    ed.setEnPassantSquare(Square::E3);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "En-passant target square"));
}

TEST(editor_validation_rejects_ep_with_no_pawn) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    ed.setSideToMove(Color::White);
    // ep on E6 implies black just pushed a pawn from E7 to E5 — but no pawn
    // is present on E5.
    ed.setEnPassantSquare(Square::E6);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(!v.legal);
    CHECK(hasIssue(v, "supposedly-just-moved pawn"));
}

TEST(editor_validation_accepts_valid_ep_position) {
    PositionEditor ed;
    ed.placePiece(Square::E1, Piece{Color::White, PieceType::King});
    ed.placePiece(Square::E8, Piece{Color::Black, PieceType::King});
    // Black just played e7-e5; ep target = E6, side to move = White, pawn on E5.
    ed.placePiece(Square::E5, Piece{Color::Black, PieceType::Pawn});
    ed.setSideToMove(Color::White);
    ed.setEnPassantSquare(Square::E6);
    EditorValidation v = validateForEditor(ed.build());
    CHECK(v.legal);
}
