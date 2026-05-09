#include "chess/PositionEditor.h"

#include "chess/Fen.h"

namespace cnnv::chess {

namespace {

std::uint8_t maskFor(Color c, PositionEditor::CastleSide side) noexcept {
    if (c == Color::White) {
        return side == PositionEditor::CastleSide::Kingside  ? WhiteKing  : WhiteQueen;
    }
    return     side == PositionEditor::CastleSide::Kingside  ? BlackKing  : BlackQueen;
}

}  // namespace

PositionEditor::PositionEditor() {
    m_position.clear();
}

PositionEditor::PositionEditor(const Position& seed) : m_position(seed) {}

void PositionEditor::clearBoard() noexcept {
    m_position.clear();
}

void PositionEditor::resetToStartpos() noexcept {
    m_position.setStartpos();
}

void PositionEditor::placePiece(Square sq, Piece p) noexcept {
    m_position.placePiece(sq, p);
}

void PositionEditor::removePiece(Square sq) noexcept {
    m_position.removePiece(sq);
}

void PositionEditor::setCastlingRight(Color c, CastleSide side, bool on) noexcept {
    std::uint8_t mask = maskFor(c, side);
    std::uint8_t cur  = m_position.castlingRights();
    m_position.setCastlingRights(on ? (cur | mask) : (cur & ~mask));
}

bool PositionEditor::castlingRight(Color c, CastleSide side) const noexcept {
    return (m_position.castlingRights() & maskFor(c, side)) != 0;
}

void PositionEditor::setEnPassantSquare(std::optional<Square> sq) noexcept {
    m_position.setEpSquare(sq.value_or(Square::None));
}

std::optional<Square> PositionEditor::enPassantSquare() const noexcept {
    Square s = m_position.epSquare();
    if (s == Square::None) return std::nullopt;
    return s;
}

Position PositionEditor::build() const {
    Position p = m_position;
    p.anchorHashHistory();
    return p;
}

std::string PositionEditor::fen() const {
    return Fen::format(m_position);
}

EditorValidation validateForEditor(const Position& pos) {
    EditorValidation v;

    auto add = [&](std::string s) { v.issues.push_back(std::move(s)); };

    // ---- (1) exactly one king per side ----
    int whiteKings = popcount(pos.pieceBitboard(Color::White, PieceType::King));
    int blackKings = popcount(pos.pieceBitboard(Color::Black, PieceType::King));
    if (whiteKings != 1) {
        add("White must have exactly one king (found " +
            std::to_string(whiteKings) + ").");
    }
    if (blackKings != 1) {
        add("Black must have exactly one king (found " +
            std::to_string(blackKings) + ").");
    }

    // ---- (2) no pawns on rank 1 or rank 8 ----
    Bitboard wp = pos.pieceBitboard(Color::White, PieceType::Pawn);
    Bitboard bp = pos.pieceBitboard(Color::Black, PieceType::Pawn);
    Bitboard pawnsOnEnds = (wp | bp) & (kRankMasks[0] | kRankMasks[7]);
    if (pawnsOnEnds) {
        add("Pawns may not occupy rank 1 or rank 8.");
    }

    // ---- (3) side-not-to-move must not be in check ----
    // If both kings are present we can ask: is the *opposing* king attacked
    // by the side-to-move? If yes, that opposing player would have ended
    // their last turn in check, which is illegal.
    if (whiteKings == 1 && blackKings == 1) {
        Color stm = pos.sideToMove();
        Color opp = other(stm);
        Square oppKing = pos.kingSquare(opp);
        if (oppKing != Square::None &&
            pos.isSquareAttacked(oppKing, stm)) {
            add(stm == Color::White
                ? "Black is in check but it's White to move — illegal."
                : "White is in check but it's Black to move — illegal.");
        }
    }

    // ---- (4) castling rights need king + rook on home squares ----
    auto requireCastle = [&](Color c, PositionEditor::CastleSide side,
                             Square king, Square rook,
                             const char* label) {
        std::uint8_t mask = (c == Color::White
            ? (side == PositionEditor::CastleSide::Kingside ? WhiteKing : WhiteQueen)
            : (side == PositionEditor::CastleSide::Kingside ? BlackKing : BlackQueen));
        if ((pos.castlingRights() & mask) == 0) return;
        Piece k = pos.pieceAt(king);
        Piece r = pos.pieceAt(rook);
        bool kingOk = (k.type == PieceType::King && k.color == c);
        bool rookOk = (r.type == PieceType::Rook && r.color == c);
        if (!kingOk || !rookOk) {
            add(std::string(label) + " castling right is set but the king or rook is not on its home square.");
        }
    };
    requireCastle(Color::White, PositionEditor::CastleSide::Kingside,
                  Square::E1, Square::H1, "White king-side");
    requireCastle(Color::White, PositionEditor::CastleSide::Queenside,
                  Square::E1, Square::A1, "White queen-side");
    requireCastle(Color::Black, PositionEditor::CastleSide::Kingside,
                  Square::E8, Square::H8, "Black king-side");
    requireCastle(Color::Black, PositionEditor::CastleSide::Queenside,
                  Square::E8, Square::A8, "Black queen-side");

    // ---- (5) en-passant square consistency ----
    if (pos.epSquare() != Square::None) {
        int epRank = rankOf(pos.epSquare());
        Color stm = pos.sideToMove();
        // ep target on rank 6 (index 5) means black just played; white to move.
        // ep target on rank 3 (index 2) means white just played; black to move.
        bool rankOk = (stm == Color::White && epRank == 5) ||
                      (stm == Color::Black && epRank == 2);
        if (!rankOk) {
            add("En-passant target square is not consistent with the side to move.");
        } else {
            // The pawn that supposedly just moved sits one rank further along
            // its direction of travel from the ep target.
            int pawnRank = (stm == Color::White) ? 4 : 3;
            Square pawnSq = makeSquare(fileOf(pos.epSquare()), pawnRank);
            Piece p = pos.pieceAt(pawnSq);
            Color expectedColor = (stm == Color::White) ? Color::Black : Color::White;
            if (p.type != PieceType::Pawn || p.color != expectedColor) {
                add("En-passant target is set but the supposedly-just-moved pawn is missing.");
            }
        }
    }

    v.legal = v.issues.empty();
    return v;
}

}  // namespace cnnv::chess
