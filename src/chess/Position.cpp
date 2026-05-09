#include "chess/Position.h"

#include "chess/MoveGenerator.h"
#include "chess/SlidingAttacks.h"
#include "chess/Zobrist.h"

#include <cstdlib>

namespace cnnv::chess {

namespace {

// Castling-rights mask delta when a piece moves from or onto a given square.
// Any movement involving a corner kills the corresponding castling right;
// any king movement kills both rights for that color. The table is OR-applied
// to a "rights to clear" accumulator over both the from and to squares.
constexpr std::array<std::uint8_t, 64> kCastleClear = [] {
    std::array<std::uint8_t, 64> t{};
    t[squareIndex(Square::A1)] = WhiteQueen;
    t[squareIndex(Square::H1)] = WhiteKing;
    t[squareIndex(Square::E1)] = WhiteKing | WhiteQueen;
    t[squareIndex(Square::A8)] = BlackQueen;
    t[squareIndex(Square::H8)] = BlackKing;
    t[squareIndex(Square::E8)] = BlackKing | BlackQueen;
    return t;
}();

}  // namespace

Position::Position() {
    clear();
}

void Position::clear() {
    for (auto& row : m_pieces) row.fill(0);
    m_byColor.fill(0);
    m_occupied = 0;
    m_board.fill(kNoPiece);
    m_sideToMove = Color::White;
    m_castlingRights = NoCastling;
    m_epSquare = Square::None;
    m_halfmoveClock = 0;
    m_fullmoveNumber = 1;
    m_history.clear();
    m_hashHistory.clear();
}

void Position::setStartpos() {
    clear();
    static constexpr PieceType backRank[8] = {
        PieceType::Rook, PieceType::Knight, PieceType::Bishop, PieceType::Queen,
        PieceType::King, PieceType::Bishop, PieceType::Knight, PieceType::Rook,
    };
    for (int f = 0; f < 8; ++f) {
        placePiece(makeSquare(f, 0), Piece{Color::White, backRank[f]});
        placePiece(makeSquare(f, 1), Piece{Color::White, PieceType::Pawn});
        placePiece(makeSquare(f, 6), Piece{Color::Black, PieceType::Pawn});
        placePiece(makeSquare(f, 7), Piece{Color::Black, backRank[f]});
    }
    m_sideToMove = Color::White;
    m_castlingRights = AnyCastling;
    m_epSquare = Square::None;
    m_halfmoveClock = 0;
    m_fullmoveNumber = 1;
    anchorHashHistory();
}

void Position::placePiece(Square sq, Piece p) noexcept {
    int idx = squareIndex(sq);
    if (!m_board[idx].isNone()) removePiece(sq);
    if (p.isNone()) return;
    m_board[idx] = p;
    Bitboard bit = Bitboard{1} << idx;
    m_pieces[static_cast<int>(p.color)][static_cast<int>(p.type) - 1] |= bit;
    m_byColor[static_cast<int>(p.color)] |= bit;
    m_occupied |= bit;
}

void Position::removePiece(Square sq) noexcept {
    int idx = squareIndex(sq);
    Piece p = m_board[idx];
    if (p.isNone()) return;
    Bitboard bit = Bitboard{1} << idx;
    m_pieces[static_cast<int>(p.color)][static_cast<int>(p.type) - 1] &= ~bit;
    m_byColor[static_cast<int>(p.color)] &= ~bit;
    m_occupied &= ~bit;
    m_board[idx] = kNoPiece;
}

void Position::rebuildOccupancy() noexcept {
    m_byColor[0] = 0;
    m_byColor[1] = 0;
    for (int c = 0; c < 2; ++c) {
        for (int t = 0; t < 6; ++t) m_byColor[c] |= m_pieces[c][t];
    }
    m_occupied = m_byColor[0] | m_byColor[1];
}

Square Position::kingSquare(Color c) const noexcept {
    Bitboard bb = m_pieces[static_cast<int>(c)][static_cast<int>(PieceType::King) - 1];
    if (!bb) return Square::None;
    return static_cast<Square>(lsb(bb));
}

bool Position::isSquareAttacked(Square sq, Color bySide) const noexcept {
    int idx = squareIndex(sq);
    int by = static_cast<int>(bySide);

    // Pawn attacks: a pawn of bySide attacks `sq` iff `sq` is in the pawn's
    // attack set. Equivalently, a pawn of the OTHER color on `sq` would be
    // attacked by bySide-pawns sitting on the squares pawnAttacks(sq, !bySide)
    // returns. We use the latter form.
    Bitboard pawnAttackers = sliding::pawnAttacks(idx, other(bySide))
        & m_pieces[by][static_cast<int>(PieceType::Pawn) - 1];
    if (pawnAttackers) return true;

    if (sliding::knightAttacks(idx)
        & m_pieces[by][static_cast<int>(PieceType::Knight) - 1]) {
        return true;
    }
    if (sliding::kingAttacks(idx)
        & m_pieces[by][static_cast<int>(PieceType::King) - 1]) {
        return true;
    }

    Bitboard bishopsQueens =
        m_pieces[by][static_cast<int>(PieceType::Bishop) - 1] |
        m_pieces[by][static_cast<int>(PieceType::Queen) - 1];
    if (sliding::bishopAttacks(idx, m_occupied) & bishopsQueens) return true;

    Bitboard rooksQueens =
        m_pieces[by][static_cast<int>(PieceType::Rook) - 1] |
        m_pieces[by][static_cast<int>(PieceType::Queen) - 1];
    if (sliding::rookAttacks(idx, m_occupied) & rooksQueens) return true;

    return false;
}

bool Position::inCheck() const noexcept {
    Square ks = kingSquare(m_sideToMove);
    if (ks == Square::None) return false;
    return isSquareAttacked(ks, other(m_sideToMove));
}

void Position::make(Move m) {
    StateInfo st;
    st.moveMade = m;
    st.prevCastlingRights = m_castlingRights;
    st.prevEpSquare = m_epSquare;
    st.prevHalfmoveClock = m_halfmoveClock;

    Square from = m.from();
    Square to   = m.to();
    Piece moving = m_board[squareIndex(from)];
    Piece captured = m_board[squareIndex(to)];

    bool isPawnMove = moving.type == PieceType::Pawn;
    bool isCapture = !captured.isNone();

    Square newEp = Square::None;

    // En passant detection: a pawn moves diagonally to the EP square. The
    // captured pawn sits one rank behind `to` from the moving side's view.
    bool isEnPassant = isPawnMove
        && to == m_epSquare
        && fileOf(from) != fileOf(to);
    if (isEnPassant) {
        int dir = (moving.color == Color::White) ? -8 : 8;
        Square capSq = static_cast<Square>(squareIndex(to) + dir);
        st.capturedPiece = m_board[squareIndex(capSq)];
        removePiece(capSq);
        isCapture = true;
    } else {
        st.capturedPiece = captured;
        if (isCapture) removePiece(to);
    }

    // Castling: king moves two squares; also move the corresponding rook.
    bool isCastle = moving.type == PieceType::King
        && std::abs(squareIndex(to) - squareIndex(from)) == 2;
    if (isCastle) {
        bool kingside = squareIndex(to) > squareIndex(from);
        Square rookFrom, rookTo;
        if (moving.color == Color::White) {
            rookFrom = kingside ? Square::H1 : Square::A1;
            rookTo   = kingside ? Square::F1 : Square::D1;
        } else {
            rookFrom = kingside ? Square::H8 : Square::A8;
            rookTo   = kingside ? Square::F8 : Square::D8;
        }
        Piece rook = m_board[squareIndex(rookFrom)];
        removePiece(rookFrom);
        placePiece(rookTo, rook);
    }

    // Pawn double push: set the EP target as the square the pawn passed over.
    if (isPawnMove && std::abs(squareIndex(to) - squareIndex(from)) == 16) {
        newEp = static_cast<Square>((squareIndex(from) + squareIndex(to)) / 2);
    }

    // Move the piece itself, applying promotion if specified.
    removePiece(from);
    Piece landing = moving;
    if (m.promotion() != Move::Promotion::None) {
        switch (m.promotion()) {
            case Move::Promotion::Knight: landing.type = PieceType::Knight; break;
            case Move::Promotion::Bishop: landing.type = PieceType::Bishop; break;
            case Move::Promotion::Rook:   landing.type = PieceType::Rook;   break;
            case Move::Promotion::Queen:  landing.type = PieceType::Queen;  break;
            case Move::Promotion::None: break;
        }
    }
    placePiece(to, landing);

    // Castling-rights update: clear any rights tied to the moved-from or
    // captured-on squares.
    m_castlingRights &= ~(kCastleClear[squareIndex(from)] | kCastleClear[squareIndex(to)]);

    m_epSquare = newEp;

    if (isPawnMove || isCapture) {
        m_halfmoveClock = 0;
    } else {
        ++m_halfmoveClock;
    }

    if (m_sideToMove == Color::Black) ++m_fullmoveNumber;
    m_sideToMove = other(m_sideToMove);

    m_history.push_back(st);
    m_hashHistory.push_back(computeZobristHash(*this));
}

void Position::unmake() {
    if (m_history.empty()) return;
    StateInfo st = m_history.back();
    m_history.pop_back();

    m_sideToMove = other(m_sideToMove);
    if (m_sideToMove == Color::Black) --m_fullmoveNumber;

    Move m = st.moveMade;
    Square from = m.from();
    Square to   = m.to();

    Piece landed = m_board[squareIndex(to)];
    Piece original = landed;

    // Reverse promotion: the piece on `to` may not be a pawn; the `from`
    // piece that moved was a pawn.
    if (m.promotion() != Move::Promotion::None) {
        original.type = PieceType::Pawn;
    }

    removePiece(to);
    placePiece(from, original);

    // Restore captured piece. For en-passant, it goes on the square one rank
    // behind `to` from the moving side's perspective. We can detect en-passant
    // here by: original is a pawn, no promotion, capturedPiece is a pawn of
    // the other color, and `to` was empty before the move (no piece on `to`
    // when we look at the captured square being elsewhere). The simplest
    // discriminator: capturedPiece was a pawn and from-file != to-file and
    // the to-square is currently empty (we just removed our piece) and the
    // captured-piece slot matched neither the to-square's piece nor any
    // previously-on-to. We just store en-passant via a position-derivable
    // check: if the move is a pawn move with file change but capturedPiece
    // sits at a different square, we infer EP from pawn + diagonal + EP-flag
    // on the snapshot's prevEpSquare matching `to`.
    bool isEnPassant = original.type == PieceType::Pawn
        && fileOf(from) != fileOf(to)
        && to == st.prevEpSquare
        && !st.capturedPiece.isNone()
        && st.capturedPiece.type == PieceType::Pawn;
    if (!st.capturedPiece.isNone()) {
        if (isEnPassant) {
            int dir = (original.color == Color::White) ? -8 : 8;
            Square capSq = static_cast<Square>(squareIndex(to) + dir);
            placePiece(capSq, st.capturedPiece);
        } else {
            placePiece(to, st.capturedPiece);
        }
    }

    // Castling rook reversal.
    bool isCastle = original.type == PieceType::King
        && std::abs(squareIndex(to) - squareIndex(from)) == 2;
    if (isCastle) {
        bool kingside = squareIndex(to) > squareIndex(from);
        Square rookFrom, rookTo;
        if (original.color == Color::White) {
            rookFrom = kingside ? Square::H1 : Square::A1;
            rookTo   = kingside ? Square::F1 : Square::D1;
        } else {
            rookFrom = kingside ? Square::H8 : Square::A8;
            rookTo   = kingside ? Square::F8 : Square::D8;
        }
        Piece rook = m_board[squareIndex(rookTo)];
        removePiece(rookTo);
        placePiece(rookFrom, rook);
    }

    m_castlingRights = st.prevCastlingRights;
    m_epSquare = st.prevEpSquare;
    m_halfmoveClock = st.prevHalfmoveClock;
    if (!m_hashHistory.empty()) m_hashHistory.pop_back();
}

void Position::anchorHashHistory() {
    m_hashHistory.clear();
    m_hashHistory.push_back(computeZobristHash(*this));
}

std::uint64_t Position::hash() const noexcept {
    return computeZobristHash(*this);
}

bool Position::hasLegalMoves() const {
    Position copy = *this;
    MoveList moves;
    MoveGenerator::generateLegal(copy, moves);
    return !moves.empty();
}

bool Position::isCheckmate() const {
    return inCheck() && !hasLegalMoves();
}

bool Position::isStalemate() const {
    return !inCheck() && !hasLegalMoves();
}

bool Position::isInsufficientMaterial() const noexcept {
    // Standard FIDE-ish list of trivially-drawn material configurations:
    //   K vs K, K+B vs K, K+N vs K, K+B vs K+B with bishops on same colour.
    // We deliberately stop short of the more elaborate "no possible mate"
    // logic — the visualizer just needs to show "Draw" in clearly drawn
    // endings.
    auto count = [&](Color c, PieceType t) {
        return popcount(pieceBitboard(c, t));
    };
    int wQ = count(Color::White, PieceType::Queen);
    int wR = count(Color::White, PieceType::Rook);
    int wP = count(Color::White, PieceType::Pawn);
    int bQ = count(Color::Black, PieceType::Queen);
    int bR = count(Color::Black, PieceType::Rook);
    int bP = count(Color::Black, PieceType::Pawn);
    if (wQ || wR || wP || bQ || bR || bP) return false;

    int wB = count(Color::White, PieceType::Bishop);
    int wN = count(Color::White, PieceType::Knight);
    int bB = count(Color::Black, PieceType::Bishop);
    int bN = count(Color::Black, PieceType::Knight);
    int total = wB + wN + bB + bN;
    if (total == 0) return true;                 // K vs K
    if (total == 1) return true;                 // K+minor vs K
    if (total == 2 && wB == 1 && bB == 1) {
        // Both bishops on same colour squares?
        Bitboard wb = pieceBitboard(Color::White, PieceType::Bishop);
        Bitboard bb = pieceBitboard(Color::Black, PieceType::Bishop);
        auto squareColor = [](int idx) {
            return ((idx >> 3) + (idx & 7)) & 1;
        };
        return squareColor(lsb(wb)) == squareColor(lsb(bb));
    }
    return false;
}

bool Position::isThreefoldRepetition() const noexcept {
    if (m_hashHistory.empty()) return false;
    std::uint64_t cur = m_hashHistory.back();
    // Only positions reachable from the most recent irreversible move can
    // repeat with the current one, so we only need to scan back that far.
    std::size_t window = static_cast<std::size_t>(m_halfmoveClock);
    if (window > m_hashHistory.size()) window = m_hashHistory.size();
    int matches = 0;
    for (std::size_t i = 0; i < window; ++i) {
        if (m_hashHistory[m_hashHistory.size() - 1 - i] == cur) ++matches;
        if (matches >= 3) return true;
    }
    return false;
}

}  // namespace cnnv::chess
