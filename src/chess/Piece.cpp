#include "chess/Piece.h"

namespace cnnv::chess {

char pieceToFenChar(Piece p) noexcept {
    if (p.type == PieceType::None) return '.';
    char c = '.';
    switch (p.type) {
        case PieceType::Pawn:   c = 'P'; break;
        case PieceType::Knight: c = 'N'; break;
        case PieceType::Bishop: c = 'B'; break;
        case PieceType::Rook:   c = 'R'; break;
        case PieceType::Queen:  c = 'Q'; break;
        case PieceType::King:   c = 'K'; break;
        case PieceType::None:   return '.';
    }
    return p.color == Color::White ? c : static_cast<char>(c + ('a' - 'A'));
}

Piece pieceFromFenChar(char c) noexcept {
    Color col = (c >= 'a' && c <= 'z') ? Color::Black : Color::White;
    char up = (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
    PieceType t;
    switch (up) {
        case 'P': t = PieceType::Pawn;   break;
        case 'N': t = PieceType::Knight; break;
        case 'B': t = PieceType::Bishop; break;
        case 'R': t = PieceType::Rook;   break;
        case 'Q': t = PieceType::Queen;  break;
        case 'K': t = PieceType::King;   break;
        default:  return kNoPiece;
    }
    return Piece{col, t};
}

}  // namespace cnnv::chess
