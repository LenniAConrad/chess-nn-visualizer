#include "chess/Fen.h"

#include <sstream>

namespace cnnv::chess {

namespace {

bool parseField1(const std::string& s, Position& pos) {
    int rank = 7;
    int file = 0;
    for (char c : s) {
        if (c == '/') {
            if (file != 8) return false;
            --rank;
            file = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += (c - '0');
            if (file > 8) return false;
            continue;
        }
        Piece p = pieceFromFenChar(c);
        if (p.isNone()) return false;
        if (rank < 0 || file >= 8) return false;
        pos.placePiece(makeSquare(file, rank), p);
        ++file;
    }
    return rank == 0 && file == 8;
}

std::string formatField1(const Position& pos) {
    std::string s;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Piece p = pos.pieceAt(makeSquare(file, rank));
            if (p.isNone()) {
                ++empty;
            } else {
                if (empty > 0) {
                    s += static_cast<char>('0' + empty);
                    empty = 0;
                }
                s += pieceToFenChar(p);
            }
        }
        if (empty > 0) s += static_cast<char>('0' + empty);
        if (rank > 0) s += '/';
    }
    return s;
}

bool parseCastling(const std::string& s, Position& pos) {
    std::uint8_t r = NoCastling;
    if (s == "-") {
        pos.setCastlingRights(0);
        return true;
    }
    for (char c : s) {
        switch (c) {
            case 'K': r |= WhiteKing;  break;
            case 'Q': r |= WhiteQueen; break;
            case 'k': r |= BlackKing;  break;
            case 'q': r |= BlackQueen; break;
            default:  return false;
        }
    }
    pos.setCastlingRights(r);
    return true;
}

std::string formatCastling(std::uint8_t r) {
    if (r == 0) return "-";
    std::string s;
    if (r & WhiteKing)  s += 'K';
    if (r & WhiteQueen) s += 'Q';
    if (r & BlackKing)  s += 'k';
    if (r & BlackQueen) s += 'q';
    return s;
}

bool parseEp(const std::string& s, Position& pos) {
    if (s == "-") { pos.setEpSquare(Square::None); return true; }
    if (s.size() != 2) return false;
    int f = s[0] - 'a';
    int r = s[1] - '1';
    if (f < 0 || f > 7 || r < 0 || r > 7) return false;
    pos.setEpSquare(makeSquare(f, r));
    return true;
}

std::string formatEp(Square sq) {
    if (sq == Square::None) return "-";
    std::string s;
    s += static_cast<char>('a' + fileOf(sq));
    s += static_cast<char>('1' + rankOf(sq));
    return s;
}

}  // namespace

std::optional<Position> Fen::parse(const std::string& fen) {
    Position pos;
    pos.clear();

    std::istringstream iss(fen);
    std::string field1, field2, field3, field4, field5, field6;
    if (!(iss >> field1 >> field2 >> field3 >> field4)) return std::nullopt;
    if (!(iss >> field5)) field5 = "0";
    if (!(iss >> field6)) field6 = "1";

    if (!parseField1(field1, pos)) return std::nullopt;

    if (field2 == "w") pos.setSideToMove(Color::White);
    else if (field2 == "b") pos.setSideToMove(Color::Black);
    else return std::nullopt;

    if (!parseCastling(field3, pos)) return std::nullopt;
    if (!parseEp(field4, pos)) return std::nullopt;

    try {
        pos.setHalfmoveClock(std::stoi(field5));
        pos.setFullmoveNumber(std::stoi(field6));
    } catch (...) {
        return std::nullopt;
    }

    pos.anchorHashHistory();
    return pos;
}

std::string Fen::format(const Position& pos) {
    std::string s = formatField1(pos);
    s += pos.sideToMove() == Color::White ? " w " : " b ";
    s += formatCastling(pos.castlingRights());
    s += ' ';
    s += formatEp(pos.epSquare());
    s += ' ';
    s += std::to_string(pos.halfmoveClock());
    s += ' ';
    s += std::to_string(pos.fullmoveNumber());
    return s;
}

}  // namespace cnnv::chess
