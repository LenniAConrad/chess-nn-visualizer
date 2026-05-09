#include "chess/San.h"

#include "chess/MoveGenerator.h"
#include "chess/MoveList.h"

#include <cctype>
#include <cstdlib>

namespace cnnv::chess {

namespace {

char pieceLetter(PieceType t) {
    switch (t) {
        case PieceType::Knight: return 'N';
        case PieceType::Bishop: return 'B';
        case PieceType::Rook:   return 'R';
        case PieceType::Queen:  return 'Q';
        case PieceType::King:   return 'K';
        default:                return '\0';
    }
}

char promoLetter(Move::Promotion p) {
    switch (p) {
        case Move::Promotion::Knight: return 'N';
        case Move::Promotion::Bishop: return 'B';
        case Move::Promotion::Rook:   return 'R';
        case Move::Promotion::Queen:  return 'Q';
        case Move::Promotion::None:   return '\0';
    }
    return '\0';
}

std::string squareName(Square sq) {
    std::string s;
    s += static_cast<char>('a' + fileOf(sq));
    s += static_cast<char>('1' + rankOf(sq));
    return s;
}

PieceType movingPieceType(const Position& pos, Move m) {
    Piece p = pos.pieceAt(m.from());
    return p.type;
}

bool isCastleMove(const Position& pos, Move m) {
    Piece p = pos.pieceAt(m.from());
    if (p.type != PieceType::King) return false;
    return std::abs(squareIndex(m.to()) - squareIndex(m.from())) == 2;
}

bool isEnPassantMove(const Position& pos, Move m) {
    Piece p = pos.pieceAt(m.from());
    if (p.type != PieceType::Pawn) return false;
    if (fileOf(m.from()) == fileOf(m.to())) return false;
    return pos.pieceAt(m.to()).isNone() && m.to() == pos.epSquare();
}

bool isCaptureMove(const Position& pos, Move m) {
    if (!pos.pieceAt(m.to()).isNone()) return true;
    return isEnPassantMove(pos, m);
}

}  // namespace

std::string San::toSan(const Position& pos, Move m) {
    // Resolve the legal move list once so we can disambiguate and tag check.
    Position copy = pos;
    MoveList legal;
    MoveGenerator::generateLegal(copy, legal);

    std::string out;

    if (isCastleMove(pos, m)) {
        out = (squareIndex(m.to()) > squareIndex(m.from())) ? "O-O" : "O-O-O";
    } else {
        PieceType type = movingPieceType(pos, m);
        bool capture = isCaptureMove(pos, m);

        if (type == PieceType::Pawn) {
            if (capture) {
                out += static_cast<char>('a' + fileOf(m.from()));
                out += 'x';
            }
            out += squareName(m.to());
            if (m.promotion() != Move::Promotion::None) {
                out += '=';
                out += promoLetter(m.promotion());
            }
        } else {
            out += pieceLetter(type);

            // Disambiguation: collect all legal moves where the SAME piece
            // type (and color) lands on the same target square.
            bool needFile = false, needRank = false, needBoth = false;
            int matches = 0;
            for (Move o : legal) {
                if (o == m) continue;
                if (o.to() != m.to()) continue;
                if (movingPieceType(pos, o) != type) continue;
                ++matches;
                if (fileOf(o.from()) == fileOf(m.from())) needRank = true;
                if (rankOf(o.from()) == rankOf(m.from())) needFile = true;
            }
            if (matches > 0 && !needFile && !needRank) needFile = true;
            if (needFile && needRank) { needBoth = true; needFile = false; needRank = false; }

            if (needBoth) {
                out += static_cast<char>('a' + fileOf(m.from()));
                out += static_cast<char>('1' + rankOf(m.from()));
            } else if (needFile) {
                out += static_cast<char>('a' + fileOf(m.from()));
            } else if (needRank) {
                out += static_cast<char>('1' + rankOf(m.from()));
            }

            if (capture) out += 'x';
            out += squareName(m.to());
        }
    }

    // Check / checkmate suffix.
    Position after = pos;
    after.make(m);
    if (after.inCheck()) {
        out += after.hasLegalMoves() ? '+' : '#';
    }
    return out;
}

Move San::parse(const Position& pos, const std::string& s) {
    if (s.empty()) return Move::none();

    // Strip trailing '+' or '#' decorations and any annotation glyphs.
    std::string t = s;
    while (!t.empty()) {
        char c = t.back();
        if (c == '+' || c == '#' || c == '!' || c == '?') t.pop_back();
        else break;
    }
    if (t.empty()) return Move::none();

    Position copy = pos;
    MoveList legal;
    MoveGenerator::generateLegal(copy, legal);

    // Castling.
    if (t == "O-O" || t == "0-0") {
        for (Move m : legal) {
            if (movingPieceType(pos, m) == PieceType::King
                && squareIndex(m.to()) - squareIndex(m.from()) == 2) {
                return m;
            }
        }
        return Move::none();
    }
    if (t == "O-O-O" || t == "0-0-0") {
        for (Move m : legal) {
            if (movingPieceType(pos, m) == PieceType::King
                && squareIndex(m.from()) - squareIndex(m.to()) == 2) {
                return m;
            }
        }
        return Move::none();
    }

    // Promotion suffix: "=Q", "=R", "=B", "=N". Also tolerate bare "Q" / "q".
    Move::Promotion promo = Move::Promotion::None;
    if (t.size() >= 2 && t[t.size() - 2] == '=') {
        switch (std::toupper(static_cast<unsigned char>(t.back()))) {
            case 'N': promo = Move::Promotion::Knight; break;
            case 'B': promo = Move::Promotion::Bishop; break;
            case 'R': promo = Move::Promotion::Rook;   break;
            case 'Q': promo = Move::Promotion::Queen;  break;
            default:  return Move::none();
        }
        t.resize(t.size() - 2);
    } else if (!t.empty()) {
        char c = t.back();
        if (c == 'N' || c == 'B' || c == 'R' || c == 'Q'
            || c == 'n' || c == 'b' || c == 'r' || c == 'q') {
            // Only treat as promotion suffix if the rest of the string ends
            // in a valid destination square (file+rank). Otherwise this is a
            // piece-letter prefix like "Nf3".
            if (t.size() >= 3) {
                char r = t[t.size() - 2];
                char f = t[t.size() - 3];
                if (f >= 'a' && f <= 'h' && r >= '1' && r <= '8') {
                    switch (std::toupper(static_cast<unsigned char>(c))) {
                        case 'N': promo = Move::Promotion::Knight; break;
                        case 'B': promo = Move::Promotion::Bishop; break;
                        case 'R': promo = Move::Promotion::Rook;   break;
                        case 'Q': promo = Move::Promotion::Queen;  break;
                    }
                    t.pop_back();
                }
            }
        }
    }
    if (t.size() < 2) return Move::none();

    // Destination square = last two characters of what remains.
    char rChar = t[t.size() - 1];
    char fChar = t[t.size() - 2];
    if (fChar < 'a' || fChar > 'h' || rChar < '1' || rChar > '8') {
        return Move::none();
    }
    Square to = makeSquare(fChar - 'a', rChar - '1');
    t.resize(t.size() - 2);

    // Optional 'x' capture indicator.
    if (!t.empty() && t.back() == 'x') t.pop_back();

    // Piece-type prefix.
    PieceType wantedType = PieceType::Pawn;
    if (!t.empty()) {
        char c = t.front();
        switch (c) {
            case 'N': wantedType = PieceType::Knight; t.erase(t.begin()); break;
            case 'B': wantedType = PieceType::Bishop; t.erase(t.begin()); break;
            case 'R': wantedType = PieceType::Rook;   t.erase(t.begin()); break;
            case 'Q': wantedType = PieceType::Queen;  t.erase(t.begin()); break;
            case 'K': wantedType = PieceType::King;   t.erase(t.begin()); break;
            default:
                if (c >= 'a' && c <= 'h') {
                    // Pawn capture file specifier or pawn move starting square
                    // — leave for disambiguation parsing below.
                }
                break;
        }
    }

    // Disambiguation: at most one file char, at most one rank char.
    int wantFile = -1, wantRank = -1;
    for (char c : t) {
        if (c >= 'a' && c <= 'h') wantFile = c - 'a';
        else if (c >= '1' && c <= '8') wantRank = c - '1';
        else return Move::none();
    }

    Move match = Move::none();
    int matchCount = 0;
    for (Move m : legal) {
        if (m.to() != to) continue;
        if (m.promotion() != promo) continue;
        if (movingPieceType(pos, m) != wantedType) continue;
        if (wantFile != -1 && fileOf(m.from()) != wantFile) continue;
        if (wantRank != -1 && rankOf(m.from()) != wantRank) continue;
        match = m;
        ++matchCount;
    }
    return matchCount == 1 ? match : Move::none();
}

}  // namespace cnnv::chess
