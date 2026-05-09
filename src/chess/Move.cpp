#include "chess/Move.h"

namespace cnnv::chess {

namespace {

char fileChar(Square sq) noexcept {
    return static_cast<char>('a' + fileOf(sq));
}

char rankChar(Square sq) noexcept {
    return static_cast<char>('1' + rankOf(sq));
}

char promotionChar(Move::Promotion p) noexcept {
    switch (p) {
        case Move::Promotion::Knight: return 'n';
        case Move::Promotion::Bishop: return 'b';
        case Move::Promotion::Rook:   return 'r';
        case Move::Promotion::Queen:  return 'q';
        case Move::Promotion::None:   return '\0';
    }
    return '\0';
}

}  // namespace

std::string Move::toUci() const {
    if (isNone()) return {};
    std::string s;
    s.reserve(5);
    s += fileChar(from());
    s += rankChar(from());
    s += fileChar(to());
    s += rankChar(to());
    char p = promotionChar(promotion());
    if (p) s += p;
    return s;
}

Move Move::parseUci(const std::string& s) {
    if (s.size() < 4 || s.size() > 5) return Move::none();
    int ff = s[0] - 'a';
    int fr = s[1] - '1';
    int tf = s[2] - 'a';
    int tr = s[3] - '1';
    if (ff < 0 || ff > 7 || fr < 0 || fr > 7 ||
        tf < 0 || tf > 7 || tr < 0 || tr > 7) {
        return Move::none();
    }
    Promotion p = Promotion::None;
    if (s.size() == 5) {
        switch (s[4]) {
            case 'n': p = Promotion::Knight; break;
            case 'b': p = Promotion::Bishop; break;
            case 'r': p = Promotion::Rook;   break;
            case 'q': p = Promotion::Queen;  break;
            default: return Move::none();
        }
    }
    return Move(makeSquare(ff, fr), makeSquare(tf, tr), p);
}

}  // namespace cnnv::chess
