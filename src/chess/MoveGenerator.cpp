#include "chess/MoveGenerator.h"

#include "chess/SlidingAttacks.h"

namespace cnnv::chess {

namespace {

void emitWithPromotions(MoveList& out, Square from, Square to, Color color) {
    int toRank = rankOf(to);
    bool promoting = (color == Color::White && toRank == 7)
                  || (color == Color::Black && toRank == 0);
    if (promoting) {
        out.push(Move(from, to, Move::Promotion::Queen));
        out.push(Move(from, to, Move::Promotion::Rook));
        out.push(Move(from, to, Move::Promotion::Bishop));
        out.push(Move(from, to, Move::Promotion::Knight));
    } else {
        out.push(Move(from, to));
    }
}

void generatePseudoLegal(const Position& pos, MoveList& out) {
    Color us = pos.sideToMove();
    Color them = other(us);
    Bitboard ourPieces = pos.colorBitboard(us);
    Bitboard theirPieces = pos.colorBitboard(them);
    Bitboard occupied = pos.occupied();
    Bitboard empty = ~occupied;

    // ---------------- Pawn moves ----------------
    Bitboard pawns = pos.pieceBitboard(us, PieceType::Pawn);
    int forward = (us == Color::White) ? 8 : -8;
    Bitboard rank2 = (us == Color::White) ? kRankMasks[1] : kRankMasks[6];
    while (pawns) {
        int from = popLsb(pawns);
        // Single push.
        int oneAhead = from + forward;
        if (oneAhead >= 0 && oneAhead < 64 && (empty & (Bitboard{1} << oneAhead))) {
            emitWithPromotions(out, static_cast<Square>(from),
                               static_cast<Square>(oneAhead), us);
            // Double push from starting rank.
            if ((Bitboard{1} << from) & rank2) {
                int twoAhead = from + 2 * forward;
                if (empty & (Bitboard{1} << twoAhead)) {
                    out.push(Move(static_cast<Square>(from),
                                  static_cast<Square>(twoAhead)));
                }
            }
        }
        // Captures (including promotions).
        Bitboard atk = sliding::pawnAttacks(from, us);
        Bitboard caps = atk & theirPieces;
        while (caps) {
            int to = popLsb(caps);
            emitWithPromotions(out, static_cast<Square>(from),
                               static_cast<Square>(to), us);
        }
        // En passant.
        if (pos.epSquare() != Square::None) {
            int epIdx = squareIndex(pos.epSquare());
            if (atk & (Bitboard{1} << epIdx)) {
                out.push(Move(static_cast<Square>(from), pos.epSquare()));
            }
        }
    }

    // ---------------- Knight moves ----------------
    Bitboard knights = pos.pieceBitboard(us, PieceType::Knight);
    while (knights) {
        int from = popLsb(knights);
        Bitboard atk = sliding::knightAttacks(from) & ~ourPieces;
        while (atk) {
            int to = popLsb(atk);
            out.push(Move(static_cast<Square>(from), static_cast<Square>(to)));
        }
    }

    // ---------------- Bishop / Rook / Queen ----------------
    Bitboard bishops = pos.pieceBitboard(us, PieceType::Bishop);
    while (bishops) {
        int from = popLsb(bishops);
        Bitboard atk = sliding::bishopAttacks(from, occupied) & ~ourPieces;
        while (atk) {
            int to = popLsb(atk);
            out.push(Move(static_cast<Square>(from), static_cast<Square>(to)));
        }
    }
    Bitboard rooks = pos.pieceBitboard(us, PieceType::Rook);
    while (rooks) {
        int from = popLsb(rooks);
        Bitboard atk = sliding::rookAttacks(from, occupied) & ~ourPieces;
        while (atk) {
            int to = popLsb(atk);
            out.push(Move(static_cast<Square>(from), static_cast<Square>(to)));
        }
    }
    Bitboard queens = pos.pieceBitboard(us, PieceType::Queen);
    while (queens) {
        int from = popLsb(queens);
        Bitboard atk = sliding::queenAttacks(from, occupied) & ~ourPieces;
        while (atk) {
            int to = popLsb(atk);
            out.push(Move(static_cast<Square>(from), static_cast<Square>(to)));
        }
    }

    // ---------------- King moves ----------------
    Square ks = pos.kingSquare(us);
    if (ks != Square::None) {
        int from = squareIndex(ks);
        Bitboard atk = sliding::kingAttacks(from) & ~ourPieces;
        while (atk) {
            int to = popLsb(atk);
            out.push(Move(static_cast<Square>(from), static_cast<Square>(to)));
        }

        // Castling: can't be in check, can't pass through check, squares in
        // between must be empty. The check filtering happens in the legality
        // pass below; here we only verify emptiness and the rights flag.
        std::uint8_t cr = pos.castlingRights();
        bool inCheckNow = pos.isSquareAttacked(ks, them);
        if (!inCheckNow) {
            if (us == Color::White) {
                if ((cr & WhiteKing)
                    && !(occupied & ((Bitboard{1} << squareIndex(Square::F1))
                                   | (Bitboard{1} << squareIndex(Square::G1))))
                    && !pos.isSquareAttacked(Square::F1, them)) {
                    out.push(Move(Square::E1, Square::G1));
                }
                if ((cr & WhiteQueen)
                    && !(occupied & ((Bitboard{1} << squareIndex(Square::B1))
                                   | (Bitboard{1} << squareIndex(Square::C1))
                                   | (Bitboard{1} << squareIndex(Square::D1))))
                    && !pos.isSquareAttacked(Square::D1, them)) {
                    out.push(Move(Square::E1, Square::C1));
                }
            } else {
                if ((cr & BlackKing)
                    && !(occupied & ((Bitboard{1} << squareIndex(Square::F8))
                                   | (Bitboard{1} << squareIndex(Square::G8))))
                    && !pos.isSquareAttacked(Square::F8, them)) {
                    out.push(Move(Square::E8, Square::G8));
                }
                if ((cr & BlackQueen)
                    && !(occupied & ((Bitboard{1} << squareIndex(Square::B8))
                                   | (Bitboard{1} << squareIndex(Square::C8))
                                   | (Bitboard{1} << squareIndex(Square::D8))))
                    && !pos.isSquareAttacked(Square::D8, them)) {
                    out.push(Move(Square::E8, Square::C8));
                }
            }
        }
    }
}

}  // namespace

void MoveGenerator::generateLegal(Position& pos, MoveList& out) {
    out.clear();
    MoveList pseudo;
    generatePseudoLegal(pos, pseudo);

    Color us = pos.sideToMove();
    for (Move m : pseudo) {
        pos.make(m);
        // After make(), our king is the side that just moved (i.e. `us`).
        Square ks = pos.kingSquare(us);
        bool kingInCheck = (ks != Square::None)
            && pos.isSquareAttacked(ks, pos.sideToMove());
        pos.unmake();
        if (!kingInCheck) out.push(m);
    }
}

}  // namespace cnnv::chess
