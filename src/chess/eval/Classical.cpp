/**
 * @file Classical.cpp
 * @brief Numerically-exact port of chess-rtk's handcrafted classical evaluator.
 *
 * This is a faithful transliteration of `chess-rtk/src/chess/classical/Wdl.java`
 * (plus its `AttackInfo`, `EvalBuffers`, `EvalScan`, `MinorMaterialState`
 * helpers). It reproduces the same eleven-term white-perspective breakdown, the
 * same win/draw/loss mapping, and the same piece-square tables.
 *
 * Coordinate handling: the chess core indexes squares a1=0..h8=63 (rank-major),
 * whereas the Java source indexes a8=0..h1=63. To keep this a literal port of
 * the Java logic, the position is snapshotted once into Java a8=0 coordinates
 * (`JavaBoard`) and every computation below runs in that space exactly as the
 * Java does. Bitboards are vertically flipped at the boundary; the public
 * `pieceSquareTable` already returns the a8=0 ordering the heatmap expects.
 */

#include "chess/eval/Classical.h"
#include "chess/eval/ClassicalTables.h"

#include "chess/Bitboard.h"
#include "chess/Piece.h"
#include "chess/Position.h"
#include "chess/SlidingAttacks.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace cnnv::chess::eval {
namespace {

// ---------------------------------------------------------------------------
// Coordinate bridge: chess core (a1=0) -> chess-rtk Java convention (a8=0).
// ---------------------------------------------------------------------------

using Bb = std::uint64_t;

/** @brief Vertical flip of a bitboard (a1<->a8); converts a1=0 <-> a8=0. */
inline Bb vflip(Bb bb) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(bb);
#else
    bb = ((bb >> 8) & 0x00FF00FF00FF00FFULL) | ((bb & 0x00FF00FF00FF00FFULL) << 8);
    bb = ((bb >> 16) & 0x0000FFFF0000FFFFULL) | ((bb & 0x0000FFFF0000FFFFULL) << 16);
    bb = (bb >> 32) | (bb << 32);
    return bb;
#endif
}

// Java chess.core.Bits masks, expressed in the a8=0 convention.
// Files are identical to the a1=0 convention (a file = indices 0,8,16,..56).
// Ranks are vertically mirrored: RANK_8 occupies indices 0..7, RANK_1 56..63.
constexpr Bb kBitsFileA = 0x0101010101010101ULL;
constexpr Bb kBitsFileH = kBitsFileA << 7;
constexpr Bb kBitsFileC = kBitsFileA << 2;
constexpr Bb kBitsFileD = kBitsFileA << 3;
constexpr Bb kBitsFileE = kBitsFileA << 4;
constexpr Bb kBitsFileF = kBitsFileA << 5;

constexpr Bb kBitsRank8 = 0x00000000000000FFULL;  // a8..h8 (indices 0..7)
constexpr Bb kBitsRank7 = kBitsRank8 << 8;         // rank 7  (indices 8..15)
constexpr Bb kBitsRank6 = kBitsRank8 << 16;        // rank 6
constexpr Bb kBitsRank5 = kBitsRank8 << 24;        // rank 5
constexpr Bb kBitsRank4 = kBitsRank8 << 32;        // rank 4
constexpr Bb kBitsRank3 = kBitsRank8 << 40;        // rank 3
constexpr Bb kBitsRank2 = kBitsRank8 << 48;        // rank 2
constexpr Bb kBitsRank1 = kBitsRank8 << 56;        // rank 1 (indices 56..63)

// chess.core.Field constants (a8=0 convention) used by the evaluator.
constexpr int kFieldC8 = 2;
constexpr int kFieldG8 = 6;
constexpr int kFieldD5 = 27;
constexpr int kFieldE5 = 28;
constexpr int kFieldD4 = 35;
constexpr int kFieldE4 = 36;
constexpr int kFieldC1 = 58;
constexpr int kFieldG1 = 62;

// Java Piece type codes (absolute).
constexpr int kPawn = 1;
constexpr int kKnight = 2;
constexpr int kBishop = 3;
constexpr int kRook = 4;
constexpr int kQueen = 5;
constexpr int kKing = 6;

// Java Position piece-index codes (used by pieces()).
constexpr int kWhitePawn = 0;
constexpr int kWhiteKnight = 1;
constexpr int kWhiteBishop = 2;
constexpr int kWhiteRook = 3;
constexpr int kWhiteQueen = 4;
constexpr int kWhiteKing = 5;
constexpr int kBlackPawn = 6;
constexpr int kBlackKnight = 7;
constexpr int kBlackBishop = 8;
constexpr int kBlackRook = 9;
constexpr int kBlackQueen = 10;
constexpr int kBlackKing = 11;

// Java Position castling-right constants.
constexpr int kWhiteKingside = 1;
constexpr int kWhiteQueenside = 2;
constexpr int kBlackKingside = 4;
constexpr int kBlackQueenside = 8;

constexpr int kValuePawn = 100;
constexpr int kValueKnight = 300;
constexpr int kValueBishop = 300;
constexpr int kValueRook = 500;
constexpr int kValueQueen = 900;

constexpr int kStartTotalMaterialCp =
    2 * (8 * kValuePawn + 2 * kValueKnight + 2 * kValueBishop + 2 * kValueRook + kValueQueen);

/** @brief Center four files (C,D,E,F). */
constexpr Bb kCenterFiles = kBitsFileC | kBitsFileD | kBitsFileE | kBitsFileF;
/** @brief Center four squares (d4,e4,d5,e5). */
constexpr Bb kCenterSquares =
    (Bb{1} << kFieldD4) | (Bb{1} << kFieldE4) | (Bb{1} << kFieldD5) | (Bb{1} << kFieldE5);
constexpr Bb kWhiteSpaceMask = kCenterFiles & (kBitsRank2 | kBitsRank3 | kBitsRank4);
constexpr Bb kBlackSpaceMask = kCenterFiles & (kBitsRank7 | kBitsRank6 | kBitsRank5);

/** @brief Light/dark square complexes in the a8=0 convention. */
constexpr Bb buildSquareColor(int color) {
    Bb mask = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if ((((sq & 7) + (sq >> 3)) & 1) == color) {
            mask |= (Bb{1} << sq);
        }
    }
    return mask;
}
constexpr Bb kLightSquares = buildSquareColor(0);
constexpr Bb kDarkSquares = ~kLightSquares;

/** @brief A8=0 -> a1=0 board square (Java square to chess-core square). */
inline int toCoreSquare(int javaSq) noexcept { return javaSq ^ 56; }

// Attack generators in the a8=0 (Java) convention. The chess core generators
// work in a1=0; flipping the source square, the occupancy, and the result is
// equivalent for all these (vertical-flip-symmetric) piece attacks.
inline Bb knightAttacksJ(int sq) noexcept {
    return vflip(sliding::knightAttacks(toCoreSquare(sq)));
}
inline Bb kingAttacksJ(int sq) noexcept {
    return vflip(sliding::kingAttacks(toCoreSquare(sq)));
}
inline Bb bishopAttacksJ(int sq, Bb occJava) noexcept {
    return vflip(sliding::bishopAttacks(toCoreSquare(sq), vflip(occJava)));
}
inline Bb rookAttacksJ(int sq, Bb occJava) noexcept {
    return vflip(sliding::rookAttacks(toCoreSquare(sq), vflip(occJava)));
}

inline int bitCount(Bb bb) noexcept { return popcount(bb); }
inline int trailingZeros(Bb bb) noexcept { return lsb(bb); }

/** @brief Numbers.clamp01 from chess-rtk. */
inline double clamp01(double v) noexcept {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

/**
 * @brief Java's Math.round semantics: floor(x + 0.5) (round half toward +inf).
 *
 * This differs from std::lround, which rounds half away from zero, so for
 * negative half-integer values the two disagree. The classical evaluator's
 * blended terms can be negative, so matching Java exactly is required.
 */
inline int javaRound(double x) noexcept {
    return static_cast<int>(std::floor(x + 0.5));
}

// ---------------------------------------------------------------------------
// JavaBoard: a snapshot of the position in chess-rtk's a8=0 coordinate space.
// ---------------------------------------------------------------------------

struct JavaBoard {
    // Signed mailbox: +1..+6 white pawn..king, -1..-6 black, 0 empty (a8=0).
    std::array<int, 64> board{};
    // Java Position piece bitboards indexed 0..11 (a8=0 convention).
    std::array<Bb, 12> pieces{};
    Bb occAll = 0;
    Bb occWhite = 0;
    Bb occBlack = 0;
    int whiteKing = -1;  // a8=0 square or -1
    int blackKing = -1;
    bool whiteToMove = true;
    bool inCheck = false;
    int castling = 0;  // Java castling mask

    Bb piecesOf(int idx) const noexcept { return pieces[static_cast<std::size_t>(idx)]; }
    Bb occupancy() const noexcept { return occAll; }
    Bb occupancy(bool white) const noexcept { return white ? occWhite : occBlack; }
    int kingSquare(bool white) const noexcept { return white ? whiteKing : blackKing; }
    bool isWhiteToMove() const noexcept { return whiteToMove; }
    bool canCastle(int right) const noexcept { return (castling & right) != 0; }
    int pieceAt(int sq) const noexcept { return board[static_cast<std::size_t>(sq)]; }
};

JavaBoard snapshot(const Position& pos) {
    JavaBoard jb;
    // Bitboards: convert each core (a1=0) bitboard to Java (a8=0) by vflip.
    const Color colors[2] = {Color::White, Color::Black};
    const PieceType types[6] = {PieceType::Pawn,   PieceType::Knight, PieceType::Bishop,
                                PieceType::Rook,    PieceType::Queen,  PieceType::King};
    for (int c = 0; c < 2; ++c) {
        for (int t = 0; t < 6; ++t) {
            Bb bb = vflip(pos.pieceBitboard(colors[c], types[t]));
            jb.pieces[static_cast<std::size_t>(c * 6 + t)] = bb;
        }
    }
    jb.occWhite = vflip(pos.colorBitboard(Color::White));
    jb.occBlack = vflip(pos.colorBitboard(Color::Black));
    jb.occAll = vflip(pos.occupied());

    // Signed mailbox in Java coordinates.
    for (int js = 0; js < 64; ++js) {
        Piece p = pos.pieceAt(toCoreSquare(js));
        int code = 0;
        if (!p.isNone()) {
            int type = static_cast<int>(p.type);  // 1..6
            code = (p.color == Color::White) ? type : -type;
        }
        jb.board[static_cast<std::size_t>(js)] = code;
    }

    Square wk = pos.kingSquare(Color::White);
    Square bk = pos.kingSquare(Color::Black);
    jb.whiteKing = (wk == Square::None) ? -1 : (squareIndex(wk) ^ 56);
    jb.blackKing = (bk == Square::None) ? -1 : (squareIndex(bk) ^ 56);

    jb.whiteToMove = pos.sideToMove() == Color::White;
    jb.inCheck = pos.inCheck();

    std::uint8_t cr = pos.castlingRights();
    int mask = 0;
    if (cr & WhiteKing) mask |= kWhiteKingside;
    if (cr & WhiteQueen) mask |= kWhiteQueenside;
    if (cr & BlackKing) mask |= kBlackKingside;
    if (cr & BlackQueen) mask |= kBlackQueenside;
    jb.castling = mask;
    return jb;
}

// ---------------------------------------------------------------------------
// Small Java helpers, ported verbatim (a8=0 convention).
// ---------------------------------------------------------------------------

/** @brief chess.classical.Wdl.flip: vertical square flip (a8<->a1). */
inline int flip(int square) noexcept { return square ^ 56; }

/** @brief chess.core.Bits.rank: chess rank 0..7 (0 == rank 1). */
inline int bitsRank(int square) noexcept { return 7 - (square >> 3); }

inline int sideIndex(bool white) noexcept { return white ? kWhite : kBlack; }

/** @brief Wdl.pawnAttackMask. */
inline Bb pawnAttackMask(bool white, Bb pawns) noexcept {
    if (white) {
        return ((pawns & ~kBitsFileA) >> 9) | ((pawns & ~kBitsFileH) >> 7);
    }
    return ((pawns & ~kBitsFileA) << 7) | ((pawns & ~kBitsFileH) << 9);
}

/** @brief Wdl.pawnPushMask. */
inline Bb pawnPushMask(bool white, Bb pawns, Bb empty) noexcept {
    Bb single = white ? ((pawns >> 8) & empty) : ((pawns << 8) & empty);
    if (white) {
        return single | (((single & kBitsRank3) >> 8) & empty);
    }
    return single | (((single & kBitsRank6) << 8) & empty);
}

/** @brief Wdl.fileBitboard. */
inline Bb fileBitboard(int file) noexcept { return kBitsFileA << file; }

inline Bb adjacentFileBitboard(int file) noexcept {
    Bb mask = 0;
    if (file > 0) mask |= fileBitboard(file - 1);
    if (file < 7) mask |= fileBitboard(file + 1);
    return mask;
}

inline int kingDistance(int a, int b) noexcept {
    return std::max(std::abs((a & 7) - (b & 7)), std::abs((a >> 3) - (b >> 3)));
}

inline int edgeFileDistance(int file) noexcept { return std::min(file, 7 - file); }

inline int blend(int mg, int eg, double phase) noexcept {
    return javaRound(mg * phase + eg * (1.0 - phase));
}

inline int mobilityScore(int type, int mobility) noexcept {
    switch (type) {
        case 1: {
            int idx = std::min<int>(mobility, static_cast<int>(kBishopMobilityCp.size()) - 1);
            return kBishopMobilityCp[static_cast<std::size_t>(idx)];
        }
        case 2: {
            int idx = std::min<int>(mobility, static_cast<int>(kRookMobilityCp.size()) - 1);
            return kRookMobilityCp[static_cast<std::size_t>(idx)];
        }
        case 3: {
            int idx = std::min<int>(mobility, static_cast<int>(kQueenMobilityCp.size()) - 1);
            return kQueenMobilityCp[static_cast<std::size_t>(idx)];
        }
        default: {
            int idx = std::min<int>(mobility, static_cast<int>(kKnightMobilityCp.size()) - 1);
            return kKnightMobilityCp[static_cast<std::size_t>(idx)];
        }
    }
}

inline int mobilityEgScore(int type, int mobility) noexcept {
    switch (type) {
        case 1: {
            int idx = std::min<int>(mobility, static_cast<int>(kBishopMobilityEgCp.size()) - 1);
            return kBishopMobilityEgCp[static_cast<std::size_t>(idx)];
        }
        case 2: {
            int idx = std::min<int>(mobility, static_cast<int>(kRookMobilityEgCp.size()) - 1);
            return kRookMobilityEgCp[static_cast<std::size_t>(idx)];
        }
        case 3: {
            int idx = std::min<int>(mobility, static_cast<int>(kQueenMobilityEgCp.size()) - 1);
            return kQueenMobilityEgCp[static_cast<std::size_t>(idx)];
        }
        default: {
            int idx = std::min<int>(mobility, static_cast<int>(kKnightMobilityEgCp.size()) - 1);
            return kKnightMobilityEgCp[static_cast<std::size_t>(idx)];
        }
    }
}

inline bool isOutpost(bool white, int square, Bb ownPawnAttacks, Bb enemyPawnAttacks) noexcept {
    Bb bit = Bb{1} << square;
    if ((ownPawnAttacks & bit) == 0 || (enemyPawnAttacks & bit) != 0) {
        return false;
    }
    int relativeRank = white ? bitsRank(square) : 7 - bitsRank(square);
    return relativeRank >= 3 && relativeRank <= 5;
}

inline int badBishopPenalty(int bishop, int mobility, Bb ownPawns) noexcept {
    if (mobility > 4) {
        return 0;
    }
    Bb colorMask = (((bishop & 7) + (bishop >> 3)) & 1) == 0 ? kLightSquares : kDarkSquares;
    int sameColorPawns = bitCount(ownPawns & colorMask);
    return sameColorPawns <= 4 ? 0 : (sameColorPawns - 4) * 4 + (4 - mobility) * 3;
}

// ---------------------------------------------------------------------------
// EvalScan / EvalBuffers equivalents.
// ---------------------------------------------------------------------------

struct EvalScan {
    int whiteMaterial = 0;
    int blackMaterial = 0;
    int score = 0;
    int whiteBishops = 0;
    int blackBishops = 0;
    int whitePawns = 0;
    int blackPawns = 0;
};

struct EvalBuffers {
    std::array<int, 8> whitePawnsPerFile{};
    std::array<int, 8> blackPawnsPerFile{};
    std::array<int, 8> minWhitePawnRank{};
    std::array<int, 8> maxBlackPawnRank{};
    std::array<int, 8> minBlackPawnRank{};
    std::array<int, 8> maxWhitePawnRank{};
    std::array<int, 8> whiteRooksFileCount{};
    std::array<int, 8> blackRooksFileCount{};
    EvalScan scan;
    double phase = 1.0;

    EvalBuffers() { reset(); }

    void reset() {
        for (int i = 0; i < 8; ++i) {
            whitePawnsPerFile[static_cast<std::size_t>(i)] = 0;
            blackPawnsPerFile[static_cast<std::size_t>(i)] = 0;
            minWhitePawnRank[static_cast<std::size_t>(i)] = 8;
            maxBlackPawnRank[static_cast<std::size_t>(i)] = -1;
            minBlackPawnRank[static_cast<std::size_t>(i)] = 8;
            maxWhitePawnRank[static_cast<std::size_t>(i)] = -1;
            whiteRooksFileCount[static_cast<std::size_t>(i)] = 0;
            blackRooksFileCount[static_cast<std::size_t>(i)] = 0;
        }
        scan = EvalScan{};
        phase = 1.0;
    }
};

// ---------------------------------------------------------------------------
// AttackInfo equivalent.
// ---------------------------------------------------------------------------

struct AttackInfo {
    // attackedBy[side][type]; type 0 == ALL_ATTACKS, 1..6 == Piece types.
    std::array<std::array<Bb, 7>, 2> attackedBy{};
    std::array<Bb, 2> attackedBy2{};
    std::array<Bb, 2> kingZone{};
    std::array<int, 2> kingAttackersCount{};
    std::array<int, 2> kingAttackersWeight{};
    std::array<int, 2> kingAttacksCount{};
    std::array<int, 2> mobilityMg{};
    std::array<int, 2> mobilityEg{};
    std::array<int, 2> pieceMg{};
    std::array<int, 2> pieceEg{};

    void reset() {
        for (int side = 0; side < 2; ++side) {
            attackedBy2[static_cast<std::size_t>(side)] = 0;
            kingZone[static_cast<std::size_t>(side)] = 0;
            kingAttackersCount[static_cast<std::size_t>(side)] = 0;
            kingAttackersWeight[static_cast<std::size_t>(side)] = 0;
            kingAttacksCount[static_cast<std::size_t>(side)] = 0;
            mobilityMg[static_cast<std::size_t>(side)] = 0;
            mobilityEg[static_cast<std::size_t>(side)] = 0;
            pieceMg[static_cast<std::size_t>(side)] = 0;
            pieceEg[static_cast<std::size_t>(side)] = 0;
            for (auto& v : attackedBy[static_cast<std::size_t>(side)]) {
                v = 0;
            }
        }
    }

    Bb ab(int side, int type) const noexcept {
        return attackedBy[static_cast<std::size_t>(side)][static_cast<std::size_t>(type)];
    }

    Bb kingZoneMask(int king) const noexcept {
        if (king < 0) return 0;
        return (Bb{1} << king) | kingAttacksJ(king);
    }

    int activityType(int type) const noexcept {
        switch (type) {
            case kKnight: return 0;
            case kBishop: return 1;
            case kRook: return 2;
            case kQueen: return 3;
            default: return 0;
        }
    }

    Bb attacksForPieceType(bool white, int type, int from, Bb occupancy) const noexcept {
        switch (type) {
            case kPawn: return pawnAttackMask(white, Bb{1} << from);
            case kKnight: return knightAttacksJ(from);
            case kBishop: return bishopAttacksJ(from, occupancy);
            case kRook: return rookAttacksJ(from, occupancy);
            case kQueen: return bishopAttacksJ(from, occupancy) | rookAttacksJ(from, occupancy);
            case kKing: return kingAttacksJ(from);
            default: return 0;
        }
    }

    void addAttacks(int side, int type, Bb attacks) {
        std::size_t s = static_cast<std::size_t>(side);
        attackedBy2[s] |= attackedBy[s][kAllAttacks] & attacks;
        attackedBy[s][kAllAttacks] |= attacks;
        attackedBy[s][static_cast<std::size_t>(type)] |= attacks;
    }

    Bb outpostMask(bool white, Bb ownPawnAttacks, Bb enemyPawnAttacks) const noexcept {
        Bb ranks = white ? (kBitsRank4 | kBitsRank5 | kBitsRank6)
                         : (kBitsRank5 | kBitsRank4 | kBitsRank3);
        return ranks & ownPawnAttacks & ~enemyPawnAttacks;
    }

    bool minorBehindPawn(bool white, int square, Bb ownPawns) const noexcept {
        int pawnSquare = white ? square - 8 : square + 8;
        return pawnSquare >= 0 && pawnSquare < 64 && ((Bb{1} << pawnSquare) & ownPawns) != 0;
    }

    int relativeRank(bool white, int square) const noexcept {
        int rank = bitsRank(square);
        return white ? rank : 7 - rank;
    }

    bool sameFlank(int a, int b) const noexcept {
        return ((a & 7) <= 3) == ((b & 7) <= 3);
    }

    bool canCastleEither(const JavaBoard& pos, bool white) const noexcept {
        return white ? (pos.canCastle(kWhiteKingside) || pos.canCastle(kWhiteQueenside))
                     : (pos.canCastle(kBlackKingside) || pos.canCastle(kBlackQueenside));
    }

    void addQueenActivity(int side, bool white, int square, Bb enemyPawnAttacks) {
        std::size_t s = static_cast<std::size_t>(side);
        if (relativeRank(white, square) >= 4 && ((Bb{1} << square) & enemyPawnAttacks) == 0) {
            pieceMg[s] += 7;
            pieceEg[s] += 10;
        }
    }

    void addRookActivity(const JavaBoard& pos, int side, bool white, int square, int mobility) {
        std::size_t s = static_cast<std::size_t>(side);
        Bb file = fileBitboard(square & 7);
        if (((pos.piecesOf(kWhiteQueen) | pos.piecesOf(kBlackQueen)) & file) != 0) {
            pieceMg[s] += 6;
        }
        int relRank = relativeRank(white, square);
        Bb seventh = white ? kBitsRank7 : kBitsRank2;
        if (relRank == 6 && (pos.occupancy(!white) & seventh) != 0) {
            pieceMg[s] += 18;
            pieceEg[s] += 28;
        }
        int king = pos.kingSquare(white);
        if (mobility <= 3 && king >= 0 && sameFlank(square, king)) {
            int penalty = canCastleEither(pos, white) ? 10 : 22;
            pieceMg[s] -= penalty;
            pieceEg[s] -= penalty / 2;
        }
    }

    void addPieceActivity(const JavaBoard& pos, int side, bool white, int type, int square,
                          int mobility, Bb attacks, Bb own, Bb ownPawns, Bb enemyPawns,
                          Bb enemyPawnAttacks, int ownKing) {
        std::size_t s = static_cast<std::size_t>(side);
        if (type == kKnight || type == kBishop) {
            Bb pawnAttacks = attackedBy[s][kPawn];
            if (isOutpost(white, square, pawnAttacks, enemyPawnAttacks)) {
                int bonus = type == kKnight ? kKnightOutpostCp : kBishopOutpostCp;
                pieceMg[s] += bonus;
                pieceEg[s] += bonus / 2;
            } else if ((attacks & outpostMask(white, pawnAttacks, enemyPawnAttacks) & ~own) != 0) {
                int bonus = type == kKnight ? 12 : 7;
                pieceMg[s] += bonus;
                pieceEg[s] += bonus / 2;
            }
            if (minorBehindPawn(white, square, ownPawns)) {
                pieceMg[s] += 8;
                pieceEg[s] += 5;
            }
            if (ownKing >= 0) {
                int protectorDistance = kingDistance(ownKing, square);
                int weight = type == kKnight ? 3 : 2;
                pieceMg[s] -= std::max(0, protectorDistance - 1) * weight;
            }
        }
        if (type == kBishop) {
            int penalty = badBishopPenalty(square, mobility, ownPawns);
            pieceMg[s] -= penalty;
            pieceEg[s] -= penalty / 2;
            if (bitCount(bishopAttacksJ(square, ownPawns | enemyPawns) & kCenterSquares) >= 2) {
                pieceMg[s] += 7;
                pieceEg[s] += 4;
            }
        } else if (type == kRook) {
            addRookActivity(pos, side, white, square, mobility);
        } else if (type == kQueen) {
            addQueenActivity(side, white, square, enemyPawnAttacks);
        }
    }

    void scanPieces(const JavaBoard& pos, Bb pieces, bool white, int type, Bb occupancy, Bb own,
                    Bb ownPawns, Bb enemyPawns, Bb enemyKing, Bb enemyPawnAttacks, int side,
                    int ownKing) {
        std::size_t s = static_cast<std::size_t>(side);
        Bb mobilityArea = ~(own | enemyKing | enemyPawnAttacks);
        int enemy = 1 - side;
        while (pieces != 0) {
            int from = trailingZeros(pieces);
            pieces &= pieces - 1;
            Bb attacks = attacksForPieceType(white, type, from, occupancy);
            addAttacks(side, type, attacks);
            if (type != kPawn && type != kKing) {
                int mobility = bitCount(attacks & mobilityArea);
                int at = activityType(type);
                mobilityMg[s] += mobilityScore(at, mobility);
                mobilityEg[s] += mobilityEgScore(at, mobility);
                addPieceActivity(pos, side, white, type, from, mobility, attacks, own, ownPawns,
                                 enemyPawns, enemyPawnAttacks, ownKing);
            }
            Bb kingHits = attacks & kingZone[static_cast<std::size_t>(enemy)];
            if (kingHits != 0 && type != kKing) {
                kingAttackersCount[s]++;
                kingAttackersWeight[s] += kKingAttackWeight[static_cast<std::size_t>(type)];
                kingAttacksCount[s] += bitCount(kingHits);
            }
        }
    }

    void scanSide(const JavaBoard& pos, bool white, Bb occupancy, Bb enemyPawnAttacks) {
        int side = sideIndex(white);
        Bb own = pos.occupancy(white);
        Bb ownPawns = pos.piecesOf(white ? kWhitePawn : kBlackPawn);
        Bb enemyPawns = pos.piecesOf(white ? kBlackPawn : kWhitePawn);
        Bb enemyKing = pos.piecesOf(white ? kBlackKing : kWhiteKing);
        int ownKing = pos.kingSquare(white);
        scanPieces(pos, ownPawns, white, kPawn, occupancy, own, ownPawns, enemyPawns, enemyKing,
                   enemyPawnAttacks, side, ownKing);
        scanPieces(pos, pos.piecesOf(white ? kWhiteKnight : kBlackKnight), white, kKnight, occupancy,
                   own, ownPawns, enemyPawns, enemyKing, enemyPawnAttacks, side, ownKing);
        scanPieces(pos, pos.piecesOf(white ? kWhiteBishop : kBlackBishop), white, kBishop, occupancy,
                   own, ownPawns, enemyPawns, enemyKing, enemyPawnAttacks, side, ownKing);
        scanPieces(pos, pos.piecesOf(white ? kWhiteRook : kBlackRook), white, kRook, occupancy, own,
                   ownPawns, enemyPawns, enemyKing, enemyPawnAttacks, side, ownKing);
        scanPieces(pos, pos.piecesOf(white ? kWhiteQueen : kBlackQueen), white, kQueen, occupancy,
                   own, ownPawns, enemyPawns, enemyKing, enemyPawnAttacks, side, ownKing);
        scanPieces(pos, pos.piecesOf(white ? kWhiteKing : kBlackKing), white, kKing, occupancy, own,
                   ownPawns, enemyPawns, enemyKing, enemyPawnAttacks, side, ownKing);
    }

    void build(const JavaBoard& pos) {
        reset();
        int whiteKing = pos.kingSquare(true);
        int blackKing = pos.kingSquare(false);
        kingZone[kWhite] = kingZoneMask(whiteKing);
        kingZone[kBlack] = kingZoneMask(blackKing);
        Bb occupancy = pos.occupancy();
        Bb whitePawnAttacks = pawnAttackMask(true, pos.piecesOf(kWhitePawn));
        Bb blackPawnAttacks = pawnAttackMask(false, pos.piecesOf(kBlackPawn));
        scanSide(pos, true, occupancy, blackPawnAttacks);
        scanSide(pos, false, occupancy, whitePawnAttacks);
    }
};

// ---------------------------------------------------------------------------
// Material / PST scan.
// ---------------------------------------------------------------------------

void applyPawn(EvalScan& scan, EvalBuffers& buffers, bool white, int square, int psq) {
    int file = square & 7;
    int rank = square >> 3;
    int pawnPst = kPawnPst[static_cast<std::size_t>(psq)];
    scan.score += white ? pawnPst : -pawnPst;
    std::size_t f = static_cast<std::size_t>(file);
    if (white) {
        scan.whitePawns++;
        buffers.whitePawnsPerFile[f]++;
        if (rank < buffers.minWhitePawnRank[f]) buffers.minWhitePawnRank[f] = rank;
        if (rank > buffers.maxWhitePawnRank[f]) buffers.maxWhitePawnRank[f] = rank;
    } else {
        scan.blackPawns++;
        buffers.blackPawnsPerFile[f]++;
        if (rank > buffers.maxBlackPawnRank[f]) buffers.maxBlackPawnRank[f] = rank;
        if (rank < buffers.minBlackPawnRank[f]) buffers.minBlackPawnRank[f] = rank;
    }
}

void applyKnight(EvalScan& scan, bool white, int psq) {
    int v = kKnightPst[static_cast<std::size_t>(psq)];
    scan.score += white ? v : -v;
}

void applyBishop(EvalScan& scan, bool white, int psq) {
    int v = kBishopPst[static_cast<std::size_t>(psq)];
    scan.score += white ? v : -v;
    if (white) scan.whiteBishops++;
    else scan.blackBishops++;
}

void applyRook(EvalScan& scan, EvalBuffers& buffers, bool white, int square, int psq) {
    int file = square & 7;
    int v = kRookPst[static_cast<std::size_t>(psq)];
    scan.score += white ? v : -v;
    if (white) buffers.whiteRooksFileCount[static_cast<std::size_t>(file)]++;
    else buffers.blackRooksFileCount[static_cast<std::size_t>(file)]++;
}

void applyQueen(EvalScan& scan, bool white, int psq) {
    int v = kQueenPst[static_cast<std::size_t>(psq)];
    scan.score += white ? v : -v;
}

void applyMaterial(EvalScan& scan, int type, bool white) {
    int value = 0;
    switch (type) {
        case kPawn: value = kValuePawn; break;
        case kKnight: value = kValueKnight; break;
        case kBishop: value = kValueBishop; break;
        case kRook: value = kValueRook; break;
        case kQueen: value = kValueQueen; break;
        default: value = 0; break;  // king
    }
    if (white) scan.whiteMaterial += value;
    else scan.blackMaterial += value;
}

void applyPieceSquareAndStructure(EvalScan& scan, EvalBuffers& buffers, int type, int square,
                                  bool white, int psq) {
    switch (type) {
        case kPawn: applyPawn(scan, buffers, white, square, psq); break;
        case kKnight: applyKnight(scan, white, psq); break;
        case kBishop: applyBishop(scan, white, psq); break;
        case kRook: applyRook(scan, buffers, white, square, psq); break;
        case kQueen: applyQueen(scan, white, psq); break;
        default: break;
    }
}

void scanMaterialAndPst(const JavaBoard& pos, EvalBuffers& buffers, EvalScan& scan) {
    scan = EvalScan{};
    for (int square = 0; square < 64; ++square) {
        int piece = pos.board[static_cast<std::size_t>(square)];
        if (piece == 0 || piece == kKing || piece == -kKing) {
            continue;
        }
        bool white = piece > 0;
        int type = std::abs(piece);
        int psq = white ? square : flip(square);
        applyMaterial(scan, type, white);
        applyPieceSquareAndStructure(scan, buffers, type, square, white, psq);
    }
}

double updatePhase(const EvalScan& scan, EvalBuffers& buffers) {
    int totalMaterial = scan.whiteMaterial + scan.blackMaterial;
    double phase = clamp01(totalMaterial / static_cast<double>(kStartTotalMaterialCp));
    buffers.phase = phase;
    return phase;
}

// ---------------------------------------------------------------------------
// Term: bishop pair.
// ---------------------------------------------------------------------------

int bishopPairCp(int whiteBishops, int blackBishops) {
    int score = 0;
    if (whiteBishops >= 2) score += kBishopPairCp;
    if (blackBishops >= 2) score -= kBishopPairCp;
    return score;
}

// ---------------------------------------------------------------------------
// Term: pawn structure.
// ---------------------------------------------------------------------------

int fileMaskOf(const std::array<int, 8>& pawnsPerFile) {
    int mask = 0;
    for (int f = 0; f < 8; ++f) {
        if (pawnsPerFile[static_cast<std::size_t>(f)] != 0) mask |= (1 << f);
    }
    return mask;
}

int adjacentFileMask(int file) {
    int mask = 0;
    if (file > 0) mask |= (1 << (file - 1));
    if (file < 7) mask |= (1 << (file + 1));
    return mask;
}

int doubledPawnsScore(const std::array<int, 8>& whitePawnsPerFile,
                      const std::array<int, 8>& blackPawnsPerFile) {
    int score = 0;
    for (int f = 0; f < 8; ++f) {
        int w = whitePawnsPerFile[static_cast<std::size_t>(f)];
        int b = blackPawnsPerFile[static_cast<std::size_t>(f)];
        if (w > 1) score -= (w - 1) * 12;
        if (b > 1) score += (b - 1) * 12;
    }
    return score;
}

int isolatedPawnsScore(const std::array<int, 8>& whitePawnsPerFile,
                       const std::array<int, 8>& blackPawnsPerFile, int whiteFileMask,
                       int blackFileMask) {
    int score = 0;
    for (int f = 0; f < 8; ++f) {
        int adj = adjacentFileMask(f);
        if (whitePawnsPerFile[static_cast<std::size_t>(f)] != 0 && (whiteFileMask & adj) == 0) {
            score -= 10;
        }
        if (blackPawnsPerFile[static_cast<std::size_t>(f)] != 0 && (blackFileMask & adj) == 0) {
            score += 10;
        }
    }
    return score;
}

bool enemyPawnInFrontForWhite(int file, int whiteRank, const std::array<int, 8>& minBlackPawnRank) {
    for (int df = -1; df <= 1; ++df) {
        int f = file + df;
        if (f < 0 || f > 7) continue;
        if (minBlackPawnRank[static_cast<std::size_t>(f)] < whiteRank) return true;
    }
    return false;
}

bool enemyPawnInFrontForBlack(int file, int blackRank, const std::array<int, 8>& maxWhitePawnRank) {
    for (int df = -1; df <= 1; ++df) {
        int f = file + df;
        if (f < 0 || f > 7) continue;
        if (maxWhitePawnRank[static_cast<std::size_t>(f)] > blackRank) return true;
    }
    return false;
}

Bb passedPawnMask(bool white, Bb pawns, const std::array<int, 8>& enemyFrontRanks) {
    Bb passed = 0;
    while (pawns != 0) {
        int square = trailingZeros(pawns);
        pawns &= pawns - 1;
        int file = square & 7;
        int rank = square >> 3;
        bool blocked = white ? enemyPawnInFrontForWhite(file, rank, enemyFrontRanks)
                             : enemyPawnInFrontForBlack(file, rank, enemyFrontRanks);
        if (!blocked) passed |= Bb{1} << square;
    }
    return passed;
}

Bb forwardFileMask(bool white, int square) {
    Bb mask = 0;
    int file = square & 7;
    int row = square >> 3;
    if (white) {
        for (int r = row - 1; r >= 0; --r) mask |= Bb{1} << ((r << 3) | file);
    } else {
        for (int r = row + 1; r < 8; ++r) mask |= Bb{1} << ((r << 3) | file);
    }
    return mask;
}

bool hasConnectedPasser(Bb passed, int square) {
    int file = square & 7;
    int rank = square >> 3;
    Bb adjacent = passed & adjacentFileBitboard(file);
    while (adjacent != 0) {
        int other = trailingZeros(adjacent);
        adjacent &= adjacent - 1;
        if (std::abs((other >> 3) - rank) <= 1) return true;
    }
    return false;
}

int passedPawnScore(const JavaBoard& pos, const AttackInfo& attacks, bool white, Bb passed, Bb pawns,
                    double phase) {
    int mg = 0;
    int eg = 0;
    Bb defendedByPawn = pawnAttackMask(white, pawns);
    int us = sideIndex(white);
    int them = 1 - us;
    Bb remaining = passed;
    while (remaining != 0) {
        int square = trailingZeros(remaining);
        remaining &= remaining - 1;
        int relRank = white ? 7 - (square >> 3) : square >> 3;
        int baseMg = 8 + 7 * relRank + relRank * relRank;
        int baseEg = 18 + 10 * relRank + relRank * relRank * 2;
        Bb bit = Bb{1} << square;
        if ((defendedByPawn & bit) != 0) {
            baseMg += 8 + 2 * relRank;
            baseEg += 11 + 3 * relRank;
        }
        if (hasConnectedPasser(passed, square)) {
            baseMg += 7 + 2 * relRank;
            baseEg += 9 + 3 * relRank;
        }
        int blockSquare = white ? square - 8 : square + 8;
        if (blockSquare >= 0 && blockSquare < 64) {
            Bb blockBit = Bb{1} << blockSquare;
            if ((pos.occupancy() & blockBit) != 0) {
                baseMg -= 10 + 3 * relRank;
                baseEg -= 13 + 4 * relRank;
            } else {
                Bb path = forwardFileMask(white, square);
                Bb unsafe = path & attacks.ab(them, kAllAttacks) & ~attacks.ab(us, kAllAttacks);
                if (unsafe == 0) {
                    baseMg += 10 + 4 * relRank;
                    baseEg += 18 + 6 * relRank;
                } else if ((unsafe & blockBit) == 0) {
                    baseMg += 5 + 2 * relRank;
                    baseEg += 9 + 3 * relRank;
                }
                int ownKing = pos.kingSquare(white);
                int enemyKing = pos.kingSquare(!white);
                if (ownKing >= 0 && enemyKing >= 0 && relRank >= 4) {
                    int ownDistance = kingDistance(ownKing, blockSquare);
                    int enemyDistance = kingDistance(enemyKing, blockSquare);
                    baseEg += (enemyDistance - ownDistance) * (3 + relRank);
                }
            }
        }
        mg += baseMg - edgeFileDistance(square & 7) * 3;
        eg += baseEg - edgeFileDistance(square & 7) * 2;
    }
    return blend(mg, eg, phase);
}

int passedPawnsScore(const JavaBoard& pos, const std::array<int, 8>& minBlackPawnRank,
                     const std::array<int, 8>& maxWhitePawnRank, const AttackInfo& attacks,
                     double phase) {
    Bb whitePassed = passedPawnMask(true, pos.piecesOf(kWhitePawn), minBlackPawnRank);
    Bb blackPassed = passedPawnMask(false, pos.piecesOf(kBlackPawn), maxWhitePawnRank);
    int white = passedPawnScore(pos, attacks, true, whitePassed, pos.piecesOf(kWhitePawn), phase);
    int black = passedPawnScore(pos, attacks, false, blackPassed, pos.piecesOf(kBlackPawn), phase);
    return white - black;
}

int pawnStructureCp(const JavaBoard& pos, const std::array<int, 8>& whitePawnsPerFile,
                    const std::array<int, 8>& blackPawnsPerFile,
                    const std::array<int, 8>& minBlackPawnRank,
                    const std::array<int, 8>& maxWhitePawnRank, const AttackInfo& attacks,
                    double phase) {
    int whiteFileMask = fileMaskOf(whitePawnsPerFile);
    int blackFileMask = fileMaskOf(blackPawnsPerFile);
    int score = 0;
    score += doubledPawnsScore(whitePawnsPerFile, blackPawnsPerFile);
    score += isolatedPawnsScore(whitePawnsPerFile, blackPawnsPerFile, whiteFileMask, blackFileMask);
    score += passedPawnsScore(pos, minBlackPawnRank, maxWhitePawnRank, attacks, phase);
    return score;
}

// ---------------------------------------------------------------------------
// Term: rook files.
// ---------------------------------------------------------------------------

int rookFileCp(const std::array<int, 8>& whiteRooksFileCount,
               const std::array<int, 8>& blackRooksFileCount,
               const std::array<int, 8>& whitePawnsPerFile,
               const std::array<int, 8>& blackPawnsPerFile) {
    int score = 0;
    for (int f = 0; f < 8; ++f) {
        std::size_t i = static_cast<std::size_t>(f);
        bool hasAnyPawn = (whitePawnsPerFile[i] + blackPawnsPerFile[i]) != 0;
        bool hasWhitePawn = whitePawnsPerFile[i] != 0;
        bool hasBlackPawn = blackPawnsPerFile[i] != 0;
        int whiteRooks = whiteRooksFileCount[i];
        int blackRooks = blackRooksFileCount[i];
        if (whiteRooks != 0) {
            if (!hasAnyPawn) score += 14 * whiteRooks;
            else if (!hasWhitePawn && hasBlackPawn) score += 8 * whiteRooks;
        }
        if (blackRooks != 0) {
            if (!hasAnyPawn) score -= 14 * blackRooks;
            else if (!hasBlackPawn && hasWhitePawn) score -= 8 * blackRooks;
        }
    }
    return score;
}

// ---------------------------------------------------------------------------
// Term: king safety.
// ---------------------------------------------------------------------------

int shieldScore(int distance) {
    switch (distance) {
        case 1: return 14;
        case 2: return 6;
        case 3: return -4;
        default: return -16;
    }
}

int stormPenalty(int distance) {
    switch (distance) {
        case 1: return 22;
        case 2: return 15;
        case 3: return 9;
        case 4: return 4;
        default: return 0;
    }
}

int ownShieldDistance(const EvalBuffers& buffers, bool white, int file, int rank) {
    std::size_t f = static_cast<std::size_t>(file);
    if (white) {
        int pawnRank = buffers.maxWhitePawnRank[f];
        return pawnRank >= 0 && pawnRank < rank ? rank - pawnRank : 8;
    }
    int pawnRank = buffers.minBlackPawnRank[f];
    return pawnRank < 8 && pawnRank > rank ? pawnRank - rank : 8;
}

int enemyStormDistance(const EvalBuffers& buffers, bool white, int file, int rank) {
    std::size_t f = static_cast<std::size_t>(file);
    if (white) {
        int pawnRank = buffers.maxBlackPawnRank[f];
        return pawnRank >= 0 && pawnRank < rank ? rank - pawnRank : 8;
    }
    int pawnRank = buffers.minWhitePawnRank[f];
    return pawnRank < 8 && pawnRank > rank ? pawnRank - rank : 8;
}

int shelterAt(const EvalBuffers& buffers, bool white, int king) {
    int file = king & 7;
    int rank = king >> 3;
    int centerFile = std::max(1, std::min(6, file));
    int score = 6;
    for (int f = centerFile - 1; f <= centerFile + 1; ++f) {
        int ownDistance = ownShieldDistance(buffers, white, f, rank);
        int enemyDistance = enemyStormDistance(buffers, white, f, rank);
        score += shieldScore(ownDistance);
        score -= stormPenalty(enemyDistance);
        if (f == file && ownDistance > 2) {
            score -= 8;
        }
        std::size_t fi = static_cast<std::size_t>(f);
        bool hasOwnPawn = (white ? buffers.whitePawnsPerFile[fi] : buffers.blackPawnsPerFile[fi]) != 0;
        bool hasEnemyPawn = (white ? buffers.blackPawnsPerFile[fi] : buffers.whitePawnsPerFile[fi]) != 0;
        if (!hasOwnPawn) {
            score -= hasEnemyPawn ? 7 : 13;
        }
    }
    return score;
}

int bestShelterCp(const JavaBoard& pos, const EvalBuffers& buffers, bool white, int king) {
    int shelter = shelterAt(buffers, white, king);
    if (white) {
        if (pos.canCastle(kWhiteKingside)) shelter = std::max(shelter, shelterAt(buffers, true, kFieldG1));
        if (pos.canCastle(kWhiteQueenside)) shelter = std::max(shelter, shelterAt(buffers, true, kFieldC1));
    } else {
        if (pos.canCastle(kBlackKingside)) shelter = std::max(shelter, shelterAt(buffers, false, kFieldG8));
        if (pos.canCastle(kBlackQueenside)) shelter = std::max(shelter, shelterAt(buffers, false, kFieldC8));
    }
    return shelter;
}

Bb kingFlankMask(int king) {
    int file = king & 7;
    int center = std::max(1, std::min(6, file));
    return fileBitboard(center - 1) | fileBitboard(center) | fileBitboard(center + 1);
}

Bb campMask(bool white) {
    return white ? ~(kBitsRank6 | kBitsRank7 | kBitsRank8) : ~(kBitsRank1 | kBitsRank2 | kBitsRank3);
}

int kingZonePressureCp(const JavaBoard& pos, const AttackInfo& attacks, bool whiteKing, int king) {
    int us = sideIndex(whiteKing);
    int them = 1 - us;
    int attackers = attacks.kingAttackersCount[static_cast<std::size_t>(them)];
    int weight = attacks.kingAttackersWeight[static_cast<std::size_t>(them)];
    if (attackers == 0) {
        return 0;
    }
    Bb weak = attacks.ab(them, kAllAttacks) & ~attacks.attackedBy2[static_cast<std::size_t>(us)] &
              (~attacks.ab(us, kAllAttacks) | attacks.ab(us, kKing) | attacks.ab(us, kQueen));
    Bb flank = kingFlankMask(king) & campMask(whiteKing);
    int flankAttack = bitCount(attacks.ab(them, kAllAttacks) & flank) +
                      bitCount(attacks.attackedBy2[static_cast<std::size_t>(them)] & flank);
    int flankDefense = bitCount(attacks.ab(us, kAllAttacks) & flank);
    int pressure = weight + attackers * attackers * 4 +
                   12 * attacks.kingAttacksCount[static_cast<std::size_t>(them)] +
                   9 * bitCount(attacks.kingZone[static_cast<std::size_t>(us)] & weak) +
                   flankAttack * flankAttack / 2 +
                   std::max(0, attacks.mobilityMg[static_cast<std::size_t>(them)] -
                                   attacks.mobilityMg[static_cast<std::size_t>(us)]) /
                       2 -
                   flankDefense * 3;
    if (((pos.piecesOf(kWhitePawn) | pos.piecesOf(kBlackPawn)) & flank) == 0) {
        pressure += 18;
    }
    return attackers == 1 ? pressure / 2 : pressure;
}

int sideKingSafetyCp(const JavaBoard& pos, const EvalBuffers& buffers, const AttackInfo& attacks,
                     bool white, int king, double phase) {
    if (phase <= 0.0) {
        return 0;
    }
    int shelter = bestShelterCp(pos, buffers, white, king);
    int pressure = kingZonePressureCp(pos, attacks, white, king);
    Bb enemyQueens = pos.piecesOf(white ? kBlackQueen : kWhiteQueen);
    double enemyQueenFactor = 0.55;
    if (enemyQueens != 0) {
        enemyQueenFactor = 1.0;
    }
    return javaRound((shelter - pressure * enemyQueenFactor) * phase);
}

int kingSafetyCp(const JavaBoard& pos, const EvalBuffers& buffers, const AttackInfo& attacks,
                 double phase) {
    int score = 0;
    int whiteKing = pos.kingSquare(true);
    if (whiteKing >= 0) {
        std::size_t psq = static_cast<std::size_t>(whiteKing);
        score += javaRound(kKingPstOpening[psq] * phase + kKingPstEndgame[psq] * (1.0 - phase));
        score += sideKingSafetyCp(pos, buffers, attacks, true, whiteKing, phase);
    }
    int blackKing = pos.kingSquare(false);
    if (blackKing >= 0) {
        std::size_t psq = static_cast<std::size_t>(flip(blackKing));
        score -= javaRound(kKingPstOpening[psq] * phase + kKingPstEndgame[psq] * (1.0 - phase));
        score -= sideKingSafetyCp(pos, buffers, attacks, false, blackKing, phase);
    }
    return score;
}

// ---------------------------------------------------------------------------
// Term: activity (mobility + piece placement).
// ---------------------------------------------------------------------------

int activityCp(const AttackInfo& attacks, double phase) {
    int mg = attacks.mobilityMg[kWhite] + attacks.pieceMg[kWhite] - attacks.mobilityMg[kBlack] -
             attacks.pieceMg[kBlack];
    int eg = attacks.mobilityEg[kWhite] + attacks.pieceEg[kWhite] - attacks.mobilityEg[kBlack] -
             attacks.pieceEg[kBlack];
    return blend(mg, eg, phase);
}

// ---------------------------------------------------------------------------
// Term: threats.
// ---------------------------------------------------------------------------

int minorThreatMg(int type) {
    switch (type) {
        case kPawn: return 6;
        case kKnight:
        case kBishop: return 28;
        case kRook: return 44;
        case kQueen: return 58;
        default: return 0;
    }
}

int minorThreatEg(int type) {
    switch (type) {
        case kPawn: return 16;
        case kKnight:
        case kBishop: return 24;
        case kRook: return 38;
        case kQueen: return 70;
        default: return 0;
    }
}

int rookThreatMg(int type) {
    switch (type) {
        case kPawn: return 4;
        case kKnight:
        case kBishop: return 22;
        case kQueen: return 42;
        default: return 0;
    }
}

int rookThreatEg(int type) {
    switch (type) {
        case kPawn: return 28;
        case kKnight:
        case kBishop: return 34;
        case kRook: return 18;
        case kQueen: return 36;
        default: return 0;
    }
}

int sideThreatsCp(const JavaBoard& pos, const AttackInfo& attacks, bool white, double phase) {
    int us = sideIndex(white);
    int them = 1 - us;
    Bb enemies = pos.occupancy(!white);
    Bb nonPawnEnemies = enemies & ~pos.piecesOf(white ? kBlackPawn : kWhitePawn);
    Bb stronglyProtected =
        attacks.ab(them, kPawn) | (attacks.attackedBy2[static_cast<std::size_t>(them)] &
                                   ~attacks.attackedBy2[static_cast<std::size_t>(us)]);
    Bb weak = enemies & ~stronglyProtected & attacks.ab(us, kAllAttacks);

    int mg = 0;
    int eg = 0;
    Bb minorTargets = (weak | (nonPawnEnemies & stronglyProtected)) &
                      (attacks.ab(us, kKnight) | attacks.ab(us, kBishop));
    while (minorTargets != 0) {
        int square = trailingZeros(minorTargets);
        minorTargets &= minorTargets - 1;
        int type = std::abs(pos.pieceAt(square));
        mg += minorThreatMg(type);
        eg += minorThreatEg(type);
    }

    Bb rookTargets = weak & attacks.ab(us, kRook);
    while (rookTargets != 0) {
        int square = trailingZeros(rookTargets);
        rookTargets &= rookTargets - 1;
        int type = std::abs(pos.pieceAt(square));
        mg += rookThreatMg(type);
        eg += rookThreatEg(type);
    }

    Bb hanging = weak & (~attacks.ab(them, kAllAttacks) | attacks.attackedBy2[static_cast<std::size_t>(us)]);
    int hangingCount = bitCount(hanging);
    mg += hangingCount * 22;
    eg += hangingCount * 14;

    Bb restricted = attacks.ab(them, kAllAttacks) & ~stronglyProtected & attacks.ab(us, kAllAttacks);
    mg += bitCount(restricted) * 4;

    Bb safe = ~attacks.ab(them, kAllAttacks) | attacks.ab(us, kAllAttacks);
    Bb safePawnThreats = attacks.ab(us, kPawn) & nonPawnEnemies & safe;
    mg += bitCount(safePawnThreats) * 32;
    eg += bitCount(safePawnThreats) * 26;

    Bb pushedPawns = pawnPushMask(white, pos.piecesOf(white ? kWhitePawn : kBlackPawn), ~pos.occupancy());
    Bb pawnPushThreats =
        pawnAttackMask(white, pushedPawns) & nonPawnEnemies & ~attacks.ab(them, kPawn) & safe;
    mg += bitCount(pawnPushThreats) * 18;
    eg += bitCount(pawnPushThreats) * 16;

    Bb enemyQueens = pos.piecesOf(white ? kBlackQueen : kWhiteQueen);
    if (enemyQueens != 0) {
        int queen = trailingZeros(enemyQueens);
        Bb safeQueenAttackers = attacks.ab(us, kKnight) & knightAttacksJ(queen) & safe;
        mg += bitCount(safeQueenAttackers) * 16;
        eg += bitCount(safeQueenAttackers) * 10;
        Bb sliderAttackers =
            (attacks.ab(us, kBishop) & bishopAttacksJ(queen, pos.occupancy())) |
            (attacks.ab(us, kRook) & rookAttacksJ(queen, pos.occupancy()));
        int sliderPressure =
            bitCount(sliderAttackers & attacks.attackedBy2[static_cast<std::size_t>(us)] & safe);
        mg += sliderPressure * 18;
        eg += sliderPressure * 12;
    }
    return blend(mg, eg, phase);
}

int threatsCp(const JavaBoard& pos, const AttackInfo& attacks, double phase) {
    int white = sideThreatsCp(pos, attacks, true, phase);
    int black = sideThreatsCp(pos, attacks, false, phase);
    return white - black;
}

// ---------------------------------------------------------------------------
// Term: space.
// ---------------------------------------------------------------------------

int nonPawnMaterial(const EvalScan& scan) {
    return scan.whiteMaterial + scan.blackMaterial - (scan.whitePawns + scan.blackPawns) * kValuePawn;
}

Bb shiftBackward(bool white, Bb mask) {
    return white ? (mask << 8) : (mask >> 8);
}

int blockedPawnCount(const JavaBoard& pos, bool white) {
    Bb pawns = pos.piecesOf(white ? kWhitePawn : kBlackPawn);
    Bb occupied = pos.occupancy();
    Bb blocked = white ? ((pawns >> 8) & occupied) : ((pawns << 8) & occupied);
    return bitCount(blocked);
}

int sideSpaceCp(const JavaBoard& pos, const AttackInfo& attacks, bool white) {
    int them = sideIndex(!white);
    Bb pawns = pos.piecesOf(white ? kWhitePawn : kBlackPawn);
    Bb mask = white ? kWhiteSpaceMask : kBlackSpaceMask;
    Bb safe = mask & ~pawns & ~attacks.ab(them, kPawn);
    Bb behind = pawns;
    Bb oneBehind = shiftBackward(white, pawns);
    Bb twoBehind = shiftBackward(white, oneBehind);
    Bb threeBehind = shiftBackward(white, twoBehind);
    behind |= oneBehind | twoBehind | threeBehind;
    int bonus = bitCount(safe) + bitCount(behind & safe & ~attacks.ab(them, kAllAttacks));
    int blocked = blockedPawnCount(pos, white);
    int weight = std::max(0, bitCount(pos.occupancy(white)) - 3 + std::min(blocked, 9));
    return bonus * weight * weight / 12;
}

int spaceCp(const JavaBoard& pos, const AttackInfo& attacks, const EvalScan& scan, double phase) {
    if (phase < 0.45 || nonPawnMaterial(scan) < 2400) {
        return 0;
    }
    int white = sideSpaceCp(pos, attacks, true);
    int black = sideSpaceCp(pos, attacks, false);
    return javaRound((white - black) * phase);
}

// ---------------------------------------------------------------------------
// Term: tempo / check.
// ---------------------------------------------------------------------------

int tempoCp(const JavaBoard& pos) { return pos.isWhiteToMove() ? kTempoCp : -kTempoCp; }

int checkPenaltyCp(const JavaBoard& pos) {
    if (!pos.inCheck) {
        return 0;
    }
    return pos.isWhiteToMove() ? -kInCheckCp : kInCheckCp;
}

// ---------------------------------------------------------------------------
// WDL mapping.
// ---------------------------------------------------------------------------

double sigmoid(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

WdlTriplet fromProbabilities(double pWin, double pDraw, double pLoss) {
    pWin = clamp01(pWin);
    pDraw = clamp01(pDraw);
    pLoss = clamp01(pLoss);

    double sum = pWin + pDraw + pLoss;
    if (sum <= 0.0) {
        return WdlTriplet{0, kWdlTotal, 0};
    }
    pWin /= sum;
    pDraw /= sum;
    pLoss /= sum;

    int winBase = static_cast<int>(std::floor(pWin * kWdlTotal));
    int drawBase = static_cast<int>(std::floor(pDraw * kWdlTotal));
    int lossBase = static_cast<int>(std::floor(pLoss * kWdlTotal));

    double winFrac = (pWin * kWdlTotal) - winBase;
    double drawFrac = (pDraw * kWdlTotal) - drawBase;
    double lossFrac = (pLoss * kWdlTotal) - lossBase;

    int sumBase = winBase + drawBase + lossBase;
    int remainder = kWdlTotal - sumBase;

    int win = winBase;
    int draw = drawBase;
    int loss = lossBase;

    for (int i = 0; i < remainder; ++i) {
        if (winFrac >= drawFrac && winFrac >= lossFrac) {
            win++;
            winFrac = -1.0;
        } else if (drawFrac >= lossFrac) {
            draw++;
            drawFrac = -1.0;
        } else {
            loss++;
            lossFrac = -1.0;
        }
    }

    return WdlTriplet{win, draw, loss};
}

// ---------------------------------------------------------------------------
// Core white-perspective evaluation.
// ---------------------------------------------------------------------------

Breakdown computeBreakdown(const JavaBoard& pos) {
    EvalBuffers buffers;
    EvalScan& scan = buffers.scan;
    scanMaterialAndPst(pos, buffers, scan);
    double phase = updatePhase(scan, buffers);
    AttackInfo attacks;
    attacks.build(pos);

    Breakdown b;
    b.whiteToMove = pos.isWhiteToMove();
    b.phase = phase;
    b.material = scan.whiteMaterial - scan.blackMaterial;
    b.pieceSquare = scan.score;
    b.bishopPair = bishopPairCp(scan.whiteBishops, scan.blackBishops);
    b.pawnStructure = pawnStructureCp(pos, buffers.whitePawnsPerFile, buffers.blackPawnsPerFile,
                                      buffers.minBlackPawnRank, buffers.maxWhitePawnRank, attacks, phase);
    b.rookFile = rookFileCp(buffers.whiteRooksFileCount, buffers.blackRooksFileCount,
                            buffers.whitePawnsPerFile, buffers.blackPawnsPerFile);
    b.kingSafety = kingSafetyCp(pos, buffers, attacks, phase);
    b.activity = activityCp(attacks, phase);
    b.threats = threatsCp(pos, attacks, phase);
    b.space = spaceCp(pos, attacks, scan, phase);
    b.tempo = tempoCp(pos);
    b.checkPenalty = checkPenaltyCp(pos);
    return b;
}

// MinorMaterialState equivalent for insufficient-material detection.
bool isInsufficientMaterial(const JavaBoard& pos) {
    int whiteKnights = 0, whiteBishops = 0, blackKnights = 0, blackBishops = 0;
    int whiteBishopColor = -1, blackBishopColor = -1;
    for (int square = 0; square < 64; ++square) {
        int piece = pos.board[static_cast<std::size_t>(square)];
        if (piece == 0 || piece == kKing || piece == -kKing) {
            continue;
        }
        int type = std::abs(piece);
        bool white = piece > 0;
        if (type == kPawn || type == kRook || type == kQueen) {
            return false;
        }
        if (type == kKnight) {
            if (white) whiteKnights++;
            else blackKnights++;
            continue;
        }
        // bishop
        int color = ((square & 7) + (square >> 3)) & 1;
        if (white) {
            whiteBishops++;
            whiteBishopColor = color;
        } else {
            blackBishops++;
            blackBishopColor = color;
        }
    }
    int whiteMinors = whiteKnights + whiteBishops;
    int blackMinors = blackKnights + blackBishops;
    bool sameColorBishopDraw = whiteKnights == 0 && blackKnights == 0 && whiteBishops == 1 &&
                               blackBishops == 1 && whiteBishopColor == blackBishopColor;
    return (whiteMinors == 0 && blackMinors == 0) || (whiteMinors == 1 && blackMinors == 0) ||
           (whiteMinors == 0 && blackMinors == 1) || sameColorBishopDraw;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

Breakdown evaluateWhiteBreakdown(const Position& pos) {
    JavaBoard jb = snapshot(pos);
    return computeBreakdown(jb);
}

WdlTriplet evaluateWdl(const Position& pos, bool terminalAware) {
    if (terminalAware) {
        if (pos.isCheckmate()) {
            return WdlTriplet{0, 0, kWdlTotal};
        }
        if (pos.isStalemate()) {
            return WdlTriplet{0, kWdlTotal, 0};
        }
    }

    JavaBoard jb = snapshot(pos);
    if (isInsufficientMaterial(jb)) {
        return WdlTriplet{0, kWdlTotal, 0};
    }

    Breakdown b = computeBreakdown(jb);
    int whiteScoreCp = b.whiteTotal();
    int stmScoreCp = jb.isWhiteToMove() ? whiteScoreCp : -whiteScoreCp;

    double materialFactor = b.phase;
    double endgame = 1.0 - materialFactor;

    double margin = kDrawMarginCp * (1.0 + 0.40 * endgame);
    double scale = kScaleCp * (1.0 + 0.20 * endgame);

    double pWin = sigmoid((stmScoreCp - margin) / scale);
    double pLoss = sigmoid((-stmScoreCp - margin) / scale);

    pWin = clamp01(pWin);
    pLoss = clamp01(pLoss);
    double winLossSum = pWin + pLoss;
    if (winLossSum > 1.0) {
        double renorm = winLossSum != 0.0 ? (1.0 / winLossSum) : 0.0;
        pWin *= renorm;
        pLoss *= renorm;
    }
    double extraDraw = endgame * kEndgameDrawBonus;
    pWin *= (1.0 - extraDraw);
    pLoss *= (1.0 - extraDraw);
    double pDraw = 1.0 - pWin - pLoss;

    return fromProbabilities(pWin, pDraw, pLoss);
}

std::array<int, 64> pieceSquareTable(int pieceType) {
    switch (pieceType) {
        case 1: return kPawnPst;
        case 2: return kKnightPst;
        case 3: return kBishopPst;
        case 4: return kRookPst;
        case 5: return kQueenPst;
        case 6: return kKingPstOpening;
        default: return std::array<int, 64>{};
    }
}

std::array<float, 64> pieceSquareHeatmap(const Position& pos) {
    std::array<float, 64> heat{};
    for (int sq = 0; sq < 64; ++sq) {
        const Piece p = pos.pieceAt(sq);
        if (p.isNone()) continue;

        const std::array<int, 64> pst =
            pieceSquareTable(static_cast<int>(p.type));
        const auto [minIt, maxIt] = std::minmax_element(pst.begin(), pst.end());
        const int range = *maxIt - *minIt;
        if (range <= 0) continue;

        const int tableSq = p.color == Color::White ? (sq ^ 56) : sq;
        const float t = static_cast<float>(
            pst[static_cast<std::size_t>(tableSq)] - *minIt) /
            static_cast<float>(range);
        heat[static_cast<std::size_t>(sq)] = std::clamp(t, 0.0f, 1.0f);
    }
    return heat;
}

}  // namespace cnnv::chess::eval
