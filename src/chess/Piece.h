#pragma once

/**
 * @file Piece.h
 * @brief Chess piece, color, and material-value helpers.
 */

#include <cstdint>

namespace cnnv::chess {

/**
 * @brief Player color.
 */
enum class Color : std::uint8_t {
    White = 0,
    Black = 1,
};

/**
 * @brief Returns the opposite side.
 * @param c Input side.
 * @return `Color::Black` for white and `Color::White` for black.
 */
constexpr Color other(Color c) noexcept {
    return c == Color::White ? Color::Black : Color::White;
}

/**
 * @brief Chess piece kind independent of color.
 */
enum class PieceType : std::uint8_t {
    None   = 0,
    Pawn   = 1,
    Knight = 2,
    Bishop = 3,
    Rook   = 4,
    Queen  = 5,
    King   = 6,
};

/**
 * @brief One color-qualified chess piece.
 *
 * Real pieces map to stable indices 0..11: white pieces occupy 0..5 and black
 * pieces occupy 6..11. `PieceType::None` is represented by `kNoPiece` and maps
 * to index 12 through `pieceIndex()`.
 */
struct Piece {
    Color color = Color::White;
    PieceType type = PieceType::None;

    /**
     * @brief Tests whether this object represents an empty square.
     * @return True when `type == PieceType::None`.
     */
    constexpr bool isNone() const noexcept { return type == PieceType::None; }

    constexpr bool operator==(const Piece& o) const noexcept {
        return color == o.color && type == o.type;
    }
    constexpr bool operator!=(const Piece& o) const noexcept { return !(*this == o); }
};

/**
 * @brief Sentinel piece used for empty squares and failed parses.
 */
constexpr Piece kNoPiece = Piece{Color::White, PieceType::None};

/**
 * @brief Maps a piece to its compact table index.
 * @param p Piece to encode.
 * @return 0..11 for real pieces, or 12 for `PieceType::None`.
 */
constexpr int pieceIndex(Piece p) noexcept {
    if (p.type == PieceType::None) return 12;
    return static_cast<int>(p.color) * 6 + (static_cast<int>(p.type) - 1);
}

/**
 * @brief Converts a piece to the standard FEN/SAN character.
 * @param p Piece to encode.
 * @return Uppercase white piece, lowercase black piece, or `'.'` for none.
 */
char pieceToFenChar(Piece p) noexcept;

/**
 * @brief Parses a FEN piece character.
 * @param c Character such as `P`, `n`, or `q`.
 * @return Parsed piece, or `kNoPiece` for unknown characters.
 */
Piece pieceFromFenChar(char c) noexcept;

/**
 * @brief Returns a simple centipawn material value for UI heuristics.
 * @param t Piece type to score.
 * @return Conventional material value; kings and empty squares return zero.
 *
 * Neural-network inference does not depend on these values.
 */
constexpr int centipawnValue(PieceType t) noexcept {
    switch (t) {
        case PieceType::Pawn:   return 100;
        case PieceType::Knight: return 300;
        case PieceType::Bishop: return 300;
        case PieceType::Rook:   return 500;
        case PieceType::Queen:  return 900;
        case PieceType::King:   return 0;
        case PieceType::None:   return 0;
    }
    return 0;
}

}  // namespace cnnv::chess
