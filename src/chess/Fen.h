#pragma once

/**
 * @file Fen.h
 * @brief Standard Forsyth-Edwards Notation parsing and formatting.
 */

#include "chess/Position.h"

#include <optional>
#include <string>

namespace cnnv::chess {

/**
 * @brief Static helpers for converting between `Position` and six-field FEN.
 */
class Fen {
public:
    /**
     * @brief Standard chess starting position in six-field FEN form.
     */
    static constexpr const char* kStartpos =
        "rnbqkbnr/pppppppp/8/8/8/8/8/PPPPPPPP w KQkq - 0 1";

    /**
     * @brief Parses a standard six-field FEN string.
     * @param fen Text containing piece placement, side, castling, en-passant,
     * halfmove clock, and fullmove number.
     * @return Fully populated position, or `std::nullopt` when malformed.
     */
    static std::optional<Position> parse(const std::string& fen);

    /**
     * @brief Formats a position as a standard six-field FEN string.
     * @param pos Position to serialize.
     * @return FEN representation of `pos`.
     */
    static std::string format(const Position& pos);
};

}  // namespace cnnv::chess
