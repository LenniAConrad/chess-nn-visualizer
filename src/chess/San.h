#pragma once

/**
 * @file San.h
 * @brief Standard Algebraic Notation conversion for legal chess moves.
 */

#include "chess/Move.h"
#include "chess/Position.h"

#include <string>

namespace cnnv::chess {

/**
 * @brief Static helpers for rendering and parsing SAN in a position context.
 */
class San {
public:
    /**
     * @brief Renders a move as Standard Algebraic Notation.
     * @param pos Position before the move.
     * @param m Legal move to render.
     * @return SAN text, including `+` or `#` for check/checkmate.
     */
    static std::string toSan(const Position& pos, Move m);

    /**
     * @brief Parses a SAN move in the context of a position.
     * @param pos Position before the move.
     * @param s SAN text to resolve.
     * @return Matching legal move, or `Move::none()` if not unique/valid.
     *
     * The parser ignores trailing `+`/`#` and tolerates capture markers and
     * `=Q`-style promotion suffixes.
     */
    static Move parse(const Position& pos, const std::string& s);
};

}  // namespace cnnv::chess
