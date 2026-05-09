#pragma once

/**
 * @file MoveGenerator.h
 * @brief Legal move generation entry point.
 */

#include "chess/MoveList.h"
#include "chess/Position.h"

namespace cnnv::chess {

/**
 * @brief Stateless legal move generator for the current position.
 *
 * The generator builds pseudo-legal moves and filters them with make/unmake
 * plus an in-check test. That is sufficient for the visualizer; a search engine
 * would typically add pinned-piece tracking to avoid the make/unmake filter.
 */
class MoveGenerator {
public:
    /**
     * @brief Appends every legal move for `pos.sideToMove()` to `out`.
     * @param pos Position to generate from. It is restored before return.
     * @param out Destination list. Existing contents are preserved.
     */
    static void generateLegal(Position& pos, MoveList& out);
};

}  // namespace cnnv::chess
