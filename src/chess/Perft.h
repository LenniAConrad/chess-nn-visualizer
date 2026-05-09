#pragma once

/**
 * @file Perft.h
 * @brief Move-generator correctness counter.
 */

#include "chess/Position.h"

#include <cstdint>

namespace cnnv::chess {

/**
 * @brief Counts leaf nodes in a legal-move make/unmake tree.
 * @param pos Position to search. It is restored before return.
 * @param depth Number of plies to expand.
 * @return Number of leaf nodes at `depth`.
 *
 * Perft is the canonical correctness test for chess move generators.
 */
std::uint64_t perft(Position& pos, int depth);

}  // namespace cnnv::chess
