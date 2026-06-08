#pragma once

/**
 * @file PolicyEncoder.h
 * @brief LC0 73-plane move->policy-index encoder (transpiled from chess-rtk's
 *        chess.nn.lc0.cnn.PolicyEncoder).
 *
 * Policy planes are indexed by `planeIndex * 64 + fromSquare`, where fromSquare
 * uses a1=0..h8=63 ordering and the move is encoded from the side-to-move
 * perspective (black is rank-mirrored). This is the same raw index space the
 * CNN weight file's policyMap entries live in, so it lets a chess move be mapped
 * to its slot in the compressed policy logits.
 *
 * Plane layout (AlphaZero/LC0 classical): 0..55 queen-like (8 dirs x 7 dist),
 * 56..63 knight, 64..72 underpromotions (N,B,R) x (fwd, fwd-left, fwd-right).
 */

#include "chess/Move.h"
#include "chess/Position.h"

namespace cnnv::nn::lc0_cnn {

/** @brief Number of policy planes in the uncompressed LC0 encoding. */
constexpr int kPolicyPlanes = 73;

/** @brief Number of raw policy logits (planes x 64). */
constexpr int kRawPolicySize = kPolicyPlanes * 64;

/**
 * @brief Raw LC0 policy index for a move, or -1 if it cannot be encoded.
 * @param pos Position the move is played from (side-to-move matters).
 * @param move Move to encode.
 */
int rawPolicyIndex(const cnnv::chess::Position& pos, cnnv::chess::Move move);

}  // namespace cnnv::nn::lc0_cnn
