#pragma once

/**
 * @file Bt4RealForward.h
 * @brief Native CPU forward pass for the real CRTK BT4 transformer.
 */

#include "nn/lc0_bt4/Bt4RealWeights.h"

namespace cnnv::chess {
class Position;
}

namespace cnnv::nn {
class ActivationSnapshot;
}

namespace cnnv::nn::lc0_bt4 {

/**
 * @brief Runs the real BT4 CPU forward pass and captures activations.
 * @param w Parsed BT4 weight bundle.
 * @param pos Position to evaluate (batch size 1).
 * @param out Snapshot destination, filled with the BT4 view's stable keys.
 *
 * Ports chess-rtk's batch-1 CPU path: the BT4_CANONICAL_112 input encoder, the
 * full input embedding stack (preproc / embedding / LN / gates / FFN), the
 * encoder blocks with smolgen attention bias, the attention policy head gathered
 * to LC0's 1858 logits, and the WDL value head. Per-block attention is stored as
 * `[heads, 64, 64]` post-softmax. WDL is ordered win, draw, loss; the value
 * scalar is `win - loss`.
 */
void evaluateBt4Real(const Bt4RealWeights& w, const cnnv::chess::Position& pos,
                     cnnv::nn::ActivationSnapshot& out);

}  // namespace cnnv::nn::lc0_bt4
