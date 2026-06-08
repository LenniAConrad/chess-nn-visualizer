#pragma once

/**
 * @file Bt4BinLoader.h
 * @brief Loader for the CRTK BT4J v2 binary weight format.
 */

#include "nn/lc0_bt4/Bt4RealWeights.h"

#include <string>

namespace cnnv::nn::lc0_bt4 {

/**
 * @brief Loads a real BT4 network from a CRTK BT4J v2 binary file.
 * @param path Path to the `.crtkbin`/`BT4J` weights file.
 * @param out Destination weight bundle, populated on success.
 * @param error Receives a human-readable message on failure.
 * @return True on success; false on any parse error (with `error` set).
 *
 * Validates the `BT4J` magic and version 2, parses the full v2 layout
 * (architecture, input stack, encoder blocks, shared smolgen projection,
 * policy head, value head), and fails if trailing bytes remain. The format is
 * little-endian; this reader matches the byte order of chess-rtk's
 * `BinLoader`.
 */
bool loadBt4Real(const std::string& path, Bt4RealWeights& out,
                 std::string& error);

}  // namespace cnnv::nn::lc0_bt4
