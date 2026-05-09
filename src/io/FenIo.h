#pragma once

/**
 * @file FenIo.h
 * @brief File helpers for loading and saving single-position FEN documents.
 */

#include "chess/Position.h"

#include <optional>
#include <string>

namespace cnnv::io {

/**
 * @brief Loads a chess position from a text file containing FEN.
 * @param path Source path.
 * @return Parsed position, or `std::nullopt` when the file cannot be opened or
 * the FEN is malformed.
 *
 * The first non-empty line is parsed; subsequent lines are ignored.
 */
std::optional<cnnv::chess::Position> loadPositionFromFenFile(const std::string& path);

/**
 * @brief Saves a position as FEN text.
 * @param path Destination path, overwritten if it exists.
 * @param pos Position to serialize.
 * @return True on success; false on I/O error.
 */
bool savePositionToFenFile(const std::string& path,
                           const cnnv::chess::Position& pos);

}  // namespace cnnv::io
