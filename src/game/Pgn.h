#pragma once

/**
 * @file Pgn.h
 * @brief PGN export helpers for completed or in-progress games.
 */

#include "game/Game.h"

#include <string>

namespace cnnv::game {

/**
 * @brief Exports a game's move list to PGN text.
 * @param g Game whose history should be exported.
 * @param whiteName White player tag value.
 * @param blackName Black player tag value.
 * @return PGN document including a `Result` tag derived from `Game::status()`.
 *
 * The project currently supports PGN export only, not PGN parsing.
 */
std::string exportPgn(const Game& g, const std::string& whiteName = "Human",
                      const std::string& blackName = "Human");

/**
 * @brief Writes a game's PGN export to a file.
 * @param path Destination path.
 * @param g Game to export.
 * @param whiteName White player tag value.
 * @param blackName Black player tag value.
 * @return True on success; false on I/O error.
 */
bool writePgnFile(const std::string& path, const Game& g,
                  const std::string& whiteName = "Human",
                  const std::string& blackName = "Human");

}  // namespace cnnv::game
