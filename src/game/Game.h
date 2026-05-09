#pragma once

/**
 * @file Game.h
 * @brief High-level chess game controller with legal moves and undo/redo.
 */

#include "chess/Move.h"
#include "chess/MoveList.h"
#include "chess/Position.h"
#include "game/MoveHistory.h"

#include <string>

namespace cnnv::game {

/**
 * @brief User-facing terminal state of a chess game.
 */
enum class GameStatus {
    Ongoing,
    WhiteWins,
    BlackWins,
    DrawStalemate,
    DrawInsufficientMaterial,
    DrawFiftyMove,
    DrawThreefold,
};

/**
 * @brief Owns one chess game: position, move history, and status evaluation.
 *
 * All normal move-mutation paths go through this class so the `Position` and
 * `MoveHistory` stay synchronized for the board, move list, PGN export, and
 * undo/redo controls.
 */
class Game {
public:
    /**
     * @brief Creates a game at the standard starting position.
     */
    Game();

    /**
     * @brief Resets to the standard starting position and clears history.
     */
    void reset();

    /**
     * @brief Loads a new root position from FEN and clears history.
     * @param fen Six-field FEN string.
     * @return True on success; false if parsing failed.
     */
    bool loadFen(const std::string& fen);

    /**
     * @brief Validates and applies a move.
     * @param m Candidate move.
     * @return True when the move was legal and applied.
     *
     * On success, a `MoveRecord` containing SAN plus before/after FEN is
     * appended to history.
     */
    bool tryMove(cnnv::chess::Move m);

    /**
     * @brief Undoes the most recent move.
     * @return True if a move was undone.
     */
    bool undo();

    /**
     * @brief Redoes the next move when a forward branch exists.
     * @return True if a move was redone.
     */
    bool redo();

    /**
     * @brief Navigates to a ply in the linear move history.
     * @param targetPly Half-move index, where 0 is the root and history size
     * is the tip.
     * @return True on success; false if the target is out of range.
     */
    bool jumpToPly(std::size_t targetPly);

    /** @brief True when at least one move can be undone. */
    bool canUndo() const noexcept { return m_history.canUndo(); }

    /** @brief True when at least one move can be redone. */
    bool canRedo() const noexcept { return m_history.canRedo(); }

    /** @brief Number of half-moves currently applied. */
    std::size_t currentPly() const noexcept { return m_history.plyCount(); }

    /** @brief Current board position. */
    const cnnv::chess::Position& position() const noexcept { return m_position; }

    /** @brief Linear move history. */
    const MoveHistory& history() const noexcept { return m_history; }

    /**
     * @brief Appends legal moves from the current position.
     * @param out Destination list. Existing contents are preserved.
     *
     * Non-const because `MoveGenerator` temporarily makes/unmakes moves; the
     * position is restored before return.
     */
    void generateLegalMoves(cnnv::chess::MoveList& out);

    /**
     * @brief Computes the current game status.
     */
    GameStatus status() const;

    /**
     * @brief Formats a human-readable status string for the UI banner.
     */
    std::string statusText() const;

private:
    cnnv::chess::Position m_position;
    MoveHistory m_history;
};

}  // namespace cnnv::game
