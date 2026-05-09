#pragma once

/**
 * @file MoveHistory.h
 * @brief Linked-list move history used for undo, redo, and move-list display.
 */

#include "chess/Move.h"

#include <cstddef>
#include <string>

namespace cnnv::game {

/**
 * @brief One node in the linear move history.
 *
 * Stores enough metadata to render SAN, jump to any prior position via FEN,
 * and re-apply a move on redo.
 */
struct MoveRecord {
    /** @brief Encoded move that was played. */
    cnnv::chess::Move move;

    /** @brief FEN before the move was applied. */
    std::string fenBefore;

    /** @brief FEN after the move was applied. */
    std::string fenAfter;

    /** @brief SAN text for display and PGN export. */
    std::string san;

    /** @brief Previous node in the line, or null for the first move. */
    MoveRecord* prev = nullptr;

    /** @brief Next node in the line, or null at the current tip. */
    MoveRecord* next = nullptr;
};

/**
 * @brief Hand-rolled doubly linked list of `MoveRecord` nodes.
 *
 * Conceptual model:
 * - The list represents one line of moves from the root position.
 * - `current()` points to the most recently applied move, or null at the root.
 * - `undo()` moves `current` one node toward the root.
 * - `redo()` moves `current` one node toward the tip when a forward node exists.
 * - `pushMove()` appends after `current`, truncating any forward branch first.
 */
class MoveHistory {
public:
    MoveHistory();
    ~MoveHistory();

    MoveHistory(const MoveHistory&) = delete;
    MoveHistory& operator=(const MoveHistory&) = delete;

    /**
     * @brief Deletes all nodes and returns to the root state.
     */
    void clear();

    /**
     * @brief Appends a move record after the current node.
     * @param rec Record to take by value and store as a new node.
     *
     * Any forward nodes after `current` are deleted; branching is not retained.
     */
    void pushMove(MoveRecord rec);

    /** @brief True when `undo()` can move back one ply. */
    bool canUndo() const noexcept { return m_current != nullptr; }

    /** @brief True when `redo()` can move forward one ply. */
    bool canRedo() const noexcept {
        return m_current ? m_current->next != nullptr : m_head != nullptr;
    }

    /**
     * @brief Pointer to the most recently applied move.
     * @return Current node, or null at the root.
     */
    const MoveRecord* current() const noexcept { return m_current; }

    /**
     * @brief Steps `current` one node back.
     * @return Record that was current before the step, or null when nothing
     * could be undone.
     */
    const MoveRecord* undo() noexcept;

    /**
     * @brief Steps `current` one node forward.
     * @return Record that becomes current, or null when no redo node exists.
     */
    const MoveRecord* redo() noexcept;

    /**
     * @brief First node in the line.
     * @return Head node, or null for an empty history.
     */
    const MoveRecord* head() const noexcept { return m_head; }

    /**
     * @brief Counts total nodes from head to tail.
     */
    std::size_t size() const noexcept;

    /**
     * @brief Counts how many half-moves are currently applied.
     * @return 0 at the root, otherwise number of nodes through `current`.
     */
    std::size_t plyCount() const noexcept;

private:
    void truncateAfter(MoveRecord* node) noexcept;

    MoveRecord* m_head = nullptr;
    MoveRecord* m_current = nullptr;
};

}  // namespace cnnv::game
