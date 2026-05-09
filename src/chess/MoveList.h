#pragma once

/**
 * @file MoveList.h
 * @brief Fixed-capacity legal move list used by the chess core.
 */

#include "chess/Move.h"

#include <array>
#include <cstddef>

namespace cnnv::chess {

/**
 * @brief Heap-free fixed-capacity list of chess moves.
 *
 * Capacity 256 is above the known maximum number of legal moves in a chess
 * position. The type is intentionally simple so move generation can run
 * without allocations in the hot path.
 */
class MoveList {
public:
    /**
     * @brief Maximum number of moves stored by the list.
     */
    static constexpr std::size_t kCapacity = 256;

    MoveList() = default;

    /**
     * @brief Appends a move.
     * @param m Move to append.
     *
     * The caller must ensure capacity is not exceeded.
     */
    void push(Move m) noexcept { m_moves[m_size++] = m; }

    /**
     * @brief Removes all moves without clearing underlying storage.
     */
    void clear() noexcept { m_size = 0; }

    /**
     * @brief Number of moves currently stored.
     */
    std::size_t size() const noexcept { return m_size; }

    /**
     * @brief Tests whether the list has no moves.
     */
    bool empty() const noexcept { return m_size == 0; }

    /**
     * @brief Reads a move by index.
     * @param i Index in the range 0..size()-1.
     */
    Move operator[](std::size_t i) const noexcept { return m_moves[i]; }

    /**
     * @brief Mutably accesses a move by index.
     * @param i Index in the range 0..size()-1.
     */
    Move& operator[](std::size_t i) noexcept { return m_moves[i]; }

    Move* begin() noexcept { return m_moves.data(); }
    Move* end() noexcept { return m_moves.data() + m_size; }
    const Move* begin() const noexcept { return m_moves.data(); }
    const Move* end() const noexcept { return m_moves.data() + m_size; }

    /**
     * @brief Removes one move in O(1) by replacing it with the last move.
     * @param i Index to remove.
     *
     * Order is not preserved.
     */
    void swapErase(std::size_t i) noexcept {
        --m_size;
        if (i != m_size) m_moves[i] = m_moves[m_size];
    }

private:
    std::array<Move, kCapacity> m_moves{};
    std::size_t m_size = 0;
};

}  // namespace cnnv::chess
