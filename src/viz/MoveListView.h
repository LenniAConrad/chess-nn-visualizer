#pragma once

/**
 * @file MoveListView.h
 * @brief Clickable SAN move-history panel.
 */

#include "game/Game.h"
#include "viz/Theme.h"

#include <raylib.h>

#include <cstddef>
#include <optional>

namespace cnnv::viz {

/**
 * @brief Two-column SAN move list rendered into a fixed rectangle.
 *
 * The view reads directly from `Game::history()` and owns no chess state.
 * Clicking a SAN cell produces a target ply for `Game::jumpToPly()`: ply 0 is
 * the root, ply N is the position after the N-th half-move. The header strip
 * also acts as a reset-to-root click target.
 */
class MoveListView {
public:
    /** @brief Creates a view bound to a game. */
    explicit MoveListView(const cnnv::game::Game& game) : m_game(&game) {}

    /** @brief Sets view bounds. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /** @brief Returns view bounds. */
    Rectangle bounds() const noexcept { return m_bounds; }

    /**
     * @brief Processes mouse input and stores a pending target ply.
     */
    void update();

    /**
     * @brief Retrieves and clears the latest clicked target ply.
     */
    std::optional<std::size_t> takeClickedPly() noexcept {
        auto out = m_pendingClick;
        m_pendingClick.reset();
        return out;
    }

    /** @brief Draws the move list. */
    void draw(const Theme& theme) const;

private:
    struct CellLayout {
        Rectangle moveNumber;
        Rectangle white;
        Rectangle black;
        float rowTop;
        float rowHeight;
    };

    CellLayout rowLayout(int row) const;
    int        visibleRows() const;
    int        scrollOffsetClamped(int totalRows) const;

    const cnnv::game::Game* m_game;
    Rectangle m_bounds{0, 0, 0, 0};
    int m_scrollRows = 0;
    std::optional<std::size_t> m_pendingClick;
};

}  // namespace cnnv::viz
