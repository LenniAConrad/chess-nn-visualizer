#pragma once

/**
 * @file PromotionPicker.h
 * @brief Modal promotion-choice widget for pawn moves.
 */

#include "chess/Move.h"
#include "chess/Piece.h"
#include "viz/PieceSprites.h"
#include "viz/Theme.h"

#include <raylib.h>

#include <optional>

namespace cnnv::viz {

/**
 * @brief Modal chooser shown when a pawn reaches the back rank.
 *
 * Renders the four promotion piece icons, captures clicks, and returns the
 * completed move once the user chooses a piece.
 */
class PromotionPicker {
public:
    /** @brief Creates a picker using the shared piece sprites. */
    PromotionPicker(const PieceSprites& sprites);

    /**
     * @brief Opens the picker for a pending pawn move.
     * @param pendingMove Move without promotion bits set.
     * @param color Color of the promoting pawn.
     * @param anchor Board square rectangle used for popup placement.
     */
    void open(cnnv::chess::Move pendingMove, cnnv::chess::Color color, Rectangle anchor);

    /** @brief Closes the picker without selecting a promotion. */
    void close() noexcept { m_active = false; }

    /** @brief True while the modal is visible. */
    bool active() const noexcept { return m_active; }

    /**
     * @brief Processes one input frame.
     * @return Completed promotion move exactly once after a piece is selected.
     */
    std::optional<cnnv::chess::Move> update();

    /** @brief Draws the promotion picker. */
    void draw(const Theme& theme = defaultTheme()) const;

private:
    Rectangle iconRect(int index) const;

    const PieceSprites* m_sprites;
    bool m_active = false;
    cnnv::chess::Move m_pending;
    cnnv::chess::Color m_color = cnnv::chess::Color::White;
    Rectangle m_anchor{0, 0, 0, 0};
};

}  // namespace cnnv::viz
