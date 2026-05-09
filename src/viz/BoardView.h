#pragma once

/**
 * @file BoardView.h
 * @brief Chessboard rendering, hit-testing, and board overlay state.
 */

#include "chess/Bitboard.h"
#include "chess/Position.h"
#include "viz/PieceSprites.h"
#include "viz/Theme.h"

#include <raylib.h>

#include <optional>

namespace cnnv::viz {

/**
 * @brief Renders one chessboard inside a fixed rectangle.
 *
 * The live chess state is owned elsewhere and referenced through a `Position`
 * pointer. `BoardView` stores only presentation state: orientation, selection,
 * legal-target highlights, last move, activation overlay, and drag indicator.
 */
class BoardView {
public:
    /**
     * @brief Creates a board view for a position and sprite atlas.
     */
    BoardView(const cnnv::chess::Position& position,
              const PieceSprites& sprites,
              Rectangle bounds);

    /**
     * @brief Re-aims the view at a different position.
     */
    void setPosition(const cnnv::chess::Position& position) noexcept {
        m_position = &position;
    }

    /** @brief Sets the board's layout rectangle. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /** @brief Returns the board's layout rectangle. */
    Rectangle bounds() const noexcept { return m_bounds; }

    /** @brief Sets whether the board is rendered from black's perspective. */
    void setFlipped(bool flipped) noexcept { m_flipped = flipped; }

    /** @brief Returns whether the board is flipped. */
    bool flipped() const noexcept { return m_flipped; }

    /**
     * @brief Converts a raylib screen coordinate to a board square.
     * @param pixel Screen-space cursor or touch position.
     * @return Square under the pixel, or `std::nullopt` outside the board.
     */
    std::optional<cnnv::chess::Square> squareAtPixel(Vector2 pixel) const;

    /**
     * @brief Renders board, pieces, highlights, and coordinates.
     */
    void draw(const Theme& theme = defaultTheme()) const;

    /** @name Highlight Overlays */
    ///@{
    /** @brief Sets the selected square highlight. */
    void setSelection(std::optional<cnnv::chess::Square> sq) noexcept { m_selection = sq; }

    /** @brief Sets legal move destination highlights. */
    void setLegalDestinations(cnnv::chess::Bitboard bb) noexcept { m_legalTargets = bb; }

    /** @brief Sets last-move origin and destination highlights. */
    void setLastMove(std::optional<cnnv::chess::Move> m) noexcept { m_lastMove = m; }

    /**
     * @brief Sets a per-square neural activation heatmap.
     * @param intensities Array indexed by raw square, expected in range [0, 1].
     */
    void setActivationOverlay(const std::array<float, 64>& intensities) noexcept;

    /** @brief Sets activation overlay opacity. */
    void setActivationOverlayOpacity(float opacity) noexcept;

    /** @brief Clears the activation heatmap. */
    void clearActivationOverlay() noexcept;

    /** @brief Dims pieces not participating in the active overlay. */
    void setDimInactivePieces(bool enabled) noexcept { m_dimInactivePieces = enabled; }

    /** @brief Dims squares not participating in the active overlay. */
    void setDimInactiveSquares(bool enabled) noexcept { m_dimInactiveSquares = enabled; }
    ///@}

    /**
     * @brief Draws a dragged piece centered at the cursor.
     * @param from Origin square whose normal piece rendering is suppressed.
     * @param cursor Screen-space cursor position.
     */
    void setDragging(cnnv::chess::Square from, Vector2 cursor) noexcept;

    /**
     * @brief Clears drag rendering state.
     */
    void clearDragging() noexcept { m_dragFrom.reset(); }

    /**
     * @brief Returns the pixel size of one square at the current bounds.
     */
    float squareSize() const;

private:
    Rectangle squareRect(int file, int rank) const;
    Vector2 boardOriginAndSquareSize(float& outSquareSize) const;

    const cnnv::chess::Position* m_position;
    const PieceSprites* m_sprites;
    Rectangle m_bounds;
    bool m_flipped = false;

    std::optional<cnnv::chess::Square> m_selection;
    cnnv::chess::Bitboard m_legalTargets = 0;
    std::optional<cnnv::chess::Move> m_lastMove;

    bool m_overlayActive = false;
    bool m_previousOverlayActive = false;
    float m_overlayOpacity = 1.0f;
    std::array<float, 64> m_overlay{};
    std::array<float, 64> m_previousOverlay{};
    bool m_dimInactivePieces = false;
    bool m_dimInactiveSquares = false;

    std::optional<cnnv::chess::Square> m_dragFrom;
    Vector2 m_dragCursor{0.0f, 0.0f};
};

}  // namespace cnnv::viz
