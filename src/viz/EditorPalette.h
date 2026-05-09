#pragma once

/**
 * @file EditorPalette.h
 * @brief Piece and action palette for board setup mode.
 */

#include "viz/EditorMode.h"
#include "viz/PieceSprites.h"
#include "viz/Theme.h"

#include <raylib.h>

namespace cnnv::viz {

/**
 * @brief Piece picker shown next to the board during setup mode.
 *
 * Renders twelve piece icons in two rows plus Eraser, Clear, and Startpos
 * actions. The palette mutates shared `EditorMode` by changing the brush or
 * issuing direct board edits.
 */
class EditorPalette {
public:
    /**
     * @brief Creates a palette bound to editor state and piece sprites.
     */
    explicit EditorPalette(EditorMode& mode, const PieceSprites& sprites)
        : m_mode(&mode), m_sprites(&sprites) {}

    /** @brief Sets the palette bounds. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /** @brief Returns the palette bounds. */
    Rectangle bounds() const noexcept { return m_bounds; }

    /**
     * @brief Preferred vertical space for layout.
     */
    static float recommendedHeight() noexcept;

    /** @brief Processes mouse input for one frame. */
    void update();

    /** @brief Draws the palette. */
    void draw(const Theme& theme) const;

private:
    Rectangle pieceCell(int col, int row) const;
    Rectangle eraserButton() const;
    Rectangle clearButton() const;
    Rectangle startposButton() const;

    EditorMode* m_mode;
    const PieceSprites* m_sprites;
    Rectangle m_bounds{0, 0, 0, 0};
};

}  // namespace cnnv::viz
