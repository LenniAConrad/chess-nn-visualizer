#pragma once

/**
 * @file EditorMode.h
 * @brief Shared mutable state for the board setup/editor workflow.
 */

#include "chess/Piece.h"
#include "chess/PositionEditor.h"

#include <optional>

namespace cnnv::viz {

/**
 * @brief Shared state for the position-editor experience.
 *
 * Owns the in-progress `PositionEditor` plus the selected brush that controls
 * what left-click places. `EditorPalette` mutates the brush, `EditorPanel`
 * mutates side/castling/counter fields, and `BoardView` is re-pointed at the
 * editable position while the mode is active.
 */
class EditorMode {
public:
    /** @brief Type of square edit performed by a left click. */
    enum class BrushKind { Eraser, Piece };

    /** @brief Current editor brush. */
    struct Brush {
        BrushKind kind = BrushKind::Eraser;
        cnnv::chess::Piece piece{};
    };

    EditorMode() = default;

    /** @brief True while setup mode is active. */
    bool active() const noexcept { return m_active; }

    /**
     * @brief Enters editor mode seeded from a position.
     *
     * Resets the brush to eraser to avoid placing stale pieces unexpectedly.
     */
    void enter(const cnnv::chess::Position& seed);

    /** @brief Leaves editor mode without committing. */
    void leave() noexcept { m_active = false; }

    /** @brief Mutable editor model for UI controls. */
    cnnv::chess::PositionEditor& editor() noexcept { return m_editor; }

    /** @brief Const editor model for read-only views. */
    const cnnv::chess::PositionEditor& editor() const noexcept { return m_editor; }

    /** @brief Position currently being edited. */
    const cnnv::chess::Position& position() const noexcept {
        return m_editor.position();
    }

    /** @name Brush State */
    ///@{
    /** @brief Current brush. */
    Brush brush() const noexcept { return m_brush; }

    /** @brief Selects a piece-placement brush. */
    void  setBrushPiece(cnnv::chess::Piece p) noexcept;

    /** @brief Selects the eraser brush. */
    void  setBrushEraser() noexcept;
    ///@}

    /**
     * @brief Applies the active brush to a square.
     */
    void applyLeftClick(cnnv::chess::Square sq);

    /**
     * @brief Erases a square regardless of the active brush.
     */
    void applyRightClick(cnnv::chess::Square sq);

    /**
     * @brief Re-runs editor legality validation and stores the result.
     */
    void validate();

    /** @brief Last validation result. */
    const cnnv::chess::EditorValidation& lastValidation() const noexcept {
        return m_validation;
    }

    /** @brief True once validation has been requested at least once. */
    bool hasBeenValidated() const noexcept { return m_validated; }

    /**
     * @brief Monotonic revision bumped after each edit.
     */
    std::uint64_t revision() const noexcept { return m_revision; }

private:
    void bump() noexcept;

    bool m_active = false;
    cnnv::chess::PositionEditor m_editor;
    Brush m_brush;
    cnnv::chess::EditorValidation m_validation;
    bool m_validated = false;
    std::uint64_t m_revision = 0;
};

}  // namespace cnnv::viz
