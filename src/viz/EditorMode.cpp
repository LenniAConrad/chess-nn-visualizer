#include "viz/EditorMode.h"

namespace cnnv::viz {

void EditorMode::enter(const cnnv::chess::Position& seed) {
    m_editor = cnnv::chess::PositionEditor(seed);
    m_brush  = Brush{};  // eraser by default
    m_validation = {};
    m_validated = false;
    m_active = true;
    bump();
}

void EditorMode::setBrushPiece(cnnv::chess::Piece p) noexcept {
    m_brush.kind  = BrushKind::Piece;
    m_brush.piece = p;
}

void EditorMode::setBrushEraser() noexcept {
    m_brush.kind  = BrushKind::Eraser;
    m_brush.piece = cnnv::chess::Piece{};
}

void EditorMode::applyLeftClick(cnnv::chess::Square sq) {
    if (!m_active) return;
    if (m_brush.kind == BrushKind::Eraser || m_brush.piece.isNone()) {
        m_editor.removePiece(sq);
    } else {
        m_editor.placePiece(sq, m_brush.piece);
    }
    m_validated = false;
    bump();
}

void EditorMode::applyRightClick(cnnv::chess::Square sq) {
    if (!m_active) return;
    m_editor.removePiece(sq);
    m_validated = false;
    bump();
}

void EditorMode::validate() {
    m_validation = cnnv::chess::validateForEditor(m_editor.position());
    m_validated = true;
}

void EditorMode::bump() noexcept {
    ++m_revision;
}

}  // namespace cnnv::viz
