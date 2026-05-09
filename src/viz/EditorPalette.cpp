#include "viz/EditorPalette.h"

namespace cnnv::viz {

namespace {

constexpr float kPad        = 10.0f;
constexpr float kCellSize   = 62.0f;
constexpr float kCellGap    = 6.0f;
constexpr float kHeader     = 32.0f;
constexpr float kButtonH    = 38.0f;
constexpr int   kCols       = 6;

constexpr cnnv::chess::PieceType kRow[6] = {
    cnnv::chess::PieceType::Pawn,
    cnnv::chess::PieceType::Knight,
    cnnv::chess::PieceType::Bishop,
    cnnv::chess::PieceType::Rook,
    cnnv::chess::PieceType::Queen,
    cnnv::chess::PieceType::King,
};

}  // namespace

float EditorPalette::recommendedHeight() noexcept {
    return kHeader
         + 2 * kCellSize + kCellGap   // two piece rows
         + kPad
         + kButtonH
         + 2 * kPad;
}

Rectangle EditorPalette::pieceCell(int col, int row) const {
    float gridW = kCols * kCellSize + (kCols - 1) * kCellGap;
    float ox = m_bounds.x + (m_bounds.width - gridW) * 0.5f;
    float oy = m_bounds.y + kHeader;
    return Rectangle{
        ox + col * (kCellSize + kCellGap),
        oy + row * (kCellSize + kCellGap),
        kCellSize, kCellSize,
    };
}

Rectangle EditorPalette::eraserButton() const {
    float gridW = kCols * kCellSize + (kCols - 1) * kCellGap;
    float ox = m_bounds.x + (m_bounds.width - gridW) * 0.5f;
    float oy = m_bounds.y + kHeader + 2 * kCellSize + kCellGap + kPad;
    float w = (gridW - 2 * kCellGap) / 3.0f;
    return Rectangle{ox, oy, w, kButtonH};
}

Rectangle EditorPalette::clearButton() const {
    Rectangle e = eraserButton();
    return Rectangle{e.x + e.width + kCellGap, e.y, e.width, e.height};
}

Rectangle EditorPalette::startposButton() const {
    Rectangle c = clearButton();
    return Rectangle{c.x + c.width + kCellGap, c.y, c.width, c.height};
}

void EditorPalette::update() {
    if (!m_mode || !m_mode->active()) return;
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return;
    Vector2 mp = GetMousePosition();

    for (int row = 0; row < 2; ++row) {
        cnnv::chess::Color c = (row == 0) ? cnnv::chess::Color::White
                                          : cnnv::chess::Color::Black;
        for (int col = 0; col < kCols; ++col) {
            if (CheckCollisionPointRec(mp, pieceCell(col, row))) {
                m_mode->setBrushPiece(cnnv::chess::Piece{c, kRow[col]});
                return;
            }
        }
    }
    if (CheckCollisionPointRec(mp, eraserButton())) {
        m_mode->setBrushEraser();
        return;
    }
    if (CheckCollisionPointRec(mp, clearButton())) {
        m_mode->editor().clearBoard();
        m_mode->validate();
        return;
    }
    if (CheckCollisionPointRec(mp, startposButton())) {
        m_mode->editor().resetToStartpos();
        m_mode->validate();
        return;
    }
}

void EditorPalette::draw(const Theme& theme) const {
    if (!m_mode || !m_mode->active()) return;

    drawText(theme, "Piece palette",
             static_cast<int>(m_bounds.x + kPad),
             static_cast<int>(m_bounds.y + 6),
             20, theme.textPrimary);

    EditorMode::Brush brush = m_mode->brush();
    Vector2 mp = GetMousePosition();

    for (int row = 0; row < 2; ++row) {
        cnnv::chess::Color c = (row == 0) ? cnnv::chess::Color::White
                                          : cnnv::chess::Color::Black;
        for (int col = 0; col < kCols; ++col) {
            Rectangle r = pieceCell(col, row);
            bool selected = brush.kind == EditorMode::BrushKind::Piece
                         && brush.piece.color == c
                         && brush.piece.type  == kRow[col];
            bool hovered = CheckCollisionPointRec(mp, r);
            Color fill = selected ? theme.buttonActive
                       : hovered  ? theme.buttonHover
                                  : theme.buttonIdle;
            DrawRectangleRec(r, fill);
            DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
            const Texture2D& tex = m_sprites->textureFor(
                cnnv::chess::Piece{c, kRow[col]});
            if (tex.id != 0) {
                Rectangle src{0, 0,
                    static_cast<float>(tex.width),
                    static_cast<float>(tex.height)};
                DrawTexturePro(tex, src, r, Vector2{0, 0}, 0.0f, WHITE);
            }
        }
    }

    auto drawActionButton = [&](Rectangle r, const char* label,
                                bool active, bool emphatic) {
        Color fill = active   ? theme.buttonActive
                   : emphatic ? theme.buttonHover
                              : theme.buttonIdle;
        if (CheckCollisionPointRec(mp, r) && !active) fill = theme.buttonHover;
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        int fontSize = 17;
        int w = measureText(theme, label, fontSize);
        drawText(theme, label,
                 static_cast<int>(r.x + (r.width - w) / 2),
                 static_cast<int>(r.y + (r.height - static_cast<float>(fontSize)) / 2),
                 fontSize, theme.textPrimary);
    };
    bool eraserActive = brush.kind == EditorMode::BrushKind::Eraser;
    drawActionButton(eraserButton(),   "Eraser",   eraserActive, false);
    drawActionButton(clearButton(),    "Clear",    false,        false);
    drawActionButton(startposButton(), "Startpos", false,        false);
}

}  // namespace cnnv::viz
