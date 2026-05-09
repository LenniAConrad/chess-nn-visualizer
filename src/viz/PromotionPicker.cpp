#include "viz/PromotionPicker.h"

namespace cnnv::viz {

namespace {

constexpr cnnv::chess::PieceType kOptions[4] = {
    cnnv::chess::PieceType::Queen,
    cnnv::chess::PieceType::Rook,
    cnnv::chess::PieceType::Bishop,
    cnnv::chess::PieceType::Knight,
};

cnnv::chess::Move::Promotion promotionFor(cnnv::chess::PieceType t) {
    switch (t) {
        case cnnv::chess::PieceType::Queen:  return cnnv::chess::Move::Promotion::Queen;
        case cnnv::chess::PieceType::Rook:   return cnnv::chess::Move::Promotion::Rook;
        case cnnv::chess::PieceType::Bishop: return cnnv::chess::Move::Promotion::Bishop;
        case cnnv::chess::PieceType::Knight: return cnnv::chess::Move::Promotion::Knight;
        default: return cnnv::chess::Move::Promotion::None;
    }
}

}  // namespace

PromotionPicker::PromotionPicker(const PieceSprites& sprites)
    : m_sprites(&sprites) {}

void PromotionPicker::open(cnnv::chess::Move pendingMove,
                           cnnv::chess::Color color,
                           Rectangle anchor) {
    m_pending = pendingMove;
    m_color = color;
    m_anchor = anchor;
    m_active = true;
}

Rectangle PromotionPicker::iconRect(int index) const {
    float side = m_anchor.width;
    return Rectangle{
        m_anchor.x + index * side,
        m_anchor.y,
        side, side,
    };
}

std::optional<cnnv::chess::Move> PromotionPicker::update() {
    if (!m_active) return std::nullopt;
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return std::nullopt;
    Vector2 mp = GetMousePosition();
    for (int i = 0; i < 4; ++i) {
        if (CheckCollisionPointRec(mp, iconRect(i))) {
            cnnv::chess::Move chosen(m_pending.from(), m_pending.to(),
                                     promotionFor(kOptions[i]));
            m_active = false;
            return chosen;
        }
    }
    // Click outside cancels.
    m_active = false;
    return std::nullopt;
}

void PromotionPicker::draw(const Theme& theme) const {
    if (!m_active) return;
    // Backdrop covering the four-piece strip.
    Rectangle backdrop{
        m_anchor.x, m_anchor.y,
        m_anchor.width * 4, m_anchor.height,
    };
    DrawRectangleRec(backdrop, theme.panelBackground);
    DrawRectangleLinesEx(backdrop, 2.0f, theme.panelBorder);

    for (int i = 0; i < 4; ++i) {
        Rectangle r = iconRect(i);
        cnnv::chess::Piece p{m_color, kOptions[i]};
        const Texture2D& tex = m_sprites->textureFor(p);
        if (tex.id == 0) continue;
        Rectangle src{0, 0,
            static_cast<float>(tex.width),
            static_cast<float>(tex.height)};
        DrawTexturePro(tex, src, r, Vector2{0, 0}, 0.0f, WHITE);
        Vector2 mp = GetMousePosition();
        if (CheckCollisionPointRec(mp, r)) {
            DrawRectangleLinesEx(r, 3.0f, theme.selection);
        }
    }
}

}  // namespace cnnv::viz
