#include "viz/Controls.h"

namespace cnnv::viz {

namespace {

constexpr float kButtonHeight = 36.0f;
constexpr float kButtonSpacing = 6.0f;
constexpr float kPanelPadding = 10.0f;

}  // namespace

void Controls::addButton(std::string label, Callback onPress) {
    m_buttons.push_back({std::move(label), std::move(onPress)});
}

Rectangle Controls::buttonRect(std::size_t index) const {
    const int cols = m_bounds.width >= 620.0f ? 5
                   : m_bounds.width >= 480.0f ? 3
                   : m_bounds.width >= 360.0f ? 2
                                              : 1;
    const int row = static_cast<int>(index) / cols;
    const int col = static_cast<int>(index) % cols;
    const float w = (m_bounds.width - 2.0f * kPanelPadding
                     - static_cast<float>(cols - 1) * kButtonSpacing)
                  / static_cast<float>(cols);
    float y = m_bounds.y + kPanelPadding +
              static_cast<float>(row) * (kButtonHeight + kButtonSpacing);
    return Rectangle{
        m_bounds.x + kPanelPadding +
            static_cast<float>(col) * (w + kButtonSpacing),
        y,
        w,
        kButtonHeight,
    };
}

void Controls::update() {
    Vector2 mp = GetMousePosition();
    m_hoverIndex = -1;
    for (std::size_t i = 0; i < m_buttons.size(); ++i) {
        if (CheckCollisionPointRec(mp, buttonRect(i))) {
            m_hoverIndex = static_cast<int>(i);
            break;
        }
    }

    if (m_hoverIndex >= 0 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        m_pressedIndex = m_hoverIndex;
    }
    if (m_pressedIndex >= 0 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        if (m_pressedIndex == m_hoverIndex
            && static_cast<std::size_t>(m_pressedIndex) < m_buttons.size()
            && m_buttons[m_pressedIndex].onPress) {
            m_buttons[m_pressedIndex].onPress();
        }
        m_pressedIndex = -1;
    }
}

void Controls::draw(const Theme& theme) const {
    for (std::size_t i = 0; i < m_buttons.size(); ++i) {
        Rectangle r = buttonRect(i);
        Color fill = theme.buttonIdle;
        if (static_cast<int>(i) == m_pressedIndex) fill = theme.buttonActive;
        else if (static_cast<int>(i) == m_hoverIndex) fill = theme.buttonHover;
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        const char* text = m_buttons[i].label.c_str();
        int fontSize = 17;
        int textWidth = measureText(theme, text, fontSize);
        int tx = static_cast<int>(r.x + (r.width - textWidth) / 2);
        int ty = static_cast<int>(r.y + (r.height - fontSize) / 2);
        drawText(theme, text, tx, ty, fontSize, theme.textPrimary);
    }
}

}  // namespace cnnv::viz
