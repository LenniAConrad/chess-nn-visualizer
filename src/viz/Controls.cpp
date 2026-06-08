#include "viz/Controls.h"

namespace cnnv::viz {

namespace {

constexpr float kButtonHeight = 36.0f;
constexpr float kButtonSpacing = 6.0f;
constexpr float kPanelPadding = 10.0f;

Color blendColor(Color a, Color b, float t) noexcept {
    const auto mix = [t](unsigned char av, unsigned char bv) {
        return static_cast<unsigned char>(static_cast<float>(av)
            + (static_cast<float>(bv) - static_cast<float>(av)) * t);
    };
    return Color{mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a)};
}

}  // namespace

void Controls::addButton(std::string label, Callback onPress,
                         ActivePredicate isActive) {
    m_buttons.push_back({std::move(label), std::move(onPress),
                         std::move(isActive)});
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
        const bool latched =
            m_buttons[i].isActive && m_buttons[i].isActive();
        const bool hovered = static_cast<int>(i) == m_hoverIndex;
        const bool pressed = static_cast<int>(i) == m_pressedIndex;
        Color fill = theme.buttonIdle;
        if (latched) {
            fill = blendColor(hovered ? theme.buttonHover : theme.buttonActive,
                              theme.accentGreen, 0.42f);
        } else if (pressed) {
            fill = theme.buttonActive;
        } else if (hovered) {
            fill = theme.buttonHover;
        }
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, latched ? 2.0f : 1.0f,
                             latched ? theme.accentGreen : theme.panelBorder);
        if (latched) {
            DrawRectangleRec(Rectangle{r.x, r.y, 4.0f, r.height},
                             theme.accentGreen);
        }
        const char* text = m_buttons[i].label.c_str();
        int fontSize = 17;
        int textWidth = measureText(theme, text, fontSize);
        int tx = static_cast<int>(r.x + (r.width - textWidth) / 2);
        int ty = static_cast<int>(r.y + (r.height - fontSize) / 2);
        drawText(theme, text, tx, ty, fontSize, theme.textPrimary);
    }
}

}  // namespace cnnv::viz
