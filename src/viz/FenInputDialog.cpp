#include "viz/FenInputDialog.h"

#include "chess/Fen.h"

namespace cnnv::viz {

void FenInputDialog::open(const std::string& initial) {
    m_buffer = initial;
    m_errorMessage.clear();
    m_active = true;
}

std::optional<cnnv::chess::Position> FenInputDialog::update() {
    if (!m_active) return std::nullopt;

    if (IsKeyPressed(KEY_ESCAPE)) {
        m_active = false;
        return std::nullopt;
    }

    int c = GetCharPressed();
    while (c > 0) {
        if (c >= 32 && c < 127) {
            m_buffer += static_cast<char>(c);
        }
        c = GetCharPressed();
    }

    if (IsKeyPressed(KEY_BACKSPACE) && !m_buffer.empty()) {
        m_buffer.pop_back();
    }

    if ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL))
        && IsKeyPressed(KEY_V)) {
        const char* text = GetClipboardText();
        if (text != nullptr) {
            for (const char* p = text; *p != '\0'; ++p) {
                if (*p >= 32 && *p < 127) m_buffer += *p;
            }
        }
    }

    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
        auto opt = cnnv::chess::Fen::parse(m_buffer);
        if (opt.has_value()) {
            m_active = false;
            return opt;
        }
        m_errorMessage = "Invalid FEN — try again or press Esc to cancel.";
    }
    return std::nullopt;
}

void FenInputDialog::draw(const Theme& theme) const {
    if (!m_active) return;
    DrawRectangleRec(m_bounds, theme.panelBackground);
    DrawRectangleLinesEx(m_bounds, 2.0f, theme.panelBorder);

    int padding = 12;
    drawText(theme, "Enter FEN, then press Enter (Esc to cancel):",
             static_cast<int>(m_bounds.x) + padding,
             static_cast<int>(m_bounds.y) + padding,
             18, theme.textMuted);

    Rectangle field{
        m_bounds.x + padding,
        m_bounds.y + padding + 28,
        m_bounds.width - 2 * padding,
        36.0f,
    };
    DrawRectangleRec(field, theme.buttonIdle);
    DrawRectangleLinesEx(field, 1.0f, theme.panelBorder);
    drawTextMono(theme, m_buffer.c_str(),
                 static_cast<int>(field.x) + 8,
                 static_cast<int>(field.y) + 8,
                 18, theme.textPrimary);

    if (!m_errorMessage.empty()) {
        drawText(theme, m_errorMessage.c_str(),
                 static_cast<int>(m_bounds.x) + padding,
                 static_cast<int>(field.y + field.height + 8),
                 15, theme.checkWarning);
    }
}

}  // namespace cnnv::viz
