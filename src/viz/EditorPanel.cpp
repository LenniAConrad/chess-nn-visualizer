#include "viz/EditorPanel.h"

#include "chess/Bitboard.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>

namespace cnnv::viz {

namespace ce = cnnv::chess;

namespace {

constexpr float kPad         = 4.0f;
constexpr float kRowHeight   = 20.0f;
constexpr float kSmallButton = 20.0f;
constexpr float kSpacing     = 2.0f;

// Order of EP target squares the cycle visits.
//   0 → none, 1..8 → A3..H3, 9..16 → A6..H6.
ce::Square epForIndex(int idx) {
    if (idx <= 0) return ce::Square::None;
    int rank = (idx <= 8) ? 2 : 5;
    int file = (idx - 1) % 8;
    return ce::makeSquare(file, rank);
}

int indexForEp(ce::Square sq) {
    if (sq == ce::Square::None) return 0;
    int file = ce::fileOf(sq);
    int rank = ce::rankOf(sq);
    if (rank == 2) return 1 + file;
    if (rank == 5) return 9 + file;
    return 0;
}

const char* epLabel(int idx) {
    static char buf[4];
    if (idx == 0) return "none";
    int rank = (idx <= 8) ? 3 : 6;
    int file = (idx - 1) % 8;
    buf[0] = static_cast<char>('a' + file);
    buf[1] = static_cast<char>('0' + rank);
    buf[2] = '\0';
    return buf;
}

}  // namespace

EditorPanel::LayoutY EditorPanel::layout() const noexcept {
    float y = m_bounds.y + kPad;
    LayoutY L{};
    L.side       = y;                    y += kRowHeight + kSpacing;
    L.castling   = y;                    y += kRowHeight + kSpacing;
    L.ep         = y;                    y += kRowHeight + kSpacing;
    L.halfmove   = y;                    y += kRowHeight + kSpacing;
    L.fullmove   = y;                    y += kRowHeight + kSpacing;
    L.fenLabel   = y;                    y += 14.0f;
    L.fenText    = y;                    y += 30.0f;  // wraps to two lines
    L.issuesLabel = y + 2.0f;
    L.buttonsBottom = m_bounds.y + m_bounds.height - kPad - kRowHeight;
    return L;
}

Rectangle EditorPanel::sideButton(int idx) const {
    LayoutY L = layout();
    float w = (m_bounds.width - 2 * kPad - kSpacing) * 0.5f;
    return Rectangle{
        m_bounds.x + kPad + idx * (w + kSpacing),
        L.side, w, kRowHeight,
    };
}

Rectangle EditorPanel::castleButton(int idx) const {
    LayoutY L = layout();
    float w = (m_bounds.width - 2 * kPad - 3 * kSpacing) / 4.0f;
    return Rectangle{
        m_bounds.x + kPad + idx * (w + kSpacing),
        L.castling, w, kRowHeight,
    };
}

Rectangle EditorPanel::epPrevButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - 2 * kSmallButton - kSpacing,
        L.ep, kSmallButton, kRowHeight,
    };
}
Rectangle EditorPanel::epNextButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - kSmallButton,
        L.ep, kSmallButton, kRowHeight,
    };
}
Rectangle EditorPanel::halfMinusButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - 2 * kSmallButton - kSpacing,
        L.halfmove, kSmallButton, kRowHeight,
    };
}
Rectangle EditorPanel::halfPlusButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - kSmallButton,
        L.halfmove, kSmallButton, kRowHeight,
    };
}
Rectangle EditorPanel::fullMinusButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - 2 * kSmallButton - kSpacing,
        L.fullmove, kSmallButton, kRowHeight,
    };
}
Rectangle EditorPanel::fullPlusButton() const {
    LayoutY L = layout();
    return Rectangle{
        m_bounds.x + m_bounds.width - kPad - kSmallButton,
        L.fullmove, kSmallButton, kRowHeight,
    };
}

Rectangle EditorPanel::validateButton() const {
    LayoutY L = layout();
    float w = (m_bounds.width - 2 * kPad - 2 * kSpacing) / 3.0f;
    return Rectangle{
        m_bounds.x + kPad,
        L.buttonsBottom, w, kRowHeight,
    };
}
Rectangle EditorPanel::applyButton() const {
    Rectangle v = validateButton();
    return Rectangle{v.x + v.width + kSpacing, v.y, v.width, v.height};
}
Rectangle EditorPanel::cancelButton() const {
    Rectangle a = applyButton();
    return Rectangle{a.x + a.width + kSpacing, a.y, a.width, a.height};
}

void EditorPanel::cycleEp(int direction) {
    int idx = indexForEp(m_mode->editor().enPassantSquare()
                            .value_or(ce::Square::None));
    int total = 1 + 16;  // none + 16 valid squares
    idx = (idx + direction + total) % total;
    m_mode->editor().setEnPassantSquare(
        idx == 0 ? std::optional<ce::Square>{}
                 : std::optional<ce::Square>{epForIndex(idx)});
}

void EditorPanel::update() {
    if (!m_mode || !m_mode->active()) return;
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return;
    Vector2 mp = GetMousePosition();
    auto& edit = m_mode->editor();

    if (CheckCollisionPointRec(mp, sideButton(0))) {
        edit.setSideToMove(ce::Color::White); return;
    }
    if (CheckCollisionPointRec(mp, sideButton(1))) {
        edit.setSideToMove(ce::Color::Black); return;
    }

    using CS = ce::PositionEditor::CastleSide;
    struct CastleSlot { ce::Color c; CS side; };
    const std::array<CastleSlot, 4> slots = {{
        {ce::Color::White, CS::Kingside},
        {ce::Color::White, CS::Queenside},
        {ce::Color::Black, CS::Kingside},
        {ce::Color::Black, CS::Queenside},
    }};
    for (int i = 0; i < 4; ++i) {
        if (CheckCollisionPointRec(mp, castleButton(i))) {
            bool cur = edit.castlingRight(slots[i].c, slots[i].side);
            edit.setCastlingRight(slots[i].c, slots[i].side, !cur);
            return;
        }
    }

    if (CheckCollisionPointRec(mp, epPrevButton()))    { cycleEp(-1); return; }
    if (CheckCollisionPointRec(mp, epNextButton()))    { cycleEp(+1); return; }
    if (CheckCollisionPointRec(mp, halfMinusButton())) {
        edit.setHalfmoveClock(edit.halfmoveClock() - 1); return;
    }
    if (CheckCollisionPointRec(mp, halfPlusButton())) {
        edit.setHalfmoveClock(edit.halfmoveClock() + 1); return;
    }
    if (CheckCollisionPointRec(mp, fullMinusButton())) {
        edit.setFullmoveNumber(edit.fullmoveNumber() - 1); return;
    }
    if (CheckCollisionPointRec(mp, fullPlusButton())) {
        edit.setFullmoveNumber(edit.fullmoveNumber() + 1); return;
    }

    if (CheckCollisionPointRec(mp, validateButton())) {
        m_pending = Action::Validate; return;
    }
    if (CheckCollisionPointRec(mp, applyButton())) {
        m_pending = Action::Apply; return;
    }
    if (CheckCollisionPointRec(mp, cancelButton())) {
        m_pending = Action::Cancel; return;
    }
}

void EditorPanel::draw(const Theme& theme) const {
    if (!m_mode || !m_mode->active()) return;

    const auto& edit = m_mode->editor();
    LayoutY L = layout();

    auto drawToggle = [&](Rectangle r, const char* label, bool active) {
        Color fill = active ? theme.buttonActive : theme.buttonIdle;
        if (!active && CheckCollisionPointRec(GetMousePosition(), r)) {
            fill = theme.buttonHover;
        }
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        int w = measureText(theme, label, 13);
        drawText(theme, label,
                 static_cast<int>(r.x + (r.width - w) / 2),
                 static_cast<int>(r.y + (r.height - 13) / 2),
                 13, theme.textPrimary);
    };

    auto drawSmallButton = [&](Rectangle r, const char* label) {
        Color fill = theme.buttonIdle;
        if (CheckCollisionPointRec(GetMousePosition(), r)) fill = theme.buttonHover;
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        int w = measureText(theme, label, 13);
        drawText(theme, label,
                 static_cast<int>(r.x + (r.width - w) / 2),
                 static_cast<int>(r.y + (r.height - 13) / 2),
                 13, theme.textPrimary);
    };

    // Side to move.
    drawText(theme, "Side to move",
             static_cast<int>(m_bounds.x + kPad),
             static_cast<int>(L.side - 14),
             12, theme.textMuted);
    drawToggle(sideButton(0), "White", edit.sideToMove() == ce::Color::White);
    drawToggle(sideButton(1), "Black", edit.sideToMove() == ce::Color::Black);

    // Castling.
    drawText(theme, "Castling",
             static_cast<int>(m_bounds.x + kPad),
             static_cast<int>(L.castling - 14),
             12, theme.textMuted);
    using CS = ce::PositionEditor::CastleSide;
    drawToggle(castleButton(0), "K",
               edit.castlingRight(ce::Color::White, CS::Kingside));
    drawToggle(castleButton(1), "Q",
               edit.castlingRight(ce::Color::White, CS::Queenside));
    drawToggle(castleButton(2), "k",
               edit.castlingRight(ce::Color::Black, CS::Kingside));
    drawToggle(castleButton(3), "q",
               edit.castlingRight(ce::Color::Black, CS::Queenside));

    // EP.
    {
        int idx = indexForEp(edit.enPassantSquare().value_or(ce::Square::None));
        char line[32];
        std::snprintf(line, sizeof(line), "En passant: %s", epLabel(idx));
        drawText(theme, line,
                 static_cast<int>(m_bounds.x + kPad),
                 static_cast<int>(L.ep + 3),
                 13, theme.textPrimary);
        drawSmallButton(epPrevButton(), "<");
        drawSmallButton(epNextButton(), ">");
    }

    // Halfmove clock.
    {
        char line[48];
        std::snprintf(line, sizeof(line), "Halfmove clock: %d",
                      edit.halfmoveClock());
        drawText(theme, line,
                 static_cast<int>(m_bounds.x + kPad),
                 static_cast<int>(L.halfmove + 3),
                 13, theme.textPrimary);
        drawSmallButton(halfMinusButton(), "-");
        drawSmallButton(halfPlusButton(),  "+");
    }
    {
        char line[48];
        std::snprintf(line, sizeof(line), "Fullmove number: %d",
                      edit.fullmoveNumber());
        drawText(theme, line,
                 static_cast<int>(m_bounds.x + kPad),
                 static_cast<int>(L.fullmove + 3),
                 13, theme.textPrimary);
        drawSmallButton(fullMinusButton(), "-");
        drawSmallButton(fullPlusButton(),  "+");
    }

    // FEN display (mono, wrapped to two rows for readability).
    drawText(theme, "FEN",
             static_cast<int>(m_bounds.x + kPad),
             static_cast<int>(L.fenLabel),
             11, theme.textMuted);
    std::string fen = edit.fen();
    Rectangle fenBox{
        m_bounds.x + kPad, L.fenText - 2,
        m_bounds.width - 2 * kPad, 28.0f,
    };
    DrawRectangleRec(fenBox, theme.fenStripBackground);
    DrawRectangleLinesEx(fenBox, 1.0f, theme.panelBorder);
    // Split the FEN at the first space (after the board section) so the long
    // piece-placement field sits on its own line.
    std::string firstLine = fen;
    std::string secondLine;
    auto spacePos = fen.find(' ');
    if (spacePos != std::string::npos) {
        firstLine = fen.substr(0, spacePos);
        secondLine = fen.substr(spacePos + 1);
    }
    drawTextMono(theme, firstLine.c_str(),
                 static_cast<int>(fenBox.x + 4),
                 static_cast<int>(fenBox.y + 2),
                 11, theme.fenStripText);
    drawTextMono(theme, secondLine.c_str(),
                 static_cast<int>(fenBox.x + 4),
                 static_cast<int>(fenBox.y + 15),
                 11, theme.fenStripText);

    // Issues panel.
    if (m_mode->hasBeenValidated()) {
        const auto& v = m_mode->lastValidation();
        if (v.legal) {
            drawText(theme, "Position is legal.",
                     static_cast<int>(m_bounds.x + kPad),
                     static_cast<int>(L.issuesLabel),
                     12, theme.textPrimary);
        } else {
            float y = L.issuesLabel;
            drawText(theme, "Issues:",
                     static_cast<int>(m_bounds.x + kPad),
                     static_cast<int>(y),
                     12, theme.checkWarning);
            y += 12;
            float maxY = L.buttonsBottom - 2;
            for (const auto& issue : v.issues) {
                if (y + 11 > maxY) break;
                drawText(theme, issue.c_str(),
                         static_cast<int>(m_bounds.x + kPad),
                         static_cast<int>(y),
                         11, theme.textMuted);
                y += 12;
            }
        }
    } else {
        drawText(theme, "Click Validate to check legality.",
                 static_cast<int>(m_bounds.x + kPad),
                 static_cast<int>(L.issuesLabel),
                 12, theme.textDim);
    }

    // Action buttons. Apply is disabled (greyed) until validation passes.
    auto drawActionButton = [&](Rectangle r, const char* label, bool enabled) {
        Color fill = enabled ? theme.buttonIdle : theme.buttonIdle;
        Color text = enabled ? theme.textPrimary : theme.textDim;
        if (enabled && CheckCollisionPointRec(GetMousePosition(), r)) {
            fill = theme.buttonHover;
        }
        DrawRectangleRec(r, fill);
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        int w = measureText(theme, label, 13);
        drawText(theme, label,
                 static_cast<int>(r.x + (r.width - w) / 2),
                 static_cast<int>(r.y + (r.height - 13) / 2),
                 13, text);
    };
    bool applyEnabled = m_mode->hasBeenValidated() &&
                        m_mode->lastValidation().legal;
    drawActionButton(validateButton(), "Validate", true);
    drawActionButton(applyButton(),    "Apply",    applyEnabled);
    drawActionButton(cancelButton(),   "Cancel",   true);
}

}  // namespace cnnv::viz
