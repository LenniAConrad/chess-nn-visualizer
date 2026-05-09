#include "viz/MoveListView.h"

#include "game/MoveHistory.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace cnnv::viz {

namespace {

constexpr float kHeaderHeight = 28.0f;
constexpr float kRowHeight    = 24.0f;
constexpr float kPadding      = 10.0f;
constexpr float kNumberColW   = 36.0f;

}  // namespace

int MoveListView::visibleRows() const {
    float usable = m_bounds.height - kHeaderHeight - 2 * kPadding;
    if (usable <= 0) return 0;
    return std::max(0, static_cast<int>(usable / kRowHeight));
}

int MoveListView::scrollOffsetClamped(int totalRows) const {
    int vis = visibleRows();
    if (totalRows <= vis) return 0;
    return std::clamp(m_scrollRows, 0, totalRows - vis);
}

MoveListView::CellLayout MoveListView::rowLayout(int row) const {
    float rowTop = m_bounds.y + kHeaderHeight + kPadding +
                   static_cast<float>(row) * kRowHeight;
    float left   = m_bounds.x + kPadding;
    float colW   = (m_bounds.width - 2 * kPadding - kNumberColW) * 0.5f;
    Rectangle num{left, rowTop, kNumberColW, kRowHeight};
    Rectangle whitec{left + kNumberColW,            rowTop, colW, kRowHeight};
    Rectangle blackc{left + kNumberColW + colW,     rowTop, colW, kRowHeight};
    return CellLayout{num, whitec, blackc, rowTop, kRowHeight};
}

void MoveListView::update() {
    if (!m_game) return;

    // Scroll wheel adjusts the row offset only when the cursor is inside the
    // panel — otherwise other widgets (none yet, but BoardView later) keep
    // ownership of the wheel.
    Vector2 mp = GetMousePosition();
    bool inside = CheckCollisionPointRec(mp, m_bounds);
    if (inside) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            m_scrollRows -= static_cast<int>(wheel);
        }
    }

    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || !inside) return;

    // Header click → jump to start.
    Rectangle header{m_bounds.x, m_bounds.y, m_bounds.width, kHeaderHeight};
    if (CheckCollisionPointRec(mp, header)) {
        m_pendingClick = 0;
        return;
    }

    const auto& history = m_game->history();
    std::size_t total = history.size();
    int rows = static_cast<int>((total + 1) / 2);
    int offset = scrollOffsetClamped(rows);
    int vis = visibleRows();

    for (int i = 0; i < vis; ++i) {
        int row = offset + i;
        if (row >= rows) break;
        CellLayout cl = rowLayout(i);
        if (CheckCollisionPointRec(mp, cl.white)) {
            std::size_t ply = static_cast<std::size_t>(row) * 2 + 1;
            if (ply <= total) m_pendingClick = ply;
            return;
        }
        if (CheckCollisionPointRec(mp, cl.black)) {
            std::size_t ply = static_cast<std::size_t>(row) * 2 + 2;
            if (ply <= total) m_pendingClick = ply;
            return;
        }
    }
}

void MoveListView::draw(const Theme& theme) const {
    if (!m_game) return;

    DrawRectangleRec(m_bounds, theme.panelBackground);
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    Rectangle header{m_bounds.x, m_bounds.y, m_bounds.width, kHeaderHeight};
    DrawRectangleRec(header, theme.buttonIdle);
    DrawRectangleLinesEx(header, 1.0f, theme.panelBorder);
    drawText(theme, "Move history",
             static_cast<int>(header.x + kPadding),
             static_cast<int>(header.y + 6),
             16, theme.textPrimary);

    const auto& history = m_game->history();
    std::size_t currentPly = m_game->currentPly();

    // Walk the linked list once into a stable per-ply array of SAN strings.
    // Total ply count == history.size(); index 0 is the very first half-move.
    std::vector<const cnnv::game::MoveRecord*> nodes;
    nodes.reserve(history.size());
    for (const cnnv::game::MoveRecord* n = history.head(); n; n = n->next) {
        nodes.push_back(n);
    }

    int rows = static_cast<int>((nodes.size() + 1) / 2);
    int offset = scrollOffsetClamped(rows);
    int vis = visibleRows();

    for (int i = 0; i < vis; ++i) {
        int row = offset + i;
        if (row >= rows) break;
        CellLayout cl = rowLayout(i);

        // Striped background for legibility.
        if ((row & 1) == 0) {
            DrawRectangleRec(
                Rectangle{cl.moveNumber.x, cl.rowTop,
                          cl.moveNumber.width + cl.white.width + cl.black.width,
                          cl.rowHeight},
                theme.buttonIdle);
        }

        // Move number.
        char numBuf[16];
        std::snprintf(numBuf, sizeof(numBuf), "%d.", row + 1);
        drawText(theme, numBuf,
                 static_cast<int>(cl.moveNumber.x + 4),
                 static_cast<int>(cl.rowTop + 4),
                 15, theme.textMuted);

        std::size_t whiteIdx = static_cast<std::size_t>(row) * 2;
        std::size_t blackIdx = whiteIdx + 1;

        auto drawCell = [&](Rectangle r, const cnnv::game::MoveRecord* node,
                            std::size_t ply) {
            if (!node) return;
            bool isCurrent = (ply == currentPly);
            if (isCurrent) {
                DrawRectangleRec(r, theme.selection);
            }
            Color tint = isCurrent ? theme.background : theme.textPrimary;
            drawText(theme, node->san.c_str(),
                     static_cast<int>(r.x + 6),
                     static_cast<int>(r.y + 4),
                     16, tint);
        };

        const cnnv::game::MoveRecord* whiteNode =
            whiteIdx < nodes.size() ? nodes[whiteIdx] : nullptr;
        const cnnv::game::MoveRecord* blackNode =
            blackIdx < nodes.size() ? nodes[blackIdx] : nullptr;

        drawCell(cl.white, whiteNode, whiteIdx + 1);
        drawCell(cl.black, blackNode, blackIdx + 1);
    }

    // Scroll indicator (footer hint when overflowing).
    if (rows > vis) {
        char hint[64];
        std::snprintf(hint, sizeof(hint), "rows %d-%d / %d  (scroll)",
                      offset + 1,
                      std::min(rows, offset + vis),
                      rows);
        drawText(theme, hint,
                 static_cast<int>(m_bounds.x + kPadding),
                 static_cast<int>(m_bounds.y + m_bounds.height - 18),
                 13, theme.textDim);
    }
}

}  // namespace cnnv::viz
