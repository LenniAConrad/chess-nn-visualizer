#include "viz/ClassicalView.h"

#include "viz/TensorViz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace cnnv::viz {

namespace {

namespace ce = cnnv::chess::eval;

// The eleven breakdown terms in display order, with the labels the Java view
// uses (which differ slightly from the struct field names).
struct TermRef {
    const char* label;
    int ce::Breakdown::* member;
};

const std::array<TermRef, 11> kTerms = {{
    {"material",       &ce::Breakdown::material},
    {"piece-square",   &ce::Breakdown::pieceSquare},
    {"bishop pair",    &ce::Breakdown::bishopPair},
    {"pawn structure", &ce::Breakdown::pawnStructure},
    {"rook files",     &ce::Breakdown::rookFile},
    {"king safety",    &ce::Breakdown::kingSafety},
    {"activity",       &ce::Breakdown::activity},
    {"threats",        &ce::Breakdown::threats},
    {"space",          &ce::Breakdown::space},
    {"tempo",          &ce::Breakdown::tempo},
    {"check",          &ce::Breakdown::checkPenalty},
}};

const std::array<const char*, 6> kPstNames = {
    {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"}};

// Draws one centered signed bar (green right for +, red left for -).
void drawSignedBar(Rectangle area, int value, int maxAbs, const Theme& theme) {
    DrawRectangleRec(area, tensorviz::withAlpha(theme.buttonIdle, 220));
    const float mid = area.x + area.width * 0.5f;
    const float halfW = area.width * 0.5f;
    const float norm =
        std::clamp(static_cast<float>(value) / static_cast<float>(std::max(1, maxAbs)),
                   -1.0f, 1.0f);
    const float len = std::max(1.0f, halfW * std::fabs(norm));
    const Color c = value >= 0 ? theme.accentGreen : theme.accentRed;
    if (value >= 0) {
        DrawRectangleRec(Rectangle{mid, area.y + 2.0f, len, area.height - 4.0f}, c);
    } else {
        DrawRectangleRec(Rectangle{mid - len, area.y + 2.0f, len, area.height - 4.0f}, c);
    }
    DrawLineV(Vector2{mid, area.y}, Vector2{mid, area.y + area.height},
              theme.panelBorder);
}

// Static schematic of the evaluation pipeline: feature groups -> total -> WDL.
void drawClassicalDiagram(Rectangle bounds, const ce::Breakdown& b,
                          const ce::WdlTriplet& wdl, Color signature,
                          const Theme& theme) {
    struct Group { const char* title; int value; };
    const std::array<Group, 4> groups = {{
        {"material + PST", b.material + b.pieceSquare},
        {"pawns + rooks + bishops", b.pawnStructure + b.rookFile + b.bishopPair},
        {"king safety", b.kingSafety},
        {"activity + threats + space", b.activity + b.threats + b.space + b.tempo +
                                           b.checkPenalty},
    }};
    int maxAbs = std::abs(b.whiteTotal());
    for (const auto& g : groups) maxAbs = std::max(maxAbs, std::abs(g.value));
    maxAbs = std::max(1, maxAbs);

    const float gap = 12.0f;
    const float colW = (bounds.width - 3.0f * gap) * 0.25f;
    const float topH = std::clamp(bounds.height * 0.26f, 70.0f, 110.0f);
    std::array<Rectangle, 4> boxes{};
    for (int i = 0; i < 4; ++i) {
        boxes[static_cast<std::size_t>(i)] =
            Rectangle{bounds.x + static_cast<float>(i) * (colW + gap), bounds.y,
                      colW, topH};
    }
    const float midY = bounds.y + topH + 56.0f;
    const Rectangle totalR{bounds.x + bounds.width * 0.5f - colW * 0.75f, midY,
                           colW * 1.5f, topH};
    const float wdlY = midY + topH + 48.0f;
    const Rectangle wdlR{bounds.x + bounds.width * 0.5f - colW * 0.75f, wdlY,
                         colW * 1.5f, std::min(topH, 84.0f)};

    for (int i = 0; i < 4; ++i) {
        tensorviz::drawElbowConnection(boxes[static_cast<std::size_t>(i)], totalR,
                                       signature, true, false, theme);
    }
    tensorviz::drawElbowConnection(totalR, wdlR,
                                   b.whiteTotal() >= 0 ? theme.accentGreen
                                                       : theme.accentRed,
                                   true, false, theme);

    for (int i = 0; i < 4; ++i) {
        char detail[24];
        std::snprintf(detail, sizeof(detail), "%+d cp",
                      groups[static_cast<std::size_t>(i)].value);
        const float act = std::clamp(
            static_cast<float>(std::abs(groups[static_cast<std::size_t>(i)].value)) /
                static_cast<float>(maxAbs),
            0.0f, 1.0f);
        tensorviz::drawAbstractBlock(boxes[static_cast<std::size_t>(i)],
                                     groups[static_cast<std::size_t>(i)].title,
                                     detail, act, signature, false, theme);
    }
    char totalDetail[40];
    std::snprintf(totalDetail, sizeof(totalDetail), "%+d cp (White)", b.whiteTotal());
    tensorviz::drawAbstractBlock(totalR, "sum of terms", totalDetail, 1.0f,
                                 b.whiteTotal() >= 0 ? theme.accentGreen
                                                     : theme.accentRed,
                                 false, theme);
    const int total = std::max(1, wdl.win + wdl.draw + wdl.loss);
    char wdlDetail[48];
    std::snprintf(wdlDetail, sizeof(wdlDetail), "W %.0f%%  D %.0f%%  L %.0f%%",
                  100.0f * wdl.win / total, 100.0f * wdl.draw / total,
                  100.0f * wdl.loss / total);
    tensorviz::drawAbstractBlock(wdlR, "win / draw / loss", wdlDetail, 1.0f,
                                 signature, false, theme);
}

}  // namespace

void ClassicalView::update(const cnnv::nn::ActivationSnapshot& snap) {
    (void)snap;  // model-free; the position is fed via setPosition().
}

void ClassicalView::setPosition(const cnnv::chess::Position& pos) {
    const std::uint64_t h = pos.hash();
    if (m_hasData && h == m_hash) return;
    m_hash = h;
    m_breakdown = ce::evaluateWhiteBreakdown(pos);
    m_wdl = ce::evaluateWdl(pos, /*terminalAware=*/true);
    m_hasData = true;
}

void ClassicalView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    float y = m_bounds.y + pad;
    drawText(theme, "Classical evaluator",
             static_cast<int>(m_bounds.x + pad), static_cast<int>(y), 23,
             theme.textPrimary);
    m_mode = tensorviz::drawModeSwitcher5(
        Rectangle{m_bounds.x + m_bounds.width - pad - 300.0f, y + 1.0f, 300.0f,
                  30.0f},
        m_mode, m_signature, theme);
    y += 36.0f;

    if (!m_hasData) {
        drawText(theme, "Load a position to see the handcrafted breakdown.",
                 static_cast<int>(m_bounds.x + pad), static_cast<int>(y), 16,
                 theme.textMuted);
        return;
    }

    const int stm = m_breakdown.stmTotal();
    char subtitle[128];
    std::snprintf(subtitle, sizeof(subtitle),
                  "handcrafted cp  -  %s to move  -  %+d cp (side to move)",
                  m_breakdown.whiteToMove ? "white" : "black", stm);
    drawText(theme, subtitle, static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 13, theme.textMuted);
    y += 22.0f;

    if (m_mode == 4) {
        drawClassicalDiagram(
            Rectangle{m_bounds.x + pad, y, m_bounds.width - 2.0f * pad,
                      m_bounds.y + m_bounds.height - y - pad},
            m_breakdown, m_wdl, m_signature, theme);
        return;
    }

    const float innerX = m_bounds.x + pad;
    const float innerW = m_bounds.width - 2.0f * pad;

    const float leftW = innerW;

    // Phase bar.
    const float phaseW = std::min(220.0f, leftW);
    Rectangle phaseTrack{innerX, y, phaseW, 14.0f};
    DrawRectangleRec(phaseTrack, tensorviz::withAlpha(theme.buttonIdle, 230));
    const float phaseFill =
        std::round(phaseW * (1.0f - static_cast<float>(m_breakdown.phase)));
    DrawRectangleRec(Rectangle{phaseTrack.x, phaseTrack.y, phaseFill, phaseTrack.height},
                     tensorviz::withAlpha(m_signature, 200));
    drawText(theme, "opening", static_cast<int>(phaseTrack.x + 3.0f),
             static_cast<int>(phaseTrack.y + 1.0f), 9, theme.textPrimary);
    {
        const int tw = measureText(theme, "endgame", 9);
        drawText(theme, "endgame",
                 static_cast<int>(phaseTrack.x + phaseTrack.width - tw - 3.0f),
                 static_cast<int>(phaseTrack.y + 1.0f), 9, theme.textPrimary);
    }
    float ly = y + 22.0f;

    // WDL bar.
    {
        const int total = std::max(1, m_wdl.win + m_wdl.draw + m_wdl.loss);
        Rectangle bar{innerX, ly, phaseW, 12.0f};
        const float wW = std::round(bar.width * static_cast<float>(m_wdl.win) / total);
        const float dW = std::round(bar.width * static_cast<float>(m_wdl.draw) / total);
        DrawRectangleRec(Rectangle{bar.x, bar.y, wW, bar.height}, theme.accentGreen);
        DrawRectangleRec(Rectangle{bar.x + wW, bar.y, dW, bar.height},
                         tensorviz::withAlpha(theme.textMuted, 160));
        DrawRectangleRec(Rectangle{bar.x + wW + dW, bar.y,
                                   bar.width - wW - dW, bar.height},
                         theme.accentRed);
        char wdlText[48];
        std::snprintf(wdlText, sizeof(wdlText), "W %.0f%%  D %.0f%%  L %.0f%%",
                      100.0f * m_wdl.win / total, 100.0f * m_wdl.draw / total,
                      100.0f * m_wdl.loss / total);
        drawText(theme, wdlText, static_cast<int>(bar.x),
                 static_cast<int>(bar.y + 16.0f), 11, theme.textMuted);
    }
    ly += 36.0f;

    // Term contributions chart.
    int maxAbs = std::abs(m_breakdown.whiteTotal());
    for (const auto& t : kTerms) maxAbs = std::max(maxAbs, std::abs(m_breakdown.*(t.member)));
    maxAbs = std::max(1, maxAbs);

    drawText(theme, "term contributions (White - Black, centipawns)",
             static_cast<int>(innerX), static_cast<int>(ly), 11, theme.textMuted);
    ly += 18.0f;

    const float labelW = 104.0f;
    const float rowH = 18.0f;
    const float rowGap = 4.0f;
    const float barLeft = innerX + labelW;
    const float barW = std::max(80.0f, leftW - labelW);
    const float chartBottom = m_bounds.y + m_bounds.height - pad -
                              (m_mode >= 1 && m_mode <= 3 ? 132.0f : 0.0f);
    for (const auto& t : kTerms) {
        if (ly + rowH > chartBottom) break;
        const int v = m_breakdown.*(t.member);
        drawText(theme, t.label, static_cast<int>(innerX),
                 static_cast<int>(ly + 3.0f), 12, theme.textMuted);
        drawSignedBar(Rectangle{barLeft, ly, barW, rowH}, v, maxAbs, theme);
        char vt[16];
        std::snprintf(vt, sizeof(vt), "%+d", v);
        const float mid = barLeft + barW * 0.5f;
        if (v >= 0) {
            drawText(theme, vt, static_cast<int>(mid + 4.0f),
                     static_cast<int>(ly + 3.0f), 9, theme.textPrimary);
        } else {
            const int tw = measureText(theme, vt, 9);
            drawText(theme, vt, static_cast<int>(mid - 4.0f - tw),
                     static_cast<int>(ly + 3.0f), 9, theme.textPrimary);
        }
        ly += rowH + rowGap;
    }
    // Total row.
    if (ly + rowH <= chartBottom) {
        DrawLineV(Vector2{innerX, ly}, Vector2{innerX + leftW, ly},
                  theme.panelBorder);
        ly += 4.0f;
        drawText(theme, "total", static_cast<int>(innerX),
                 static_cast<int>(ly + 2.0f), 12, theme.textPrimary);
        drawSignedBar(Rectangle{barLeft, ly, barW, rowH}, m_breakdown.whiteTotal(),
                      maxAbs, theme);
        ly += rowH + rowGap;
    }

    // PST heatmaps (Trace/All/Atlas modes show them; Overview hides them).
    if (m_mode >= 1 && m_mode <= 3) {
        const float heatTop = m_bounds.y + m_bounds.height - pad - 124.0f;
        drawText(theme, "piece-square tables (White perspective)",
                 static_cast<int>(innerX), static_cast<int>(heatTop), 11,
                 theme.textMuted);
        const float top = heatTop + 16.0f;
        const float gap = 14.0f;
        const float availW = innerW - gap * 5.0f;
        const float cell = std::clamp(
            std::min(availW / 6.0f, (m_bounds.y + m_bounds.height - pad - top)) / 8.0f,
            5.0f, 12.0f);
        const float boardW = cell * 8.0f;
        for (int type = 1; type <= 6; ++type) {
            const std::array<int, 64> pst =
                cnnv::chess::eval::pieceSquareTable(type);
            int pstMax = 1;
            for (int v : pst) pstMax = std::max(pstMax, std::abs(v));
            const float ox = innerX + (type - 1) * (boardW + gap);
            for (int idx = 0; idx < 64; ++idx) {
                const int row = idx / 8;  // 0 = a8 row (rank 8 at top)
                const int col = idx % 8;
                Rectangle sq{ox + col * cell, top + row * cell, cell, cell};
                const int v = pst[static_cast<std::size_t>(idx)];
                const float norm = static_cast<float>(v) / static_cast<float>(pstMax);
                const Color base = v >= 0 ? theme.accentGreen : theme.accentRed;
                const unsigned char a = static_cast<unsigned char>(
                    std::clamp(40.0f + 200.0f * std::fabs(norm), 0.0f, 255.0f));
                DrawRectangleRec(sq, tensorviz::withAlpha(base, a));
            }
            DrawRectangleLinesEx(Rectangle{ox, top, boardW, boardW}, 1.0f,
                                 tensorviz::withAlpha(theme.panelBorder, 200));
            drawText(theme, kPstNames[static_cast<std::size_t>(type - 1)],
                     static_cast<int>(ox), static_cast<int>(top + boardW + 2.0f),
                     10, theme.textDim);
        }
    }
}

}  // namespace cnnv::viz
