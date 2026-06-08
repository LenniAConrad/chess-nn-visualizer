#include "viz/TensorViz.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cnnv::viz::tensorviz {

namespace {

Camera2D g_activeCamera{};
bool g_hasActiveCamera = false;

struct GridSpec {
    int rows = 1;
    int cols = 1;
    bool planes = false;
    int planeRows = 1;
    int planeCols = 1;
    int planesPerRow = 1;
};

float clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

void drawLineSegment(Vector2 a, Vector2 b, float thickness, Color color) {
    DrawLineEx(a, b, thickness, color);
}

void drawArrowHead(Vector2 end, Vector2 previous, Color color, float size) {
    const float dx = end.x - previous.x;
    const float dy = end.y - previous.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1.0f) return;
    const float ux = dx / len;
    const float uy = dy / len;
    const float wing = size * 0.48f;
    const Vector2 base{end.x - ux * size, end.y - uy * size};
    DrawTriangle(end,
                 Vector2{base.x - uy * wing, base.y + ux * wing},
                 Vector2{base.x + uy * wing, base.y - ux * wing},
                 color);
}

GridSpec inferGrid(const std::vector<float>& values,
                   const std::vector<std::size_t>& shape,
                   Rectangle bounds) {
    GridSpec grid;
    const std::size_t n = values.size();
    if (n == 0) return grid;

    if (shape.size() == 3 && shape[1] > 0 && shape[2] > 0 &&
        shape[0] * shape[1] * shape[2] == n) {
        const int planes = static_cast<int>(shape[0]);
        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);
        const float aspect = std::max(0.3f, bounds.width / std::max(bounds.height, 1.0f));
        int perRow = static_cast<int>(std::ceil(std::sqrt(
            static_cast<float>(planes) * aspect * static_cast<float>(h) /
            std::max(1.0f, static_cast<float>(w)))));
        perRow = std::clamp(perRow, 1, planes);
        grid.planes = true;
        grid.planeRows = h;
        grid.planeCols = w;
        grid.planesPerRow = perRow;
        grid.cols = perRow * w;
        grid.rows = ((planes + perRow - 1) / perRow) * h;
        return grid;
    }

    if (shape.size() == 2 && shape[0] > 0 && shape[1] > 0 &&
        shape[0] * shape[1] == n) {
        grid.rows = static_cast<int>(shape[0]);
        grid.cols = static_cast<int>(shape[1]);
        return grid;
    }

    if (shape.size() == 1 && shape[0] == n) {
        if (n == 64) {
            grid.rows = 8;
            grid.cols = 8;
            return grid;
        }
        if (n == 3) {
            grid.rows = 1;
            grid.cols = 3;
            return grid;
        }
    }

    const float aspect = std::max(0.2f, bounds.width / std::max(bounds.height, 1.0f));
    int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(n) * aspect)));
    cols = std::max(1, cols);
    grid.cols = cols;
    grid.rows = std::max(1, static_cast<int>(
        (n + static_cast<std::size_t>(cols) - 1) /
        static_cast<std::size_t>(cols)));
    return grid;
}

Vector2 cellPosition(std::size_t index, const GridSpec& grid) {
    if (grid.planes) {
        const int planeSize = grid.planeRows * grid.planeCols;
        const int plane = static_cast<int>(index / static_cast<std::size_t>(planeSize));
        const int offset = static_cast<int>(index % static_cast<std::size_t>(planeSize));
        const int py = plane / grid.planesPerRow;
        const int px = plane % grid.planesPerRow;
        const int y = offset / grid.planeCols;
        const int x = offset % grid.planeCols;
        return Vector2{static_cast<float>(px * grid.planeCols + x),
                       static_cast<float>(py * grid.planeRows + y)};
    }

    return Vector2{static_cast<float>(index % static_cast<std::size_t>(grid.cols)),
                   static_cast<float>(index / static_cast<std::size_t>(grid.cols))};
}

void drawGridGuides(Rectangle bounds,
                    const GridSpec& grid,
                    const Theme& theme,
                    bool highDetail) {
    const float cellW = bounds.width / static_cast<float>(std::max(1, grid.cols));
    const float cellH = bounds.height / static_cast<float>(std::max(1, grid.rows));
    Color guide = withAlpha(theme.panelBorder, highDetail ? 105 : 62);

    if (cellW >= 6.0f) {
        for (int c = 1; c < grid.cols; ++c) {
            const float x = bounds.x + static_cast<float>(c) * cellW;
            DrawLineV(Vector2{x, bounds.y}, Vector2{x, bounds.y + bounds.height}, guide);
        }
    }
    if (cellH >= 6.0f) {
        for (int r = 1; r < grid.rows; ++r) {
            const float y = bounds.y + static_cast<float>(r) * cellH;
            DrawLineV(Vector2{bounds.x, y}, Vector2{bounds.x + bounds.width, y}, guide);
        }
    }

    if (grid.planes && (cellW * static_cast<float>(grid.planeCols) >= 10.0f ||
                        cellH * static_cast<float>(grid.planeRows) >= 10.0f)) {
        Color planeGuide = withAlpha(theme.textDim, 90);
        for (int c = grid.planeCols; c < grid.cols; c += grid.planeCols) {
            const float x = bounds.x + static_cast<float>(c) * cellW;
            DrawLineEx(Vector2{x, bounds.y},
                       Vector2{x, bounds.y + bounds.height},
                       1.0f, planeGuide);
        }
        for (int r = grid.planeRows; r < grid.rows; r += grid.planeRows) {
            const float y = bounds.y + static_cast<float>(r) * cellH;
            DrawLineEx(Vector2{bounds.x, y},
                       Vector2{bounds.x + bounds.width, y},
                       1.0f, planeGuide);
        }
    }
}

void drawBadge(Rectangle bounds,
               const char* label,
               const char* value,
               const Theme& theme) {
    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 235));
    DrawRectangleLinesEx(bounds, 1.0f, withAlpha(theme.panelBorder, 215));
    drawTextMono(theme, label,
                 static_cast<int>(bounds.x + 6.0f),
                 static_cast<int>(bounds.y + 4.0f),
                 10, theme.textDim);
    drawTextMono(theme, value,
                 static_cast<int>(bounds.x + 6.0f),
                 static_cast<int>(bounds.y + 18.0f),
                 12, theme.textPrimary);
}

void drawStatsBadges(Rectangle bounds,
                     const TensorStats& stats,
                     const Theme& theme) {
    char nText[32];
    char rmsText[32];
    char meanText[32];
    char minText[32];
    char maxText[32];
    std::snprintf(nText, sizeof(nText), "%zu", stats.count);
    std::snprintf(rmsText, sizeof(rmsText), "%.3f", stats.rms);
    std::snprintf(meanText, sizeof(meanText), "%.3f", stats.mean);
    std::snprintf(minText, sizeof(minText), "%.3f", stats.min);
    std::snprintf(maxText, sizeof(maxText), "%.3f", stats.max);

    const char* labels[] = {"n", "rms", "mean", "min", "max"};
    const char* values[] = {nText, rmsText, meanText, minText, maxText};
    const int count = 5;
    const float gap = 6.0f;
    const float w = (bounds.width - gap * static_cast<float>(count - 1)) /
                    static_cast<float>(count);
    for (int i = 0; i < count; ++i) {
        drawBadge(Rectangle{bounds.x + static_cast<float>(i) * (w + gap),
                            bounds.y, w, bounds.height},
                  labels[i], values[i], theme);
    }
}

}  // namespace

TensorStats computeStats(const std::vector<float>& values) {
    TensorStats stats;
    stats.count = values.size();
    if (values.empty()) return stats;

    stats.min = values[0];
    stats.max = values[0];
    double sum = 0.0;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    for (float v : values) {
        stats.min = std::min(stats.min, v);
        stats.max = std::max(stats.max, v);
        stats.maxAbs = std::max(stats.maxAbs, std::fabs(v));
        sum += v;
        sumAbs += std::fabs(v);
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double inv = 1.0 / static_cast<double>(values.size());
    stats.mean = static_cast<float>(sum * inv);
    stats.meanAbs = static_cast<float>(sumAbs * inv);
    stats.rms = static_cast<float>(std::sqrt(sumSq * inv));
    return stats;
}

float maxAbs(const std::vector<float>& values) {
    float m = 0.0f;
    for (float v : values) m = std::max(m, std::fabs(v));
    return m;
}

const char* squareName(int square) {
    static const char* names[] = {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    };
    return (square >= 0 && square < 64) ? names[square] : "??";
}

std::string fitText(const Theme& theme,
                    std::string text,
                    int fontSize,
                    bool mono,
                    float maxWidth) {
    if (maxWidth <= 1.0f || text.empty()) return "";
    auto measure = [&]() {
        return mono ? measureTextMono(theme, text.c_str(), fontSize)
                    : measureText(theme, text.c_str(), fontSize);
    };
    if (measure() <= static_cast<int>(maxWidth)) return text;

    const std::string ellipsis = "...";
    while (!text.empty()) {
        text.pop_back();
        std::string candidate = text + ellipsis;
        const int w = mono ? measureTextMono(theme, candidate.c_str(), fontSize)
                           : measureText(theme, candidate.c_str(), fontSize);
        if (w <= static_cast<int>(maxWidth)) return candidate;
    }
    return ellipsis;
}

Rectangle lerfSquare(Rectangle board, int square, bool whiteDown) {
    const int file = square & 7;
    const int rank = square >> 3;
    const int col = whiteDown ? file : 7 - file;
    const int drawRank = whiteDown ? 7 - rank : rank;
    const float cw = board.width / 8.0f;
    const float ch = board.height / 8.0f;
    return Rectangle{board.x + static_cast<float>(col) * cw,
                     board.y + static_cast<float>(drawRank) * ch, cw, ch};
}

void drawMiniBoard(Rectangle board, const Theme& theme) {
    for (int sq = 0; sq < 64; ++sq) {
        const Rectangle cell = lerfSquare(board, sq);
        const int file = sq & 7;
        const int rank = sq >> 3;
        const bool light = ((file + rank) & 1) != 0;
        DrawRectangleRec(cell, light ? theme.squareLight : theme.squareDark);
    }
    DrawRectangleLinesEx(board, 1.0f, theme.panelBorder);
}

void drawCenteredText(const Theme& theme,
                      const char* text,
                      Rectangle bounds,
                      int fontSize,
                      Color color,
                      bool mono) {
    const int w = mono ? measureTextMono(theme, text, fontSize)
                       : measureText(theme, text, fontSize);
    const int x = static_cast<int>(
        bounds.x + (bounds.width - static_cast<float>(w)) * 0.5f);
    const int y = static_cast<int>(
        bounds.y + (bounds.height - static_cast<float>(fontSize)) * 0.5f);
    if (mono) {
        drawTextMono(theme, text, x, y, fontSize, color);
    } else {
        drawText(theme, text, x, y, fontSize, color);
    }
}

Color blend(Color a, Color b, float t) {
    t = clamp01(t);
    auto mix = [t](unsigned char x, unsigned char y) {
        return static_cast<unsigned char>(
            static_cast<float>(x) + (static_cast<float>(y) - static_cast<float>(x)) * t);
    };
    return Color{mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a)};
}

Color withAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

Color signedColor(float value, float scale, const Theme& theme) {
    const float safeScale = std::max(scale, 1e-6f);
    const float t = std::sqrt(clamp01(std::fabs(value) / safeScale));
    const Color zero = Color{45, 48, 54, 245};
    if (value < 0.0f) return blend(zero, theme.accentBlue, t);
    return blend(zero, theme.overlayHot, t);
}

void setActiveCamera(const Camera2D& camera) {
    g_activeCamera = camera;
    g_hasActiveCamera = true;
}

void clearActiveCamera() {
    g_hasActiveCamera = false;
}

Vector2 panelMousePosition() {
    if (!g_hasActiveCamera) return GetMousePosition();
    return GetScreenToWorld2D(GetMousePosition(), g_activeCamera);
}

void drawSectionHeader(Rectangle bounds,
                       const char* title,
                       const char* detail,
                       const Theme& theme,
                       Color accent) {
    DrawRectangleRec(bounds, withAlpha(theme.panelBackground, 240));
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);
    const bool hasAccent = accent.a != 0;
    const float textLeft = hasAccent ? 14.0f : 9.0f;
    if (hasAccent) {
        DrawRectangleRec(Rectangle{bounds.x, bounds.y, 4.0f, bounds.height},
                         accent);
    }
    const std::string titleText =
        fitText(theme, title ? title : "", 15, false, bounds.width - textLeft - 9.0f);
    drawText(theme, titleText.c_str(),
             static_cast<int>(bounds.x + textLeft),
             static_cast<int>(bounds.y + 6.0f),
             15, theme.textPrimary);
    if (detail && detail[0] != '\0') {
        const std::string detailText =
            fitText(theme, detail, 11, true, bounds.width - textLeft - 9.0f);
        drawTextMono(theme, detailText.c_str(),
                     static_cast<int>(bounds.x + textLeft),
                     static_cast<int>(bounds.y + 26.0f),
                     11, theme.textMuted);
    }
}

bool drawModeToggle(Rectangle bounds, bool detailed, const Theme& theme,
                    Color accent) {
    const Rectangle left{bounds.x, bounds.y, bounds.width * 0.5f, bounds.height};
    const Rectangle right{bounds.x + bounds.width * 0.5f, bounds.y,
                          bounds.width * 0.5f, bounds.height};
    const Vector2 mouse = panelMousePosition();
    bool next = detailed;
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (CheckCollisionPointRec(mouse, left)) next = false;
        if (CheckCollisionPointRec(mouse, right)) next = true;
    }

    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 225));
    const Color activeFill =
        accent.a != 0 ? blend(theme.buttonActive, accent, 0.55f)
                      : theme.buttonActive;
    DrawRectangleRec(next ? right : left, activeFill);
    if (CheckCollisionPointRec(mouse, left) || CheckCollisionPointRec(mouse, right)) {
        DrawRectangleRec(CheckCollisionPointRec(mouse, right) ? right : left,
                         withAlpha(theme.buttonHover, 155));
    }
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);
    DrawLineV(Vector2{right.x, right.y},
              Vector2{right.x, right.y + right.height},
              theme.panelBorder);

    const char* leftText = "abstract";
    const char* rightText = "detailed";
    const int size = 13;
    int w = measureText(theme, leftText, size);
    drawText(theme, leftText,
             static_cast<int>(left.x + (left.width - static_cast<float>(w)) * 0.5f),
             static_cast<int>(left.y + (left.height - 15.0f) * 0.5f),
             size, next ? theme.textMuted : theme.textPrimary);
    w = measureText(theme, rightText, size);
    drawText(theme, rightText,
             static_cast<int>(right.x + (right.width - static_cast<float>(w)) * 0.5f),
             static_cast<int>(right.y + (right.height - 15.0f) * 0.5f),
             size, next ? theme.textPrimary : theme.textMuted);
    return next;
}

int drawModeSwitcher5(Rectangle bounds, int mode, Color accent,
                      const Theme& theme) {
    static const char* labels[5] = {"Overview", "Trace", "All", "Atlas",
                                    "Diagram"};
    const float segW = bounds.width / 5.0f;
    const Vector2 mouse = panelMousePosition();
    int next = std::clamp(mode, 0, 4);
    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 225));
    for (int i = 0; i < 5; ++i) {
        const Rectangle seg{bounds.x + segW * static_cast<float>(i), bounds.y,
                            segW, bounds.height};
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
            CheckCollisionPointRec(mouse, seg)) {
            next = i;
        }
    }
    const Rectangle activeSeg{bounds.x + segW * static_cast<float>(next), bounds.y,
                              segW, bounds.height};
    const Color activeFill = accent.a != 0
                                 ? blend(theme.buttonActive, accent, 0.55f)
                                 : theme.buttonActive;
    DrawRectangleRec(activeSeg, activeFill);
    for (int i = 0; i < 5; ++i) {
        const Rectangle seg{bounds.x + segW * static_cast<float>(i), bounds.y,
                            segW, bounds.height};
        if (CheckCollisionPointRec(mouse, seg) && i != next) {
            DrawRectangleRec(seg, withAlpha(theme.buttonHover, 150));
        }
        if (i != 0) {
            DrawLineV(Vector2{seg.x, seg.y}, Vector2{seg.x, seg.y + seg.height},
                      theme.panelBorder);
        }
        const int w = measureText(theme, labels[i], 11);
        drawText(theme, labels[i],
                 static_cast<int>(seg.x + (seg.width - static_cast<float>(w)) * 0.5f),
                 static_cast<int>(seg.y + (seg.height - 13.0f) * 0.5f), 11,
                 i == next ? theme.textPrimary : theme.textMuted);
    }
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);
    return next;
}

int drawModeSelector(Rectangle bounds, int mode, const Theme& theme,
                     Color accent) {
    const char* labels[3] = {"abstract", "detailed", "diagram"};
    const float segW = bounds.width / 3.0f;
    const Vector2 mouse = panelMousePosition();
    int next = mode < 0 ? 0 : (mode > 2 ? 2 : mode);

    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 225));
    for (int i = 0; i < 3; ++i) {
        const Rectangle seg{bounds.x + segW * static_cast<float>(i), bounds.y,
                            segW, bounds.height};
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
            CheckCollisionPointRec(mouse, seg)) {
            next = i;
        }
    }
    const Rectangle activeSeg{bounds.x + segW * static_cast<float>(next), bounds.y,
                              segW, bounds.height};
    const Color activeFill =
        accent.a != 0 ? blend(theme.buttonActive, accent, 0.55f)
                      : theme.buttonActive;
    DrawRectangleRec(activeSeg, activeFill);
    for (int i = 0; i < 3; ++i) {
        const Rectangle seg{bounds.x + segW * static_cast<float>(i), bounds.y,
                            segW, bounds.height};
        if (CheckCollisionPointRec(mouse, seg) && i != next) {
            DrawRectangleRec(seg, withAlpha(theme.buttonHover, 150));
        }
        if (i != 0) {
            DrawLineV(Vector2{seg.x, seg.y}, Vector2{seg.x, seg.y + seg.height},
                      theme.panelBorder);
        }
        const int size = 12;
        const int w = measureText(theme, labels[i], size);
        drawText(theme, labels[i],
                 static_cast<int>(seg.x + (seg.width - static_cast<float>(w)) * 0.5f),
                 static_cast<int>(seg.y + (seg.height - 14.0f) * 0.5f), size,
                 i == next ? theme.textPrimary : theme.textMuted);
    }
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);
    return next;
}

void drawAbstractBlock(Rectangle bounds,
                       const std::string& title,
                       const std::string& detail,
                       float activity,
                       Color accent,
                       bool selected,
                       const Theme& theme) {
    activity = clamp01(activity);
    const Color fill = blend(theme.panelBackground, accent, selected ? 0.18f : 0.08f);
    DrawRectangleRec(bounds, fill);
    DrawRectangleLinesEx(bounds, selected ? 2.0f : 1.0f,
                         selected ? theme.selection : theme.panelBorder);
    const int titleSize = bounds.width < 110.0f ? 13 : 16;
    const int detailSize = bounds.width < 110.0f ? 10 : 12;
    const std::string titleText =
        fitText(theme, title, titleSize, false, bounds.width - 20.0f);
    drawText(theme, titleText.c_str(),
             static_cast<int>(bounds.x + 10.0f),
             static_cast<int>(bounds.y + 10.0f),
             titleSize,
             theme.textPrimary);
    if (!detail.empty()) {
        const std::string detailText =
            fitText(theme, detail, detailSize, true, bounds.width - 20.0f);
        drawTextMono(theme, detailText.c_str(),
                     static_cast<int>(bounds.x + 10.0f),
                     static_cast<int>(bounds.y + 34.0f),
                     detailSize,
                     theme.textMuted);
    }
    const Rectangle track{bounds.x + 10.0f, bounds.y + bounds.height - 14.0f,
                          bounds.width - 20.0f, 5.0f};
    DrawRectangleRec(track, withAlpha(theme.buttonActive, 105));
    DrawRectangleRec(Rectangle{track.x, track.y, track.width * activity, track.height},
                     blend(theme.accentBlue, theme.overlayHot, activity));
}

void drawLayerCard(Rectangle bounds,
                   const std::string& name,
                   const std::string& shape,
                   const std::vector<float>& values,
                   const std::vector<std::size_t>& shapeDims,
                   float rmsScale,
                   Color accent,
                   bool selected,
                   bool dimmed,
                   const Theme& theme) {
    const TensorStats stats = computeStats(values);
    const float activity = clamp01(stats.rms / std::max(rmsScale, 1e-6f));
    Color fill = blend(theme.panelBackground, accent, selected ? 0.12f : 0.035f);
    if (dimmed) fill = blend(theme.background, fill, 0.72f);
    DrawRectangleRec(bounds, fill);
    DrawRectangleLinesEx(bounds, selected ? 2.0f : 1.0f,
                         selected ? theme.selection : withAlpha(theme.panelBorder, 210));

    const int nameSize = bounds.width < 82.0f ? 11 : 13;
    const int shapeSize = bounds.width < 82.0f ? 9 : 10;
    const std::string nameText =
        fitText(theme, name, nameSize, false, bounds.width - 14.0f);
    drawText(theme, nameText.c_str(),
             static_cast<int>(bounds.x + 7.0f),
             static_cast<int>(bounds.y + 6.0f),
             nameSize, dimmed ? theme.textMuted : theme.textPrimary);
    const std::string shapeText =
        fitText(theme, shape, shapeSize, true, bounds.width * 0.48f);
    drawTextMono(theme, shapeText.c_str(),
                 static_cast<int>(bounds.x + 7.0f),
                 static_cast<int>(bounds.y + 25.0f),
                 shapeSize, theme.textDim);

    char rmsText[32];
    std::snprintf(rmsText, sizeof(rmsText), "rms %.2f", stats.rms);
    const int rmsW = measureTextMono(theme, rmsText, shapeSize);
    drawTextMono(theme, rmsText,
                 static_cast<int>(bounds.x + bounds.width - static_cast<float>(rmsW) - 7.0f),
                 static_cast<int>(bounds.y + 25.0f),
                 shapeSize, theme.textMuted);

    Rectangle heatmap{
        bounds.x + 7.0f,
        bounds.y + 42.0f,
        bounds.width - 14.0f,
        std::max(12.0f, bounds.height - 61.0f),
    };
    drawTensorPreview(values, shapeDims, heatmap, theme, stats.maxAbs, false);

    Rectangle track{bounds.x + 7.0f, bounds.y + bounds.height - 11.0f,
                    bounds.width - 14.0f, 5.0f};
    DrawRectangleRec(track, withAlpha(theme.buttonActive, 120));
    DrawRectangleRec(Rectangle{track.x, track.y, track.width * activity, track.height},
                     blend(theme.accentBlue, theme.overlayHot, activity));
}

void drawTensorPreview(const std::vector<float>& values,
                       const std::vector<std::size_t>& shapeDims,
                       Rectangle bounds,
                       const Theme& theme,
                       float scale,
                       bool highDetail) {
    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 165));
    if (values.empty() || bounds.width <= 1.0f || bounds.height <= 1.0f) {
        return;
    }

    if (scale < 1e-6f) scale = computeStats(values).maxAbs;
    if (scale < 1e-6f) scale = 1.0f;

    const GridSpec grid = inferGrid(values, shapeDims, bounds);
    const float cellW = bounds.width / static_cast<float>(std::max(1, grid.cols));
    const float cellH = bounds.height / static_cast<float>(std::max(1, grid.rows));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const Vector2 pos = cellPosition(i, grid);
        const float x = bounds.x + pos.x * cellW;
        const float y = bounds.y + pos.y * cellH;
        DrawRectangleRec(Rectangle{x, y,
                                   std::max(0.45f, cellW - 0.25f),
                                   std::max(0.45f, cellH - 0.25f)},
                         signedColor(values[i], scale, theme));
    }

    if (highDetail) {
        drawGridGuides(bounds, grid, theme, highDetail);
        DrawRectangleLinesEx(bounds, 1.0f, withAlpha(theme.panelBorder, 230));
        drawTextMono(theme, "0",
                     static_cast<int>(bounds.x + 4.0f),
                     static_cast<int>(bounds.y + 3.0f),
                     10, withAlpha(theme.textMuted, 150));
        drawTextMono(theme, "end",
                     static_cast<int>(bounds.x + bounds.width - 30.0f),
                     static_cast<int>(bounds.y + bounds.height - 15.0f),
                     10, withAlpha(theme.textMuted, 150));
    }
}

void drawInspector(Rectangle bounds,
                   const std::string& title,
                   const std::string& shape,
                   const std::vector<float>& values,
                   const std::vector<std::size_t>& shapeDims,
                   const std::string& note,
                   const Theme& theme) {
    DrawRectangleRec(bounds, theme.panelBackground);
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);

    const TensorStats stats = computeStats(values);
    const std::string titleText =
        fitText(theme, title, 18, false, bounds.width - 20.0f);
    drawText(theme, titleText.c_str(),
             static_cast<int>(bounds.x + 10.0f),
             static_cast<int>(bounds.y + 8.0f),
             18, theme.textPrimary);

    std::string meta = shape;
    if (!note.empty()) {
        meta += "  |  ";
        meta += note;
    }
    meta = fitText(theme, meta, 11, true, bounds.width - 20.0f);
    drawTextMono(theme, meta.c_str(),
                 static_cast<int>(bounds.x + 10.0f),
                 static_cast<int>(bounds.y + 33.0f),
                 11, theme.textMuted);

    const float pad = 10.0f;
    const float top = bounds.y + 56.0f;
    const float contentH = std::max(20.0f, bounds.height - 66.0f);
    const bool wide = bounds.width >= 560.0f;
    Rectangle heat{};
    Rectangle side{};
    if (wide) {
        side = Rectangle{bounds.x + bounds.width - std::min(250.0f, bounds.width * 0.31f) - pad,
                         top,
                         std::min(250.0f, bounds.width * 0.31f),
                         contentH};
        heat = Rectangle{bounds.x + pad, top,
                         side.x - bounds.x - 2.0f * pad,
                         contentH};
    } else {
        heat = Rectangle{bounds.x + pad, top,
                         bounds.width - 2.0f * pad,
                         contentH * 0.60f};
        side = Rectangle{bounds.x + pad, heat.y + heat.height + 8.0f,
                         bounds.width - 2.0f * pad,
                         contentH - heat.height - 8.0f};
    }

    drawTensorPreview(values, shapeDims, heat, theme, stats.maxAbs, true);

    const float legendH = 42.0f;
    const float badgeH = 47.0f;
    drawDivergingLegend(Rectangle{side.x, side.y, side.width, legendH},
                        stats.maxAbs, theme);
    drawHistogram(Rectangle{side.x, side.y + legendH + 10.0f,
                            side.width,
                            std::max(36.0f, side.height - legendH - badgeH - 30.0f)},
                  values, stats.maxAbs, theme);
    drawStatsBadges(Rectangle{side.x, side.y + side.height - badgeH,
                              side.width, badgeH},
                    stats, theme);
}

void drawHistogram(Rectangle bounds,
                   const std::vector<float>& values,
                   float scale,
                   const Theme& theme) {
    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 180));
    DrawRectangleLinesEx(bounds, 1.0f, withAlpha(theme.panelBorder, 210));
    drawTextMono(theme, "distribution",
                 static_cast<int>(bounds.x + 7.0f),
                 static_cast<int>(bounds.y + 5.0f),
                 11, theme.textMuted);
    if (values.empty()) return;

    if (scale < 1e-6f) scale = computeStats(values).maxAbs;
    if (scale < 1e-6f) scale = 1.0f;

    constexpr int kBins = 32;
    int bins[kBins] = {};
    int maxBin = 1;
    for (float v : values) {
        float normalized = (std::clamp(v, -scale, scale) + scale) / (2.0f * scale);
        int idx = static_cast<int>(normalized * static_cast<float>(kBins));
        idx = std::clamp(idx, 0, kBins - 1);
        maxBin = std::max(maxBin, ++bins[idx]);
    }

    const Rectangle plot{bounds.x + 7.0f, bounds.y + 24.0f,
                         bounds.width - 14.0f, bounds.height - 31.0f};
    const float zeroX = plot.x + plot.width * 0.5f;
    DrawLineV(Vector2{zeroX, plot.y}, Vector2{zeroX, plot.y + plot.height},
              withAlpha(theme.textDim, 95));
    const float binW = plot.width / static_cast<float>(kBins);
    for (int i = 0; i < kBins; ++i) {
        const float h = plot.height * static_cast<float>(bins[i]) /
                        static_cast<float>(maxBin);
        const float x = plot.x + static_cast<float>(i) * binW;
        const float center = -scale + (static_cast<float>(i) + 0.5f) *
                             (2.0f * scale / static_cast<float>(kBins));
        DrawRectangleRec(Rectangle{x, plot.y + plot.height - h,
                                   std::max(1.0f, binW - 1.0f), h},
                         signedColor(center, scale, theme));
    }
}

void drawDivergingLegend(Rectangle bounds,
                         float scale,
                         const Theme& theme) {
    DrawRectangleRec(bounds, withAlpha(theme.buttonIdle, 180));
    DrawRectangleLinesEx(bounds, 1.0f, withAlpha(theme.panelBorder, 210));

    const Rectangle ramp{bounds.x + 7.0f, bounds.y + 8.0f,
                         bounds.width - 14.0f, 9.0f};
    const int steps = std::max(8, static_cast<int>(ramp.width));
    for (int i = 0; i < steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps - 1);
        const float v = -scale + 2.0f * scale * t;
        DrawRectangleRec(Rectangle{ramp.x + t * ramp.width,
                                   ramp.y,
                                   std::max(1.0f, ramp.width / static_cast<float>(steps) + 1.0f),
                                   ramp.height},
                         signedColor(v, scale, theme));
    }
    char lo[32];
    char mid[16];
    char hi[32];
    std::snprintf(lo, sizeof(lo), "-%.2f", scale);
    std::snprintf(mid, sizeof(mid), "0");
    std::snprintf(hi, sizeof(hi), "+%.2f", scale);
    drawTextMono(theme, lo, static_cast<int>(ramp.x),
                 static_cast<int>(ramp.y + 17.0f), 10, theme.textMuted);
    const int midW = measureTextMono(theme, mid, 10);
    drawTextMono(theme, mid, static_cast<int>(ramp.x + (ramp.width - midW) * 0.5f),
                 static_cast<int>(ramp.y + 17.0f), 10, theme.textMuted);
    const int hiW = measureTextMono(theme, hi, 10);
    drawTextMono(theme, hi, static_cast<int>(ramp.x + ramp.width - static_cast<float>(hiW)),
                 static_cast<int>(ramp.y + 17.0f), 10, theme.textMuted);
}

void drawMetricBars(Rectangle bounds,
                    const std::vector<std::string>& labels,
                    const std::vector<float>& values,
                    float scale,
                    const Theme& theme) {
    DrawRectangleRec(bounds, theme.panelBackground);
    DrawRectangleLinesEx(bounds, 1.0f, theme.panelBorder);
    if (labels.empty() || values.empty()) return;

    if (scale < 1e-6f) {
        for (float v : values) scale = std::max(scale, std::fabs(v));
    }
    scale = std::max(scale, 1e-6f);
    bool signedRange = false;
    for (float v : values) signedRange = signedRange || v < 0.0f;

    const float pad = 8.0f;
    const float rowH = std::max(15.0f,
        (bounds.height - 2.0f * pad) / static_cast<float>(values.size()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float y = bounds.y + pad + static_cast<float>(i) * rowH;
        drawTextMono(theme, labels[i].c_str(),
                     static_cast<int>(bounds.x + pad),
                     static_cast<int>(y + 1.0f),
                     11, theme.textMuted);
        const float labelW = 54.0f;
        const Rectangle track{bounds.x + labelW, y + 3.0f,
                              bounds.width - labelW - pad,
                              std::max(5.0f, rowH - 7.0f)};
        DrawRectangleRec(track, withAlpha(theme.buttonActive, 105));
        const float value = values[i];
        if (!signedRange) {
            DrawRectangleRec(Rectangle{track.x, track.y,
                                       track.width * clamp01(value / scale),
                                       track.height},
                             signedColor(value, scale, theme));
        } else if (value >= 0.0f) {
            const float half = track.width * 0.5f;
            DrawLineV(Vector2{track.x + half, track.y},
                      Vector2{track.x + half, track.y + track.height},
                      withAlpha(theme.textDim, 90));
            DrawRectangleRec(Rectangle{track.x + half, track.y,
                                       half * clamp01(value / scale),
                                       track.height},
                             signedColor(value, scale, theme));
        } else {
            const float half = track.width * 0.5f;
            DrawLineV(Vector2{track.x + half, track.y},
                      Vector2{track.x + half, track.y + track.height},
                      withAlpha(theme.textDim, 90));
            DrawRectangleRec(Rectangle{track.x + half - half * clamp01(-value / scale),
                                       track.y,
                                       half * clamp01(-value / scale),
                                       track.height},
                             signedColor(value, scale, theme));
        }
    }
}

void drawElbowConnection(Rectangle from,
                         Rectangle to,
                         Color color,
                         bool active,
                         bool dimmed,
                         const Theme& theme) {
    (void)theme;
    color.a = active ? 170 : (dimmed ? 36 : 74);
    const float thickness = active ? 2.2f : 1.1f;
    const float fromCy = from.y + from.height * 0.5f;
    const float toCy = to.y + to.height * 0.5f;

    Vector2 start{};
    Vector2 end{};
    if (std::fabs(fromCy - toCy) < std::min(from.height, to.height) * 0.75f) {
        if (to.x >= from.x) {
            start = Vector2{from.x + from.width + 1.0f, fromCy};
            end = Vector2{to.x - 1.0f, toCy};
        } else {
            start = Vector2{from.x - 1.0f, fromCy};
            end = Vector2{to.x + to.width + 1.0f, toCy};
        }
        drawLineSegment(start, end, thickness, color);
        drawArrowHead(end, start, color, active ? 9.0f : 7.0f);
        return;
    }

    if (to.y >= from.y) {
        start = Vector2{from.x + from.width * 0.5f, from.y + from.height + 1.0f};
        end = Vector2{to.x + to.width * 0.5f, to.y - 1.0f};
    } else {
        start = Vector2{from.x + from.width * 0.5f, from.y - 1.0f};
        end = Vector2{to.x + to.width * 0.5f, to.y + to.height + 1.0f};
    }

    const float midY = (start.y + end.y) * 0.5f;
    const Vector2 a{start.x, midY};
    const Vector2 b{end.x, midY};
    drawLineSegment(start, a, thickness, color);
    drawLineSegment(a, b, thickness, color);
    drawLineSegment(b, end, thickness, color);
    drawArrowHead(end, b, color, active ? 9.0f : 7.0f);
}

void drawNodeFanConnections(const std::vector<float>& fromValues,
                            const std::vector<std::size_t>& fromShape,
                            Rectangle fromBounds,
                            const std::vector<float>& toValues,
                            const std::vector<std::size_t>& toShape,
                            Rectangle toBounds,
                            Color color,
                            std::size_t maxEdges,
                            const Theme& theme) {
    if (fromValues.empty() || toValues.empty()) return;
    const std::size_t edges = fromValues.size() * toValues.size();
    if (edges == 0 || edges > maxEdges) {
        drawElbowConnection(fromBounds, toBounds, color, true, false, theme);
        return;
    }

    const GridSpec fromGrid = inferGrid(fromValues, fromShape, fromBounds);
    const GridSpec toGrid = inferGrid(toValues, toShape, toBounds);
    const float fromCellH = fromBounds.height / static_cast<float>(std::max(1, fromGrid.rows));
    const float toCellH = toBounds.height / static_cast<float>(std::max(1, toGrid.rows));

    Color edge = color;
    edge.a = edges > 1200 ? 14 : edges > 420 ? 22 : 36;
    for (std::size_t i = 0; i < fromValues.size(); ++i) {
        const Vector2 fp = cellPosition(i, fromGrid);
        const Vector2 start{
            fromBounds.x + fromBounds.width,
            fromBounds.y + (fp.y + 0.5f) * fromCellH,
        };
        for (std::size_t j = 0; j < toValues.size(); ++j) {
            const Vector2 tp = cellPosition(j, toGrid);
            const Vector2 end{
                toBounds.x,
                toBounds.y + (tp.y + 0.5f) * toCellH,
            };
            DrawLineV(start, end, edge);
        }
    }
}

}  // namespace cnnv::viz::tensorviz
