#include "viz/NnueView.h"

#include "nn/ActivationSnapshot.h"
#include "nn/nnue/NnueNetwork.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::nnue::snapshot_keys;

void copyFloats(const cnnv::nn::ActivationSnapshot& snap, const char* key,
                std::vector<float>& dst) {
    const float* p = snap.data(key);
    const std::size_t n = snap.size(key);
    if (p == nullptr || n == 0) {
        dst.clear();
        return;
    }
    dst.assign(p, p + n);
}

struct NodeLayerDraw {
    Rectangle bounds{};
    Rectangle nodeArea{};
    std::vector<Rectangle> nodes;
    int hover = -1;
};

float maxAbsValue(const std::vector<float>& values) {
    float m = 0.0f;
    for (float v : values) m = std::max(m, std::fabs(v));
    return std::max(m, 1e-6f);
}

float maxAbsRange(const float* data, std::size_t count) {
    float m = 0.0f;
    if (!data) return 1.0f;
    for (std::size_t i = 0; i < count; ++i) {
        m = std::max(m, std::fabs(data[i]));
    }
    return std::max(m, 1e-6f);
}

float clippedRelu(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

std::string fitMonoText(const Theme& theme,
                        std::string text,
                        int fontSize,
                        float maxWidth) {
    if (maxWidth <= 1.0f || text.empty()) return "";
    if (measureTextMono(theme, text.c_str(), fontSize) <= static_cast<int>(maxWidth)) {
        return text;
    }
    const std::string ellipsis = "...";
    while (!text.empty()) {
        text.pop_back();
        const std::string candidate = text + ellipsis;
        if (measureTextMono(theme, candidate.c_str(), fontSize) <=
            static_cast<int>(maxWidth)) {
            return candidate;
        }
    }
    return ellipsis;
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
    return square >= 0 && square < 64 ? names[square] : "??";
}

const char* planeName(int plane) {
    static const char* names[] = {
        "own pawn", "own knight", "own bishop", "own rook", "own queen",
        "enemy pawn", "enemy knight", "enemy bishop", "enemy rook", "enemy queen",
    };
    return plane >= 0 && plane < 10 ? names[plane] : "unknown";
}

std::string featureDescription(float featureValue, bool whitePerspective) {
    const int feature = static_cast<int>(std::round(featureValue));
    const int pieceSq = feature % 64;
    const int plane = (feature / 64) % 10;
    const int kingSq = feature / (10 * 64);
    const int boardPieceSq = whitePerspective ? pieceSq : (pieceSq ^ 56);
    const int boardKingSq = whitePerspective ? kingSq : (kingSq ^ 56);
    std::string text = whitePerspective ? "W " : "B ";
    text += "f";
    text += std::to_string(feature);
    text += " king ";
    text += squareName(boardKingSq);
    text += " ";
    text += planeName(plane);
    text += " ";
    text += squareName(boardPieceSq);
    return text;
}

int drawEdgeModeControl(Rectangle bounds, int mode, const Theme& theme) {
    const char* labels[] = {"all", "top", "+", "-", "sel", "impact"};
    constexpr int count = 6;
    mode = std::clamp(mode, 0, count - 1);
    const float w = bounds.width / static_cast<float>(count);
    const Vector2 mouse = GetMousePosition();
    for (int i = 0; i < count; ++i) {
        Rectangle r{bounds.x + static_cast<float>(i) * w, bounds.y,
                    w, bounds.height};
        if (CheckCollisionPointRec(mouse, r) &&
            IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            mode = i;
        }
    }

    DrawRectangleRec(bounds, tensorviz::withAlpha(theme.buttonIdle, 225));
    for (int i = 0; i < count; ++i) {
        Rectangle r{bounds.x + static_cast<float>(i) * w, bounds.y,
                    w, bounds.height};
        if (i == mode) DrawRectangleRec(r, theme.buttonActive);
        else if (CheckCollisionPointRec(mouse, r)) {
            DrawRectangleRec(r, tensorviz::withAlpha(theme.buttonHover, 130));
        }
        DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
        const int tw = measureText(theme, labels[i], 12);
        drawText(theme, labels[i],
                 static_cast<int>(r.x + (r.width - static_cast<float>(tw)) * 0.5f),
                 static_cast<int>(r.y + (r.height - 14.0f) * 0.5f),
                 12, i == mode ? theme.textPrimary : theme.textMuted);
    }
    return mode;
}

void drawRibbonStep(Rectangle bounds,
                    int step,
                    const char* title,
                    const char* formula,
                    const char* detail,
                    float activity,
                    Color accent,
                    const Theme& theme) {
    activity = std::clamp(activity, 0.0f, 1.0f);
    DrawRectangleRec(bounds, tensorviz::blend(theme.panelBackground, accent, 0.055f));
    DrawRectangleLinesEx(bounds, 1.0f, tensorviz::withAlpha(theme.panelBorder, 230));

    const Rectangle badge{bounds.x + 8.0f, bounds.y + 8.0f, 24.0f, 24.0f};
    DrawRectangleRec(badge, tensorviz::withAlpha(accent, 180));
    DrawRectangleLinesEx(badge, 1.0f, tensorviz::withAlpha(theme.textPrimary, 80));
    char indexText[8];
    std::snprintf(indexText, sizeof(indexText), "%d", step);
    const int indexW = measureText(theme, indexText, 13);
    drawText(theme, indexText,
             static_cast<int>(badge.x + (badge.width - static_cast<float>(indexW)) * 0.5f),
             static_cast<int>(badge.y + 4.0f),
             13, theme.textPrimary);

    const std::string titleText =
        fitMonoText(theme, title, 14, bounds.width - 46.0f);
    drawText(theme, titleText.c_str(),
             static_cast<int>(bounds.x + 40.0f),
             static_cast<int>(bounds.y + 7.0f),
             14, theme.textPrimary);

    const std::string formulaText =
        fitMonoText(theme, formula, 10, bounds.width - 16.0f);
    drawTextMono(theme, formulaText.c_str(),
                 static_cast<int>(bounds.x + 8.0f),
                 static_cast<int>(bounds.y + 39.0f),
                 10, theme.textMuted);

    const std::string detailText =
        fitMonoText(theme, detail, 10, bounds.width - 16.0f);
    drawTextMono(theme, detailText.c_str(),
                 static_cast<int>(bounds.x + 8.0f),
                 static_cast<int>(bounds.y + 56.0f),
                 10, theme.textDim);

    const Rectangle track{bounds.x + 8.0f, bounds.y + bounds.height - 9.0f,
                          bounds.width - 16.0f, 4.0f};
    DrawRectangleRec(track, tensorviz::withAlpha(theme.buttonActive, 110));
    DrawRectangleRec(Rectangle{track.x, track.y, track.width * activity, track.height},
                     tensorviz::blend(theme.accentBlue, accent, activity));
}

void drawNnueComputationRibbon(Rectangle bounds,
                               const std::vector<float>& featuresUs,
                               const std::vector<float>& featuresThem,
                               const std::vector<float>& accumUs,
                               const std::vector<float>& accumThem,
                               const std::vector<float>& clippedUs,
                               const std::vector<float>& clippedThem,
                               const std::vector<float>& outputContributionUs,
                               const std::vector<float>& outputContributionThem,
                               float centipawns,
                               bool clippedUsFromWhite,
                               const Theme& theme) {
    const float gap = 8.0f;
    const float stepW = (bounds.width - 4.0f * gap) / 5.0f;
    if (stepW < 92.0f || bounds.height < 58.0f) return;

    const tensorviz::TensorStats accumStats = tensorviz::computeStats(accumUs);
    const tensorviz::TensorStats accumThemStats = tensorviz::computeStats(accumThem);
    const tensorviz::TensorStats clippedStats = tensorviz::computeStats(clippedUs);
    const tensorviz::TensorStats clippedThemStats = tensorviz::computeStats(clippedThem);
    const float rawScale = std::max({accumStats.rms, accumThemStats.rms, 1e-6f});
    const float clipScale = std::max({clippedStats.rms, clippedThemStats.rms, 1e-6f});

    int liveClipped = 0;
    int saturated = 0;
    auto countClip = [&](const std::vector<float>& values) {
        for (float v : values) {
            if (v > 0.0f) ++liveClipped;
            if (v >= 0.999f) ++saturated;
        }
    };
    countClip(clippedUs);
    countClip(clippedThem);

    const float outputScale =
        std::max(maxAbsValue(outputContributionUs), maxAbsValue(outputContributionThem));

    char featureDetail[72];
    char rawDetail[72];
    char perspectiveDetail[72];
    char clipDetail[72];
    char outputDetail[72];
    std::snprintf(featureDetail, sizeof(featureDetail), "%zu white + %zu black active ids",
                  featuresUs.size(), featuresThem.size());
    std::snprintf(rawDetail, sizeof(rawDetail), "2 x %zu hidden raw nodes",
                  std::max(accumUs.size(), accumThem.size()));
    std::snprintf(perspectiveDetail, sizeof(perspectiveDetail), "%s accumulator maps to us",
                  clippedUsFromWhite ? "white" : "black");
    std::snprintf(clipDetail, sizeof(clipDetail), "%d live, %d saturated",
                  liveClipped, saturated);
    std::snprintf(outputDetail, sizeof(outputDetail), "%.1f centipawns", centipawns);

    Rectangle cards[5]{};
    for (int i = 0; i < 5; ++i) {
        cards[i] = Rectangle{bounds.x + static_cast<float>(i) * (stepW + gap),
                             bounds.y, stepW, bounds.height};
    }
    for (int i = 0; i < 4; ++i) {
        tensorviz::drawElbowConnection(cards[i], cards[i + 1],
                                       i < 2 ? theme.accentOrange
                                             : i == 2 ? theme.accentBlue
                                                      : theme.accentYellow,
                                       true, false, theme);
    }

    drawRibbonStep(cards[0], 1, "HalfKP features",
                   "feature = side + king square + piece plane + piece square",
                   featureDetail,
                   std::clamp(static_cast<float>(featuresUs.size() + featuresThem.size()) / 64.0f,
                              0.0f, 1.0f),
                   theme.accentGreen, theme);
    drawRibbonStep(cards[1], 2, "Affine accumulators",
                   "acc[h] = bias[h] + sum active w[f,h]",
                   rawDetail,
                   std::clamp(rawScale, 0.0f, 1.0f),
                   theme.accentOrange, theme);
    drawRibbonStep(cards[2], 3, "Perspective split",
                   "white/black accumulators become us/them",
                   perspectiveDetail,
                   0.65f,
                   theme.accentBlue, theme);
    drawRibbonStep(cards[3], 4, "Clipped ReLU",
                   "clip[h] = clamp(acc[h], 0, 1)",
                   clipDetail,
                   std::clamp(clipScale, 0.0f, 1.0f),
                   theme.accentMagenta, theme);
    drawRibbonStep(cards[4], 5, "Centipawn output",
                   "cp = scale * (bias + sum out[h] * clip[h])",
                   outputDetail,
                   std::clamp(outputScale / 40.0f, 0.0f, 1.0f),
                   theme.accentYellow, theme);
}

float clippedError(const std::vector<float>& raw,
                   const std::vector<float>& clipped) {
    const std::size_t n = std::min(raw.size(), clipped.size());
    if (n == 0) return 1e30f;
    double sum = static_cast<double>(std::max(raw.size(), clipped.size()) - n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(clippedRelu(raw[i]) - clipped[i]);
        sum += d * d;
    }
    return static_cast<float>(sum);
}

void drawActiveFeatureGroups(const NodeLayerDraw& layer,
                             std::size_t whiteCount,
                             const Theme& theme) {
    if (layer.nodes.empty() || whiteCount == 0 || whiteCount >= layer.nodes.size()) return;
    const std::size_t split = std::min<std::size_t>(whiteCount, layer.nodes.size() - 1);
    const Rectangle upper = layer.nodes[split - 1];
    const Rectangle lower = layer.nodes[split];
    const float y = (upper.y + upper.height + lower.y) * 0.5f;
    DrawLineEx(Vector2{layer.nodeArea.x + 3.0f, y},
               Vector2{layer.nodeArea.x + layer.nodeArea.width - 3.0f, y},
               2.0f, tensorviz::withAlpha(theme.selection, 180));

    auto drawTag = [&](const char* text, float tagY, Color accent) {
        const int font = 10;
        const int tw = measureTextMono(theme, text, font);
        Rectangle tag{layer.nodeArea.x + 6.0f, tagY,
                      static_cast<float>(tw) + 12.0f, 19.0f};
        DrawRectangleRec(tag, tensorviz::withAlpha(theme.buttonIdle, 225));
        DrawRectangleLinesEx(tag, 1.0f, tensorviz::withAlpha(accent, 185));
        drawTextMono(theme, text,
                     static_cast<int>(tag.x + 6.0f),
                     static_cast<int>(tag.y + 5.0f),
                     font, theme.textPrimary);
    };
    drawTag("white HalfKP", layer.nodeArea.y + 6.0f, theme.accentOrange);
    drawTag("black HalfKP", y + 6.0f, theme.accentBlue);
}

std::vector<Rectangle> layoutNodeRects(Rectangle area, std::size_t count) {
    std::vector<Rectangle> rects;
    rects.reserve(count);
    if (count == 0 || area.width <= 1.0f || area.height <= 1.0f) return rects;

    const int maxCols = std::max(1, static_cast<int>(area.width / 15.0f));
    int cols = 1;
    for (; cols < maxCols; ++cols) {
        const int rows = static_cast<int>((count + static_cast<std::size_t>(cols) - 1) /
                                          static_cast<std::size_t>(cols));
        if (area.height / static_cast<float>(rows) >= 12.0f) break;
    }
    const int rows = std::max(1, static_cast<int>(
        (count + static_cast<std::size_t>(cols) - 1) / static_cast<std::size_t>(cols)));
    const float cellW = area.width / static_cast<float>(cols);
    const float cellH = area.height / static_cast<float>(rows);
    for (std::size_t i = 0; i < count; ++i) {
        const int row = static_cast<int>(i / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(i % static_cast<std::size_t>(cols));
        const float inset = std::min(3.0f, std::max(1.0f, std::min(cellW, cellH) * 0.13f));
        rects.push_back(Rectangle{
            area.x + static_cast<float>(col) * cellW + inset,
            area.y + static_cast<float>(row) * cellH + inset,
            std::max(2.0f, cellW - 2.0f * inset),
            std::max(2.0f, cellH - 2.0f * inset),
        });
    }
    return rects;
}

Rectangle nodeAreaForPanel(Rectangle bounds) {
    return Rectangle{bounds.x + 8.0f, bounds.y + 50.0f,
                     bounds.width - 16.0f,
                     std::max(10.0f, bounds.height - 58.0f)};
}

Vector2 leftCenter(Rectangle r) {
    return Vector2{r.x, r.y + r.height * 0.5f};
}

Vector2 rightCenter(Rectangle r) {
    return Vector2{r.x + r.width, r.y + r.height * 0.5f};
}

struct WeightedEdge {
    std::size_t from = 0;
    std::size_t to = 0;
    float value = 0.0f;
};

bool shouldDrawWeightedEdge(int mode,
                            float value,
                            std::size_t from,
                            std::size_t to,
                            int selectedFrom,
                            int selectedTo) {
    switch (mode) {
        case 2: return value > 0.0f;
        case 3: return value < 0.0f;
        case 4:
            return (selectedFrom >= 0 && from == static_cast<std::size_t>(selectedFrom)) ||
                   (selectedTo >= 0 && to == static_cast<std::size_t>(selectedTo));
        case 5: return false;
        default: return true;
    }
}

void drawWeightedEdges(const std::vector<Rectangle>& from,
                       const std::vector<Rectangle>& to,
                       const std::vector<float>& weights,
                       int mode,
                       int selectedFrom,
                       int selectedTo,
                       const Theme& theme) {
    if (from.empty() || to.empty() ||
        weights.size() < from.size() * to.size()) {
        return;
    }

    const float scale = maxAbsRange(weights.data(), weights.size());
    std::vector<WeightedEdge> edges;
    if (mode == 1) {
        edges.reserve(from.size() * to.size());
        for (std::size_t f = 0; f < from.size(); ++f) {
            for (std::size_t t = 0; t < to.size(); ++t) {
                const float v = weights[f * to.size() + t];
                edges.push_back(WeightedEdge{f, t, v});
            }
        }
        const std::size_t keep = std::min<std::size_t>(180, edges.size());
        std::partial_sort(edges.begin(), edges.begin() + keep, edges.end(),
                          [](const WeightedEdge& a, const WeightedEdge& b) {
                              return std::fabs(a.value) > std::fabs(b.value);
                          });
        edges.resize(keep);
    }

    const std::size_t total = mode == 1 ? edges.size() : from.size() * to.size();
    const std::size_t target = mode == 0 ? 120000 : 45000;
    const std::size_t step = std::max<std::size_t>(1, total / target);
    std::size_t serial = 0;
    auto drawOne = [&](std::size_t f, std::size_t t, float v) {
        if (!shouldDrawWeightedEdge(mode, v, f, t, selectedFrom, selectedTo)) return;
        if ((serial++ % step) != 0) return;
        Color c = tensorviz::signedColor(v, scale, theme);
        const float strength = std::sqrt(std::clamp(std::fabs(v) / scale, 0.0f, 1.0f));
        c.a = static_cast<unsigned char>(mode == 4 ? 86 : 15 + 56 * strength);
        DrawLineEx(rightCenter(from[f]), leftCenter(to[t]),
                   mode == 4 ? 1.6f : 0.7f + 1.1f * strength, c);
    };

    if (mode == 1) {
        for (const WeightedEdge& e : edges) drawOne(e.from, e.to, e.value);
        return;
    }
    for (std::size_t f = 0; f < from.size(); ++f) {
        for (std::size_t t = 0; t < to.size(); ++t) {
            drawOne(f, t, weights[f * to.size() + t]);
        }
    }
}

void drawOutputEdges(const std::vector<Rectangle>& from,
                     const std::vector<Rectangle>& to,
                     const std::vector<float>& contributions,
                     int mode,
                     int selectedFrom,
                     bool selectedOutput,
                     const Theme& theme) {
    if (from.empty() || to.empty() || contributions.empty()) return;
    const float scale = maxAbsValue(contributions);
    std::vector<WeightedEdge> edges;
    for (std::size_t i = 0; i < std::min(from.size(), contributions.size()); ++i) {
        const float v = contributions[i];
        bool draw = true;
        if (mode == 1) {
            edges.push_back(WeightedEdge{i, 0, v});
            draw = false;
        } else if (mode == 2) draw = v > 0.0f;
        else if (mode == 3) draw = v < 0.0f;
        else if (mode == 4) draw = selectedOutput ||
            (selectedFrom >= 0 && i == static_cast<std::size_t>(selectedFrom));
        if (!draw) continue;
        Color c = tensorviz::signedColor(v, scale, theme);
        const float strength = std::sqrt(std::clamp(std::fabs(v) / scale, 0.0f, 1.0f));
        c.a = static_cast<unsigned char>((mode == 4 || mode == 5) ? 110 : 28 + 72 * strength);
        DrawLineEx(rightCenter(from[i]), leftCenter(to[0]),
                   (mode == 4 || mode == 5) ? 1.9f : 0.9f + 1.5f * strength, c);
    }
    if (mode != 1) return;

    const std::size_t keep = std::min<std::size_t>(80, edges.size());
    std::partial_sort(edges.begin(), edges.begin() + keep, edges.end(),
                      [](const WeightedEdge& a, const WeightedEdge& b) {
                          return std::fabs(a.value) > std::fabs(b.value);
                      });
    for (std::size_t i = 0; i < keep; ++i) {
        const WeightedEdge& e = edges[i];
        Color c = tensorviz::signedColor(e.value, scale, theme);
        const float strength = std::sqrt(std::clamp(std::fabs(e.value) / scale, 0.0f, 1.0f));
        c.a = static_cast<unsigned char>(42 + 88 * strength);
        DrawLineEx(rightCenter(from[e.from]), leftCenter(to[0]),
                   1.0f + 1.8f * strength, c);
    }
}

void drawNodePairs(const std::vector<Rectangle>& from,
                   const std::vector<Rectangle>& to,
                   Color color) {
    const std::size_t count = std::min(from.size(), to.size());
    if (count == 0) return;
    color.a = count > 128 ? 30 : 52;
    for (std::size_t i = 0; i < count; ++i) {
        DrawLineV(rightCenter(from[i]), leftCenter(to[i]), color);
    }
}

NodeLayerDraw drawNodeLayer(Rectangle bounds,
                            const char* title,
                            const char* detail,
                            const std::vector<float>& values,
                            const std::vector<std::string>& labels,
                            float scale,
                            Color accent,
                            int selectedIndex,
                            bool selectedLayer,
                            const std::vector<float>* impact,
                            const Theme& theme) {
    NodeLayerDraw out;
    out.bounds = bounds;
    DrawRectangleRec(bounds, tensorviz::withAlpha(theme.panelBackground, 236));
    DrawRectangleLinesEx(bounds, selectedLayer ? 2.0f : 1.0f,
                         selectedLayer ? theme.selection : theme.panelBorder);

    const std::string titleText =
        fitMonoText(theme, title, 15, bounds.width - 16.0f);
    drawText(theme, titleText.c_str(),
             static_cast<int>(bounds.x + 8.0f),
             static_cast<int>(bounds.y + 7.0f),
             15, theme.textPrimary);
    const std::string detailText =
        fitMonoText(theme, detail, 10, bounds.width * 0.68f);
    drawTextMono(theme, detailText.c_str(),
                 static_cast<int>(bounds.x + 8.0f),
                 static_cast<int>(bounds.y + 29.0f),
                 10, theme.textMuted);

    const tensorviz::TensorStats stats = tensorviz::computeStats(values);
    char statText[72];
    std::snprintf(statText, sizeof(statText), "n=%zu  rms=%.3f",
                  values.size(), stats.rms);
    const int statW = measureTextMono(theme, statText, 10);
    drawTextMono(theme, statText,
                 static_cast<int>(bounds.x + bounds.width - statW - 8.0f),
                 static_cast<int>(bounds.y + 29.0f),
                 10, theme.textDim);

    out.nodeArea = nodeAreaForPanel(bounds);
    DrawRectangleRec(out.nodeArea, tensorviz::withAlpha(theme.buttonIdle, 120));
    out.nodes = layoutNodeRects(out.nodeArea, values.size());

    const Vector2 mouse = GetMousePosition();
    for (std::size_t i = 0; i < out.nodes.size(); ++i) {
        const Rectangle node = out.nodes[i];
        const bool hover = CheckCollisionPointRec(mouse, node);
        const bool selected = selectedIndex == static_cast<int>(i);
        if (hover) out.hover = static_cast<int>(i);
        Color fill = values[i] == 0.0f
            ? tensorviz::blend(theme.buttonIdle, accent, 0.22f)
            : tensorviz::signedColor(values[i], scale, theme);
        if (title[0] == 'a') {
            fill = values[i] < 0.0f
                ? tensorviz::blend(theme.buttonIdle, theme.accentBlue, 0.72f)
                : tensorviz::blend(theme.buttonIdle, theme.accentOrange, 0.72f);
        }
        DrawRectangleRec(node, fill);
        DrawRectangleLinesEx(node, (hover || selected) ? 2.0f : 1.0f,
                             selected ? theme.selection
                                      : hover ? theme.selection
                                              : tensorviz::withAlpha(theme.panelBorder, 150));
        if (impact && i < impact->size() && node.width >= 5.0f && node.height >= 5.0f) {
            const float impactScale = maxAbsValue(*impact);
            Color impactColor =
                tensorviz::signedColor((*impact)[i], impactScale, theme);
            impactColor.a = 225;
            DrawRectangleRec(Rectangle{node.x, node.y + node.height - 3.0f,
                                       node.width, 3.0f},
                             impactColor);
        }
        if (node.width >= 24.0f && node.height >= 13.0f) {
            std::string text = (i < labels.size() && labels[i].size() <= 5)
                ? labels[i]
                : std::to_string(i);
            if (text.size() > 5) text.resize(5);
            const int font = node.height >= 18.0f ? 10 : 9;
            const int tw = measureTextMono(theme, text.c_str(), font);
            drawTextMono(theme, text.c_str(),
                         static_cast<int>(node.x + (node.width - tw) * 0.5f),
                         static_cast<int>(node.y + (node.height - font) * 0.5f),
                         font, theme.textPrimary);
        }
    }

    return out;
}

std::vector<std::string> sequentialLabels(std::size_t count, const char* prefix) {
    std::vector<std::string> labels;
    labels.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        labels.emplace_back(std::string(prefix) + std::to_string(i));
    }
    return labels;
}

void updateNodeSelection(int layer,
                         const std::vector<Rectangle>& nodes,
                         Rectangle panel,
                         int& selectedLayer,
                         int& selectedNode) {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return;
    const Vector2 mouse = GetMousePosition();
    if (!CheckCollisionPointRec(mouse, panel)) return;
    selectedLayer = layer;
    selectedNode = -1;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (CheckCollisionPointRec(mouse, nodes[i])) {
            selectedNode = static_cast<int>(i);
            return;
        }
    }
}

std::string selectedNodeSummary(int selectedLayer,
                                int selectedNode,
                                const std::vector<float>& activeNodes,
                                const std::vector<std::string>& activeLabels,
                                const std::vector<float>& accumUs,
                                const std::vector<std::string>& hiddenLabels,
                                const std::vector<float>& accumThem,
                                const std::vector<float>& clippedUs,
                                const std::vector<std::string>& usLabels,
                                const std::vector<float>& clippedThem,
                                const std::vector<std::string>& themLabels,
                                const std::vector<float>& outputValues,
                                const std::vector<std::string>& outputLabels) {
    const char* names[] = {
        "active features", "accumulator white", "accumulator black",
        "clipped us", "clipped them", "linear output",
    };
    const std::vector<float>* values[] = {
        &activeNodes, &accumUs, &accumThem, &clippedUs, &clippedThem, &outputValues,
    };
    const std::vector<std::string>* labels[] = {
        &activeLabels, &hiddenLabels, &hiddenLabels, &usLabels, &themLabels, &outputLabels,
    };
    if (selectedLayer < 0 || selectedLayer >= 6) return "selected: none";
    std::string out = std::string("selected: ") + names[selectedLayer];
    const auto& v = *values[selectedLayer];
    const auto& l = *labels[selectedLayer];
    if (selectedNode >= 0 && static_cast<std::size_t>(selectedNode) < v.size()) {
        out += "  node ";
        out += static_cast<std::size_t>(selectedNode) < l.size()
            ? l[static_cast<std::size_t>(selectedNode)]
            : std::to_string(selectedNode);
        char value[48];
        std::snprintf(value, sizeof(value), " = %.5f",
                      v[static_cast<std::size_t>(selectedNode)]);
        out += value;
    }
    return out;
}

std::string selectedNodeNote(int selectedLayer,
                             int selectedNode,
                             const std::vector<float>& activeNodes,
                             const std::vector<std::string>& activeLabels,
                             const std::vector<float>& accumUs,
                             const std::vector<std::string>& hiddenLabels,
                             const std::vector<float>& accumThem,
                             const std::vector<float>& clippedUs,
                             const std::vector<std::string>& usLabels,
                             const std::vector<float>& clippedThem,
                             const std::vector<std::string>& themLabels,
                             const std::vector<float>& outputValues,
                             const std::vector<std::string>& outputLabels) {
    const std::vector<float>* values[] = {
        &activeNodes, &accumUs, &accumThem, &clippedUs, &clippedThem, &outputValues,
    };
    const std::vector<std::string>* labels[] = {
        &activeLabels, &hiddenLabels, &hiddenLabels, &usLabels, &themLabels, &outputLabels,
    };
    if (selectedLayer < 0 || selectedLayer >= 6) return "no node selected";
    const auto& v = *values[selectedLayer];
    const auto& l = *labels[selectedLayer];
    if (selectedNode < 0 || static_cast<std::size_t>(selectedNode) >= v.size()) {
        return "click a node for its exact value";
    }

    std::string label = static_cast<std::size_t>(selectedNode) < l.size()
        ? l[static_cast<std::size_t>(selectedNode)]
        : std::to_string(selectedNode);
    char value[96];
    std::snprintf(value, sizeof(value), "node %s = %.5f",
                  label.c_str(), v[static_cast<std::size_t>(selectedNode)]);
    return value;
}

std::vector<std::string> topFeatureContributors(
    const std::vector<float>& features,
    const std::vector<std::string>& labels,
    const std::vector<float>& activeWeights,
    std::size_t hiddenSize,
    int hiddenNode,
    const char* prefix) {
    std::vector<std::string> lines;
    if (hiddenNode < 0 || hiddenSize == 0 ||
        activeWeights.size() < features.size() * hiddenSize) {
        return lines;
    }
    struct Item {
        std::size_t index = 0;
        float value = 0.0f;
    };
    std::vector<Item> items;
    items.reserve(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) {
        items.push_back(Item{
            i,
            activeWeights[i * hiddenSize + static_cast<std::size_t>(hiddenNode)],
        });
    }
    const std::size_t keep = std::min<std::size_t>(4, items.size());
    std::partial_sort(items.begin(), items.begin() + keep, items.end(),
                      [](const Item& a, const Item& b) {
                          return std::fabs(a.value) > std::fabs(b.value);
                      });
    lines.emplace_back(prefix);
    for (std::size_t i = 0; i < keep; ++i) {
        const Item& item = items[i];
        char text[160];
        const std::string& label =
            item.index < labels.size() ? labels[item.index] : std::to_string(item.index);
        std::snprintf(text, sizeof(text), "%s  %.5f",
                      label.c_str(), item.value);
        lines.emplace_back(text);
    }
    return lines;
}

std::vector<std::string> selectedDetailLines(
    int selectedLayer,
    int selectedNode,
    bool clippedUsFromWhite,
    const std::vector<float>& featuresUs,
    const std::vector<std::string>& whiteLabels,
    const std::vector<float>& featureWeightsUs,
    const std::vector<float>& featuresThem,
    const std::vector<std::string>& blackLabels,
    const std::vector<float>& featureWeightsThem,
    const std::vector<float>& featureBias,
    const std::vector<float>& accumUs,
    const std::vector<float>& accumThem,
    const std::vector<float>& clippedUs,
    const std::vector<float>& clippedThem,
    const std::vector<float>& outputWeightsUs,
    const std::vector<float>& outputWeightsThem,
    const std::vector<float>& outputContributionUs,
    const std::vector<float>& outputContributionThem,
    const std::vector<float>& outputAffine) {
    std::vector<std::string> lines;
    const std::size_t hiddenSize = std::max(accumUs.size(), accumThem.size());
    auto pushLine = [&](const char* fmt, float a, float b = 0.0f, float c = 0.0f) {
        char text[128];
        std::snprintf(text, sizeof(text), fmt, a, b, c);
        lines.emplace_back(text);
    };

    if (selectedLayer == 0 && selectedNode >= 0) {
        const bool whiteFeature = static_cast<std::size_t>(selectedNode) < featuresUs.size();
        const std::size_t idx = whiteFeature
            ? static_cast<std::size_t>(selectedNode)
            : static_cast<std::size_t>(selectedNode) - featuresUs.size();
        const std::vector<float>& weights = whiteFeature ? featureWeightsUs
                                                         : featureWeightsThem;
        const std::vector<std::string>& labels = whiteFeature ? whiteLabels
                                                              : blackLabels;
        if (idx < labels.size()) lines.push_back(labels[idx]);
        if (weights.size() >= (idx + 1) * hiddenSize && hiddenSize > 0) {
            struct Item { std::size_t h; float w; };
            std::vector<Item> items;
            items.reserve(hiddenSize);
            for (std::size_t h = 0; h < hiddenSize; ++h) {
                items.push_back(Item{h, weights[idx * hiddenSize + h]});
            }
            const std::size_t keep = std::min<std::size_t>(4, items.size());
            std::partial_sort(items.begin(), items.begin() + keep, items.end(),
                              [](const Item& a, const Item& b) {
                                  return std::fabs(a.w) > std::fabs(b.w);
                              });
            lines.emplace_back("strongest hidden weights");
            for (std::size_t i = 0; i < keep; ++i) {
                char text[96];
                std::snprintf(text, sizeof(text), "h%zu  %.5f",
                              items[i].h, items[i].w);
                lines.emplace_back(text);
            }
        }
        return lines;
    }

    auto rawLines = [&](bool whiteAccumulator) {
        const std::vector<float>& features = whiteAccumulator ? featuresUs : featuresThem;
        const std::vector<std::string>& labels = whiteAccumulator ? whiteLabels : blackLabels;
        const std::vector<float>& weights = whiteAccumulator ? featureWeightsUs
                                                            : featureWeightsThem;
        const std::vector<float>& raw = whiteAccumulator ? accumUs : accumThem;
        if (selectedNode >= 0 &&
            static_cast<std::size_t>(selectedNode) < raw.size()) {
            const std::size_t h = static_cast<std::size_t>(selectedNode);
            const float bias = h < featureBias.size() ? featureBias[h] : 0.0f;
            pushLine("raw %.5f  bias %.5f", raw[h], bias);
            const bool mapsToUs = (whiteAccumulator && clippedUsFromWhite) ||
                                  (!whiteAccumulator && !clippedUsFromWhite);
            const std::vector<float>& contrib =
                mapsToUs ? outputContributionUs : outputContributionThem;
            if (h < contrib.size()) pushLine("output impact %.5f", contrib[h]);
            auto top = topFeatureContributors(features, labels, weights,
                                              raw.size(), selectedNode,
                                              "top feature contributors");
            lines.insert(lines.end(), top.begin(), top.end());
        }
    };

    if (selectedLayer == 1) {
        rawLines(true);
    } else if (selectedLayer == 2) {
        rawLines(false);
    } else if ((selectedLayer == 3 || selectedLayer == 4) && selectedNode >= 0) {
        const bool us = selectedLayer == 3;
        const std::vector<float>& clipped = us ? clippedUs : clippedThem;
        const std::vector<float>& outW = us ? outputWeightsUs : outputWeightsThem;
        const std::vector<float>& contrib = us ? outputContributionUs : outputContributionThem;
        const std::size_t h = static_cast<std::size_t>(selectedNode);
        if (h < clipped.size()) pushLine("clipped %.5f", clipped[h]);
        if (h < outW.size()) pushLine("output weight %.5f", outW[h]);
        if (h < contrib.size()) pushLine("output impact %.5f cp", contrib[h]);
    } else if (selectedLayer == 5) {
        if (outputAffine.size() >= 4) {
            pushLine("bias %.5f  scale %.5f", outputAffine[0], outputAffine[1]);
            pushLine("raw %.5f  cp %.5f", outputAffine[2], outputAffine[3]);
        }
        struct Item { const char* side; std::size_t h; float v; };
        std::vector<Item> items;
        for (std::size_t h = 0; h < outputContributionUs.size(); ++h) {
            items.push_back(Item{"us", h, outputContributionUs[h]});
        }
        for (std::size_t h = 0; h < outputContributionThem.size(); ++h) {
            items.push_back(Item{"them", h, outputContributionThem[h]});
        }
        const std::size_t keep = std::min<std::size_t>(5, items.size());
        std::partial_sort(items.begin(), items.begin() + keep, items.end(),
                          [](const Item& a, const Item& b) {
                              return std::fabs(a.v) > std::fabs(b.v);
                          });
        lines.emplace_back("top output impacts");
        for (std::size_t i = 0; i < keep; ++i) {
            char text[96];
            std::snprintf(text, sizeof(text), "%s h%zu  %.5f cp",
                          items[i].side, items[i].h, items[i].v);
            lines.emplace_back(text);
        }
    }
    return lines;
}

void drawNnueAbstract(Rectangle bounds, const Theme& theme,
                      const std::vector<float>& featuresUs,
                      const std::vector<float>& featuresThem,
                      const std::vector<float>& accumUs,
                      const std::vector<float>& accumThem,
                      const std::vector<float>& clippedUs,
                      const std::vector<float>& clippedThem,
                      float centipawns) {
    std::vector<float> activeNodes;
    activeNodes.reserve(featuresUs.size() + featuresThem.size());
    activeNodes.insert(activeNodes.end(), featuresUs.size(), 1.0f);
    activeNodes.insert(activeNodes.end(), featuresThem.size(), -1.0f);

    const tensorviz::TensorStats activeStats = tensorviz::computeStats(activeNodes);
    const tensorviz::TensorStats accumStats = tensorviz::computeStats(accumUs);
    const tensorviz::TensorStats accumThemStats = tensorviz::computeStats(accumThem);
    const tensorviz::TensorStats clipStats = tensorviz::computeStats(clippedUs);
    const tensorviz::TensorStats clipThemStats = tensorviz::computeStats(clippedThem);
    const float scale = std::max({activeStats.rms, accumStats.rms,
                                  accumThemStats.rms, clipStats.rms,
                                  clipThemStats.rms, 1e-6f});

    constexpr float kHeaderH = 48.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "NNUE abstract flow",
        "sparse features -> hidden accumulators -> clipped pair -> scalar",
        theme);

    const float gap = 10.0f;
    const float x = bounds.x + 8.0f;
    const float y = bounds.y + kHeaderH + 14.0f;
    const float w = bounds.width - 16.0f;
    const float h = std::max(240.0f, bounds.height - kHeaderH - 24.0f);
    const float flowH = std::min(h, 520.0f);
    const float colW = (w - 3.0f * gap) * 0.25f;
    const float stackGap = 10.0f;
    const float stackH = std::clamp((flowH - stackGap) * 0.5f, 105.0f, 210.0f);
    const float singleH = std::clamp(flowH * 0.42f, 96.0f, 170.0f);
    const Rectangle active{x, y + (2.0f * stackH + stackGap - singleH) * 0.5f,
                           colW, singleH};
    const Rectangle accumW{x + colW + gap, y, colW, stackH};
    const Rectangle accumB{x + colW + gap, y + stackH + stackGap, colW, stackH};
    const Rectangle clippedUsR{x + 2.0f * (colW + gap), y, colW, stackH};
    const Rectangle clippedThemR{x + 2.0f * (colW + gap), y + stackH + stackGap, colW, stackH};
    const Rectangle output{x + 3.0f * (colW + gap),
                           y + (2.0f * stackH + stackGap - singleH) * 0.5f,
                           colW, singleH};

    tensorviz::drawElbowConnection(active, accumW, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(active, accumB, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(accumW, clippedUsR, theme.accentBlue, true, false, theme);
    tensorviz::drawElbowConnection(accumB, clippedThemR, theme.accentMagenta, true, false, theme);
    tensorviz::drawElbowConnection(clippedUsR, output, theme.accentYellow, true, false, theme);
    tensorviz::drawElbowConnection(clippedThemR, output, theme.accentYellow, true, false, theme);

    char activeDetail[64];
    std::snprintf(activeDetail, sizeof(activeDetail), "white %zu | black %zu",
                  featuresUs.size(), featuresThem.size());
    tensorviz::drawAbstractBlock(active, "active features", activeDetail,
                                 activeStats.rms / scale, theme.accentGreen,
                                 false, theme);
    tensorviz::drawAbstractBlock(accumW, "accumulator W",
                                 std::to_string(accumUs.size()) + " hidden",
                                 accumStats.rms / scale, theme.accentOrange,
                                 false, theme);
    tensorviz::drawAbstractBlock(accumB, "accumulator B",
                                 std::to_string(accumThem.size()) + " hidden",
                                 accumThemStats.rms / scale, theme.accentOrange,
                                 false, theme);
    tensorviz::drawAbstractBlock(clippedUsR, "clipped us",
                                 std::to_string(clippedUs.size()) + " hidden",
                                 clipStats.rms / scale, theme.accentBlue,
                                 false, theme);
    tensorviz::drawAbstractBlock(clippedThemR, "clipped them",
                                 std::to_string(clippedThem.size()) + " hidden",
                                 clipThemStats.rms / scale, theme.accentMagenta,
                                 false, theme);
    char cp[48];
    std::snprintf(cp, sizeof(cp), "%.1f cp", centipawns);
    tensorviz::drawAbstractBlock(output, "linear output", cp,
                                 std::min(1.0f, std::fabs(centipawns) / 800.0f),
                                 theme.accentYellow, false, theme);

    const float summaryY = y + 2.0f * stackH + stackGap + 16.0f;
    const float summaryH = bounds.y + bounds.height - summaryY;
    if (summaryH >= 84.0f) {
        const float panelGap = 10.0f;
        const float metricW = std::min(340.0f, (w - panelGap) * 0.45f);
        tensorviz::drawMetricBars(
            Rectangle{x, summaryY, metricW, std::min(128.0f, summaryH)},
            std::vector<std::string>{"white", "black"},
            std::vector<float>{static_cast<float>(featuresUs.size()),
                               static_cast<float>(featuresThem.size())},
            std::max(1.0f, static_cast<float>(
                std::max(featuresUs.size(), featuresThem.size()))),
            theme);
        tensorviz::drawMetricBars(
            Rectangle{x + metricW + panelGap, summaryY,
                      w - metricW - panelGap, std::min(128.0f, summaryH)},
            std::vector<std::string>{"acc W", "acc B", "clip U", "clip T"},
            std::vector<float>{accumStats.rms, accumThemStats.rms,
                               clipStats.rms, clipThemStats.rms},
            scale, theme);
    }
}

void drawNnueLayerStack(Rectangle bounds, const Theme& theme,
                        const std::vector<float>& featuresUs,
                        const std::vector<float>& featuresThem,
                        const std::vector<float>& accumUs,
                        const std::vector<float>& accumThem,
                        const std::vector<float>& clippedUs,
                        const std::vector<float>& clippedThem,
                        const std::vector<float>& featureBias,
                        const std::vector<float>& featureWeightsUs,
                        const std::vector<float>& featureWeightsThem,
                        const std::vector<float>& outputWeightsUs,
                        const std::vector<float>& outputWeightsThem,
                        const std::vector<float>& outputContributionUs,
                        const std::vector<float>& outputContributionThem,
                        const std::vector<float>& outputAffine,
                        float centipawns,
                        int& selectedLayer,
                        int& selectedNode,
                        int& edgeMode) {
    std::vector<float> activeNodes;
    activeNodes.reserve(featuresUs.size() + featuresThem.size());
    activeNodes.insert(activeNodes.end(), featuresUs.size(), 1.0f);
    activeNodes.insert(activeNodes.end(), featuresThem.size(), -1.0f);
    std::vector<std::string> activeLabels;
    std::vector<std::string> whiteFeatureLabels;
    std::vector<std::string> blackFeatureLabels;
    activeLabels.reserve(activeNodes.size());
    for (float id : featuresUs) {
        whiteFeatureLabels.push_back(featureDescription(id, true));
        activeLabels.push_back(whiteFeatureLabels.back());
    }
    for (float id : featuresThem) {
        blackFeatureLabels.push_back(featureDescription(id, false));
        activeLabels.push_back(blackFeatureLabels.back());
    }
    const std::vector<std::string> hiddenLabels =
        sequentialLabels(std::max(accumUs.size(), accumThem.size()), "h");
    const std::vector<std::string> usLabels =
        sequentialLabels(clippedUs.size(), "u");
    const std::vector<std::string> themLabels =
        sequentialLabels(clippedThem.size(), "t");
    const std::vector<float> outputValues{centipawns};
    const std::vector<std::string> outputLabels{"cp"};
    const bool clippedUsFromWhite =
        clippedError(accumUs, clippedUs) <= clippedError(accumThem, clippedUs);

    const bool compactHeader = bounds.width < 760.0f;
    const float kHeaderH = compactHeader ? 86.0f : 58.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "detailed NNUE node graph",
        compactHeader
            ? "HalfKP -> affine accumulators -> clipped ReLU -> centipawns"
            : "HalfKP sparse features | affine accumulators | perspective split | clipped ReLU | centipawn head",
        theme);
    const Rectangle edgeControlRect = compactHeader
        ? Rectangle{bounds.x + 8.0f, bounds.y + 50.0f,
                    std::min(326.0f, bounds.width - 16.0f), 28.0f}
        : Rectangle{bounds.x + bounds.width - 336.0f, bounds.y + 16.0f,
                    326.0f, 28.0f};
    edgeMode = drawEdgeModeControl(
        edgeControlRect,
        edgeMode, theme);

    const float gap = 8.0f;
    const float ribbonH = bounds.height >= 700.0f ? 88.0f : 76.0f;
    drawNnueComputationRibbon(
        Rectangle{bounds.x, bounds.y + kHeaderH + 8.0f, bounds.width, ribbonH},
        featuresUs, featuresThem, accumUs, accumThem, clippedUs, clippedThem,
        outputContributionUs, outputContributionThem, centipawns,
        clippedUsFromWhite, theme);

    float inspectorH = std::clamp(bounds.height * 0.31f, 185.0f, 285.0f);
    inspectorH = std::min(inspectorH, std::max(140.0f, bounds.height - 260.0f));
    const float top = bounds.y + kHeaderH + ribbonH + 18.0f;
    const float graphH = std::max(
        180.0f, bounds.height - kHeaderH - ribbonH - 26.0f - inspectorH);
    const float colW = (bounds.width - 16.0f - 3.0f * gap) / 4.0f;
    const float rowH = (graphH - gap) * 0.5f;
    const float x = bounds.x + 8.0f;
    const Rectangle panels[] = {
        Rectangle{x, top, colW, rowH * 2.0f + gap},
        Rectangle{x + colW + gap, top, colW, rowH},
        Rectangle{x + colW + gap, top + rowH + gap, colW, rowH},
        Rectangle{x + 2.0f * (colW + gap), top, colW, rowH},
        Rectangle{x + 2.0f * (colW + gap), top + rowH + gap, colW, rowH},
        Rectangle{x + 3.0f * (colW + gap), top + (rowH + gap) * 0.5f, colW, rowH},
    };

    const Rectangle activeArea = nodeAreaForPanel(panels[0]);
    const Rectangle accumWArea = nodeAreaForPanel(panels[1]);
    const Rectangle accumBArea = nodeAreaForPanel(panels[2]);
    const Rectangle clippedUsArea = nodeAreaForPanel(panels[3]);
    const Rectangle clippedThemArea = nodeAreaForPanel(panels[4]);
    const Rectangle outputArea = nodeAreaForPanel(panels[5]);

    const std::vector<Rectangle> activeRects =
        layoutNodeRects(activeArea, activeNodes.size());
    std::vector<Rectangle> whiteActiveRects;
    std::vector<Rectangle> blackActiveRects;
    whiteActiveRects.reserve(featuresUs.size());
    blackActiveRects.reserve(featuresThem.size());
    for (std::size_t i = 0; i < activeRects.size(); ++i) {
        if (i < featuresUs.size()) whiteActiveRects.push_back(activeRects[i]);
        else blackActiveRects.push_back(activeRects[i]);
    }
    const std::vector<Rectangle> accumWRects =
        layoutNodeRects(accumWArea, accumUs.size());
    const std::vector<Rectangle> accumBRects =
        layoutNodeRects(accumBArea, accumThem.size());
    const std::vector<Rectangle> clippedUsRects =
        layoutNodeRects(clippedUsArea, clippedUs.size());
    const std::vector<Rectangle> clippedThemRects =
        layoutNodeRects(clippedThemArea, clippedThem.size());
    const std::vector<Rectangle> outputRects =
        layoutNodeRects(outputArea, outputValues.size());

    if (selectedLayer < 0 || selectedLayer >= 6) {
        selectedLayer = 3;
        selectedNode = -1;
    }
    updateNodeSelection(0, activeRects, panels[0], selectedLayer, selectedNode);
    updateNodeSelection(1, accumWRects, panels[1], selectedLayer, selectedNode);
    updateNodeSelection(2, accumBRects, panels[2], selectedLayer, selectedNode);
    updateNodeSelection(3, clippedUsRects, panels[3], selectedLayer, selectedNode);
    updateNodeSelection(4, clippedThemRects, panels[4], selectedLayer, selectedNode);
    updateNodeSelection(5, outputRects, panels[5], selectedLayer, selectedNode);

    const int selectedWhiteFeature =
        selectedLayer == 0 && selectedNode >= 0 &&
        static_cast<std::size_t>(selectedNode) < featuresUs.size()
            ? selectedNode : -1;
    const int selectedBlackFeature =
        selectedLayer == 0 && selectedNode >= 0 &&
        static_cast<std::size_t>(selectedNode) >= featuresUs.size()
            ? selectedNode - static_cast<int>(featuresUs.size()) : -1;
    drawWeightedEdges(whiteActiveRects, accumWRects, featureWeightsUs,
                      edgeMode, selectedWhiteFeature,
                      selectedLayer == 1 ? selectedNode : -1, theme);
    drawWeightedEdges(blackActiveRects, accumBRects, featureWeightsThem,
                      edgeMode, selectedBlackFeature,
                      selectedLayer == 2 ? selectedNode : -1, theme);
    if (clippedUsFromWhite) {
        drawNodePairs(accumWRects, clippedUsRects, theme.accentBlue);
        drawNodePairs(accumBRects, clippedThemRects, theme.accentMagenta);
    } else {
        drawNodePairs(accumBRects, clippedUsRects, theme.accentBlue);
        drawNodePairs(accumWRects, clippedThemRects, theme.accentMagenta);
    }
    drawOutputEdges(clippedUsRects, outputRects, outputContributionUs,
                    edgeMode, selectedLayer == 3 ? selectedNode : -1,
                    selectedLayer == 5, theme);
    drawOutputEdges(clippedThemRects, outputRects, outputContributionThem,
                    edgeMode, selectedLayer == 4 ? selectedNode : -1,
                    selectedLayer == 5, theme);

    char activeDetail[80];
    std::snprintf(activeDetail, sizeof(activeDetail), "white %zu | black %zu",
                  featuresUs.size(), featuresThem.size());
    const NodeLayerDraw activeDraw =
        drawNodeLayer(panels[0], "active HalfKP features", activeDetail,
                      activeNodes, activeLabels, 1.0f, theme.accentGreen,
                      selectedLayer == 0 ? selectedNode : -1,
                      selectedLayer == 0, nullptr, theme);
    drawActiveFeatureGroups(activeDraw, featuresUs.size(), theme);
    const std::vector<float>& whiteImpact = clippedUsFromWhite
        ? outputContributionUs : outputContributionThem;
    const std::vector<float>& blackImpact = clippedUsFromWhite
        ? outputContributionThem : outputContributionUs;
    drawNodeLayer(panels[1], "accumulator white", "acc[h]=bias[h]+sum active w[f,h]",
                  accumUs, hiddenLabels, maxAbsValue(accumUs),
                  theme.accentOrange,
                  selectedLayer == 1 ? selectedNode : -1,
                  selectedLayer == 1, &whiteImpact, theme);
    drawNodeLayer(panels[2], "accumulator black", "acc[h]=bias[h]+sum active w[f,h]",
                  accumThem, hiddenLabels, maxAbsValue(accumThem),
                  theme.accentOrange,
                  selectedLayer == 2 ? selectedNode : -1,
                  selectedLayer == 2, &blackImpact, theme);
    drawNodeLayer(panels[3], "clipped us", clippedUsFromWhite ? "from white accumulator"
                                                              : "from black accumulator",
                  clippedUs, usLabels, 1.0f, theme.accentBlue,
                  selectedLayer == 3 ? selectedNode : -1,
                  selectedLayer == 3, &outputContributionUs, theme);
    drawNodeLayer(panels[4], "clipped them", clippedUsFromWhite ? "from black accumulator"
                                                                : "from white accumulator",
                  clippedThem, themLabels, 1.0f, theme.accentMagenta,
                  selectedLayer == 4 ? selectedNode : -1,
                  selectedLayer == 4, &outputContributionThem, theme);
    char outputDetail[64];
    std::snprintf(outputDetail, sizeof(outputDetail), "%.1f centipawns", centipawns);
    drawNodeLayer(panels[5], "linear output", outputDetail,
                  outputValues, outputLabels,
                  std::max(800.0f, std::fabs(centipawns)),
                  theme.accentYellow,
                  selectedLayer == 5 ? selectedNode : -1,
                  selectedLayer == 5, nullptr, theme);

    const char* layerTitles[] = {
        "active HalfKP features", "accumulator white", "accumulator black",
        "clipped us", "clipped them", "linear output",
    };
    const char* layerShapes[] = {
        "sparse active features", "hidden raw", "hidden raw",
        "hidden clipped", "hidden clipped", "scalar",
    };
    const std::vector<float>* layerValues[] = {
        &activeNodes, &accumUs, &accumThem, &clippedUs, &clippedThem, &outputValues,
    };
    const std::vector<std::size_t> layerShapesDims[] = {
        std::vector<std::size_t>{activeNodes.size()},
        std::vector<std::size_t>{accumUs.size()},
        std::vector<std::size_t>{accumThem.size()},
        std::vector<std::size_t>{clippedUs.size()},
        std::vector<std::size_t>{clippedThem.size()},
        std::vector<std::size_t>{1},
    };
    selectedLayer = std::clamp(selectedLayer, 0, 5);
    Rectangle detail{bounds.x, bounds.y + bounds.height - inspectorH,
                     bounds.width, inspectorH};
    Rectangle inspector = detail;
    Rectangle summary{};
    if (detail.width >= 760.0f) {
        summary = Rectangle{detail.x + detail.width - 260.0f, detail.y,
                            260.0f, detail.height};
        inspector.width -= summary.width + gap;
    }

    const std::string note =
        selectedNodeNote(selectedLayer, selectedNode,
                         activeNodes, activeLabels,
                         accumUs, hiddenLabels,
                         accumThem,
                         clippedUs, usLabels,
                         clippedThem, themLabels,
                         outputValues, outputLabels);
    tensorviz::drawInspector(inspector,
                             layerTitles[selectedLayer],
                             layerShapes[selectedLayer],
                             *layerValues[selectedLayer],
                             layerShapesDims[selectedLayer],
                             note,
                             theme);

    if (summary.width <= 0.0f) return;

    DrawRectangleRec(summary, theme.panelBackground);
    DrawRectangleLinesEx(summary, 1.0f, theme.panelBorder);
    drawText(theme, "selected computation",
             static_cast<int>(summary.x + 9.0f),
             static_cast<int>(summary.y + 8.0f),
             16, theme.textPrimary);
    const char* stageNames[] = {
        "HalfKP feature", "white accumulator", "black accumulator",
        "us clipped node", "them clipped node", "centipawn head",
    };
    const char* stageFormulas[] = {
        "feature = king bucket + piece plane + square",
        "acc[h] = bias[h] + sum active w[f,h]",
        "acc[h] = bias[h] + sum active w[f,h]",
        "clip[h] = clamp(raw[h], 0, 1)",
        "clip[h] = clamp(raw[h], 0, 1)",
        "cp = scale * (bias + sum out[h] * clip[h])",
    };
    selectedLayer = std::clamp(selectedLayer, 0, 5);

    Rectangle stageBox{summary.x + 9.0f, summary.y + 32.0f,
                       summary.width - 18.0f, 56.0f};
    Color stageAccent = theme.accentGreen;
    if (selectedLayer == 1 || selectedLayer == 2) stageAccent = theme.accentOrange;
    else if (selectedLayer == 3) stageAccent = theme.accentBlue;
    else if (selectedLayer == 4) stageAccent = theme.accentMagenta;
    else if (selectedLayer == 5) stageAccent = theme.accentYellow;
    DrawRectangleRec(stageBox, tensorviz::blend(theme.buttonIdle, stageAccent, 0.10f));
    DrawRectangleLinesEx(stageBox, 1.0f, tensorviz::withAlpha(stageAccent, 170));
    drawText(theme, stageNames[selectedLayer],
             static_cast<int>(stageBox.x + 8.0f),
             static_cast<int>(stageBox.y + 7.0f),
             13, theme.textPrimary);
    const std::string formula =
        fitMonoText(theme, stageFormulas[selectedLayer], 10, stageBox.width - 16.0f);
    drawTextMono(theme, formula.c_str(),
                 static_cast<int>(stageBox.x + 8.0f),
                 static_cast<int>(stageBox.y + 30.0f),
                 10, theme.textMuted);

    char nodeCounts[96];
    char edgeCounts[96];
    const std::size_t featureEdges =
        featuresUs.size() * accumUs.size() + featuresThem.size() * accumThem.size();
    const std::size_t clipEdges = clippedUsFromWhite
        ? std::min(accumUs.size(), clippedUs.size()) +
          std::min(accumThem.size(), clippedThem.size())
        : std::min(accumThem.size(), clippedUs.size()) +
          std::min(accumUs.size(), clippedThem.size());
    const std::size_t outputEdges = clippedUs.size() + clippedThem.size();
    std::snprintf(nodeCounts, sizeof(nodeCounts),
                  "nodes: %zu active, %zu raw, %zu clipped, 1 output",
                  activeNodes.size(), accumUs.size() + accumThem.size(),
                  clippedUs.size() + clippedThem.size());
    std::snprintf(edgeCounts, sizeof(edgeCounts),
                  "edges: %zu feature->hidden, %zu clip, %zu output",
                  featureEdges, clipEdges, outputEdges);
    const std::string nodeText =
        fitMonoText(theme, nodeCounts, 10, summary.width - 18.0f);
    const std::string edgeText =
        fitMonoText(theme, edgeCounts, 10, summary.width - 18.0f);
    const std::string selectedTextRaw =
        selectedNodeSummary(selectedLayer, selectedNode,
                            activeNodes, activeLabels,
                            accumUs, hiddenLabels,
                            accumThem,
                            clippedUs, usLabels,
                            clippedThem, themLabels,
                            outputValues, outputLabels);
    const std::string selectedText =
        fitMonoText(theme, selectedTextRaw, 10, summary.width - 18.0f);
    drawText(theme, "current node",
             static_cast<int>(summary.x + 9.0f),
             static_cast<int>(summary.y + 99.0f),
             12, theme.textMuted);
    drawTextMono(theme, selectedText.c_str(),
                 static_cast<int>(summary.x + 9.0f),
                 static_cast<int>(summary.y + 118.0f),
                 10, theme.textPrimary);
    const std::vector<std::string> detailLines =
        selectedDetailLines(selectedLayer, selectedNode, clippedUsFromWhite,
                            featuresUs, whiteFeatureLabels, featureWeightsUs,
                            featuresThem, blackFeatureLabels, featureWeightsThem,
                            featureBias,
                            accumUs, accumThem,
                            clippedUs, clippedThem,
                            outputWeightsUs, outputWeightsThem,
                            outputContributionUs, outputContributionThem,
                            outputAffine);
    drawText(theme, "math details",
             static_cast<int>(summary.x + 9.0f),
             static_cast<int>(summary.y + 140.0f),
             12, theme.textMuted);
    float lineY = summary.y + 159.0f;
    for (std::size_t i = 0; i < detailLines.size() && i < 7; ++i) {
        if (lineY > summary.y + summary.height - 128.0f) break;
        const std::string line =
            fitMonoText(theme, detailLines[i], 10, summary.width - 18.0f);
        drawTextMono(theme, line.c_str(),
                     static_cast<int>(summary.x + 9.0f),
                     static_cast<int>(lineY),
                     10, i == 0 ? theme.textPrimary : theme.textMuted);
        lineY += 14.0f;
    }
    const float countsY = summary.y + summary.height - 124.0f;
    if (countsY > lineY + 8.0f) {
        drawTextMono(theme, nodeText.c_str(),
                     static_cast<int>(summary.x + 9.0f),
                     static_cast<int>(countsY),
                     10, theme.textDim);
        drawTextMono(theme, edgeText.c_str(),
                     static_cast<int>(summary.x + 9.0f),
                     static_cast<int>(countsY + 14.0f),
                     10, theme.textDim);
    }
    tensorviz::drawMetricBars(
        Rectangle{summary.x + 9.0f, summary.y + summary.height - 96.0f,
                  summary.width - 18.0f, 34.0f},
        std::vector<std::string>{"cp"},
        std::vector<float>{centipawns},
        800.0f, theme);
    const float featureScale =
        std::max<float>(1.0f, static_cast<float>(
            std::max(featuresUs.size(), featuresThem.size())));
    tensorviz::drawMetricBars(
        Rectangle{summary.x + 9.0f, summary.y + summary.height - 52.0f,
                  summary.width - 18.0f, 42.0f},
        std::vector<std::string>{"white", "black"},
        std::vector<float>{static_cast<float>(featuresUs.size()),
                           static_cast<float>(featuresThem.size())},
        featureScale, theme);
}

}  // namespace

void NnueView::update(const cnnv::nn::ActivationSnapshot& snap) {
    // The snapshot publishes side-to-move accumulators under fixed-color
    // keys. The "us / them" mapping for clipped accumulators is already
    // resolved by NnueNetwork; for the raw accumulators we just show
    // White / Black.
    copyFloats(snap, keys::kAccumulatorWhite, m_accumUs);
    copyFloats(snap, keys::kAccumulatorBlack, m_accumThem);
    copyFloats(snap, keys::kAccumulatorClippedUs, m_clippedUs);
    copyFloats(snap, keys::kAccumulatorClippedThem, m_clippedThem);
    copyFloats(snap, keys::kFeatureActiveWhite, m_featuresUs);
    copyFloats(snap, keys::kFeatureActiveBlack, m_featuresThem);
    copyFloats(snap, keys::kFeatureBias, m_featureBias);
    copyFloats(snap, keys::kFeatureWeightsActiveWhite, m_featureWeightsUs);
    copyFloats(snap, keys::kFeatureWeightsActiveBlack, m_featureWeightsThem);
    copyFloats(snap, keys::kOutputWeightsUs, m_outputWeightsUs);
    copyFloats(snap, keys::kOutputWeightsThem, m_outputWeightsThem);
    copyFloats(snap, keys::kOutputContributionUs, m_outputContributionUs);
    copyFloats(snap, keys::kOutputContributionThem, m_outputContributionThem);
    copyFloats(snap, keys::kOutputAffine, m_outputAffine);
    if (snap.has(keys::kValueCentipawns)) {
        m_centipawns = snap.data(keys::kValueCentipawns)[0];
    } else {
        m_centipawns = 0.0f;
    }
    m_activeFeaturesUs   = static_cast<int>(snap.size(keys::kFeatureActiveWhite));
    m_activeFeaturesThem = static_cast<int>(snap.size(keys::kFeatureActiveBlack));
    m_hasData = !m_accumUs.empty();
}

void NnueView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;

    // Subtle inner border so the view's bounds are obvious against the
    // panel background.
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    float y = m_bounds.y + pad;

    char header[96];
    if (m_hasData) {
        std::snprintf(header, sizeof(header),
                      "NNUE tensor layers  H=%d  active w/b: %d / %d",
                      static_cast<int>(m_accumUs.size()),
                      m_activeFeaturesUs, m_activeFeaturesThem);
    } else {
        std::snprintf(header, sizeof(header), "NNUE tensor layers  (no evaluation yet)");
    }
    drawText(theme, header, static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 23, theme.textPrimary);
    m_detailed = tensorviz::drawModeToggle(
        Rectangle{m_bounds.x + m_bounds.width - pad - 190.0f, y + 1.0f,
                  190.0f, 30.0f},
        m_detailed, theme);
    y += 36.0f;

    if (!m_hasData) return;

    Rectangle layers{m_bounds.x + pad, y,
                     m_bounds.width - 2.0f * pad,
                     m_bounds.y + m_bounds.height - y - pad};
    if (m_detailed) {
        drawNnueLayerStack(layers, theme,
                           m_featuresUs, m_featuresThem,
                           m_accumUs, m_accumThem,
                           m_clippedUs, m_clippedThem,
                           m_featureBias,
                           m_featureWeightsUs, m_featureWeightsThem,
                           m_outputWeightsUs, m_outputWeightsThem,
                           m_outputContributionUs, m_outputContributionThem,
                           m_outputAffine,
                           m_centipawns,
                           m_selectedLayer, m_selectedNode, m_edgeMode);
    } else {
        drawNnueAbstract(layers, theme,
                         m_featuresUs, m_featuresThem,
                         m_accumUs, m_accumThem,
                         m_clippedUs, m_clippedThem,
                         m_centipawns);
    }
}

}  // namespace cnnv::viz
