#include "viz/CnnView.h"

#include "nn/ActivationSnapshot.h"
#include "nn/lc0_cnn/Lc0CnnNetwork.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::lc0_cnn::snapshot_keys;
using tensorviz::maxAbs;
using tensorviz::squareName;
using tensorviz::drawCenteredText;
using tensorviz::drawMiniBoard;
using tensorviz::lerfSquare;

// ---------------------------------------------------------------------------
// Small shared helpers (transpiled from chess-rtk TensorViz / CnnView). The
// C++ project keeps the orange/blue diverging palette (tensorviz::signedColor)
// rather than crtk's green/coral, so "POSITIVE" reads as orange-hot and
// "NEGATIVE" as blue here; the spatial/sign semantics match crtk.
//
// crtk's CnnView paints boards with real pieces (TensorViz.drawPositionPieces,
// fed by the snapshot FEN). The C++ CNN snapshot carries no FEN and there is no
// sprite access, so every board here renders only theme.squareLight/squareDark
// squares plus the signed activation overlay; pieces are omitted. See the
// crtk call sites at CnnView.java:608-609 (atlas board), :918-919 (policy
// attention) and :1186-1187 (trace board).
// ---------------------------------------------------------------------------

int safeLen(const std::vector<float>& v) { return static_cast<int>(v.size()); }

// Signed activation overlay over a board's 64 squares (no pieces). Mirrors
// TensorViz.drawSquareOverlay.
void drawSquareOverlay(Rectangle board, const std::vector<float>& values, float scale,
                       const Theme& theme) {
    if (safeLen(values) < 64) return;
    scale = std::max(scale, 1e-6f);
    for (int sq = 0; sq < 64; ++sq) {
        const float v = values[static_cast<std::size_t>(sq)] / scale;
        const Color c = tensorviz::signedColor(std::clamp(v, -1.0f, 1.0f), 1.0f, theme);
        Color overlay = c;
        overlay.a = 190;
        DrawRectangleRec(lerfSquare(board, sq), overlay);
    }
}

// Signed gamma heatmap of a flat row-major slice (cols x rows). Mirrors
// TensorViz.drawSignedGammaHeatmap / drawHeatmap used by the channel atlas.
void drawHeatmap(Rectangle r, const std::vector<float>& values, int cols, int rows,
                 float scale, const Theme& theme, bool border = true) {
    if (cols <= 0 || rows <= 0 || values.empty()) {
        DrawRectangleRec(r, tensorviz::withAlpha(theme.buttonIdle, 120));
        return;
    }
    scale = std::max(scale, 1e-6f);
    const float cw = r.width / static_cast<float>(cols);
    const float ch = r.height / static_cast<float>(rows);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int idx = row * cols + col;
            if (idx >= static_cast<int>(values.size())) continue;
            const Color c =
                tensorviz::signedColor(values[static_cast<std::size_t>(idx)], scale, theme);
            DrawRectangleRec(Rectangle{r.x + static_cast<float>(col) * cw,
                                       r.y + static_cast<float>(row) * ch,
                                       std::ceil(cw), std::ceil(ch)},
                             c);
        }
    }
    if (border) DrawRectangleLinesEx(r, 1.0f, tensorviz::withAlpha(theme.panelBorder, 160));
}

void drawCard(Rectangle r, const char* title, const char* subtitle, Color accent,
              const Theme& theme) {
    DrawRectangleRec(r, tensorviz::withAlpha(theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
    if (accent.a != 0) {
        DrawRectangleRec(Rectangle{r.x, r.y, 4.0f, r.height}, accent);
    }
    if (title != nullptr && title[0] != '\0') {
        const std::string t = tensorviz::fitText(theme, title, 13, false, r.width - 24.0f);
        drawText(theme, t.c_str(), static_cast<int>(r.x + 10.0f),
                 static_cast<int>(r.y + 7.0f), 13, theme.textPrimary);
    }
    if (subtitle != nullptr && subtitle[0] != '\0') {
        const std::string s = tensorviz::fitText(theme, subtitle, 10, true, r.width - 24.0f);
        drawTextMono(theme, s.c_str(), static_cast<int>(r.x + 10.0f),
                     static_cast<int>(r.y + 24.0f), 10, theme.textMuted);
    }
}

bool clickedIn(Rectangle r) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
           CheckCollisionPointRec(tensorviz::panelMousePosition(), r);
}

// crtk colorFor(name): policy=POLICY(blue), value=VALUE(magenta),
// input=POSITIVE(orange-ish), else TRUNK(orange). CnnView.java:1524.
Color colorFor(const std::string& name, const Theme& theme) {
    if (!name.empty() && name[0] == 'p') return theme.accentBlue;
    if ((!name.empty() && name[0] == 'v') || name == "WDL") return theme.accentMagenta;
    if (name == "input") return theme.accentGreen;
    return theme.accentOrange;
}

// crtk importanceLabel (CnnView.java:1685).
const char* importanceLabel(float ratio) {
    if (ratio >= 0.82f) return "primary";
    if (ratio >= 0.55f) return "high";
    if (ratio >= 0.28f) return "medium";
    return "low";
}

}  // namespace

// ===========================================================================
// update(): rebuild the cached layer list (crtk rebuildLayers, :1540) and the
// per-mode derived quantities.
// ===========================================================================

void CnnView::update(const cnnv::nn::ActivationSnapshot& snap) {
    auto copy = [&](const char* key, std::vector<float>& dst) {
        const float* p = snap.data(key);
        const std::size_t n = snap.size(key);
        if (p == nullptr || n == 0) {
            dst.clear();
        } else {
            dst.assign(p, p + n);
        }
    };

    auto shapeText = [](const std::vector<std::size_t>& shape) {
        if (shape.empty()) return std::string("-");
        std::string out;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) out += "x";
            out += std::to_string(shape[i]);
        }
        return out;
    };

    auto stats = [&](LayerStat& s) {
        if (s.values.empty()) return;
        s.min = s.values[0];
        s.max = s.values[0];
        double sum = 0.0, sumAbs = 0.0, sumSq = 0.0;
        for (float v : s.values) {
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
            sum += v;
            sumAbs += std::fabs(v);
            sumSq += static_cast<double>(v) * static_cast<double>(v);
        }
        const double inv = 1.0 / static_cast<double>(s.values.size());
        s.mean = static_cast<float>(sum * inv);
        s.meanAbs = static_cast<float>(sumAbs * inv);
        s.rms = static_cast<float>(std::sqrt(sumSq * inv));
    };

    m_layers.clear();
    auto addLayer = [&](const char* name, const std::string& key) {
        const auto* entry = snap.find(key);
        if (entry == nullptr) return;
        LayerStat s;
        s.name = name;
        s.shape = shapeText(entry->shape);
        s.shapeDims = entry->shape;
        s.values = entry->data;
        stats(s);
        m_layers.push_back(std::move(s));
    };

    // Count residual blocks (crtk countLayersWithPrefix("B")).
    m_blockCount = 0;
    for (int i = 0; i < 256; ++i) {
        if (snap.has(keys::blockReluKey(i))) ++m_blockCount;
    }

    addLayer("input", keys::kInputPlanes);
    addLayer("stem", keys::kStemRelu);
    for (int i = 0; i < 256; ++i) {
        const std::string key = keys::blockReluKey(i);
        if (!snap.has(key)) continue;
        char label[48];
        std::snprintf(label, sizeof(label), "B%d", i + 1);
        addLayer(label, key);
    }
    addLayer("final", keys::kFinalActivation);
    addLayer("pStem", keys::kPolicyHidden);
    addLayer("pPlane", keys::kPolicyPlanes);
    addLayer("pLogit", keys::kPolicyLogits);
    addLayer("vConv", keys::kValueConv);
    addLayer("fc1", keys::kValueFc1);
    addLayer("vLogit", keys::kValueLogits);
    addLayer("WDL", keys::kValueWdl);

    copy(keys::kFinalActivation, m_finalMap);
    copy(keys::kPolicyLogits, m_policyLogits);
    m_hasFinalMap = m_finalMap.size() >= 64;

    m_hasWdl = false;
    if (snap.size(keys::kValueWdl) == 3) {
        const float* p = snap.data(keys::kValueWdl);
        m_wdl[0] = p[0];
        m_wdl[1] = p[1];
        m_wdl[2] = p[2];
        m_hasWdl = true;
    }
    if (snap.size(keys::kValueScalar) == 1) {
        m_valueScalar = snap.data(keys::kValueScalar)[0];
    } else if (m_hasWdl) {
        m_valueScalar = m_wdl[0] - m_wdl[2];
    } else {
        m_valueScalar = 0.0f;
    }

    m_topPolicy.clear();
    m_hasPolicy = !m_policyLogits.empty();
    if (m_hasPolicy) {
        std::vector<int> indices(m_policyLogits.size());
        std::iota(indices.begin(), indices.end(), 0);
        const std::size_t keep = std::min<std::size_t>(8, indices.size());
        std::partial_sort(indices.begin(),
                          indices.begin() + static_cast<std::ptrdiff_t>(keep), indices.end(),
                          [&](int a, int b) {
                              return m_policyLogits[static_cast<std::size_t>(a)] >
                                     m_policyLogits[static_cast<std::size_t>(b)];
                          });
        for (std::size_t i = 0; i < keep; ++i) {
            const int idx = indices[i];
            m_topPolicy.push_back({idx, m_policyLogits[static_cast<std::size_t>(idx)]});
        }
    }

    if (m_selectedLayer >= static_cast<int>(m_layers.size())) m_selectedLayer = -1;
}

// ===========================================================================
// Internal drawing namespace: each crtk paint* method as a free function that
// receives the cached state via a small context. Kept in the .cpp so the header
// stays slim (mirrors NnueView.cpp's Ctx pattern).
// ===========================================================================

namespace {

struct Ctx {
    const Theme& theme;
    Color signature;
    const std::vector<CnnView::LayerStat>& layers;
    const std::vector<float>& finalMap;
    const std::vector<std::pair<int, float>>& topPolicy;
    const float* wdl;
    float valueScalar;
    int blockCount;
    bool hasFinalMap;
    bool hasWdl;
    bool hasPolicy;
    int& selectedLayer;
    int& rawZoomLayer;
};

const CnnView::LayerStat* findLayer(const Ctx& c, const char* name) {
    for (const auto& l : c.layers) {
        if (l.name == name) return &l;
    }
    return nullptr;
}

int indexOf(const Ctx& c, const char* name) {
    for (int i = 0; i < static_cast<int>(c.layers.size()); ++i) {
        if (c.layers[static_cast<std::size_t>(i)].name == name) return i;
    }
    return -1;
}

// crtk defaultLayer(): "final", else the last layer (CnnView.java:1448).
int defaultLayer(const Ctx& c) {
    const int idx = indexOf(c, "final");
    if (idx >= 0) return idx;
    return c.layers.empty() ? -1 : static_cast<int>(c.layers.size()) - 1;
}

bool isSpatial(const CnnView::LayerStat& l) {
    return l.shapeDims.size() == 3 && l.shapeDims[1] == 8 && l.shapeDims[2] == 8;
}

std::vector<const CnnView::LayerStat*> spatialLayers(const Ctx& c) {
    std::vector<const CnnView::LayerStat*> out;
    for (const auto& l : c.layers) {
        if (isSpatial(l)) out.push_back(&l);
    }
    return out;
}

float maxScale(const Ctx& c) {
    float scale = 0.0f;
    for (const auto& l : c.layers) scale = std::max(scale, l.rms);
    return std::max(scale, 1e-6f);
}

float activity(const CnnView::LayerStat* l, float scale) {
    if (l == nullptr) return 0.0f;
    return std::clamp(l->rms / std::max(scale, 1e-6f), 0.0f, 1.0f);
}

// One channel as a 64-cell 8x8 slice (crtk channelSlice).
std::vector<float> channelSlice(const CnnView::LayerStat& l, int channel) {
    std::vector<float> out(64, 0.0f);
    if (l.shapeDims.empty()) return out;
    const int channels = static_cast<int>(l.shapeDims[0]);
    if (channel < 0 || channel >= channels) return out;
    const int off = channel * 64;
    if (off + 64 <= safeLen(l.values)) {
        std::copy(l.values.begin() + off, l.values.begin() + off + 64, out.begin());
    }
    return out;
}

float channelRms(const CnnView::LayerStat& l, int channel) {
    const std::vector<float> slice = channelSlice(l, channel);
    double sumSq = 0.0;
    for (float v : slice) sumSq += static_cast<double>(v) * v;
    return static_cast<float>(std::sqrt(sumSq / 64.0));
}

// Channel-averaged per-square activity for a spatial layer (crtk meanPerSquare).
std::vector<float> meanPerSquare(const CnnView::LayerStat* l) {
    if (l == nullptr || !isSpatial(*l)) return {};
    const int channels = static_cast<int>(l->shapeDims[0]);
    std::vector<float> out(64, 0.0f);
    for (int ch = 0; ch < channels; ++ch) {
        const int off = ch * 64;
        if (off + 64 > safeLen(l->values)) break;
        for (int s = 0; s < 64; ++s) {
            out[static_cast<std::size_t>(s)] += l->values[static_cast<std::size_t>(off + s)];
        }
    }
    const float inv = channels > 0 ? 1.0f / static_cast<float>(channels) : 1.0f;
    for (float& v : out) v *= inv;
    return out;
}

int strongestSquare(const std::vector<float>& v) {
    if (safeLen(v) < 64) return -1;
    int best = 0;
    float bestV = std::fabs(v[0]);
    for (int sq = 1; sq < 64; ++sq) {
        const float a = std::fabs(v[static_cast<std::size_t>(sq)]);
        if (a > bestV) { bestV = a; best = sq; }
    }
    return best;
}

void highlightSquareRing(Rectangle board, int square, Color color) {
    if (square < 0 || square >= 64) return;
    DrawRectangleLinesEx(lerfSquare(board, square), 2.0f, color);
}

}  // namespace

// ---------------------------------------------------------------------------
// 5-segment switcher (self-drawn — drawModeSelector only has 3 segments).
// Transpiled from NnueView.cpp's drawSwitcher.
// ---------------------------------------------------------------------------

namespace {

int drawSwitcher(Rectangle bounds, int mode, Color accent, const Theme& theme) {
    static const char* labels[5] = {"Overview", "Trace", "All", "Atlas", "Diagram"};
    const float segW = bounds.width / 5.0f;
    const Vector2 mouse = tensorviz::panelMousePosition();
    int next = std::clamp(mode, 0, 4);
    DrawRectangleRec(bounds, tensorviz::withAlpha(theme.buttonIdle, 225));
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
                                 ? tensorviz::blend(theme.buttonActive, accent, 0.55f)
                                 : theme.buttonActive;
    DrawRectangleRec(activeSeg, activeFill);
    for (int i = 0; i < 5; ++i) {
        const Rectangle seg{bounds.x + segW * static_cast<float>(i), bounds.y,
                            segW, bounds.height};
        if (CheckCollisionPointRec(mouse, seg) && i != next) {
            DrawRectangleRec(seg, tensorviz::withAlpha(theme.buttonHover, 150));
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

// ===========================================================================
// Mode 0 — Overview (ABSTRACT): trunk pipeline + per-block strip + WDL card on
// the left, policy-attention board + top policy bars on the right.
// Transpiled from CnnView.paintAbstract / paintAbstractPipeline /
// paintPolicyHeatmap (CnnView.java:699-935).
// ===========================================================================

// Stacked-rectangle silhouette behind the trunk block (crtk drawTrunkStack,
// CnnView.java:844).
void drawTrunkStack(Rectangle r, int blocks, float act, const Ctx& c) {
    const int layers = std::clamp(blocks / 4, 2, 4);
    const Color edge = c.theme.accentOrange;
    const Color fill = tensorviz::withAlpha(
        edge, static_cast<unsigned char>(30 + std::lround(20.0f * std::clamp(act, 0.0f, 1.0f))));
    for (int i = layers; i >= 1; --i) {
        const float d = static_cast<float>(i) * 4.0f;
        const Rectangle off{r.x + d, r.y - d, r.width, r.height};
        DrawRectangleRec(off, fill);
        DrawRectangleLinesEx(off, 1.0f, edge);
    }
    char detail[40];
    std::snprintf(detail, sizeof(detail), "%d blocks", blocks);
    tensorviz::drawAbstractBlock(r, "trunk", detail, act, c.theme.accentOrange,
                                 false, c.theme);
}

// Per-block RMS bar strip (crtk paintTrunkStrip, CnnView.java:867).
void paintTrunkStrip(Rectangle r, const Ctx& c) {
    std::vector<const CnnView::LayerStat*> blocks;
    for (const auto& l : c.layers) {
        if (!l.name.empty() && l.name[0] == 'B') blocks.push_back(&l);
    }
    if (blocks.empty()) return;
    float scale = 0.0f;
    for (const auto* b : blocks) scale = std::max(scale, b->rms);
    scale = std::max(scale, 1e-6f);
    const int gap = 3;
    const int n = static_cast<int>(blocks.size());
    const float cellW =
        std::max(8.0f, (r.width - static_cast<float>(gap * (n - 1))) / static_cast<float>(n));
    for (int i = 0; i < n; ++i) {
        const float x = r.x + static_cast<float>(i) * (cellW + static_cast<float>(gap));
        const float h = std::max(2.0f, r.height * (blocks[static_cast<std::size_t>(i)]->rms / scale));
        DrawRectangleRec(Rectangle{x, r.y + r.height - h, cellW, h}, c.theme.accentOrange);
    }
    DrawRectangleLinesEx(r, 1.0f, c.theme.panelBorder);
}

void paintAbstractPipeline(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 40.0f}, "abstract flow",
        "input -> stem -> residual blocks -> policy/value",
        c.theme, c.signature);

    const CnnView::LayerStat* input = findLayer(c, "input");
    const CnnView::LayerStat* stem = findLayer(c, "stem");
    const CnnView::LayerStat* finalLayer = findLayer(c, "final");
    const CnnView::LayerStat* policy = findLayer(c, "pLogit");
    const CnnView::LayerStat* value = findLayer(c, "WDL");

    float scale = 0.0f, trunkRms = 0.0f;
    int trunkSeen = 0;
    for (const auto& l : c.layers) {
        scale = std::max(scale, l.rms);
        if (!l.name.empty() && l.name[0] == 'B') { trunkRms += l.rms; ++trunkSeen; }
    }
    if (trunkSeen > 0) trunkRms /= static_cast<float>(trunkSeen);
    scale = std::max(scale, 1e-6f);

    const float top = r.y + 56.0f;
    const float blockH = 80.0f;
    const float gap = 16.0f;
    const float cellW = (r.width - 5.0f * gap) / 4.0f;
    const Rectangle inputR{r.x + gap, top, cellW, blockH};
    const Rectangle stemR{inputR.x + cellW + gap, top, cellW, blockH};
    const Rectangle trunkR{stemR.x + cellW + gap, top, cellW, blockH};
    const Rectangle finalR{trunkR.x + cellW + gap, top, cellW, blockH};

    tensorviz::drawElbowConnection(inputR, stemR, c.theme.accentOrange, true, false, c.theme);
    tensorviz::drawElbowConnection(stemR, trunkR, c.theme.accentOrange, true, false, c.theme);
    tensorviz::drawElbowConnection(trunkR, finalR, c.theme.accentOrange, true, false, c.theme);

    tensorviz::drawAbstractBlock(inputR, "input",
                                 input ? input->shape : "112x8x8",
                                 activity(input, scale), c.theme.accentGreen, false, c.theme);
    tensorviz::drawAbstractBlock(stemR, "stem", stem ? stem->shape : "-",
                                 activity(stem, scale), c.theme.accentOrange, false, c.theme);
    drawTrunkStack(trunkR, c.blockCount, trunkRms / scale, c);
    tensorviz::drawAbstractBlock(finalR, "final",
                                 finalLayer ? finalLayer->shape : "8x8",
                                 activity(finalLayer, scale), c.theme.accentYellow, false, c.theme);

    // Both heads sit under the final feature map.
    const float headTop = top + blockH + 40.0f;
    const float headW = (finalR.width - 8.0f) / 2.0f;
    const float headH = blockH - 16.0f;
    const Rectangle policyR{finalR.x, headTop, headW, headH};
    const Rectangle valueR{finalR.x + headW + 8.0f, headTop, headW, headH};
    tensorviz::drawElbowConnection(finalR, policyR, c.theme.accentBlue, true, false, c.theme);
    tensorviz::drawElbowConnection(finalR, valueR, c.theme.accentMagenta, true, false, c.theme);
    tensorviz::drawAbstractBlock(policyR, "P", "", activity(policy, scale),
                                 c.theme.accentBlue, false, c.theme);
    tensorviz::drawAbstractBlock(valueR, "V", "", activity(value, scale),
                                 c.theme.accentMagenta, false, c.theme);

    // Per-block activity strip.
    const float stripY = policyR.y + policyR.height + 28.0f;
    if (stripY + 26.0f <= r.y + r.height) {
        const Rectangle stripR{r.x + gap, stripY, r.width - 2.0f * gap, 26.0f};
        drawText(c.theme, "per-block activity (early -> late block)",
                 static_cast<int>(stripR.x), static_cast<int>(stripR.y - 14.0f), 10,
                 c.theme.textMuted);
        paintTrunkStrip(stripR, c);
    }

    // Value-head WDL card fills the lower column (crtk paintValueCard).
    const float valueTop = stripY + 26.0f + 22.0f;
    if (valueTop + 96.0f <= r.y + r.height) {
        const Rectangle card{r.x + gap, valueTop, r.width - 2.0f * gap, r.y + r.height - valueTop};
        drawCard(card, "value head", nullptr, c.theme.accentMagenta, c.theme);
        if (c.hasWdl) {
            tensorviz::drawMetricBars(
                Rectangle{card.x + 10.0f, card.y + 30.0f, std::min(280.0f, card.width - 20.0f),
                          std::max(30.0f, card.height - 58.0f)},
                std::vector<std::string>{"W", "D", "L"},
                std::vector<float>{c.wdl[0], c.wdl[1], c.wdl[2]}, 1.0f, c.theme);
        }
        char scalar[48];
        std::snprintf(scalar, sizeof(scalar), "value scalar %+.3f", c.valueScalar);
        drawText(c.theme, scalar, static_cast<int>(card.x + 10.0f),
                 static_cast<int>(card.y + card.height - 22.0f), 11, c.theme.textMuted);
    }
}

void paintPolicyHeatmap(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 40.0f}, "policy attention",
        "final-map mean over 8x8 board",
        c.theme, c.signature);
    const float size = std::min(r.width, r.height - 60.0f);
    const Rectangle board{r.x + (r.width - size) * 0.5f, r.y + 50.0f, size - 16.0f, size - 16.0f};
    drawMiniBoard(board, c.theme);
    if (c.hasFinalMap) {
        const float s = std::max(maxAbs(c.finalMap), 1e-6f);
        drawSquareOverlay(board, c.finalMap, s, c.theme);
        highlightSquareRing(board, strongestSquare(c.finalMap),
                            tensorviz::withAlpha(c.theme.accentYellow, 230));
    }

    // Top legal-move policy is unavailable (no move decoder / FEN in C++); show
    // top policy by raw logit index instead (crtk paintTopPolicy fallback,
    // CnnView.java:996).
    if (!c.topPolicy.empty()) {
        const float barsTop = board.y + board.height + 22.0f;
        const Rectangle bars{r.x + 4.0f, barsTop, r.width - 8.0f,
                             std::max(1.0f, r.y + r.height - barsTop - 8.0f)};
        if (bars.height >= 40.0f) {
            drawText(c.theme, "top policy logits (by raw move index)",
                     static_cast<int>(bars.x), static_cast<int>(bars.y - 14.0f), 10,
                     c.theme.textMuted);
            std::vector<std::string> labels;
            std::vector<float> values;
            const std::size_t keep = std::min<std::size_t>(6, c.topPolicy.size());
            float pScale = 0.0f;
            for (std::size_t i = 0; i < keep; ++i) {
                char label[48];
                std::snprintf(label, sizeof(label), "#%d", c.topPolicy[i].first);
                labels.emplace_back(label);
                values.push_back(c.topPolicy[i].second);
                pScale = std::max(pScale, std::fabs(c.topPolicy[i].second));
            }
            tensorviz::drawMetricBars(bars, labels, values, pScale, c.theme);
        }
    }
}

void paintOverview(Rectangle body, const Ctx& c) {
    if (c.layers.empty()) {
        drawText(c.theme, "No CNN layers in this snapshot.",
                 static_cast<int>(body.x + 8.0f), static_cast<int>(body.y + 8.0f), 14,
                 c.theme.textMuted);
        return;
    }
    const float leftW = body.width * 0.62f;
    const Rectangle left{body.x, body.y, leftW, body.height};
    const Rectangle right{body.x + leftW + 10.0f, body.y, body.width - leftW - 10.0f,
                          body.height};
    paintAbstractPipeline(left, c);
    paintPolicyHeatmap(right, c);
}

// ===========================================================================
// Mode 1 — Trace (DETAILED): trace path strip (click to select a layer) +
// selected-layer board overlay + per-channel grid / input-plane mini-boards.
// Transpiled from CnnView.paintDetailed / paintTracePath / paintDetailedBoard /
// paintDetailedChannels / paintInputPlanes / paintChannelGrid (:1086-1366).
// ===========================================================================

// Returns the trace-path bottom y. Click-selects a layer cell into
// c.selectedLayer (crtk paintTracePath, CnnView.java:1114).
float paintTracePath(Rectangle r, const Ctx& c) {
    if (c.layers.empty()) return r.y + r.height;
    const int focused = (c.selectedLayer < 0 || c.selectedLayer >= static_cast<int>(c.layers.size()))
                            ? defaultLayer(c)
                            : c.selectedLayer;
    const float scale = maxScale(c);
    const float top = r.y + 4.0f;
    const float h = std::max(28.0f, r.height - 8.0f);
    const int count = static_cast<int>(c.layers.size());
    const int gap = count > 48 ? 1 : 3;
    const float cellW =
        std::max(3.0f, (r.width - static_cast<float>(gap * (count - 1))) / static_cast<float>(count));
    float x = r.x;
    for (int i = 0; i < count; ++i) {
        const CnnView::LayerStat& info = c.layers[static_cast<std::size_t>(i)];
        const float w = (i == count - 1) ? std::max(8.0f, r.x + r.width - x) : cellW;
        const Rectangle cell{x, top, w, h};
        const Color accent = colorFor(info.name, c.theme);
        const bool selected = i == focused;
        const float act = std::clamp(info.rms / scale, 0.0f, 1.0f);
        DrawRectangleRec(cell, tensorviz::blend(c.theme.panelBackground, accent,
                                                0.10f + act * 0.38f));
        DrawRectangleLinesEx(cell, 1.0f, selected ? c.theme.selection : c.theme.panelBorder);
        if (w > 8.0f) {
            const float barW = std::max(1.0f, (w - 6.0f) * act);
            DrawRectangleRec(Rectangle{cell.x + 3.0f, cell.y + cell.height - 7.0f, barW, 3.0f},
                             accent);
        }
        if (w >= 34.0f) {
            const std::string label = tensorviz::fitText(c.theme, info.name, 10, false, std::max(8.0f, w - 6.0f));
            drawText(c.theme, label.c_str(), static_cast<int>(cell.x + 3.0f),
                     static_cast<int>(cell.y + 3.0f), 10,
                     selected ? c.theme.textPrimary : c.theme.textMuted);
        }
        if (selected) {
            DrawRectangleLinesEx(Rectangle{cell.x + 1.0f, cell.y + 1.0f, cell.width - 3.0f,
                                           cell.height - 3.0f},
                                 1.0f, c.theme.selection);
        }
        if (clickedIn(cell)) c.selectedLayer = i;
        x += w + static_cast<float>(gap);
    }
    return r.y + r.height;
}

void paintDetailedBoard(Rectangle r, const Ctx& c, int idx) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 18.0f}, "position",
                                 "mean activity of the selected layer (no pieces)",
                                 c.theme, c.signature);
    const float size = std::min(r.width - 8.0f, r.height - 80.0f);
    const Rectangle board{r.x + (r.width - size) * 0.5f, r.y + 24.0f, size, size};
    drawMiniBoard(board, c.theme);
    if (idx < 0 || idx >= static_cast<int>(c.layers.size())) return;
    const CnnView::LayerStat& info = c.layers[static_cast<std::size_t>(idx)];
    // Spatial layers average per square; the "final" layer is already the
    // per-square mean map ([64]) so it overlays its raw values directly.
    std::vector<float> perSquare = meanPerSquare(&info);
    if (perSquare.empty() && info.name == "final" && safeLen(info.values) >= 64) {
        perSquare.assign(info.values.begin(), info.values.begin() + 64);
    }
    if (!perSquare.empty()) {
        const float s = std::max(maxAbs(perSquare), 1e-6f);
        drawSquareOverlay(board, perSquare, s, c.theme);
        highlightSquareRing(board, strongestSquare(perSquare),
                            tensorviz::withAlpha(c.theme.accentYellow, 230));
    }
    char line[96];
    std::snprintf(line, sizeof(line), "layer: %s  -  %s", info.name.c_str(), info.shape.c_str());
    drawText(c.theme, line, static_cast<int>(r.x + 6.0f),
             static_cast<int>(board.y + board.height + 6.0f), 11, c.theme.textMuted);
    char stat[120];
    std::snprintf(stat, sizeof(stat), "rms %.3f  mean %+.3f  range %+.2f..%+.2f", info.rms,
                  info.mean, info.min, info.max);
    drawText(c.theme, stat, static_cast<int>(r.x + 6.0f),
             static_cast<int>(board.y + board.height + 22.0f), 11, c.theme.textMuted);
    const int focusSq = strongestSquare(perSquare);
    if (focusSq >= 0) {
        char fs[48];
        std::snprintf(fs, sizeof(fs), "strongest square: %s", squareName(focusSq));
        drawText(c.theme, fs, static_cast<int>(r.x + 6.0f),
                 static_cast<int>(board.y + board.height + 38.0f), 11, c.theme.textMuted);
    }
}

// First input planes as labelled mini-boards (crtk paintInputPlanes, :1290).
void paintInputPlanes(Rectangle r, const Ctx& c, const CnnView::LayerStat& info) {
    static const char* labels[] = {
        "own P", "own N", "own B", "own R", "own Q", "own K",
        "enemy P", "enemy N", "enemy B", "enemy R", "enemy Q", "enemy K",
        "rep 1", "rep 2"};
    const int channels =
        std::min(static_cast<int>(sizeof(labels) / sizeof(labels[0])),
                 static_cast<int>(info.shapeDims.empty() ? 0 : info.shapeDims[0]));
    if (channels <= 0) return;
    const int cols = std::min(7, channels);
    const int rows = (channels + cols - 1) / cols;
    const float labelH = 14.0f;
    const float cellW = (r.width - static_cast<float>(cols - 1) * 8.0f) / static_cast<float>(cols);
    const float cellH =
        (r.height - static_cast<float>(rows - 1) * 10.0f - static_cast<float>(rows) * labelH) /
        static_cast<float>(rows);
    const float side = std::max(16.0f, std::min(cellW, cellH));
    for (int ch = 0; ch < channels; ++ch) {
        const int row = ch / cols;
        const int col = ch % cols;
        const float x = r.x + static_cast<float>(col) * (cellW + 8.0f);
        const float y = r.y + static_cast<float>(row) * (side + 10.0f + labelH);
        const Rectangle board{x, y + labelH, side, side};
        drawMiniBoard(board, c.theme);
        const std::vector<float> slice = channelSlice(info, ch);
        drawSquareOverlay(board, slice, std::max(maxAbs(slice), 1e-6f), c.theme);
        drawText(c.theme, labels[ch], static_cast<int>(x), static_cast<int>(y), 10,
                 c.theme.textMuted);
    }
}

// Grid of per-channel 8x8 heatmaps (crtk paintChannelGrid, :1333).
void paintChannelGrid(Rectangle r, const Ctx& c, const CnnView::LayerStat& info) {
    const int channels = static_cast<int>(info.shapeDims.empty() ? 0 : info.shapeDims[0]);
    const int show = std::min(channels, 32);
    if (show <= 0) return;
    const int cols = std::min(show, std::max(4, static_cast<int>(r.width / 60.0f)));
    const int rows = (show + cols - 1) / cols;
    const float cellW = (r.width - static_cast<float>(cols - 1) * 6.0f) / static_cast<float>(cols);
    const float cellH = std::min(cellW, (r.height - static_cast<float>(rows - 1) * 6.0f) /
                                            static_cast<float>(rows));
    const float scale = std::max(maxAbs(info.values), 1e-6f);
    for (int ch = 0; ch < show; ++ch) {
        const int row = ch / cols;
        const int col = ch % cols;
        const Rectangle cell{r.x + static_cast<float>(col) * (cellW + 6.0f),
                             r.y + static_cast<float>(row) * (cellH + 6.0f), cellW, cellH};
        drawHeatmap(cell, channelSlice(info, ch), 8, 8, scale, c.theme);
    }
}

void paintDetailedChannels(Rectangle r, const Ctx& c, int idx) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 18.0f}, "channels",
                                 "per-channel 8x8 heatmaps for the focused layer", c.theme,
                                 c.signature);
    if (idx < 0 || idx >= static_cast<int>(c.layers.size())) return;
    const CnnView::LayerStat& info = c.layers[static_cast<std::size_t>(idx)];
    const Rectangle content{r.x, r.y + 24.0f, r.width, r.height - 24.0f};
    if (info.name == "input" && isSpatial(info)) {
        paintInputPlanes(content, c, info);
    } else if (isSpatial(info)) {
        paintChannelGrid(content, c, info);
    } else if (!info.values.empty()) {
        // Flat vector: single-row signed heatmap bar strip (crtk paintBarStrip).
        const int cols = std::min(safeLen(info.values), 512);
        std::vector<float> sampled(static_cast<std::size_t>(cols));
        for (int i = 0; i < cols; ++i) {
            const int s = static_cast<int>(static_cast<long long>(i) * safeLen(info.values) / cols);
            sampled[static_cast<std::size_t>(i)] = info.values[static_cast<std::size_t>(s)];
        }
        const Rectangle bar{content.x, content.y + content.height * 0.5f - 24.0f, content.width,
                            48.0f};
        drawHeatmap(bar, sampled, cols, 1, std::max(maxAbs(info.values), 1e-6f), c.theme);
    }
}

void paintTrace(Rectangle body, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, 40.0f}, "CNN Trace - one selected layer",
        "click a stage in the path; the board and channel panel inspect only that layer", c.theme,
        c.signature);
    if (c.layers.empty()) {
        drawText(c.theme, "No CNN layers in this snapshot.", static_cast<int>(body.x + 8.0f),
                 static_cast<int>(body.y + 48.0f), 14, c.theme.textMuted);
        return;
    }
    const float top = body.y + 50.0f;
    const float pathH = std::min(104.0f, std::max(78.0f, body.height / 5.0f));
    const float inspectorTop = paintTracePath(Rectangle{body.x, top, body.width, pathH}, c) + 14.0f;
    const float inspectorH = std::max(220.0f, body.y + body.height - inspectorTop - 4.0f);
    const float boardW = std::min(360.0f, std::max(220.0f, inspectorH));
    const int idx = (c.selectedLayer < 0 || c.selectedLayer >= static_cast<int>(c.layers.size()))
                        ? defaultLayer(c)
                        : c.selectedLayer;
    paintDetailedBoard(Rectangle{body.x, inspectorTop, boardW, inspectorH}, c, idx);
    paintDetailedChannels(
        Rectangle{body.x + boardW + 12.0f, inspectorTop, body.width - boardW - 12.0f, inspectorH},
        c, idx);
}

// ===========================================================================
// Mode 2 — All (RAW): dense all-layers atlas (rows = spatial layer, cols =
// channel, each cell an 8x8 gamma heatmap). Click a row to zoom that layer.
// Transpiled from CnnView.paintRaw / paintRawZoom (CnnView.java:216-371).
// ===========================================================================

void paintRawZoom(Rectangle body, const Ctx& c, const CnnView::LayerStat& info) {
    const float headerH = 38.0f;
    const int channels = static_cast<int>(info.shapeDims.empty() ? 0 : info.shapeDims[0]);
    char title[80];
    std::snprintf(title, sizeof(title), "layer %s - %d channels", info.name.c_str(), channels);
    tensorviz::drawSectionHeader(Rectangle{body.x, body.y, body.width, headerH}, title,
                                 "8x8 activation per channel - click anywhere to zoom out", c.theme,
                                 c.signature);
    const float scale = std::max(maxAbs(info.values), 1e-6f);
    const float gridTop = body.y + headerH + 6.0f;
    const float gridW = body.width;
    const float gridH = body.height - headerH - 10.0f;
    const int cols = std::max(1, static_cast<int>(std::ceil(std::sqrt(
                                  static_cast<double>(channels) * gridW / std::max(1.0f, gridH)))));
    const int rows = (channels + cols - 1) / cols;
    const float cell = std::max(8.0f, std::min(gridW / static_cast<float>(cols),
                                               gridH / static_cast<float>(std::max(1, rows))));
    for (int ch = 0; ch < channels; ++ch) {
        const int col = ch % cols;
        const int row = ch / cols;
        const Rectangle r{body.x + static_cast<float>(col) * cell + 1.0f,
                          gridTop + static_cast<float>(row) * cell + 1.0f, cell - 2.0f, cell - 2.0f};
        drawHeatmap(r, channelSlice(info, ch), 8, 8, scale, c.theme);
        if (cell >= 26.0f) {
            char idx[48];
            std::snprintf(idx, sizeof(idx), "%d", ch);
            drawText(c.theme, idx, static_cast<int>(r.x + 2.0f), static_cast<int>(r.y + 1.0f), 9,
                     c.theme.textMuted);
        }
    }
    if (clickedIn(Rectangle{body.x, body.y, body.width, body.height})) c.rawZoomLayer = -1;
}

void paintRaw(Rectangle body, const Ctx& c) {
    const float headerH = 38.0f;
    const std::vector<const CnnView::LayerStat*> spatial = spatialLayers(c);
    if (spatial.empty()) {
        tensorviz::drawSectionHeader(Rectangle{body.x, body.y, body.width, headerH},
                                     "raw channel atlas", "no spatial layers in this snapshot",
                                     c.theme, c.signature);
        return;
    }
    if (c.rawZoomLayer >= 0 && c.rawZoomLayer < static_cast<int>(spatial.size())) {
        paintRawZoom(body, c, *spatial[static_cast<std::size_t>(c.rawZoomLayer)]);
        return;
    }
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, headerH},
        "raw activation atlas - every CNN layer at once",
        "rows = layer - cols = channel - signed palette - click a row to focus", c.theme,
        c.signature);

    int maxChannels = 0;
    for (const auto* info : spatial) {
        maxChannels = std::max(maxChannels, static_cast<int>(info->shapeDims[0]));
    }
    maxChannels = std::max(1, maxChannels);
    const float rowLabelW = 96.0f;
    const float colLabelH = 16.0f;
    const float gridLeft = body.x + rowLabelW;
    const float gridTop = body.y + headerH + 4.0f + colLabelH;
    const float gridW = body.width - rowLabelW - 8.0f;
    const float gridH = body.height - headerH - 4.0f - colLabelH - 4.0f;
    const float cellW = std::max(2.0f, gridW / static_cast<float>(maxChannels));
    const float cellH = std::max(8.0f, gridH / static_cast<float>(spatial.size()));

    // Sparse column header (channel index).
    const int step = std::max(1, maxChannels / 16);
    for (int ch = 0; ch < maxChannels; ch += step) {
        char lbl[48];
        std::snprintf(lbl, sizeof(lbl), "%d", ch);
        const int lw = measureText(c.theme, lbl, 9);
        drawText(c.theme, lbl,
                 static_cast<int>(gridLeft + static_cast<float>(ch) * cellW +
                                  (cellW - static_cast<float>(lw)) * 0.5f),
                 static_cast<int>(body.y + headerH + 4.0f), 9, c.theme.textMuted);
    }

    for (int li = 0; li < static_cast<int>(spatial.size()); ++li) {
        const CnnView::LayerStat& info = *spatial[static_cast<std::size_t>(li)];
        const int channels = static_cast<int>(info.shapeDims[0]);
        const float y = gridTop + static_cast<float>(li) * cellH;
        drawText(c.theme, info.name.c_str(), static_cast<int>(body.x + 4.0f),
                 static_cast<int>(y + cellH * 0.5f - 9.0f), 10, c.theme.textMuted);
        drawText(c.theme, info.shape.c_str(), static_cast<int>(body.x + 4.0f),
                 static_cast<int>(y + cellH * 0.5f + 3.0f), 9, c.theme.textDim);
        const float scale = std::max(maxAbs(info.values), 1e-6f);
        for (int ch = 0; ch < channels; ++ch) {
            const Rectangle cell{gridLeft + static_cast<float>(ch) * cellW + 1.0f, y + 1.0f,
                                 std::max(1.0f, cellW - 2.0f), std::max(1.0f, cellH - 2.0f)};
            drawHeatmap(cell, channelSlice(info, ch), 8, 8, scale, c.theme, false);
        }
        // One zoom hit region per layer row.
        if (clickedIn(Rectangle{body.x, y, body.width, cellH})) c.rawZoomLayer = li;
    }
}

// ===========================================================================
// Mode 3 — Atlas: layer x channel RMS fingerprint + strongest-channel cards +
// board-level final-map / policy-plane footprints. Transpiled from
// CnnView.paintAtlas / paintCnnLayerFingerprint / paintCnnTopChannels /
// paintCnnSpatialAtlas (CnnView.java:146-617). The compare/variant/sort and
// hover-inspect infra are simplified out (see report).
// ===========================================================================

struct ChannelPick {
    const CnnView::LayerStat* layer = nullptr;
    int channel = 0;
    float score = 0.0f;
};

std::vector<ChannelPick> strongestChannels(const Ctx& c, int limit) {
    std::vector<ChannelPick> picks;
    for (const auto* info : spatialLayers(c)) {
        if (info->name == "input") continue;
        const int channels = static_cast<int>(info->shapeDims[0]);
        for (int ch = 0; ch < channels; ++ch) {
            picks.push_back(ChannelPick{info, ch, channelRms(*info, ch)});
        }
    }
    std::sort(picks.begin(), picks.end(),
              [](const ChannelPick& a, const ChannelPick& b) { return a.score > b.score; });
    if (static_cast<int>(picks.size()) > limit) picks.resize(static_cast<std::size_t>(limit));
    return picks;
}

void paintCnnLayerFingerprint(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 38.0f}, "layer x channel fingerprint",
        "bright cells carry the most signal now; pale cells are low-signal background", c.theme,
        c.signature);
    const std::vector<const CnnView::LayerStat*> spatial = spatialLayers(c);
    if (spatial.empty()) return;
    const float rowLabelW = 64.0f;
    const float colLabelH = 16.0f;
    const float gridX = r.x + rowLabelW;
    const float gridY = r.y + 38.0f + colLabelH + 4.0f;
    const float gridW = r.width - rowLabelW - 4.0f;
    const float gridH = r.y + r.height - gridY - 4.0f;
    int maxChannels = 1;
    for (const auto* info : spatial) {
        maxChannels = std::max(maxChannels, static_cast<int>(info->shapeDims[0]));
    }
    const float cellW = std::max(2.0f, gridW / static_cast<float>(maxChannels));
    const float rowH = std::max(10.0f, gridH / static_cast<float>(spatial.size()));

    float scale = 0.0f;
    std::vector<std::vector<float>> scores(spatial.size());
    for (std::size_t i = 0; i < spatial.size(); ++i) {
        const int channels = static_cast<int>(spatial[i]->shapeDims[0]);
        scores[i].assign(static_cast<std::size_t>(channels), 0.0f);
        for (int ch = 0; ch < channels; ++ch) {
            const float v = channelRms(*spatial[i], ch);
            scores[i][static_cast<std::size_t>(ch)] = v;
            scale = std::max(scale, v);
        }
    }
    scale = std::max(scale, 1e-6f);
    const float importantCutoff = scale * 0.78f;

    // Sparse column header.
    const int step = std::max(1, maxChannels / 8);
    for (int ch = 0; ch < maxChannels; ch += step) {
        char lbl[48];
        std::snprintf(lbl, sizeof(lbl), "%d", ch);
        drawText(c.theme, lbl, static_cast<int>(gridX + static_cast<float>(ch) * cellW),
                 static_cast<int>(gridY - 14.0f), 9, c.theme.textMuted);
    }

    for (std::size_t i = 0; i < spatial.size(); ++i) {
        const CnnView::LayerStat& info = *spatial[i];
        const float y = gridY + static_cast<float>(i) * rowH;
        drawText(c.theme, info.name.c_str(), static_cast<int>(r.x + 4.0f),
                 static_cast<int>(y + std::max(10.0f, rowH * 0.5f) - 5.0f), 10, c.theme.textMuted);
        const int channels = static_cast<int>(info.shapeDims[0]);
        const Color accent = colorFor(info.name, c.theme);
        for (int ch = 0; ch < channels; ++ch) {
            const float v = std::sqrt(std::clamp(scores[i][static_cast<std::size_t>(ch)] / scale,
                                                 0.0f, 1.0f));
            const Color fill =
                tensorviz::blend(c.theme.panelBackground, accent, 0.18f + 0.78f * v);
            const float x = gridX + static_cast<float>(ch) * cellW;
            DrawRectangleRec(Rectangle{x, y, std::max(1.0f, cellW - 1.0f), std::max(1.0f, rowH - 1.0f)},
                             fill);
            if (scores[i][static_cast<std::size_t>(ch)] >= importantCutoff && cellW >= 4.0f &&
                rowH >= 8.0f) {
                DrawRectangleLinesEx(Rectangle{x, y, std::max(1.0f, cellW - 1.0f),
                                               std::max(1.0f, rowH - 1.0f)},
                                     1.0f, tensorviz::withAlpha(c.theme.textPrimary, 120));
            }
        }
    }
    DrawRectangleLinesEx(Rectangle{gridX, gridY,
                                   std::min(gridW, static_cast<float>(maxChannels) * cellW),
                                   std::min(gridH, static_cast<float>(spatial.size()) * rowH)},
                         1.0f, c.theme.panelBorder);
}

void paintCnnTopChannels(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 38.0f}, "strongest filters",
                                 "ranked channels explaining most of this position's spatial signal",
                                 c.theme, c.signature);
    const std::vector<ChannelPick> picks = strongestChannels(c, 8);
    if (picks.empty()) return;
    const float bestScore = std::max(1e-6f, picks[0].score);
    const Rectangle content{r.x + 2.0f, r.y + 44.0f, r.width - 4.0f, r.height - 46.0f};
    const int show = std::min<int>(8, static_cast<int>(picks.size()));
    const int cols = std::min(4, std::max(1, static_cast<int>(content.width / 104.0f)));
    const int rows = (show + cols - 1) / cols;
    const float gap = 8.0f;
    const float cellW =
        std::max(60.0f, (content.width - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols));
    const float cellH =
        std::max(48.0f, (content.height - gap * static_cast<float>(rows - 1)) / static_cast<float>(rows));
    for (int i = 0; i < show; ++i) {
        const ChannelPick& pick = picks[static_cast<std::size_t>(i)];
        const int row = i / cols;
        const int col = i % cols;
        const Rectangle card{content.x + static_cast<float>(col) * (cellW + gap),
                             content.y + static_cast<float>(row) * (cellH + gap), cellW, cellH};
        char title[40];
        std::snprintf(title, sizeof(title), "%s c%d", pick.layer->name.c_str(), pick.channel);
        char sub[48];
        std::snprintf(sub, sizeof(sub), "%s - rms %.3f", importanceLabel(pick.score / bestScore),
                      pick.score);
        drawCard(card, title, sub, colorFor(pick.layer->name, c.theme), c.theme);
        const float side = std::max(24.0f, std::min(card.width - 14.0f, card.height - 34.0f));
        const Rectangle heat{card.x + 7.0f, card.y + card.height - side - 7.0f, side, side};
        const std::vector<float> slice = channelSlice(*pick.layer, pick.channel);
        drawHeatmap(heat, slice, 8, 8, std::max(maxAbs(slice), 1e-6f), c.theme);
    }
}

void drawCnnAtlasBoard(Rectangle board, const char* title, const std::vector<float>& values,
                       const Ctx& c) {
    const int focusSq = strongestSquare(values);
    char label[64];
    if (focusSq >= 0) {
        std::snprintf(label, sizeof(label), "%s - focus %s", title, squareName(focusSq));
    } else {
        std::snprintf(label, sizeof(label), "%s", title);
    }
    drawText(c.theme, tensorviz::fitText(c.theme, label, 10, false, board.width + 10.0f).c_str(),
             static_cast<int>(board.x), static_cast<int>(board.y - 14.0f), 10, c.theme.textMuted);
    drawMiniBoard(board, c.theme);
    if (safeLen(values) >= 64) {
        drawSquareOverlay(board, values, std::max(maxAbs(values), 1e-6f), c.theme);
        highlightSquareRing(board, focusSq, tensorviz::withAlpha(c.theme.accentYellow, 230));
    }
}

void paintCnnSpatialAtlas(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 38.0f}, "board footprint",
                                 "final-map and policy-plane activation projected onto the board",
                                 c.theme, c.signature);
    const Rectangle content{r.x + 4.0f, r.y + 46.0f, r.width - 8.0f, r.height - 50.0f};
    const CnnView::LayerStat* finalLayer = findLayer(c, "final");
    const CnnView::LayerStat* policyPlanes = findLayer(c, "pPlane");
    std::vector<float> finalMap = c.hasFinalMap ? c.finalMap : meanPerSquare(finalLayer);
    const std::vector<float> policyMap = meanPerSquare(policyPlanes);
    const float gap = 10.0f;
    float boardSide =
        std::min((content.width - gap) * 0.5f, std::max(60.0f, content.height - 22.0f));
    if (boardSide < 72.0f) {
        boardSide = std::min(content.width, std::max(48.0f, content.height - 22.0f));
        drawCnnAtlasBoard(Rectangle{content.x, content.y + 14.0f, boardSide, boardSide}, "final",
                          finalMap, c);
        return;
    }
    drawCnnAtlasBoard(Rectangle{content.x, content.y + 16.0f, boardSide, boardSide}, "final map",
                      finalMap, c);
    drawCnnAtlasBoard(
        Rectangle{content.x + boardSide + gap, content.y + 16.0f, boardSide, boardSide},
        "policy planes", policyMap, c);
}

void paintAtlas(Rectangle body, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, 40.0f}, "CNN activation atlas",
        "layer/channel fingerprint - strongest spatial filters - board policy/value footprint",
        c.theme, c.signature);
    if (c.layers.empty()) {
        drawText(c.theme, "No CNN layers in this snapshot.", static_cast<int>(body.x + 8.0f),
                 static_cast<int>(body.y + 48.0f), 14, c.theme.textMuted);
        return;
    }
    const float topY = body.y + 50.0f;
    const float h = body.height - 50.0f;
    const float gap = 12.0f;
    if (body.width < 820.0f) {
        const float part = std::max(160.0f, (h - 2.0f * gap) / 3.0f);
        const Rectangle fingerprint{body.x, topY, body.width, part};
        const Rectangle channels{body.x, fingerprint.y + fingerprint.height + gap, body.width, part};
        const Rectangle boards{
            body.x, channels.y + channels.height + gap, body.width,
            std::max(120.0f, body.y + body.height - (channels.y + channels.height + gap))};
        paintCnnLayerFingerprint(fingerprint, c);
        paintCnnTopChannels(channels, c);
        paintCnnSpatialAtlas(boards, c);
        return;
    }
    const float leftW = std::max(430.0f, std::min(body.width * 0.56f, body.width - 360.0f));
    const Rectangle fingerprint{body.x, topY, leftW, h};
    const float rightX = fingerprint.x + fingerprint.width + gap;
    const float rightW = body.x + body.width - rightX;
    const float topH = std::max(220.0f, std::min(h * 0.52f, 320.0f));
    const Rectangle channels{rightX, topY, rightW, topH};
    const Rectangle boards{rightX, channels.y + channels.height + gap, rightW,
                           std::max(140.0f, body.y + body.height - (channels.y + channels.height + gap))};
    paintCnnLayerFingerprint(fingerprint, c);
    paintCnnTopChannels(channels, c);
    paintCnnSpatialAtlas(boards, c);
}

// ===========================================================================
// Mode 4 — Diagram: static five-box architecture schematic with live dims and
// value-scalar caption. Transpiled from CnnView.paintDiagram (:1596).
// ===========================================================================

void paintDiagram(Rectangle body, const Ctx& c) {
    const float headerH = 40.0f;
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, headerH}, "LC0 CNN architecture",
        "input planes -> stem -> residual blocks -> policy / value heads", c.theme, c.signature);

    const CnnView::LayerStat* input = findLayer(c, "input");
    const CnnView::LayerStat* stem = findLayer(c, "stem");
    const CnnView::LayerStat* policy = findLayer(c, "pLogit");

    char blockTitle[32];
    std::snprintf(blockTitle, sizeof(blockTitle), "%d residual blocks", c.blockCount);
    const std::array<const char*, 5> titles{
        {"input encoding", "stem", blockTitle, "policy head", "value head"}};

    const std::array<std::string, 5> subs{
        {input ? input->shape : "112 input planes (8x8)",
         stem ? stem->shape : "3x3 conv - ReLU", "3x3 conv x2 + skip",
         policy ? policy->shape : "logits 1858", "WDL + scalar eval"}};

    const float boxW = 200.0f;
    const float boxH = 90.0f;
    const float gap = 30.0f;
    const float totalW = boxW * 5.0f + gap * 4.0f;
    const float startX = body.x + (body.width - totalW) * 0.5f;
    const float y = body.y + headerH + 60.0f;
    for (int i = 0; i < 5; ++i) {
        const float x = startX + static_cast<float>(i) * (boxW + gap);
        const Rectangle box{x, y, boxW, boxH};
        DrawRectangleRec(box, tensorviz::withAlpha(c.theme.panelBackground, 240));
        DrawRectangleLinesEx(box, 1.0f, i == 4 ? c.theme.accentMagenta : c.theme.panelBorder);
        drawCenteredText(c.theme, titles[static_cast<std::size_t>(i)],
                     Rectangle{box.x, box.y + 8.0f, box.width, 24.0f}, 13, c.theme.textPrimary);
        const std::string sub =
            tensorviz::fitText(c.theme, subs[static_cast<std::size_t>(i)], 11, false, boxW - 12.0f);
        drawCenteredText(c.theme, sub.c_str(), Rectangle{box.x, box.y + 40.0f, box.width, 20.0f}, 11,
                     c.theme.textMuted);
        if (i < 4) {
            const float ax1 = x + boxW;
            const float ax2 = x + boxW + gap;
            const float ay = y + boxH * 0.5f;
            DrawLine(static_cast<int>(ax1 + 2.0f), static_cast<int>(ay),
                     static_cast<int>(ax2 - 6.0f), static_cast<int>(ay), c.theme.accentOrange);
            DrawTriangle(Vector2{ax2 - 6.0f, ay}, Vector2{ax2 - 12.0f, ay + 5.0f},
                         Vector2{ax2 - 12.0f, ay - 5.0f}, c.theme.accentOrange);
        }
    }
    char caption[48];
    std::snprintf(caption, sizeof(caption), "value scalar %+.3f", c.valueScalar);
    const int cw = measureText(c.theme, caption, 12);
    drawText(c.theme, caption,
             static_cast<int>(body.x + (body.width - static_cast<float>(cw)) * 0.5f),
             static_cast<int>(y + boxH + 28.0f), 12, c.theme.textMuted);
}

}  // namespace

// ===========================================================================
// Public entry point: header band + 5-segment switcher + mode dispatch.
// ===========================================================================

void CnnView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    float y = m_bounds.y + pad;

    // 5-segment switcher (Overview / Trace / All / Atlas / Diagram).
    const Rectangle switcher{
        m_bounds.x + m_bounds.width - pad - 310.0f, y + 1.0f,
        310.0f, 28.0f};
    const std::string title = tensorviz::fitText(
        theme, "LC0 CNN activations", 22, false,
        switcher.x - (m_bounds.x + pad) - 10.0f);
    drawText(theme, title.c_str(), static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 22, theme.textPrimary);
    const int prevMode = m_mode;
    m_mode = drawSwitcher(switcher, m_mode, m_signature, theme);
    if (m_mode != prevMode) m_rawZoomLayer = -1;  // crtk onViewModeChanged.
    y += 34.0f;

    char summary[112];
    std::snprintf(summary, sizeof(summary),
                  "112 input planes -> stem -> %d residual blocks -> policy / value heads",
                  m_blockCount);
    drawText(theme, summary, static_cast<int>(m_bounds.x + pad), static_cast<int>(y), 13,
             theme.textMuted);
    y += 22.0f;

    const Rectangle body{m_bounds.x + pad, y, m_bounds.width - 2.0f * pad,
                         m_bounds.y + m_bounds.height - y - pad};

    const Ctx c{theme,        m_signature, m_layers,        m_finalMap,
                m_topPolicy,  m_wdl,       m_valueScalar,   m_blockCount,
                m_hasFinalMap, m_hasWdl,   m_hasPolicy,     m_selectedLayer,
                m_rawZoomLayer};

    // Diagram works without any activation data; every other mode needs layers.
    if (m_mode == 4) {
        paintDiagram(body, c);
        return;
    }

    if (m_layers.empty() && !m_hasFinalMap && !m_hasWdl && !m_hasPolicy) {
        drawText(theme, "No CNN snapshot.", static_cast<int>(body.x), static_cast<int>(body.y + 8.0f),
                 16, theme.textMuted);
        drawText(theme, "Set model.lc0_cnn to a converted .bin file.", static_cast<int>(body.x),
                 static_cast<int>(body.y + 32.0f), 14, theme.textDim);
        return;
    }

    switch (m_mode) {
        case 1: paintTrace(body, c); break;
        case 2: paintRaw(body, c); break;
        case 3: paintAtlas(body, c); break;
        case 0:
        default: paintOverview(body, c); break;
    }
}

}  // namespace cnnv::viz
