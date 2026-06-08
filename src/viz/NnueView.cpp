#include "viz/NnueView.h"

#include "nn/ActivationSnapshot.h"
#include "nn/nnue/NnueNetwork.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::nnue::snapshot_keys;
using tensorviz::maxAbs;
using tensorviz::squareName;
using tensorviz::drawCenteredText;
using tensorviz::drawMiniBoard;
using tensorviz::lerfSquare;

// ---------------------------------------------------------------------------
// Small shared helpers (transpiled from chess-rtk TensorViz / NnueAtlas /
// NnueDrawing). The C++ project keeps the orange/blue diverging palette
// (tensorviz::signedColor) rather than crtk's green/coral, so "POSITIVE" reads
// as orange-hot and "NEGATIVE" as blue here; semantics (raise vs lower) match.
// ---------------------------------------------------------------------------

float valueAt(const std::vector<float>& v, int i) {
    return (i >= 0 && i < static_cast<int>(v.size())) ? v[static_cast<std::size_t>(i)] : 0.0f;
}

int safeLen(const std::vector<float>& v) { return static_cast<int>(v.size()); }

const char* planeNameLong(int plane) {
    static const char* names[] = {
        "own pawn", "own knight", "own bishop", "own rook", "own queen",
        "enemy pawn", "enemy knight", "enemy bishop", "enemy rook", "enemy queen",
    };
    return (plane >= 0 && plane < 10) ? names[plane] : "kings";
}

char planeShort(int plane) {
    static const char own[] = {'P', 'N', 'B', 'R', 'Q'};
    static const char enemy[] = {'p', 'n', 'b', 'r', 'q'};
    if (plane >= 0 && plane < 5) return own[plane];
    if (plane >= 5 && plane < 10) return enemy[plane - 5];
    return 'K';
}

// Half-KP feature decode: feature = (kingSq*10 + plane)*64 + pieceSq.
// Black perspective squares are mirrored with `^ 56` for board display.
struct HalfKp {
    int kingSquare = 0;
    int pieceCode = 0;  // 0..9, own {P,N,B,R,Q} then enemy {p,n,b,r,q}.
    int pieceSquare = 0;
    bool valid = false;
};

HalfKp decodeHalfKp(int feature, bool whitePerspective) {
    constexpr int kFeatureCount = 64 * 10 * 64;
    if (feature < 0 || feature >= kFeatureCount) return HalfKp{};
    const int pieceSq = feature % 64;
    const int packed = feature / 64;
    const int plane = packed % 10;
    const int kingSq = packed / 10;
    const int displayKing = whitePerspective ? kingSq : (kingSq ^ 56);
    const int displayPiece = whitePerspective ? pieceSq : (pieceSq ^ 56);
    return HalfKp{displayKing, plane, displayPiece, true};
}

std::string decodeHalfKpLabel(int feature, bool whitePerspective) {
    const HalfKp f = decodeHalfKp(feature, whitePerspective);
    if (!f.valid) return "invalid";
    const char pieceChar = "PNBRQpnbrq"[f.pieceCode < 10 ? f.pieceCode : 0];
    return std::string("K") + squareName(f.kingSquare) + " / " + pieceChar +
           squareName(f.pieceSquare);
}

// Heatmap of a flat vector reshaped as rows x cols, diverging or magnitude.
void drawHeatmap(Rectangle r, const std::vector<float>& values, int cols, int rows,
                 float scale, bool signedColor, const Theme& theme) {
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
            const float v = values[static_cast<std::size_t>(idx)];
            Color c;
            if (signedColor) {
                c = tensorviz::signedColor(v, scale, theme);
            } else {
                const float t = std::sqrt(std::clamp(std::fabs(v) / scale, 0.0f, 1.0f));
                c = tensorviz::blend(tensorviz::withAlpha(theme.buttonIdle, 255),
                                     theme.overlayHot, t);
            }
            DrawRectangleRec(Rectangle{r.x + static_cast<float>(col) * cw,
                                       r.y + static_cast<float>(row) * ch,
                                       std::ceil(cw), std::ceil(ch)},
                             c);
        }
    }
    DrawRectangleLinesEx(r, 1.0f, tensorviz::withAlpha(theme.panelBorder, 160));
}

// Single signed node (filled circle, sign-colored, optional selection ring).
void drawNode(int cx, int cy, int radius, float norm, bool selected,
              const Theme& theme) {
    norm = std::clamp(norm, -1.0f, 1.0f);
    const Color c = tensorviz::signedColor(norm, 1.0f, theme);
    DrawCircle(cx, cy, static_cast<float>(radius), c);
    DrawCircleLines(cx, cy, static_cast<float>(radius),
                    selected ? theme.selection
                             : tensorviz::withAlpha(theme.panelBorder, 200));
    if (selected) {
        DrawCircleLines(cx, cy, static_cast<float>(radius + 2), theme.selection);
    }
}

// Weighted edge between two points: sign-colored, opacity & width by magnitude.
void drawTraceEdge(int x1, int y1, int x2, int y2, float strength, bool emphasized,
                   const Theme& theme) {
    strength = std::clamp(strength, -1.0f, 1.0f);
    const float mag = std::sqrt(std::fabs(strength));
    Color c = tensorviz::signedColor(strength, 1.0f, theme);
    c.a = static_cast<unsigned char>(emphasized ? std::min(255.0f, 120.0f + 120.0f * mag)
                                                : 18.0f + 92.0f * mag);
    DrawLineEx(Vector2{static_cast<float>(x1), static_cast<float>(y1)},
               Vector2{static_cast<float>(x2), static_cast<float>(y2)},
               (emphasized ? 1.7f : 0.7f) + 1.4f * mag, c);
}

void drawCard(Rectangle r, const char* title, const char* subtitle, Color accent,
              const Theme& theme) {
    DrawRectangleRec(r, tensorviz::withAlpha(theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
    if (accent.a != 0) {
        DrawRectangleRec(Rectangle{r.x, r.y, 4.0f, r.height}, accent);
    }
    if (title != nullptr && title[0] != '\0') {
        const std::string t = tensorviz::fitText(theme, title, 14, false, r.width - 24.0f);
        drawText(theme, t.c_str(), static_cast<int>(r.x + 10.0f),
                 static_cast<int>(r.y + 7.0f), 14, theme.textPrimary);
    }
    if (subtitle != nullptr && subtitle[0] != '\0') {
        const std::string s = tensorviz::fitText(theme, subtitle, 10, true, r.width - 24.0f);
        drawTextMono(theme, s.c_str(), static_cast<int>(r.x + 10.0f),
                     static_cast<int>(r.y + 25.0f), 10, theme.textMuted);
    }
}

bool clickedIn(Rectangle r) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
           CheckCollisionPointRec(tensorviz::panelMousePosition(), r);
}

}  // namespace

// ===========================================================================
// update(): copy backend tensors and derive crtk-equivalent quantities.
// ===========================================================================

void NnueView::update(const cnnv::nn::ActivationSnapshot& snap) {
    auto copy = [&](const char* key, std::vector<float>& dst) {
        const float* p = snap.data(key);
        const std::size_t n = snap.size(key);
        if (p == nullptr || n == 0) {
            dst.clear();
        } else {
            dst.assign(p, p + n);
        }
    };

    copy(keys::kAccumulatorWhite, m_accumWhite);
    copy(keys::kAccumulatorBlack, m_accumBlack);
    copy(keys::kAccumulatorClippedUs, m_clippedUs);
    copy(keys::kAccumulatorClippedThem, m_clippedThem);
    copy(keys::kFeatureActiveWhite, m_featuresUs);
    copy(keys::kFeatureActiveBlack, m_featuresThem);
    copy(keys::kFeatureWeightsActiveWhite, m_featureWeightsUs);
    copy(keys::kFeatureWeightsActiveBlack, m_featureWeightsThem);
    copy(keys::kOutputWeightsUs, m_outputWeightsUs);
    copy(keys::kOutputWeightsThem, m_outputWeightsThem);
    copy(keys::kOutputContributionUs, m_outputContributionUs);
    copy(keys::kOutputContributionThem, m_outputContributionThem);

    copy(keys::kAtlasWeights, m_atlasWeights);
    copy(keys::kAtlasKing, m_atlasKing);
    copy(keys::kAtlasOutput, m_atlasOutput);
    {
        const auto& shape = snap.shape(keys::kAtlasWeights);
        m_atlasPlanes = shape.size() >= 3 ? static_cast<int>(shape[1]) : 0;
        m_atlasSquares = shape.size() >= 3 ? static_cast<int>(shape[2]) : 0;
    }

    m_hidden = static_cast<int>(m_accumWhite.size());
    m_activeFeaturesUs = static_cast<int>(m_featuresUs.size());
    m_activeFeaturesThem = static_cast<int>(m_featuresThem.size());

    if (snap.has(keys::kValueCentipawns) && snap.size(keys::kValueCentipawns) > 0) {
        m_centipawns = snap.data(keys::kValueCentipawns)[0];
    } else {
        m_centipawns = 0.0f;
    }

    // Resolve which raw accumulator (white/black) feeds the "us" clipped vector
    // by picking the perspective whose clamped values best match clipped.us.
    auto clipError = [&](const std::vector<float>& raw) {
        const std::size_t n = std::min(raw.size(), m_clippedUs.size());
        if (n == 0) return 1e30;
        double sum = static_cast<double>(std::max(raw.size(), m_clippedUs.size()) - n);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(std::clamp(raw[i], 0.0f, 1.0f) -
                                                 m_clippedUs[i]);
            sum += d * d;
        }
        return sum;
    };
    m_clippedUsFromWhite = clipError(m_accumWhite) <= clipError(m_accumBlack);

    // Total per-slot contribution (crtk nnue.output.contribution.total).
    m_contributionTotal.assign(static_cast<std::size_t>(m_hidden), 0.0f);
    for (int i = 0; i < m_hidden; ++i) {
        m_contributionTotal[static_cast<std::size_t>(i)] =
            valueAt(m_outputContributionUs, i) + valueAt(m_outputContributionThem, i);
    }

    // Per-feature signed impact (crtk emits nnue.features.us.impact; we
    // approximate it as the L1-weighted sum of that feature's row against the
    // current contribution direction: sum_h w[f,h] * sign-weighted slot impact).
    auto computeImpact = [&](const std::vector<float>& features,
                             const std::vector<float>& weights,
                             const std::vector<float>& contribution,
                             std::vector<float>& impact) {
        const int rows = static_cast<int>(features.size());
        impact.assign(static_cast<std::size_t>(rows), 0.0f);
        if (m_hidden <= 0 ||
            static_cast<int>(weights.size()) < rows * m_hidden ||
            contribution.empty()) {
            return;
        }
        // Normalise each feature's weight row by its L1 mass and project onto
        // the per-slot contribution so the resulting cp-scaled impact sums to
        // roughly the eval, the way crtk's exact impact does.
        for (int row = 0; row < rows; ++row) {
            double total = 0.0;
            double sum = 0.0;
            for (int h = 0; h < m_hidden; ++h) {
                const float w =
                    weights[static_cast<std::size_t>(row * m_hidden + h)];
                total += std::fabs(w);
            }
            if (total <= 1e-9) continue;
            for (int h = 0; h < m_hidden; ++h) {
                const float w =
                    weights[static_cast<std::size_t>(row * m_hidden + h)];
                sum += static_cast<double>(w) / total *
                       static_cast<double>(valueAt(contribution, h));
            }
            impact[static_cast<std::size_t>(row)] = static_cast<float>(sum);
        }
    };
    computeImpact(m_featuresUs, m_featureWeightsUs, m_outputContributionUs, m_impactUs);
    computeImpact(m_featuresThem, m_featureWeightsThem, m_outputContributionThem,
                  m_impactThem);

    m_hasData = !m_accumWhite.empty();
}

// ===========================================================================
// Internal drawing namespace: each crtk paint* method as a free function that
// receives the cached tensors via a small context. Kept in the .cpp so the
// header stays slim.
// ===========================================================================

namespace {

/**
 * @brief Buendelt alle Daten, die ein NNUE-Zeichenmodus braucht.
 *
 * Die einzelnen Modi bleiben dadurch freie Zeichenfunktionen: sie lesen die
 * Snapshot-Vektoren nur ueber Referenzen und schreiben nur die kleinen
 * Auswahl-Cursor zurueck. Das haelt den grossen NnueView-Zustand von den
 * mode-spezifischen Layoutfunktionen getrennt.
 */
struct Ctx {
    const Theme& theme;
    Color signature;
    int hidden;
    bool clippedUsFromWhite;
    float centipawns;
    const std::vector<float>& featuresUs;
    const std::vector<float>& featuresThem;
    const std::vector<float>& impactUs;
    const std::vector<float>& impactThem;
    const std::vector<float>& featureWeightsUs;
    const std::vector<float>& featureWeightsThem;
    const std::vector<float>& accumWhite;
    const std::vector<float>& accumBlack;
    const std::vector<float>& clippedUs;
    const std::vector<float>& clippedThem;
    const std::vector<float>& outputWeightsUs;
    const std::vector<float>& outputWeightsThem;
    const std::vector<float>& contribUs;
    const std::vector<float>& contribThem;
    const std::vector<float>& contribTotal;
    const std::vector<float>& atlasWeights;
    const std::vector<float>& atlasKing;
    const std::vector<float>& atlasOutput;
    int atlasPlanes;
    int atlasSquares;
    int& selectedFeature;
    int& selectedSlot;
    int& atlasSelected;
    int& atlasPlane;
};

// Side to move is unknown without a FEN; crtk defaults to White when the FEN is
// absent, so the C++ view does too. The "us" accumulator is white when
// clippedUsFromWhite resolves that way.
bool sideToMoveWhite(const Ctx& c) { return c.clippedUsFromWhite; }

void highlightSquare(Rectangle board, int square, Color color, const Theme& theme) {
    if (square < 0 || square >= 64) return;
    const Rectangle cell = lerfSquare(board, square, true);
    DrawRectangleRec(cell, color);
    DrawRectangleLinesEx(cell, 2.0f, tensorviz::withAlpha(theme.selection, 220));
}

// ----- strongest-driver helpers (NnueOverviewView) -----

struct Driver {
    int row = -1;
    int featureIndex = -1;
    float impact = 0.0f;
    bool valid = false;
};

Driver strongestDriver(const Ctx& c, bool positive) {
    Driver best;
    float bestAbs = -1.0f;
    const int n = std::min(safeLen(c.featuresUs), safeLen(c.impactUs));
    for (int i = 0; i < n; ++i) {
        const float v = c.impactUs[static_cast<std::size_t>(i)];
        if (positive ? v <= 0.0f : v >= 0.0f) continue;
        if (std::fabs(v) > bestAbs) {
            bestAbs = std::fabs(v);
            best = Driver{i, static_cast<int>(std::lround(
                                 c.featuresUs[static_cast<std::size_t>(i)])),
                          v, true};
        }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// 5-segment switcher (self-drawn — drawModeSelector only has 3 segments).
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
// Mode 0 — Overview (ABSTRACT): summary band + signed contributor columns.
// Transpiled from NnueOverviewView.paintAbstract, minus the reconstructed
// HalfKP board because the main board already carries the position context.
// ===========================================================================

void drawSummaryChip(Rectangle r, const char* label, const std::string& value,
                     const std::string& sub, Color accent, const Theme& theme) {
    DrawRectangleRec(r, tensorviz::withAlpha(theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, theme.panelBorder);
    DrawRectangleRec(Rectangle{r.x, r.y, 3.0f, r.height}, accent);
    drawText(theme, label, static_cast<int>(r.x + 9.0f),
             static_cast<int>(r.y + 6.0f), 10, theme.textMuted);
    const std::string v = tensorviz::fitText(theme, value, 17, false, r.width - 16.0f);
    drawText(theme, v.c_str(), static_cast<int>(r.x + 9.0f),
             static_cast<int>(r.y + 20.0f), 17, accent);
    const std::string s = tensorviz::fitText(theme, sub, 10, true, r.width - 16.0f);
    drawTextMono(theme, s.c_str(), static_cast<int>(r.x + 9.0f),
                 static_cast<int>(r.y + 42.0f), 10, theme.textMuted);
}

void paintOverviewSummary(Rectangle r, const Ctx& c) {
    drawCard(r, nullptr, nullptr, c.signature, c.theme);
    const float cp = c.centipawns;
    const Driver raiser = strongestDriver(c, true);
    const Driver lowerer = strongestDriver(c, false);
    const bool white = sideToMoveWhite(c);
    const int chipGap = 8;
    const int chipCount = r.width < 780.0f ? 3 : 4;
    const float chipW =
        std::max(1.0f, (r.width - 20.0f - static_cast<float>(chipGap) *
                                              static_cast<float>(chipCount - 1)) /
                           static_cast<float>(chipCount));
    const float chipH = std::max(46.0f, r.height - 18.0f);
    float x = r.x + 10.0f;
    char evalText[32];
    std::snprintf(evalText, sizeof(evalText), "%+d cp", static_cast<int>(std::lround(cp)));
    drawSummaryChip(Rectangle{x, r.y + 9.0f, chipW, chipH}, "eval", evalText,
                    white ? "side-to-move view" : "side-to-move view",
                    cp >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue, c.theme);
    x += chipW + static_cast<float>(chipGap);
    char raiseVal[24];
    std::snprintf(raiseVal, sizeof(raiseVal), "%+.1f cp", raiser.valid ? raiser.impact : 0.0f);
    drawSummaryChip(Rectangle{x, r.y + 9.0f, chipW, chipH}, "top raiser",
                    raiser.valid ? decodeHalfKpLabel(raiser.featureIndex, white) : "-",
                    raiser.valid ? raiseVal : "none", c.theme.overlayHot, c.theme);
    x += chipW + static_cast<float>(chipGap);
    char lowerVal[24];
    std::snprintf(lowerVal, sizeof(lowerVal), "%+.1f cp", lowerer.valid ? lowerer.impact : 0.0f);
    drawSummaryChip(Rectangle{x, r.y + 9.0f, chipW, chipH}, "top lowerer",
                    lowerer.valid ? decodeHalfKpLabel(lowerer.featureIndex, white) : "-",
                    lowerer.valid ? lowerVal : "none", c.theme.accentBlue, c.theme);
    if (chipCount > 3) {
        x += chipW + static_cast<float>(chipGap);
        char halfkp[24];
        std::snprintf(halfkp, sizeof(halfkp), "%d / %d",
                      safeLen(c.featuresUs), safeLen(c.featuresThem));
        drawSummaryChip(Rectangle{x, r.y + 9.0f, chipW, chipH}, "HalfKP", halfkp,
                        "us / them active", c.signature, c.theme);
    }
}

void paintContributorColumn(Rectangle r, const Ctx& c, bool positive) {
    // Collect signed features and rank by |impact|.
    struct Row { int idx; int feature; float impact; };
    std::vector<Row> rows;
    const int n = std::min(safeLen(c.featuresUs), safeLen(c.impactUs));
    for (int i = 0; i < n; ++i) {
        const float v = c.impactUs[static_cast<std::size_t>(i)];
        if (positive ? v > 0.0f : v < 0.0f) {
            rows.push_back(Row{i, static_cast<int>(std::lround(
                                     c.featuresUs[static_cast<std::size_t>(i)])),
                               v});
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return std::fabs(a.impact) > std::fabs(b.impact);
    });

    DrawRectangleRec(r, tensorviz::withAlpha(c.theme.panelBackground, 235));
    DrawRectangleLinesEx(r, 1.0f, c.theme.panelBorder);

    const Color accent = positive ? c.theme.overlayHot : c.theme.accentBlue;
    drawText(c.theme, positive ? "raise" : "lower",
             static_cast<int>(r.x + 6.0f), static_cast<int>(r.y + 6.0f), 11, accent);

    const int headerH = 24;
    const int rowGap = 4;
    const int rowMin = 38;
    const int rowMax = 48;
    const int maxRows = 10;
    const int availableH = std::max(0, static_cast<int>(r.height) - headerH);
    const int rowsByHeight = std::max(0, (availableH + rowGap) / (rowMin + rowGap));
    const int shown = std::min({static_cast<int>(rows.size()), maxRows, rowsByHeight});
    char count[40];
    if (shown == static_cast<int>(rows.size())) {
        std::snprintf(count, sizeof(count), "%d features", shown);
    } else {
        std::snprintf(count, sizeof(count), "top %d of %d", shown,
                      static_cast<int>(rows.size()));
    }
    drawText(c.theme, count,
             static_cast<int>(r.x + 6.0f + static_cast<float>(
                 measureText(c.theme, positive ? "raise" : "lower", 11)) + 12.0f),
             static_cast<int>(r.y + 7.0f), 10, c.theme.textMuted);
    if (shown == 0) {
        drawText(c.theme,
                 rows.empty() ? "(no signed features active)"
                              : "(not enough vertical room)",
                 static_cast<int>(r.x + 6.0f), static_cast<int>(r.y + 30.0f), 10,
                 c.theme.textMuted);
        return;
    }
    const int rowH = std::min(rowMax,
                              std::max(rowMin, (availableH - rowGap * std::max(0, shown - 1)) /
                                                   std::max(1, shown)));
    float colMax = 0.0f;
    for (int i = 0; i < shown; ++i) {
        colMax = std::max(colMax, std::fabs(rows[static_cast<std::size_t>(i)].impact));
    }
    colMax = std::max(colMax, 1e-6f);
    const bool white = sideToMoveWhite(c);
    for (int i = 0; i < shown; ++i) {
        const Row& row = rows[static_cast<std::size_t>(i)];
        const float y = r.y + static_cast<float>(headerH) +
                        static_cast<float>(i) * static_cast<float>(rowH + rowGap);
        const Rectangle rowR{r.x, y, r.width, static_cast<float>(rowH)};
        const bool selected = row.feature == c.selectedFeature;
        DrawRectangleRec(rowR, tensorviz::withAlpha(c.theme.buttonIdle, 200));
        DrawRectangleLinesEx(rowR, 1.0f, selected ? c.signature : c.theme.panelBorder);
        if (selected) {
            DrawRectangleRec(Rectangle{rowR.x, rowR.y, 3.0f, rowR.height}, c.signature);
        }
        // Mini glyph board for this feature (king + piece tiny board).
        const float glyphSide = std::min(38.0f, std::max(28.0f, static_cast<float>(rowH) - 8.0f));
        const Rectangle glyph{rowR.x + 5.0f, rowR.y + (rowR.height - glyphSide) * 0.5f,
                              glyphSide, glyphSide};
        const HalfKp f = decodeHalfKp(row.feature, white);
        drawMiniBoard(glyph, c.theme);
        if (f.valid) {
            highlightSquare(glyph, f.kingSquare,
                            tensorviz::withAlpha(c.signature, 150), c.theme);
            highlightSquare(glyph, f.pieceSquare, tensorviz::withAlpha(accent, 170), c.theme);
        }
        const float textX = glyph.x + glyph.width + 10.0f;
        char tail[16];
        std::snprintf(tail, sizeof(tail), "%+5.1f cp", row.impact);
        const int tailW = measureText(c.theme, tail, 11);
        const float valueX = rowR.x + rowR.width - static_cast<float>(tailW) - 8.0f;
        const std::string label =
            tensorviz::fitText(c.theme, decodeHalfKpLabel(row.feature, white), 11, false,
                std::max(24.0f, valueX - textX - 8.0f));
        drawText(c.theme, label.c_str(), static_cast<int>(textX),
                 static_cast<int>(rowR.y + 5.0f), 11, c.theme.textPrimary);
        // Magnitude bar.
        const float barW = std::max(1.0f, valueX - textX - 8.0f);
        if (barW > 18.0f) {
            const float barH = rowH < 42 ? 7.0f : 8.0f;
            const Rectangle bar{textX, rowR.y + rowR.height - barH - 8.0f, barW, barH};
            DrawRectangleRec(bar, tensorviz::withAlpha(c.theme.buttonIdle, 230));
            const float ratio = std::min(1.0f, std::fabs(row.impact) / colMax);
            const float fillW = std::max(2.0f, barW * ratio);
            const float fillX = positive ? bar.x : bar.x + barW - fillW;
            DrawRectangleRec(Rectangle{fillX, bar.y, fillW, barH}, accent);
        }
        drawText(c.theme, tail, static_cast<int>(valueX),
                 static_cast<int>(rowR.y + 5.0f), 11,
                 row.impact >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue);
        if (clickedIn(rowR)) {
            c.selectedFeature = (c.selectedFeature == row.feature) ? -1 : row.feature;
        }
    }
}

void paintTopContributors(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 36.0f},
                                 "top feature drivers",
                                 "largest half-KP features pushing the eval up or down",
                                 c.theme, c.signature);
    if (c.featuresUs.empty() || c.impactUs.empty()) return;
    const int colGap = 12;
    const float colW = (r.width - static_cast<float>(colGap)) * 0.5f;
    const float colY = r.y + 40.0f;
    const float colH = std::max(1.0f, r.height - 40.0f);
    paintContributorColumn(Rectangle{r.x, colY, colW, colH}, c, true);
    paintContributorColumn(Rectangle{r.x + colW + static_cast<float>(colGap), colY, colW, colH},
                           c, false);
}

void paintOverview(Rectangle body, const Ctx& c) {
    const int gap = 12;
    const float summaryH = body.width < 760.0f ? 78.0f : 84.0f;
    const Rectangle summary{body.x, body.y, body.width,
                            std::min(summaryH, body.height)};
    paintOverviewSummary(summary, c);
    const Rectangle rest{body.x, summary.y + summary.height + static_cast<float>(gap),
                         body.width,
                         std::max(1.0f, body.y + body.height - summary.y -
                                            summary.height - static_cast<float>(gap))};
    if (rest.height <= 1.0f) return;
    paintTopContributors(rest, c);
}

// ===========================================================================
// Mode 1 — Trace (DETAILED): five-column wired node graph
// (feature -> accumulator -> clipped -> contribution -> output) with weighted
// edges, the layer-stack ribbon and a selected-slot footer.
// Transpiled from NnueTraceView.paintDetailed (classic HalfKP branch only;
// the Stockfish HalfKAv2 branch does not apply to the C++ half-KP backend).
// ===========================================================================

constexpr int kTraceMaxSlots = 16;
constexpr int kTraceFeatureLanes = 30;  // FeatureEncoder.MAX_ACTIVE_FEATURES.

// Ranks slots by |total contribution| and returns the visible-slot indices
// (NnueViewBase.rebuildVisibleSelection).
std::vector<int> visibleSlots(const Ctx& c) {
    std::vector<int> order(static_cast<std::size_t>(c.hidden));
    for (int i = 0; i < c.hidden; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::fabs(valueAt(c.contribTotal, a)) >
               std::fabs(valueAt(c.contribTotal, b));
    });
    if (static_cast<int>(order.size()) > kTraceMaxSlots) {
        order.resize(static_cast<std::size_t>(kTraceMaxSlots));
    }
    return order;
}

struct TraceLayout {
    int featureCx = 0, accumCx = 0, clippedCx = 0, contribCx = 0, outputCx = 0;
    int slotRadius = 8;
    int graphTop = 0, graphBottom = 0;
    int startY = 0, bottomY = 0;
    int featureStartY = 0;
    int slotPitch = 22, featurePitch = 40;
    int labelY = 0;
    int slots = 0, features = 0;
};

int adaptivePitch(int usable, int count, int preferred) {
    if (count <= 1) return preferred;
    return std::clamp(usable / (count - 1), 12, preferred);
}

TraceLayout traceLayout(Rectangle r, int slots, int features) {
    TraceLayout out;
    const int leftCx = static_cast<int>(r.x) + 72;
    const int rightCx = std::max(leftCx + 4, static_cast<int>(r.x + r.width) - 102);
    const int gap = std::max(1, (rightCx - leftCx) / 4);
    out.featureCx = leftCx;
    out.accumCx = leftCx + gap;
    out.clippedCx = leftCx + gap * 2;
    out.contribCx = leftCx + gap * 3;
    out.outputCx = leftCx + gap * 4;
    out.slotRadius = 8;
    const int graphTop = static_cast<int>(r.y) + 40;
    const int graphBottom = static_cast<int>(r.y) +
                            std::max(72, static_cast<int>(r.height) - 28);
    const int usable = std::max(1, graphBottom - graphTop);
    out.graphTop = graphTop;
    out.graphBottom = graphBottom;
    out.slots = std::max(1, slots);
    out.features = std::max(1, features);
    out.slotPitch = adaptivePitch(usable, out.slots, 22);
    out.featurePitch = adaptivePitch(usable, out.features, 40);
    const int slotSpan = (out.slots - 1) * out.slotPitch;
    const int featureSpan = (out.features - 1) * out.featurePitch;
    out.startY = graphTop + std::max(0, (usable - slotSpan) / 2);
    out.featureStartY = graphTop + std::max(0, (usable - featureSpan) / 2);
    out.bottomY = out.startY + (out.slots - 1) * out.slotPitch;
    out.labelY = std::max(static_cast<int>(r.y) + 18,
                          std::min(out.startY, out.featureStartY) - 18);
    return out;
}

void drawTraceStage(Rectangle r, const char* title, const std::string& subtitle,
                    const std::vector<float>& values, bool signedScale, Color accent,
                    const Ctx& c) {
    drawCard(r, title, subtitle.c_str(), accent, c.theme);
    const Rectangle strip{r.x + 10.0f, r.y + r.height - 26.0f,
                          std::max(8.0f, r.width - 20.0f), 12.0f};
    if (values.empty()) {
        DrawRectangleRec(strip, tensorviz::withAlpha(c.theme.buttonIdle, 120));
    } else {
        drawHeatmap(strip, values, static_cast<int>(values.size()), 1,
                    std::max(1e-4f, maxAbs(values)), signedScale, c.theme);
    }
}

void drawTracePipelineRibbon(Rectangle r, const Ctx& c) {
    const int stages = 6;
    const int gap = 8;
    const float stageW =
        std::max(72.0f, (r.width - static_cast<float>(gap) * (stages - 1)) / stages);
    std::array<Rectangle, 6> cards{};
    for (int i = 0; i < stages; ++i) {
        cards[static_cast<std::size_t>(i)] =
            Rectangle{r.x + static_cast<float>(i) * (stageW + static_cast<float>(gap)),
                      r.y,
                      i == stages - 1 ? r.x + r.width - (r.x + static_cast<float>(i) *
                                                                     (stageW + static_cast<float>(gap)))
                                      : stageW,
                      r.height};
    }
    for (int i = 0; i < stages - 1; ++i) {
        const Rectangle& a = cards[static_cast<std::size_t>(i)];
        const int y = static_cast<int>(a.y + a.height * 0.5f);
        const int x1 = static_cast<int>(a.x + a.width + 1.0f);
        const int x2 = static_cast<int>(cards[static_cast<std::size_t>(i + 1)].x - 2.0f);
        if (x2 > x1) {
            const Color col = tensorviz::withAlpha(c.signature, 140);
            DrawLine(x1, y, x2, y, col);
            DrawLine(x2, y, x2 - 4, y - 3, col);
            DrawLine(x2, y, x2 - 4, y + 3, col);
        }
    }
    auto count = [](const std::vector<float>& v, const char* unit) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(v.size()), unit);
        return std::string(buf);
    };
    char cpText[24];
    std::snprintf(cpText, sizeof(cpText), "%+d cp",
                  static_cast<int>(std::lround(c.centipawns)));
    char inputs[40];
    std::snprintf(inputs, sizeof(inputs), "%d us / %d them", safeLen(c.featuresUs),
                  safeLen(c.featuresThem));
    std::vector<float> activeSplit{static_cast<float>(c.featuresUs.size()),
                                   -static_cast<float>(c.featuresThem.size())};
    std::vector<float> cpVec{c.centipawns};
    drawTraceStage(cards[0], "HalfKP inputs", inputs, activeSplit, true,
                   c.theme.overlayHot, c);
    drawTraceStage(cards[1], "accumulator us", count(c.accumWhite, "raw neurons"),
                   c.accumWhite, true, c.theme.accentBlue, c);
    drawTraceStage(cards[2], "accumulator them", count(c.accumBlack, "raw neurons"),
                   c.accumBlack, true, c.theme.accentBlue, c);
    drawTraceStage(cards[3], "clipped us", count(c.clippedUs, "post-ReLU"),
                   c.clippedUs, false, c.theme.accentMagenta, c);
    drawTraceStage(cards[4], "clipped them", count(c.clippedThem, "post-ReLU"),
                   c.clippedThem, false, c.theme.accentYellow, c);
    drawTraceStage(cards[5], "linear output", cpText, cpVec, true,
                   c.theme.accentYellow, c);
}

void drawStageHeader(int cx, int y, int w, const char* index, const char* title,
                     const std::string& detail, const Ctx& c) {
    const Rectangle box{static_cast<float>(cx) - static_cast<float>(w) * 0.5f,
                        static_cast<float>(y), static_cast<float>(w), 30.0f};
    char badge[8];
    std::snprintf(badge, sizeof(badge), "%s", index);
    const Rectangle badgeR{box.x, box.y, 16.0f, 16.0f};
    DrawRectangleRec(badgeR, tensorviz::withAlpha(c.signature, 180));
    drawCenteredText(c.theme, badge, badgeR, 11, c.theme.textPrimary);
    const std::string t = tensorviz::fitText(c.theme, title, 12, false,
                              static_cast<float>(w) - 20.0f);
    drawText(c.theme, t.c_str(), static_cast<int>(box.x + 20.0f),
             static_cast<int>(box.y + 1.0f), 12, c.theme.textPrimary);
    const std::string d = tensorviz::fitText(c.theme, detail, 9, true, static_cast<float>(w));
    drawTextMono(c.theme, d.c_str(), static_cast<int>(box.x),
                 static_cast<int>(box.y + 17.0f), 9, c.theme.textMuted);
}

std::string visibleStageDetail(const std::vector<float>& values, int shown,
                               const char* unit) {
    const int total = safeLen(values);
    char buf[40];
    if (total > 0 && shown > 0 && shown < total) {
        std::snprintf(buf, sizeof(buf), "top %d/%d", shown, total);
    } else {
        std::snprintf(buf, sizeof(buf), "%d %s", total, unit);
    }
    return buf;
}

void paintTrace(Rectangle body, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, 48.0f}, "detailed NNUE node graph",
        "30 active-feature lanes | accumulator -> clipped ReLU -> contribution -> cp",
        c.theme, c.signature);
    const Rectangle ribbon{body.x, body.y + 48.0f + 8.0f, body.width, 92.0f};
    drawTracePipelineRibbon(ribbon, c);

    const int graphTop = static_cast<int>(ribbon.y + ribbon.height) + 14;
    const Rectangle wire{body.x + 4.0f, static_cast<float>(graphTop),
                         std::max(1.0f, body.width - 8.0f),
                         body.height - (static_cast<float>(graphTop) - body.y)};

    const std::vector<int> slots = visibleSlots(c);
    const int featureLanes = std::min(kTraceFeatureLanes, safeLen(c.featuresUs));
    const TraceLayout L =
        traceLayout(wire, static_cast<int>(slots.size()), kTraceFeatureLanes);

    // Stage band backdrops.
    const int bandTop = static_cast<int>(wire.y) + 4;
    const int bandBottom = std::min(static_cast<int>(wire.y + wire.height) - 24,
                                    L.graphBottom + 12);
    const int bandW = std::max(20, std::min(96, (L.accumCx - L.featureCx) - 12));
    const std::array<int, 5> centers{{L.featureCx, L.accumCx, L.clippedCx,
                                      L.contribCx, L.outputCx}};
    const std::array<Color, 5> tints{{c.theme.overlayHot, c.theme.accentBlue,
                                      c.theme.accentMagenta, c.theme.accentYellow,
                                      c.signature}};
    if (bandBottom > bandTop) {
        for (int i = 0; i < 5; ++i) {
            const Rectangle band{static_cast<float>(centers[static_cast<std::size_t>(i)] - bandW / 2),
                                 static_cast<float>(bandTop), static_cast<float>(bandW),
                                 static_cast<float>(bandBottom - bandTop)};
            DrawRectangleRec(band, tensorviz::withAlpha(tints[static_cast<std::size_t>(i)], 16));
            DrawRectangleLinesEx(band, 1.0f, tensorviz::withAlpha(c.theme.panelBorder, 105));
        }
    }

    // Column labels.
    const int w = bandW - 8;
    drawStageHeader(L.featureCx, L.labelY, w, "1", "features",
                    visibleStageDetail(c.featuresUs, featureLanes, "active"), c);
    drawStageHeader(L.accumCx, L.labelY, w, "2", "accumulator",
                    visibleStageDetail(c.accumWhite, static_cast<int>(slots.size()), "slots"), c);
    drawStageHeader(L.clippedCx, L.labelY, w, "3", "clipped",
                    visibleStageDetail(c.clippedUs, static_cast<int>(slots.size()), "slots"), c);
    drawStageHeader(L.contribCx, L.labelY, w, "4", "contrib",
                    visibleStageDetail(c.contribTotal, static_cast<int>(slots.size()), "terms"), c);
    drawStageHeader(L.outputCx, L.labelY, w, "5", "output", "centipawns", c);

    // Edges: features -> accumulator (every active lane to every visible slot),
    // then trunk lanes accumulator -> clipped -> contribution -> output.
    const int hiddenStride = c.hidden;
    float featWeightScale = 1e-6f;
    for (int fi = 0; fi < featureLanes; ++fi) {
        for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
            const int w2 = fi * hiddenStride + slots[static_cast<std::size_t>(si)];
            if (w2 < safeLen(c.featureWeightsUs)) {
                featWeightScale = std::max(featWeightScale,
                                           std::fabs(c.featureWeightsUs[static_cast<std::size_t>(w2)]));
            }
        }
    }
    float contribScale = 1e-6f;
    float clippedScale = 1e-6f;
    for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
        const int idx = slots[static_cast<std::size_t>(si)];
        contribScale = std::max(contribScale, std::fabs(valueAt(c.contribTotal, idx)));
        clippedScale = std::max(clippedScale,
                                std::fabs(valueAt(c.clippedUs, idx) -
                                          valueAt(c.clippedThem, idx)));
    }
    clippedScale = std::max(clippedScale, 1e-6f);
    contribScale = std::max(contribScale, 1e-6f);
    const bool focused = c.selectedSlot >= 0 && c.selectedSlot < static_cast<int>(slots.size());

    for (int fi = 0; fi < featureLanes; ++fi) {
        const int featureY = L.featureStartY + fi * L.featurePitch;
        for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
            const int slotY = L.startY + si * L.slotPitch;
            const int slotIdx = slots[static_cast<std::size_t>(si)];
            float strength = 0.0f;
            const int w2 = fi * hiddenStride + slotIdx;
            if (w2 < safeLen(c.featureWeightsUs)) {
                strength = c.featureWeightsUs[static_cast<std::size_t>(w2)] / featWeightScale;
            }
            drawTraceEdge(L.featureCx + L.slotRadius, featureY,
                          L.accumCx - L.slotRadius, slotY, strength,
                          focused && si == c.selectedSlot, c.theme);
        }
    }
    const int outCy = (L.startY + L.bottomY) / 2;
    for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
        const int slotY = L.startY + si * L.slotPitch;
        const int idx = slots[static_cast<std::size_t>(si)];
        const float w2 = valueAt(c.contribTotal, idx) / contribScale;
        const float gate =
            (valueAt(c.clippedUs, idx) - valueAt(c.clippedThem, idx)) / clippedScale;
        const bool sel = si == c.selectedSlot;
        drawTraceEdge(L.accumCx + L.slotRadius, slotY, L.clippedCx - L.slotRadius,
                      slotY, gate, sel, c.theme);
        drawTraceEdge(L.clippedCx + L.slotRadius, slotY, L.contribCx - L.slotRadius,
                      slotY, w2, sel, c.theme);
        drawTraceEdge(L.contribCx + L.slotRadius, slotY,
                      L.outputCx - (L.slotRadius + 6), outCy, w2, sel, c.theme);
    }

    // Feature column nodes.
    const float impactMax = std::max(maxAbs(c.impactUs), 1e-6f);
    for (int i = 0; i < kTraceFeatureLanes; ++i) {
        const int y = L.featureStartY + i * L.featurePitch;
        if (i >= featureLanes) {
            DrawCircleLines(L.featureCx, y, static_cast<float>(L.slotRadius),
                            tensorviz::withAlpha(c.theme.panelBorder, 130));
            continue;
        }
        const int featureIdx =
            static_cast<int>(std::lround(c.featuresUs[static_cast<std::size_t>(i)]));
        const float v = valueAt(c.impactUs, i) / impactMax;
        drawNode(L.featureCx, y, L.slotRadius, v,
                 featureIdx == c.selectedFeature, c.theme);
        char label[16];
        std::snprintf(label, sizeof(label), "#%d", featureIdx);
        const int lw = measureText(c.theme, label, 10);
        drawText(c.theme, label, L.featureCx - lw - 14, y - 6, 10, c.theme.textMuted);
        const Rectangle hit{static_cast<float>(L.featureCx - lw - 18),
                            static_cast<float>(y - L.slotRadius - 2),
                            static_cast<float>(lw + L.slotRadius * 2 + 22),
                            static_cast<float>(L.slotRadius * 2 + 4)};
        if (clickedIn(hit)) {
            c.selectedFeature = (c.selectedFeature == featureIdx) ? -1 : featureIdx;
        }
    }

    // Trunk node columns: accumulator (net us-them), clipped, contribution.
    float accScale = 1e-6f;
    for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
        const int idx = slots[static_cast<std::size_t>(si)];
        accScale = std::max(accScale, std::fabs(valueAt(c.accumWhite, idx) -
                                                valueAt(c.accumBlack, idx)));
    }
    accScale = std::max(accScale, 1e-6f);
    for (int si = 0; si < static_cast<int>(slots.size()); ++si) {
        const int idx = slots[static_cast<std::size_t>(si)];
        const int y = L.startY + si * L.slotPitch;
        const bool sel = si == c.selectedSlot;
        drawNode(L.accumCx, y, L.slotRadius,
                 (valueAt(c.accumWhite, idx) - valueAt(c.accumBlack, idx)) / accScale,
                 sel, c.theme);
        drawNode(L.clippedCx, y, L.slotRadius,
                 (valueAt(c.clippedUs, idx) - valueAt(c.clippedThem, idx)) / clippedScale,
                 sel, c.theme);
        drawNode(L.contribCx, y, L.slotRadius,
                 valueAt(c.contribTotal, idx) / contribScale, sel, c.theme);
        const Rectangle hit{static_cast<float>(L.accumCx - L.slotRadius - 2),
                            static_cast<float>(y - L.slotRadius - 2),
                            static_cast<float>(L.contribCx - L.accumCx +
                                               L.slotRadius * 2 + 4),
                            static_cast<float>(L.slotRadius * 2 + 4)};
        if (clickedIn(hit)) {
            c.selectedSlot = (c.selectedSlot == si) ? -1 : si;
        }
    }

    // Output column: single node + bar.
    const float outScale = std::max(80.0f, std::fabs(c.centipawns) * 1.4f);
    drawNode(L.outputCx, outCy, L.slotRadius + 6, c.centipawns / outScale, false, c.theme);
    const Rectangle bar{static_cast<float>(L.outputCx + 18), static_cast<float>(outCy - 8),
                        80.0f, 16.0f};
    DrawRectangleRec(bar, tensorviz::withAlpha(c.theme.buttonIdle, 230));
    const float mid = bar.x + bar.width * 0.5f;
    const float norm = std::clamp(c.centipawns / outScale, -1.0f, 1.0f);
    const float len = std::max(1.0f, bar.width * 0.5f * std::fabs(norm));
    DrawRectangleRec(norm >= 0.0f ? Rectangle{mid, bar.y + 2.0f, len, bar.height - 4.0f}
                                  : Rectangle{mid - len, bar.y + 2.0f, len, bar.height - 4.0f},
                     norm >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue);
    char cpText[24];
    std::snprintf(cpText, sizeof(cpText), "%+d cp", static_cast<int>(std::lround(c.centipawns)));
    drawText(c.theme, cpText, L.outputCx + 18, outCy + 16, 11, c.theme.textPrimary);

    // Footer readout.
    if (c.selectedSlot >= 0 && c.selectedSlot < static_cast<int>(slots.size())) {
        const int idx = slots[static_cast<std::size_t>(c.selectedSlot)];
        char text[160];
        std::snprintf(text, sizeof(text),
                      "slot %3d | us %+.3f->%.3f x %+.3f | them %+.3f->%.3f x %+.3f | net %+.3f",
                      idx, valueAt(c.accumWhite, idx), valueAt(c.clippedUs, idx),
                      valueAt(c.outputWeightsUs, idx), valueAt(c.accumBlack, idx),
                      valueAt(c.clippedThem, idx), valueAt(c.outputWeightsThem, idx),
                      valueAt(c.contribTotal, idx));
        drawText(c.theme, tensorviz::fitText(c.theme, text, 11, false, body.width - 16.0f).c_str(),
                 static_cast<int>(body.x + 8.0f),
                 static_cast<int>(body.y + body.height - 18.0f), 11, c.theme.textPrimary);
    } else {
        drawText(c.theme,
                 "Full trace: features -> raw accumulator -> clipped ReLU -> contribution -> cp"
                 "  |  orange raises, blue lowers, opacity = weight size",
                 static_cast<int>(body.x + 8.0f),
                 static_cast<int>(body.y + body.height - 18.0f), 10, c.theme.textMuted);
    }
}

// ===========================================================================
// Mode 2 — All (RAW): dense feature x slot heatmap matrix plus the summary
// vector rows. Transpiled from NnueOverviewView.paintRaw.
// ===========================================================================

void drawRawFeatureRow(Rectangle body, int y, int rowH, int gridLeft, int gridW,
                       const std::vector<float>& indices,
                       const std::vector<float>& impact,
                       const std::vector<float>& weights, int hidden, int sourceIndex,
                       float scale, bool whitePerspective, const Ctx& c) {
    const int featureIdx =
        static_cast<int>(std::lround(indices[static_cast<std::size_t>(sourceIndex)]));
    const float v = valueAt(impact, sourceIndex);
    // Tiny glyph board.
    const int glyphSize = std::max(7, std::min(rowH - 2, 26));
    const Rectangle glyph{body.x + 4.0f, static_cast<float>(y + (rowH - glyphSize) / 2),
                          static_cast<float>(glyphSize), static_cast<float>(glyphSize)};
    drawMiniBoard(glyph, c.theme);
    const HalfKp f = decodeHalfKp(featureIdx, whitePerspective);
    if (f.valid) {
        highlightSquare(glyph, f.kingSquare, tensorviz::withAlpha(c.signature, 150), c.theme);
        highlightSquare(glyph, f.pieceSquare,
                        tensorviz::withAlpha(c.theme.overlayHot, 170), c.theme);
    }
    char label[64];
    std::snprintf(label, sizeof(label), "#%d  %s", featureIdx,
                  decodeHalfKpLabel(featureIdx, whitePerspective).c_str());
    const std::string lbl =
        tensorviz::fitText(c.theme, label, 10, false,
            static_cast<float>(gridLeft) - glyph.x - static_cast<float>(glyphSize) - 68.0f);
    drawText(c.theme, lbl.c_str(), static_cast<int>(glyph.x + glyphSize + 6),
             y + rowH / 2 - 5, 10, c.theme.textPrimary);
    char cp[16];
    std::snprintf(cp, sizeof(cp), "%+5.1f cp", v);
    drawText(c.theme, cp, gridLeft - 56, y + rowH / 2 - 5, 10,
             v >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue);
    // Weight row heatmap.
    std::vector<float> row(static_cast<std::size_t>(hidden), 0.0f);
    const int rowStart = sourceIndex * hidden;
    for (int s = 0; s < hidden && rowStart + s < safeLen(weights); ++s) {
        row[static_cast<std::size_t>(s)] = weights[static_cast<std::size_t>(rowStart + s)];
    }
    const Rectangle cell{static_cast<float>(gridLeft), static_cast<float>(y + 1),
                         static_cast<float>(gridW), static_cast<float>(std::max(1, rowH - 2))};
    drawHeatmap(cell, row, hidden, 1, scale, true, c.theme);
}

int drawRawVectorRow(Rectangle body, int gridTop, int rowH, int gridLeft, int gridW,
                     int row, const char* label, const std::vector<float>& values,
                     int hidden, float scale, bool signedScale, const Ctx& c) {
    if (values.empty()) return row;
    const int y = gridTop + row * rowH;
    drawText(c.theme, label, static_cast<int>(body.x + 4.0f), y + rowH / 2 - 5, 10,
             c.theme.textPrimary);
    const int cols = std::min(safeLen(values), hidden);
    const Rectangle cell{static_cast<float>(gridLeft), static_cast<float>(y + 1),
                         static_cast<float>(gridW), static_cast<float>(std::max(1, rowH - 2))};
    drawHeatmap(cell, values, cols, 1, std::max(scale, 1e-4f), signedScale, c.theme);
    return row + 1;
}

void drawRawGroupLabel(Rectangle body, int y, int rowH, const char* title,
                       const std::string& detail, const Ctx& c) {
    DrawLine(static_cast<int>(body.x), y + rowH - 1,
             static_cast<int>(body.x + body.width), y + rowH - 1, c.theme.panelBorder);
    drawText(c.theme, title, static_cast<int>(body.x + 4.0f), y + rowH / 2 - 5, 10,
             c.theme.textPrimary);
    drawText(c.theme, detail.c_str(), static_cast<int>(body.x + 140.0f), y + rowH / 2 - 5,
             10, c.theme.textMuted);
}

void paintRaw(Rectangle body, const Ctx& c) {
    const int headerH = 38;
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, static_cast<float>(headerH)},
        "all NNUE activations",
        "every active half-KP row plus accumulator, clipped, output-weight and contribution slots",
        c.theme, c.signature);
    if (c.featuresUs.empty() || c.featuresThem.empty() || c.featureWeightsUs.empty() ||
        c.hidden <= 0) {
        drawText(c.theme, "Raw NNUE data unavailable for this snapshot.",
                 static_cast<int>(body.x + 12.0f),
                 static_cast<int>(body.y + headerH + 30.0f), 12, c.theme.textMuted);
        return;
    }
    const int hidden = c.hidden;
    const int usCount = safeLen(c.featuresUs);
    const int themCount = safeLen(c.featuresThem);
    const int featureLanes = std::max(kTraceFeatureLanes, std::max(usCount, themCount));
    const bool whiteToMove = sideToMoveWhite(c);
    const int groupRows = 2;
    const int summaryRows = 8;
    const int totalRows = featureLanes * 2 + groupRows + summaryRows;
    const int rowLabelW = 230;
    const int gridTop = static_cast<int>(body.y) + headerH + 6;
    const int gridH = static_cast<int>(body.height) - headerH - 12;
    const int rowH = std::max(9, gridH / std::max(1, totalRows));
    const int gridLeft = static_cast<int>(body.x) + rowLabelW;
    const int gridW = static_cast<int>(body.width) - rowLabelW - 4;

    const float featScale = std::max({maxAbs(c.featureWeightsUs),
                                      maxAbs(c.featureWeightsThem), 1e-6f});
    const float preReluScale = std::max({maxAbs(c.accumWhite), maxAbs(c.accumBlack), 1e-6f});
    const float clippedScale = std::max({maxAbs(c.clippedUs), maxAbs(c.clippedThem), 1e-6f});
    const float outputWeightScale =
        std::max({maxAbs(c.outputWeightsUs), maxAbs(c.outputWeightsThem), 1e-6f});
    const float contribScale = std::max({maxAbs(c.contribUs), maxAbs(c.contribThem), 1e-6f});

    int row = 0;
    char detailBuf[40];
    std::snprintf(detailBuf, sizeof(detailBuf), "%d / %d active sparse inputs", usCount,
                  featureLanes);
    drawRawGroupLabel(body, gridTop + row * rowH, rowH, "side to move features",
                      detailBuf, c);
    ++row;
    for (int i = 0; i < featureLanes; ++i) {
        if (i < usCount) {
            drawRawFeatureRow(body, gridTop + row * rowH, rowH, gridLeft, gridW,
                              c.featuresUs, c.impactUs, c.featureWeightsUs, hidden, i,
                              featScale, whiteToMove, c);
        } else {
            DrawLine(gridLeft, gridTop + row * rowH + rowH - 1, gridLeft + gridW,
                     gridTop + row * rowH + rowH - 1,
                     tensorviz::withAlpha(c.theme.panelBorder, 90));
        }
        ++row;
    }
    std::snprintf(detailBuf, sizeof(detailBuf), "%d / %d active sparse inputs", themCount,
                  featureLanes);
    drawRawGroupLabel(body, gridTop + row * rowH, rowH, "opponent features", detailBuf, c);
    ++row;
    for (int i = 0; i < featureLanes; ++i) {
        if (i < themCount) {
            drawRawFeatureRow(body, gridTop + row * rowH, rowH, gridLeft, gridW,
                              c.featuresThem, c.impactThem, c.featureWeightsThem, hidden, i,
                              featScale, !whiteToMove, c);
        } else {
            DrawLine(gridLeft, gridTop + row * rowH + rowH - 1, gridLeft + gridW,
                     gridTop + row * rowH + rowH - 1,
                     tensorviz::withAlpha(c.theme.panelBorder, 90));
        }
        ++row;
    }

    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "pre-ReLU accumulator (us)", c.accumWhite, hidden,
                           preReluScale, false, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "pre-ReLU accumulator (them)", c.accumBlack, hidden,
                           preReluScale, false, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "post-ReLU clipped (us)", c.clippedUs, hidden, clippedScale,
                           false, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "post-ReLU clipped (them)", c.clippedThem, hidden, clippedScale,
                           false, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "output weights (us)", c.outputWeightsUs, hidden,
                           outputWeightScale, true, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "output weights (them)", c.outputWeightsThem, hidden,
                           outputWeightScale, true, c);
    row = drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                           "output contribution (us)", c.contribUs, hidden, contribScale,
                           true, c);
    drawRawVectorRow(body, gridTop, rowH, gridLeft, gridW, row,
                     "output contribution (them)", c.contribThem, hidden, contribScale,
                     true, c);
}

// ===========================================================================
// Mode 3 — Atlas (ATLAS): whole pixel-plane learned-weight overview + a
// selected-slot 8x8 plane board with plane chips + a slot explanation.
// Transpiled from NnueAtlasView.paintAtlas / paintAtlasBrowser. The compare/
// diff, variant grid, output-overlay, sort-mode and per-slot all-planes "zoom"
// sub-modes are intentionally simplified out (single magnitude-sorted atlas);
// see report notes.
// ===========================================================================

const char* atlasPlaneLongName(int plane, int planes) {
    return plane < 10 ? planeNameLong(plane) : (planes > 10 ? "kings" : "plane");
}

// Per-slot max-abs scale over its plane x square span.
float atlasSlotScale(const Ctx& c, int slot) {
    const int span = c.atlasPlanes * c.atlasSquares;
    const int base = slot * span;
    float m = 1e-6f;
    for (int i = 0; i < span; ++i) {
        const int idx = base + i;
        if (idx < safeLen(c.atlasWeights)) {
            m = std::max(m, std::fabs(c.atlasWeights[static_cast<std::size_t>(idx)]));
        }
    }
    return m;
}

int strongestAtlasPlane(const Ctx& c, int slot) {
    int best = -1;
    float bestMass = -1.0f;
    for (int p = 0; p < c.atlasPlanes; ++p) {
        float sum = 0.0f;
        const int base = (slot * c.atlasPlanes + p) * c.atlasSquares;
        for (int sq = 0; sq < c.atlasSquares; ++sq) {
            const int idx = base + sq;
            if (idx < safeLen(c.atlasWeights)) {
                sum += std::fabs(c.atlasWeights[static_cast<std::size_t>(idx)]);
            }
        }
        if (sum > bestMass) {
            bestMass = sum;
            best = p;
        }
    }
    return best;
}

// Magnitude-sorted slot order (by |output weight|, NnueAtlasView.sortNeurons
// default branch).
std::vector<int> atlasOrder(const Ctx& c) {
    std::vector<int> order(static_cast<std::size_t>(c.hidden));
    for (int i = 0; i < c.hidden; ++i) order[static_cast<std::size_t>(i)] = i;
    if (safeLen(c.atlasOutput) == c.hidden) {
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::fabs(c.atlasOutput[static_cast<std::size_t>(a)]) >
                   std::fabs(c.atlasOutput[static_cast<std::size_t>(b)]);
        });
    }
    return order;
}

// Draws one slot's plane x square bank as 8px-per-square cells: each plane is an
// 8x8 board column, all planes side by side. Returns nothing; used by the whole
// pixel-plane overview (NnueAtlas.atlasPlaneImage equivalent, drawn directly).
void drawAtlasBank(Rectangle r, const Ctx& c, int slot, float scale) {
    const int planes = c.atlasPlanes;
    if (planes <= 0) return;
    const float planeW = r.width / static_cast<float>(planes);
    const float cellW = planeW / 8.0f;
    const float cellH = r.height / 8.0f;
    for (int p = 0; p < planes; ++p) {
        const int base = (slot * planes + p) * c.atlasSquares;
        for (int sq = 0; sq < 64; ++sq) {
            const int file = sq & 7;
            const int rank = sq >> 3;
            const int drawRank = 7 - rank;  // rank 8 at top (white-down).
            const int idx = base + sq;
            const float v = idx < safeLen(c.atlasWeights)
                                ? c.atlasWeights[static_cast<std::size_t>(idx)] / scale
                                : 0.0f;
            const Color col = tensorviz::signedColor(std::clamp(v, -1.0f, 1.0f), 1.0f, c.theme);
            DrawRectangleRec(Rectangle{r.x + static_cast<float>(p) * planeW +
                                           static_cast<float>(file) * cellW,
                                       r.y + static_cast<float>(drawRank) * cellH,
                                       std::ceil(cellW), std::ceil(cellH)},
                             col);
        }
    }
}

void paintAtlasOverview(Rectangle r, const Ctx& c, const std::vector<int>& order,
                        int selectedSlot) {
    drawCard(r, "whole pixel-plane atlas",
             "every row = one slot's king-averaged weight footprint; click a row to select",
             c.signature, c.theme);
    const Rectangle inner{r.x + 10.0f, r.y + 38.0f, std::max(1.0f, r.width - 20.0f),
                          std::max(1.0f, r.height - 48.0f)};
    if (c.atlasSquares != 64 || c.hidden <= 0 || c.atlasPlanes <= 0 ||
        inner.width <= 12.0f || inner.height <= 16.0f) {
        DrawRectangleRec(inner, tensorviz::withAlpha(c.theme.buttonIdle, 120));
        return;
    }
    const float labelH = 14.0f;
    const int columnGap = 8;
    const float rowAreaH = std::max(1.0f, inner.height - labelH);
    // Wrap the hidden slots into vertical banks so the whole atlas fits.
    const int nativeW = std::max(8, c.atlasPlanes * 8);
    const int preferredW = std::min(320, std::max(160, nativeW * 2));
    const int maxColumns = std::max(1, (static_cast<int>(inner.width) + columnGap) /
                                           (preferredW + columnGap));
    const int neededColumns = std::max(1, (c.hidden + 127) / 128);
    const int columns = std::max(1, std::min(std::min(c.hidden, maxColumns), neededColumns));
    const int rowsPerColumn = std::max(1, (c.hidden + columns - 1) / columns);
    const float columnW =
        std::max(1.0f, (inner.width - static_cast<float>(columnGap) *
                                          static_cast<float>(std::max(0, columns - 1))) /
                           static_cast<float>(columns));
    const int slotPitch =
        std::max(1, std::min(c.hidden >= 512 ? 6 : 8,
                             static_cast<int>(rowAreaH) / rowsPerColumn));

    for (int column = 0; column < columns; ++column) {
        const int startRow = column * rowsPerColumn;
        const int rowCount = std::min(rowsPerColumn, c.hidden - startRow);
        if (rowCount <= 0) continue;
        const float columnX =
            inner.x + static_cast<float>(column) * (columnW + static_cast<float>(columnGap));
        // Plane labels.
        if (columnW / static_cast<float>(c.atlasPlanes) >= 10.0f) {
            for (int p = 0; p < c.atlasPlanes; ++p) {
                const float x0 = columnX + static_cast<float>(p) * columnW /
                                               static_cast<float>(c.atlasPlanes);
                char lab[2] = {planeShort(p), '\0'};
                drawText(c.theme, lab, static_cast<int>(x0 + 1.0f),
                         static_cast<int>(inner.y), 9, c.theme.textMuted);
            }
        }
        const Rectangle imageRect{columnX, inner.y + labelH, columnW,
                                  static_cast<float>(std::min(static_cast<int>(rowAreaH),
                                                              rowCount * slotPitch))};
        for (int rr = 0; rr < rowCount; ++rr) {
            const int slot = order[static_cast<std::size_t>(startRow + rr)];
            const Rectangle bankRow{imageRect.x,
                                    imageRect.y + static_cast<float>(rr * slotPitch),
                                    imageRect.width, static_cast<float>(slotPitch)};
            drawAtlasBank(bankRow, c, slot, atlasSlotScale(c, slot));
            if (clickedIn(bankRow)) {
                c.atlasSelected = slot;
            }
            if (slot == selectedSlot) {
                DrawRectangleLinesEx(bankRow, 2.0f, c.signature);
            }
        }
        DrawRectangleLinesEx(imageRect, 1.0f, c.theme.panelBorder);
    }
}

void paintAtlasSlotDetail(Rectangle r, const Ctx& c, int slot, float scale) {
    char title[40];
    std::snprintf(title, sizeof(title), "slot #%d atlas", slot);
    char sub[40];
    std::snprintf(sub, sizeof(sub), "%s weights",
                  atlasPlaneLongName(c.atlasPlane, c.atlasPlanes));
    drawCard(r, title, sub, c.signature, c.theme);
    const Rectangle inner{r.x + 12.0f, r.y + 38.0f, std::max(1.0f, r.width - 24.0f),
                          std::max(1.0f, r.height - 50.0f)};
    // Plane chips.
    const int chipGap = 5;
    const float chipH = 24.0f;
    const float chipW =
        std::max(28.0f, std::min(58.0f, (inner.width - static_cast<float>(chipGap) *
                                                          static_cast<float>(std::max(0, c.atlasPlanes - 1))) /
                                            static_cast<float>(std::max(1, c.atlasPlanes))));
    for (int p = 0; p < c.atlasPlanes; ++p) {
        const Rectangle chip{inner.x + static_cast<float>(p) * (chipW + static_cast<float>(chipGap)),
                             inner.y, chipW, chipH};
        const bool selected = p == c.atlasPlane;
        DrawRectangleRec(chip, selected ? tensorviz::withAlpha(c.signature, 90)
                                        : tensorviz::withAlpha(c.theme.buttonIdle, 230));
        DrawRectangleLinesEx(chip, 1.0f, selected ? c.signature : c.theme.panelBorder);
        char lab[2] = {planeShort(p), '\0'};
        drawCenteredText(c.theme, lab, chip, 11, c.theme.textPrimary);
        if (clickedIn(chip)) c.atlasPlane = p;
    }
    // Board for selected plane.
    const float boardTop = inner.y + chipH + 14.0f;
    const float side = std::max(80.0f, std::min(inner.width,
                                                std::max(80.0f, inner.y + inner.height -
                                                                    boardTop - 30.0f)));
    const Rectangle board{inner.x + std::max(0.0f, (inner.width - side) * 0.5f), boardTop,
                          side, side};
    const int offset = (slot * c.atlasPlanes + c.atlasPlane) * c.atlasSquares;
    drawMiniBoard(board, c.theme);
    for (int sq = 0; sq < 64; ++sq) {
        const Rectangle cell = lerfSquare(board, sq, true);
        const int idx = offset + sq;
        const float v = idx < safeLen(c.atlasWeights)
                            ? c.atlasWeights[static_cast<std::size_t>(idx)] /
                                  std::max(scale, 1e-6f)
                            : 0.0f;
        DrawRectangleRec(cell, tensorviz::signedColor(std::clamp(v, -1.0f, 1.0f), 1.0f, c.theme));
    }
    DrawRectangleLinesEx(board, 1.0f, c.theme.panelBorder);
}

int drawAtlasMetricLine(Rectangle bounds, int y, const char* label,
                        const std::string& value, Color accent, const Ctx& c) {
    drawText(c.theme, label, static_cast<int>(bounds.x), y + 1, 10, c.theme.textMuted);
    const std::string v = tensorviz::fitText(c.theme, value, 12, false, std::max(20.0f, bounds.width - 120.0f));
    drawText(c.theme, v.c_str(), static_cast<int>(bounds.x + 120.0f), y, 12, accent);
    return y + 22;
}

void paintAtlasSlotExplanation(Rectangle r, const Ctx& c, int slot) {
    const int topPlane = strongestAtlasPlane(c, slot);
    char sub[48];
    std::snprintf(sub, sizeof(sub), "dominant: %s",
                  topPlane >= 0 ? atlasPlaneLongName(topPlane, c.atlasPlanes) : "mixed");
    drawCard(r, "slot explanation", sub, c.signature, c.theme);
    const Rectangle inner{r.x + 12.0f, r.y + 42.0f, std::max(1.0f, r.width - 24.0f),
                          std::max(1.0f, r.height - 54.0f)};
    int y = static_cast<int>(inner.y);
    const float contrib = valueAt(c.contribTotal, slot);
    const float out = valueAt(c.atlasOutput, slot);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.3f", contrib);
    y = drawAtlasMetricLine(inner, y, "current contribution", buf,
                            contrib >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue, c);
    std::snprintf(buf, sizeof(buf), "%+.3f", out);
    y = drawAtlasMetricLine(inner, y, "output weight", buf,
                            out >= 0.0f ? c.theme.overlayHot : c.theme.accentBlue, c);
    y = drawAtlasMetricLine(inner, y, "dominant plane",
                            topPlane >= 0 ? atlasPlaneLongName(topPlane, c.atlasPlanes) : "-",
                            c.signature, c);
    // Strongest +/- squares in the currently selected plane.
    const int offset = (slot * c.atlasPlanes + c.atlasPlane) * c.atlasSquares;
    int posSq = -1, negSq = -1;
    float posV = -1e30f, negV = 1e30f;
    for (int sq = 0; sq < c.atlasSquares; ++sq) {
        const int idx = offset + sq;
        const float v = idx < safeLen(c.atlasWeights)
                            ? c.atlasWeights[static_cast<std::size_t>(idx)]
                            : 0.0f;
        if (v > posV) { posV = v; posSq = sq; }
        if (v < negV) { negV = v; negSq = sq; }
    }
    char sqBuf[32];
    std::snprintf(sqBuf, sizeof(sqBuf), "%s %+.3f", posSq >= 0 ? squareName(posSq) : "-", posV);
    y = drawAtlasMetricLine(inner, y, "strongest + square", sqBuf, c.theme.overlayHot, c);
    std::snprintf(sqBuf, sizeof(sqBuf), "%s %+.3f", negSq >= 0 ? squareName(negSq) : "-", negV);
    y = drawAtlasMetricLine(inner, y, "strongest - square", sqBuf, c.theme.accentBlue, c);
    // King-preference mini board.
    if (safeLen(c.atlasKing) >= (slot + 1) * c.atlasSquares && y < inner.y + inner.height - 90.0f) {
        y += 8;
        drawText(c.theme, "king preference", static_cast<int>(inner.x), y, 11,
                 c.theme.textPrimary);
        y += 18;
        const float side = std::min(inner.width, static_cast<float>(inner.y + inner.height) -
                                                     static_cast<float>(y) - 4.0f);
        if (side >= 40.0f) {
            const Rectangle board{inner.x, static_cast<float>(y), side, side};
            const int kBase = slot * c.atlasSquares;
            float kScale = 1e-6f;
            for (int i = 0; i < c.atlasSquares; ++i) {
                kScale = std::max(kScale,
                                  std::fabs(c.atlasKing[static_cast<std::size_t>(kBase + i)]));
            }
            drawMiniBoard(board, c.theme);
            for (int sq = 0; sq < 64; ++sq) {
                const Rectangle cell = lerfSquare(board, sq, true);
                const float v = c.atlasKing[static_cast<std::size_t>(kBase + sq)] / kScale;
                DrawRectangleRec(cell, tensorviz::signedColor(std::clamp(v, -1.0f, 1.0f), 1.0f,
                                                              c.theme));
            }
            DrawRectangleLinesEx(board, 1.0f, c.theme.panelBorder);
        }
    }
}

void paintAtlas(Rectangle body, const Ctx& c) {
    if (c.atlasWeights.empty() || c.atlasPlanes < 1 || c.atlasSquares < 1) {
        drawText(c.theme, "Atlas data unavailable.", static_cast<int>(body.x + 12.0f),
                 static_cast<int>(body.y + 32.0f), 13, c.theme.textMuted);
        return;
    }
    const int headerH = 46;
    char headerSub[96];
    std::snprintf(headerSub, sizeof(headerSub),
                  "%d accumulator slots x %d piece planes - magnitude sorted",
                  c.hidden, c.atlasPlanes);
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, static_cast<float>(headerH)},
        "slot atlas browser", headerSub, c.theme, c.signature);
    const Rectangle content{body.x, body.y + static_cast<float>(headerH) + 10.0f, body.width,
                            std::max(1.0f, body.height - static_cast<float>(headerH) - 10.0f)};
    const std::vector<int> order = atlasOrder(c);
    const int selectedSlot = (c.atlasSelected >= 0 && c.atlasSelected < c.hidden)
                                 ? c.atlasSelected
                                 : (order.empty() ? 0 : order[0]);
    c.atlasPlane = std::clamp(c.atlasPlane, 0, std::max(0, c.atlasPlanes - 1));

    const int gap = 12;
    if (content.width >= 900.0f) {
        const float rightW = std::min(460.0f, std::max(340.0f, content.width / 4.0f));
        const float leftW = std::max(1.0f, content.width - rightW - static_cast<float>(gap));
        const Rectangle overview{content.x, content.y, leftW, content.height};
        float detailH = std::min(content.height,
                                 std::max(260.0f, std::min(content.height * 0.58f, rightW + 110.0f)));
        if (content.height - detailH - static_cast<float>(gap) < 150.0f) {
            detailH = std::max(220.0f, content.height - static_cast<float>(gap) - 150.0f);
        }
        const Rectangle detail{overview.x + overview.width + static_cast<float>(gap), content.y,
                               rightW, std::max(1.0f, detailH)};
        const Rectangle explanation{detail.x, detail.y + detail.height + static_cast<float>(gap),
                                    rightW,
                                    std::max(1.0f, content.y + content.height - detail.y -
                                                       detail.height - static_cast<float>(gap))};
        paintAtlasOverview(overview, c, order, selectedSlot);
        paintAtlasSlotDetail(detail, c, selectedSlot, atlasSlotScale(c, selectedSlot));
        paintAtlasSlotExplanation(explanation, c, selectedSlot);
    } else {
        const float detailH = std::max(230.0f, std::min(330.0f, content.height / 3.0f));
        const float explanationH = std::max(150.0f, std::min(230.0f, content.height / 4.0f));
        const float overviewH =
            std::max(160.0f, content.height - detailH - explanationH - static_cast<float>(gap) * 2.0f);
        const Rectangle overview{content.x, content.y, content.width,
                                 std::min(content.height, overviewH)};
        const Rectangle detail{content.x, overview.y + overview.height + static_cast<float>(gap),
                               content.width,
                               std::max(1.0f, std::min(detailH, content.y + content.height -
                                                                    overview.y - overview.height -
                                                                    static_cast<float>(gap)))};
        const Rectangle explanation{content.x, detail.y + detail.height + static_cast<float>(gap),
                                    content.width,
                                    std::max(1.0f, content.y + content.height - detail.y -
                                                       detail.height - static_cast<float>(gap))};
        paintAtlasOverview(overview, c, order, selectedSlot);
        paintAtlasSlotDetail(detail, c, selectedSlot, atlasSlotScale(c, selectedSlot));
        paintAtlasSlotExplanation(explanation, c, selectedSlot);
    }
}

// ===========================================================================
// Mode 4 — Diagram (DIAGRAM): static five-box architecture schematic.
// Transpiled from NnueAtlasView.paintDiagram.
// ===========================================================================

void paintDiagram(Rectangle body, const Ctx& c) {
    const int headerH = 40;
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, static_cast<float>(headerH)},
        "architecture diagram", "schematic of the loaded NNUE half-KP evaluator",
        c.theme, c.signature);
    const int boxW = 200;
    const int boxH = 90;
    const int gap = 36;
    const int hidden = c.hidden;
    const int planes = c.atlasPlanes;
    const int squares = c.atlasSquares;
    static const char* titles[5] = {"input features", "feature transformer",
                                    "ReLU + scale", "output stack", "centipawn eval"};
    char sub0[48], sub1[48], sub3[48], sub4[48];
    std::snprintf(sub0, sizeof(sub0), "halfKP %d planes x %d squares",
                  planes == 0 ? 10 : planes, squares == 0 ? 64 : squares);
    std::snprintf(sub1, sizeof(sub1), "[features -> %d hidden]", hidden);
    std::snprintf(sub3, sizeof(sub3), "%d -> output (cp)", hidden);
    std::snprintf(sub4, sizeof(sub4), "active us=%d - them=%d", safeLen(c.featuresUs),
                  safeLen(c.featuresThem));
    const char* subs[5] = {sub0, sub1, "post-ReLU clipped activations", sub3, sub4};

    const int totalW = boxW * 5 + gap * 4;
    const int startX = static_cast<int>(body.x) +
                       (static_cast<int>(body.width) - totalW) / 2;
    const int y = static_cast<int>(body.y) + headerH + 60;
    for (int i = 0; i < 5; ++i) {
        const int x = startX + i * (boxW + gap);
        const Rectangle box{static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(boxW), static_cast<float>(boxH)};
        DrawRectangleRec(box, tensorviz::withAlpha(c.theme.panelBackground, 240));
        DrawRectangleLinesEx(box, 1.0f,
                             i == 4 ? c.theme.accentYellow : c.theme.panelBorder);
        drawCenteredText(c.theme, titles[i],
                     Rectangle{box.x, box.y + 8.0f, box.width, 24.0f}, 13,
                     c.theme.textPrimary);
        const std::string sub = tensorviz::fitText(c.theme, subs[i], 11, false, static_cast<float>(boxW) - 12.0f);
        drawCenteredText(c.theme, sub.c_str(),
                     Rectangle{box.x, box.y + 40.0f, box.width, 20.0f}, 11,
                     c.theme.textMuted);
        if (i < 4) {
            const int ax1 = x + boxW;
            const int ax2 = x + boxW + gap;
            const int ay = y + boxH / 2;
            DrawLine(ax1 + 2, ay, ax2 - 6, ay, c.signature);
            DrawTriangle(Vector2{static_cast<float>(ax2 - 6), static_cast<float>(ay)},
                         Vector2{static_cast<float>(ax2 - 12), static_cast<float>(ay + 5)},
                         Vector2{static_cast<float>(ax2 - 12), static_cast<float>(ay - 5)},
                         c.signature);
        }
    }
    char caption[32];
    std::snprintf(caption, sizeof(caption), "eval %+d cp",
                  static_cast<int>(std::lround(c.centipawns)));
    const int cw = measureText(c.theme, caption, 12);
    drawText(c.theme, caption,
             static_cast<int>(body.x) + (static_cast<int>(body.width) - cw) / 2,
             y + boxH + 28, 12, c.theme.textMuted);
}

}  // namespace

// ===========================================================================
// Public entry point: header band + 5-segment switcher + mode dispatch.
// ===========================================================================

void NnueView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    float y = m_bounds.y + pad;

    // 5-segment switcher (Overview / Trace / All / Atlas / Diagram).
    const Rectangle switcher{
        m_bounds.x + m_bounds.width - pad - 310.0f, y + 1.0f,
        310.0f, 28.0f};
    const std::string title =
        tensorviz::fitText(theme, "NNUE inspector", 22, false,
            switcher.x - (m_bounds.x + pad) - 10.0f);
    drawText(theme, title.c_str(), static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 22, theme.textPrimary);
    m_mode = drawSwitcher(switcher, m_mode, m_signature, theme);
    y += 34.0f;

    char subtitle[128];
    if (m_hasData) {
        std::snprintf(subtitle, sizeof(subtitle),
                      "half-KP fp32  -  %+d cp (side to move)  -  H=%d  -  active w/b %d/%d",
                      static_cast<int>(std::lround(m_centipawns)), m_hidden,
                      m_activeFeaturesUs, m_activeFeaturesThem);
    } else {
        std::snprintf(subtitle, sizeof(subtitle), "half-KP fp32  -  no evaluation yet");
    }
    const std::string subtitleFit =
        tensorviz::fitText(theme, subtitle, 13, false, m_bounds.width - 2.0f * pad);
    drawText(theme, subtitleFit.c_str(), static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 13, theme.textMuted);
    y += 22.0f;

    const Rectangle body{m_bounds.x + pad, y, m_bounds.width - 2.0f * pad,
                         m_bounds.y + m_bounds.height - y - pad};

    /**
     * @brief Gemeinsamer Zeichenkontext fuer alle NNUE-Unteransichten.
     *
     * Der Kontext ist auch im Leerzustand gueltig, weil er nur Referenzen auf
     * die gecachten Vektoren enthaelt. So braucht der Diagramm-Modus keinen
     * zweiten Initialisierungsblock.
     */
    const Ctx c{theme,
                m_signature,
                m_hidden,
                m_clippedUsFromWhite,
                m_centipawns,
                m_featuresUs,
                m_featuresThem,
                m_impactUs,
                m_impactThem,
                m_featureWeightsUs,
                m_featureWeightsThem,
                m_accumWhite,
                m_accumBlack,
                m_clippedUs,
                m_clippedThem,
                m_outputWeightsUs,
                m_outputWeightsThem,
                m_outputContributionUs,
                m_outputContributionThem,
                m_contributionTotal,
                m_atlasWeights,
                m_atlasKing,
                m_atlasOutput,
                m_atlasPlanes,
                m_atlasSquares,
                m_selectedFeature,
                m_selectedSlot,
                m_atlasSelected,
                m_atlasPlane};

    // Diagram works without an evaluation; every other mode needs data.
    if (m_mode == 4) {
        paintDiagram(body, c);
        return;
    }

    if (!m_hasData) {
        drawText(theme, "Run an evaluation to populate this NNUE view.",
                 static_cast<int>(body.x), static_cast<int>(body.y + 8.0f), 16,
                 theme.textMuted);
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
