#include "viz/Bt4View.h"

#include "chess/Position.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"
#include "viz/PieceSprites.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::lc0_bt4::snapshot_keys;
using tensorviz::maxAbs;
using tensorviz::squareName;
using tensorviz::drawCenteredText;
using tensorviz::drawMiniBoard;
using tensorviz::lerfSquare;

constexpr int kTokens = cnnv::nn::lc0_bt4::Network::kTokens;  // 64.

// ---------------------------------------------------------------------------
// Small shared helpers (transpiled from chess-rtk TensorViz / Bt4View). The
// C++ project keeps the orange/blue diverging palette (tensorviz::signedColor)
// rather than crtk's green/coral, so crtk's POSITIVE reads as orange-hot and
// NEGATIVE as blue here; crtk's POLICY/VALUE/FOCUS map to the theme accents
// below. Most boards are theme square colours; the trace attention board can
// draw live pieces when App supplies the position and sprite atlas.
// ---------------------------------------------------------------------------

int detectBlockCount(const cnnv::nn::ActivationSnapshot& snap) {
    const std::string prefix = "bt4.block";
    int count = 0;
    for (const auto& item : snap.entries()) {
        const std::string& key = item.first;
        if (key.rfind(prefix, 0) != 0) continue;
        std::size_t pos = prefix.size();
        if (pos >= key.size() ||
            !std::isdigit(static_cast<unsigned char>(key[pos]))) {
            continue;
        }
        int idx = 0;
        while (pos < key.size() &&
               std::isdigit(static_cast<unsigned char>(key[pos]))) {
            idx = idx * 10 + (key[pos] - '0');
            ++pos;
        }
        if (pos < key.size() && key[pos] == '.') {
            count = std::max(count, idx + 1);
        }
    }
    return count;
}

bool clickedIn(Rectangle r) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
           CheckCollisionPointRec(tensorviz::panelMousePosition(), r);
}

void drawBoardPieces(Rectangle board, const cnnv::chess::Position* position,
                     const PieceSprites* sprites) {
    if (position == nullptr || sprites == nullptr) return;
    for (int sq = 0; sq < 64; ++sq) {
        const cnnv::chess::Piece p = position->pieceAt(sq);
        if (p.isNone()) continue;
        const Texture2D& tex = sprites->textureFor(p);
        if (tex.id == 0) continue;
        DrawTexturePro(tex,
                       Rectangle{0.0f, 0.0f, static_cast<float>(tex.width),
                                 static_cast<float>(tex.height)},
                       lerfSquare(board, sq), Vector2{0.0f, 0.0f}, 0.0f,
                       WHITE);
    }
}

// Heatmap of a flat row-major vector reshaped as rows x cols.
//
// crtk uses a green ramp (positive) for attention magnitudes; here we use the
// theme's magnitude ramp (buttonIdle -> overlayHot) when signedScale is false,
// or the diverging signed palette when it is true. gamma==true applies the
// sqrt-gamma that keeps small attention values visible (crtk drawGammaHeatmap /
// drawSignedGammaHeatmap), which is load-bearing for the dense attention maps.
void drawHeatmap(Rectangle r, const std::vector<float>& values, int cols, int rows,
                 float scale, bool signedScale, bool gamma, const Theme& theme,
                 bool border = true) {
    if (cols <= 0 || rows <= 0 || values.empty()) {
        DrawRectangleRec(r, tensorviz::withAlpha(theme.buttonIdle, 120));
        if (border) DrawRectangleLinesEx(r, 1.0f, tensorviz::withAlpha(theme.panelBorder, 160));
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
            if (signedScale) {
                c = tensorviz::signedColor(v, scale, theme);
            } else {
                float t = std::clamp(std::fabs(v) / scale, 0.0f, 1.0f);
                if (gamma) t = std::sqrt(t);
                c = tensorviz::blend(tensorviz::withAlpha(theme.buttonIdle, 255),
                                     theme.overlayHot, t);
            }
            DrawRectangleRec(Rectangle{r.x + static_cast<float>(col) * cw,
                                       r.y + static_cast<float>(row) * ch,
                                       std::ceil(cw), std::ceil(ch)},
                             c);
        }
    }
    if (border) DrawRectangleLinesEx(r, 1.0f, tensorviz::withAlpha(theme.panelBorder, 160));
}

[[maybe_unused]] void drawCard(Rectangle r, const char* title, const char* subtitle, Color accent,
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

// Highlights one board square with a translucent overlay + selection ring.
[[maybe_unused]] void highlightSquare(Rectangle board, int square, Color color, const Theme& theme) {
    if (square < 0 || square >= 64) return;
    const Rectangle cell = lerfSquare(board, square);
    DrawRectangleRec(cell, color);
    DrawRectangleLinesEx(cell, 2.0f, tensorviz::withAlpha(theme.selection, 220));
}

// Mean attention received per board square for one head (crtk
// headReceivedEnergy): energy[to] = (sum over from of attn[from][to]) / 64.
std::vector<float> headReceivedEnergy(const std::vector<float>& attention, int head) {
    std::vector<float> out(static_cast<std::size_t>(kTokens), 0.0f);
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;
    const std::size_t off = static_cast<std::size_t>(head) * plane;
    if (head < 0 || off + plane > attention.size()) return out;
    for (int from = 0; from < kTokens; ++from) {
        for (int to = 0; to < kTokens; ++to) {
            out[static_cast<std::size_t>(to)] +=
                attention[off + static_cast<std::size_t>(from) * kTokens + to];
        }
    }
    for (int sq = 0; sq < kTokens; ++sq) {
        out[static_cast<std::size_t>(sq)] /= static_cast<float>(kTokens);
    }
    return out;
}

float headFocusScore(const std::vector<float>& energy) {
    float peak = 0.0f;
    for (float v : energy) peak = std::max(peak, v);
    return peak;
}

int strongestSquare(const std::vector<float>& values) {
    if (values.size() < 64) return -1;
    int best = 0;
    float bestValue = values[0];
    for (int sq = 1; sq < 64; ++sq) {
        if (values[static_cast<std::size_t>(sq)] > bestValue) {
            bestValue = values[static_cast<std::size_t>(sq)];
            best = sq;
        }
    }
    return best;
}

const char* attentionImportance(float ratio) {
    if (ratio >= 0.82f) return "primary";
    if (ratio >= 0.55f) return "high";
    if (ratio >= 0.28f) return "medium";
    return "low";
}

}  // namespace

// ===========================================================================
// update(): copy backend tensors and derive crtk-equivalent quantities.
// ===========================================================================

void Bt4View::update(const cnnv::nn::ActivationSnapshot& snap) {
    auto copy = [&](const std::string& key, std::vector<float>& dst) {
        const float* p = snap.data(key);
        const std::size_t n = snap.size(key);
        if (p == nullptr || n == 0) {
            dst.clear();
        } else {
            dst.assign(p, p + n);
        }
    };

    copy(keys::kInputPlanes, m_inputPlanes);
    copy(keys::kTokenFeatures, m_tokenFeatures);
    copy(keys::kEmbedding, m_embedding);
    copy(keys::kFinalTokens, m_finalTokens);
    copy(keys::kFinalTokenMagnitude, m_tokenEnergy);
    copy(keys::kBoardSalience, m_boardSalience);

    m_hasTokenEnergy = m_tokenEnergy.size() >= static_cast<std::size_t>(kTokens);

    const int blockCount = detectBlockCount(snap);
    m_blocks.assign(static_cast<std::size_t>(std::max(0, blockCount)), Block{});
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;
    for (int i = 0; i < blockCount; ++i) {
        Block& block = m_blocks[static_cast<std::size_t>(i)];
        copy(keys::blockAttentionKey(i), block.attention);
        copy(keys::blockFfnKey(i), block.ffn);
        // Derive the head count from the tensor instead of hardcoding it.
        block.heads = (plane > 0 && !block.attention.empty() &&
                       block.attention.size() % plane == 0)
                          ? static_cast<int>(block.attention.size() / plane)
                          : 0;
        // attentionFocus = mean attention magnitude over the whole tensor.
        if (!block.attention.empty()) {
            double sum = 0.0;
            for (float v : block.attention) sum += v;
            block.attentionFocus =
                static_cast<float>(sum / static_cast<double>(block.attention.size()));
        }
        // ffnRms = RMS of the FFN output (crtk TensorViz.summarize[2]).
        if (!block.ffn.empty()) {
            double sumSq = 0.0;
            for (float v : block.ffn) sumSq += static_cast<double>(v) * v;
            block.ffnRms =
                static_cast<float>(std::sqrt(sumSq / static_cast<double>(block.ffn.size())));
        }
        block.have = block.heads > 0 || !block.ffn.empty();
    }

    copy(keys::kPolicyLogits, m_policy);
    m_hasPolicy = !m_policy.empty();

    m_hasWdl = false;
    if (snap.size(keys::kValueWdl) == 3) {
        const float* w = snap.data(keys::kValueWdl);
        m_wdl = {w[0], w[1], w[2]};
        m_hasWdl = true;
    }
    if (snap.size(keys::kValueScalar) == 1) {
        m_valueScalar = snap.data(keys::kValueScalar)[0];
    } else if (m_hasWdl) {
        m_valueScalar = m_wdl[0] - m_wdl[2];
    } else {
        m_valueScalar = 0.0f;
    }

    m_hasData = false;
    for (const Block& b : m_blocks) m_hasData |= b.have;
    m_hasData |= !m_inputPlanes.empty() || !m_finalTokens.empty();

    if (m_blocks.empty()) {
        m_selectedBlock = 0;
    } else {
        m_selectedBlock =
            std::clamp(m_selectedBlock, 0, static_cast<int>(m_blocks.size()) - 1);
    }
}

// ===========================================================================
// Internal drawing namespace: each crtk paint* method as a free function that
// receives the cached tensors through a small read-only context plus the
// mutable selection cursors (mirrors NnueView's Ctx pattern).
// ===========================================================================

namespace {

struct BlockView {
    const std::vector<float>* attention;
    const std::vector<float>* ffn;
    int heads;
    float attentionFocus;
    float ffnRms;
    bool have;
};

struct Ctx {
    const Theme& theme;
    Color signature;
    const cnnv::chess::Position* position;
    const PieceSprites* sprites;
    const std::vector<BlockView>& blocks;
    const std::vector<float>& tokenEnergy;
    const std::vector<float>& boardSalience;
    const std::vector<float>& inputPlanes;
    const std::vector<float>& tokenFeatures;
    const std::vector<float>& embedding;
    const std::vector<float>& finalTokens;
    const std::vector<float>& policy;
    const std::array<float, 3>& wdl;
    float valueScalar;
    bool hasWdl;
    bool hasPolicy;
    bool hasTokenEnergy;
    int& selectedBlock;
    int& selectedHead;
    int& selectedSquare;
};

const std::vector<float>& blockAttention(const Ctx& c, int block) {
    static const std::vector<float> empty;
    if (block < 0 || block >= static_cast<int>(c.blocks.size())) return empty;
    return *c.blocks[static_cast<std::size_t>(block)].attention;
}

int blockHeads(const Ctx& c, int block) {
    if (block < 0 || block >= static_cast<int>(c.blocks.size())) return 0;
    return c.blocks[static_cast<std::size_t>(block)].heads;
}

// Largest mean-received focus over every block/head (crtk topAttentionFocus).
float topAttentionFocus(const Ctx& c) {
    float top = 0.0f;
    for (int b = 0; b < static_cast<int>(c.blocks.size()); ++b) {
        const std::vector<float>& attn = blockAttention(c, b);
        const int heads = blockHeads(c, b);
        for (int h = 0; h < heads; ++h) {
            top = std::max(top, headFocusScore(headReceivedEnergy(attn, h)));
        }
    }
    return top;
}

int strongestAttentionBlock(const Ctx& c) {
    int best = 0;
    float bestValue = -1e30f;
    for (int b = 0; b < static_cast<int>(c.blocks.size()); ++b) {
        const float v = c.blocks[static_cast<std::size_t>(b)].attentionFocus;
        if (v > bestValue) { bestValue = v; best = b; }
    }
    return best;
}

int strongestFfnBlock(const Ctx& c) {
    int best = 0;
    float bestValue = -1e30f;
    for (int b = 0; b < static_cast<int>(c.blocks.size()); ++b) {
        const float v = c.blocks[static_cast<std::size_t>(b)].ffnRms;
        if (v > bestValue) { bestValue = v; best = b; }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// 5-segment switcher (self-drawn — tensorviz::drawModeSelector only has 3).
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
// The 64x64 attention matrix + the on-board overlays are shared by Trace and
// Atlas, so factor them out once.
// ===========================================================================

// Draws the selected head's 64x64 attention matrix with the from/to crosshair
// (crtk paintAttentionMatrix). rows = from-token (query), cols = to-token (key).
void paintAttentionMatrix(Rectangle r, const Ctx& c) {
    const int headerH = 38;
    char sub[64];
    std::snprintf(sub, sizeof(sub), "64x64 matrix - rows = from - cols = to");
    char title[48];
    std::snprintf(title, sizeof(title), "head %d - full attention", c.selectedHead + 1);
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, static_cast<float>(headerH)},
                                 title, sub, c.theme, c.signature);
    const std::vector<float>& attn = blockAttention(c, c.selectedBlock);
    const int matrixTop = static_cast<int>(r.y) + headerH + 4;
    const float size = std::min(r.width - 32.0f, r.height - static_cast<float>(headerH) - 16.0f);
    const Rectangle map{r.x + 24.0f, static_cast<float>(matrixTop), std::max(8.0f, size),
                        std::max(8.0f, size)};
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;
    const std::size_t off = static_cast<std::size_t>(c.selectedHead) * plane;
    if (attn.empty() || off + plane > attn.size()) {
        drawHeatmap(map, {}, 1, 1, 1.0f, false, false, c.theme);
        return;
    }
    std::vector<float> slice(attn.begin() + static_cast<std::ptrdiff_t>(off),
                             attn.begin() + static_cast<std::ptrdiff_t>(off + plane));
    drawHeatmap(map, slice, kTokens, kTokens, std::max(1e-4f, maxAbs(slice)), false, true,
                c.theme);
    if (c.selectedSquare >= 0) {
        const Color cross = tensorviz::withAlpha(c.theme.accentBlue, 90);
        const float rowY = map.y + static_cast<float>(c.selectedSquare) * (map.height / 64.0f);
        const float cellH = std::max(1.0f, map.height / 64.0f);
        DrawRectangleRec(Rectangle{map.x, rowY, map.width, cellH}, cross);
        const float colX = map.x + static_cast<float>(c.selectedSquare) * (map.width / 64.0f);
        const float cellW = std::max(1.0f, map.width / 64.0f);
        DrawRectangleRec(Rectangle{colX, map.y, cellW, map.height}, cross);
        drawText(c.theme, squareName(c.selectedSquare), static_cast<int>(map.x - 22.0f),
                 static_cast<int>(rowY + cellH - 10.0f), 10, c.theme.textMuted);
        drawText(c.theme, squareName(c.selectedSquare), static_cast<int>(colX),
                 static_cast<int>(map.y - 14.0f), 10, c.theme.textMuted);
    }
}

// ===========================================================================
// Mode 0 — Overview (ABSTRACT): left pipeline (flow cards + per-block attn/ffn
// strip + WDL card), right token-energy board + top-move bars.
// Transpiled from Bt4View.paintAbstract / paintAbstractPipeline /
// paintBlockStrip / paintTokenEnergyBoard / paintTopPolicy.
// ===========================================================================

void paintBlockStrip(Rectangle r, const Ctx& c) {
    const int gap = 3;
    const int blocks = static_cast<int>(c.blocks.size());
    const int cellW = std::max(8, (static_cast<int>(r.width) - gap * (blocks - 1)) /
                                       std::max(1, blocks));
    const int half = static_cast<int>(r.height) / 2;
    DrawRectangleRec(r, tensorviz::withAlpha(c.theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, c.theme.panelBorder);
    float maxAttn = 1e-6f;
    float maxFfn = 1e-6f;
    for (const BlockView& b : c.blocks) {
        maxAttn = std::max(maxAttn, b.attentionFocus);
        maxFfn = std::max(maxFfn, b.ffnRms);
    }
    const int bestAttnBlock = strongestAttentionBlock(c);
    const int bestFfnBlock = strongestFfnBlock(c);
    for (int b = 0; b < blocks; ++b) {
        const BlockView& blk = c.blocks[static_cast<std::size_t>(b)];
        const int x = static_cast<int>(r.x) + b * (cellW + gap);
        const int hAttn = static_cast<int>(std::lround(blk.attentionFocus / maxAttn *
                                                       static_cast<float>(half - 2)));
        const int hFfn = static_cast<int>(std::lround(blk.ffnRms / maxFfn *
                                                      static_cast<float>(half - 2)));
        DrawRectangleRec(Rectangle{static_cast<float>(x),
                                   r.y + static_cast<float>(half - hAttn),
                                   static_cast<float>(cellW), static_cast<float>(hAttn)},
                         c.theme.accentBlue);  // crtk POLICY.
        DrawRectangleRec(Rectangle{static_cast<float>(x), r.y + static_cast<float>(half + 1),
                                   static_cast<float>(cellW), static_cast<float>(hFfn)},
                         c.theme.accentMagenta);  // crtk VALUE.
        if (b == bestAttnBlock || b == bestFfnBlock) {
            DrawRectangleLinesEx(
                Rectangle{static_cast<float>(x), r.y + 1.0f,
                          static_cast<float>(std::max(1, cellW - 1)), r.height - 3.0f},
                1.0f, b == bestAttnBlock ? c.theme.accentBlue : c.theme.accentMagenta);
        }
        char label[32];
        std::snprintf(label, sizeof(label), "%d", b + 1);
        const int lw = measureText(c.theme, label, 9);
        drawText(c.theme, label, x + std::max(0, (cellW - lw) / 2),
                 static_cast<int>(r.y + r.height) - 11, 9, c.theme.textMuted);
        if (clickedIn(Rectangle{static_cast<float>(x), r.y,
                                static_cast<float>(cellW + gap), r.height})) {
            c.selectedBlock = b;
        }
    }
}

// crtk drawWdlCard.
void paintWdlCard(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 40.0f},
                                 "value head - win / draw / loss",
                                 "predicted game outcome for the side to move", c.theme,
                                 c.signature);
    const float barTop = r.y + 50.0f;
    const float barH = std::min(34.0f, std::max(20.0f, r.height - 80.0f));
    const Rectangle bar{r.x + 8.0f, barTop, std::max(1.0f, r.width - 16.0f), barH};
    if (c.hasWdl) {
        float w = std::max(0.0f, c.wdl[0]);
        float d = std::max(0.0f, c.wdl[1]);
        float l = std::max(0.0f, c.wdl[2]);
        const float sum = std::max(1e-4f, w + d + l);
        w /= sum; d /= sum; l /= sum;
        const float ww = std::round(bar.width * w);
        const float dw = std::round(bar.width * d);
        const float lw = std::max(0.0f, bar.width - ww - dw);
        const Color winFill = tensorviz::blend(c.theme.panelBackground, c.theme.overlayHot, 0.85f);
        const Color drawFill = tensorviz::blend(c.theme.panelBackground, c.theme.textMuted, 0.55f);
        const Color lossFill = tensorviz::blend(c.theme.panelBackground, c.theme.accentBlue, 0.85f);
        DrawRectangleRec(Rectangle{bar.x, bar.y, ww, bar.height}, winFill);
        DrawRectangleRec(Rectangle{bar.x + ww, bar.y, dw, bar.height}, drawFill);
        DrawRectangleRec(Rectangle{bar.x + ww + dw, bar.y, lw, bar.height}, lossFill);
        DrawRectangleLinesEx(bar, 1.0f, c.theme.panelBorder);
        auto segLabel = [&](const char* prefix, int pct, float x, float segW) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), "%s %d%%", prefix, pct);
            if (segW > static_cast<float>(measureText(c.theme, buf, 11)) + 6.0f) {
                drawCenteredText(c.theme, buf, Rectangle{x, bar.y, segW, bar.height}, 11,
                             c.theme.textPrimary);
            }
        };
        segLabel("W", static_cast<int>(std::lround(w * 100.0f)), bar.x, ww);
        segLabel("D", static_cast<int>(std::lround(d * 100.0f)), bar.x + ww, dw);
        segLabel("L", static_cast<int>(std::lround(l * 100.0f)), bar.x + ww + dw, lw);
    } else {
        drawText(c.theme, "No W/D/L output in this snapshot.", static_cast<int>(bar.x),
                 static_cast<int>(bar.y + 16.0f), 11, c.theme.textMuted);
    }
    char scalar[64];
    std::snprintf(scalar, sizeof(scalar),
                  "value scalar %+.3f  (+1 = winning, -1 = losing)", c.valueScalar);
    drawText(c.theme, scalar, static_cast<int>(bar.x),
             static_cast<int>(bar.y + bar.height + 22.0f), 11, c.theme.textMuted);
}

void paintAbstractPipeline(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 40.0f}, "abstract flow",
        "input -> tokens -> transformer blocks -> heads", c.theme,
        c.signature);
    const float top = r.y + 56.0f;
    const float gap = 8.0f;
    const float cardW = (r.width - 32.0f) / 3.0f;
    const float cardH = 70.0f;
    const Rectangle inR{r.x + 8.0f, top, cardW, cardH};
    const Rectangle emR{inR.x + cardW + gap, top, cardW, cardH};
    const Rectangle outR{emR.x + cardW + gap, top, cardW, cardH};
    tensorviz::drawElbowConnection(inR, emR, c.signature, true, false, c.theme);
    tensorviz::drawElbowConnection(emR, outR, c.signature, true, false, c.theme);
    char embedDetail[32];
    std::snprintf(embedDetail, sizeof(embedDetail), "64 x %zu",
                  c.embedding.empty() ? std::size_t{1024}
                                      : c.embedding.size() / static_cast<std::size_t>(kTokens));
    tensorviz::drawAbstractBlock(inR, "input", "112x8x8", 1.0f, c.theme.accentGreen,
                                 false, c.theme);
    tensorviz::drawAbstractBlock(emR, "tokens", embedDetail, 0.7f,
                                 c.signature, false, c.theme);
    tensorviz::drawAbstractBlock(outR, "final tokens", embedDetail, 0.6f,
                                 c.theme.accentYellow, false, c.theme);

    const float stripY = top + cardH + 28.0f;
    const Rectangle strip{r.x + 8.0f, stripY, r.width - 16.0f, 60.0f};
    char strHint[96];
    std::snprintf(strHint, sizeof(strHint),
                  "important blocks: attention B%d - ffn B%d  (small bars are low-signal)",
                  strongestAttentionBlock(c) + 1, strongestFfnBlock(c) + 1);
    drawText(c.theme, strHint, static_cast<int>(strip.x),
             static_cast<int>(strip.y - 14.0f), 10, c.theme.textMuted);
    paintBlockStrip(strip, c);

    const float valueY = stripY + strip.height + 24.0f;
    if (valueY + 96.0f <= r.y + r.height) {
        const Rectangle valueR{r.x + 8.0f, valueY, r.width - 16.0f,
                               r.y + r.height - valueY - 4.0f};
        paintWdlCard(valueR, c);
    }
}

// crtk paintTopPolicy: without a FEN we cannot decode legal moves, so show the
// raw top-K policy logits as a softmax-probability bar chart instead. See
// report notes (Bt4View.java:901 decodeTopMoves needs a Position).
void paintTopPolicy(Rectangle r, const Ctx& c) {
    if (!c.hasPolicy) return;
    // Top-6 logits -> softmax over that subset for a readable probability.
    struct Move { int index; float logit; };
    std::vector<Move> all;
    all.reserve(c.policy.size());
    for (int i = 0; i < static_cast<int>(c.policy.size()); ++i) {
        all.push_back(Move{i, c.policy[static_cast<std::size_t>(i)]});
    }
    const int k = std::min<int>(6, static_cast<int>(all.size()));
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [](const Move& a, const Move& b) { return a.logit > b.logit; });
    all.resize(static_cast<std::size_t>(k));
    double denom = 0.0;
    for (const Move& m : all) denom += std::exp(static_cast<double>(m.logit - all.front().logit));
    std::vector<std::string> labels;
    std::vector<float> values;
    float scale = 0.0f;
    for (const Move& m : all) {
        const float p = static_cast<float>(
            std::exp(static_cast<double>(m.logit - all.front().logit)) / std::max(1e-9, denom));
        char buf[16];
        std::snprintf(buf, sizeof(buf), "#%d", m.index);
        labels.emplace_back(buf);
        values.push_back(p);
        scale = std::max(scale, p);
    }
    drawText(c.theme, "top policy logits",
             static_cast<int>(r.x), static_cast<int>(r.y - 14.0f), 10, c.theme.textMuted);
    tensorviz::drawMetricBars(r, labels, values, std::max(scale, 1e-4f), c.theme);
}

void paintTokenEnergyBoard(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 40.0f}, "token energy + top moves",
        "attention tint; top moves",
        c.theme, c.signature);
    const float boardArea =
        std::min(r.width, std::max(160.0f, (r.height - 50.0f) * 3.0f / 5.0f));
    const Rectangle board{r.x + std::max(8.0f, (r.width - boardArea) / 2.0f), r.y + 50.0f,
                          boardArea - 16.0f, boardArea - 16.0f};
    drawMiniBoard(board, c.theme);
    if (c.hasTokenEnergy) {
        const std::vector<float> energy(c.tokenEnergy.begin(),
                                        c.tokenEnergy.begin() + kTokens);
        const float s = std::max(1e-4f, maxAbs(energy));
        for (int sq = 0; sq < 64; ++sq) {
            const float t = std::sqrt(std::clamp(energy[static_cast<std::size_t>(sq)] / s,
                                                 0.0f, 1.0f));
            DrawRectangleRec(lerfSquare(board, sq),
                             tensorviz::withAlpha(c.theme.overlayHot,
                                                  static_cast<unsigned char>(20.0f + 200.0f * t)));
        }
        const int focus = strongestSquare(energy);
        if (focus >= 0) {
            DrawRectangleLinesEx(lerfSquare(board, focus), 2.0f, c.theme.accentYellow);
        }
        DrawRectangleLinesEx(board, 1.0f, c.theme.panelBorder);
    }
    const float barsTop = board.y + board.height + 30.0f;
    if (barsTop + 40.0f <= r.y + r.height) {
        const Rectangle barsR{r.x + 8.0f, barsTop, r.width - 16.0f,
                              r.y + r.height - barsTop - 6.0f};
        paintTopPolicy(barsR, c);
    }
}

void paintOverview(Rectangle body, const Ctx& c) {
    const float leftW = body.width * 0.58f;
    const Rectangle left{body.x, body.y, leftW, body.height};
    const Rectangle right{body.x + leftW + 10.0f, body.y, body.width - leftW - 10.0f,
                          body.height};
    paintAbstractPipeline(left, c);
    paintTokenEnergyBoard(right, c);
}

// ===========================================================================
// Mode 1 — Trace (DETAILED): block selector strip + 4x8 head grid of 8x8
// received-energy thumbnails + the selected head's 64x64 matrix + the on-board
// two-triangle from/to overlay + a head readout.
// Transpiled from Bt4View.paintDetailed / paintBlockSelector / paintHeadGrid /
// paintBoardOverlay / drawTriangleOverlay / paintHeadReadout.
// ===========================================================================

void paintBlockSelector(Rectangle r, const Ctx& c) {
    DrawRectangleRec(r, tensorviz::withAlpha(c.theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, c.theme.panelBorder);
    const int blocks = static_cast<int>(c.blocks.size());
    const int cellW = static_cast<int>(r.width) / std::max(1, blocks);
    for (int b = 0; b < blocks; ++b) {
        const int x = static_cast<int>(r.x) + b * cellW;
        const bool sel = b == c.selectedBlock;
        const Rectangle cell{static_cast<float>(x + 1), r.y + 1.0f,
                             static_cast<float>(cellW - 2), r.height - 2.0f};
        DrawRectangleRec(cell, sel ? tensorviz::blend(c.theme.buttonActive, c.signature, 0.55f)
                                   : c.theme.buttonIdle);
        char label[32];
        std::snprintf(label, sizeof(label), "%d", b + 1);
        const int lw = measureText(c.theme, label, 11);
        drawText(c.theme, label, x + std::max(0, (cellW - lw) / 2),
                 static_cast<int>(r.y + r.height) - 15, 11,
                 sel ? c.theme.textPrimary : c.theme.textMuted);
        if (clickedIn(Rectangle{static_cast<float>(x), r.y, static_cast<float>(cellW),
                                r.height})) {
            c.selectedBlock = b;
        }
    }
}

void paintHeadGrid(Rectangle r, const Ctx& c) {
    const int headerH = 38;
    char headerTitle[40];
    std::snprintf(headerTitle, sizeof(headerTitle), "heads of block %d", c.selectedBlock + 1);
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, static_cast<float>(headerH)},
                                 headerTitle, "8x8 attention-received per head - click one",
                                 c.theme, c.signature);
    const Rectangle grid{r.x, r.y + static_cast<float>(headerH) + 4.0f, r.width,
                         r.height - static_cast<float>(headerH) - 4.0f};
    const int heads = blockHeads(c, c.selectedBlock);
    const std::vector<float>& attn = blockAttention(c, c.selectedBlock);
    if (heads <= 0 || attn.empty()) {
        drawText(c.theme, "No attention for this block.", static_cast<int>(grid.x + 4.0f),
                 static_cast<int>(grid.y + 20.0f), 11, c.theme.textMuted);
        return;
    }
    const int cols = 4;
    const int rows = (heads + cols - 1) / cols;
    const float cellW = grid.width / static_cast<float>(cols);
    const float cellH = grid.height / static_cast<float>(rows);
    // Per-head received energy + a shared scale, so heads are comparable.
    std::vector<std::vector<float>> energy(static_cast<std::size_t>(heads));
    float globalMax = 1e-6f;
    for (int h = 0; h < heads; ++h) {
        energy[static_cast<std::size_t>(h)] = headReceivedEnergy(attn, h);
        globalMax = std::max(globalMax, headFocusScore(energy[static_cast<std::size_t>(h)]));
    }
    for (int h = 0; h < heads; ++h) {
        const int col = h % cols;
        const int row = h / cols;
        const float x = grid.x + static_cast<float>(col) * cellW;
        const float y = grid.y + static_cast<float>(row) * cellH;
        const float padTop = 14.0f;
        const Rectangle heat{x + 3.0f, y + padTop, std::max(4.0f, cellW - 6.0f),
                             std::max(4.0f, cellH - padTop - 4.0f)};
        drawHeatmap(heat, energy[static_cast<std::size_t>(h)], 8, 8, globalMax, false, false,
                    c.theme, false);
        // Selected-square marker inside each thumbnail.
        if (c.selectedSquare >= 0) {
            const int file = c.selectedSquare & 7;
            const int drawRank = 7 - (c.selectedSquare >> 3);
            const float mw = heat.width / 8.0f;
            const float mh = heat.height / 8.0f;
            DrawRectangleLinesEx(Rectangle{heat.x + static_cast<float>(file) * mw,
                                           heat.y + static_cast<float>(drawRank) * mh, mw, mh},
                                 1.0f, c.theme.accentYellow);
        }
        const Rectangle cell{x, y, cellW, cellH};
        if (h == c.selectedHead) {
            DrawRectangleLinesEx(cell, 2.0f, c.signature);
        } else {
            DrawRectangleLinesEx(cell, 1.0f, c.theme.panelBorder);
        }
        char hl[32];
        std::snprintf(hl, sizeof(hl), "h%d", h + 1);
        drawText(c.theme, hl, static_cast<int>(x + 5.0f), static_cast<int>(y + 2.0f), 11,
                 c.theme.textPrimary);
        if (clickedIn(cell)) c.selectedHead = h;
    }
}

// Two-triangle on-board overlay (crtk drawTriangleOverlay): upper-right =
// outgoing (selected -> this), lower-left = incoming (this -> selected). Both
// share one scale. outgoing[k]=attn[sel*64+k], incoming[k]=attn[k*64+sel].
void paintBoardOverlay(Rectangle r, const Ctx& c) {
    const int headerH = 38;
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, static_cast<float>(headerH)}, "attention on the board",
        "lower-left = incoming (this -> sel) - upper-right = outgoing (sel -> this)", c.theme,
        c.signature);
    const float top = r.y + static_cast<float>(headerH) + 6.0f;
    const float bottomReserve = 22.0f;
    const float availW = r.width - 16.0f;
    const float availH = r.height - static_cast<float>(headerH) - 6.0f - bottomReserve;
    const float size = std::max(40.0f, std::min(availW, availH));
    const Rectangle board{r.x + (r.width - size) / 2.0f, top, size, size};
    drawMiniBoard(board, c.theme);

    const std::vector<float>& attn = blockAttention(c, c.selectedBlock);
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;
    const std::size_t off = static_cast<std::size_t>(c.selectedHead) * plane;
    if (c.selectedSquare >= 0 && !attn.empty() && off + plane <= attn.size()) {
        const int sel = c.selectedSquare;  // identity token mapping (no FEN).
        std::vector<float> outgoing(64, 0.0f);  // selected -> this.
        std::vector<float> incoming(64, 0.0f);  // this -> selected.
        for (int token = 0; token < 64; ++token) {
            outgoing[static_cast<std::size_t>(token)] =
                attn[off + static_cast<std::size_t>(sel) * kTokens + token];
            incoming[static_cast<std::size_t>(token)] =
                attn[off + static_cast<std::size_t>(token) * kTokens + sel];
        }
        const float scale = std::max({maxAbs(outgoing), maxAbs(incoming), 1e-6f});
        const Color diag = tensorviz::withAlpha(c.theme.textPrimary, 58);
        for (int sq = 0; sq < 64; ++sq) {
            const Rectangle cell = lerfSquare(board, sq);
            const float xd = cell.x;
            const float yd = cell.y;
            const float xr = cell.x + cell.width;
            const float yb = cell.y + cell.height;
            const float vTo = std::clamp(outgoing[static_cast<std::size_t>(sq)] / scale, 0.0f, 1.0f);
            const float vFrom = std::clamp(incoming[static_cast<std::size_t>(sq)] / scale, 0.0f, 1.0f);
            // Upper-right triangle = outgoing (orange-hot ~ crtk POSITIVE).
            DrawTriangle(Vector2{xd, yd}, Vector2{xr, yb}, Vector2{xr, yd},
                         tensorviz::withAlpha(c.theme.overlayHot,
                                              static_cast<unsigned char>(20.0f + 200.0f * vTo)));
            // Lower-left triangle = incoming (blue ~ crtk NEGATIVE).
            DrawTriangle(Vector2{xd, yd}, Vector2{xd, yb}, Vector2{xr, yb},
                         tensorviz::withAlpha(c.theme.accentBlue,
                                              static_cast<unsigned char>(20.0f + 200.0f * vFrom)));
            DrawLineV(Vector2{xd, yd}, Vector2{xr, yb}, diag);
        }
    }
    drawBoardPieces(board, c.position, c.sprites);
    if (c.selectedSquare >= 0) {
        DrawRectangleLinesEx(lerfSquare(board, c.selectedSquare), 2.0f,
                             c.theme.accentYellow);
    }
    DrawRectangleLinesEx(board, 1.0f, c.theme.panelBorder);

    // Click handling: select / deselect a board square.
    if (clickedIn(board)) {
        const float cw = board.width / 8.0f;
        const float ch = board.height / 8.0f;
        const Vector2 m = tensorviz::panelMousePosition();
        const int file = std::clamp(static_cast<int>((m.x - board.x) / cw), 0, 7);
        const int drawRank = std::clamp(static_cast<int>((m.y - board.y) / ch), 0, 7);
        const int rank = 7 - drawRank;
        const int sq = rank * 8 + file;
        c.selectedSquare = (sq == c.selectedSquare) ? -1 : sq;
    }

    char hint[64];
    if (c.selectedSquare < 0) {
        std::snprintf(hint, sizeof(hint), "no square selected - click a board square");
    } else {
        std::snprintf(hint, sizeof(hint), "selected: %s - click again to clear",
                      squareName(c.selectedSquare));
    }
    drawText(c.theme, hint, static_cast<int>(r.x + 8.0f),
             static_cast<int>(r.y + r.height - 16.0f), 10, c.theme.textMuted);
}

void paintHeadReadout(Rectangle r, const Ctx& c) {
    const std::vector<float>& attn = blockAttention(c, c.selectedBlock);
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;
    const std::size_t off = static_cast<std::size_t>(c.selectedHead) * plane;
    float headMag = 0.0f;
    float maxEntry = 0.0f;
    int argFrom = 0;
    int argTo = 0;
    if (!attn.empty() && off + plane <= attn.size()) {
        double sum = 0.0;
        for (std::size_t i = 0; i < plane; ++i) {
            const float v = attn[off + i];
            sum += v;
            if (v > maxEntry) {
                maxEntry = v;
                argFrom = static_cast<int>(i / kTokens);
                argTo = static_cast<int>(i % kTokens);
            }
        }
        headMag = static_cast<float>(sum / static_cast<double>(plane));
    }
    DrawRectangleRec(r, tensorviz::withAlpha(c.theme.panelBackground, 240));
    DrawRectangleLinesEx(r, 1.0f, c.theme.panelBorder);
    char line1[96];
    std::snprintf(line1, sizeof(line1), "block %d   |   head %d/%d   |   mean attention %.4f",
                  c.selectedBlock + 1, c.selectedHead + 1, blockHeads(c, c.selectedBlock),
                  headMag);
    drawText(c.theme, line1, static_cast<int>(r.x + 8.0f), static_cast<int>(r.y + 6.0f), 11,
             c.theme.textPrimary);
    const float selectedFocus =
        headFocusScore(headReceivedEnergy(attn, c.selectedHead));
    const float topFocus = std::max(1e-6f, topAttentionFocus(c));
    char line2[160];
    std::snprintf(line2, sizeof(line2),
                  "importance: %s (%.0f%% of top head) - hottest pair: %s -> %s (%.3f)",
                  attentionImportance(selectedFocus / topFocus),
                  std::min(100.0f, 100.0f * selectedFocus / topFocus), squareName(argFrom),
                  squareName(argTo), maxEntry);
    drawText(c.theme, tensorviz::fitText(c.theme, line2, 11, false, r.width - 16.0f).c_str(),
             static_cast<int>(r.x + 8.0f), static_cast<int>(r.y + 22.0f), 11, c.theme.textMuted);
}

void paintTrace(Rectangle body, const Ctx& c) {
    char subtitle[96];
    std::snprintf(subtitle, sizeof(subtitle),
                  "block 1..%d - head 1..N - click a head, then click any board square",
                  std::max(1, static_cast<int>(c.blocks.size())));
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, 40.0f}, "per-head attention",
        subtitle, c.theme, c.signature);
    const float top = body.y + 50.0f;
    const Rectangle blockBar{body.x, top, body.width, 22.0f};
    paintBlockSelector(blockBar, c);

    const float columnTop = blockBar.y + blockBar.height + 12.0f;
    const float columnH = body.height - (columnTop - body.y) - 8.0f;
    const float gridW = std::min(560.0f, std::max(360.0f, body.width / 3.0f));
    const float matrixW = std::min(420.0f, std::max(320.0f, (body.width - gridW - 24.0f) / 3.0f));
    const Rectangle gridR{body.x, columnTop, gridW, std::min(columnH, gridW * 2.0f)};
    const Rectangle matrixR{body.x + gridW + 12.0f, columnTop, matrixW,
                            std::min(matrixW + 32.0f, columnH)};
    const Rectangle boardR{matrixR.x + matrixW + 12.0f, columnTop,
                           body.width - (matrixR.x + matrixW + 12.0f - body.x), columnH};

    // Clamp the head selection to the current block's head count.
    const int heads = blockHeads(c, c.selectedBlock);
    if (heads > 0) c.selectedHead = std::clamp(c.selectedHead, 0, heads - 1);

    paintHeadGrid(gridR, c);
    paintAttentionMatrix(matrixR, c);
    paintBoardOverlay(boardR, c);

    const Rectangle bottom{body.x, gridR.y + gridR.height + 12.0f, gridW + matrixW + 12.0f,
                           body.y + body.height - (gridR.y + gridR.height + 12.0f) - 4.0f};
    if (bottom.height > 24.0f) paintHeadReadout(bottom, c);
}

// ===========================================================================
// Mode 2 — All (RAW): the dense block x head grid; each block row split into a
// top 64x64 attention-matrix sub-cell and a bottom 8x8 mean-received sub-cell,
// both via sqrt-gamma heatmaps with per-block colour normalisation.
// Transpiled from Bt4View.paintRaw / paintBt4AttentionGrid.
// ===========================================================================

void paintRaw(Rectangle body, const Ctx& c) {
    const int headerH = 38;
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, static_cast<float>(headerH)},
        "raw attention atlas - every block x head at once",
        "top grid: 64x64 attention matrices - bottom grid: 8x8 mean attention-received",
        c.theme, c.signature);

    // Heads can vary per block; use the max so the grid has uniform columns.
    int maxHeads = 0;
    for (const BlockView& b : c.blocks) maxHeads = std::max(maxHeads, b.heads);
    if (maxHeads <= 0) {
        drawText(c.theme, "No attention tensors in this snapshot.",
                 static_cast<int>(body.x + 12.0f), static_cast<int>(body.y + headerH + 30.0f),
                 12, c.theme.textMuted);
        return;
    }
    const int blocks = static_cast<int>(c.blocks.size());
    const int rowLabelW = 28;
    const int colLabelH = 16;
    const int gridTop = static_cast<int>(body.y) + headerH + 4 + colLabelH;
    const int gridLeft = static_cast<int>(body.x) + rowLabelW;
    const int gridW = static_cast<int>(body.width) - rowLabelW - 4;
    const int gridH = static_cast<int>(body.height) - headerH - 4 - colLabelH - 4;
    const int cellW = std::max(2, gridW / maxHeads);
    const int cellH = std::max(4, gridH / std::max(1, blocks));
    const int subPad = 1;
    const int matrixH = static_cast<int>(std::lround(cellH * 0.62));
    const int boardH = cellH - matrixH - 2;
    const std::size_t plane = static_cast<std::size_t>(kTokens) * kTokens;

    // Column (head) labels — every other to avoid clutter.
    for (int h = 0; h < maxHeads; h += 2) {
        char lbl[32];
        std::snprintf(lbl, sizeof(lbl), "h%d", h + 1);
        const int lw = measureText(c.theme, lbl, 9);
        drawText(c.theme, lbl, gridLeft + h * cellW + std::max(0, (cellW - lw) / 2),
                 static_cast<int>(body.y) + headerH + 4 + colLabelH - 12, 9, c.theme.textMuted);
    }

    for (int b = 0; b < blocks; ++b) {
        const BlockView& blk = c.blocks[static_cast<std::size_t>(b)];
        const std::vector<float>& attn = *blk.attention;
        const int y = gridTop + b * cellH;
        char rowLabel[32];
        std::snprintf(rowLabel, sizeof(rowLabel), "B%d", b + 1);
        drawText(c.theme, rowLabel, static_cast<int>(body.x) + 4, y + cellH / 2 - 5, 10,
                 c.theme.textMuted);
        // Per-block colour normalisation (separate for matrix vs energy).
        float matrixMax = 1e-6f;
        float energyMax = 1e-6f;
        std::vector<std::vector<float>> energy(static_cast<std::size_t>(blk.heads));
        for (int h = 0; h < blk.heads; ++h) {
            const std::size_t off = static_cast<std::size_t>(h) * plane;
            if (off + plane > attn.size()) continue;
            for (std::size_t i = 0; i < plane; ++i) matrixMax = std::max(matrixMax, attn[off + i]);
            energy[static_cast<std::size_t>(h)] = headReceivedEnergy(attn, h);
            energyMax = std::max(energyMax, headFocusScore(energy[static_cast<std::size_t>(h)]));
        }
        for (int h = 0; h < maxHeads; ++h) {
            const int x = gridLeft + h * cellW;
            const Rectangle matrixCell{static_cast<float>(x + subPad),
                                       static_cast<float>(y + subPad),
                                       static_cast<float>(std::max(1, cellW - 2 * subPad)),
                                       static_cast<float>(std::max(1, matrixH - 2 * subPad))};
            const Rectangle boardCell{static_cast<float>(x + subPad),
                                      static_cast<float>(y + matrixH + 1),
                                      static_cast<float>(std::max(1, cellW - 2 * subPad)),
                                      static_cast<float>(std::max(1, boardH - 2))};
            const std::size_t off = static_cast<std::size_t>(h) * plane;
            if (h < blk.heads && off + plane <= attn.size()) {
                std::vector<float> slice(attn.begin() + static_cast<std::ptrdiff_t>(off),
                                         attn.begin() + static_cast<std::ptrdiff_t>(off + plane));
                drawHeatmap(matrixCell, slice, kTokens, kTokens, matrixMax, false, true, c.theme,
                            false);
                drawHeatmap(boardCell, energy[static_cast<std::size_t>(h)], 8, 8, energyMax,
                            false, true, c.theme, false);
            } else {
                drawHeatmap(matrixCell, {}, 1, 1, 1.0f, false, false, c.theme, false);
                drawHeatmap(boardCell, {}, 1, 1, 1.0f, false, false, c.theme, false);
            }
            const bool selected = b == c.selectedBlock && h == c.selectedHead;
            const Rectangle cell{static_cast<float>(x), static_cast<float>(y),
                                 static_cast<float>(cellW), static_cast<float>(cellH)};
            DrawRectangleLinesEx(cell, selected ? 2.0f : 1.0f,
                                 selected ? c.signature : c.theme.panelBorder);
            if (selected) DrawRectangleLinesEx(cell, 1.0f, c.signature);
            if (h < blk.heads && clickedIn(cell)) {
                c.selectedBlock = b;
                c.selectedHead = h;
            }
        }
    }
}

// ===========================================================================
// Mode 3 — Atlas (ATLAS): block x head focus fingerprint + selected-head /
// token-energy board footprints + the selected head's 64x64 matrix.
// Transpiled from Bt4View.paintAtlas / paintBt4HeadFingerprint /
// paintBt4AtlasBoards / drawBt4AtlasBoard.
// ===========================================================================

void drawAtlasBoard(Rectangle board, const char* title, const std::vector<float>& values,
                    const Ctx& c) {
    const int focus = strongestSquare(values);
    char head[64];
    std::snprintf(head, sizeof(head), "%s%s%s", title, focus >= 0 ? " - focus " : "",
                  focus >= 0 ? squareName(focus) : "");
    drawText(c.theme, tensorviz::fitText(c.theme, head, 10, false, board.width + 10.0f).c_str(),
             static_cast<int>(board.x), static_cast<int>(board.y - 14.0f), 10, c.theme.textMuted);
    drawMiniBoard(board, c.theme);
    if (values.size() >= 64) {
        const float s = std::max(1e-4f, maxAbs(values));
        for (int sq = 0; sq < 64; ++sq) {
            const float t = std::sqrt(std::clamp(values[static_cast<std::size_t>(sq)] / s,
                                                 0.0f, 1.0f));
            DrawRectangleRec(lerfSquare(board, sq),
                             tensorviz::withAlpha(c.theme.overlayHot,
                                                  static_cast<unsigned char>(20.0f + 200.0f * t)));
        }
        if (focus >= 0) {
            DrawRectangleLinesEx(lerfSquare(board, focus), 2.0f, c.theme.accentYellow);
        }
    }
    DrawRectangleLinesEx(board, 1.0f, c.theme.panelBorder);
}

void paintAtlasFingerprint(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{r.x, r.y, r.width, 38.0f}, "block x head fingerprint",
        "bright cells are focused heads; pale cells are diffuse or low-impact", c.theme,
        c.signature);
    int maxHeads = 0;
    for (const BlockView& b : c.blocks) maxHeads = std::max(maxHeads, b.heads);
    if (maxHeads <= 0) {
        drawText(c.theme, "No attention tensors in this snapshot.",
                 static_cast<int>(r.x + 4.0f), static_cast<int>(r.y + 52.0f), 11,
                 c.theme.textMuted);
        return;
    }
    const int blocks = static_cast<int>(c.blocks.size());
    const int rowLabelW = 38;
    const int colLabelH = 18;
    const float gridX = r.x + static_cast<float>(rowLabelW);
    const float gridY = r.y + 38.0f + static_cast<float>(colLabelH) + 4.0f;
    const float gridW = r.width - static_cast<float>(rowLabelW) - 4.0f;
    const float gridH = r.y + r.height - gridY - 4.0f;
    const float cellW = std::max(4.0f, gridW / static_cast<float>(maxHeads));
    const float cellH = std::max(8.0f, gridH / static_cast<float>(blocks));

    // Per-cell focus score + global scale + a "% quiet" summary.
    std::vector<std::vector<float>> scores(static_cast<std::size_t>(blocks));
    float scale = 1e-6f;
    int topBlock = 0;
    int topHead = 0;
    for (int b = 0; b < blocks; ++b) {
        const BlockView& blk = c.blocks[static_cast<std::size_t>(b)];
        scores[static_cast<std::size_t>(b)].assign(static_cast<std::size_t>(maxHeads), 0.0f);
        for (int h = 0; h < blk.heads; ++h) {
            const float s = headFocusScore(headReceivedEnergy(*blk.attention, h));
            scores[static_cast<std::size_t>(b)][static_cast<std::size_t>(h)] = s;
            if (s > scale) { scale = s; topBlock = b; topHead = h; }
        }
    }
    const float quietCutoff = scale * 0.16f;
    const float importantCutoff = scale * 0.78f;
    int quietHeads = 0;
    int totalHeads = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int h = 0; h < c.blocks[static_cast<std::size_t>(b)].heads; ++h) {
            ++totalHeads;
            if (scores[static_cast<std::size_t>(b)][static_cast<std::size_t>(h)] <= quietCutoff) {
                ++quietHeads;
            }
        }
    }
    if (r.width >= 760.0f && totalHeads > 0) {
        char imp[48];
        std::snprintf(imp, sizeof(imp), "important B%d h%d", topBlock + 1, topHead + 1);
        drawText(c.theme, imp, static_cast<int>(r.x + r.width - 322.0f),
                 static_cast<int>(r.y + 11.0f), 11, c.theme.accentYellow);
        char quiet[48];
        std::snprintf(quiet, sizeof(quiet), "%d%% quiet",
                      static_cast<int>(std::lround(100.0f * static_cast<float>(quietHeads) /
                                                   static_cast<float>(totalHeads))));
        drawText(c.theme, quiet, static_cast<int>(r.x + r.width - 150.0f),
                 static_cast<int>(r.y + 11.0f), 11, c.theme.textMuted);
    }

    for (int h = 0; h < maxHeads; h += 4) {
        char lbl[32];
        std::snprintf(lbl, sizeof(lbl), "h%d", h + 1);
        drawText(c.theme, lbl, static_cast<int>(gridX + static_cast<float>(h) * cellW + 1.0f),
                 static_cast<int>(gridY - 14.0f), 9, c.theme.textMuted);
    }
    for (int b = 0; b < blocks; ++b) {
        const float y = gridY + static_cast<float>(b) * cellH;
        char rowLabel[32];
        std::snprintf(rowLabel, sizeof(rowLabel), "B%d", b + 1);
        drawText(c.theme, rowLabel, static_cast<int>(r.x + 5.0f),
                 static_cast<int>(y + std::max(10.0f, cellH / 2.0f - 4.0f)), 10,
                 c.theme.textMuted);
        for (int h = 0; h < c.blocks[static_cast<std::size_t>(b)].heads; ++h) {
            const float x = gridX + static_cast<float>(h) * cellW;
            const float v = std::sqrt(std::clamp(
                scores[static_cast<std::size_t>(b)][static_cast<std::size_t>(h)] / scale, 0.0f,
                1.0f));
            const Rectangle cell{x, y, std::max(1.0f, cellW - 1.0f), std::max(1.0f, cellH - 1.0f)};
            DrawRectangleRec(cell, tensorviz::blend(c.theme.panelBackground, c.theme.overlayHot,
                                                    0.12f + 0.82f * v));
            if (scores[static_cast<std::size_t>(b)][static_cast<std::size_t>(h)] >=
                    importantCutoff &&
                cellW >= 4.0f && cellH >= 8.0f) {
                DrawRectangleLinesEx(cell, 1.0f, tensorviz::withAlpha(c.theme.textPrimary, 120));
            }
            if (b == c.selectedBlock && h == c.selectedHead) {
                DrawRectangleLinesEx(Rectangle{x, y, cellW, cellH}, 2.0f, c.signature);
            }
            if (clickedIn(Rectangle{x, y, cellW, cellH})) {
                c.selectedBlock = b;
                c.selectedHead = h;
            }
        }
    }
    DrawRectangleLinesEx(Rectangle{gridX, gridY, static_cast<float>(maxHeads) * cellW,
                                   static_cast<float>(blocks) * cellH},
                         1.0f, c.theme.panelBorder);
}

void paintAtlasBoards(Rectangle r, const Ctx& c) {
    tensorviz::drawSectionHeader(Rectangle{r.x, r.y, r.width, 38.0f}, "board footprint",
                                 "selected head vs final token-energy summary", c.theme,
                                 c.signature);
    const Rectangle content{r.x + 4.0f, r.y + 48.0f, r.width - 8.0f, r.height - 52.0f};
    const std::vector<float> selectedEnergy =
        headReceivedEnergy(blockAttention(c, c.selectedBlock), c.selectedHead);
    const float gap = 10.0f;
    char selTitle[48];
    std::snprintf(selTitle, sizeof(selTitle), "B%d h%d", c.selectedBlock + 1,
                  c.selectedHead + 1);
    float boardSide = std::min((content.width - gap) / 2.0f,
                               std::max(64.0f, content.height - 22.0f));
    if (boardSide < 72.0f) {
        boardSide = std::min(content.width, std::max(48.0f, content.height - 22.0f));
        drawAtlasBoard(Rectangle{content.x, content.y + 16.0f, boardSide, boardSide}, selTitle,
                       selectedEnergy, c);
        return;
    }
    drawAtlasBoard(Rectangle{content.x, content.y + 16.0f, boardSide, boardSide}, selTitle,
                   selectedEnergy, c);
    std::vector<float> tokenEnergy;
    if (c.hasTokenEnergy) {
        tokenEnergy.assign(c.tokenEnergy.begin(), c.tokenEnergy.begin() + kTokens);
    }
    drawAtlasBoard(Rectangle{content.x + boardSide + gap, content.y + 16.0f, boardSide, boardSide},
                   "token energy", tokenEnergy, c);
}

void paintAtlas(Rectangle body, const Ctx& c) {
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, 40.0f}, "BT4 attention atlas",
        "block/head fingerprint - board footprint pair - selected head's 64x64 matrix", c.theme,
        c.signature);
    const float topY = body.y + 50.0f;
    const float h = body.height - 50.0f;
    const float gap = 12.0f;
    if (body.width < 860.0f) {
        const float part = std::max(160.0f, (h - 2.0f * gap) / 3.0f);
        const Rectangle fingerprint{body.x, topY, body.width, part};
        const Rectangle boards{body.x, fingerprint.y + fingerprint.height + gap, body.width, part};
        const Rectangle heads{body.x, boards.y + boards.height + gap, body.width,
                              std::max(120.0f, body.y + body.height -
                                                   (boards.y + boards.height + gap))};
        paintAtlasFingerprint(fingerprint, c);
        paintAtlasBoards(boards, c);
        paintAttentionMatrix(heads, c);
        return;
    }
    const float leftW = std::max(500.0f, std::min(body.width * 0.58f, body.width - 380.0f));
    const Rectangle fingerprint{body.x, topY, leftW, h};
    const float rightX = fingerprint.x + fingerprint.width + gap;
    const float rightW = body.x + body.width - rightX;
    const float boardH = std::max(240.0f, std::min(h * 0.52f, 330.0f));
    const Rectangle boards{rightX, topY, rightW, boardH};
    const Rectangle heads{rightX, boards.y + boards.height + gap, rightW,
                          std::max(140.0f, body.y + body.height - (boards.y + boards.height + gap))};
    paintAtlasFingerprint(fingerprint, c);
    paintAtlasBoards(boards, c);
    paintAttentionMatrix(heads, c);
}

// ===========================================================================
// Mode 4 — Diagram (DIAGRAM): the static five-box architecture schematic.
// Transpiled from Bt4View.paintDiagram.
// ===========================================================================

void paintDiagram(Rectangle body, const Ctx& c) {
    const int headerH = 40;
    const int blocks = static_cast<int>(c.blocks.size());
    char headerSub[96];
    if (blocks > 0) {
        std::snprintf(headerSub, sizeof(headerSub),
                      "input -> embedding -> %d transformer blocks -> policy / value heads",
                      blocks);
    } else {
        std::snprintf(headerSub, sizeof(headerSub),
                      "input -> embedding -> transformer blocks -> policy / value heads");
    }
    tensorviz::drawSectionHeader(
        Rectangle{body.x, body.y, body.width, static_cast<float>(headerH)},
        "LC0 BT4 architecture", headerSub, c.theme, c.signature);

    static const char* titles[5] = {"input encoding", "embedding", "transformer blocks",
                                    "policy head", "value head"};
    int heads = 0;
    for (const BlockView& b : c.blocks) heads = std::max(heads, b.heads);
    char block2sub[40];
    if (heads > 0) {
        std::snprintf(block2sub, sizeof(block2sub), "%d self-attn heads x MLP", heads);
    } else {
        std::snprintf(block2sub, sizeof(block2sub), "self-attn heads x MLP");
    }
    const char* subs[5] = {"112 input planes (8x8)", "linear projection", block2sub,
                           "logits over moves", "WDL"};

    const int boxW = 200;
    const int boxH = 90;
    const int gap = 30;
    const int totalW = boxW * 5 + gap * 4;
    const int startX = static_cast<int>(body.x) + (static_cast<int>(body.width) - totalW) / 2;
    const int y = static_cast<int>(body.y) + headerH + 60;
    for (int i = 0; i < 5; ++i) {
        const int x = startX + i * (boxW + gap);
        const Rectangle box{static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(boxW), static_cast<float>(boxH)};
        DrawRectangleRec(box, tensorviz::withAlpha(c.theme.panelBackground, 240));
        DrawRectangleLinesEx(box, 1.0f,
                             i == 3   ? c.theme.accentBlue
                             : i == 4 ? c.theme.accentMagenta
                                      : c.theme.panelBorder);
        drawCenteredText(c.theme, titles[i], Rectangle{box.x, box.y + 8.0f, box.width, 24.0f}, 13,
                     c.theme.textPrimary);
        const std::string sub = tensorviz::fitText(c.theme, subs[i], 11, false, static_cast<float>(boxW) - 12.0f);
        drawCenteredText(c.theme, sub.c_str(), Rectangle{box.x, box.y + 40.0f, box.width, 20.0f}, 11,
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
}

}  // namespace

// ===========================================================================
// Public entry point: header band + 5-segment switcher + mode dispatch.
// ===========================================================================

void Bt4View::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;
    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    const float innerX = m_bounds.x + pad;
    const float innerW = m_bounds.width - 2.0f * pad;
    float y = m_bounds.y + pad;

    // 5-segment switcher (Overview / Trace / All / Atlas / Diagram).
    const Rectangle switcher{
        m_bounds.x + m_bounds.width - pad - 320.0f, y + 1.0f,
        320.0f, 28.0f};
    const std::string title = tensorviz::fitText(
        theme, "LC0 BT4 activations", 22, false,
        switcher.x - innerX - 10.0f);
    drawText(theme, title.c_str(), static_cast<int>(innerX),
             static_cast<int>(y), 22, theme.textPrimary);
    m_mode = drawSwitcher(switcher, m_mode, m_signature, theme);
    y += 34.0f;

    // Derive model dimensions from the active snapshot.
    int heads = 0;
    for (const Block& b : m_blocks) heads = std::max(heads, b.heads);
    std::size_t dim = 0;
    if (!m_finalTokens.empty()) {
        dim = m_finalTokens.size() / static_cast<std::size_t>(kTokens);
    } else if (!m_embedding.empty()) {
        dim = m_embedding.size() / static_cast<std::size_t>(kTokens);
    }
    char sub[120];
    std::snprintf(sub, sizeof(sub),
                  "%zu blocks - %d heads - %d tokens (squares) - %zu-d embedding",
                  m_blocks.size(), heads, kTokens, dim);
    drawTextMono(theme, sub, static_cast<int>(innerX), static_cast<int>(y), 13, theme.textMuted);
    y += 26.0f;

    const Rectangle body{innerX, y, innerW, m_bounds.y + m_bounds.height - y - pad};

    // Build the read-only context referencing the cached tensors.
    std::vector<BlockView> blocks;
    blocks.reserve(m_blocks.size());
    for (const auto& b : m_blocks) {
        blocks.push_back(BlockView{&b.attention, &b.ffn, b.heads, b.attentionFocus, b.ffnRms,
                               b.have});
    }
    const Ctx c{theme,
                m_signature,
                m_position,
                m_sprites,
                blocks,
                m_tokenEnergy,
                m_boardSalience,
                m_inputPlanes,
                m_tokenFeatures,
                m_embedding,
                m_finalTokens,
                m_policy,
                m_wdl,
                m_valueScalar,
                m_hasWdl,
                m_hasPolicy,
                m_hasTokenEnergy,
                m_selectedBlock,
                m_selectedHead,
                m_selectedSquare};

    // Diagram works without data; every other mode needs a snapshot.
    if (m_mode == 4) {
        paintDiagram(body, c);
        return;
    }
    if (!m_hasData) {
        drawText(theme, "Run an evaluation to populate this BT4 view.",
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
