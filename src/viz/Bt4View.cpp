#include "viz/Bt4View.h"

#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::lc0_bt4::snapshot_keys;
using Bt4Net = cnnv::nn::lc0_bt4::Network;

std::string shapeText(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return "-";
    std::string out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out += "x";
        out += std::to_string(shape[i]);
    }
    return out;
}

Bt4View::LayerStat layerStats(
    const char* name,
    const cnnv::nn::ActivationSnapshot::Entry& entry) {
    Bt4View::LayerStat s;
    s.name = name;
    s.shape = shapeText(entry.shape);
    s.shapeDims = entry.shape;
    s.values = entry.data;
    if (entry.data.empty()) return s;
    s.min = entry.data[0];
    s.max = entry.data[0];
    double sum = 0.0;
    double sumSq = 0.0;
    for (float v : entry.data) {
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum += v;
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double inv = 1.0 / static_cast<double>(entry.data.size());
    s.mean = static_cast<float>(sum * inv);
    s.rms = static_cast<float>(std::sqrt(sumSq * inv));
    return s;
}

bool readLayerStat(const cnnv::nn::ActivationSnapshot& snap,
                   const char* name,
                   const std::string& key,
                   Bt4View::LayerStat& out) {
    const auto* entry = snap.find(key);
    if (!entry) return false;
    out = layerStats(name, *entry);
    return true;
}

void copyFloats(const cnnv::nn::ActivationSnapshot& snap,
                const std::string& key,
                std::vector<float>& dst) {
    const float* data = snap.data(key);
    const std::size_t n = snap.size(key);
    if (!data || n == 0) {
        dst.clear();
        return;
    }
    dst.assign(data, data + n);
}

bool averageAttentionGrid(const cnnv::nn::ActivationSnapshot& snap,
                          int block,
                          std::vector<float>& average,
                          float& normalizedFocus) {
    const float* data = snap.data(keys::blockAttentionKey(block));
    const std::size_t n = snap.size(keys::blockAttentionKey(block));
    constexpr int kH = Bt4Net::kHeads;
    constexpr int kT = Bt4Net::kTokens;
    if (!data || n != static_cast<std::size_t>(kH * kT * kT)) return false;

    average.assign(static_cast<std::size_t>(kT * kT), 0.0f);
    float maxSum = 0.0f;
    for (int head = 0; head < kH; ++head) {
        for (int from = 0; from < kT; ++from) {
            float rowMax = 0.0f;
            for (int to = 0; to < kT; ++to) {
                const std::size_t idx =
                    (static_cast<std::size_t>(head) * kT + from) * kT + to;
                const float v = data[idx];
                average[static_cast<std::size_t>(from * kT + to)] +=
                    v / static_cast<float>(kH);
                rowMax = std::max(rowMax, v);
            }
            maxSum += rowMax;
        }
    }

    const float uniform = 1.0f / static_cast<float>(kT);
    const float meanMax = maxSum / static_cast<float>(kH * kT);
    normalizedFocus = std::clamp((meanMax - uniform) / (1.0f - uniform),
                                 0.0f, 1.0f);
    return true;
}

int findLayer(const std::vector<const Bt4View::LayerStat*>& layers,
              const char* name) {
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] && layers[i]->name == name) return static_cast<int>(i);
    }
    return -1;
}

float layerActivity(const Bt4View::LayerStat& layer, float scale) {
    return std::clamp(layer.rms / std::max(scale, 1e-6f), 0.0f, 1.0f);
}

void drawBt4Abstract(const Bt4View::LayerStat& inputPlanes,
                     const Bt4View::LayerStat& tokenFeatures,
                     const Bt4View::LayerStat& embedding,
                     const Bt4View::LayerStat& finalTokens,
                     const std::vector<Bt4View::BlockStat>& blocks,
                     const Bt4View::LayerStat& policy,
                     bool hasPolicy,
                     const Bt4View::LayerStat& valueLogits,
                     const Bt4View::LayerStat& wdlLayer,
                     bool hasWdl,
                     float valueScalar,
                     Rectangle bounds,
                     const Theme& theme) {
    float scale = std::max({inputPlanes.rms, tokenFeatures.rms, embedding.rms,
                            finalTokens.rms, policy.rms, valueLogits.rms,
                            wdlLayer.rms, 1e-6f});
    float blockRms = 0.0f;
    float focus = 0.0f;
    int blockSeen = 0;
    for (const auto& block : blocks) {
        if (!block.output.values.empty()) {
            blockRms += block.output.rms;
            focus += block.attentionFocus;
            ++blockSeen;
        }
    }
    if (blockSeen > 0) {
        blockRms /= static_cast<float>(blockSeen);
        focus /= static_cast<float>(blockSeen);
        scale = std::max(scale, blockRms);
    }

    constexpr float kHeaderH = 48.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "BT4 abstract flow",
        "board planes -> tokens -> transformer stack -> policy/value heads",
        theme);

    const float gap = 10.0f;
    const float x = bounds.x + 8.0f;
    const float y = bounds.y + kHeaderH + 14.0f;
    const float w = bounds.width - 16.0f;
    const float h = std::max(230.0f, bounds.height - kHeaderH - 24.0f);
    const float flowH = std::min(h, 500.0f);
    const float topH = std::clamp(flowH * 0.29f, 90.0f, 138.0f);
    const float bottomH = std::clamp(flowH * 0.25f, 82.0f, 126.0f);
    const float colW = (w - 4.0f * gap) * 0.20f;

    const Rectangle inputR{x, y, colW, topH};
    const Rectangle tokenR{x + colW + gap, y, colW, topH};
    const Rectangle embedR{x + 2.0f * (colW + gap), y, colW, topH};
    const Rectangle stackR{x + 3.0f * (colW + gap), y, colW, topH};
    const Rectangle finalR{x + 4.0f * (colW + gap), y, colW, topH};
    const float branchY = y + topH + 52.0f;
    const Rectangle policyR{x + 3.0f * (colW + gap), branchY, colW, bottomH};
    const Rectangle valueR{x + 4.0f * (colW + gap), branchY, colW, bottomH};

    tensorviz::drawElbowConnection(inputR, tokenR, theme.accentGreen, true, false, theme);
    tensorviz::drawElbowConnection(tokenR, embedR, theme.accentGreen, true, false, theme);
    tensorviz::drawElbowConnection(embedR, stackR, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(stackR, finalR, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(finalR, policyR, theme.accentBlue, true, false, theme);
    tensorviz::drawElbowConnection(finalR, valueR, theme.accentMagenta, true, false, theme);

    tensorviz::drawAbstractBlock(inputR, "input planes", inputPlanes.shape,
                                 layerActivity(inputPlanes, scale),
                                 theme.accentGreen, false, theme);
    tensorviz::drawAbstractBlock(tokenR, "tokens", tokenFeatures.shape,
                                 layerActivity(tokenFeatures, scale),
                                 theme.accentGreen, false, theme);
    tensorviz::drawAbstractBlock(embedR, "embedding", embedding.shape,
                                 layerActivity(embedding, scale),
                                 theme.accentOrange, false, theme);
    char stackDetail[64];
    std::snprintf(stackDetail, sizeof(stackDetail), "%zu blocks | focus %.2f",
                  blocks.size(), focus);
    tensorviz::drawAbstractBlock(stackR, "transformer", stackDetail,
                                 blockRms / scale,
                                 theme.accentOrange, false, theme);
    if (!blocks.empty()) {
        const int ticks = std::min<int>(static_cast<int>(blocks.size()), 24);
        const float tickGap = 2.0f;
        const float tickW = (stackR.width - 20.0f - tickGap * static_cast<float>(ticks - 1)) /
                            static_cast<float>(ticks);
        const float ty = stackR.y + stackR.height - 26.0f;
        for (int i = 0; i < ticks; ++i) {
            const float f = blocks[static_cast<std::size_t>(i)].attentionFocus;
            DrawRectangleRec(Rectangle{stackR.x + 10.0f + static_cast<float>(i) * (tickW + tickGap),
                                       ty, std::max(1.0f, tickW), 7.0f},
                             tensorviz::blend(tensorviz::withAlpha(theme.accentBlue, 120),
                                               tensorviz::withAlpha(theme.accentYellow, 185),
                                               f));
        }
    }
    tensorviz::drawAbstractBlock(finalR, "final tokens", finalTokens.shape,
                                 layerActivity(finalTokens, scale),
                                 theme.accentYellow, false, theme);
    tensorviz::drawAbstractBlock(policyR, "policy head",
                                 hasPolicy ? policy.shape : "1858",
                                 layerActivity(policy, scale),
                                 theme.accentBlue, false, theme);
    char valueDetail[64];
    std::snprintf(valueDetail, sizeof(valueDetail), "WDL | v %.3f", valueScalar);
    tensorviz::drawAbstractBlock(valueR, "value head", valueDetail,
                                 hasWdl ? layerActivity(wdlLayer, scale)
                                        : layerActivity(valueLogits, scale),
                                 theme.accentMagenta, false, theme);

    Rectangle out{x, branchY + bottomH + 16.0f, std::min(340.0f, w),
                  bounds.y + bounds.height - (branchY + bottomH + 28.0f)};
    if (hasWdl && out.height >= 54.0f) {
        tensorviz::drawMetricBars(out,
                                  std::vector<std::string>{"W", "D", "L"},
                                  wdlLayer.values,
                                  1.0f, theme);
    }
}

void drawBt4LayerList(const std::vector<const Bt4View::LayerStat*>& layers,
                      const std::vector<float>& attention,
                      bool hasAttention,
                      const std::vector<float>& tokenEnergy,
                      bool hasTokenEnergy,
                      const std::vector<float>& valueSalience,
                      bool hasValueSalience,
                      float valueScalar,
                      int& selectedLayer,
                      Rectangle bounds,
                      const Theme& theme) {
    if (layers.empty()) return;

    float scale = 0.0f;
    for (const auto* layer : layers) {
        if (layer) scale = std::max(scale, layer->rms);
    }
    scale = std::max(scale, 1e-6f);

    const float gap = 8.0f;
    constexpr float kHeaderH = 48.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "detailed token tensors",
        "64 tokens | 15 blocks | policy/value heads | signed activations",
        theme);

    const std::size_t cols = bounds.width >= 780.0f ? 5 :
                             bounds.width >= 620.0f ? 4 : 3;
    const std::size_t rows = (layers.size() + cols - 1) / cols;
    const float inspectorMinH = std::clamp(bounds.height * 0.24f, 230.0f, 340.0f);
    const float top = bounds.y + kHeaderH + 8.0f;
    const float bottom = bounds.y + bounds.height;
    const float cardAreaH = std::max(80.0f, bottom - top - inspectorMinH - gap);
    const float cellW = (bounds.width - 16.0f -
                         gap * static_cast<float>(cols - 1)) /
                        static_cast<float>(cols);
    const float rawCellH =
        (cardAreaH - gap * static_cast<float>(rows - 1)) /
        static_cast<float>(std::max<std::size_t>(1, rows));
    const float cellH = std::clamp(rawCellH, 54.0f, 94.0f);

    std::vector<Rectangle> cards;
    cards.reserve(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::size_t row = i / cols;
        const std::size_t col = i % cols;
        const std::size_t visualCol = (row % 2 == 0) ? col : cols - 1 - col;
        Rectangle card{
            bounds.x + 8.0f + static_cast<float>(visualCol) * (cellW + gap),
            top + static_cast<float>(row) * (cellH + gap),
            cellW,
            cellH,
        };
        if (card.y + card.height > bottom - inspectorMinH - 6.0f) break;
        cards.push_back(card);
    }
    if (cards.empty()) return;

    const int finalIdx = findLayer(layers, "final 64d");
    const int policyIdx = findLayer(layers, "policy");
    const int valueIdx = findLayer(layers, "value");
    const int wdlIdx = findLayer(layers, "WDL");

    std::vector<int> parent(layers.size(), -1);
    auto setParent = [&](int child, int p) {
        if (child >= 0 && p >= 0 &&
            static_cast<std::size_t>(child) < parent.size() &&
            static_cast<std::size_t>(p) < parent.size()) {
            parent[static_cast<std::size_t>(child)] = p;
        }
    };
    if (finalIdx > 0) {
        for (int i = 1; i <= finalIdx; ++i) setParent(i, i - 1);
        setParent(policyIdx, finalIdx);
        setParent(valueIdx, finalIdx);
        setParent(wdlIdx, valueIdx);
    } else {
        for (std::size_t i = 1; i < layers.size(); ++i) {
            parent[i] = static_cast<int>(i - 1);
        }
    }

    const int fallbackSelected =
        finalIdx >= 0 ? finalIdx : static_cast<int>(layers.size()) - 1;
    if (selectedLayer < 0 ||
        selectedLayer >= static_cast<int>(cards.size())) {
        selectedLayer = fallbackSelected;
    }
    const Vector2 mouse = GetMousePosition();
    for (std::size_t i = 0; i < cards.size(); ++i) {
        if (CheckCollisionPointRec(mouse, cards[i]) &&
            IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            selectedLayer = static_cast<int>(i);
        }
    }
    const int selected = std::clamp(selectedLayer, 0,
                                    static_cast<int>(cards.size()) - 1);
    selectedLayer = selected;

    std::vector<bool> path(layers.size(), false);
    for (int cur = selected; cur >= 0 && static_cast<std::size_t>(cur) < parent.size();
         cur = parent[static_cast<std::size_t>(cur)]) {
        path[static_cast<std::size_t>(cur)] = true;
    }

    auto edgeActive = [&](int from, int to) {
        return to >= 0 && static_cast<std::size_t>(to) < parent.size() &&
               parent[static_cast<std::size_t>(to)] == from &&
               path[static_cast<std::size_t>(to)];
    };
    auto connect = [&](int from, int to, Color color) {
        if (from >= 0 && to >= 0 &&
            static_cast<std::size_t>(from) < cards.size() &&
            static_cast<std::size_t>(to) < cards.size()) {
            const bool active = edgeActive(from, to);
            tensorviz::drawElbowConnection(cards[static_cast<std::size_t>(from)],
                                           cards[static_cast<std::size_t>(to)],
                                           color, active, !active, theme);
        }
    };

    Color trunk = theme.accentOrange;
    Color policy = theme.accentBlue;
    Color value = theme.accentMagenta;
    if (finalIdx > 0) {
        for (int i = 1; i <= finalIdx; ++i) connect(i - 1, i, trunk);
        connect(finalIdx, policyIdx, policy);
        connect(finalIdx, valueIdx, value);
        connect(valueIdx, wdlIdx, value);
    } else {
        for (std::size_t i = 1; i < cards.size(); ++i) {
            connect(static_cast<int>(i - 1), static_cast<int>(i), trunk);
        }
    }

    for (std::size_t i = 0; i < cards.size(); ++i) {
        Color accent = theme.accentOrange;
        if (static_cast<int>(i) == policyIdx) {
            accent = theme.accentBlue;
        } else if (static_cast<int>(i) == valueIdx ||
                   static_cast<int>(i) == wdlIdx) {
            accent = theme.accentMagenta;
        } else if (i < 3) {
            accent = theme.accentGreen;
        }
        const bool onPath = i < path.size() && path[i];
        tensorviz::drawLayerCard(cards[i], layers[i]->name, layers[i]->shape,
                                 layers[i]->values, layers[i]->shapeDims,
                                 scale, accent, static_cast<int>(i) == selected,
                                 !onPath, theme);
    }

    const Bt4View::LayerStat& focus = *layers[static_cast<std::size_t>(selected)];
    char note[96];
    std::snprintf(note, sizeof(note), "mean %.3f | value %.3f",
                  focus.mean, valueScalar);
    const float cardsBottom = cards.empty() ? top :
        cards.back().y + cards.back().height;
    const float detailY = std::min(bottom - 160.0f, cardsBottom + 12.0f);
    Rectangle detail{bounds.x, detailY, bounds.width,
                     std::max(160.0f, bottom - detailY)};
    Rectangle inspector = detail;
    Rectangle diagnostics{};
    if (detail.width >= 760.0f &&
        (hasAttention || hasTokenEnergy || hasValueSalience)) {
        diagnostics = Rectangle{detail.x + detail.width - 250.0f, detail.y,
                                250.0f, detail.height};
        inspector.width -= diagnostics.width + gap;
    }
    tensorviz::drawInspector(inspector, focus.name, focus.shape, focus.values,
                             focus.shapeDims, note, theme);

    if (diagnostics.width > 0.0f) {
        DrawRectangleRec(diagnostics, theme.panelBackground);
        DrawRectangleLinesEx(diagnostics, 1.0f, theme.panelBorder);
        drawText(theme, "token diagnostics",
                 static_cast<int>(diagnostics.x + 9.0f),
                 static_cast<int>(diagnostics.y + 8.0f),
                 15, theme.textPrimary);

        const float x = diagnostics.x + 8.0f;
        float y = diagnostics.y + 34.0f;
        const float w = diagnostics.width - 16.0f;
        if (hasAttention) {
            drawTextMono(theme, "last block attention",
                         static_cast<int>(x), static_cast<int>(y),
                         10, theme.textMuted);
            tensorviz::drawTensorPreview(
                attention, std::vector<std::size_t>{64, 64},
                Rectangle{x, y + 14.0f, w, std::min(94.0f, diagnostics.height * 0.36f)},
                theme, 0.0f, true);
            y += std::min(112.0f, diagnostics.height * 0.42f);
        }
        const float miniH = std::max(44.0f, (diagnostics.y + diagnostics.height - y - 8.0f) * 0.5f);
        if (hasTokenEnergy) {
            drawTextMono(theme, "token rms",
                         static_cast<int>(x), static_cast<int>(y),
                         10, theme.textMuted);
            tensorviz::drawTensorPreview(tokenEnergy, std::vector<std::size_t>{64},
                                         Rectangle{x, y + 14.0f, w, miniH - 16.0f},
                                         theme, 0.0f, true);
            y += miniH + 4.0f;
        }
        if (hasValueSalience) {
            drawTextMono(theme, "value salience",
                         static_cast<int>(x), static_cast<int>(y),
                         10, theme.textMuted);
            tensorviz::drawTensorPreview(valueSalience, std::vector<std::size_t>{64},
                                         Rectangle{x, y + 14.0f, w,
                                                   diagnostics.y + diagnostics.height - y - 22.0f},
                                         theme, 0.0f, true);
        }
    }
}

}  // namespace

void Bt4View::update(const cnnv::nn::ActivationSnapshot& snap) {
    m_inputPlanes = {};
    m_tokenFeatures = {};
    m_embedding = {};
    m_finalTokens = {};
    m_policy = {};
    m_valueLogits = {};
    m_blocks.clear();
    m_attention.clear();
    m_tokenEnergy.clear();
    m_valueSalience.clear();
    m_wdl = {0.0f, 0.0f, 0.0f};
    m_valueScalar = 0.0f;
    m_hasOverview = false;
    m_hasAttention = false;
    m_hasTokenEnergy = false;
    m_hasValueSalience = false;
    m_hasPolicy = false;
    m_hasWdl = false;

    bool overview = false;
    overview |= readLayerStat(snap, "112 planes", keys::kInputPlanes, m_inputPlanes);
    overview |= readLayerStat(snap, "64 tokens", keys::kTokenFeatures, m_tokenFeatures);
    overview |= readLayerStat(snap, "embed 64d", keys::kEmbedding, m_embedding);
    overview |= readLayerStat(snap, "final 64d", keys::kFinalTokens, m_finalTokens);
    m_hasOverview = overview;

    for (int i = 0; i < Bt4Net::kBlocks; ++i) {
        BlockStat block;
        block.index = i + 1;
        block.have |= readLayerStat(snap, "attn", keys::blockAttentionOutKey(i),
                                    block.attentionOut);
        block.have |= readLayerStat(snap, "ffn", keys::blockFfnKey(i),
                                    block.ffn);
        block.have |= readLayerStat(snap, "out", keys::blockOutputKey(i),
                                    block.output);
        if (!block.output.values.empty()) {
            char label[8];
            std::snprintf(label, sizeof(label), "B%02d", i + 1);
            block.output.name = label;
        }

        std::vector<float> avgAttention;
        float focus = 0.0f;
        if (averageAttentionGrid(snap, i, avgAttention, focus)) {
            block.attentionFocus = focus;
            block.have = true;
            if (i == Bt4Net::kBlocks - 1) {
                m_attention = std::move(avgAttention);
                m_hasAttention = true;
            }
        }

        if (block.have) m_blocks.push_back(std::move(block));
    }

    copyFloats(snap, keys::kFinalTokenMagnitude, m_tokenEnergy);
    m_hasTokenEnergy = m_tokenEnergy.size() == static_cast<std::size_t>(Bt4Net::kTokens);

    copyFloats(snap, keys::kBoardSalience, m_valueSalience);
    m_hasValueSalience = m_valueSalience.size() == static_cast<std::size_t>(Bt4Net::kTokens);

    m_hasPolicy = readLayerStat(snap, "policy", keys::kPolicyLogits, m_policy);
    (void)readLayerStat(snap, "value", keys::kValueLogits, m_valueLogits);

    if (snap.size(keys::kValueWdl) == 3) {
        const float* w = snap.data(keys::kValueWdl);
        m_wdl = {w[0], w[1], w[2]};
        m_hasWdl = true;
    }
    if (snap.size(keys::kValueScalar) == 1) {
        m_valueScalar = snap.data(keys::kValueScalar)[0];
    } else if (m_hasWdl) {
        m_valueScalar = m_wdl[0] - m_wdl[2];
    }
}

void Bt4View::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;

    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    const float innerX = m_bounds.x + pad;
    const float innerW = m_bounds.width - 2.0f * pad;
    float y = m_bounds.y + pad;

    drawText(theme, "BT4 layers and activations",
             static_cast<int>(innerX),
             static_cast<int>(y), 23, theme.textPrimary);
    m_detailed = tensorviz::drawModeToggle(
        Rectangle{m_bounds.x + m_bounds.width - pad - 190.0f, y + 1.0f,
                  190.0f, 30.0f},
        m_detailed, theme);
    y += 32.0f;
    drawTextMono(theme, "64 board tokens x 176 -> 64d | 15 blocks | 8 heads | policy 1858 + WDL",
                 static_cast<int>(innerX),
                 static_cast<int>(y), 13, theme.textMuted);
    y += 26.0f;

    if (!m_hasOverview && m_blocks.empty()) {
        drawText(theme, "No BT4 snapshot yet.",
                 static_cast<int>(innerX),
                 static_cast<int>(y), 17, theme.textMuted);
        return;
    }

    LayerStat wdlLayer;
    wdlLayer.name = "WDL";
    wdlLayer.shape = "3";
    wdlLayer.values = {m_wdl[0], m_wdl[1], m_wdl[2]};
    if (m_hasWdl) {
        wdlLayer.min = std::min({m_wdl[0], m_wdl[1], m_wdl[2]});
        wdlLayer.max = std::max({m_wdl[0], m_wdl[1], m_wdl[2]});
        double sumSq = 0.0;
        for (float v : wdlLayer.values) sumSq += static_cast<double>(v) * v;
        wdlLayer.rms = static_cast<float>(std::sqrt(sumSq / 3.0));
    }
    wdlLayer.shapeDims = {3};

    std::vector<const LayerStat*> layers;
    if (m_hasOverview) {
        layers.push_back(&m_inputPlanes);
        layers.push_back(&m_tokenFeatures);
        layers.push_back(&m_embedding);
    }
    for (const auto& block : m_blocks) {
        layers.push_back(&block.output);
    }
    if (m_hasOverview) layers.push_back(&m_finalTokens);
    if (m_hasPolicy) layers.push_back(&m_policy);
    if (!m_valueLogits.values.empty()) layers.push_back(&m_valueLogits);
    if (m_hasWdl) layers.push_back(&wdlLayer);

    Rectangle content{innerX, y, innerW, m_bounds.y + m_bounds.height - y - pad};
    if (m_detailed) {
        drawBt4LayerList(layers,
                         m_attention, m_hasAttention,
                         m_tokenEnergy, m_hasTokenEnergy,
                         m_valueSalience, m_hasValueSalience,
                         m_valueScalar,
                         m_selectedLayer,
                         content,
                         theme);
    } else {
        drawBt4Abstract(m_inputPlanes, m_tokenFeatures, m_embedding, m_finalTokens,
                        m_blocks, m_policy, m_hasPolicy, m_valueLogits,
                        wdlLayer, m_hasWdl, m_valueScalar, content, theme);
    }
}

}  // namespace cnnv::viz
