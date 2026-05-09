#include "viz/CnnView.h"

#include "nn/ActivationSnapshot.h"
#include "nn/lc0_cnn/Lc0CnnNetwork.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <utility>

namespace cnnv::viz {

namespace {

namespace keys = cnnv::nn::lc0_cnn::snapshot_keys;

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

std::string shapeText(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return "-";
    std::string out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out += "x";
        out += std::to_string(shape[i]);
    }
    return out;
}

CnnView::LayerStat layerStats(const char* name,
                              const cnnv::nn::ActivationSnapshot::Entry& e) {
    CnnView::LayerStat s;
    s.name = name;
    s.shape = shapeText(e.shape);
    s.shapeDims = e.shape;
    s.values = e.data;
    if (e.data.empty()) return s;

    s.min = e.data[0];
    s.max = e.data[0];
    double sum = 0.0;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    for (float v : e.data) {
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum += v;
        sumAbs += std::fabs(v);
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double inv = 1.0 / static_cast<double>(e.data.size());
    s.mean = static_cast<float>(sum * inv);
    s.meanAbs = static_cast<float>(sumAbs * inv);
    s.rms = static_cast<float>(std::sqrt(sumSq * inv));
    return s;
}

int findLayer(const std::vector<CnnView::LayerStat>& layers, const char* name) {
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

const CnnView::LayerStat* layerByName(const std::vector<CnnView::LayerStat>& layers,
                                      const char* name) {
    const int idx = findLayer(layers, name);
    return idx >= 0 ? &layers[static_cast<std::size_t>(idx)] : nullptr;
}

float layerActivity(const CnnView::LayerStat* layer, float scale) {
    if (!layer) return 0.0f;
    return std::clamp(layer->rms / std::max(scale, 1e-6f), 0.0f, 1.0f);
}

void drawCnnAbstract(const std::vector<CnnView::LayerStat>& layers,
                     int blockCount,
                     bool hasWdl,
                     const float wdl[3],
                     float valueScalar,
                     const std::vector<std::pair<int, float>>& topPolicy,
                     Rectangle bounds,
                     const Theme& theme) {
    if (layers.empty()) return;

    const CnnView::LayerStat* input = layerByName(layers, "input");
    const CnnView::LayerStat* stem = layerByName(layers, "stem");
    const CnnView::LayerStat* final = layerByName(layers, "final");
    const CnnView::LayerStat* policy = layerByName(layers, "pLogit");
    const CnnView::LayerStat* value = layerByName(layers, "WDL");

    float scale = 0.0f;
    float trunkRms = 0.0f;
    int trunkSeen = 0;
    for (const auto& layer : layers) {
        scale = std::max(scale, layer.rms);
        if (!layer.name.empty() && layer.name[0] == 'B') {
            trunkRms += layer.rms;
            ++trunkSeen;
        }
    }
    if (trunkSeen > 0) trunkRms /= static_cast<float>(trunkSeen);
    scale = std::max(scale, 1e-6f);

    constexpr float kHeaderH = 48.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "CNN abstract flow",
        "input planes -> convolution trunk -> policy/value heads",
        theme);

    const float gap = 10.0f;
    const float x = bounds.x + 8.0f;
    const float y = bounds.y + kHeaderH + 14.0f;
    const float w = bounds.width - 16.0f;
    const float h = std::max(220.0f, bounds.height - kHeaderH - 24.0f);
    const float flowH = std::min(h, 470.0f);
    const float topH = std::clamp(flowH * 0.30f, 92.0f, 140.0f);
    const float bottomH = std::clamp(flowH * 0.26f, 82.0f, 126.0f);
    const float colW = (w - 3.0f * gap) * 0.25f;

    const Rectangle inputR{x, y, colW, topH};
    const Rectangle stemR{x + colW + gap, y, colW, topH};
    const Rectangle trunkR{x + 2.0f * (colW + gap), y, colW, topH};
    const Rectangle finalR{x + 3.0f * (colW + gap), y, colW, topH};
    const float branchY = y + topH + 48.0f;
    const Rectangle policyR{x + 2.0f * (colW + gap), branchY, colW, bottomH};
    const Rectangle valueR{x + 3.0f * (colW + gap), branchY, colW, bottomH};

    tensorviz::drawElbowConnection(inputR, stemR, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(stemR, trunkR, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(trunkR, finalR, theme.accentOrange, true, false, theme);
    tensorviz::drawElbowConnection(finalR, policyR, theme.accentBlue, true, false, theme);
    tensorviz::drawElbowConnection(finalR, valueR, theme.accentMagenta, true, false, theme);

    tensorviz::drawAbstractBlock(inputR, "input planes",
                                 input ? input->shape : "112x8x8",
                                 layerActivity(input, scale), theme.accentGreen,
                                 false, theme);
    tensorviz::drawAbstractBlock(stemR, "stem conv",
                                 stem ? stem->shape : "-",
                                 layerActivity(stem, scale), theme.accentOrange,
                                 false, theme);
    char trunkDetail[64];
    std::snprintf(trunkDetail, sizeof(trunkDetail), "%d residual blocks", blockCount);
    tensorviz::drawAbstractBlock(trunkR, "residual trunk", trunkDetail,
                                 trunkRms / scale, theme.accentOrange,
                                 false, theme);
    if (blockCount > 0) {
        const int ticks = std::min(blockCount, 32);
        const float tickGap = 2.0f;
        const float tickW = (trunkR.width - 20.0f - tickGap * static_cast<float>(ticks - 1)) /
                            static_cast<float>(ticks);
        const float ty = trunkR.y + trunkR.height - 26.0f;
        for (int i = 0; i < ticks; ++i) {
            DrawRectangleRec(Rectangle{trunkR.x + 10.0f + static_cast<float>(i) * (tickW + tickGap),
                                       ty, std::max(1.0f, tickW), 7.0f},
                             tensorviz::withAlpha(theme.accentOrange, 145));
        }
    }
    tensorviz::drawAbstractBlock(finalR, "final map",
                                 final ? final->shape : "64",
                                 layerActivity(final, scale), theme.accentYellow,
                                 false, theme);
    tensorviz::drawAbstractBlock(policyR, "policy head",
                                 policy ? policy->shape : "logits",
                                 layerActivity(policy, scale), theme.accentBlue,
                                 false, theme);
    char valueDetail[64];
    std::snprintf(valueDetail, sizeof(valueDetail), "WDL | v %.3f", valueScalar);
    tensorviz::drawAbstractBlock(valueR, "value head", valueDetail,
                                 layerActivity(value, scale), theme.accentMagenta,
                                 false, theme);

    Rectangle side{x, branchY + bottomH + 16.0f, w,
                   bounds.y + bounds.height - (branchY + bottomH + 28.0f)};
    if (side.height >= 54.0f) {
        if (hasWdl) {
            tensorviz::drawMetricBars(
                Rectangle{side.x, side.y, std::min(260.0f, side.width * 0.45f), side.height},
                std::vector<std::string>{"W", "D", "L"},
                std::vector<float>{wdl[0], wdl[1], wdl[2]},
                1.0f, theme);
        }
        if (!topPolicy.empty() && side.width >= 500.0f) {
            std::vector<std::string> labels;
            std::vector<float> values;
            const std::size_t keep = std::min<std::size_t>(4, topPolicy.size());
            labels.reserve(keep);
            values.reserve(keep);
            float policyScale = 0.0f;
            for (std::size_t i = 0; i < keep; ++i) {
                char label[16];
                std::snprintf(label, sizeof(label), "#%d", topPolicy[i].first);
                labels.emplace_back(label);
                values.push_back(topPolicy[i].second);
                policyScale = std::max(policyScale, std::fabs(topPolicy[i].second));
            }
            tensorviz::drawMetricBars(
                Rectangle{side.x + side.width * 0.48f, side.y,
                          side.width * 0.52f, side.height},
                labels, values, policyScale, theme);
        }
    }
}

void drawLayerPipeline(const std::vector<CnnView::LayerStat>& layers,
                       Rectangle bounds,
                       bool hasWdl,
                       const float wdl[3],
                       float valueScalar,
                       const std::vector<std::pair<int, float>>& topPolicy,
                       int& selectedLayer,
                       const Theme& theme) {
    if (layers.empty()) return;

    float scale = 0.0f;
    for (const auto& layer : layers) scale = std::max(scale, layer.rms);
    scale = std::max(scale, 1e-6f);

    const float gap = 8.0f;
    constexpr float kHeaderH = 48.0f;
    tensorviz::drawSectionHeader(
        Rectangle{bounds.x, bounds.y, bounds.width, kHeaderH},
        "detailed tensor graph",
        "backbone -> policy/value heads | signed activations | RMS activity",
        theme);

    const std::size_t cols = bounds.width >= 760.0f ? 6 :
                             bounds.width >= 610.0f ? 5 : 4;
    const std::size_t rows = (layers.size() + cols - 1) / cols;
    const float inspectorMinH = std::clamp(bounds.height * 0.24f, 220.0f, 330.0f);
    const float top = bounds.y + kHeaderH + 8.0f;
    const float bottom = bounds.y + bounds.height;
    const float cardAreaH = std::max(80.0f, bottom - top - inspectorMinH - gap);
    const float cellW = (bounds.width - 16.0f -
                         gap * static_cast<float>(cols - 1)) /
                        static_cast<float>(cols);
    const float rawCellH =
        (cardAreaH - gap * static_cast<float>(rows - 1)) /
        static_cast<float>(std::max<std::size_t>(1, rows));
    const float cellH = std::clamp(rawCellH, 54.0f, 92.0f);
    std::vector<Rectangle> cells;
    cells.reserve(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::size_t row = i / cols;
        const std::size_t col = i % cols;
        const std::size_t visualCol = (row % 2 == 0) ? col : cols - 1 - col;
        Rectangle cell{
            bounds.x + 8.0f + static_cast<float>(visualCol) * (cellW + gap),
            top + static_cast<float>(row) * (cellH + gap),
            cellW,
            cellH,
        };
        if (cell.y + cell.height > bottom - inspectorMinH - 6.0f) break;
        cells.push_back(cell);
    }
    if (cells.empty()) return;

    const int finalIdx = findLayer(layers, "final");
    const int pStemIdx = findLayer(layers, "pStem");
    const int pPlaneIdx = findLayer(layers, "pPlane");
    const int pLogitIdx = findLayer(layers, "pLogit");
    const int vConvIdx = findLayer(layers, "vConv");
    const int fc1Idx = findLayer(layers, "fc1");
    const int vLogitIdx = findLayer(layers, "vLogit");
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
        setParent(pStemIdx, finalIdx);
        setParent(pPlaneIdx, pStemIdx);
        setParent(pLogitIdx, pPlaneIdx);
        setParent(vConvIdx, finalIdx);
        setParent(fc1Idx, vConvIdx);
        setParent(vLogitIdx, fc1Idx);
        setParent(wdlIdx, vLogitIdx);
    } else {
        for (std::size_t i = 1; i < layers.size(); ++i) {
            parent[i] = static_cast<int>(i - 1);
        }
    }

    const int fallbackSelected =
        finalIdx >= 0 ? finalIdx : static_cast<int>(layers.size()) - 1;
    if (selectedLayer < 0 ||
        selectedLayer >= static_cast<int>(cells.size())) {
        selectedLayer = fallbackSelected;
    }
    const Vector2 mouse = GetMousePosition();
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (CheckCollisionPointRec(mouse, cells[i]) &&
            IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            selectedLayer = static_cast<int>(i);
        }
    }
    const int selected = std::clamp(selectedLayer, 0,
                                    static_cast<int>(cells.size()) - 1);
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
            static_cast<std::size_t>(from) < cells.size() &&
            static_cast<std::size_t>(to) < cells.size()) {
            const bool active = edgeActive(from, to);
            tensorviz::drawElbowConnection(cells[static_cast<std::size_t>(from)],
                                           cells[static_cast<std::size_t>(to)],
                                           color, active, !active, theme);
        }
    };

    Color trunk = theme.accentOrange;
    Color policy = theme.accentBlue;
    Color value = theme.accentMagenta;
    if (finalIdx > 0) {
        for (int i = 1; i <= finalIdx; ++i) connect(i - 1, i, trunk);
        connect(finalIdx, pStemIdx, policy);
        connect(pStemIdx, pPlaneIdx, policy);
        connect(pPlaneIdx, pLogitIdx, policy);
        connect(finalIdx, vConvIdx, value);
        connect(vConvIdx, fc1Idx, value);
        connect(fc1Idx, vLogitIdx, value);
        connect(vLogitIdx, wdlIdx, value);
    } else {
        for (std::size_t i = 1; i < cells.size(); ++i) {
            connect(static_cast<int>(i - 1), static_cast<int>(i), trunk);
        }
    }
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const bool onPath = i < path.size() && path[i];
        Color accent = theme.accentOrange;
        if (static_cast<int>(i) == pStemIdx || static_cast<int>(i) == pPlaneIdx ||
            static_cast<int>(i) == pLogitIdx) {
            accent = theme.accentBlue;
        } else if (static_cast<int>(i) == vConvIdx || static_cast<int>(i) == fc1Idx ||
                   static_cast<int>(i) == vLogitIdx || static_cast<int>(i) == wdlIdx) {
            accent = theme.accentMagenta;
        }
        tensorviz::drawLayerCard(cells[i], layers[i].name, layers[i].shape,
                                 layers[i].values, layers[i].shapeDims,
                                 scale, accent, static_cast<int>(i) == selected,
                                 !onPath, theme);
    }

    const CnnView::LayerStat& focus = layers[static_cast<std::size_t>(selected)];
    char note[128];
    std::snprintf(note, sizeof(note), "mean %.3f | abs %.3f | value %.3f",
                  focus.mean, focus.meanAbs, valueScalar);

    const float cardsBottom = cells.empty() ? top :
        cells.back().y + cells.back().height;
    const float detailY = std::min(bottom - 150.0f, cardsBottom + 12.0f);
    Rectangle detail{bounds.x, detailY, bounds.width,
                     std::max(150.0f, bottom - detailY)};
    Rectangle inspector = detail;
    Rectangle outputs{};
    if (detail.width >= 760.0f && (hasWdl || !topPolicy.empty())) {
        outputs = Rectangle{detail.x + detail.width - 226.0f, detail.y,
                            226.0f, detail.height};
        inspector.width -= outputs.width + gap;
    }
    tensorviz::drawInspector(inspector, focus.name, focus.shape, focus.values,
                             focus.shapeDims, note, theme);

    if (outputs.width > 0.0f) {
        DrawRectangleRec(outputs, theme.panelBackground);
        DrawRectangleLinesEx(outputs, 1.0f, theme.panelBorder);
        drawText(theme, "outputs",
                 static_cast<int>(outputs.x + 9.0f),
                 static_cast<int>(outputs.y + 8.0f), 15, theme.textPrimary);
        if (hasWdl) {
            tensorviz::drawMetricBars(
                Rectangle{outputs.x + 8.0f, outputs.y + 34.0f,
                          outputs.width - 16.0f, 76.0f},
                std::vector<std::string>{"W", "D", "L"},
                std::vector<float>{wdl[0], wdl[1], wdl[2]},
                1.0f, theme);
        }
        if (!topPolicy.empty()) {
            std::vector<std::string> labels;
            std::vector<float> values;
            const std::size_t keep = std::min<std::size_t>(5, topPolicy.size());
            labels.reserve(keep);
            values.reserve(keep);
            float policyScale = 0.0f;
            for (std::size_t i = 0; i < keep; ++i) {
                char label[16];
                std::snprintf(label, sizeof(label), "#%d", topPolicy[i].first);
                labels.emplace_back(label);
                values.push_back(topPolicy[i].second);
                policyScale = std::max(policyScale, std::fabs(topPolicy[i].second));
            }
            tensorviz::drawMetricBars(
                Rectangle{outputs.x + 8.0f, outputs.y + 120.0f,
                          outputs.width - 16.0f,
                          outputs.height - 128.0f},
                labels, values, policyScale, theme);
        }
    }
}

}  // namespace

void CnnView::update(const cnnv::nn::ActivationSnapshot& snap) {
    copyFloats(snap, keys::kFinalActivation, m_heatmap);
    copyFloats(snap, keys::kPolicyLogits, m_policyLogits);
    m_hasHeatmap = m_heatmap.size() == 64;
    m_blockCount = 0;
    m_layers.clear();
    for (const auto& [key, entry] : snap.entries()) {
        (void)entry;
        if (key.find("cnn.block") == 0 && key.find(".relu") != std::string::npos) {
            ++m_blockCount;
        }
    }
    auto addLayer = [&](const char* name, const std::string& key) {
        if (const auto* entry = snap.find(key)) {
            m_layers.push_back(layerStats(name, *entry));
        }
    };
    addLayer("input", keys::kInputPlanes);
    addLayer("stem", keys::kStemRelu);
    for (int i = 0, seen = 0; seen < m_blockCount && i < 256; ++i) {
        const std::string key = keys::blockReluKey(i);
        if (!snap.has(key)) continue;
        char label[8];
        std::snprintf(label, sizeof(label), "B%d", i + 1);
        addLayer(label, key);
        ++seen;
    }
    addLayer("final", keys::kFinalActivation);
    addLayer("pStem", keys::kPolicyHidden);
    addLayer("pPlane", keys::kPolicyPlanes);
    addLayer("pLogit", keys::kPolicyLogits);
    addLayer("vConv", keys::kValueConv);
    addLayer("fc1", keys::kValueFc1);
    addLayer("vLogit", keys::kValueLogits);
    addLayer("WDL", keys::kValueWdl);

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
        std::partial_sort(indices.begin(), indices.begin() + keep, indices.end(),
                          [&](int a, int b) {
                              return m_policyLogits[static_cast<std::size_t>(a)] >
                                     m_policyLogits[static_cast<std::size_t>(b)];
                          });
        for (std::size_t i = 0; i < keep; ++i) {
            const int idx = indices[i];
            m_topPolicy.push_back({idx, m_policyLogits[static_cast<std::size_t>(idx)]});
        }
    }
}

void CnnView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;

    DrawRectangleLinesEx(m_bounds, 1.0f, theme.panelBorder);

    const float pad = 12.0f;
    float y = m_bounds.y + pad;

    drawText(theme, "CNN layers and activations",
             static_cast<int>(m_bounds.x + pad),
             static_cast<int>(y), 23, theme.textPrimary);
    m_detailed = tensorviz::drawModeToggle(
        Rectangle{m_bounds.x + m_bounds.width - pad - 190.0f, y + 1.0f,
                  190.0f, 30.0f},
        m_detailed, theme);
    y += 36.0f;

    if (!m_hasHeatmap && !m_hasWdl && !m_hasPolicy) {
        drawText(theme, "No CNN snapshot.",
                 static_cast<int>(m_bounds.x + pad),
                 static_cast<int>(y), 17, theme.textMuted);
        drawText(theme, "Set model.lc0_cnn to a converted .bin file.",
                 static_cast<int>(m_bounds.x + pad),
                 static_cast<int>(y + 24.0f), 16, theme.textDim);
        return;
    }

    const float pipelineH = m_bounds.y + m_bounds.height - y - pad;
    Rectangle pipeline{m_bounds.x + pad, y, m_bounds.width - 2.0f * pad, pipelineH};
    if (m_detailed) {
        drawLayerPipeline(m_layers, pipeline, m_hasWdl, m_wdl, m_valueScalar,
                          m_topPolicy, m_selectedLayer, theme);
    } else {
        drawCnnAbstract(m_layers, m_blockCount, m_hasWdl, m_wdl,
                        m_valueScalar, m_topPolicy, pipeline, theme);
    }
}

}  // namespace cnnv::viz
