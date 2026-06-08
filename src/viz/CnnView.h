#pragma once

/**
 * @file CnnView.h
 * @brief Activation panel for LC0 CNN snapshots.
 *
 * Transpiled from chess-rtk's CnnView.java (engine-lab network view). Mirrors
 * the five crtk paint modes via a self-drawn 5-segment switcher:
 *   Overview (ABSTRACT) | Trace (DETAILED) | All (RAW) | Atlas | Diagram.
 */

#include "viz/IActivationView.h"

#include <raylib.h>

#include <string>
#include <utility>
#include <vector>

namespace cnnv::viz {

/**
 * @brief Renders LC0 CNN trunk, policy, value, and heatmap activations.
 */
class CnnView : public IActivationView {
public:
    CnnView() = default;

    /** @brief Copies relevant CNN tensors from the latest snapshot. */
    void update(const cnnv::nn::ActivationSnapshot& snap) override;

    /** @brief Sets the panel bounds. */
    void setBounds(Rectangle r) override { m_bounds = r; }

    /** @brief Draws the panel. */
    void draw(const Theme& theme = defaultTheme()) const override;

    /** @brief Short architecture label. */
    std::string name() const override { return "CNN"; }

    /**
     * @brief Summary statistics and raw values for one displayed CNN layer.
     */
    struct LayerStat {
        std::string name;
        std::string shape;
        std::vector<std::size_t> shapeDims;
        float mean = 0.0f;
        float meanAbs = 0.0f;
        float rms = 0.0f;
        float min = 0.0f;
        float max = 0.0f;
        std::vector<float> values;
    };

private:
    Rectangle m_bounds{0, 0, 0, 0};
    std::vector<float> m_finalMap;      ///< cnn.final.relu_mean [64].
    std::vector<float> m_policyLogits;  ///< cnn.policy.logits [P].
    std::vector<std::pair<int, float>> m_topPolicy;
    std::vector<LayerStat> m_layers;
    float m_wdl[3] = {0.0f, 0.0f, 0.0f};
    float m_valueScalar = 0.0f;
    int m_blockCount = 0;
    bool m_hasFinalMap = false;
    bool m_hasWdl = false;
    bool m_hasPolicy = false;

    // Mutable cursors updated during const draw() (mirrors crtk's mutable
    // selection fields that live on the Swing component).
    mutable int m_mode = 0;          ///< 0=Overview 1=Trace 2=All 3=Atlas 4=Diagram.
    mutable int m_selectedLayer = -1;  ///< Trace-mode focused layer.
    mutable int m_rawZoomLayer = -1;   ///< All-mode zoomed spatial layer, -1 = atlas.
};

}  // namespace cnnv::viz
