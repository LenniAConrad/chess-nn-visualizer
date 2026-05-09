#pragma once

/**
 * @file CnnView.h
 * @brief Activation panel for LC0 CNN snapshots.
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
    std::vector<float> m_heatmap;
    std::vector<float> m_policyLogits;
    std::vector<std::pair<int, float>> m_topPolicy;
    std::vector<LayerStat> m_layers;
    float m_wdl[3] = {0.0f, 0.0f, 0.0f};
    float m_valueScalar = 0.0f;
    int m_blockCount = 0;
    bool m_hasHeatmap = false;
    bool m_hasWdl = false;
    bool m_hasPolicy = false;
    mutable bool m_detailed = false;
    mutable int m_selectedLayer = -1;
};

}  // namespace cnnv::viz
