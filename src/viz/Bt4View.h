#pragma once

/**
 * @file Bt4View.h
 * @brief Activation panel for BT4-style token-transformer snapshots.
 */

#include "viz/IActivationView.h"

#include <raylib.h>

#include <array>
#include <string>
#include <vector>

namespace cnnv::viz {

/**
 * @brief Renders BT4 token, attention, FFN, policy, and value activations.
 */
class Bt4View : public IActivationView {
public:
    Bt4View() = default;

    /** @brief Copies relevant BT4 tensors from the latest snapshot. */
    void update(const cnnv::nn::ActivationSnapshot& snap) override;

    /** @brief Sets the panel bounds. */
    void setBounds(Rectangle r) override { m_bounds = r; }

    /** @brief Draws the panel. */
    void draw(const Theme& theme = defaultTheme()) const override;

    /** @brief Short architecture label. */
    std::string name() const override { return "BT4"; }

    /**
     * @brief Summary statistics and raw values for one displayed tensor.
     */
    struct LayerStat {
        std::string name;
        std::string shape;
        std::vector<std::size_t> shapeDims;
        float mean = 0.0f;
        float rms = 0.0f;
        float min = 0.0f;
        float max = 0.0f;
        std::vector<float> values;
    };

    /**
     * @brief Per-transformer-block summary used by the overview mode.
     */
    struct BlockStat {
        int index = 0;
        float attentionFocus = 0.0f;
        LayerStat attentionOut;
        LayerStat ffn;
        LayerStat output;
        bool have = false;
    };

private:
    Rectangle m_bounds{0, 0, 0, 0};
    LayerStat m_inputPlanes;
    LayerStat m_tokenFeatures;
    LayerStat m_embedding;
    LayerStat m_finalTokens;
    LayerStat m_policy;
    LayerStat m_valueLogits;
    std::vector<BlockStat> m_blocks;
    std::vector<float> m_attention;
    std::vector<float> m_tokenEnergy;
    std::vector<float> m_valueSalience;
    std::array<float, 3> m_wdl{0.0f, 0.0f, 0.0f};
    float m_valueScalar = 0.0f;
    bool m_hasOverview = false;
    bool m_hasAttention = false;
    bool m_hasTokenEnergy = false;
    bool m_hasValueSalience = false;
    bool m_hasPolicy = false;
    bool m_hasWdl = false;
    mutable bool m_detailed = false;
    mutable int m_selectedLayer = -1;
};

}  // namespace cnnv::viz
