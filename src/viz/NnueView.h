#pragma once

/**
 * @file NnueView.h
 * @brief Activation panel for NNUE snapshots.
 */

#include "viz/IActivationView.h"

#include <raylib.h>

#include <string>
#include <vector>

namespace cnnv::viz {

/**
 * @brief Renders NNUE accumulators, feature activity, and output contribution.
 *
 * The panel can show a compact overview or a detailed layer/node view. Empty
 * snapshots are handled gracefully by drawing the header and omitting charts.
 */
class NnueView : public IActivationView {
   public:
    NnueView() = default;

    /** @brief Copies relevant NNUE tensors from the latest snapshot. */
    void update(const cnnv::nn::ActivationSnapshot& snap) override;

    /** @brief Sets panel bounds. */
    void setBounds(Rectangle r) override { m_bounds = r; }

    /** @brief Draws the panel. */
    void draw(const Theme& theme = defaultTheme()) const override;

    /** @brief Short architecture label. */
    std::string name() const override { return "NNUE"; }

   private:
    Rectangle m_bounds{0, 0, 0, 0};

    /**
     * @brief Cached vectors copied from the snapshot so `draw()` stays const
     * and cheap.
     */
    std::vector<float> m_accumUs;
    std::vector<float> m_accumThem;
    std::vector<float> m_clippedUs;
    std::vector<float> m_clippedThem;
    std::vector<float> m_featuresUs;
    std::vector<float> m_featuresThem;
    std::vector<float> m_featureBias;
    std::vector<float> m_featureWeightsUs;
    std::vector<float> m_featureWeightsThem;
    std::vector<float> m_outputWeightsUs;
    std::vector<float> m_outputWeightsThem;
    std::vector<float> m_outputContributionUs;
    std::vector<float> m_outputContributionThem;
    std::vector<float> m_outputAffine;
    float m_centipawns = 0.0f;
    int m_activeFeaturesUs = 0;
    int m_activeFeaturesThem = 0;
    bool m_hasData = false;
    mutable bool m_detailed = false;
    mutable int m_selectedLayer = 3;
    mutable int m_selectedNode = -1;
    mutable int m_edgeMode = 0;
};

}  // namespace cnnv::viz
