#pragma once

/**
 * @file NnueView.h
 * @brief Activation panel for NNUE snapshots.
 *
 * Transpiled from the chess-rtk Swing "engine-lab" NNUE views
 * (application.gui.workbench.network.Nnue*View) so the C++/raylib panel renders
 * the same five modes with faithful layouts:
 *   - Overview (ABSTRACT): score summary band + signed contributor columns.
 *   - Trace (DETAILED): five-column wired node graph
 *     (feature -> accumulator -> clipped -> contribution -> output) plus the
 *     layer-stack ribbon and selected-slot footer.
 *   - All (RAW): dense feature x slot heatmap matrix plus summary vector rows.
 *   - Atlas (ATLAS): whole pixel-plane learned-weight overview + selected-slot
 *     8x8 plane board + slot explanation.
 *   - Diagram (DIAGRAM): static five-box architecture schematic.
 *
 * Mode switching is internal (a self-drawn 5-segment selector). The view reads
 * the C++ NNUE backend keys (cnnv::nn::nnue::snapshot_keys); where crtk relies
 * on per-feature impact or a position FEN that the C++ backend does not emit,
 * the view derives equivalents from the available tensors (see NnueView.cpp).
 */

#include "viz/IActivationView.h"

#include <raylib.h>

#include <string>
#include <vector>

namespace cnnv::viz {

/**
 * @brief Renders NNUE accumulators, feature activity, output contribution, and
 * the learned-weight atlas across five faithful crtk-style modes.
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

    // ---- Cached snapshot tensors (copied so draw() stays const/cheap) ----
    std::vector<float> m_accumWhite;   ///< raw accumulator, white perspective.
    std::vector<float> m_accumBlack;   ///< raw accumulator, black perspective.
    std::vector<float> m_clippedUs;    ///< post-ReLU activations, side to move.
    std::vector<float> m_clippedThem;  ///< post-ReLU activations, opponent.
    std::vector<float> m_featuresUs;   ///< active feature ids, side to move.
    std::vector<float> m_featuresThem; ///< active feature ids, opponent.
    std::vector<float> m_featureWeightsUs;    ///< [activeUs, hidden] weight rows.
    std::vector<float> m_featureWeightsThem;  ///< [activeThem, hidden] rows.
    std::vector<float> m_outputWeightsUs;     ///< output-head weights, us slots.
    std::vector<float> m_outputWeightsThem;   ///< output-head weights, them.
    std::vector<float> m_outputContributionUs;    ///< clipped*weight*scale, us.
    std::vector<float> m_outputContributionThem;  ///< clipped*weight*scale, them.

    // Atlas tensors (position-independent learned-weight footprint).
    std::vector<float> m_atlasWeights;  ///< [hidden, planes, squares].
    std::vector<float> m_atlasKing;     ///< [hidden, squares].
    std::vector<float> m_atlasOutput;   ///< [hidden] output-weight per slot.
    int m_atlasPlanes = 0;
    int m_atlasSquares = 0;

    // Derived per-feature signed impact (crtk emits nnue.features.us.impact;
    // the C++ backend does not, so we approximate it as the sum of that
    // feature's output contribution over the hidden layer).
    std::vector<float> m_impactUs;
    std::vector<float> m_impactThem;

    // Total per-slot output contribution (us + them); crtk's
    // nnue.output.contribution.total, which the C++ backend does not emit.
    std::vector<float> m_contributionTotal;

    float m_centipawns = 0.0f;
    int m_activeFeaturesUs = 0;
    int m_activeFeaturesThem = 0;
    int m_hidden = 0;
    bool m_hasData = false;
    bool m_clippedUsFromWhite = true;  ///< resolved us/them -> white/black map.

    // ---- Mode + selection state (mutable so draw() stays const) ----
    /// 0=Overview, 1=Trace, 2=All, 3=Atlas, 4=Diagram.
    mutable int m_mode = 0;
    mutable int m_selectedFeature = -1;   ///< sparse feature id, or -1.
    mutable int m_selectedSlot = -1;      ///< trace visible-slot index, or -1.
    mutable int m_atlasSelected = -1;     ///< atlas hidden slot, or -1.
    mutable int m_atlasPlane = 0;         ///< selected atlas piece plane.
};

}  // namespace cnnv::viz
