#pragma once

/**
 * @file ClassicalView.h
 * @brief Activation-panel view for the handcrafted (classical) evaluator.
 *
 * Unlike the neural views this one consumes no activation snapshot. It is fed
 * the live position and renders the 11-term centipawn breakdown, a win/draw/loss
 * bar, a game-phase bar, and (in detailed mode) the six piece-square-table
 * heatmaps. It mirrors chess-rtk's `ClassicalView`.
 */

#include "chess/Position.h"
#include "chess/eval/Classical.h"
#include "viz/IActivationView.h"

#include <cstdint>

namespace cnnv::viz {

/**
 * @brief Renders the classical evaluator's term breakdown and PST tables.
 */
class ClassicalView : public IActivationView {
   public:
    /** @brief No-op: the classical view is model-free (uses setPosition). */
    void update(const cnnv::nn::ActivationSnapshot& snap) override;

    /** @brief Sets the panel bounds. */
    void setBounds(Rectangle r) override { m_bounds = r; }

    /** @brief Renders the breakdown, bars, and (detailed) PST heatmaps. */
    void draw(const Theme& theme = defaultTheme()) const override;

    /** @brief Short label for the architecture switcher. */
    std::string name() const override { return "Classical"; }

    /**
     * @brief Feeds the live position and recomputes when its hash changes.
     */
    void setPosition(const cnnv::chess::Position& pos);

   private:
    Rectangle m_bounds{0, 0, 0, 0};
    bool m_hasData = false;
    std::uint64_t m_hash = 0;
    cnnv::chess::eval::Breakdown m_breakdown;
    cnnv::chess::eval::WdlTriplet m_wdl;

    /** @brief View mode: 0 = abstract, 1 = detailed (PSTs), 2 = diagram. */
    mutable int m_mode = 0;
};

}  // namespace cnnv::viz
