#pragma once

/**
 * @file Bt4View.h
 * @brief Activation panel for the LC0 BT4 token-transformer.
 *
 * Transpiled from chess-rtk's Bt4View (Java/Swing) to C++/raylib. The view
 * mirrors all five crtk modes behind a self-drawn 5-segment switcher:
 *   - Overview (ABSTRACT): pipeline + per-block attn/ffn strip + WDL +
 *     token-energy board + top-policy bars.
 *   - Trace (DETAILED): block selector + 4x8 head grid of 8x8 received-energy
 *     thumbnails + the selected head's 64x64 attention matrix + an on-board
 *     two-triangle from/to overlay + a head readout.
 *   - All (RAW): the dense block x head grid of 64x64 attention matrices
 *     stacked over 8x8 mean-received energy, via sqrt-gamma heatmaps.
 *   - Atlas (ATLAS): block x head focus fingerprint + selected-head/token board
 *     footprints + the selected head's 64x64 attention matrix.
 *   - Diagram (DIAGRAM): static architecture schematic.
 *
 * Boards are drawn with theme square colours; the trace attention board can
 * overlay live-position pieces supplied by `App`. Token-square <-> board-square
 * mapping is identity here.
 * Head count is derived from each block's attention tensor rather than a
 * compile-time constant.
 */

#include "viz/IActivationView.h"

#include <raylib.h>

#include <array>
#include <string>
#include <vector>

namespace cnnv::chess {
class Position;
}

namespace cnnv::viz {

class PieceSprites;

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

    /** @brief Provides the live board position for attention-board pieces. */
    void setPosition(const cnnv::chess::Position* position) noexcept {
        m_position = position;
    }

    /** @brief Provides piece sprites used by attention-board overlays. */
    void setSprites(const PieceSprites* sprites) noexcept { m_sprites = sprites; }

    /** @brief Draws the panel. */
    void draw(const Theme& theme = defaultTheme()) const override;

    /** @brief Short architecture label. */
    std::string name() const override { return "BT4"; }

private:
    /** @brief One transformer block's cached attention + FFN tensors. */
    struct Block {
        std::vector<float> attention;  ///< [heads, 64, 64] post-softmax.
        std::vector<float> ffn;        ///< [64, D] feed-forward output.
        int heads = 0;                 ///< Derived from attention.size()/(64*64).
        float attentionFocus = 0.0f;   ///< Mean attention magnitude (block strip).
        float ffnRms = 0.0f;           ///< FFN RMS (block strip).
        bool have = false;
    };

    Rectangle m_bounds{0, 0, 0, 0};
    const cnnv::chess::Position* m_position = nullptr;
    const PieceSprites* m_sprites = nullptr;

    // Cached snapshot tensors.
    std::vector<Block> m_blocks;
    std::vector<float> m_inputPlanes;
    std::vector<float> m_tokenFeatures;
    std::vector<float> m_embedding;
    std::vector<float> m_finalTokens;
    std::vector<float> m_tokenEnergy;     ///< [64] final token rms.
    std::vector<float> m_boardSalience;   ///< [64] positive board salience.
    std::vector<float> m_policy;          ///< [1858] policy logits.
    std::array<float, 3> m_wdl{0.0f, 0.0f, 0.0f};
    float m_valueScalar = 0.0f;
    bool m_hasData = false;
    bool m_hasWdl = false;
    bool m_hasPolicy = false;
    bool m_hasTokenEnergy = false;

    // Mutable selection cursors (the view is drawn from a const method).
    mutable int m_mode = 0;            ///< 0=Overview 1=Trace 2=All 3=Atlas 4=Diagram.
    mutable int m_selectedBlock = 0;   ///< 0..blocks-1.
    mutable int m_selectedHead = 0;    ///< 0..heads-1.
    mutable int m_selectedSquare = -1; ///< -1 = none, else 0..63.
};

}  // namespace cnnv::viz
