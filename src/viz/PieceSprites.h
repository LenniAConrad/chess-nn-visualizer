#pragma once

/**
 * @file PieceSprites.h
 * @brief raylib texture owner for chess piece SVG sprites.
 */

#include "chess/Piece.h"

#include <raylib.h>

#include <array>
#include <string>

namespace cnnv::viz {

/**
 * @brief Owns the twelve piece textures loaded from `assets/pieces`.
 *
 * Construct only after raylib has initialized a window because `LoadTexture`
 * needs a graphics context. Destroy before `CloseWindow()`.
 */
class PieceSprites {
public:
    /**
     * @brief Loads piece textures from an asset directory.
     */
    explicit PieceSprites(const std::string& assetDir = "assets/pieces");

    /**
     * @brief Unloads loaded textures.
     */
    ~PieceSprites();

    PieceSprites(const PieceSprites&) = delete;
    PieceSprites& operator=(const PieceSprites&) = delete;

    /**
     * @brief Returns the texture for a piece.
     * @return A blank zeroed texture when `p` is none or failed to load.
     */
    const Texture2D& textureFor(cnnv::chess::Piece p) const;

    /**
     * @brief True when every piece texture loaded successfully.
     */
    bool allLoaded() const noexcept { return m_loaded; }

private:
    std::array<Texture2D, 12> m_textures{};
    Texture2D m_blank{};
    bool m_loaded = false;
};

}  // namespace cnnv::viz
