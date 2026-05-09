#pragma once

/**
 * @file NnueFeatureEncoder.h
 * @brief Half-KP feature encoder used by the NNUE network.
 */

#include "chess/Bitboard.h"
#include "chess/Piece.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cnnv::chess {
class Position;
}

namespace cnnv::nn::nnue {

/**
 * @brief Encodes chess positions into NNUE half-KP sparse feature indices.
 *
 * One feature is emitted for each `(perspective king square, piece plane, piece
 * square)` tuple:
 *
 * `feature = (kingSq * 10 + plane) * 64 + pieceSq`
 *
 * Planes are own `{P,N,B,R,Q}` followed by enemy `{P,N,B,R,Q}`. Kings are not
 * emitted as piece features because the perspective king square selects the
 * bucket. Black perspective flips ranks with `square ^ 56`.
 */
class FeatureEncoder {
   public:
    /** @brief Number of board squares. */
    static constexpr int kSquares = 64;

    /** @brief Number of non-king piece planes per perspective. */
    static constexpr int kPiecePlanes = 10;

    /** @brief Total number of half-KP sparse features. */
    static constexpr int kFeatureCount = kSquares * kPiecePlanes * kSquares;

    /** @brief Maximum active non-king piece features in one legal position. */
    static constexpr int kMaxActiveFeatures = 30;

    /** @name Plane Indices */
    ///@{
    static constexpr int kOwnPawn   = 0;
    static constexpr int kOwnKnight = 1;
    static constexpr int kOwnBishop = 2;
    static constexpr int kOwnRook   = 3;
    static constexpr int kOwnQueen  = 4;
    static constexpr int kEnemyPawn   = 5;
    static constexpr int kEnemyKnight = 6;
    static constexpr int kEnemyBishop = 7;
    static constexpr int kEnemyRook   = 8;
    static constexpr int kEnemyQueen  = 9;
    ///@}

    /**
     * @brief Collects all active feature indices for one perspective.
     * @param pos Position to encode.
     * @param whitePerspective True for white perspective, false for black.
     * @return Feature indices, one per non-king piece.
     */
    static std::vector<int> activeFeatures(const cnnv::chess::Position& pos,
                                           bool whitePerspective);

    /**
     * @brief Orients a square for one NNUE perspective.
     */
    static int orientSquare(int square, bool whitePerspective) noexcept {
        return whitePerspective ? square : (square ^ 56);
    }

    /**
     * @brief Maps a piece to its perspective-relative feature plane.
     * @return Plane index 0..9, or -1 for empty squares and kings.
     */
    static int piecePlane(cnnv::chess::Piece piece,
                          bool whitePerspective) noexcept;

    /**
     * @brief Packs king square, plane, and piece square into one feature index.
     */
    static int encodeFeature(int kingSquare, int piecePlane,
                             int pieceSquare) noexcept {
        return (kingSquare * kPiecePlanes + piecePlane) * kSquares +
               pieceSquare;
    }

    /**
     * @brief Returns the oriented king square for one perspective.
     */
    static int perspectiveKingSquare(const cnnv::chess::Position& pos,
                                     bool whitePerspective);
};

}  // namespace cnnv::nn::nnue
