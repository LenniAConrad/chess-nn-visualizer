#pragma once

/**
 * @file Lc0CnnEncoder.h
 * @brief LC0 classical 112-plane board encoder.
 */

#include <cstddef>
#include <vector>

namespace cnnv::chess {
class Position;
}

namespace cnnv::nn::lc0_cnn {

/**
 * @brief Encodes a position into LC0's classical 112-plane input tensor.
 *
 * Output layout is channel-major `float[112 * 64]`. Each plane uses the same
 * a1..h1, a2..h2, ..., a8..h8 indexing as the chess core. Inputs are oriented
 * from the side-to-move perspective, so black-to-move positions are rank
 * mirrored.
 *
 * Planes:
 * - 0..103: eight repeated history blocks of 12 piece planes plus repetition.
 * - 104..107: castling rights, ordered we-Q, we-K, they-Q, they-K.
 * - 108: black-to-move indicator.
 * - 109: halfmove clock constant plane.
 * - 110: reserved zero plane.
 * - 111: edge plane filled with 1.0.
 */
class Encoder {
   public:
    /** @brief Number of LC0 input planes. */
    static constexpr int kPlanes = 112;

    /** @brief Number of squares per plane. */
    static constexpr int kSquares = 64;

    /** @brief Number of repeated history blocks. */
    static constexpr int kHistoryBlocks = 8;

    /** @brief Piece plus repetition planes per history block. */
    static constexpr int kPlanesPerBoard = 13;

    /** @brief First auxiliary plane index. */
    static constexpr int kAuxBase = kHistoryBlocks * kPlanesPerBoard;  // 104

    /** @brief Flat input tensor length. */
    static constexpr std::size_t kInputSize =
        static_cast<std::size_t>(kPlanes) * kSquares;

    /**
     * @brief Encodes a position as LC0 input planes.
     */
    static std::vector<float> encode(const cnnv::chess::Position& pos);
};

}  // namespace cnnv::nn::lc0_cnn
