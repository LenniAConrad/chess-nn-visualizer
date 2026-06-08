#pragma once

/**
 * @file Classical.h
 * @brief Native port of chess-rtk's handcrafted ("classical") centipawn
 *        evaluator, exposing a term-by-term breakdown for visualization.
 *
 * This mirrors `chess-rtk/src/chess/classical/Wdl.java`. It produces the same
 * 11-term white-perspective centipawn decomposition, a win/draw/loss triplet,
 * and the per-piece-type piece-square tables the Classical view renders. It is
 * model-free: it depends only on the chess core, not on any neural network.
 */

#include "chess/Position.h"

#include <array>

namespace cnnv::chess::eval {

/**
 * @brief White-perspective centipawn breakdown of the classical evaluator.
 *
 * Every term is in centipawns from White's point of view (positive favors
 * White). The eleven terms sum exactly to `whiteTotal()`. Field order matches
 * the chess-rtk `Wdl.Breakdown` record so the view can render them in the same
 * order as the Java app.
 */
struct Breakdown {
    /** @brief True when White is the side to move in the evaluated position. */
    bool whiteToMove = true;

    /** @brief Game phase in [0,1]: 1 = opening, 0 = endgame. */
    double phase = 1.0;

    int material = 0;       ///< Material balance (White minus Black).
    int pieceSquare = 0;    ///< Summed piece-square-table bonuses.
    int bishopPair = 0;     ///< Bishop-pair bonus (+/-30 per pair).
    int pawnStructure = 0;  ///< Doubled/isolated/passed pawn terms.
    int rookFile = 0;       ///< Rook open/semi-open file bonuses.
    int kingSafety = 0;     ///< King PST + shelter/storm - enemy pressure.
    int activity = 0;       ///< Mobility plus per-piece activity bonuses.
    int threats = 0;        ///< Hanging/weak pieces and safe pawn threats.
    int space = 0;          ///< Safe central space (opening only).
    int tempo = 0;          ///< Side-to-move tempo (+/-8).
    int checkPenalty = 0;   ///< In-check penalty (+/-35).

    /** @brief Sum of the eleven terms (White perspective). */
    int whiteTotal() const noexcept {
        return material + pieceSquare + bishopPair + pawnStructure + rookFile +
               kingSafety + activity + threats + space + tempo + checkPenalty;
    }

    /** @brief Total from the side-to-move perspective. */
    int stmTotal() const noexcept {
        return whiteToMove ? whiteTotal() : -whiteTotal();
    }
};

/**
 * @brief Win/draw/loss probabilities scaled to integers summing to 1000.
 *
 * From the side-to-move perspective, matching chess-rtk `Wdl.evaluate`.
 */
struct WdlTriplet {
    int win = 0;
    int draw = 0;
    int loss = 0;
};

/**
 * @brief Computes the eleven-term white-perspective breakdown for a position.
 */
Breakdown evaluateWhiteBreakdown(const Position& pos);

/**
 * @brief Maps a position to a win/draw/loss triplet (side-to-move perspective).
 * @param terminalAware When true, applies terminal short-circuits
 *        (checkmate/stalemate/insufficient material) like the Java evaluator.
 */
WdlTriplet evaluateWdl(const Position& pos, bool terminalAware);

/**
 * @brief Returns the 64-entry piece-square table for a piece type.
 * @param pieceType 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King.
 * @return Centipawns, White perspective, indexed 0 = a8 .. 63 = h1 to match the
 *         heatmap layout. The king returns the opening table; unknown types
 *         return a zeroed array.
 */
std::array<int, 64> pieceSquareTable(int pieceType);

/**
 * @brief Builds an occupied-square heatmap from each piece's own PST.
 * @return Intensities indexed by raw board square (a1=0), normalized to [0,1].
 *
 * Empty squares are zero. White pieces use their table in normal orientation;
 * black pieces use the mirrored table square, matching the evaluator's
 * piece-square term.
 */
std::array<float, 64> pieceSquareHeatmap(const Position& pos);

}  // namespace cnnv::chess::eval
