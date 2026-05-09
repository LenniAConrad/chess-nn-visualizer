#pragma once

/**
 * @file PositionEditor.h
 * @brief Mutable setup-mode wrapper and validation rules for custom positions.
 */

#include "chess/Bitboard.h"
#include "chess/Piece.h"
#include "chess/Position.h"

#include <optional>
#include <string>
#include <vector>

namespace cnnv::chess {

/**
 * @brief Editor-mode wrapper around `Position` for composing custom boards.
 *
 * The play loop owns a `Position` with make/unmake history. This wrapper lets
 * setup UI mutate a separate board safely, then build a fresh ready-to-play
 * position. Castling rights are exposed as discrete booleans, and all edits
 * are allowed even on illegal boards; legality is reported by
 * `validateForEditor()`.
 */
class PositionEditor {
public:
    /**
     * @brief Castling side used by the editor controls.
     */
    enum class CastleSide { Kingside, Queenside };

    /**
     * @brief Creates an empty editable position with default counters.
     */
    PositionEditor();

    /**
     * @brief Seeds the editor from an existing position.
     * @param seed Position to copy into the editable buffer.
     */
    explicit PositionEditor(const Position& seed);

    /** @name Piece Placement */
    ///@{
    /** @brief Removes every piece from the board. */
    void clearBoard() noexcept;

    /** @brief Resets the editable position to standard chess startpos. */
    void resetToStartpos() noexcept;

    /** @brief Places or replaces a piece on a square. */
    void placePiece(Square sq, Piece p) noexcept;

    /** @brief Removes any piece from a square. */
    void removePiece(Square sq) noexcept;

    /** @brief Reads the piece currently on a square. */
    Piece pieceAt(Square sq) const noexcept { return m_position.pieceAt(sq); }
    ///@}

    /** @name Side State */
    ///@{
    /** @brief Sets which color moves next. */
    void setSideToMove(Color c) noexcept { m_position.setSideToMove(c); }

    /** @brief Returns which color moves next. */
    Color sideToMove() const noexcept { return m_position.sideToMove(); }
    ///@}

    /** @name Castling Rights */
    ///@{
    /** @brief Enables or disables one castling right. */
    void setCastlingRight(Color c, CastleSide side, bool on) noexcept;

    /** @brief Tests whether one castling right is enabled. */
    bool castlingRight(Color c, CastleSide side) const noexcept;

    /** @brief Returns the raw castling-right mask. */
    std::uint8_t castlingMask() const noexcept { return m_position.castlingRights(); }
    ///@}

    /** @name En-Passant Target */
    ///@{
    /**
     * @brief Sets the en-passant target square.
     * @param sq Target square, or `std::nullopt` when no en-passant capture is
     * available.
     *
     * The target is the square behind a pawn that just advanced two ranks: the
     * square an opposing pawn would occupy after capturing en passant.
     */
    void setEnPassantSquare(std::optional<Square> sq) noexcept;

    /**
     * @brief Returns the current en-passant target, if any.
     */
    std::optional<Square> enPassantSquare() const noexcept;
    ///@}

    /** @name Counters */
    ///@{
    /** @brief Sets the halfmove clock, clamped to zero or above. */
    void setHalfmoveClock(int n) noexcept { m_position.setHalfmoveClock(n < 0 ? 0 : n); }

    /** @brief Returns the halfmove clock. */
    int  halfmoveClock() const noexcept { return m_position.halfmoveClock(); }

    /** @brief Sets the fullmove number, clamped to one or above. */
    void setFullmoveNumber(int n) noexcept { m_position.setFullmoveNumber(n < 1 ? 1 : n); }

    /** @brief Returns the fullmove number. */
    int  fullmoveNumber() const noexcept { return m_position.fullmoveNumber(); }
    ///@}

    /**
     * @brief Returns a read-only view of the in-progress position.
     */
    const Position& position() const noexcept { return m_position; }

    /**
     * @brief Builds a fresh ready-to-play `Position`.
     * @return Position with hash history anchored at the editor state.
     */
    Position build() const;

    /**
     * @brief Formats the in-progress position as FEN.
     */
    std::string fen() const;

private:
    Position m_position;
};

/**
 * @brief Result of a legality sweep over an editor-built position.
 */
struct EditorValidation {
    /**
     * @brief True when `issues` is empty.
     */
    bool legal = false;

    /**
     * @brief Human-readable validation failures shown in the editor panel.
     */
    std::vector<std::string> issues;
};

/**
 * @brief Runs editor-specific legality checks over a custom position.
 * @param pos Position to validate.
 * @return Validation result with every detected issue.
 *
 * Checks include king count, illegal pawn ranks, impossible check state,
 * castling-right consistency, and en-passant target consistency.
 */
EditorValidation validateForEditor(const Position& pos);

}  // namespace cnnv::chess
