#pragma once

/**
 * @file Position.h
 * @brief Mutable chess board state, move make/unmake, and game-status helpers.
 */

#include "chess/Bitboard.h"
#include "chess/Move.h"
#include "chess/Piece.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cnnv::chess {

/**
 * @brief Castling-right flags stored as a four-bit mask.
 */
enum CastlingRight : std::uint8_t {
    NoCastling   = 0,
    WhiteKing    = 1,
    WhiteQueen   = 2,
    BlackKing    = 4,
    BlackQueen   = 8,
    AnyCastling  = WhiteKing | WhiteQueen | BlackKing | BlackQueen,
};

/**
 * @brief Irrecoverable state saved before each `Position::make()` call.
 *
 * `unmake()` consumes the top state object to restore fields that cannot be
 * reconstructed from the move alone: captured piece, castling rights,
 * en-passant target, halfmove clock, and the move that was made.
 */
struct StateInfo {
    Piece capturedPiece = kNoPiece;
    std::uint8_t prevCastlingRights = NoCastling;
    Square prevEpSquare = Square::None;
    int prevHalfmoveClock = 0;
    Move moveMade;
};

/**
 * @brief Complete mutable chess position with bitboards, mailbox, and history.
 *
 * `Position` owns the live board state used by both the game loop and neural
 * encoders. It stores redundant piece bitboards plus a mailbox array so attack
 * generation and UI piece lookups can both be efficient. Legal-play callers
 * should mutate positions through `make()` and `unmake()`; setup tools such as
 * FEN parsing and the editor may use the direct setters and then call
 * `anchorHashHistory()`.
 */
class Position {
public:
    Position();

    /**
     * @brief Resets to an empty board with default counters.
     *
     * After this call the FEN is `8/8/8/8/8/8/8/8 w - - 0 1`.
     */
    void clear();

    /**
     * @brief Resets to the standard chess starting position.
     */
    void setStartpos();

    /** @name Accessors */
    ///@{
    /**
     * @brief Returns the piece on a square.
     * @param sq Square to inspect.
     */
    Piece pieceAt(Square sq) const noexcept { return m_board[squareIndex(sq)]; }

    /**
     * @brief Returns the piece at a raw square index.
     * @param idx Square index in the range 0..63.
     */
    Piece pieceAt(int idx)  const noexcept { return m_board[idx]; }

    /**
     * @brief Returns the bitboard for pieces of one color and type.
     */
    Bitboard pieceBitboard(Color c, PieceType t) const noexcept {
        return m_pieces[static_cast<int>(c)][static_cast<int>(t) - 1];
    }

    /**
     * @brief Returns occupancy for all pieces of one color.
     */
    Bitboard colorBitboard(Color c) const noexcept {
        return m_byColor[static_cast<int>(c)];
    }

    /**
     * @brief Returns occupancy for all pieces.
     */
    Bitboard occupied() const noexcept { return m_occupied; }

    /**
     * @brief Side whose turn it is to move.
     */
    Color sideToMove() const noexcept { return m_sideToMove; }

    /**
     * @brief Current castling-right bit mask.
     */
    std::uint8_t castlingRights() const noexcept { return m_castlingRights; }

    /**
     * @brief Current en-passant target square, or `Square::None`.
     */
    Square epSquare() const noexcept { return m_epSquare; }

    /**
     * @brief Halfmove clock used for the fifty-move rule.
     */
    int halfmoveClock() const noexcept { return m_halfmoveClock; }

    /**
     * @brief Fullmove number shown in FEN/PGN.
     */
    int fullmoveNumber() const noexcept { return m_fullmoveNumber; }
    ///@}

    /** @name Setup Mutators */
    ///@{
    /**
     * @brief Places a piece on a square, replacing any existing piece.
     * @param sq Destination square.
     * @param p Piece to place.
     *
     * Used by FEN and editor setup. The live game loop should use `make()`.
     */
    void placePiece(Square sq, Piece p) noexcept;

    /**
     * @brief Removes any piece from a square.
     * @param sq Square to clear.
     */
    void removePiece(Square sq) noexcept;

    /** @brief Sets the side to move. */
    void setSideToMove(Color c) noexcept { m_sideToMove = c; }

    /** @brief Replaces the castling-right bit mask. */
    void setCastlingRights(std::uint8_t r) noexcept { m_castlingRights = r; }

    /** @brief Sets the en-passant target square, or `Square::None`. */
    void setEpSquare(Square sq) noexcept { m_epSquare = sq; }

    /** @brief Sets the fifty-move halfmove clock. */
    void setHalfmoveClock(int n) noexcept { m_halfmoveClock = n; }

    /** @brief Sets the FEN fullmove number. */
    void setFullmoveNumber(int n) noexcept { m_fullmoveNumber = n; }
    ///@}

    /** @name Move Make/Unmake */
    ///@{
    /**
     * @brief Applies a move and pushes enough state for `unmake()`.
     * @param m Legal move to apply.
     *
     * The caller must ensure `m` is legal for this position. Pseudo-legal
     * moves that leave the king in check are filtered by `MoveGenerator`
     * before they reach this method.
     */
    void make(Move m);

    /**
     * @brief Reverses the most recent `make()` call.
     */
    void unmake();
    ///@}

    /** @name Attack And Check Helpers */
    ///@{
    /**
     * @brief Tests whether a square is attacked by one side.
     * @param sq Target square.
     * @param bySide Attacking side.
     * @return True if any `bySide` piece attacks `sq`.
     */
    bool isSquareAttacked(Square sq, Color bySide) const noexcept;

    /**
     * @brief Tests whether the side to move is in check.
     */
    bool inCheck() const noexcept;

    /**
     * @brief Finds the king square for a side.
     * @param c Color whose king should be found.
     * @return Square containing that king.
     *
     * Assumes legal positions with exactly one king of that color.
     */
    Square kingSquare(Color c) const noexcept;
    ///@}

    /** @name Game Status Helpers */
    ///@{
    /**
     * @brief Checks whether the side to move has at least one legal move.
     */
    bool hasLegalMoves() const;

    /**
     * @brief Checks whether the current player is checkmated.
     */
    bool isCheckmate() const;

    /**
     * @brief Checks whether the current player is stalemated.
     */
    bool isStalemate() const;

    /**
     * @brief Detects draw by insufficient mating material.
     */
    bool isInsufficientMaterial() const noexcept;

    /**
     * @brief Detects draw eligibility by the fifty-move rule.
     */
    bool isFiftyMoveDraw() const noexcept { return m_halfmoveClock >= 100; }

    /**
     * @brief Detects threefold repetition in the anchored hash history.
     */
    bool isThreefoldRepetition() const noexcept;
    ///@}

    /**
     * @brief Computes the Zobrist hash of the current state.
     * @return Deterministic 64-bit hash including board, side, castling, and
     * en-passant state.
     *
     * The hash is recomputed from scratch. That is cheap enough for the
     * visualizer and keeps the move path simple.
     */
    std::uint64_t hash() const noexcept;

    /**
     * @brief Re-anchors repetition history at the current state.
     *
     * Call after assembling a position via setup mutators or FEN parsing so
     * threefold detection starts from the new root.
     */
    void anchorHashHistory();

private:
    /** @brief Twelve piece bitboards indexed as `[color][pieceType - 1]`. */
    std::array<std::array<Bitboard, 6>, 2> m_pieces{};
    std::array<Bitboard, 2> m_byColor{};
    Bitboard m_occupied = 0;

    /**
     * @brief Mailbox board for O(1) `pieceAt()` lookups.
     *
     * This duplicates the bitboard state but is small and simplifies UI and
     * FEN code that need direct square access.
     */
    std::array<Piece, 64> m_board{};

    Color m_sideToMove = Color::White;
    std::uint8_t m_castlingRights = NoCastling;
    Square m_epSquare = Square::None;
    int m_halfmoveClock = 0;
    int m_fullmoveNumber = 1;

    std::vector<StateInfo> m_history;
    std::vector<std::uint64_t> m_hashHistory;

    /**
     * @brief Rebuilds aggregate color and occupied bitboards from pieces.
     */
    void rebuildOccupancy() noexcept;
};

}  // namespace cnnv::chess
