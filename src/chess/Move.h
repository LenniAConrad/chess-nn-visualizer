#pragma once

/**
 * @file Move.h
 * @brief Compact chess move representation and UCI conversion helpers.
 */

#include "chess/Bitboard.h"
#include "chess/Piece.h"

#include <cstdint>
#include <string>

namespace cnnv::chess {

/**
 * @brief Compact 16-bit move encoded for fast storage and comparison.
 *
 * The layout mirrors chess-rtk `Move.java`, which allows direct cross-checks
 * against reference dumps:
 * - bits 0..5: from-square (0..63)
 * - bits 6..11: to-square (0..63)
 * - bits 12..14: promotion code (0=none, 1=N, 2=B, 3=R, 4=Q)
 * - bit 15: unused
 *
 * Castling and en-passant are inferred from the live position when a move is
 * generated or made, so they do not need dedicated flag bits.
 */
class Move {
public:
    /**
     * @brief Raw sentinel value used by `Move::none()`.
     */
    static constexpr std::uint16_t kNoMove = 0xFFFF;

    /**
     * @brief Promotion piece encoded in the high move bits.
     */
    enum class Promotion : std::uint8_t {
        None   = 0,
        Knight = 1,
        Bishop = 2,
        Rook   = 3,
        Queen  = 4,
    };

    constexpr Move() = default;

    /**
     * @brief Constructs a move from endpoints and an optional promotion.
     * @param from Origin square.
     * @param to Destination square.
     * @param promo Promotion choice, or `Promotion::None` for normal moves.
     */
    constexpr Move(Square from, Square to,
                   Promotion promo = Promotion::None) noexcept
        : m_value(static_cast<std::uint16_t>(
              squareIndex(from) |
              (squareIndex(to) << 6) |
              (static_cast<int>(promo) << 12))) {}

    /**
     * @brief Creates the invalid/no-move sentinel.
     * @return Move whose raw value is `kNoMove`.
     */
    static constexpr Move none() noexcept {
        Move m;
        m.m_value = kNoMove;
        return m;
    }

    /**
     * @brief Origin square.
     * @return The encoded from-square.
     */
    constexpr Square from() const noexcept {
        return static_cast<Square>(m_value & 0x3F);
    }

    /**
     * @brief Destination square.
     * @return The encoded to-square.
     */
    constexpr Square to() const noexcept {
        return static_cast<Square>((m_value >> 6) & 0x3F);
    }

    /**
     * @brief Promotion piece for this move.
     * @return Promotion code, or `Promotion::None`.
     */
    constexpr Promotion promotion() const noexcept {
        return static_cast<Promotion>((m_value >> 12) & 0x07);
    }

    /**
     * @brief Raw 16-bit encoded move value.
     * @return Encoded value suitable for stable comparison or serialization.
     */
    constexpr std::uint16_t raw() const noexcept { return m_value; }

    /**
     * @brief Tests whether this is the no-move sentinel.
     * @return True when `raw() == kNoMove`.
     */
    constexpr bool isNone() const noexcept { return m_value == kNoMove; }

    constexpr bool operator==(const Move& o) const noexcept {
        return m_value == o.m_value;
    }
    constexpr bool operator!=(const Move& o) const noexcept {
        return m_value != o.m_value;
    }

    /**
     * @brief Converts the move to long-algebraic UCI notation.
     * @return Text such as `e2e4`, `e7e8q`, or an empty string for no-move.
     */
    std::string toUci() const;

    /**
     * @brief Parses UCI notation into a `Move`.
     * @param s Text such as `e2e4` or `e7e8q`.
     * @return Parsed move, or `Move::none()` on malformed input.
     *
     * This converts notation only; it does not validate legality against a
     * particular position.
     */
    static Move parseUci(const std::string& s);

private:
    std::uint16_t m_value = kNoMove;
};

}  // namespace cnnv::chess
