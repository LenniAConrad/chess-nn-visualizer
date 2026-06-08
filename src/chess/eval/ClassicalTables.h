#pragma once

/**
 * @file ClassicalTables.h
 * @brief Piece-square tables and named magic constants for the classical
 *        evaluator.
 *
 * Every table and constant here is a numerically-exact copy of the chess-rtk
 * Java source `chess/classical/Wdl.java`. The piece-square tables follow the
 * Java internal board convention where index 0 == a8 and index 63 == h1
 * (rank 8 first, file a first). The evaluator (`Classical.cpp`) operates in
 * that same a8=0 coordinate space, remapping the a1=0 chess core at the
 * boundary, so these tables are consumed directly without re-indexing.
 */

#include <array>

namespace cnnv::chess::eval {

/** @brief Total WDL scaling target (UCI "wdl" convention). */
constexpr int kWdlTotal = 1000;

/** @brief Width of the central band that favors draws (centipawns). */
constexpr int kDrawMarginCp = 200;

/** @brief Logistic scale turning centipawns into probabilities (centipawns). */
constexpr double kScaleCp = 170.0;

/** @brief Extra draw mass in low-material endgames. */
constexpr double kEndgameDrawBonus = 0.12;

/** @brief Small tempo term from White's perspective (centipawns). */
constexpr int kTempoCp = 8;

/** @brief Penalty for being in check, applied to the checked side (centipawns). */
constexpr int kInCheckCp = 35;

/** @brief Bishop-pair bonus (centipawns). */
constexpr int kBishopPairCp = 30;

/** @brief Knight midgame outpost bonus. */
constexpr int kKnightOutpostCp = 22;

/** @brief Bishop midgame outpost bonus. */
constexpr int kBishopOutpostCp = 12;

/** @brief Attack-table slot containing all piece attacks for one side. */
constexpr int kAllAttacks = 0;

/** @brief Attack-table side index for White. */
constexpr int kWhite = 0;

/** @brief Attack-table side index for Black. */
constexpr int kBlack = 1;

/**
 * @brief Midgame weight assigned to attackers of the enemy king zone.
 *
 * Indexed by absolute piece type (0..6); only knight..queen carry weight.
 */
constexpr std::array<int, 7> kKingAttackWeight = {0, 0, 11, 9, 13, 18, 0};

/** @brief Knight midgame mobility score by reachable safe targets. */
constexpr std::array<int, 9> kKnightMobilityCp = {-18, -12, -6, -2, 3, 7, 10, 13, 15};

/** @brief Knight endgame mobility score by reachable safe targets. */
constexpr std::array<int, 9> kKnightMobilityEgCp = {-24, -16, -8, -2, 4, 9, 13, 16, 18};

/** @brief Bishop midgame mobility score by reachable safe targets. */
constexpr std::array<int, 14> kBishopMobilityCp =
    {-14, -8, -2, 4, 9, 14, 18, 22, 25, 28, 30, 32, 34, 35};

/** @brief Bishop endgame mobility score by reachable safe targets. */
constexpr std::array<int, 14> kBishopMobilityEgCp =
    {-18, -10, -2, 5, 11, 17, 23, 28, 32, 35, 38, 40, 42, 44};

/** @brief Rook midgame mobility score by reachable safe targets. */
constexpr std::array<int, 15> kRookMobilityCp =
    {-12, -7, -2, 2, 6, 10, 14, 17, 20, 22, 24, 26, 28, 30, 31};

/** @brief Rook endgame mobility score by reachable safe targets. */
constexpr std::array<int, 15> kRookMobilityEgCp =
    {-18, -9, -1, 7, 15, 23, 30, 36, 41, 45, 49, 52, 55, 57, 59};

/** @brief Queen midgame mobility score by reachable safe targets. */
constexpr std::array<int, 28> kQueenMobilityCp = {
    -8, -5, -2, 0, 3, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34};

/** @brief Queen endgame mobility score by reachable safe targets. */
constexpr std::array<int, 28> kQueenMobilityEgCp = {
    -12, -8, -4, 0, 5, 10, 14, 18, 22, 26, 30, 34, 37, 40, 43, 46,
    49, 52, 55, 58, 61, 64, 66, 68, 70, 72, 74, 76};

/**
 * @brief Pawn piece-square table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kPawnPst = {
    0, 0, 0, 0, 0, 0, 0, 0,
    10, 12, 12, 14, 14, 12, 12, 10,
    8, 10, 12, 16, 16, 12, 10, 8,
    6, 8, 10, 14, 14, 10, 8, 6,
    4, 6, 8, 12, 12, 8, 6, 4,
    2, 4, 6, 8, 8, 6, 4, 2,
    0, 0, 0, -6, -6, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0};

/**
 * @brief Knight piece-square table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kKnightPst = {
    -40, -25, -15, -10, -10, -15, -25, -40,
    -25, -10, 0, 5, 5, 0, -10, -25,
    -15, 0, 10, 15, 15, 10, 0, -15,
    -10, 5, 15, 20, 20, 15, 5, -10,
    -10, 5, 15, 20, 20, 15, 5, -10,
    -15, 0, 10, 15, 15, 10, 0, -15,
    -25, -10, 0, 5, 5, 0, -10, -25,
    -40, -25, -15, -10, -10, -15, -25, -40};

/**
 * @brief Bishop piece-square table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kBishopPst = {
    -15, -10, -10, -10, -10, -10, -10, -15,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 8, 8, 5, 0, -10,
    -10, 3, 8, 12, 12, 8, 3, -10,
    -10, 3, 8, 12, 12, 8, 3, -10,
    -10, 0, 5, 8, 8, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -15, -10, -10, -10, -10, -10, -10, -15};

/**
 * @brief Rook piece-square table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kRookPst = {
    5, 5, 5, 8, 8, 5, 5, 5,
    0, 0, 0, 4, 4, 0, 0, 0,
    -4, -4, -2, 0, 0, -2, -4, -4,
    -6, -6, -4, -2, -2, -4, -6, -6,
    -6, -6, -4, -2, -2, -4, -6, -6,
    -4, -4, -2, 0, 0, -2, -4, -4,
    0, 0, 0, 4, 4, 0, 0, 0,
    5, 5, 5, 8, 8, 5, 5, 5};

/**
 * @brief Queen piece-square table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kQueenPst = {
    -10, -8, -6, -4, -4, -6, -8, -10,
    -8, -4, -2, -1, -1, -2, -4, -8,
    -6, -2, 0, 1, 1, 0, -2, -6,
    -4, -1, 1, 2, 2, 1, -1, -4,
    -4, -1, 1, 2, 2, 1, -1, -4,
    -6, -2, 0, 1, 1, 0, -2, -6,
    -8, -4, -2, -1, -1, -2, -4, -8,
    -10, -8, -6, -4, -4, -6, -8, -10};

/**
 * @brief King opening/middlegame table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kKingPstOpening = {
    20, 25, 10, 0, 0, 10, 25, 20,
    10, 10, 0, -8, -8, 0, 10, 10,
    0, 0, -10, -15, -15, -10, 0, 0,
    -10, -10, -15, -20, -20, -15, -10, -10,
    -15, -15, -20, -25, -25, -20, -15, -15,
    -20, -20, -25, -30, -30, -25, -20, -20,
    -25, -25, -30, -35, -35, -30, -25, -25,
    -30, -30, -35, -40, -40, -35, -30, -30};

/**
 * @brief King endgame table from White's perspective (index 0 == a8).
 */
constexpr std::array<int, 64> kKingPstEndgame = {
    -10, -5, 0, 5, 5, 0, -5, -10,
    -5, 0, 5, 10, 10, 5, 0, -5,
    0, 5, 10, 15, 15, 10, 5, 0,
    5, 10, 15, 20, 20, 15, 10, 5,
    5, 10, 15, 20, 20, 15, 10, 5,
    0, 5, 10, 15, 15, 10, 5, 0,
    -5, 0, 5, 10, 10, 5, 0, -5,
    -10, -5, 0, 5, 5, 0, -5, -10};

}  // namespace cnnv::chess::eval
