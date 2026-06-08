#pragma once

/**
 * @file Mcts.h
 * @brief Native PUCT Monte-Carlo tree search for the search-tree visualizer.
 *
 * A single-threaded port of chess-rtk's PUCT search (chess/engine/Mcts.java),
 * scoped to what the visualizer needs: select -> expand -> evaluate -> backup,
 * policy priors + a network value at the leaf, terminal handling, principal-
 * variation extraction, and a bounded tree snapshot for rendering.
 *
 * Leaf evaluation is delegated to an @ref Evaluator callback so the engine stays
 * independent of the neural-network and UI layers; the app supplies a callback
 * that runs the currently selected architecture.
 */

#include "chess/Move.h"
#include "chess/MoveList.h"
#include "chess/Position.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cnnv::search {

/**
 * @brief Result of evaluating one leaf position.
 */
struct LeafEval {
    /** @brief Position value from the side-to-move perspective, in [-1, 1]. */
    float value = 0.0f;

    /**
     * @brief Move priors aligned 1:1 with the legal-move list passed in.
     *
     * Should sum to ~1. If shorter than the move list the engine falls back to a
     * uniform prior for the missing moves.
     */
    std::vector<float> priors;
};

/**
 * @brief Leaf-evaluation callback: maps a position + its legal moves to a value
 *        and per-move priors.
 */
using Evaluator =
    std::function<LeafEval(const cnnv::chess::Position&, const cnnv::chess::MoveList&)>;

/**
 * @brief One node in a render-friendly tree snapshot.
 */
struct SnapshotNode {
    int parent = -1;                       ///< Parent index, -1 for the root.
    std::vector<int> children;             ///< Child snapshot indices.
    std::uint16_t moveRaw = 0xFFFF;        ///< Move from parent (raw packed).
    std::string uci;                       ///< Move from parent in UCI.
    long visits = 0;                       ///< Visit count N.
    float q = 0.0f;                        ///< Mean value (this node's STM POV).
    float prior = 0.0f;                    ///< Policy prior P.
    int depth = 0;                         ///< Ply from the root.
    bool terminal = false;                 ///< Terminal (mate/draw) node.
    bool onPv = false;                     ///< On the principal variation.
    std::string fen;                       ///< Position FEN at this node.
    std::uint64_t signature = 0;           ///< Full position hash for merging.
};

/**
 * @brief Bounded snapshot of the search tree plus root summary stats.
 */
struct Snapshot {
    std::vector<SnapshotNode> nodes;       ///< node[0] is the root.
    long playouts = 0;                     ///< Total playouts run.
    std::uint16_t bestMoveRaw = 0xFFFF;    ///< Most-visited root move.
    std::string bestUci;                   ///< Most-visited root move in UCI.
    int rootCentipawns = 0;                ///< Root value in centipawns.
    std::vector<std::uint16_t> pv;         ///< Principal variation (raw moves).
    std::string exploringFen;              ///< Last leaf evaluated (follow-leaf).
};

/**
 * @brief Single-threaded PUCT search over a fixed root position.
 */
class Mcts {
   public:
    /**
     * @brief Builds a search rooted at @p root.
     * @param root Position to search from.
     * @param cpuct Exploration constant (chess-rtk default 2.8).
     * @param evaluator Leaf-evaluation callback.
     */
    Mcts(const cnnv::chess::Position& root, double cpuct, Evaluator evaluator);

    /** @brief Runs one playout (select -> expand/evaluate -> backup). */
    void iterate();

    /** @brief Total playouts run so far. */
    long playouts() const noexcept { return m_playouts; }

    /** @brief Exploration constant in use. */
    double cpuct() const noexcept { return m_cpuct; }

    /** @brief Most-visited root move, or `Move::none()` before expansion. */
    cnnv::chess::Move bestMove() const;

    /** @brief Root value in centipawns (side-to-move perspective). */
    int rootCentipawns() const;

    /** @brief FEN of the most recently evaluated leaf (for follow-leaf trace). */
    const std::string& exploringFen() const noexcept { return m_exploringFen; }

    /**
     * @brief Builds a bounded snapshot of the tree for rendering.
     * @param maxNodes Cap on total nodes emitted.
     * @param topKChildren Keep only this many highest-visit children per node
     *        (0 = keep all).
     * @param maxDepth Maximum plies emitted from the root.
     * @param minVisits Minimum visits a child needs before being emitted.
     * @param pvOnly Only follow the most-visited child at each node.
     */
    Snapshot snapshot(int maxNodes = 1000, int topKChildren = 0,
                      int maxDepth = 256, int minVisits = 0,
                      bool pvOnly = false) const;

   private:
    struct Node {
        int parent = -1;
        std::vector<int> children;
        cnnv::chess::Move move = cnnv::chess::Move::none();
        double prior = 0.0;
        int depth = 0;
        bool expanded = false;
        bool terminal = false;
        double terminalValue = 0.0;
        cnnv::chess::Position pos;
        std::uint64_t signature = 0;
        int statsIdx = -1;
    };

    struct Stats {
        long visits = 0;
        double valueSum = 0.0;
    };

    long visits(const Node& n) const noexcept;
    double nodeQ(const Node& n) const noexcept;
    int selectChild(int nodeIdx) const;
    double expandAndEvaluate(int leafIdx);
    int mostVisitedChild(int nodeIdx) const;
    double snapshotQ(int nodeIdx) const noexcept;
    int statsFor(std::uint64_t signature);
    static int valueToCentipawns(double value) noexcept;

    std::vector<Node> m_nodes;
    std::vector<Stats> m_stats;
    std::unordered_map<std::uint64_t, int> m_statsBySignature;
    double m_cpuct = 2.8;
    Evaluator m_eval;
    long m_playouts = 0;
    std::string m_exploringFen;

    static constexpr double kFpuReduction = 0.05;
};

}  // namespace cnnv::search
