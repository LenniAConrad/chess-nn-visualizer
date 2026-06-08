#include "search/Mcts.h"

#include "chess/Fen.h"
#include "chess/MoveGenerator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>

namespace cnnv::search {

namespace {

// Inverse of the tanh-style value->cp mapping chess-rtk uses (scale 600).
constexpr double kCpScale = 600.0;

}  // namespace

Mcts::Mcts(const cnnv::chess::Position& root, double cpuct, Evaluator evaluator)
    : m_cpuct(cpuct), m_eval(std::move(evaluator)) {
    Node r;
    r.parent = -1;
    r.depth = 0;
    r.pos = root;
    r.signature = root.hash();
    r.statsIdx = statsFor(r.signature);
    m_nodes.push_back(std::move(r));
}

long Mcts::visits(const Node& n) const noexcept {
    return n.statsIdx >= 0 ? m_stats[static_cast<std::size_t>(n.statsIdx)].visits
                           : 0L;
}

double Mcts::nodeQ(const Node& n) const noexcept {
    if (n.statsIdx < 0) return 0.0;
    const Stats& s = m_stats[static_cast<std::size_t>(n.statsIdx)];
    return s.visits > 0 ? s.valueSum / static_cast<double>(s.visits) : 0.0;
}

int Mcts::statsFor(std::uint64_t signature) {
    auto it = m_statsBySignature.find(signature);
    if (it != m_statsBySignature.end()) return it->second;
    const int idx = static_cast<int>(m_stats.size());
    m_stats.push_back(Stats{});
    m_statsBySignature.emplace(signature, idx);
    return idx;
}

int Mcts::valueToCentipawns(double value) noexcept {
    const double v = std::clamp(value, -0.999, 0.999);
    return static_cast<int>(std::lround(kCpScale * 0.5 *
                                        std::log((1.0 + v) / (1.0 - v))));
}

int Mcts::selectChild(int nodeIdx) const {
    const Node& parent = m_nodes[static_cast<std::size_t>(nodeIdx)];
    const double parentQ = nodeQ(parent);
    const double sqrtN =
        std::sqrt(std::max(1.0, static_cast<double>(visits(parent))));

    int best = parent.children.front();
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int c : parent.children) {
        const Node& child = m_nodes[static_cast<std::size_t>(c)];
        // child.valueSum is from the child's side-to-move perspective; from the
        // parent's perspective that is negated. Unvisited children get a
        // first-play-urgency reduction off the parent value.
        const double childPerspective =
            visits(child) == 0 ? parentQ - kFpuReduction : -nodeQ(child);
        const double u = m_cpuct * child.prior * sqrtN /
                         (1.0 + static_cast<double>(visits(child)));
        const double score = childPerspective + u;
        if (score > bestScore) {
            bestScore = score;
            best = c;
        }
    }
    return best;
}

double Mcts::expandAndEvaluate(int leafIdx) {
    // Work on copies so the stored node positions stay pristine.
    cnnv::chess::Position pos = m_nodes[static_cast<std::size_t>(leafIdx)].pos;
    cnnv::chess::Position scratch = pos;
    cnnv::chess::MoveList legal;
    cnnv::chess::MoveGenerator::generateLegal(scratch, legal);

    // Terminal: no legal moves => checkmate (loss) or stalemate (draw).
    if (legal.size() == 0) {
        const double value = pos.inCheck() ? -1.0 : 0.0;
        Node& leaf = m_nodes[static_cast<std::size_t>(leafIdx)];
        leaf.terminal = true;
        leaf.terminalValue = value;
        leaf.expanded = true;
        return value;
    }
    // Terminal draws.
    if (pos.isInsufficientMaterial() || pos.isFiftyMoveDraw() ||
        pos.isThreefoldRepetition()) {
        Node& leaf = m_nodes[static_cast<std::size_t>(leafIdx)];
        leaf.terminal = true;
        leaf.terminalValue = 0.0;
        leaf.expanded = true;
        return 0.0;
    }

    const LeafEval ev = m_eval(pos, legal);
    m_exploringFen = cnnv::chess::Fen::format(pos);

    const std::size_t n = legal.size();
    const int base = static_cast<int>(m_nodes.size());
    const int childDepth = m_nodes[static_cast<std::size_t>(leafIdx)].depth + 1;
    // Append children. This may reallocate m_nodes, so never hold a Node& across
    // these pushes; re-index afterwards.
    for (std::size_t i = 0; i < n; ++i) {
        Node child;
        child.parent = leafIdx;
        child.move = legal[i];
        child.prior = i < ev.priors.size()
                          ? static_cast<double>(ev.priors[i])
                          : 1.0 / static_cast<double>(n);
        child.depth = childDepth;
        child.pos = pos;
        child.pos.make(legal[i]);
        child.signature = child.pos.hash();
        child.statsIdx = statsFor(child.signature);
        m_nodes.push_back(std::move(child));
    }
    Node& leaf = m_nodes[static_cast<std::size_t>(leafIdx)];
    leaf.children.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        leaf.children.push_back(base + static_cast<int>(i));
    }
    leaf.expanded = true;
    return static_cast<double>(ev.value);
}

void Mcts::iterate() {
    std::vector<int> path;
    path.reserve(64);
    int idx = 0;
    path.push_back(0);
    while (true) {
        const Node& n = m_nodes[static_cast<std::size_t>(idx)];
        if (!n.expanded || n.terminal || n.children.empty()) break;
        idx = selectChild(idx);
        path.push_back(idx);
    }

    double value;
    const Node& leaf = m_nodes[static_cast<std::size_t>(idx)];
    if (leaf.terminal) {
        value = leaf.terminalValue;
    } else if (!leaf.expanded) {
        value = expandAndEvaluate(idx);  // may reallocate m_nodes
    } else {
        value = nodeQ(leaf);
    }

    // Backup with sign flip per ply (value is from the leaf's STM perspective).
    double v = value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node& n = m_nodes[static_cast<std::size_t>(*it)];
        Stats& s = m_stats[static_cast<std::size_t>(n.statsIdx)];
        s.visits += 1;
        s.valueSum += v;
        v = -v;
    }
    ++m_playouts;
}

int Mcts::mostVisitedChild(int nodeIdx) const {
    const Node& n = m_nodes[static_cast<std::size_t>(nodeIdx)];
    int best = -1;
    long bestVisits = -1;
    for (int c : n.children) {
        const long v = visits(m_nodes[static_cast<std::size_t>(c)]);
        if (v > bestVisits) {
            bestVisits = v;
            best = c;
        }
    }
    return best;
}

cnnv::chess::Move Mcts::bestMove() const {
    const int c = mostVisitedChild(0);
    return c < 0 ? cnnv::chess::Move::none()
                 : m_nodes[static_cast<std::size_t>(c)].move;
}

int Mcts::rootCentipawns() const {
    const int c = mostVisitedChild(0);
    if (c < 0) return 0;
    // Root perspective value is the negation of the best child's value.
    return valueToCentipawns(-nodeQ(m_nodes[static_cast<std::size_t>(c)]));
}

double Mcts::snapshotQ(int nodeIdx) const noexcept {
    const Node& node = m_nodes[static_cast<std::size_t>(nodeIdx)];
    if (node.parent < 0) return nodeQ(node);
    const Node& parent = m_nodes[static_cast<std::size_t>(node.parent)];
    if (visits(node) == 0) {
        return visits(parent) == 0 ? 0.0 : nodeQ(parent) - kFpuReduction;
    }
    return -nodeQ(node);
}

Snapshot Mcts::snapshot(int maxNodes, int topKChildren, int maxDepth,
                        int minVisits, bool pvOnly) const {
    Snapshot snap;
    snap.playouts = m_playouts;
    const cnnv::chess::Move best = bestMove();
    snap.bestMoveRaw = best.raw();
    snap.bestUci = best.isNone() ? std::string() : best.toUci();
    snap.rootCentipawns = rootCentipawns();
    snap.exploringFen = m_exploringFen;

    if (m_nodes.empty()) return snap;

    // Mark the principal variation in pool-index space.
    std::vector<bool> onPv(m_nodes.size(), false);
    {
        int cur = 0;
        onPv[0] = true;
        while (true) {
            const int c = mostVisitedChild(cur);
            if (c < 0 || visits(m_nodes[static_cast<std::size_t>(c)]) == 0) break;
            onPv[static_cast<std::size_t>(c)] = true;
            snap.pv.push_back(m_nodes[static_cast<std::size_t>(c)].move.raw());
            cur = c;
        }
    }

    // Best-first emit by a depth-biased visit score (chess-rtk MctsSearch
    // emitScore = visits * (1 + TREE_DEPTH_BIAS * depth), TREE_DEPTH_BIAS = 0.6).
    // Expanding the most-explored lines first but giving deeper nodes a bonus
    // means a heavily searched deep line is emitted before shallow low-relevance
    // alternatives (which always have more raw visits and would otherwise consume
    // the whole budget so the deep routes never show). A child is only enqueued
    // after its parent is emitted, so the emitted set stays a connected subtree.
    constexpr double kTreeDepthBias = 0.6;
    const int safeMaxNodes = std::max(1, maxNodes);
    const int safeMaxDepth = std::clamp(maxDepth, 0, 256);
    const int safeMinVisits = std::max(0, minVisits);
    const int safeTopK = std::clamp(topKChildren, 0, 64);
    std::vector<int> poolToSnap(m_nodes.size(), -1);
    using FrontierItem = std::tuple<double, long, int>;
    std::priority_queue<FrontierItem> frontier;  // (emitScore, visits, poolIdx)
    auto emitScore = [&](int poolIdx) -> double {
        const Node& n = m_nodes[static_cast<std::size_t>(poolIdx)];
        return static_cast<double>(visits(n)) *
               (1.0 + kTreeDepthBias * static_cast<double>(n.depth));
    };
    auto emit = [&](int poolIdx) -> int {
        const int snapIdx = static_cast<int>(snap.nodes.size());
        poolToSnap[static_cast<std::size_t>(poolIdx)] = snapIdx;
        const Node& n = m_nodes[static_cast<std::size_t>(poolIdx)];
        SnapshotNode sn;
        sn.parent = n.parent < 0 ? -1 : poolToSnap[static_cast<std::size_t>(n.parent)];
        sn.moveRaw = n.move.raw();
        sn.uci = n.move.isNone() ? std::string() : n.move.toUci();
        sn.visits = visits(n);
        sn.q = static_cast<float>(snapshotQ(poolIdx));
        sn.prior = static_cast<float>(n.prior);
        sn.depth = n.depth;
        sn.terminal = n.terminal;
        sn.onPv = onPv[static_cast<std::size_t>(poolIdx)];
        sn.fen = cnnv::chess::Fen::format(n.pos);
        sn.signature = n.signature;
        snap.nodes.push_back(std::move(sn));
        return snapIdx;
    };

    emit(0);
    frontier.push({emitScore(0), visits(m_nodes[0]), 0});
    while (!frontier.empty() &&
           static_cast<int>(snap.nodes.size()) < safeMaxNodes) {
        const int poolIdx = std::get<2>(frontier.top());
        frontier.pop();
        const int snapIdx = poolToSnap[static_cast<std::size_t>(poolIdx)];
        const Node& poolNode = m_nodes[static_cast<std::size_t>(poolIdx)];
        if (poolNode.depth >= safeMaxDepth) continue;

        std::vector<int> kids;
        if (pvOnly) {
            const int pvChild = mostVisitedChild(poolIdx);
            if (pvChild >= 0) kids.push_back(pvChild);
        } else {
            kids = poolNode.children;
        }
        std::sort(kids.begin(), kids.end(), [&](int a, int b) {
            const long av = visits(m_nodes[static_cast<std::size_t>(a)]);
            const long bv = visits(m_nodes[static_cast<std::size_t>(b)]);
            if (av != bv) return av > bv;
            return m_nodes[static_cast<std::size_t>(a)].prior >
                   m_nodes[static_cast<std::size_t>(b)].prior;
        });
        const std::size_t keep =
            safeTopK > 0
                ? std::min(kids.size(), static_cast<std::size_t>(safeTopK))
                : kids.size();
        for (std::size_t i = 0; i < keep; ++i) {
            if (static_cast<int>(snap.nodes.size()) >= safeMaxNodes) break;
            const Node& child = m_nodes[static_cast<std::size_t>(kids[i])];
            if (visits(child) < safeMinVisits) {
                continue;
            }
            const int childSnap = emit(kids[i]);
            snap.nodes[static_cast<std::size_t>(snapIdx)].children.push_back(childSnap);
            frontier.push({emitScore(kids[i]), visits(child), kids[i]});
        }
    }
    return snap;
}

}  // namespace cnnv::search
