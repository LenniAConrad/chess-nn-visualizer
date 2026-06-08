#include "TestMain.h"

#include "chess/Fen.h"
#include "chess/Move.h"
#include "chess/MoveList.h"
#include "chess/Position.h"
#include "search/Mcts.h"

#include <cstddef>
#include <cmath>

using cnnv::chess::Fen;
using cnnv::chess::Move;
using cnnv::chess::MoveList;
using cnnv::chess::Position;
using cnnv::search::LeafEval;
using cnnv::search::Mcts;

namespace {

Position parse(const char* fen) {
    auto p = Fen::parse(fen);
    if (!p.has_value()) {
        throw cnnv_test::CheckFailure(std::string("bad FEN: ") + fen);
    }
    return *p;
}

// Trivial evaluator: neutral value, uniform priors. Terminal detection in the
// engine supplies the only real signal, which is enough to find a forced mate.
LeafEval uniformEval(const Position&, const MoveList& legal) {
    LeafEval e;
    e.value = 0.0f;
    const std::size_t n = legal.size() == 0 ? 1 : legal.size();
    e.priors.assign(n, 1.0f / static_cast<float>(n));
    return e;
}

constexpr const char* kStart =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

}  // namespace

// Basic invariants: the tree grows, the root is the snapshot root, a best move
// and a principal variation exist.
TEST(mcts_startpos_runs) {
    Mcts mcts(parse(kStart), 2.8, uniformEval);
    for (int i = 0; i < 200; ++i) mcts.iterate();
    CHECK_EQ(static_cast<int>(mcts.playouts()), 200);
    CHECK(!mcts.bestMove().isNone());

    const auto snap = mcts.snapshot(500, 0);
    CHECK(snap.nodes.size() > 1);
    CHECK_EQ(snap.nodes[0].parent, -1);
    CHECK(!snap.pv.empty());
    CHECK_EQ(snap.bestMoveRaw, mcts.bestMove().raw());
}

// Terminal backup must surface a forced mate: Ra1-a8 is checkmate, so it should
// become the most-visited root move.
TEST(mcts_finds_mate_in_one) {
    Mcts mcts(parse("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"), 2.8, uniformEval);
    for (int i = 0; i < 400; ++i) mcts.iterate();
    const Move best = mcts.bestMove();
    CHECK(!best.isNone());
    CHECK(best.toUci() == std::string("a1a8"));
}

// The snapshot respects the node cap.
TEST(mcts_snapshot_respects_cap) {
    Mcts mcts(parse(kStart), 2.8, uniformEval);
    for (int i = 0; i < 300; ++i) mcts.iterate();
    const auto snap = mcts.snapshot(32, 0);
    CHECK(snap.nodes.size() <= 32);
    CHECK(snap.nodes.size() >= 1);
}

TEST(mcts_snapshot_keeps_unvisited_children_unless_filtered) {
    Mcts mcts(parse(kStart), 2.8, uniformEval);
    mcts.iterate();  // expands the root, matching chess-rtk's visible frontier.

    const auto withFrontier = mcts.snapshot(128, 0, 256, 0, false);
    CHECK(withFrontier.nodes.size() > 10);

    const auto visitedOnly = mcts.snapshot(128, 0, 256, 1, false);
    CHECK_EQ(static_cast<int>(visitedOnly.nodes.size()), 1);
}

TEST(mcts_snapshot_q_is_parent_perspective_and_carries_signature) {
    auto eval = [](const Position&, const MoveList& legal) {
        LeafEval e;
        e.value = 0.25f;
        const std::size_t n = legal.size() == 0 ? 1 : legal.size();
        e.priors.assign(n, 1.0f / static_cast<float>(n));
        return e;
    };

    Mcts mcts(parse(kStart), 2.8, eval);
    for (int i = 0; i < 2; ++i) mcts.iterate();
    const auto snap = mcts.snapshot(128, 0);

    const auto root = Fen::parse(snap.nodes[0].fen);
    CHECK(root.has_value());
    CHECK_EQ(snap.nodes[0].signature, root->hash());

    bool foundVisitedChild = false;
    for (const auto& node : snap.nodes) {
        if (node.parent == 0 && node.visits > 0) {
            const auto child = Fen::parse(node.fen);
            CHECK(child.has_value());
            CHECK_EQ(node.signature, child->hash());
            CHECK(std::fabs(node.q + 0.25f) < 0.001f);
            foundVisitedChild = true;
            break;
        }
    }
    CHECK(foundVisitedChild);
}
