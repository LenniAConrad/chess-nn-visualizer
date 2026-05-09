#include "TestMain.h"
#include "chess/Move.h"
#include "game/MoveHistory.h"

using namespace cnnv::chess;
using namespace cnnv::game;

namespace {

MoveRecord makeRecord(Move m, const char* san) {
    MoveRecord r;
    r.move = m;
    r.san = san;
    return r;
}

}  // namespace

TEST(movehistory_starts_empty) {
    MoveHistory h;
    CHECK(!h.canUndo());
    CHECK(!h.canRedo());
    CHECK_EQ(h.size(), std::size_t{0});
    CHECK_EQ(h.plyCount(), std::size_t{0});
}

TEST(movehistory_push_three_then_undo_three) {
    MoveHistory h;
    h.pushMove(makeRecord(Move(Square::E2, Square::E4), "e4"));
    h.pushMove(makeRecord(Move(Square::E7, Square::E5), "e5"));
    h.pushMove(makeRecord(Move(Square::G1, Square::F3), "Nf3"));
    CHECK_EQ(h.size(),     std::size_t{3});
    CHECK_EQ(h.plyCount(), std::size_t{3});
    CHECK(h.canUndo());
    CHECK(!h.canRedo());

    h.undo();
    CHECK_EQ(h.plyCount(), std::size_t{2});
    CHECK(h.canRedo());
    h.undo();
    h.undo();
    CHECK_EQ(h.plyCount(), std::size_t{0});
    CHECK(!h.canUndo());
    CHECK(h.canRedo());
}

TEST(movehistory_redo_walks_forward) {
    MoveHistory h;
    h.pushMove(makeRecord(Move(Square::E2, Square::E4), "e4"));
    h.pushMove(makeRecord(Move(Square::E7, Square::E5), "e5"));
    h.undo();
    h.undo();
    CHECK_EQ(h.plyCount(), std::size_t{0});
    h.redo();
    CHECK_EQ(h.plyCount(), std::size_t{1});
    h.redo();
    CHECK_EQ(h.plyCount(), std::size_t{2});
    CHECK(!h.canRedo());
}

TEST(movehistory_push_after_undo_truncates_redo_branch) {
    MoveHistory h;
    h.pushMove(makeRecord(Move(Square::E2, Square::E4), "e4"));
    h.pushMove(makeRecord(Move(Square::E7, Square::E5), "e5"));
    h.pushMove(makeRecord(Move(Square::G1, Square::F3), "Nf3"));
    h.undo();
    h.undo();
    CHECK(h.canRedo());
    h.pushMove(makeRecord(Move(Square::B1, Square::C3), "Nc3"));
    CHECK(!h.canRedo());
    CHECK_EQ(h.size(), std::size_t{2});  // e4 then Nc3
    CHECK_EQ(h.plyCount(), std::size_t{2});
}

TEST(movehistory_clear_releases_all_nodes) {
    MoveHistory h;
    for (int i = 0; i < 5; ++i) {
        h.pushMove(makeRecord(Move(Square::E2, Square::E4), "e4"));
    }
    h.clear();
    CHECK_EQ(h.size(), std::size_t{0});
    CHECK(!h.canUndo());
    CHECK(!h.canRedo());
}
