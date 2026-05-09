#include "TestMain.h"
#include "chess/San.h"
#include "game/Game.h"
#include "game/Pgn.h"

using namespace cnnv::chess;
using namespace cnnv::game;

namespace {

bool playSan(Game& g, const char* moves[], std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        Move m = San::parse(g.position(), moves[i]);
        if (m.isNone()) return false;
        if (!g.tryMove(m)) return false;
    }
    return true;
}

}  // namespace

TEST(game_fools_mate_ends_with_white_loss) {
    Game g;
    const char* moves[] = {"f3", "e5", "g4", "Qh4#"};
    CHECK(playSan(g, moves, 4));
    CHECK_EQ(g.status(), GameStatus::BlackWins);
    CHECK_EQ(g.history().plyCount(), std::size_t{4});
}

TEST(game_scholars_mate) {
    Game g;
    const char* moves[] = {"e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"};
    CHECK(playSan(g, moves, 7));
    CHECK_EQ(g.status(), GameStatus::WhiteWins);
}

TEST(game_stalemate_in_one) {
    Game g;
    CHECK(g.loadFen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"));
    CHECK_EQ(g.status(), GameStatus::DrawStalemate);
}

TEST(game_undo_redo_keeps_state_in_sync) {
    Game g;
    Move e4 = San::parse(g.position(), "e4");
    Move e5;
    CHECK(g.tryMove(e4));
    e5 = San::parse(g.position(), "e5");
    CHECK(g.tryMove(e5));
    CHECK_EQ(g.history().plyCount(), std::size_t{2});
    CHECK(g.undo());
    CHECK_EQ(g.history().plyCount(), std::size_t{1});
    CHECK(g.position().sideToMove() == Color::Black);
    CHECK(g.redo());
    CHECK_EQ(g.history().plyCount(), std::size_t{2});
    CHECK(g.position().sideToMove() == Color::White);
}

TEST(game_pgn_export_includes_all_moves_and_result) {
    Game g;
    const char* moves[] = {"f3", "e5", "g4", "Qh4#"};
    CHECK(playSan(g, moves, 4));
    std::string pgn = exportPgn(g);
    // Each SAN move should appear; the result tag should reflect black's win.
    CHECK(pgn.find("[Result \"0-1\"]") != std::string::npos);
    CHECK(pgn.find("1. f3 e5") != std::string::npos);
    CHECK(pgn.find("Qh4#") != std::string::npos);
    CHECK(pgn.find("0-1\n") != std::string::npos);
}

TEST(game_load_fen_replaces_position_and_clears_history) {
    Game g;
    const char* opening[] = {"e4", "e5"};
    CHECK(playSan(g, opening, 2));
    CHECK(g.loadFen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    CHECK_EQ(g.history().plyCount(), std::size_t{0});
    CHECK_EQ(g.history().size(),     std::size_t{0});
    CHECK(g.position().pieceAt(Square::E2).isNone());
}
