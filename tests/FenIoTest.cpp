#include "TestMain.h"
#include "chess/Fen.h"
#include "io/FenIo.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace cnnv::chess;
using namespace cnnv::io;

namespace {

std::string tempPath(const char* leaf) {
    std::string p = "/tmp/cnnv_test_";
    p += leaf;
    return p;
}

}  // namespace

TEST(fenio_save_and_load_round_trip) {
    Position p;
    p.setStartpos();
    std::string path = tempPath("startpos.fen");
    CHECK(savePositionToFenFile(path, p));
    auto loaded = loadPositionFromFenFile(path);
    CHECK(loaded.has_value());
    CHECK_EQ(Fen::format(*loaded), Fen::format(p));
    std::remove(path.c_str());
}

TEST(fenio_skips_comments_and_blank_lines) {
    std::string path = tempPath("commented.fen");
    {
        std::ofstream out(path);
        out << "# header comment\n"
            << "\n"
            << "  \t\n"
            << "rnbqkbnr/pppppppp/8/8/8/8/8/PPPPPPPP w KQkq - 0 1   # inline comment\n";
    }
    auto loaded = loadPositionFromFenFile(path);
    CHECK(loaded.has_value());
    CHECK_EQ(loaded->sideToMove(), Color::White);
    std::remove(path.c_str());
}

TEST(fenio_returns_nullopt_on_missing_file) {
    auto loaded = loadPositionFromFenFile("/tmp/this/path/does/not/exist.fen");
    CHECK(!loaded.has_value());
}

TEST(fenio_returns_nullopt_on_invalid_fen) {
    std::string path = tempPath("bad.fen");
    {
        std::ofstream out(path);
        out << "not a fen\n";
    }
    auto loaded = loadPositionFromFenFile(path);
    CHECK(!loaded.has_value());
    std::remove(path.c_str());
}
