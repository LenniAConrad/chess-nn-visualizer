#include "TestMain.h"

#include "chess/Fen.h"
#include "chess/eval/Classical.h"

#include <algorithm>
#include <array>
#include <cmath>

using cnnv::chess::Fen;
using cnnv::chess::Position;
using cnnv::chess::Square;
using cnnv::chess::squareIndex;
namespace ce = cnnv::chess::eval;

namespace {

Position parse(const char* fen) {
    auto p = Fen::parse(fen);
    if (!p.has_value()) {
        throw cnnv_test::CheckFailure(std::string("bad FEN: ") + fen);
    }
    return *p;
}

constexpr const char* kStart =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

bool approx(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

float normalizedTableValue(const std::array<int, 64>& pst, int idx) {
    const auto [minIt, maxIt] = std::minmax_element(pst.begin(), pst.end());
    const int range = *maxIt - *minIt;
    if (range <= 0) return 0.0f;
    return static_cast<float>(pst[static_cast<std::size_t>(idx)] - *minIt) /
           static_cast<float>(range);
}

}  // namespace

// At the start position every term is symmetric and cancels to zero, except the
// side-to-move tempo bonus (+8 for White). This catches color asymmetry bugs in
// the port.
TEST(classical_startpos_is_symmetric) {
    const ce::Breakdown b = ce::evaluateWhiteBreakdown(parse(kStart));
    CHECK(b.whiteToMove);
    CHECK(b.phase > 0.9);
    CHECK_EQ(b.material, 0);
    CHECK_EQ(b.pieceSquare, 0);
    CHECK_EQ(b.bishopPair, 0);
    CHECK_EQ(b.pawnStructure, 0);
    CHECK_EQ(b.rookFile, 0);
    CHECK_EQ(b.kingSafety, 0);
    CHECK_EQ(b.activity, 0);
    CHECK_EQ(b.threats, 0);
    CHECK_EQ(b.space, 0);
    CHECK_EQ(b.tempo, 8);
    CHECK_EQ(b.checkPenalty, 0);
    CHECK_EQ(b.whiteTotal(), 8);
}

// The same position with Black to move negates the white-perspective total
// (tempo flips); from the side-to-move view it is still +8.
TEST(classical_startpos_black_to_move) {
    const char* fen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
    const ce::Breakdown b = ce::evaluateWhiteBreakdown(parse(fen));
    CHECK(!b.whiteToMove);
    CHECK_EQ(b.tempo, -8);
    CHECK_EQ(b.whiteTotal(), -8);
    CHECK_EQ(b.stmTotal(), 8);
}

// Removing Black's h-pawn leaves White a clean pawn up: the material term must
// be exactly +100 (pawn value).
TEST(classical_material_up_a_pawn) {
    const char* fen =
        "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const ce::Breakdown b = ce::evaluateWhiteBreakdown(parse(fen));
    CHECK_EQ(b.material, 100);
    CHECK(b.whiteTotal() > 0);
}

// The eleven terms must sum exactly to whiteTotal() across varied positions.
TEST(classical_terms_sum_to_total) {
    const std::array<const char*, 3> fens = {{
        kStart,
        "r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    }};
    for (const char* fen : fens) {
        const ce::Breakdown b = ce::evaluateWhiteBreakdown(parse(fen));
        const int sum = b.material + b.pieceSquare + b.bishopPair +
                        b.pawnStructure + b.rookFile + b.kingSafety + b.activity +
                        b.threats + b.space + b.tempo + b.checkPenalty;
        CHECK_EQ(sum, b.whiteTotal());
    }
}

// WDL is a normalized triplet summing to exactly 1000; the start position is
// roughly balanced and a clean extra pawn raises win above loss.
TEST(classical_wdl_normalized) {
    const ce::WdlTriplet start = ce::evaluateWdl(parse(kStart), true);
    CHECK_EQ(start.win + start.draw + start.loss, 1000);

    const ce::WdlTriplet upPawn = ce::evaluateWdl(
        parse("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), true);
    CHECK_EQ(upPawn.win + upPawn.draw + upPawn.loss, 1000);
    CHECK(upPawn.win >= upPawn.loss);
}

// Insufficient material (lone kings) is a forced draw.
TEST(classical_wdl_insufficient_material_draw) {
    const ce::WdlTriplet b = ce::evaluateWdl(parse("8/8/4k3/8/8/4K3/8/8 w - - 0 1"),
                                             true);
    CHECK_EQ(b.win + b.draw + b.loss, 1000);
    CHECK(b.draw == 1000);
}

// The piece-square tables are populated and the king uses its opening table.
TEST(classical_piece_square_tables_present) {
    bool anyNonZero = false;
    for (int type = 1; type <= 6; ++type) {
        const std::array<int, 64> pst = ce::pieceSquareTable(type);
        for (int v : pst) {
            if (v != 0) anyNonZero = true;
        }
    }
    CHECK(anyNonZero);
    // Unknown types return a zeroed table.
    const std::array<int, 64> none = ce::pieceSquareTable(0);
    for (int v : none) CHECK_EQ(v, 0);
}

TEST(classical_piece_square_heatmap_tracks_occupied_piece_tables) {
    const Position pos = parse(kStart);
    const std::array<float, 64> heat = ce::pieceSquareHeatmap(pos);

    const int a2 = squareIndex(Square::A2);
    const int a7 = squareIndex(Square::A7);
    const int b1 = squareIndex(Square::B1);
    const int e4 = squareIndex(Square::E4);

    const std::array<int, 64> pawn = ce::pieceSquareTable(1);
    const std::array<int, 64> knight = ce::pieceSquareTable(2);

    CHECK(approx(heat[static_cast<std::size_t>(a2)],
                 normalizedTableValue(pawn, a2 ^ 56)));
    CHECK(approx(heat[static_cast<std::size_t>(a7)],
                 normalizedTableValue(pawn, a7)));
    CHECK(approx(heat[static_cast<std::size_t>(a2)],
                 heat[static_cast<std::size_t>(a7)]));
    CHECK(approx(heat[static_cast<std::size_t>(b1)],
                 normalizedTableValue(knight, b1 ^ 56)));
    CHECK(!approx(heat[static_cast<std::size_t>(b1)],
                  normalizedTableValue(pawn, b1 ^ 56)));
    CHECK_EQ(heat[static_cast<std::size_t>(e4)], 0.0f);
}
