#include "TestMain.h"

#include "chess/Fen.h"
#include "chess/Move.h"
#include "chess/Position.h"
#include "nn/lc0_cnn/PolicyEncoder.h"

using cnnv::chess::Fen;
using cnnv::chess::Move;
using cnnv::chess::Position;
using cnnv::chess::Square;
using cnnv::nn::lc0_cnn::rawPolicyIndex;

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

}  // namespace

// raw index = plane*64 + fromSquare (a1=0, black rank-mirrored). These exact
// values verify the transpile of chess-rtk's PolicyEncoder.rawPolicyIndex.
TEST(policy_encoder_known_indices) {
    const Position w = parse(kStart);
    // e2e4: queen-like N, distance 2 -> plane 1, fromSquare e2=12.
    CHECK_EQ(rawPolicyIndex(w, Move(Square::E2, Square::E4)), 1 * 64 + 12);
    // g1f3: knight delta (-1,+2) -> knight index 7 -> plane 63, fromSquare g1=6.
    CHECK_EQ(rawPolicyIndex(w, Move(Square::G1, Square::F3)), 63 * 64 + 6);
}

// Black-to-move moves are rank-mirrored, so e7e5 maps to the same slot as the
// white e2e4 double push.
TEST(policy_encoder_black_is_mirrored) {
    const Position b =
        parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    CHECK_EQ(rawPolicyIndex(b, Move(Square::E7, Square::E5)), 1 * 64 + 12);
}

// Underpromotions use planes 64..72; a queen promotion uses the queen-like
// plane instead.
TEST(policy_encoder_promotions) {
    const Position p = parse("8/P7/8/8/8/8/8/4k1K1 w - - 0 1");
    // a7a8=N: underpromo knight (idx 0), forward (dir 0) -> plane 64,
    // fromSquare a7=48.
    CHECK_EQ(rawPolicyIndex(p, Move(Square::A7, Square::A8, Move::Promotion::Knight)),
             64 * 64 + 48);
    // a7a8=Q: queen-like N distance 1 -> plane 0, fromSquare 48.
    CHECK_EQ(rawPolicyIndex(p, Move(Square::A7, Square::A8, Move::Promotion::Queen)),
             0 * 64 + 48);
}

// Non-encodable / null moves return -1.
TEST(policy_encoder_rejects_invalid) {
    const Position w = parse(kStart);
    CHECK_EQ(rawPolicyIndex(w, Move::none()), -1);
}
