#include "chess/Perft.h"

#include "chess/MoveGenerator.h"

namespace cnnv::chess {

std::uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveList moves;
    MoveGenerator::generateLegal(pos, moves);
    if (depth == 1) return moves.size();
    std::uint64_t total = 0;
    for (Move m : moves) {
        pos.make(m);
        total += perft(pos, depth - 1);
        pos.unmake();
    }
    return total;
}

}  // namespace cnnv::chess
