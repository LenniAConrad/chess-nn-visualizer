#include "nn/nnue/NnueFeatureEncoder.h"

#include "chess/Position.h"

#include <stdexcept>

namespace cnnv::nn::nnue {

namespace chess = cnnv::chess;

int FeatureEncoder::piecePlane(chess::Piece piece,
                               bool whitePerspective) noexcept {
    if (piece.isNone() || piece.type == chess::PieceType::King) return -1;
    const bool own = (piece.color == chess::Color::White) == whitePerspective;
    const int offset = own ? 0 : 5;
    switch (piece.type) {
        case chess::PieceType::Pawn:   return offset + kOwnPawn;
        case chess::PieceType::Knight: return offset + kOwnKnight;
        case chess::PieceType::Bishop: return offset + kOwnBishop;
        case chess::PieceType::Rook:   return offset + kOwnRook;
        case chess::PieceType::Queen:  return offset + kOwnQueen;
        default: return -1;
    }
}

int FeatureEncoder::perspectiveKingSquare(const chess::Position& pos,
                                          bool whitePerspective) {
    const chess::Color c = whitePerspective ? chess::Color::White
                                            : chess::Color::Black;
    const chess::Square ks = pos.kingSquare(c);
    if (ks == chess::Square::None) {
        throw std::invalid_argument(
            "NnueFeatureEncoder: position is missing the perspective king");
    }
    return orientSquare(chess::squareIndex(ks), whitePerspective);
}

std::vector<int> FeatureEncoder::activeFeatures(const chess::Position& pos,
                                                bool whitePerspective) {
    std::vector<int> out;
    out.reserve(kMaxActiveFeatures);
    const int king = perspectiveKingSquare(pos, whitePerspective);
    for (int sq = 0; sq < kSquares; ++sq) {
        chess::Piece p = pos.pieceAt(sq);
        const int plane = piecePlane(p, whitePerspective);
        if (plane < 0) continue;
        const int oriented = orientSquare(sq, whitePerspective);
        out.push_back(encodeFeature(king, plane, oriented));
    }
    return out;
}

}  // namespace cnnv::nn::nnue
