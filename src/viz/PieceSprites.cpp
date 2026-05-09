#include "viz/PieceSprites.h"

#include "chess/Piece.h"

#include <raylib.h>

namespace cnnv::viz {

namespace {

const char* fileNameFor(cnnv::chess::Piece p) {
    using namespace cnnv::chess;
    if (p.isNone()) return nullptr;
    bool w = p.color == Color::White;
    switch (p.type) {
        case PieceType::Pawn:   return w ? "wP" : "bP";
        case PieceType::Knight: return w ? "wN" : "bN";
        case PieceType::Bishop: return w ? "wB" : "bB";
        case PieceType::Rook:   return w ? "wR" : "bR";
        case PieceType::Queen:  return w ? "wQ" : "bQ";
        case PieceType::King:   return w ? "wK" : "bK";
        case PieceType::None:   return nullptr;
    }
    return nullptr;
}

}  // namespace

PieceSprites::PieceSprites(const std::string& assetDir) {
    using namespace cnnv::chess;
    bool ok = true;
    for (Color c : {Color::White, Color::Black}) {
        for (PieceType t : {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                            PieceType::Rook, PieceType::Queen, PieceType::King}) {
            Piece p{c, t};
            const char* name = fileNameFor(p);
            std::string pngPath = assetDir + "/" + name + ".png";
            std::string svgPath = assetDir + "/" + name + ".svg";
            Image img = LoadImage(FileExists(pngPath.c_str())
                                      ? pngPath.c_str()
                                      : svgPath.c_str());
            Texture2D tex{};
            if (img.data != nullptr) {
                tex = LoadTextureFromImage(img);
                UnloadImage(img);
            }
            if (tex.id == 0) {
                ok = false;
            } else {
                SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
            }
            m_textures[static_cast<std::size_t>(pieceIndex(p))] = tex;
        }
    }
    m_loaded = ok;
}

PieceSprites::~PieceSprites() {
    for (auto& tex : m_textures) {
        if (tex.id != 0) UnloadTexture(tex);
    }
}

const Texture2D& PieceSprites::textureFor(cnnv::chess::Piece p) const {
    if (p.isNone()) return m_blank;
    return m_textures[static_cast<std::size_t>(cnnv::chess::pieceIndex(p))];
}

}  // namespace cnnv::viz
