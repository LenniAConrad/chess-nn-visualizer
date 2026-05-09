#include "viz/PieceSprites.h"

#include "chess/Piece.h"

#include <raylib.h>

namespace cnnv::viz {

namespace {

// Render-target size for SVG rasterization. 256 px is enough for a roughly
// 90-px-per-square board and stays sharp at 2x HiDPI scaling thanks to
// bilinear filtering. Bumping to 512 doubled load time without a visible
// difference at 1280x800.
constexpr int kSvgRasterSize = 256;

const char* fileNameFor(cnnv::chess::Piece p) {
    using namespace cnnv::chess;
    if (p.isNone()) return nullptr;
    bool w = p.color == Color::White;
    switch (p.type) {
        case PieceType::Pawn:   return w ? "wP.svg" : "bP.svg";
        case PieceType::Knight: return w ? "wN.svg" : "bN.svg";
        case PieceType::Bishop: return w ? "wB.svg" : "bB.svg";
        case PieceType::Rook:   return w ? "wR.svg" : "bR.svg";
        case PieceType::Queen:  return w ? "wQ.svg" : "bQ.svg";
        case PieceType::King:   return w ? "wK.svg" : "bK.svg";
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
            std::string path = assetDir + "/" + name;
            Image img = LoadImageSvg(path.c_str(), kSvgRasterSize, kSvgRasterSize);
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
