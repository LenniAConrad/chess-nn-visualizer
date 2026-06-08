#include "viz/BoardView.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cnnv::viz {

// Note: do NOT `using namespace cnnv::chess` here. Our `chess::Color` enum
// would shadow raylib's global `Color` struct that this file uses heavily.
namespace chess = cnnv::chess;

namespace {

constexpr float kCoordLeftMargin = 30.0f;
constexpr float kCoordRightMargin = 8.0f;
constexpr float kCoordTopMargin = 8.0f;
constexpr float kCoordBottomMargin = 34.0f;
constexpr float kActivationOverlayMaxOpacity = 0.62f;

float clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

Color lerpColor(Color a, Color b, float t) {
    auto mix = [t](unsigned char x, unsigned char y) {
        return static_cast<unsigned char>(x + (y - x) * t);
    };
    return Color{mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a)};
}

}  // namespace

BoardView::BoardView(const chess::Position& position,
                     const PieceSprites& sprites,
                     Rectangle bounds)
    : m_position(&position), m_sprites(&sprites), m_bounds(bounds) {}

Vector2 BoardView::boardOriginAndSquareSize(float& outSquareSize) const {
    const float availableW = std::max(
        0.0f, m_bounds.width - kCoordLeftMargin - kCoordRightMargin);
    const float availableH = std::max(
        0.0f, m_bounds.height - kCoordTopMargin - kCoordBottomMargin);
    float side = std::min(availableW, availableH);
    outSquareSize = side / 8.0f;
    float ox = m_bounds.x + kCoordLeftMargin + (availableW - side) * 0.5f;
    float oy = m_bounds.y + kCoordTopMargin + (availableH - side) * 0.5f;
    return {ox, oy};
}

Rectangle BoardView::squareRect(int file, int rank) const {
    float sq;
    Vector2 origin = boardOriginAndSquareSize(sq);
    int displayFile = m_flipped ? 7 - file : file;
    int displayRank = m_flipped ? rank : 7 - rank;
    return Rectangle{
        origin.x + displayFile * sq,
        origin.y + displayRank * sq,
        sq, sq,
    };
}

std::optional<chess::Square> BoardView::squareAtPixel(Vector2 pixel) const {
    float sq;
    Vector2 origin = boardOriginAndSquareSize(sq);
    if (sq <= 0) return std::nullopt;
    float dx = pixel.x - origin.x;
    float dy = pixel.y - origin.y;
    if (dx < 0 || dy < 0 || dx >= 8 * sq || dy >= 8 * sq) return std::nullopt;
    int displayFile = static_cast<int>(dx / sq);
    int displayRank = static_cast<int>(dy / sq);
    int file = m_flipped ? 7 - displayFile : displayFile;
    int rank = m_flipped ? displayRank : 7 - displayRank;
    return chess::makeSquare(file, rank);
}

void BoardView::setActivationOverlay(const std::array<float, 64>& intensities) noexcept {
    if (m_overlayActive) {
        m_previousOverlay = m_overlay;
        m_previousOverlayActive = true;
    } else {
        m_previousOverlayActive = false;
    }
    m_overlay = intensities;
    m_overlayActive = true;
    m_overlayOpacity = 0.0f;
}

void BoardView::setActivationOverlayOpacity(float opacity) noexcept {
    m_overlayOpacity = clamp01(opacity);
}

void BoardView::clearActivationOverlay() noexcept {
    m_overlayActive = false;
    m_previousOverlayActive = false;
    m_overlayOpacity = 1.0f;
    m_dimInactivePieces = false;
    m_dimInactiveSquares = false;
}

void BoardView::setDragging(chess::Square from, Vector2 cursor) noexcept {
    m_dragFrom = from;
    m_dragCursor = cursor;
}

float BoardView::squareSize() const {
    float sq;
    boardOriginAndSquareSize(sq);
    return sq;
}

void BoardView::draw(const Theme& theme) const {
    if (!m_position || !m_sprites) return;

    float sq;
    Vector2 origin = boardOriginAndSquareSize(sq);

    // ----- squares -----
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            Rectangle r = squareRect(file, rank);
            bool light = ((file + rank) & 1) == 1;
            DrawRectangleRec(r, light ? theme.squareLight : theme.squareDark);
        }
    }

    // ----- last-move highlight -----
    if (m_lastMove.has_value()) {
        for (chess::Square s : {m_lastMove->from(), m_lastMove->to()}) {
            Rectangle r = squareRect(chess::fileOf(s), chess::rankOf(s));
            DrawRectangleRec(r, theme.lastMove);
        }
    }

    // ----- legal-target indicators (Lague-style dots / rings) -----
    if (m_legalTargets) {
        chess::Bitboard bb = m_legalTargets;
        while (bb) {
            int idx = chess::popLsb(bb);
            Rectangle r = squareRect(idx & 7, idx >> 3);
            chess::Piece occ = m_position->pieceAt(idx);
            Vector2 centre{r.x + r.width * 0.5f, r.y + r.height * 0.5f};
            if (occ.isNone()) {
                // Quiet move — small filled dot in the centre.
                DrawCircleV(centre, r.width * 0.16f, theme.legalDot);
            } else {
                // Capture — hollow ring hugging the piece's edge.
                float outer = r.width * 0.46f;
                float thickness = r.width * 0.07f;
                DrawRing(centre, outer - thickness, outer, 0.0f, 360.0f,
                         48, theme.legalCaptureRing);
            }
        }
    }

    // ----- selection -----
    if (m_selection.has_value()) {
        Rectangle r = squareRect(chess::fileOf(*m_selection),
                                 chess::rankOf(*m_selection));
        DrawRectangleRec(r, theme.selection);
    }

    // ----- check warning on king -----
    if (m_position->inCheck()) {
        chess::Square ks = m_position->kingSquare(m_position->sideToMove());
        if (ks != chess::Square::None) {
            Rectangle r = squareRect(chess::fileOf(ks), chess::rankOf(ks));
            DrawRectangleRec(r, theme.checkWarning);
        }
    }

    // ----- activation overlay (under pieces) -----
    auto drawActivationOverlay = [&](const std::array<float, 64>& overlay,
                                     float opacity) {
        opacity = clamp01(opacity) * kActivationOverlayMaxOpacity;
        if (opacity <= 0.001f) return;
        for (int idx = 0; idx < 64; ++idx) {
            float t = clamp01(overlay[static_cast<std::size_t>(idx)]);
            Rectangle r = squareRect(idx & 7, idx >> 3);
            if (m_dimInactiveSquares && t <= 0.001f) {
                Color tint = theme.overlayZero;
                tint.a = static_cast<unsigned char>(
                    static_cast<float>(tint.a) * opacity);
                DrawRectangleRec(r, tint);
                continue;
            }
            if (t <= 0.001f) continue;
            Color tint = lerpColor(theme.overlayZero, theme.overlayHot, t);
            tint.a = static_cast<unsigned char>(
                static_cast<float>(tint.a) * opacity);
            DrawRectangleRec(r, tint);
        }
    };
    if (m_overlayActive) {
        if (m_previousOverlayActive && m_overlayOpacity < 0.999f) {
            drawActivationOverlay(m_previousOverlay, 1.0f - m_overlayOpacity);
        }
        drawActivationOverlay(m_overlay, m_overlayOpacity);
    }

    // ----- pieces -----
    auto pieceTintFor = [&](int idx, chess::Piece piece) {
        if (!m_dimInactivePieces || !m_overlayActive) return WHITE;
        if (piece.type == chess::PieceType::King) return WHITE;
        float t = clamp01(m_overlay[static_cast<std::size_t>(idx)]);
        if (m_previousOverlayActive && m_overlayOpacity < 0.999f) {
            const float prev =
                clamp01(m_previousOverlay[static_cast<std::size_t>(idx)]);
            t = prev * (1.0f - m_overlayOpacity) + t * m_overlayOpacity;
        }
        t = t * t * (3.0f - 2.0f * t);
        return lerpColor(Color{104, 108, 116, 218}, WHITE, t);
    };
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            chess::Square s = chess::makeSquare(file, rank);
            if (m_dragFrom.has_value() && *m_dragFrom == s) continue;
            chess::Piece p = m_position->pieceAt(s);
            if (p.isNone()) continue;
            Rectangle r = squareRect(file, rank);
            const Texture2D& tex = m_sprites->textureFor(p);
            if (tex.id == 0) continue;
            Rectangle src{0, 0,
                static_cast<float>(tex.width),
                static_cast<float>(tex.height)};
            DrawTexturePro(tex, src, r, Vector2{0, 0}, 0.0f,
                           pieceTintFor(chess::squareIndex(s), p));
        }
    }

    // ----- dragged piece (drawn last so it sits above everything) -----
    if (m_dragFrom.has_value()) {
        chess::Piece p = m_position->pieceAt(*m_dragFrom);
        if (!p.isNone()) {
            const Texture2D& tex = m_sprites->textureFor(p);
            if (tex.id != 0) {
                Rectangle src{0, 0,
                    static_cast<float>(tex.width),
                    static_cast<float>(tex.height)};
                Rectangle dst{
                    m_dragCursor.x - sq * 0.5f,
                    m_dragCursor.y - sq * 0.5f,
                    sq, sq,
                };
                DrawTexturePro(tex, src, dst, Vector2{0, 0}, 0.0f, WHITE);
            }
        }
    }

    // ----- coordinate labels -----
    int fontSize = std::max(12, static_cast<int>(sq * 0.20f));
    for (int file = 0; file < 8; ++file) {
        char letter[2] = {static_cast<char>('a' + file), '\0'};
        int displayFile = m_flipped ? 7 - file : file;
        const int textW = measureText(theme, letter, fontSize);
        float x = origin.x + displayFile * sq + (sq - static_cast<float>(textW)) * 0.5f;
        float y = std::min(origin.y + 8.0f * sq + 7.0f,
                           m_bounds.y + m_bounds.height -
                               static_cast<float>(fontSize) - 2.0f);
        drawText(theme, letter, static_cast<int>(x), static_cast<int>(y),
                 fontSize, theme.coordLabel);
    }
    for (int rank = 0; rank < 8; ++rank) {
        char digit[2] = {static_cast<char>('1' + rank), '\0'};
        int displayRank = m_flipped ? rank : 7 - rank;
        const int textW = measureText(theme, digit, fontSize);
        float x = origin.x - static_cast<float>(textW) - 8.0f;
        x = std::max(m_bounds.x + 2.0f, x);
        float y = origin.y + displayRank * sq + (sq - static_cast<float>(fontSize)) * 0.5f;
        drawText(theme, digit, static_cast<int>(x), static_cast<int>(y),
                 fontSize, theme.coordLabel);
    }
}

}  // namespace cnnv::viz
