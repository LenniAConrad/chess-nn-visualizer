#include "viz/TreeView.h"

#include "chess/Fen.h"
#include "viz/PieceSprites.h"
#include "viz/TensorViz.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>

namespace cnnv::viz {

namespace {

constexpr float kNodeW = 96.0f;
constexpr float kBoardH = 96.0f;
constexpr float kCaptionH = 34.0f;
constexpr float kNodeH = kBoardH + kCaptionH;
constexpr float kHGap = 22.0f;
constexpr float kVGap = 54.0f;
constexpr float kMargin = 28.0f;
constexpr float kPartitionPad = 14.0f;
constexpr float kViewportGutter = 18.0f;
constexpr float kMinZoom = 0.01f;
constexpr float kMaxZoom = 8.0f;
constexpr float kReadableMinZoom = 0.48f;
constexpr float kDragSlop = 4.0f;
constexpr float kBoardPixelThreshold = 26.0f;
constexpr int kBlobThreshold = 2;
constexpr int kMaxDisplayNodes = 1000;

float rowStride() noexcept { return kNodeH + kVGap; }

Color qTint(float q, const Theme& theme) {
    const Color target = q >= 0.0f ? theme.accentGreen : theme.accentRed;
    const float t = 0.28f + 0.44f * std::clamp(std::fabs(q), 0.0f, 1.0f);
    return tensorviz::blend(theme.panelBackground, target, t);
}

float roundness(const Rectangle& r, float px = 10.0f) {
    return std::clamp(px / std::min(r.width, r.height), 0.0f, 0.5f);
}

void drawRoundedLines(Rectangle r, float round, int segments, float thickness,
                      Color color) {
#if defined(RAYLIB_VERSION_MAJOR) && defined(RAYLIB_VERSION_MINOR) && \
    (RAYLIB_VERSION_MAJOR > 5 ||                                      \
     (RAYLIB_VERSION_MAJOR == 5 && RAYLIB_VERSION_MINOR >= 5))
    DrawRectangleRoundedLinesEx(r, round, segments, thickness, color);
#else
    DrawRectangleRoundedLines(r, round, segments, thickness, color);
#endif
}

void compactCount(char* buf, std::size_t n, long value) {
    auto decimal = [&](double scaled, const char* suffix) {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%.1f", scaled);
        std::string text(tmp);
        if (text.size() > 2 && text.substr(text.size() - 2) == ".0") {
            text.resize(text.size() - 2);
        }
        std::snprintf(buf, n, "%s%s", text.c_str(), suffix);
    };
    const long absValue = std::labs(value);
    if (absValue >= 1000000000L) {
        decimal(static_cast<double>(value) / 1e9, "B");
    } else if (absValue >= 1000000L) {
        decimal(static_cast<double>(value) / 1e6, "M");
    } else if (absValue >= 1000L) {
        decimal(static_cast<double>(value) / 1e3, "k");
    } else {
        std::snprintf(buf, n, "%ld", value);
    }
}

Vector2 cubicPoint(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3,
                   float t) noexcept {
    const float u = 1.0f - t;
    const float uu = u * u;
    const float tt = t * t;
    const float uuu = uu * u;
    const float ttt = tt * t;
    return Vector2{uuu * p0.x + 3.0f * uu * t * p1.x +
                       3.0f * u * tt * p2.x + ttt * p3.x,
                   uuu * p0.y + 3.0f * uu * t * p1.y +
                       3.0f * u * tt * p2.y + ttt * p3.y};
}

void drawDashedBezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3,
                      float thickness, Color color) {
    constexpr int kSegments = 44;
    constexpr int kDash = 3;
    for (int i = 0; i < kSegments; ++i) {
        if (((i / kDash) & 1) != 0) continue;
        const float t0 = static_cast<float>(i) / static_cast<float>(kSegments);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(kSegments);
        DrawLineEx(cubicPoint(p0, p1, p2, p3, t0),
                   cubicPoint(p0, p1, p2, p3, t1), thickness, color);
    }
}

}  // namespace

void TreeView::setBounds(Rectangle r) {
    m_bounds = r;
    if (!m_userNavigated && !m_display.empty()) {
        anchorRoot();
    }
}

void TreeView::clear() {
    m_snap = cnnv::search::Snapshot{};
    m_boards.clear();
    m_display.clear();
    m_edges.clear();
    m_extraEdges.clear();
    m_signatureToDisplay.clear();
    m_worldW = 0.0f;
    m_worldH = 0.0f;
    m_maxDepth = 0;
    m_uniquePositions = 0;
    m_transpositionCount = 0;
    m_selected = -1;
    m_selectionChanged = false;
    m_userNavigated = false;
    m_needFit = true;
}

void TreeView::setSnapshot(const cnnv::search::Snapshot& snap) {
    const bool wasEmpty = m_snap.nodes.empty();
    m_snap = snap;
    if (m_selected >= static_cast<int>(m_snap.nodes.size())) m_selected = -1;

    m_boards.assign(m_snap.nodes.size(), std::array<cnnv::chess::Piece, 64>{});
    std::unordered_set<std::uint64_t> uniqueSignatures;
    for (std::size_t i = 0; i < m_snap.nodes.size(); ++i) {
        if (m_snap.nodes[i].signature != 0) {
            uniqueSignatures.insert(m_snap.nodes[i].signature);
        }
        const auto parsed = cnnv::chess::Fen::parse(m_snap.nodes[i].fen);
        if (!parsed.has_value()) continue;
        for (int sq = 0; sq < 64; ++sq) {
            m_boards[i][static_cast<std::size_t>(sq)] = parsed->pieceAt(sq);
        }
    }
    m_uniquePositions = uniqueSignatures.empty()
                            ? static_cast<int>(m_snap.nodes.size())
                            : static_cast<int>(uniqueSignatures.size());

    rebuildDisplay();
    if (wasEmpty || !m_userNavigated) {
        anchorRoot();
        m_needFit = false;
    }
}

void TreeView::setBranches(int branches) {
    branches = std::max(0, branches);
    if (branches == m_branches) return;
    m_branches = branches;
    rebuildDisplay();
    if (!m_userNavigated) anchorRoot();
}

void TreeView::setDepth(int depth) {
    depth = std::clamp(depth, 1, 256);
    if (depth == m_maxDepthShown) return;
    m_maxDepthShown = depth;
    rebuildDisplay();
    if (!m_userNavigated) anchorRoot();
}

void TreeView::setBatchLeaves(bool on) {
    if (on == m_batchLeaves) return;
    m_batchLeaves = on;
    rebuildDisplay();
    if (!m_userNavigated) anchorRoot();
}

void TreeView::setMergeTranspositions(bool on) {
    if (on == m_mergeTranspositions) return;
    m_mergeTranspositions = on;
    rebuildDisplay();
    if (!m_userNavigated) anchorRoot();
}

void TreeView::setShowGuides(bool on) { m_showGuides = on; }

int TreeView::buildBlob(const std::vector<int>& snapChildren, int depth) {
    if (static_cast<int>(m_display.size()) >= kMaxDisplayNodes) return -1;
    DisplayNode blob;
    blob.blob = true;
    blob.depth = depth;
    blob.blobCount = static_cast<int>(snapChildren.size());
    for (int snapIdx : snapChildren) {
        const auto& n = m_snap.nodes[static_cast<std::size_t>(snapIdx)];
        blob.blobVisits += n.visits;
        blob.onPv = blob.onPv || n.onPv;
    }
    const int idx = static_cast<int>(m_display.size());
    m_display.push_back(std::move(blob));
    m_maxDepth = std::max(m_maxDepth, depth);
    return idx;
}

int TreeView::buildNode(int snapIdx, int depth) {
    if (snapIdx < 0 || snapIdx >= static_cast<int>(m_snap.nodes.size())) return -1;
    if (static_cast<int>(m_display.size()) >= kMaxDisplayNodes) return -1;

    DisplayNode node;
    node.snap = snapIdx;
    node.depth = depth;
    node.onPv = m_snap.nodes[static_cast<std::size_t>(snapIdx)].onPv;
    const int displayIdx = static_cast<int>(m_display.size());
    m_display.push_back(std::move(node));
    const std::uint64_t signature =
        m_snap.nodes[static_cast<std::size_t>(snapIdx)].signature;
    if (m_mergeTranspositions && signature != 0) {
        m_signatureToDisplay.emplace(signature, displayIdx);
    }
    m_maxDepth = std::max(m_maxDepth, depth);

    if (depth >= m_maxDepthShown ||
        static_cast<int>(m_display.size()) >= kMaxDisplayNodes) {
        return displayIdx;
    }

    std::vector<int> kids = m_snap.nodes[static_cast<std::size_t>(snapIdx)].children;
    std::sort(kids.begin(), kids.end(), [&](int a, int b) {
        const auto& na = m_snap.nodes[static_cast<std::size_t>(a)];
        const auto& nb = m_snap.nodes[static_cast<std::size_t>(b)];
        if (na.onPv != nb.onPv) return na.onPv;
        if (na.visits != nb.visits) return na.visits > nb.visits;
        return na.prior > nb.prior;
    });
    if (m_branches > 0 && static_cast<int>(kids.size()) > m_branches) {
        kids.resize(static_cast<std::size_t>(m_branches));
    }

    std::vector<int> leafKids;
    std::vector<int> childDisplays;
    for (int childSnap : kids) {
        const auto& child = m_snap.nodes[static_cast<std::size_t>(childSnap)];
        if (m_mergeTranspositions && child.signature != 0) {
            const auto found = m_signatureToDisplay.find(child.signature);
            if (found != m_signatureToDisplay.end()) {
                const int target = found->second;
                if (target != displayIdx) {
                    m_extraEdges.push_back(DisplayEdge{displayIdx, target, true});
                    m_display[static_cast<std::size_t>(target)].transposed = true;
                }
                continue;
            }
        }
        const bool forcedLeaf = depth + 1 >= m_maxDepthShown;
        const bool leaf = child.children.empty() || forcedLeaf;
        if (m_batchLeaves && leaf && !child.onPv) {
            leafKids.push_back(childSnap);
        } else {
            const int childDisplay = buildNode(childSnap, depth + 1);
            if (childDisplay >= 0) childDisplays.push_back(childDisplay);
        }
    }

    if (m_batchLeaves && static_cast<int>(leafKids.size()) >= kBlobThreshold) {
        const int blob = buildBlob(leafKids, depth + 1);
        if (blob >= 0) childDisplays.push_back(blob);
    } else {
        for (int childSnap : leafKids) {
            const auto& child = m_snap.nodes[static_cast<std::size_t>(childSnap)];
            if (m_mergeTranspositions && child.signature != 0) {
                const auto found = m_signatureToDisplay.find(child.signature);
                if (found != m_signatureToDisplay.end()) {
                    const int target = found->second;
                    if (target != displayIdx) {
                        m_extraEdges.push_back(DisplayEdge{displayIdx, target, true});
                        m_display[static_cast<std::size_t>(target)].transposed = true;
                    }
                    continue;
                }
            }
            const int childDisplay = buildNode(childSnap, depth + 1);
            if (childDisplay >= 0) childDisplays.push_back(childDisplay);
        }
    }

    m_display[static_cast<std::size_t>(displayIdx)].children = std::move(childDisplays);
    return displayIdx;
}

float TreeView::assignX(int displayIdx, float& nextLeafX) {
    DisplayNode& node = m_display[static_cast<std::size_t>(displayIdx)];
    node.y = static_cast<float>(node.depth) * rowStride() + kMargin;

    if (node.children.empty()) {
        node.x = nextLeafX;
        nextLeafX += kNodeW + kHGap;
        return node.x + kNodeW * 0.5f;
    }

    const std::vector<int> kids = node.children;
    float first = 0.0f;
    float last = 0.0f;
    for (std::size_t i = 0; i < kids.size(); ++i) {
        const float childCx = assignX(kids[i], nextLeafX);
        if (i == 0) first = childCx;
        last = childCx;
    }
    const float cx = (first + last) * 0.5f;
    node.x = cx - kNodeW * 0.5f;
    return cx;
}

void TreeView::rebuildEdges() {
    m_edges.clear();
    m_transpositionCount = 0;
    for (std::size_t i = 0; i < m_display.size(); ++i) {
        const DisplayNode& parent = m_display[i];
        for (int child : parent.children) {
            m_edges.push_back(DisplayEdge{static_cast<int>(i), child, false});
        }
    }
    std::unordered_set<std::uint64_t> seenTranspositions;
    for (const DisplayEdge& edge : m_extraEdges) {
        if (edge.from < 0 || edge.to < 0 || edge.from == edge.to) continue;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge.from))
             << 32) |
            static_cast<std::uint32_t>(edge.to);
        if (seenTranspositions.insert(key).second) {
            m_edges.push_back(edge);
            ++m_transpositionCount;
        }
    }
}

void TreeView::rebuildDisplay() {
    m_display.clear();
    m_edges.clear();
    m_extraEdges.clear();
    m_signatureToDisplay.clear();
    m_maxDepth = 0;
    m_transpositionCount = 0;
    m_worldW = 0.0f;
    m_worldH = 0.0f;
    if (m_snap.nodes.empty()) return;

    buildNode(0, 0);
    if (m_display.empty()) return;

    float nextLeafX = kMargin;
    assignX(0, nextLeafX);
    rebuildEdges();

    float minX = m_display[0].x;
    float maxX = m_display[0].x + kNodeW;
    for (const DisplayNode& n : m_display) {
        minX = std::min(minX, n.x);
        maxX = std::max(maxX, n.x + kNodeW);
    }
    if (minX < kMargin) {
        const float shift = kMargin - minX;
        for (DisplayNode& n : m_display) n.x += shift;
        maxX += shift;
    }
    m_worldW = maxX + kMargin;
    m_worldH = static_cast<float>(m_maxDepth) * rowStride() + kNodeH +
               2.0f * kMargin;
}

Rectangle TreeView::nodeRect(const DisplayNode& d) const {
    return Rectangle{d.x, d.y, kNodeW, kNodeH};
}

Camera2D TreeView::camera() const noexcept {
    Camera2D cam{};
    cam.offset = m_pan;
    cam.target = Vector2{0.0f, 0.0f};
    cam.rotation = 0.0f;
    cam.zoom = m_zoom;
    return cam;
}

void TreeView::anchorRoot() {
    if (m_display.empty() || m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) {
        return;
    }
    const float fitZoom = std::min(m_bounds.width / std::max(1.0f, m_worldW),
                                   m_bounds.height / std::max(1.0f, m_worldH));
    m_zoom = std::clamp(std::max(kReadableMinZoom, std::min(1.0f, fitZoom)),
                        kMinZoom, kMaxZoom);
    const Rectangle root = nodeRect(m_display[0]);
    m_pan.x = m_bounds.x + m_bounds.width * 0.5f -
              (root.x + root.width * 0.5f) * m_zoom;
    m_pan.y = m_bounds.y + 20.0f - root.y * m_zoom;
    m_pan.x = std::max(m_pan.x, m_bounds.x + kViewportGutter);
}

void TreeView::fit() {
    if (m_worldW <= 0.0f || m_worldH <= 0.0f ||
        m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) {
        return;
    }
    const float sx = m_bounds.width / std::max(1.0f, m_worldW);
    const float sy = m_bounds.height / std::max(1.0f, m_worldH);
    m_zoom = std::clamp(std::min(sx, sy) * 0.94f, kMinZoom, kMaxZoom);
    m_pan.x = m_bounds.x + (m_bounds.width - m_worldW * m_zoom) * 0.5f;
    const float freeY = m_bounds.height - m_worldH * m_zoom;
    m_pan.y = m_bounds.y + std::max(14.0f, freeY * 0.5f);
    m_userNavigated = true;
}

void TreeView::resetView() {
    m_userNavigated = false;
    anchorRoot();
}

void TreeView::zoomAt(Vector2 anchor, float factor) noexcept {
    const float next = std::clamp(m_zoom * factor, kMinZoom, kMaxZoom);
    const float ratio = next / std::max(kMinZoom, m_zoom);
    m_pan.x = anchor.x - (anchor.x - m_pan.x) * ratio;
    m_pan.y = anchor.y - (anchor.y - m_pan.y) * ratio;
    m_zoom = next;
}

int TreeView::displayAtScreen(Vector2 screen) const {
    const Vector2 world = GetScreenToWorld2D(screen, camera());
    for (std::size_t i = 0; i < m_display.size(); ++i) {
        const DisplayNode& n = m_display[i];
        if (n.blob) continue;
        const Rectangle r = nodeRect(n);
        if (CheckCollisionPointRec(world, r)) return n.snap;
    }
    return -1;
}

void TreeView::update() {
    if (m_needFit && !m_display.empty()) {
        anchorRoot();
        m_needFit = false;
    }

    const Vector2 mouse = GetMousePosition();
    const bool over = CheckCollisionPointRec(mouse, m_bounds);
    if (over) {
        const float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            zoomAt(mouse, wheel > 0.0f ? 1.12f : 1.0f / 1.12f);
            m_userNavigated = true;
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            zoomAt(Vector2{m_bounds.x + m_bounds.width * 0.5f,
                           m_bounds.y + m_bounds.height * 0.5f}, 1.12f);
            m_userNavigated = true;
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            zoomAt(Vector2{m_bounds.x + m_bounds.width * 0.5f,
                           m_bounds.y + m_bounds.height * 0.5f}, 1.0f / 1.12f);
            m_userNavigated = true;
        }
        if (IsKeyPressed(KEY_ZERO) || IsKeyPressed(KEY_KP_0)) {
            resetView();
        }
    }

    if (over && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        m_panning = false;
        m_panLast = mouse;
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        if (!m_panning) {
            const float dx = mouse.x - m_panLast.x;
            const float dy = mouse.y - m_panLast.y;
            if (dx * dx + dy * dy > kDragSlop * kDragSlop) m_panning = true;
        }
        if (m_panning) {
            m_pan.x += mouse.x - m_panLast.x;
            m_pan.y += mouse.y - m_panLast.y;
            m_panLast = mouse;
            m_userNavigated = true;
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        if (!m_panning && over) {
            const int hit = displayAtScreen(mouse);
            if (hit >= 0) {
                m_selected = hit;
                m_selectionChanged = true;
            }
        }
        m_panning = false;
    }
}

bool TreeView::takeSelectionChanged() noexcept {
    const bool changed = m_selectionChanged;
    m_selectionChanged = false;
    return changed;
}

std::string TreeView::selectedFen() const {
    if (m_selected < 0 || m_selected >= static_cast<int>(m_snap.nodes.size())) {
        return std::string();
    }
    return m_snap.nodes[static_cast<std::size_t>(m_selected)].fen;
}

void TreeView::drawMiniBoard(int snapIndex, Rectangle board,
                             const Theme& theme) const {
    const float cell = board.width / 8.0f;
    const auto& squares = m_boards[static_cast<std::size_t>(snapIndex)];
    const std::uint16_t moveRaw =
        m_snap.nodes[static_cast<std::size_t>(snapIndex)].moveRaw;
    const int fromHighlight =
        moveRaw == cnnv::chess::Move::kNoMove ? -1 : (moveRaw & 0x3F);
    const int toHighlight =
        moveRaw == cnnv::chess::Move::kNoMove ? -1 : ((moveRaw >> 6) & 0x3F);

    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            const int row = 7 - rank;
            const Rectangle sq{board.x + static_cast<float>(file) * cell,
                               board.y + static_cast<float>(row) * cell,
                               cell, cell};
            const bool light = ((file + rank) & 1) != 0;
            DrawRectangleRec(sq, light ? theme.squareLight : theme.squareDark);

            const int idx = rank * 8 + file;
            if (idx == fromHighlight || idx == toHighlight) {
                DrawRectangleRec(sq, tensorviz::withAlpha(theme.lastMove, 145));
            }

            const cnnv::chess::Piece p = squares[static_cast<std::size_t>(idx)];
            if (!p.isNone() && m_sprites != nullptr) {
                const Texture2D& tex = m_sprites->textureFor(p);
                if (tex.id != 0) {
                    DrawTexturePro(
                        tex,
                        Rectangle{0.0f, 0.0f, static_cast<float>(tex.width),
                                  static_cast<float>(tex.height)},
                        sq, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
                }
            }
        }
    }
}

void TreeView::draw(const Theme& theme) const {
    if (m_bounds.width <= 0.0f || m_bounds.height <= 0.0f) return;
    DrawRectangleRec(m_bounds, theme.background);

    if (m_display.empty()) {
        drawText(theme, "No search yet - press Start to run PUCT.",
                 static_cast<int>(m_bounds.x + 24.0f),
                 static_cast<int>(m_bounds.y + 24.0f), 18, theme.textMuted);
        return;
    }

    const Camera2D cam = camera();
    const float invZoom = 1.0f / std::max(kMinZoom, m_zoom);
    const Rectangle viewWorld{
        (m_bounds.x - m_pan.x) * invZoom - 80.0f,
        (m_bounds.y - m_pan.y) * invZoom - 80.0f,
        m_bounds.width * invZoom + 160.0f,
        m_bounds.height * invZoom + 160.0f,
    };
    auto visible = [&](const DisplayNode& n) {
        const Rectangle r = nodeRect(n);
        return r.x + r.width >= viewWorld.x &&
               r.x <= viewWorld.x + viewWorld.width &&
               r.y + r.height >= viewWorld.y &&
               r.y <= viewWorld.y + viewWorld.height;
    };
    std::unordered_set<int> selectedPath;
    if (m_selected >= 0 && m_selected < static_cast<int>(m_snap.nodes.size())) {
        for (int cur = m_selected; cur >= 0;) {
            selectedPath.insert(cur);
            cur = m_snap.nodes[static_cast<std::size_t>(cur)].parent;
        }
    }

    BeginScissorMode(static_cast<int>(m_bounds.x), static_cast<int>(m_bounds.y),
                     static_cast<int>(m_bounds.width),
                     static_cast<int>(m_bounds.height));
    BeginMode2D(cam);

    if (m_showGuides) {
        if (!m_display.empty()) {
            const DisplayNode& root = m_display[0];
            std::vector<std::pair<float, float>> partitions;
            partitions.reserve(root.children.size());
            for (int child : root.children) {
                float minX = nodeRect(m_display[static_cast<std::size_t>(child)]).x;
                float maxX = minX + kNodeW;
                std::vector<int> stack{child};
                for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
                    const DisplayNode& n =
                        m_display[static_cast<std::size_t>(stack[cursor])];
                    const Rectangle nr = nodeRect(n);
                    minX = std::min(minX, nr.x);
                    maxX = std::max(maxX, nr.x + nr.width);
                    for (int kid : n.children) stack.push_back(kid);
                }
                partitions.push_back({std::max(0.0f, minX - kPartitionPad),
                                      maxX + kPartitionPad});
            }
            std::sort(partitions.begin(), partitions.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
            auto drawBoundary = [&](float x) {
                if (x < viewWorld.x - 2.0f ||
                    x > viewWorld.x + viewWorld.width + 2.0f) {
                    return;
                }
                DrawLineEx(Vector2{x, viewWorld.y},
                           Vector2{x, viewWorld.y + viewWorld.height},
                           1.0f * invZoom,
                           tensorviz::withAlpha(theme.panelBorder, 95));
            };
            if (!partitions.empty()) {
                drawBoundary(partitions.front().first);
                for (std::size_t i = 0; i + 1 < partitions.size(); ++i) {
                    drawBoundary((partitions[i].second + partitions[i + 1].first) *
                                 0.5f);
                }
                drawBoundary(partitions.back().second);
            }
        }
        for (int depth = 1; depth <= m_maxDepth; ++depth) {
            const float y = static_cast<float>(depth) * rowStride() + kMargin -
                            kVGap * 0.5f;
            DrawLineEx(Vector2{0.0f, y}, Vector2{m_worldW, y}, 1.0f * invZoom,
                       tensorviz::withAlpha(theme.panelBorder, 110));
            char label[16];
            std::snprintf(label, sizeof(label), "ply %d", depth);
            drawText(theme, label, static_cast<int>(6.0f),
                     static_cast<int>(y - 16.0f), 12,
                     tensorviz::withAlpha(theme.textDim, 205));
        }
    }

    for (const DisplayEdge& edge : m_edges) {
        if (edge.from < 0 || edge.to < 0) continue;
        const DisplayNode& parent = m_display[static_cast<std::size_t>(edge.from)];
        const DisplayNode& child = m_display[static_cast<std::size_t>(edge.to)];
        if (!visible(parent) && !visible(child)) continue;

        const Rectangle pr = nodeRect(parent);
        const Rectangle cr = nodeRect(child);
        const Vector2 from{pr.x + pr.width * 0.5f, pr.y + pr.height};
        const Vector2 to{cr.x + cr.width * 0.5f, cr.y};
        const float midY = (from.y + to.y) * 0.5f;
        const Vector2 c1{from.x, midY};
        const Vector2 c2{to.x, midY};
        const bool pvEdge = parent.onPv && child.onPv && !child.blob;
        const bool selectedEdge =
            !edge.transposition && parent.snap >= 0 && child.snap >= 0 &&
            selectedPath.count(parent.snap) != 0 &&
            selectedPath.count(child.snap) != 0 &&
            m_snap.nodes[static_cast<std::size_t>(child.snap)].parent == parent.snap;
        if (edge.transposition) {
            drawDashedBezier(from, c1, c2, to, 1.45f * invZoom,
                             tensorviz::withAlpha(theme.accentMagenta, 205));
        } else {
            const Color col =
                selectedEdge ? tensorviz::withAlpha(theme.accentGreen, 245)
                : pvEdge     ? tensorviz::withAlpha(theme.accentBlue, 235)
                             : tensorviz::withAlpha(theme.textDim, 125);
            DrawSplineSegmentBezierCubic(from, c1, c2, to,
                                         (selectedEdge ? 2.8f
                                          : pvEdge      ? 2.6f
                                                        : 1.35f) *
                                             invZoom,
                                         col);
        }
    }

    auto fontPx = [](float base, int lo, int hi) {
        return std::clamp(static_cast<int>(std::round(base)), lo, hi);
    };
    auto outerRing = [&](const Rectangle& bounds, float gapPx, float strokePx,
                         Color color) {
        const float stroke = std::max(0.05f, strokePx * invZoom);
        const float outer = gapPx * invZoom + stroke * 0.5f;
        const Rectangle ring{bounds.x - outer, bounds.y - outer,
                             bounds.width + 2.0f * outer,
                             bounds.height + 2.0f * outer};
        drawRoundedLines(ring, roundness(ring, 7.0f * invZoom), 8, stroke,
                         color);
    };

    for (std::size_t i = 0; i < m_display.size(); ++i) {
        const DisplayNode& d = m_display[i];
        if (!visible(d)) continue;
        const Rectangle r = nodeRect(d);
        const bool boardMode = r.width * m_zoom >= kBoardPixelThreshold;
        const bool showText = r.width * m_zoom >= 40.0f;

        if (d.blob) {
            for (int layer = 2; layer >= 0; --layer) {
                const float off = static_cast<float>(layer) * 3.0f;
                const Rectangle card{r.x + off, r.y + off,
                                     r.width - off, r.height - off};
                DrawRectangleRounded(card, roundness(card), 8,
                                     tensorviz::blend(theme.panelBackground,
                                                      theme.buttonIdle, 0.55f));
                drawRoundedLines(card, roundness(card), 8, 1.0f * invZoom,
                                 theme.panelBorder);
            }
            if (showText) {
                char leaves[24];
                std::snprintf(leaves, sizeof(leaves), "%d leaves", d.blobCount);
                drawText(theme, leaves, static_cast<int>(r.x + 10.0f),
                         static_cast<int>(r.y + r.height * 0.5f - 14.0f),
                         12, theme.textPrimary);
                char count[24];
                compactCount(count, sizeof(count), d.blobVisits);
                char total[32];
                std::snprintf(total, sizeof(total), "sum N=%s", count);
                drawText(theme, total, static_cast<int>(r.x + 10.0f),
                         static_cast<int>(r.y + r.height * 0.5f + 3.0f),
                         10, theme.textMuted);
            }
            continue;
        }

        const auto& n = m_snap.nodes[static_cast<std::size_t>(d.snap)];
        const Color accent = n.q >= 0.0f ? theme.accentGreen : theme.accentRed;
        if (boardMode) {
            const Rectangle board{r.x, r.y, kNodeW, kBoardH};
            drawMiniBoard(d.snap, board, theme);
            const Rectangle cap{r.x, r.y + kBoardH, kNodeW, kCaptionH};
            DrawRectangleRec(cap, tensorviz::withAlpha(theme.panelBackground, 245));
            DrawRectangleRec(Rectangle{cap.x, cap.y, 3.0f, cap.height}, accent);
            if (showText) {
                const char* moveLabel = n.parent < 0 ? "root" : n.uci.c_str();
                drawText(theme, moveLabel, static_cast<int>(cap.x + 7.0f),
                         static_cast<int>(cap.y + 3.0f),
                         fontPx(12.0f, 8, 12), theme.textPrimary);
                char count[24];
                compactCount(count, sizeof(count), n.visits);
                char stats[48];
                std::snprintf(stats, sizeof(stats), "N=%s  Q%+.2f", count,
                              static_cast<double>(n.q));
                drawText(theme, stats, static_cast<int>(cap.x + 7.0f),
                         static_cast<int>(cap.y + 18.0f),
                         fontPx(9.0f, 7, 9), theme.textMuted);
            }
        } else {
            DrawRectangleRounded(r, roundness(r), 8,
                                 tensorviz::withAlpha(qTint(n.q, theme), 235));
            if (showText) {
                const char* moveLabel = n.parent < 0 ? "root" : n.uci.c_str();
                drawText(theme, moveLabel, static_cast<int>(r.x + 7.0f),
                         static_cast<int>(r.y + r.height * 0.5f - 7.0f),
                         fontPx(12.0f, 8, 12), theme.textPrimary);
            }
        }

        if (d.transposed) {
            const float dot = 5.0f * invZoom;
            DrawCircleV(Vector2{r.x + r.width - 8.0f * invZoom,
                                r.y + 8.0f * invZoom},
                        dot, tensorviz::withAlpha(theme.accentMagenta, 230));
        }
        if (d.snap == m_selected) {
            outerRing(r, 5.0f, 2.6f, theme.selection);
        } else if (d.snap >= 0 && selectedPath.count(d.snap) != 0) {
            outerRing(r, 7.0f, 1.8f,
                      tensorviz::withAlpha(theme.accentGreen, 225));
        } else if (d.onPv) {
            outerRing(r, 5.0f, 1.8f,
                      tensorviz::withAlpha(theme.accentBlue, 220));
        }
    }

    EndMode2D();

    char legend[96];
    std::snprintf(legend, sizeof(legend), "%d positions | %d transpositions | %.0f%%",
                  m_uniquePositions, m_transpositionCount,
                  static_cast<double>(m_zoom * 100.0f));
    const int legendFont = 10;
    const int legendW = measureText(theme, legend, legendFont) + 18;
    const Rectangle legendBox{m_bounds.x + m_bounds.width -
                                  static_cast<float>(legendW) - 12.0f,
                              m_bounds.y + m_bounds.height - 28.0f,
                              static_cast<float>(legendW), 20.0f};
    DrawRectangleRounded(legendBox, roundness(legendBox, 10.0f), 8,
                         tensorviz::withAlpha(theme.panelBackground, 230));
    drawText(theme, legend, static_cast<int>(legendBox.x + 9.0f),
             static_cast<int>(legendBox.y + 5.0f), legendFont,
             theme.textMuted);
    EndScissorMode();
}

}  // namespace cnnv::viz
