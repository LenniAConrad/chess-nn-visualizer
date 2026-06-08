#pragma once

/**
 * @file TreeView.h
 * @brief Full-window pan/zoom visualizer for the MCTS search tree.
 *
 * Mirrors chess-rtk's mcts/TreeGraphView + TreePanel: a tidy layered layout of
 * mini-board nodes coloured by value, bezier edges, dashed transposition links,
 * a highlighted principal variation, click-to-select, a zoom-to-cursor +
 * drag-pan camera, plus the tree controls: top-K branches per node,
 * transposition merging, batching sibling leaves into subtree blobs, and
 * per-ply layer guides.
 */

#include "chess/Piece.h"
#include "search/Mcts.h"
#include "viz/Theme.h"

#include <raylib.h>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cnnv::viz {

class PieceSprites;

/**
 * @brief Renders and navigates an MCTS search-tree snapshot.
 */
class TreeView {
   public:
    /** @brief Provides piece sprites for the mini-board nodes (non-owning). */
    void setSprites(const PieceSprites* sprites) { m_sprites = sprites; }

    /** @brief Sets the screen rectangle the tree is drawn within. */
    void setBounds(Rectangle r);

    /** @brief Replaces the tree snapshot and recomputes the layout. */
    void setSnapshot(const cnnv::search::Snapshot& snap);

    /** @brief Clears the tree. */
    void clear();

    /** @brief Handles camera + selection input for one frame. */
    void update();

    /** @brief Draws the tree (camera-transformed, scissor-clipped to bounds). */
    void draw(const Theme& theme) const;

    /** @brief Fits the whole tree into the view. */
    void fit();

    /** @brief Resets the camera to a root-anchored default. */
    void resetView();

    /** @name Tree controls */
    ///@{
    /** @brief Max children shown per node (0 = all). Re-lays out. */
    void setBranches(int branches);
    int branches() const noexcept { return m_branches; }

    /** @brief Toggles batching sibling leaves into subtree blobs. */
    void setBatchLeaves(bool on);
    bool batchLeaves() const noexcept { return m_batchLeaves; }

    /** @brief Toggles CRTK-style merge-by-position transposition layout. */
    void setMergeTranspositions(bool on);
    bool mergeTranspositions() const noexcept { return m_mergeTranspositions; }

    /** @brief Max plies shown from the root. Re-lays out. */
    void setDepth(int depth);
    int depth() const noexcept { return m_maxDepthShown; }

    /** @brief Toggles per-ply layer guides. */
    void setShowGuides(bool on);
    bool showGuides() const noexcept { return m_showGuides; }
    ///@}

    /** @brief Index of the selected node in the current snapshot, or -1. */
    int selectedNode() const noexcept { return m_selected; }

    /** @brief FEN of the selected node, or empty. */
    std::string selectedFen() const;

    /** @brief True if a new selection was made since the last call (consumes). */
    bool takeSelectionChanged() noexcept;

    /** @brief Distinct positions in the current snapshot, for the legend. */
    int uniquePositions() const noexcept { return m_uniquePositions; }

    /** @brief Dashed transposition edge count in the current display model. */
    int transpositionEdges() const noexcept { return m_transpositionCount; }

   private:
    // One node in the laid-out display tree. A display node is either a real
    // snapshot node (snap >= 0) or a "blob" summarizing batched sibling leaves.
    struct DisplayNode {
        bool blob = false;
        int snap = -1;          // snapshot index when !blob
        int blobCount = 0;      // leaves represented when blob
        long blobVisits = 0;    // summed visits when blob
        int depth = 0;
        bool onPv = false;
        bool transposed = false;
        std::vector<int> children;  // display-node indices
        float x = 0.0f;             // world left
        float y = 0.0f;             // world top
    };

    struct DisplayEdge {
        int from = -1;
        int to = -1;
        bool transposition = false;
    };

    void rebuildDisplay();
    int buildNode(int snapIdx, int depth);
    int buildBlob(const std::vector<int>& snapChildren, int depth);
    void rebuildEdges();
    float assignX(int displayIdx, float& nextLeafX);
    void anchorRoot();
    /** @brief World-space rectangle a node is drawn in. */
    Rectangle nodeRect(const DisplayNode& d) const;
    Camera2D camera() const noexcept;
    void zoomAt(Vector2 anchor, float factor) noexcept;
    int displayAtScreen(Vector2 screen) const;
    void drawMiniBoard(int snapIndex, Rectangle board, const Theme& theme) const;

    Rectangle m_bounds{0, 0, 0, 0};
    cnnv::search::Snapshot m_snap;
    const PieceSprites* m_sprites = nullptr;
    // Per-snapshot-node parsed board (a1=0 mailbox), parallel to m_snap.nodes.
    std::vector<std::array<cnnv::chess::Piece, 64>> m_boards;
    std::vector<DisplayNode> m_display;  // m_display[0] = root
    std::vector<DisplayEdge> m_edges;
    std::vector<DisplayEdge> m_extraEdges;
    std::unordered_map<std::uint64_t, int> m_signatureToDisplay;
    int m_maxDepth = 0;
    float m_worldW = 0.0f;
    float m_worldH = 0.0f;
    int m_uniquePositions = 0;
    int m_transpositionCount = 0;

    int m_branches = 4;          // top-K children per node (0 = all)
    int m_maxDepthShown = 14;    // max plies shown from the root
    bool m_mergeTranspositions = true;  // collapse repeated signatures
    bool m_batchLeaves = true;   // batch sibling leaves into blobs
    bool m_showGuides = true;    // per-ply layer guides

    float m_zoom = 1.0f;
    Vector2 m_pan{0.0f, 0.0f};
    bool m_panning = false;
    Vector2 m_panLast{0.0f, 0.0f};
    bool m_needFit = true;
    bool m_userNavigated = false;

    int m_selected = -1;
    bool m_selectionChanged = false;
};

}  // namespace cnnv::viz
