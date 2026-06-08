#include "viz/App.h"

#include "viz/TensorViz.h"
#include "chess/Fen.h"
#include "chess/eval/Classical.h"
#include "chess/MoveGenerator.h"
#include "chess/MoveList.h"
#include "io/FenIo.h"
#include "nn/lc0_bt4/Bt4Network.h"
#include "nn/lc0_cnn/Lc0CnnNetwork.h"
#include "nn/nnue/NnueNetwork.h"
#include <raylib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cnnv::viz {

namespace chess = cnnv::chess;

namespace {

constexpr int kSearchMate = 100000;
constexpr int kMateDetectWindow = 2048;
constexpr int kSearchInf = 1000000;

// Codepoints loaded into the TTF atlas. ASCII covers all SAN, FEN and most UI
// strings; em-dash + arrow glyphs are needed for the status banner
// ("Checkmate — White wins") and the (rare) UI hints. Keep the list small —
// every codepoint expands the atlas texture.
std::vector<int> uiCodepoints() {
    std::vector<int> cps;
    cps.reserve(96 + 8);
    for (int c = 32; c < 127; ++c) cps.push_back(c);
    cps.push_back(0x00B0);  // ° (degree, occasionally useful)
    cps.push_back(0x00B7);  // · middle dot (separators)
    cps.push_back(0x03A3);  // Σ (sum, tree blob labels)
    cps.push_back(0x2212);  // − minus sign
    cps.push_back(0x2014);  // — em dash
    cps.push_back(0x2013);  // – en dash
    cps.push_back(0x2026);  // … ellipsis
    cps.push_back(0x2190);  // ← left arrow
    cps.push_back(0x2191);  // ↑ up arrow
    cps.push_back(0x2192);  // → right arrow
    cps.push_back(0x2193);  // ↓ down arrow
    return cps;
}

int evaluateSearchPosition(const chess::Position& pos) {
    int score = 0;
    for (int sq = 0; sq < 64; ++sq) {
        const chess::Piece p = pos.pieceAt(sq);
        if (p.isNone()) continue;

        const int file = sq & 7;
        const int rank = sq >> 3;
        const int center =
            7 - (std::abs(file * 2 - 7) + std::abs(rank * 2 - 7)) / 2;
        const int advance = p.color == chess::Color::White ? rank : 7 - rank;

        int value = chess::centipawnValue(p.type);
        switch (p.type) {
            case chess::PieceType::Pawn:   value += advance * 7; break;
            case chess::PieceType::Knight: value += center * 8;  break;
            case chess::PieceType::Bishop: value += center * 5;  break;
            case chess::PieceType::Rook:   value += advance * 2; break;
            case chess::PieceType::Queen:  value += center * 2;  break;
            case chess::PieceType::King:
            case chess::PieceType::None: break;
        }
        score += p.color == chess::Color::White ? value : -value;
    }
    return pos.sideToMove() == chess::Color::White ? score : -score;
}

struct SearchConfig {
    const char* name = "engine";
    int maxDepth = 6;
    int visualPauseMs = 20;
    std::uint64_t nodesPerIteration = 100000;
    std::size_t evalCacheLimit = 16384;
};

int promotionBonus(chess::Move m) {
    switch (m.promotion()) {
        case chess::Move::Promotion::Queen:  return 9000;
        case chess::Move::Promotion::Rook:   return 5000;
        case chess::Move::Promotion::Bishop: return 3200;
        case chess::Move::Promotion::Knight: return 3200;
        case chess::Move::Promotion::None:   return 0;
    }
    return 0;
}

int moveOrderingScore(const chess::Position& pos,
                      chess::Move move,
                      chess::Move ttMove) {
    if (!ttMove.isNone() && move == ttMove) return 10000000;

    int score = promotionBonus(move);
    const chess::Piece moving = pos.pieceAt(move.from());
    chess::Piece captured = pos.pieceAt(move.to());
    if (captured.isNone() &&
        moving.type == chess::PieceType::Pawn &&
        move.to() == pos.epSquare() &&
        chess::fileOf(move.from()) != chess::fileOf(move.to())) {
        captured = chess::Piece{chess::other(moving.color), chess::PieceType::Pawn};
    }
    if (!captured.isNone()) {
        score += 100000
               + 10 * chess::centipawnValue(captured.type)
               - chess::centipawnValue(moving.type);
    }

    const int to = chess::squareIndex(move.to());
    const int file = to & 7;
    const int rank = to >> 3;
    score += 7 - (std::abs(file * 2 - 7) + std::abs(rank * 2 - 7)) / 2;
    return score;
}

std::vector<chess::Move> orderedLegalMoves(chess::Position& pos,
                                           chess::Move ttMove) {
    chess::MoveList list;
    chess::MoveGenerator::generateLegal(pos, list);
    std::vector<chess::Move> moves(list.begin(), list.end());
    std::stable_sort(moves.begin(), moves.end(),
        [&](chess::Move a, chess::Move b) {
            return moveOrderingScore(pos, a, ttMove) >
                   moveOrderingScore(pos, b, ttMove);
        });
    return moves;
}

bool isEnPassantMove(const chess::Position& pos, chess::Move move,
                     chess::Piece moving) {
    return moving.type == chess::PieceType::Pawn &&
           move.to() == pos.epSquare() &&
           chess::fileOf(move.from()) != chess::fileOf(move.to());
}

bool isCastleMove(const chess::Position& pos, chess::Move move,
                  chess::Piece moving) {
    (void)pos;
    return moving.type == chess::PieceType::King &&
           std::abs(chess::squareIndex(move.to()) -
                    chess::squareIndex(move.from())) == 2;
}

chess::Piece capturedPieceFor(const chess::Position& pos, chess::Move move,
                              chess::Piece moving) {
    chess::Piece captured = pos.pieceAt(move.to());
    if (captured.isNone() && isEnPassantMove(pos, move, moving)) {
        captured = chess::Piece{chess::other(moving.color),
                                chess::PieceType::Pawn};
    }
    return captured;
}

int mctsPromotionValue(chess::Move::Promotion p) {
    switch (p) {
        case chess::Move::Promotion::Queen:  return 900;
        case chess::Move::Promotion::Rook:   return 500;
        case chess::Move::Promotion::Bishop: return 300;
        case chess::Move::Promotion::Knight: return 300;
        case chess::Move::Promotion::None:   return 0;
    }
    return 0;
}

int centerScoreForSquare(chess::Square sq) {
    const int file = chess::fileOf(sq);
    const int rank = chess::rankOf(sq);
    return 14 - 2 * (std::abs(2 * file - 7) + std::abs(2 * rank - 7));
}

int quietMovePrior(const chess::Position& pos, chess::Move move) {
    if (move.promotion() != chess::Move::Promotion::None) return 0;
    const chess::Piece moving = pos.pieceAt(move.from());
    if (moving.isNone()) return 0;
    if (isCastleMove(pos, move, moving)) return 30000;
    const chess::Piece captured = capturedPieceFor(pos, move, moving);
    if (!captured.isNone()) return 0;

    int bonus = 0;
    switch (moving.type) {
        case chess::PieceType::Knight:
        case chess::PieceType::Bishop:
            bonus += (centerScoreForSquare(move.to()) -
                      centerScoreForSquare(move.from())) * 256;
            if (chess::rankOf(move.from()) ==
                    (moving.color == chess::Color::White ? 0 : 7) &&
                chess::fileOf(move.to()) != 0 && chess::fileOf(move.to()) != 7) {
                bonus += 8000;
            }
            break;
        case chess::PieceType::Rook:
            bonus += (centerScoreForSquare(move.to()) -
                      centerScoreForSquare(move.from())) * 96;
            break;
        case chess::PieceType::Pawn: {
            const int relativeRank = moving.color == chess::Color::White
                                         ? chess::rankOf(move.to())
                                         : 7 - chess::rankOf(move.to());
            bonus += std::max(0, relativeRank - 2) * 1200;
            if (relativeRank >= 5) bonus += 3000;
            break;
        }
        case chess::PieceType::Queen:
        case chess::PieceType::King:
        case chess::PieceType::None:
            break;
    }
    return bonus;
}

int tacticalPrior(const chess::Position& pos, chess::Move move,
                  const chess::Position& child) {
    constexpr int kCheckPriorBonus = 4000;
    constexpr int kLosingCapturePriorPenalty = 8000;
    constexpr int kWinningCapturePriorScale = 8;
    int bonus = 0;
    const chess::Piece moving = pos.pieceAt(move.from());
    if (moving.isNone()) return 0;
    const chess::Piece captured = capturedPieceFor(pos, move, moving);
    if (!captured.isNone()) {
        bonus += chess::centipawnValue(captured.type) * 40;
        bonus -= chess::centipawnValue(moving.type) * 6;
        // chess-rtk uses full SEE here. This repo has no SEE helper yet, so use
        // the same branch thresholds with a material-delta proxy.
        const int simpleSee = chess::centipawnValue(captured.type) -
                              chess::centipawnValue(moving.type);
        if (simpleSee < 0) {
            bonus -= kLosingCapturePriorPenalty;
        } else if (simpleSee > 0) {
            bonus += std::min(900, simpleSee) * kWinningCapturePriorScale;
        }
    }
    if (move.promotion() != chess::Move::Promotion::None) {
        bonus += mctsPromotionValue(move.promotion()) * 35;
    }
    if (isCastleMove(pos, move, moving)) {
        bonus += 12000;
    }
    if (child.inCheck()) {
        bonus += kCheckPriorBonus;
    }
    return bonus;
}

bool isMateScore(int score) noexcept {
    return std::abs(score) >= kSearchMate - kMateDetectWindow;
}

int mateMoveDistance(int score) noexcept {
    const int plies = std::max(1, kSearchMate - std::abs(score));
    return (plies + 1) / 2;
}

float scoreForDisplay(int score) noexcept {
    if (isMateScore(score)) return score > 0 ? 99.99f : -99.99f;
    return static_cast<float>(score) / 100.0f;
}

std::string formatSearchLine(chess::Position root,
                             const std::vector<chess::Move>& line);
std::string formatCurrentNodeLine(const SearchConfig& config,
                                  chess::Position root,
                                  const std::vector<chess::Move>& path,
                                  const char* phase);

template <typename Evaluate>
int cachedSearchEval(const chess::Position& pos,
                     std::unordered_map<std::uint64_t, int>& evalCache,
                     std::size_t evalCacheLimit,
                     Evaluate& evaluate) {
    const std::uint64_t hash = pos.hash();
    if (auto it = evalCache.find(hash); it != evalCache.end()) {
        return it->second;
    }
    if (evalCache.size() >= evalCacheLimit) evalCache.clear();
    const int score = evaluate(pos);
    evalCache.emplace(hash, score);
    return score;
}

template <typename Evaluate, typename Publish>
int publishSearchEvalNode(const chess::Position& pos,
                          const SearchConfig& config,
                          std::unordered_map<std::uint64_t, int>& evalCache,
                          std::size_t evalCacheLimit,
                          Evaluate& evaluate,
                          Publish& publish,
                          const chess::Position& root,
                          const std::vector<chess::Move>& path,
                          int searchDepth,
                          std::uint64_t nodes,
                          const char* phase) {
    const int score = cachedSearchEval(pos, evalCache, evalCacheLimit, evaluate);
    publish(pos,
            path.empty() ? chess::Move::none() : path.back(),
            formatCurrentNodeLine(config, root, path, phase),
            searchDepth,
            scoreForDisplay(score),
            nodes,
            false);
    return score;
}

std::string formatNodeCount(std::uint64_t nodes) {
    std::string text = std::to_string(nodes);
    for (int pos = static_cast<int>(text.size()) - 3; pos > 0; pos -= 3) {
        text.insert(static_cast<std::size_t>(pos), ",");
    }
    return text;
}

std::string formatSearchLine(chess::Position root,
                             const std::vector<chess::Move>& line);

std::string formatCurrentNodeLine(const SearchConfig& config,
                                  chess::Position root,
                                  const std::vector<chess::Move>& path,
                                  const char* phase) {
    std::string line = config.name;
    line += " ";
    line += phase;
    if (!path.empty()) {
        line += " ";
        line += formatSearchLine(std::move(root), path);
    } else {
        line += " root";
    }
    return line;
}

std::string formatSearchLine(chess::Position root,
                             const std::vector<chess::Move>& line) {
    std::string out;
    for (chess::Move move : line) {
        if (!out.empty()) out += " ";
        out += move.toUci();
        root.make(move);
    }
    return out;
}

std::string formatSearchSummary(const SearchConfig& config,
                                int depth,
                                int score,
                                bool budgetHit,
                                const std::vector<chess::Move>& line,
                                const chess::Position& root) {
    std::string out = config.name;
    out += " d";
    out += std::to_string(depth);
    out += " ";
    if (isMateScore(score)) {
        out += score > 0 ? "mate in " : "mated in ";
        out += std::to_string(mateMoveDistance(score));
        out += " ";
    } else if (budgetHit) {
        out += "budget ";
    } else {
        out += "best ";
    }
    const std::string pv = formatSearchLine(root, line);
    out += pv.empty() ? "root" : pv;
    return out;
}

template <typename Evaluate, typename Publish>
int boundedNegamax(chess::Position& pos,
                   int depth,
                   int ply,
                   int alpha,
                   int beta,
                   std::atomic<bool>& cancel,
                   std::uint64_t& nodes,
                   std::uint64_t budgetStart,
                   std::uint64_t nodeBudget,
                   std::unordered_map<std::uint64_t, int>& evalCache,
                   std::size_t evalCacheLimit,
                   Evaluate& evaluate,
                   Publish& publish,
                   const SearchConfig& config,
                   const chess::Position& root,
                   std::vector<chess::Move>& path,
                   int searchDepth,
                   std::vector<chess::Move>& pv,
                   bool& budgetHit) {
    if (cancel.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (nodes - budgetStart >= nodeBudget) {
        budgetHit = true;
        return publishSearchEvalNode(pos, config, evalCache, evalCacheLimit,
                                     evaluate, publish, root, path,
                                     searchDepth, nodes, "budget");
    }

    ++nodes;
    if (pos.isFiftyMoveDraw() || pos.isInsufficientMaterial() ||
        pos.isThreefoldRepetition()) {
        publish(pos,
                path.empty() ? chess::Move::none() : path.back(),
                formatCurrentNodeLine(config, root, path, "draw"),
                searchDepth, 0.0f, nodes, false);
        return 0;
    }

    const bool inCheck = pos.inCheck();
    if (depth <= 0 && !inCheck) {
        return publishSearchEvalNode(pos, config, evalCache, evalCacheLimit,
                                     evaluate, publish, root, path,
                                     searchDepth, nodes, "eval");
    }

    std::vector<chess::Move> moves = orderedLegalMoves(pos, chess::Move::none());
    if (moves.empty()) {
        const int score = inCheck ? -kSearchMate + ply : 0;
        publish(pos,
                path.empty() ? chess::Move::none() : path.back(),
                formatCurrentNodeLine(config, root, path, inCheck ? "mate" : "draw"),
                searchDepth, scoreForDisplay(score), nodes, false);
        return score;
    }

    int best = -kSearchInf;
    chess::Move bestMove = chess::Move::none();
    std::vector<chess::Move> bestChildPv;
    const int nextDepth = std::max(0, depth - 1);
    for (chess::Move move : moves) {
        if (cancel.load(std::memory_order_relaxed) || budgetHit) break;

        pos.make(move);
        path.push_back(move);
        std::vector<chess::Move> childPv;
        const int score = -boundedNegamax(
            pos, nextDepth, ply + 1, -beta, -alpha, cancel,
            nodes, budgetStart, nodeBudget, evalCache, evalCacheLimit,
            evaluate, publish, config, root, path, searchDepth,
            childPv, budgetHit);
        path.pop_back();
        pos.unmake();

        if (score > best) {
            best = score;
            bestMove = move;
            bestChildPv = std::move(childPv);
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) break;
    }

    if (!bestMove.isNone()) {
        pv.clear();
        pv.push_back(bestMove);
        pv.insert(pv.end(), bestChildPv.begin(), bestChildPv.end());
    }
    return best == -kSearchInf
        ? publishSearchEvalNode(pos, config, evalCache, evalCacheLimit,
                                evaluate, publish, root, path,
                                searchDepth, nodes, "eval")
        : best;
}

template <typename Evaluate, typename Publish>
void runLiveSearch(chess::Position root,
                   std::atomic<bool>& cancel,
                   SearchConfig config,
                   Evaluate&& evaluate,
                   Publish&& publish) {
    std::uint64_t nodes = 0;
    publish(root, chess::Move::none(), std::string(config.name) + " starting",
            0, 0.0f, nodes, false);

    std::unordered_map<std::uint64_t, int> evalCache;
    const chess::Position rootStart = root;
    if (root.isFiftyMoveDraw() || root.isInsufficientMaterial() ||
        root.isThreefoldRepetition()) {
        publish(root, chess::Move::none(), std::string(config.name) + " drawn",
                0, 0.0f, nodes, true);
        return;
    }

    std::vector<chess::Move> rootMoves = orderedLegalMoves(root, chess::Move::none());
    if (rootMoves.empty()) {
        const bool mate = root.inCheck();
        publish(root, chess::Move::none(),
                mate ? std::string(config.name) + " checkmate"
                     : std::string(config.name) + " stalemate",
                0, mate ? -99.99f : 0.0f, nodes, true);
        return;
    }

    int lastScore = cachedSearchEval(root, evalCache, config.evalCacheLimit, evaluate);
    std::vector<chess::Move> lastLine;

    for (int depth = 1;
         depth <= config.maxDepth && !cancel.load(std::memory_order_relaxed);
         ++depth) {
        const std::uint64_t budgetStart = nodes;
        bool budgetHit = false;
        std::vector<chess::Move> depthLine;
        std::vector<chess::Move> currentPath;
        const int score = boundedNegamax(
            root, depth, 0, -kSearchInf, kSearchInf, cancel, nodes,
            budgetStart, config.nodesPerIteration, evalCache,
            config.evalCacheLimit, evaluate, publish, config, rootStart,
            currentPath, depth, depthLine, budgetHit);
        if (cancel.load(std::memory_order_relaxed)) break;
        if (!depthLine.empty()) {
            lastLine = depthLine;
            lastScore = score;
        }

        chess::Position preview = rootStart;
        chess::Move lastMove = chess::Move::none();
        if (!lastLine.empty()) {
            lastMove = lastLine.front();
            preview.make(lastMove);
        }

        const bool mateFound = isMateScore(lastScore);
        const bool complete = mateFound || budgetHit || depth == config.maxDepth;
        publish(std::move(preview), lastMove,
                formatSearchSummary(config, depth, lastScore, budgetHit,
                                    lastLine, rootStart),
                depth, scoreForDisplay(lastScore), nodes, complete);
        if (complete) return;
    }
}

}  // namespace

App::App(const std::string& configPath) {
    m_config.load(configPath);

    m_width  = m_config.getInt   ("window.width", 1280);
    m_height = m_config.getInt   ("window.height", 800);
    m_title  = m_config.getString("window.title", "cpp-nn-visualizer");
    bool startFullscreen = m_config.getBool("window.fullscreen", false);

    // Resizable so F11 / borderless toggles relayout cleanly, and so users
    // can drag-resize without the layout breaking. MSAA 4x smooths the
    // diagonal edges on legal-target rings, the king check halo, and any
    // rotated sprite scaling.
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(m_width, m_height, m_title.c_str());
    SetTargetFPS(60);
    SetExitKey(0);  // we handle ESC ourselves

    if (startFullscreen) {
        ToggleBorderlessWindowed();
        syncWindowSize();
    }

    loadFonts();
    applyBoardPalette(m_config.getString("board.palette", "classic"));

    std::string assetDir = m_config.getString("assets.pieces", "assets/pieces");
    m_sprites = std::make_unique<PieceSprites>(assetDir);

    std::string startFen = m_config.getString(
        "default.fen",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (!m_game.loadFen(startFen)) {
        m_game.reset();
    }

    m_clockEnabled = m_config.getBool("clock.enabled", false);
    m_clockInitialSeconds = static_cast<double>(
        std::max(1, m_config.getInt("clock.initial_seconds", 300)));
    m_clockIncrementSeconds = static_cast<double>(
        std::max(0, m_config.getInt("clock.increment_seconds", 0)));
    resetClock();

    m_promotionPicker = std::make_unique<PromotionPicker>(*m_sprites);
    m_moveList = std::make_unique<MoveListView>(m_game);
    m_nnueView = std::make_unique<NnueView>();
    m_cnnView = std::make_unique<CnnView>();
    m_bt4View = std::make_unique<Bt4View>();
    m_classicalView = std::make_unique<ClassicalView>();
    m_treeView = std::make_unique<TreeView>();
    if (m_sprites) m_treeView->setSprites(m_sprites.get());
    if (m_sprites) m_bt4View->setSprites(m_sprites.get());
    m_nnueView->setSignature(m_theme.sigNnue);
    m_cnnView->setSignature(m_theme.sigCnn);
    m_bt4View->setSignature(m_theme.sigBt4);
    m_classicalView->setSignature(m_theme.sigClassical);
    m_editorPalette = std::make_unique<EditorPalette>(m_editor, *m_sprites);
    m_editorPanel   = std::make_unique<EditorPanel>(m_editor);

    std::string nnuePath = m_config.getString(
        "model.nnue", "models/nnue-halfkp-demo.bin");
    tryLoadNnue(nnuePath);
    std::string cnnPath = m_config.getString(
        "model.lc0_cnn",
        "models/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin");
    tryLoadCnn(cnnPath);
    // BT4 is a compact real BT4J net; load it lazily on first use so a
    // launch that never opens BT4 pays nothing.
    m_bt4Path = m_config.getString(
        "model.lc0_bt4", "models/lc0-bt4-tiny-96x4x4h.bin");

    std::string startupArch = m_config.getString("startup.arch", "cnn");
    std::transform(startupArch.begin(), startupArch.end(), startupArch.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (startupArch == "cnn") {
        m_arch = Arch::Cnn;
    } else if (startupArch == "bt4") {
        m_arch = Arch::Bt4;
    } else if (startupArch == "classical") {
        m_arch = Arch::Classical;
    } else if (startupArch == "off" || startupArch == "none") {
        m_arch = Arch::Off;
    } else {
        m_arch = Arch::Nnue;
    }

    layout();
    buildControls();

    m_mctsVisitBudget = std::max(1, m_config.getInt("mcts.visits", 20000));
    {
        const std::string cpuctStr = m_config.getString("mcts.cpuct", "");
        if (!cpuctStr.empty()) {
            try {
                m_mctsCpuct = std::clamp(std::stod(cpuctStr), 0.05, 8.0);
            } catch (const std::exception&) {
            }
        }
    }

    // Optionally boot straight into the MCTS search-tree screen.
    if (m_config.getBool("startup.tree", false)) {
        enterTreeMode();
        startMcts();
    }
}

bool App::tryLoadNnue(const std::string& path) {
    if (!FileExists(path.c_str())) {
        m_nnueStatus = "NNUE: fallback (no weights at " + path + ")";
        return false;
    }
    try {
        m_nnue.load(path);
        m_nnueStatus = "NNUE: loaded H=" + std::to_string(m_nnue.hiddenSize());
        return true;
    } catch (const std::exception& e) {
        m_nnueStatus = std::string("NNUE: load failed (") + e.what() + ")";
        std::fprintf(stderr, "[App] %s\n", m_nnueStatus.c_str());
        return false;
    }
}

bool App::tryLoadCnn(const std::string& path) {
    if (!FileExists(path.c_str())) {
        m_cnnStatus = "CNN: not loaded (no weights at " + path + ")";
        m_cnnLoaded = false;
        return false;
    }
    try {
        m_cnn.load(path);
        m_cnnLoaded = m_cnn.isLoaded();
        m_cnnStatus = "CNN: loaded " + m_cnn.name();
        return m_cnnLoaded;
    } catch (const std::exception& e) {
        m_cnnStatus = std::string("CNN: load failed (") + e.what() + ")";
        std::fprintf(stderr, "[App] %s\n", m_cnnStatus.c_str());
        m_cnnLoaded = false;
        return false;
    }
}

void App::ensureBt4Loaded() {
    if (m_bt4LoadAttempted) return;
    m_bt4LoadAttempted = true;
    tryLoadBt4(m_bt4Path);
}

bool App::tryLoadBt4(const std::string& path) {
    if (!FileExists(path.c_str())) {
        return false;
    }
    try {
        m_bt4.load(path);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[App] BT4: load failed (%s)\n", e.what());
        return false;
    }
}

void App::applyBoardPalette(const std::string& paletteName) {
    std::string name = paletteName;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (name == "red" || name == "classic" || name == "warm") {
        m_theme.squareLight = Color{238, 216, 192, 255};
        m_theme.squareDark  = Color{171, 122, 101, 255};
        m_theme.coordLabel  = Color{120, 110,  95, 220};
        m_theme.lastMove    = Color{255, 209,  90, 110};
        m_theme.selection   = Color{255, 209,  90, 200};
        return;
    }

    if (name == "neural" || name == "analysis" || name == "heatmap") {
        // Muted red-analysis palette: still warm, but slightly less saturated
        // so activation overlays remain visible.
        m_theme.squareLight = Color{226, 205, 184, 255};
        m_theme.squareDark  = Color{151,  98,  88, 255};
        m_theme.coordLabel  = Color{132, 112, 100, 225};
        m_theme.lastMove    = Color{255, 209,  90, 115};
        m_theme.selection   = Color{255, 209,  90, 205};
        return;
    }

    if (name == "blue" || name == "cool") {
        m_theme.squareLight = Color{150, 158, 166, 255};
        m_theme.squareDark  = Color{ 52,  62,  78, 255};
        m_theme.coordLabel  = Color{160, 172, 184, 230};
        return;
    }

    if (name == "green" || name == "forest") {
        m_theme.squareLight = Color{151, 161, 150, 255};
        m_theme.squareDark  = Color{ 54,  70,  59, 255};
        m_theme.coordLabel  = Color{158, 174, 155, 230};
    }
}

Color App::archSignature(Arch a) const noexcept {
    switch (a) {
        case Arch::Nnue:      return m_theme.sigNnue;
        case Arch::Cnn:       return m_theme.sigCnn;
        case Arch::Bt4:       return m_theme.sigBt4;
        case Arch::Classical: return m_theme.sigClassical;
        case Arch::Off:       return m_theme.textMuted;
    }
    return m_theme.textMuted;
}

const char* App::archName(Arch a) const noexcept {
    switch (a) {
        case Arch::Nnue:      return "NNUE";
        case Arch::Cnn:       return "CNN";
        case Arch::Bt4:       return "BT4";
        case Arch::Classical: return "Classical";
        case Arch::Off:       return "Off";
    }
    return "?";
}

void App::selectArchitecture(Arch arch) {
    const bool restartSearch =
        m_arch != arch && m_searchActive && !m_editor.active();
    if (restartSearch) stopSearch();

    const std::uint64_t h = currentEvalHash();
    if (m_arch == arch &&
        (arch == Arch::Off ||
         (m_haveEval && m_lastEvalHash == h && m_lastEvalArch == arch) ||
         evalInFlightForCurrent())) {
        return;
    }
    m_arch = arch;
    resetActivationPanelCamera();
    if (m_board) layout();
    m_evalFailed = false;
    m_evalError.clear();
    if (m_arch == Arch::Off) {
        m_haveEval = false;
        m_activationFade = 0.0f;
        m_snapshot.clear();
        if (m_board) m_board->clearActivationOverlay();
        if (restartSearch) startSearch();
        return;
    }
    if (!m_editor.active()) {
        if (EvalCache* cache = cachedEval(m_arch, h)) {
            displayCachedEval(*cache, /*restartFade=*/true);
            if (restartSearch) startSearch();
            return;
        }
    }

    if (m_haveEval && m_lastEvalHash == h && m_lastEvalArch == m_arch) {
        if (restartSearch) startSearch();
        return;
    } else {
        m_haveEval = false;
        m_activationFade = 0.0f;
        m_snapshot.clear();
        if (m_board) m_board->clearActivationOverlay();
    }
    if (restartSearch) startSearch();
}

std::size_t App::evalCacheLimit(Arch arch) const noexcept {
    switch (arch) {
        case Arch::Nnue:      return 256;
        case Arch::Cnn:       return 32;
        case Arch::Bt4:       return 12;
        case Arch::Classical: return 1;
        case Arch::Off:       return 1;
    }
    return 32;
}

App::EvalCache* App::cachedEval(Arch arch, std::uint64_t hash) {
    auto& cache = m_evalCaches[static_cast<std::size_t>(arch)];
    auto it = cache.find(hash);
    if (it == cache.end()) return nullptr;
    it->second.lastUsed = ++m_evalCacheClock;
    return &it->second;
}

App::EvalCache& App::storeEvalResult(EvalResult&& result) {
    EvalCache cache;
    cache.snapshot = std::move(result.snapshot);
    cache.hash = result.hash;
    cache.arch = result.arch;
    cache.sideToMove = result.sideToMove;
    cache.have = true;
    cache.hasCentipawns = result.hasCentipawns;
    cache.hasWdl = result.hasWdl;
    cache.centipawns = result.centipawns;
    cache.wdl = result.wdl;
    cache.lastUsed = ++m_evalCacheClock;

    auto& caches = m_evalCaches[static_cast<std::size_t>(cache.arch)];
    const std::uint64_t hash = cache.hash;
    auto [it, inserted] = caches.insert_or_assign(hash, std::move(cache));
    (void)inserted;
    trimEvalCache(it->second.arch);

    return m_evalCaches[static_cast<std::size_t>(it->second.arch)].at(hash);
}

void App::trimEvalCache(Arch arch) {
    auto& cache = m_evalCaches[static_cast<std::size_t>(arch)];
    const std::size_t limit = evalCacheLimit(arch);
    while (cache.size() > limit) {
        auto oldest = std::min_element(
            cache.begin(), cache.end(),
            [](const auto& a, const auto& b) {
                return a.second.lastUsed < b.second.lastUsed;
            });
        if (oldest == cache.end()) return;
        cache.erase(oldest);
    }
}

void App::cycleArchitecture() {
    switch (m_arch) {
        case Arch::Nnue:      selectArchitecture(Arch::Cnn);       break;
        case Arch::Cnn:       selectArchitecture(Arch::Bt4);       break;
        case Arch::Bt4:       selectArchitecture(Arch::Classical); break;
        case Arch::Classical: selectArchitecture(Arch::Off);       break;
        case Arch::Off:       selectArchitecture(Arch::Nnue);      break;
    }
}

IActivationView* App::activeActivationView() noexcept {
    switch (m_arch) {
        case Arch::Nnue:      return m_nnueView.get();
        case Arch::Cnn:       return m_cnnView.get();
        case Arch::Bt4:       return m_bt4View.get();
        case Arch::Classical: return m_classicalView.get();
        case Arch::Off:       return nullptr;
    }
    return nullptr;
}

const IActivationView* App::activeActivationView() const noexcept {
    switch (m_arch) {
        case Arch::Nnue:      return m_nnueView.get();
        case Arch::Cnn:       return m_cnnView.get();
        case Arch::Bt4:       return m_bt4View.get();
        case Arch::Classical: return m_classicalView.get();
        case Arch::Off:       return nullptr;
    }
    return nullptr;
}

void App::toggleSearch() {
    if (m_searchActive) {
        stopSearch();
    } else {
        startSearch();
    }
}

void App::startSearch() {
    if (m_editor.active() || m_fenDialog.active()) return;

    stopSearch();
    const chess::Position root = m_game.position();
    m_searchRootHash = root.hash();
    m_searchCancel.store(false, std::memory_order_relaxed);
    m_searchActive = true;
    m_searchRunning = true;

    {
        std::lock_guard<std::mutex> lock(m_searchMutex);
        m_searchPreview = SearchPreview{};
        m_searchDisplayedPreview = SearchPreview{};
        m_searchPreviewSerial = 0;
        m_searchDisplayedSerial = 0;
        m_searchPreview.position = root;
        m_searchPreview.line = "starting";
        m_searchPreview.serial = ++m_searchPreviewSerial;
        m_searchPreview.have = true;
        m_searchDisplayedPreview = m_searchPreview;
        m_searchDisplayedSerial = m_searchPreview.serial;
    }

    const Arch searchArch = m_arch;
    const bool cnnLoaded = m_cnnLoaded;
    SearchConfig config;
    switch (searchArch) {
        case Arch::Nnue:
            config = SearchConfig{
                "NNUE search", 7, 12, 180000, 32768};
            break;
        case Arch::Cnn:
            config = SearchConfig{
                cnnLoaded ? "CNN value search" : "CNN fallback search",
                4, 30, 1200, 4096};
            break;
        case Arch::Bt4:
            config = SearchConfig{
                "BT4 transformer search", 3, 25, 700, 4096};
            break;
        case Arch::Classical:
            config = SearchConfig{
                "Classical eval search", 7, 8, 250000, 32768};
            break;
        case Arch::Off:
            config = SearchConfig{
                "Classical fallback", 7, 8, 250000, 32768};
            break;
    }
    m_searchFuture = std::async(std::launch::async,
        [this, root, searchArch, cnnLoaded, config]() mutable {
        runLiveSearch(root, m_searchCancel, config,
            [this, searchArch, cnnLoaded](const chess::Position& pos) {
                return evaluateSearchPositionForArch(pos, searchArch, cnnLoaded);
            },
            [this](chess::Position position,
                   chess::Move lastMove,
                   std::string line,
                   int depth,
                   float score,
                   std::uint64_t nodes,
                   bool complete) {
                publishSearchPreview(std::move(position), lastMove,
                                     std::move(line), depth, score,
                                     nodes, complete);
            });
    });
}

void App::stopSearch() {
    m_searchCancel.store(true, std::memory_order_relaxed);
    m_searchDisplayCv.notify_all();
    if (m_searchFuture.valid()) {
        try {
            m_searchFuture.wait();
            m_searchFuture.get();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[App] live search failed: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "[App] live search failed: unknown exception\n");
        }
    }
    m_searchRunning = false;
    m_searchActive = false;
    m_searchRootHash = 0;
    {
        std::lock_guard<std::mutex> lock(m_searchMutex);
        m_searchPreview = SearchPreview{};
        m_searchDisplayedPreview = SearchPreview{};
        m_searchPreviewSerial = 0;
        m_searchDisplayedSerial = 0;
    }
    m_searchDisplayCv.notify_all();
}

void App::pollSearchJob() {
    if (!m_searchRunning || !m_searchFuture.valid()) return;

    if (m_searchRootHash != 0 && m_searchRootHash != m_game.position().hash()) {
        stopSearch();
        return;
    }

    if (m_searchFuture.wait_for(std::chrono::milliseconds(0))
        != std::future_status::ready) {
        return;
    }

    try {
        m_searchFuture.get();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[App] live search failed: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[App] live search failed: unknown exception\n");
    }
    m_searchRunning = false;
}

App::SearchPreview App::currentSearchPreview() const {
    std::lock_guard<std::mutex> lock(m_searchMutex);
    return m_searchPreview;
}

App::SearchPreview App::currentDisplayedSearchPreview() const {
    std::lock_guard<std::mutex> lock(m_searchMutex);
    return m_searchDisplayedPreview;
}

bool App::searchPreviewReadyForDisplayAck(const SearchPreview& search) const {
    if (!m_searchActive || !search.have || m_editor.active()) return false;
    return true;
}

bool App::promoteSearchPreviewForDisplay(std::uint64_t serial) {
    std::lock_guard<std::mutex> lock(m_searchMutex);
    if (!m_searchPreview.have || m_searchPreview.serial != serial) {
        return false;
    }
    if (!m_searchDisplayedPreview.have ||
        serial > m_searchDisplayedPreview.serial) {
        m_searchDisplayedPreview = m_searchPreview;
    }
    return true;
}

void App::markSearchPreviewDisplayed(std::uint64_t serial) {
    {
        std::lock_guard<std::mutex> lock(m_searchMutex);
        if (serial > m_searchDisplayedSerial) {
            m_searchDisplayedSerial = serial;
        }
    }
    m_searchDisplayCv.notify_all();
}

cnnv::chess::Position App::currentEvalPosition() const {
    if (m_evalOverride.has_value()) return *m_evalOverride;
    const SearchPreview search = currentDisplayedSearchPreview();
    if (m_searchActive && search.have && !m_editor.active()) {
        return search.position;
    }
    return m_game.position();
}

std::uint64_t App::currentEvalHash() const {
    return currentEvalPosition().hash();
}

void App::publishSearchPreview(cnnv::chess::Position position,
                               cnnv::chess::Move lastMove,
                               std::string line,
                               int depth,
                               float score,
                               std::uint64_t nodes,
                               bool complete) {
    std::unique_lock<std::mutex> lock(m_searchMutex);
    const std::uint64_t serial = ++m_searchPreviewSerial;
    m_searchPreview.position = std::move(position);
    m_searchPreview.lastMove = lastMove;
    m_searchPreview.line = std::move(line);
    m_searchPreview.depth = depth;
    m_searchPreview.score = score;
    m_searchPreview.nodes = nodes;
    m_searchPreview.serial = serial;
    m_searchPreview.have = true;
    m_searchPreview.complete = complete;

    m_searchDisplayCv.wait(lock, [this, serial] {
        return m_searchCancel.load(std::memory_order_relaxed) ||
               m_searchDisplayedSerial >= serial;
    });
}

App::~App() {
    stopMcts();
    stopSearch();
    waitForEvalJob();
    m_promotionPicker.reset();
    m_moveList.reset();
    m_nnueView.reset();
    m_cnnView.reset();
    m_bt4View.reset();
    m_editorPalette.reset();
    m_editorPanel.reset();
    m_board.reset();
    m_sprites.reset();
    unloadFonts();
    if (IsWindowReady()) CloseWindow();
}

void App::loadFonts() {
    std::string uiPath = m_config.getString(
        "fonts.ui", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    std::string monoPath = m_config.getString(
        "fonts.mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
    int uiSize   = m_config.getInt("fonts.ui_size",   40);
    int monoSize = m_config.getInt("fonts.mono_size", 40);

    auto cps = uiCodepoints();
    if (FileExists(uiPath.c_str())) {
        m_theme.fontUi = LoadFontEx(uiPath.c_str(), uiSize,
                                    cps.data(), static_cast<int>(cps.size()));
        if (m_theme.fontUi.texture.id != 0) {
            SetTextureFilter(m_theme.fontUi.texture, TEXTURE_FILTER_BILINEAR);
            m_fontsLoaded = true;
        }
    }
    if (FileExists(monoPath.c_str())) {
        m_theme.fontMono = LoadFontEx(monoPath.c_str(), monoSize,
                                      cps.data(), static_cast<int>(cps.size()));
        if (m_theme.fontMono.texture.id != 0) {
            SetTextureFilter(m_theme.fontMono.texture, TEXTURE_FILTER_BILINEAR);
        }
    }
}

void App::unloadFonts() {
    if (m_theme.fontUi.texture.id   != 0) UnloadFont(m_theme.fontUi);
    if (m_theme.fontMono.texture.id != 0) UnloadFont(m_theme.fontMono);
    m_theme.fontUi   = Font{};
    m_theme.fontMono = Font{};
    m_fontsLoaded = false;
}

void App::buildControls() {
    m_controls.clear();
    m_controls.addButton("Reset",     [this] { resetGame(); });
    m_controls.addButton("Flip",      [this] { flipBoard(); });
    m_controls.addButton("Undo",      [this] { undoMove(); });
    m_controls.addButton("Redo",      [this] { redoMove(); });
    m_controls.addButton("Random",    [this] { playRandomMove(); });
    m_controls.addButton("Search",    [this] { toggleSearch(); },
                         [this] { return m_searchActive; });
    m_controls.addButton("Load FEN",  [this] {
        m_fenDialogOpensEditor = false;
        m_fenDialog.open();
    });
    m_controls.addButton("Save FEN",  [this] { saveCurrentFen(); });
    m_controls.addButton("Setup",     [this] { enterEditor(); });
    m_controls.addButton("Edit FEN",  [this] {
        m_fenDialogOpensEditor = true;
        m_fenDialog.open();
    });
}

void App::layout() {
    const float windowW = static_cast<float>(m_width);
    const float kPanelWidth = windowW * 0.5f;
    const float kStatusHeight = 44.0f;
    Rectangle boardArea{
        0.0f, kStatusHeight,
        static_cast<float>(m_width) - kPanelWidth,
        static_cast<float>(m_height) - kStatusHeight,
    };
    Rectangle panelArea{
        boardArea.x + boardArea.width, kStatusHeight,
        kPanelWidth, static_cast<float>(m_height) - kStatusHeight,
    };
    if (!m_board) {
        m_board = std::make_unique<BoardView>(m_game.position(), *m_sprites,
                                              boardArea);
    } else {
        m_board->setBounds(boardArea);
    }
    // Controls strip in the top portion of the right panel.
    const float kControlsHeight = kPanelWidth >= 620.0f ? 96.0f : 190.0f;
    Rectangle controlArea{
        panelArea.x, panelArea.y,
        panelArea.width, kControlsHeight,
    };
    m_controls.setBounds(controlArea);

    const float kArchSelectorHeight = 42.0f;
    const float kArchGap = 4.0f;
    const float kHintFooter = 44.0f;

    if (m_editor.active()) {
        // Palette sits directly under the controls; panel fills the rest.
        float paletteH = EditorPalette::recommendedHeight();
        Rectangle paletteArea{
            panelArea.x + 8,
            panelArea.y + kControlsHeight,
            panelArea.width - 16,
            paletteH,
        };
        Rectangle panelEditorArea{
            panelArea.x + 8,
            paletteArea.y + paletteH + 4,
            panelArea.width - 16,
            panelArea.height - kControlsHeight - paletteH - 4 - kHintFooter,
        };
        if (m_editorPalette) m_editorPalette->setBounds(paletteArea);
        if (m_editorPanel)   m_editorPanel->setBounds(panelEditorArea);
    } else {
        float selectorY = panelArea.y + kControlsHeight + kArchGap;
        float selectorX = panelArea.x + 8.0f;
        float selectorW = panelArea.width - 16.0f;
        // One extra cell for the "Tree" toggle tab after the architectures.
        float buttonW =
            (selectorW - static_cast<float>(kArchCount) * kArchGap) /
            static_cast<float>(kArchCount + 1);
        for (std::size_t i = 0; i < kArchCount; ++i) {
            m_archButtons[static_cast<std::size_t>(i)] = Rectangle{
                selectorX + static_cast<float>(i) * (buttonW + kArchGap),
                selectorY,
                buttonW,
                kArchSelectorHeight,
            };
        }
        m_treeTabButton = Rectangle{
            selectorX + static_cast<float>(kArchCount) * (buttonW + kArchGap),
            selectorY, buttonW, kArchSelectorHeight};

        // Split the lower panel so the active architecture view is the main
        // surface and the move list stays available as a compact footer.
        const float kAvailable =
            panelArea.height - kControlsHeight - kArchSelectorHeight
            - kArchGap - kHintFooter;
        const float contentY = selectorY + kArchSelectorHeight + kArchGap;
        if (m_arch == Arch::Off) {
            m_activationPanelBounds = Rectangle{0, 0, 0, 0};
            Rectangle moveListArea{
                panelArea.x + 8,
                contentY,
                panelArea.width - 16,
                kAvailable,
            };
            if (m_moveList) m_moveList->setBounds(moveListArea);
        } else {
            const bool showMoveList = m_game.history().size() > 0;
            const float kMoveListHeight = showMoveList
                ? std::min(76.0f, std::max(52.0f, kAvailable * 0.10f))
                : 0.0f;
            const float kMoveListGap = showMoveList ? 4.0f : 0.0f;
            const float kActivationHeight = std::max(
                260.0f, kAvailable - kMoveListHeight - kMoveListGap);
            Rectangle activationArea{
                panelArea.x + 8,
                contentY,
                panelArea.width - 16,
                kActivationHeight,
            };
            m_activationPanelBounds = activationArea;
            Rectangle moveListArea{
                panelArea.x + 8,
                activationArea.y + activationArea.height + kMoveListGap,
                panelArea.width - 16,
                kMoveListHeight,
            };
            if (m_nnueView) m_nnueView->setBounds(activationArea);
            if (m_cnnView)  m_cnnView->setBounds(activationArea);
            if (m_bt4View)  m_bt4View->setBounds(activationArea);
            if (m_classicalView) m_classicalView->setBounds(activationArea);
            if (m_moveList) m_moveList->setBounds(moveListArea);
        }
    }

    Rectangle dlg{
        boardArea.x + 40,
        boardArea.y + 80,
        boardArea.width - 80,
        140,
    };
    m_fenDialog.setBounds(dlg);
}

namespace {
// Squared pixel distance the mouse must travel after pressing on a piece
// before we treat the gesture as a drag rather than a click.
constexpr float kDragThresholdSquared = 36.0f;
constexpr double kHistoryRepeatDelay = 0.28;
constexpr double kHistoryRepeatInterval = 0.075;
}

void App::resetGame() {
    stopSearch();
    m_game.reset();
    resetClock();
    m_selected.reset();
    m_legalTargets = 0;
    m_dragFrom.reset();
    m_dragging = false;
    m_evalOverride.reset();
    layout();
}

void App::flipBoard() {
    m_board->setFlipped(!m_board->flipped());
}

void App::undoMove() {
    stopSearch();
    if (m_game.undo()) {
        m_clockLastUpdate = GetTime();
        m_selected.reset();
        m_legalTargets = 0;
        m_dragFrom.reset();
        m_dragging = false;
        m_evalOverride.reset();
        layout();
    }
}

void App::redoMove() {
    stopSearch();
    if (m_game.redo()) {
        m_clockLastUpdate = GetTime();
        m_selected.reset();
        m_legalTargets = 0;
        m_dragFrom.reset();
        m_dragging = false;
        m_evalOverride.reset();
        layout();
    }
}

void App::resetHistoryNavigationRepeat() noexcept {
    m_historyRepeatDirection = 0;
    m_historyRepeatNextAt = 0.0;
}

void App::handleHistoryNavigationKeys() {
    const bool leftDown = IsKeyDown(KEY_LEFT);
    const bool rightDown = IsKeyDown(KEY_RIGHT);

    if ((leftDown && rightDown) || (!leftDown && !rightDown)) {
        resetHistoryNavigationRepeat();
        return;
    }

    const int direction = leftDown ? -1 : 1;
    const bool pressed = direction < 0
        ? IsKeyPressed(KEY_LEFT)
        : IsKeyPressed(KEY_RIGHT);
    const double now = GetTime();

    auto step = [this](int dir) -> bool {
        if (dir < 0) {
            if (!m_game.canUndo()) return false;
            undoMove();
            return true;
        }
        if (!m_game.canRedo()) return false;
        redoMove();
        return true;
    };

    if (pressed) {
        (void)step(direction);
        m_historyRepeatDirection = direction;
        m_historyRepeatNextAt = now + kHistoryRepeatDelay;
        return;
    }

    if (m_historyRepeatDirection != direction) {
        m_historyRepeatDirection = direction;
        m_historyRepeatNextAt = now + kHistoryRepeatDelay;
        return;
    }

    if (now < m_historyRepeatNextAt) return;

    (void)step(direction);
    m_historyRepeatNextAt = now + kHistoryRepeatInterval;
}

void App::playRandomMove() {
    if (m_editor.active() || m_fenDialog.active()) return;
    if (m_game.status() != cnnv::game::GameStatus::Ongoing) return;

    stopSearch();

    chess::MoveList legal;
    m_game.generateLegalMoves(legal);
    if (legal.empty()) return;

    const int last = static_cast<int>(legal.size() - 1);
    const int index = GetRandomValue(0, last);
    if (tryMoveWithClock(legal[static_cast<std::size_t>(index)])) {
        m_selected.reset();
        m_legalTargets = 0;
        m_dragFrom.reset();
        m_dragging = false;
    }
}

void App::saveCurrentFen() {
    cnnv::io::savePositionToFenFile("position.fen", m_game.position());
}

void App::loadFenFromDialog() {
    auto opt = m_fenDialog.update();
    if (opt.has_value()) {
        stopSearch();
        if (m_fenDialogOpensEditor) {
            m_editor.enter(*opt);
            layout();
        } else {
            if (m_game.loadFen(cnnv::chess::Fen::format(*opt))) {
                resetClock();
                layout();
            }
        }
        m_selected.reset();
        m_legalTargets = 0;
        m_fenDialogOpensEditor = false;
    }
}

void App::enterEditor() {
    stopSearch();
    m_editor.enter(m_game.position());
    m_selected.reset();
    m_legalTargets = 0;
    m_dragFrom.reset();
    m_dragging = false;
    layout();
}

void App::leaveEditor(bool commit) {
    if (commit) {
        stopSearch();
        m_game.loadFen(m_editor.editor().fen());
        resetClock();
    }
    m_editor.leave();
    m_selected.reset();
    m_legalTargets = 0;
    m_dragFrom.reset();
    m_dragging = false;
    layout();
}

void App::handleEditorBoardClick(cnnv::chess::Square sq, bool isLeft) {
    if (isLeft) m_editor.applyLeftClick(sq);
    else        m_editor.applyRightClick(sq);
}

void App::toggleFullscreen() {
    ToggleBorderlessWindowed();
    syncWindowSize();
}

void App::syncWindowSize() {
    int w = GetScreenWidth();
    int h = GetScreenHeight();
    if (w == m_width && h == m_height) return;
    m_width  = w;
    m_height = h;
    layout();
}

bool App::tryMoveWithClock(cnnv::chess::Move m) {
    if (m_clockExpiredSide.has_value()) return false;
    const cnnv::chess::Color mover = m_game.position().sideToMove();
    if (!m_game.tryMove(m)) return false;
    stopSearch();
    addClockIncrement(mover);
    m_evalOverride.reset();
    layout();
    return true;
}

void App::resetClock() {
    m_clockWhiteSeconds = m_clockInitialSeconds;
    m_clockBlackSeconds = m_clockInitialSeconds;
    m_clockExpiredSide.reset();
    m_clockLastUpdate = GetTime();
}

void App::updateClock() {
    if (!m_clockEnabled) return;

    const double now = GetTime();
    const double elapsed = now - m_clockLastUpdate;
    m_clockLastUpdate = now;

    if (elapsed <= 0.0
        || m_clockExpiredSide.has_value()
        || m_editor.active()
        || m_fenDialog.active()
        || m_game.status() != cnnv::game::GameStatus::Ongoing) {
        return;
    }

    double& active = (m_game.position().sideToMove() == cnnv::chess::Color::White)
        ? m_clockWhiteSeconds
        : m_clockBlackSeconds;
    active = std::max(0.0, active - elapsed);
    if (active <= 0.0) {
        m_clockExpiredSide = m_game.position().sideToMove();
    }
}

void App::addClockIncrement(cnnv::chess::Color side) {
    if (!m_clockEnabled) return;
    double& clock = (side == cnnv::chess::Color::White)
        ? m_clockWhiteSeconds
        : m_clockBlackSeconds;
    clock += m_clockIncrementSeconds;
    m_clockLastUpdate = GetTime();
}

std::string App::statusText() const {
    if (!m_clockExpiredSide.has_value()) return m_game.statusText();
    return *m_clockExpiredSide == cnnv::chess::Color::White
        ? "Time - Black wins"
        : "Time - White wins";
}

std::string App::formatClock(double seconds) const {
    int total = static_cast<int>(std::ceil(std::max(0.0, seconds)));
    int hours = total / 3600;
    int minutes = (total / 60) % 60;
    int secs = total % 60;

    char buf[16];
    if (hours > 0) {
        std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", hours, minutes, secs);
    } else {
        std::snprintf(buf, sizeof(buf), "%d:%02d", minutes, secs);
    }
    return buf;
}

void App::drawClockArea(Rectangle statusBounds) const {
    if (!m_clockEnabled) return;

    constexpr float kBoxW = 122.0f;
    constexpr float kBoxH = 28.0f;
    constexpr float kGap = 8.0f;
    const float totalW = kBoxW * 2.0f + kGap;
    const float x = statusBounds.x + (statusBounds.width - totalW) * 0.5f;
    const float y = statusBounds.y + (statusBounds.height - kBoxH) * 0.5f;

    auto drawOne = [&](float bx, const char* label, double seconds,
                       cnnv::chess::Color side) {
        const bool active = !m_clockExpiredSide.has_value()
            && !m_editor.active()
            && m_game.status() == cnnv::game::GameStatus::Ongoing
            && m_game.position().sideToMove() == side;
        const bool expired = m_clockExpiredSide.has_value()
            && *m_clockExpiredSide == side;
        Rectangle box{bx, y, kBoxW, kBoxH};
        Color fill = expired ? m_theme.checkWarning
                   : active ? m_theme.buttonActive
                            : m_theme.buttonIdle;
        DrawRectangleRec(box, fill);
        DrawRectangleLinesEx(box, 1.0f, m_theme.panelBorder);

        std::string text = std::string(label) + " " + formatClock(seconds);
        const int fontSize = 16;
        const int textW = measureTextMono(m_theme, text.c_str(), fontSize);
        drawTextMono(m_theme, text.c_str(),
                     static_cast<int>(box.x + (box.width - textW) * 0.5f),
                     static_cast<int>(box.y + (box.height - fontSize) * 0.5f),
                     fontSize, m_theme.textPrimary);
    };

    drawOne(x, "W", m_clockWhiteSeconds, cnnv::chess::Color::White);
    drawOne(x + kBoxW + kGap, "B", m_clockBlackSeconds, cnnv::chess::Color::Black);
}

void App::updateArchitectureSelector() {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return;
    Vector2 mp = GetMousePosition();
    if (CheckCollisionPointRec(mp, m_treeTabButton)) {
        enterTreeMode();
        return;
    }
    for (std::size_t i = 0; i < kArchCount; ++i) {
        if (!CheckCollisionPointRec(mp, m_archButtons[i])) {
            continue;
        }
        selectArchitecture(static_cast<Arch>(i));
        return;
    }
}

void App::drawArchitectureSelector() const {
    static constexpr const char* kLabels[kArchCount] = {"NNUE", "CNN", "BT4",
                                                         "Classical", "OFF"};
    for (std::size_t i = 0; i < kArchCount; ++i) {
        Rectangle r = m_archButtons[i];
        const bool active = static_cast<std::size_t>(m_arch) == i;
        const bool hover = CheckCollisionPointRec(GetMousePosition(), r);
        const Color sig = archSignature(static_cast<Arch>(i));
        Color fill = active ? tensorviz::blend(m_theme.buttonActive, sig, 0.42f)
                   : hover ? m_theme.buttonHover
                           : m_theme.buttonIdle;
        DrawRectangleRec(r, fill);
        if (active) {
            DrawRectangleRec(Rectangle{r.x, r.y, r.width, 3.0f}, sig);
        }
        DrawRectangleLinesEx(r, 1.0f, active ? sig : m_theme.panelBorder);

        int fontSize = 16;
        int textWidth = measureText(m_theme, kLabels[i], fontSize);
        drawText(m_theme, kLabels[i],
                 static_cast<int>(r.x + (r.width - textWidth) * 0.5f),
                 static_cast<int>(r.y + (r.height - fontSize) * 0.5f),
                 fontSize, active ? m_theme.textPrimary : m_theme.textMuted);
    }

    // "Tree" toggle tab — switches to the full-window search-tree view.
    {
        const Rectangle r = m_treeTabButton;
        const bool hover = CheckCollisionPointRec(GetMousePosition(), r);
        DrawRectangleRec(r, hover ? m_theme.buttonHover : m_theme.buttonIdle);
        DrawRectangleRec(Rectangle{r.x, r.y, r.width, 3.0f}, m_theme.accentBlue);
        DrawRectangleLinesEx(r, 1.0f, m_theme.accentBlue);
        const int fs = 16;
        const int tw = measureText(m_theme, "Tree", fs);
        drawText(m_theme, "Tree",
                 static_cast<int>(r.x + (r.width - tw) * 0.5f),
                 static_cast<int>(r.y + (r.height - fs) * 0.5f), fs,
                 m_theme.textPrimary);
    }
}

Camera2D App::activationPanelCamera() const noexcept {
    Camera2D cam{};
    cam.offset = m_panelPan;
    cam.target = Vector2{0.0f, 0.0f};
    cam.rotation = 0.0f;
    cam.zoom = m_panelZoom;
    return cam;
}

void App::zoomActivationPanel(Vector2 anchor, float factor) noexcept {
    const float next = std::clamp(m_panelZoom * factor, 0.01f, 8.0f);
    const float ratio = next / m_panelZoom;
    // Keep the world point under `anchor` fixed (clamp applied before ratio so
    // the view does not drift at the zoom limits).
    m_panelPan.x = anchor.x - (anchor.x - m_panelPan.x) * ratio;
    m_panelPan.y = anchor.y - (anchor.y - m_panelPan.y) * ratio;
    m_panelZoom = next;
}

void App::resetActivationPanelCamera() noexcept {
    m_panelZoom = 1.0f;
    m_panelPan = Vector2{0.0f, 0.0f};
    m_panelPanning = false;
}

void App::handleActivationPanelCameraInput() {
    if (m_editor.active() || m_arch == Arch::Off) {
        m_panelPanning = false;
        return;
    }
    const Vector2 mp = GetMousePosition();
    const bool over = CheckCollisionPointRec(mp, m_activationPanelBounds);

    // Wheel zoom-to-cursor while hovering the panel (the move list owns the
    // wheel only inside its own bounds, so there is no conflict).
    if (over) {
        const float wheel = GetMouseWheelMove();
        if (wheel > 0.0f) zoomActivationPanel(mp, 1.12f);
        else if (wheel < 0.0f) zoomActivationPanel(mp, 1.0f / 1.12f);
    }

    // Right/middle-drag pan (left button stays free for view interaction).
    const bool panButtonDown = IsMouseButtonDown(MOUSE_BUTTON_RIGHT) ||
                               IsMouseButtonDown(MOUSE_BUTTON_MIDDLE);
    const bool panButtonPressed = IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) ||
                                  IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE);
    if (over && panButtonPressed) {
        m_panelPanning = true;
        m_panelPanLastMouse = mp;
    }
    if (m_panelPanning && panButtonDown) {
        m_panelPan.x += mp.x - m_panelPanLastMouse.x;
        m_panelPan.y += mp.y - m_panelPanLastMouse.y;
        m_panelPanLastMouse = mp;
    } else {
        m_panelPanning = false;
    }

    // Keyboard: +/- zoom about the panel center, 0 resets.
    const Vector2 center{
        m_activationPanelBounds.x + m_activationPanelBounds.width * 0.5f,
        m_activationPanelBounds.y + m_activationPanelBounds.height * 0.5f};
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        zoomActivationPanel(center, 1.12f);
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        zoomActivationPanel(center, 1.0f / 1.12f);
    }
    if (IsKeyPressed(KEY_ZERO) || IsKeyPressed(KEY_KP_0)) {
        resetActivationPanelCamera();
    }
}

cnnv::search::LeafEval App::mctsEvaluate(const cnnv::chess::Position& pos,
                                        const cnnv::chess::MoveList& legal,
                                        Arch arch) const {
    using cnnv::chess::Move;

    const auto classicalValue = [&]() {
        const int cp = chess::eval::evaluateWhiteBreakdown(pos).stmTotal();
        return std::tanh(static_cast<float>(cp) / 600.0f);
    };

    cnnv::search::LeafEval out;
    float value = 0.0f;
    std::vector<float> cnnLogits;  // real CNN policy logits when available
    try {
        std::lock_guard<std::mutex> lock(m_networkEvalMutex);
        cnnv::nn::ActivationSnapshot snap;
        switch (arch) {
            case Arch::Cnn: {
                if (!m_cnnLoaded) { value = classicalValue(); break; }
                m_cnn.evaluate(pos, snap);
                using namespace cnnv::nn::lc0_cnn::snapshot_keys;
                if (snap.size(kValueWdl) == 3) {
                    const float* w = snap.data(kValueWdl);
                    value = w[0] - w[2];
                }
                if (snap.size(kPolicyLogits) > 0) {
                    const float* p = snap.data(kPolicyLogits);
                    cnnLogits.assign(p, p + snap.size(kPolicyLogits));
                }
                break;
            }
            case Arch::Bt4: {
                m_bt4.evaluate(pos, snap);
                using namespace cnnv::nn::lc0_bt4::snapshot_keys;
                if (snap.size(kValueWdl) == 3) {
                    const float* w = snap.data(kValueWdl);
                    value = w[0] - w[2];
                }
                break;
            }
            case Arch::Classical:
            case Arch::Off:
                value = classicalValue();
                break;
            case Arch::Nnue:
            default: {
                m_nnue.evaluate(pos, snap);
                using namespace cnnv::nn::nnue::snapshot_keys;
                if (snap.size(kValueCentipawns) == 1) {
                    value = std::tanh(snap.data(kValueCentipawns)[0] / 600.0f);
                }
                break;
            }
        }
    } catch (...) {
        value = classicalValue();
    }
    out.value = std::clamp(value, -1.0f, 1.0f);

    const std::size_t n = legal.size();

    // Real CNN policy priors: map each legal move to its logit via the LC0
    // PolicyEncoder, then softmax (transpiled from chess-rtk
    // MctsBackendSupport.policyPriorsFromLogits). Falls through to the
    // handcrafted priors below if the policy head is unavailable.
    if (arch == Arch::Cnn && !cnnLogits.empty()) {
        std::vector<int> idx(n, -1);
        float maxLogit = -1e30f;
        int valid = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const int li = m_cnn.policyIndexForMove(pos, legal[i]);
            idx[i] = li;
            if (li >= 0 && static_cast<std::size_t>(li) < cnnLogits.size() &&
                std::isfinite(cnnLogits[static_cast<std::size_t>(li)])) {
                maxLogit = std::max(maxLogit, cnnLogits[static_cast<std::size_t>(li)]);
                ++valid;
            }
        }
        if (valid > 0 && std::isfinite(maxLogit)) {
            out.priors.assign(n, 0.0f);
            double psum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const int li = idx[i];
                if (li >= 0 && static_cast<std::size_t>(li) < cnnLogits.size() &&
                    std::isfinite(cnnLogits[static_cast<std::size_t>(li)])) {
                    out.priors[i] = std::exp(cnnLogits[static_cast<std::size_t>(li)] -
                                             maxLogit);
                    psum += out.priors[i];
                }
            }
            if (psum > 0.0) {
                for (float& p : out.priors) {
                    p = static_cast<float>(p / psum);
                }
                return out;
            }
        }
    }

    // chess-rtk fallback priors: classical quiet-move ordering score plus
    // tactical bonuses, divided by 18k and softmaxed. Used for NNUE/Classical
    // and when the CNN policy head is unavailable.
    out.priors.resize(n);
    std::vector<float> logits(n, 0.0f);
    float maxLogit = -1e30f;
    for (std::size_t i = 0; i < n; ++i) {
        const Move m = legal[i];
        cnnv::chess::Position child = pos;
        child.make(m);
        const int score = quietMovePrior(pos, m) + tacticalPrior(pos, m, child);
        logits[i] = static_cast<float>(score) / 18000.0f;
        maxLogit = std::max(maxLogit, logits[i]);
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = std::exp(logits[i] - maxLogit);
        out.priors[i] = e;
        sum += e;
    }
    if (sum > 0.0f) {
        for (float& p : out.priors) p /= sum;
    }
    return out;
}

void App::startMcts() {
    stopMcts();
    const Arch arch = m_arch == Arch::Off ? (m_cnnLoaded ? Arch::Cnn : Arch::Nnue)
                                          : m_arch;
    if (arch == Arch::Bt4) ensureBt4Loaded();
    const chess::Position root = currentEvalPosition();
    m_mcts = std::make_unique<cnnv::search::Mcts>(
        root, m_mctsCpuct,
        [this, arch](const chess::Position& p, const chess::MoveList& l) {
            return mctsEvaluate(p, l, arch);
        });
    m_mctsStop = false;
    m_mctsPaused = false;
    m_mctsAppliedPlayouts = -1;
    m_treeScrubbing = false;
    m_scrubPlaying = false;
    m_scrubIndex = 0;
    m_scrubAppliedIndex = -1;
    {
        std::lock_guard<std::mutex> lock(m_mctsMutex);
        m_mctsSnapshot = m_mcts->snapshot(1000, 0);
        m_mctsFrames.clear();
        m_mctsFrames.push_back(m_mctsSnapshot);
    }
    m_mctsRunning = true;
    m_mctsThread = std::thread([this]() { mctsWorker(); });
}

void App::mctsWorker() {
    const long budget = static_cast<long>(m_mctsVisitBudget);
    auto lastPublish = std::chrono::steady_clock::now() -
                       std::chrono::milliseconds(250);
    long lastPublishedPlayouts = -1;
    while (!m_mctsStop.load()) {
        if (m_mctsPaused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        m_mcts->iterate();
        const long n = m_mcts->playouts();
        const bool done = budget > 0 && n >= budget;
        const auto now = std::chrono::steady_clock::now();
        const bool earlyGrowth = n <= 16;
        const bool timeForFrame =
            now - lastPublish >= std::chrono::milliseconds(100);
        if ((earlyGrowth || timeForFrame || n % 32 == 0 || done) &&
            n != lastPublishedPlayouts) {
            cnnv::search::Snapshot s = m_mcts->snapshot(1000, 0);
            std::lock_guard<std::mutex> lock(m_mctsMutex);
            m_mctsSnapshot = s;
            // Record a growth frame for the scrubber, decimating to a bound.
            constexpr std::size_t kMaxFrames = 120;
            m_mctsFrames.push_back(std::move(s));
            if (m_mctsFrames.size() > kMaxFrames) {
                std::vector<cnnv::search::Snapshot> kept;
                kept.reserve(m_mctsFrames.size() / 2 + 1);
                for (std::size_t i = 0; i < m_mctsFrames.size(); i += 2) {
                    kept.push_back(std::move(m_mctsFrames[i]));
                }
                m_mctsFrames = std::move(kept);
            }
            lastPublish = now;
            lastPublishedPlayouts = n;
        }
        if (done) break;
    }
    m_mctsRunning = false;
}

void App::stopMcts() {
    m_mctsStop = true;
    if (m_mctsThread.joinable()) m_mctsThread.join();
    m_mctsRunning = false;
    if (m_mcts) {
        std::lock_guard<std::mutex> lock(m_mctsMutex);
        m_mctsSnapshot = m_mcts->snapshot(1000, 0);
        if (!m_mctsFrames.empty()) m_mctsFrames.back() = m_mctsSnapshot;
    }
}

void App::enterTreeMode() {
    m_treeMode = true;
    m_evalOverride.reset();  // root the search at the live board
    if (m_treeView) m_treeView->clear();
    m_mctsAppliedPlayouts = -1;
    layoutTree();
}

void App::leaveTreeMode() {
    stopMcts();
    m_treeMode = false;
    layout();
}

void App::layoutTree() {
    const float toolbarH = 72.0f;  // two control rows
    m_treeToolbar = Rectangle{0.0f, 0.0f, static_cast<float>(m_width), toolbarH};
    if (m_treeView) {
        m_treeView->setBounds(Rectangle{0.0f, toolbarH,
                                        static_cast<float>(m_width),
                                        static_cast<float>(m_height) - toolbarH});
    }
    const float h = 28.0f;
    const float w1 = 28.0f;  // steppers
    // Row 1: Start Stop Exit Fit | visits-/+ | cpuct-/+ | Follow.
    {
        float x = 10.0f;
        const float y = 6.0f;
        auto place = [&](std::size_t i, float w, float gapAfter) {
            m_treeButtons[i] = Rectangle{x, y, w, h};
            x += w + gapAfter;
        };
        place(0, 66.0f, 6.0f);    // Start
        place(1, 60.0f, 6.0f);    // Stop
        place(2, 84.0f, 6.0f);    // ← Board
        place(3, 50.0f, 30.0f);   // Fit
        place(4, w1, 6.0f);       // visits-
        place(5, w1, 96.0f);      // visits+ (gap for "visits N" label)
        place(6, w1, 6.0f);       // cpuct-
        place(7, w1, 96.0f);      // cpuct+ (gap for "cpuct N" label)
        place(8, 104.0f, 6.0f);   // Follow toggle
    }
    // Row 2: branches-/+ | depth-/+ | Batch | Guides | Merge.
    {
        float x = 10.0f;
        const float y = 38.0f;
        auto place = [&](std::size_t i, float w, float gapAfter) {
            m_treeButtons[i] = Rectangle{x, y, w, h};
            x += w + gapAfter;
        };
        place(9, w1, 6.0f);        // branches-
        place(10, w1, 96.0f);      // branches+ (gap for "branches N" label)
        place(13, w1, 6.0f);       // depth-
        place(14, w1, 84.0f);      // depth+ (gap for "depth N" label)
        place(11, 96.0f, 8.0f);    // Batch toggle
        place(12, 96.0f, 8.0f);    // Guides toggle
        place(15, 96.0f, 8.0f);    // Merge toggle
    }
}

void App::handleTreeInput() {
    if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_G)) {
        leaveTreeMode();
        return;
    }
    if (IsWindowResized()) {
        syncWindowSize();
        layoutTree();
    }
    if (IsKeyPressed(KEY_F11)) toggleFullscreen();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        const Vector2 mp = GetMousePosition();
        if (CheckCollisionPointRec(mp, m_treeButtons[0])) {
            startMcts();
        } else if (CheckCollisionPointRec(mp, m_treeButtons[1])) {
            stopMcts();
        } else if (CheckCollisionPointRec(mp, m_treeButtons[2])) {
            leaveTreeMode();
            return;
        } else if (CheckCollisionPointRec(mp, m_treeButtons[3])) {
            if (m_treeView) m_treeView->fit();
        } else if (CheckCollisionPointRec(mp, m_treeButtons[4])) {
            m_mctsVisitBudget = std::max(100, m_mctsVisitBudget - 5000);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[5])) {
            m_mctsVisitBudget = std::min(2000000, m_mctsVisitBudget + 5000);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[6])) {
            m_mctsCpuct = std::max(0.05, m_mctsCpuct - 0.25);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[7])) {
            m_mctsCpuct = std::min(8.0, m_mctsCpuct + 0.25);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[8])) {
            m_mctsFollow = !m_mctsFollow;
        } else if (CheckCollisionPointRec(mp, m_treeButtons[9]) && m_treeView) {
            m_treeView->setBranches(std::max(0, m_treeView->branches() - 1));
        } else if (CheckCollisionPointRec(mp, m_treeButtons[10]) && m_treeView) {
            m_treeView->setBranches(m_treeView->branches() + 1);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[11]) && m_treeView) {
            m_treeView->setBatchLeaves(!m_treeView->batchLeaves());
        } else if (CheckCollisionPointRec(mp, m_treeButtons[12]) && m_treeView) {
            m_treeView->setShowGuides(!m_treeView->showGuides());
        } else if (CheckCollisionPointRec(mp, m_treeButtons[13]) && m_treeView) {
            m_treeView->setDepth(m_treeView->depth() - 1);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[14]) && m_treeView) {
            m_treeView->setDepth(m_treeView->depth() + 1);
        } else if (CheckCollisionPointRec(mp, m_treeButtons[15]) && m_treeView) {
            m_treeView->setMergeTranspositions(!m_treeView->mergeTranspositions());
        }
    }

    // Playback scrubber over recorded growth frames (Space/Left/Right/Home/End).
    int frameCount = 0;
    {
        std::lock_guard<std::mutex> lock(m_mctsMutex);
        frameCount = static_cast<int>(m_mctsFrames.size());
    }
    if (frameCount > 0) {
        if (IsKeyPressed(KEY_LEFT)) {
            m_treeScrubbing = true;
            m_scrubPlaying = false;
            m_scrubIndex = std::max(0, m_scrubIndex - 1);
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            m_treeScrubbing = true;
            m_scrubPlaying = false;
            m_scrubIndex = std::min(frameCount - 1, m_scrubIndex + 1);
        }
        if (IsKeyPressed(KEY_HOME)) {
            m_treeScrubbing = true;
            m_scrubPlaying = false;
            m_scrubIndex = 0;
        }
        if (IsKeyPressed(KEY_END)) {
            m_treeScrubbing = false;
            m_scrubPlaying = false;
        }
        if (IsKeyPressed(KEY_SPACE)) {
            m_scrubPlaying = !m_scrubPlaying;
            if (m_scrubPlaying) {
                m_treeScrubbing = true;
                if (m_scrubIndex >= frameCount - 1) m_scrubIndex = 0;
                m_scrubNextAdvance = GetTime() + 0.11;
            }
        }
        if (m_scrubPlaying && GetTime() >= m_scrubNextAdvance) {
            m_scrubIndex = std::min(frameCount - 1, m_scrubIndex + 1);
            m_scrubNextAdvance = GetTime() + 0.11;
            if (m_scrubIndex >= frameCount - 1) m_scrubPlaying = false;
        }
        m_scrubIndex = std::clamp(m_scrubIndex, 0, frameCount - 1);
    }

    if (m_treeView) {
        m_treeView->update();
        if (m_treeView->takeSelectionChanged()) {
            // Selecting a node drives the network views to that position (the
            // "trace" link). Disables follow so the pick stays put.
            m_treeSelInfo = m_treeView->selectedFen();
            m_mctsFollow = false;
            if (auto p = chess::Fen::parse(m_treeSelInfo)) {
                m_evalOverride = *p;
            }
        }
    }

    // Live follow-leaf: mirror the currently exploring leaf into the override so
    // the network views trace it (visible after Esc returns to the panel view).
    if (m_mctsFollow) {
        std::string fen;
        {
            std::lock_guard<std::mutex> lock(m_mctsMutex);
            fen = m_mctsSnapshot.exploringFen;
        }
        if (!fen.empty()) {
            if (auto p = chess::Fen::parse(fen)) m_evalOverride = *p;
        }
    }
}

void App::drawTreeScreen() {
    // Choose what to show: a scrubbed growth frame, or the live snapshot.
    std::size_t nodeCount = 0;
    long shownPlayouts = 0;
    std::string shownBest;
    double shownCp = 0.0;
    int frameTotal = 0;
    cnnv::search::Snapshot pendingSnapshot;
    bool applySnapshot = false;
    {
        std::lock_guard<std::mutex> lock(m_mctsMutex);
        frameTotal = static_cast<int>(m_mctsFrames.size());
        if (m_treeScrubbing && !m_mctsFrames.empty()) {
            const int idx =
                std::clamp(m_scrubIndex, 0, static_cast<int>(m_mctsFrames.size()) - 1);
            m_scrubIndex = idx;
            const cnnv::search::Snapshot& f = m_mctsFrames[static_cast<std::size_t>(idx)];
            if (m_treeView && m_scrubAppliedIndex != idx) {
                pendingSnapshot = f;
                applySnapshot = true;
                m_scrubAppliedIndex = idx;
                m_mctsAppliedPlayouts = -1;  // force live re-apply on return
            }
            nodeCount = f.nodes.size();
            shownPlayouts = f.playouts;
            shownBest = f.bestUci;
            shownCp = f.rootCentipawns / 100.0;
        } else {
            m_scrubAppliedIndex = -1;
            if (m_treeView && m_mctsSnapshot.playouts != m_mctsAppliedPlayouts) {
                pendingSnapshot = m_mctsSnapshot;
                applySnapshot = true;
                m_mctsAppliedPlayouts = m_mctsSnapshot.playouts;
            }
            nodeCount = m_mctsSnapshot.nodes.size();
            shownPlayouts = m_mctsSnapshot.playouts;
            shownBest = m_mctsSnapshot.bestUci;
            shownCp = m_mctsSnapshot.rootCentipawns / 100.0;
        }
    }
    if (applySnapshot && m_treeView) {
        m_treeView->setSnapshot(pendingSnapshot);
    }
    if (m_treeView) m_treeView->draw(m_theme);

    // Toolbar.
    DrawRectangleRec(m_treeToolbar, m_theme.panelBackground);
    DrawRectangleLinesEx(m_treeToolbar, 1.0f, m_theme.panelBorder);
    const bool running = m_mctsRunning.load();
    auto button = [&](std::size_t i, const char* label, bool active) {
        const Rectangle r = m_treeButtons[i];
        DrawRectangleRec(r, active ? m_theme.buttonActive : m_theme.buttonIdle);
        DrawRectangleLinesEx(r, 1.0f, m_theme.panelBorder);
        const int tw = measureText(m_theme, label, 14);
        drawText(m_theme, label,
                 static_cast<int>(r.x + (r.width - tw) * 0.5f),
                 static_cast<int>(r.y + 7.0f), 14, m_theme.textPrimary);
    };
    const bool batchOn = m_treeView && m_treeView->batchLeaves();
    const bool guidesOn = m_treeView && m_treeView->showGuides();
    const bool mergeOn = m_treeView && m_treeView->mergeTranspositions();
    button(0, "Start", running);
    button(1, "Stop", false);
    button(2, "\xE2\x86\x90 Board", false);
    button(3, "Fit", false);
    button(4, "-", false);
    button(5, "+", false);
    button(6, "-", false);
    button(7, "+", false);
    button(8, m_mctsFollow ? "Follow: on" : "Follow: off", m_mctsFollow);
    button(9, "-", false);
    button(10, "+", false);
    button(13, "-", false);
    button(14, "+", false);
    button(11, batchOn ? "Batch: on" : "Batch: off", batchOn);
    button(12, guidesOn ? "Guides: on" : "Guides: off", guidesOn);
    button(15, mergeOn ? "Merge: on" : "Merge: off", mergeOn);

    auto label = [&](const char* s, float x, float y) {
        drawText(m_theme, s, static_cast<int>(x), static_cast<int>(y), 14,
                 m_theme.textMuted);
    };
    char visLabel[32];
    std::snprintf(visLabel, sizeof(visLabel), "visits %d", m_mctsVisitBudget);
    label(visLabel, m_treeButtons[5].x + 34.0f, m_treeButtons[5].y + 7.0f);
    char cpuctLabel[24];
    std::snprintf(cpuctLabel, sizeof(cpuctLabel), "cpuct %.2f", m_mctsCpuct);
    label(cpuctLabel, m_treeButtons[7].x + 34.0f, m_treeButtons[7].y + 7.0f);
    char brLabel[32];
    const int br = m_treeView ? m_treeView->branches() : 0;
    if (br <= 0) {
        std::snprintf(brLabel, sizeof(brLabel), "branches all");
    } else {
        std::snprintf(brLabel, sizeof(brLabel), "branches %d", br);
    }
    label(brLabel, m_treeButtons[10].x + 34.0f, m_treeButtons[10].y + 7.0f);
    char depthLabel[24];
    std::snprintf(depthLabel, sizeof(depthLabel), "depth %d",
                  m_treeView ? m_treeView->depth() : 0);
    label(depthLabel, m_treeButtons[14].x + 34.0f, m_treeButtons[14].y + 7.0f);

    // Status: concise and placed after the Follow control so it cannot collide
    // with the toolbar buttons on the default 1280px window.
    char status[200];
    if (m_treeScrubbing) {
        std::snprintf(status, sizeof(status),
                      "scrub %d/%d  %ld playouts  %zu nodes  best %s  %+.2f",
                      m_scrubIndex + 1, frameTotal, shownPlayouts, nodeCount,
                      shownBest.empty() ? "-" : shownBest.c_str(), shownCp);
    } else {
        std::snprintf(status, sizeof(status),
                      "%s  %ld playouts  %zu nodes  best %s  %+.2f  %s",
                      running ? "running" : "idle", shownPlayouts, nodeCount,
                      shownBest.empty() ? "-" : shownBest.c_str(), shownCp,
                      archName(m_arch == Arch::Off
                                   ? (m_cnnLoaded ? Arch::Cnn : Arch::Nnue)
                                   : m_arch));
    }
    const int sw = measureText(m_theme, status, 14);
    const float statusLeft = m_treeButtons[8].x + m_treeButtons[8].width + 12.0f;
    const int statusX = std::max(static_cast<int>(statusLeft), m_width - sw - 12);
    if (statusX + sw < m_width - 8) {
        drawText(m_theme, status, statusX, 13, 14, m_theme.textMuted);
    }

    if (!m_treeSelInfo.empty()) {
        const std::string info = "selected node: " + m_treeSelInfo;
        drawText(m_theme, info.c_str(), 12, m_height - 22, 13, m_theme.textDim);
    }
}

void App::handleInput() {
    updateClock();

    // The MCTS search-tree screen owns all input while active.
    if (m_treeMode) {
        handleTreeInput();
        return;
    }

    // Window resize (drag-corner or fullscreen toggle) needs a layout pass
    // before any widget reads bounds.
    if (IsWindowResized()) {
        syncWindowSize();
    }

    // F11 toggles borderless fullscreen regardless of mode (editor or play).
    if (IsKeyPressed(KEY_F11)) {
        toggleFullscreen();
    }

    // Modal dialogs and pickers consume input first.
    if (m_fenDialog.active()) {
        resetHistoryNavigationRepeat();
        loadFenFromDialog();
        return;
    }

    if (m_promotionPicker->active()) {
        resetHistoryNavigationRepeat();
        auto chosen = m_promotionPicker->update();
        if (chosen.has_value()) {
            tryMoveWithClock(*chosen);
        }
        return;
    }

    m_controls.update();

    // Activation-panel zoom/pan/keyboard (no-op in editor / Off).
    handleActivationPanelCameraInput();

    if (m_editor.active()) {
        resetHistoryNavigationRepeat();
        // Editor widgets first.
        if (m_editorPalette) m_editorPalette->update();
        if (m_editorPanel) {
            m_editorPanel->update();
            switch (m_editorPanel->takeAction()) {
                case EditorPanel::Action::Validate: m_editor.validate();        break;
                case EditorPanel::Action::Apply:    leaveEditor(/*commit=*/true);  return;
                case EditorPanel::Action::Cancel:   leaveEditor(/*commit=*/false); return;
                case EditorPanel::Action::None: break;
            }
        }

        if (IsKeyPressed(KEY_F)) flipBoard();
        if (IsKeyPressed(KEY_ESCAPE)) leaveEditor(/*commit=*/false);

        // Board clicks place / erase pieces.
        bool left  = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        bool right = IsMouseButtonPressed(MOUSE_BUTTON_RIGHT);
        if (!left && !right) return;
        Vector2 mp = GetMousePosition();
        auto sq = m_board->squareAtPixel(mp);
        if (sq.has_value()) handleEditorBoardClick(*sq, left);
        return;
    }

    updateArchitectureSelector();

    if (m_moveList) {
        m_moveList->update();
        if (auto ply = m_moveList->takeClickedPly()) {
            stopSearch();
            m_game.jumpToPly(*ply);
            m_selected.reset();
            m_legalTargets = 0;
        }
    }

    if (IsKeyPressed(KEY_F)) flipBoard();
    if (IsKeyPressed(KEY_R)) resetGame();
    if (IsKeyPressed(KEY_S)) saveCurrentFen();
    if (IsKeyPressed(KEY_A)) cycleArchitecture();
    if (IsKeyPressed(KEY_T)) enterEditor();
    if (IsKeyPressed(KEY_G)) enterTreeMode();
    if (IsKeyPressed(KEY_E)) toggleSearch();
    if (IsKeyPressed(KEY_L)) {
        stopSearch();
        m_fenDialogOpensEditor = false;
        m_fenDialog.open();
    }
    handleHistoryNavigationKeys();
    if (IsKeyPressed(KEY_HOME)) {
        stopSearch();
        m_game.jumpToPly(0);
    }
    if (IsKeyPressed(KEY_END)) {
        stopSearch();
        m_game.jumpToPly(m_game.history().size());
    }
    if (IsKeyPressed(KEY_ESCAPE)) {
        if (m_searchActive) {
            stopSearch();
        } else {
            m_selected.reset();
            m_legalTargets = 0;
        }
    }
    if (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) {
        if (IsKeyPressed(KEY_Z)) undoMove();
        if (IsKeyPressed(KEY_Y)) redoMove();
    }

    if (m_searchActive) {
        m_selected.reset();
        m_legalTargets = 0;
        m_dragFrom.reset();
        m_dragging = false;
        return;
    }

    // Drag-distance update: while the user is holding the mouse over a
    // piece they pressed, mark the gesture as a drag once the cursor moves
    // far enough.
    if (m_dragFrom.has_value() && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mp = GetMousePosition();
        float dx = mp.x - m_dragStartMouse.x;
        float dy = mp.y - m_dragStartMouse.y;
        if (!m_dragging && dx * dx + dy * dy >= kDragThresholdSquared) {
            m_dragging = true;
        }
    }

    // Drag-release: a real drag tries to make the move from origin to the
    // square under the cursor; a non-dragged release is just a click and
    // leaves the existing selection intact.
    if (m_dragFrom.has_value() && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        if (m_dragging) {
            Vector2 mp = GetMousePosition();
            auto target = m_board->squareAtPixel(mp);
            if (target.has_value() && *target != *m_dragFrom) {
                const chess::Position& pos = m_game.position();
                chess::MoveList legal;
                m_game.generateLegalMoves(legal);
                chess::Move pawnMove = chess::Move::none();
                chess::Move quietMove = chess::Move::none();
                for (chess::Move m : legal) {
                    if (m.from() != *m_dragFrom || m.to() != *target) continue;
                    if (m.promotion() != chess::Move::Promotion::None) {
                        pawnMove = chess::Move(m.from(), m.to());
                    } else {
                        quietMove = m;
                    }
                }
                if (pawnMove.raw() != chess::Move::kNoMove) {
                    float side = m_board->squareSize();
                    Rectangle anchor{mp.x, mp.y - side / 2, side, side};
                    m_promotionPicker->open(pawnMove, pos.sideToMove(), anchor);
                    m_selected.reset();
                    m_legalTargets = 0;
                } else if (!quietMove.isNone()) {
                    tryMoveWithClock(quietMove);
                    m_selected.reset();
                    m_legalTargets = 0;
                }
            }
        }
        m_dragFrom.reset();
        m_dragging = false;
        // Even if we made a move, fall through; press is not active so the
        // click block below short-circuits anyway.
    }

    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return;

    Vector2 mp = GetMousePosition();
    auto sq = m_board->squareAtPixel(mp);
    if (!sq.has_value()) {
        m_selected.reset();
        m_legalTargets = 0;
        m_dragFrom.reset();
        m_dragging = false;
        return;
    }

    const chess::Position& pos = m_game.position();
    chess::MoveList legal;
    m_game.generateLegalMoves(legal);

    // If the press is on the side-to-move's own piece, seed a potential
    // drag. The actual drag-vs-click distinction is decided on release.
    {
        chess::Piece pressed = pos.pieceAt(*sq);
        if (!pressed.isNone() && pressed.color == pos.sideToMove()) {
            m_dragFrom = sq;
            m_dragStartMouse = mp;
            m_dragging = false;
        }
    }

    if (m_selected.has_value()) {
        chess::Move pawnMove = chess::Move::none();
        chess::Move quietMove = chess::Move::none();
        for (chess::Move m : legal) {
            if (m.from() != *m_selected || m.to() != *sq) continue;
            if (m.promotion() != chess::Move::Promotion::None) {
                pawnMove = chess::Move(m.from(), m.to());
            } else {
                quietMove = m;
            }
        }
        if (pawnMove.raw() != chess::Move::kNoMove) {
            float side = m_board->bounds().height / 8.0f;
            Rectangle anchor{
                static_cast<float>(GetMouseX()),
                static_cast<float>(GetMouseY()) - side / 2,
                side, side,
            };
            m_promotionPicker->open(pawnMove, pos.sideToMove(), anchor);
            m_selected.reset();
            m_legalTargets = 0;
            return;
        }
        if (!quietMove.isNone()) {
            tryMoveWithClock(quietMove);
            m_selected.reset();
            m_legalTargets = 0;
            return;
        }
        // Click was on something else — try reselect or clear.
        chess::Piece p = pos.pieceAt(*sq);
        if (!p.isNone() && p.color == pos.sideToMove()) {
            m_selected = sq;
            m_legalTargets = 0;
            for (chess::Move m : legal) {
                if (m.from() == *sq) m_legalTargets |= chess::squareBit(m.to());
            }
        } else {
            m_selected.reset();
            m_legalTargets = 0;
        }
        return;
    }

    // No prior selection — start one if we clicked on our own piece.
    chess::Piece p = pos.pieceAt(*sq);
    if (!p.isNone() && p.color == pos.sideToMove()) {
        m_selected = sq;
        m_legalTargets = 0;
        for (chess::Move m : legal) {
            if (m.from() == *sq) m_legalTargets |= chess::squareBit(m.to());
        }
    }
}

void App::evaluateIfStale() {
    pollEvalJob();

    // Skip in editor mode — the in-progress board may be illegal.
    if (m_editor.active()) {
        m_haveEval = false;
        m_evalFailed = false;
        m_activationFade = 0.0f;
        m_snapshot.clear();
        if (m_board) m_board->clearActivationOverlay();
        return;
    }

    if (m_arch == Arch::Off) {
        m_haveEval = false;
        m_evalFailed = false;
        m_evalError.clear();
        m_activationFade = 0.0f;
        m_snapshot.clear();
        if (m_board) m_board->clearActivationOverlay();
        return;
    }

    const chess::Position pos = currentEvalPosition();
    const std::uint64_t h = pos.hash();

    if (m_arch == Arch::Classical) {
        if (m_haveEval && h == m_lastEvalHash && m_lastEvalArch == m_arch) return;

        // Model-free: show the current pieces' own piece-square-table values on
        // the board while the Classical panel computes the full term breakdown.
        m_evalFailed = false;
        m_evalError.clear();
        m_snapshot.clear();
        m_lastEvalHash = h;
        m_lastEvalArch = m_arch;
        m_lastEvalSideToMove = pos.sideToMove();
        m_haveEval = true;
        m_activationFade = 1.0f;
        if (m_board) {
            m_board->setDimInactivePieces(false);
            m_board->setDimInactiveSquares(false);
            m_board->setActivationOverlay(chess::eval::pieceSquareHeatmap(pos));
        }
        return;
    }

    if (m_arch == Arch::Bt4) ensureBt4Loaded();

    if (m_haveEval && h == m_lastEvalHash && m_arch == m_lastEvalArch) return;

    if (EvalCache* cache = cachedEval(m_arch, h)) {
        displayCachedEval(*cache, /*restartFade=*/false);
        return;
    }

    if (m_evalFailed && h == m_evalFailedHash && m_arch == m_evalFailedArch) {
        return;
    }

    const bool keepStaleDisplay = m_haveEval && m_lastEvalArch == m_arch;
    if (!keepStaleDisplay) {
        m_haveEval = false;
        m_snapshot.clear();
        m_activationFade = 0.0f;
        if (m_board) m_board->clearActivationOverlay();
    }

    if (m_evalInFlight) return;

    m_evalFailed = false;
    m_evalError.clear();
    startEvalJob(pos, h, m_arch);
}

void App::pollEvalJob() {
    if (!m_evalInFlight || !m_evalFuture.valid()) return;

    if (m_evalFuture.wait_for(std::chrono::milliseconds(0))
        != std::future_status::ready) {
        return;
    }

    EvalResult result;
    try {
        result = m_evalFuture.get();
    } catch (const std::exception& e) {
        result.requestId = m_evalInFlightRequestId;
        result.hash = m_evalInFlightHash;
        result.arch = m_evalInFlightArch;
        result.ok = false;
        result.error = e.what();
    } catch (...) {
        result.requestId = m_evalInFlightRequestId;
        result.hash = m_evalInFlightHash;
        result.arch = m_evalInFlightArch;
        result.ok = false;
        result.error = "unknown exception";
    }

    m_evalInFlight = false;
    applyEvalResult(std::move(result));
}

void App::startEvalJob(cnnv::chess::Position pos, std::uint64_t hash, Arch arch) {
    m_evalInFlight = true;
    m_evalInFlightRequestId = ++m_evalRequestSerial;
    m_evalInFlightHash = hash;
    m_evalInFlightArch = arch;

    const std::uint64_t requestId = m_evalInFlightRequestId;
    const bool cnnLoaded = m_cnnLoaded;
    m_evalFuture = std::async(std::launch::async,
        [this, pos = std::move(pos), hash, arch, requestId, cnnLoaded]() mutable {
            EvalResult result;
            result.requestId = requestId;
            result.hash = hash;
            result.arch = arch;
            result.sideToMove = pos.sideToMove();
            try {
                switch (arch) {
                    case Arch::Cnn:
                        if (cnnLoaded) {
                            std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                            m_cnn.evaluate(pos, result.snapshot);
                            if (result.snapshot.has(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl) &&
                                result.snapshot.size(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl) == 3) {
                                const float* w =
                                    result.snapshot.data(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl);
                                result.wdl = {w[0], w[1], w[2]};
                                result.hasWdl = true;
                            }
                        }
                        break;
                    case Arch::Bt4:
                        {
                            std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                            m_bt4.evaluate(pos, result.snapshot);
                        }
                        if (result.snapshot.has(cnnv::nn::lc0_bt4::snapshot_keys::kValueWdl) &&
                            result.snapshot.size(cnnv::nn::lc0_bt4::snapshot_keys::kValueWdl) == 3) {
                            const float* w =
                                result.snapshot.data(cnnv::nn::lc0_bt4::snapshot_keys::kValueWdl);
                            result.wdl = {w[0], w[1], w[2]};
                            result.hasWdl = true;
                        }
                        if (result.snapshot.has(cnnv::nn::lc0_bt4::snapshot_keys::kValueScalar)) {
                            result.centipawns =
                                result.snapshot.data(cnnv::nn::lc0_bt4::snapshot_keys::kValueScalar)[0] * 100.0f;
                            result.hasCentipawns = true;
                        }
                        break;
                    case Arch::Classical:
                    case Arch::Off:
                        break;
                    case Arch::Nnue:
                        {
                            std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                            m_nnue.evaluate(pos, result.snapshot);
                        }
                        if (result.snapshot.has(cnnv::nn::nnue::snapshot_keys::kValueCentipawns)) {
                            result.centipawns = result.snapshot.data(
                                cnnv::nn::nnue::snapshot_keys::kValueCentipawns)[0];
                            result.hasCentipawns = true;
                        }
                        break;
                }
            } catch (const std::exception& e) {
                result.ok = false;
                result.error = e.what();
            } catch (...) {
                result.ok = false;
                result.error = "unknown exception";
            }
            return result;
        });
}

int App::evaluateSearchPositionForArch(const cnnv::chess::Position& pos,
                                       Arch arch,
                                       bool cnnLoaded) const {
    cnnv::nn::ActivationSnapshot snap;
    try {
        switch (arch) {
            case Arch::Nnue: {
                std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                m_nnue.evaluate(pos, snap);
                if (snap.size(cnnv::nn::nnue::snapshot_keys::kValueCentipawns) == 1) {
                    return static_cast<int>(std::lround(
                        snap.data(cnnv::nn::nnue::snapshot_keys::kValueCentipawns)[0]));
                }
                break;
            }
            case Arch::Cnn: {
                if (!cnnLoaded) break;
                std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                m_cnn.evaluate(pos, snap);
                if (snap.size(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl) == 3) {
                    const float* w = snap.data(cnnv::nn::lc0_cnn::snapshot_keys::kValueWdl);
                    return static_cast<int>(std::lround((w[0] - w[2]) * 100.0f));
                }
                break;
            }
            case Arch::Bt4: {
                std::lock_guard<std::mutex> lock(m_networkEvalMutex);
                m_bt4.evaluate(pos, snap);
                if (snap.size(cnnv::nn::lc0_bt4::snapshot_keys::kValueScalar) == 1) {
                    return static_cast<int>(std::lround(
                        snap.data(cnnv::nn::lc0_bt4::snapshot_keys::kValueScalar)[0] * 100.0f));
                }
                if (snap.size(cnnv::nn::lc0_bt4::snapshot_keys::kValueWdl) == 3) {
                    const float* w = snap.data(cnnv::nn::lc0_bt4::snapshot_keys::kValueWdl);
                    return static_cast<int>(std::lround((w[0] - w[2]) * 100.0f));
                }
                break;
            }
            case Arch::Classical:
                return chess::eval::evaluateWhiteBreakdown(pos).stmTotal();
            case Arch::Off:
                break;
        }
    } catch (const std::exception&) {
        // Search is visual and continuous; keep it running even if a model
        // cannot evaluate one position.
    } catch (...) {
    }
    return evaluateSearchPosition(pos);
}

void App::applyEvalResult(EvalResult&& result) {
    if (result.requestId > m_evalRequestSerial) return;

    const bool targetsCurrent =
        !m_editor.active() && result.hash == currentEvalHash() && result.arch == m_arch;

    if (!result.ok) {
        if (targetsCurrent) {
            std::fprintf(stderr, "[App] %s evaluate failed: %s\n",
                         archName(result.arch), result.error.c_str());
            m_evalFailed = true;
            m_evalFailedHash = result.hash;
            m_evalFailedArch = result.arch;
            m_evalError = result.error;
            const bool keepStaleDisplay = m_haveEval && m_lastEvalArch == m_arch;
            if (!keepStaleDisplay) {
                m_haveEval = false;
                m_snapshot.clear();
                m_activationFade = 0.0f;
                if (m_board) m_board->clearActivationOverlay();
            }
        }
        return;
    }

    EvalCache& cache = storeEvalResult(std::move(result));
    if (!targetsCurrent) return;

    displayCachedEval(cache, /*restartFade=*/true);
    m_evalFailed = false;
    m_evalError.clear();
}

void App::displayCachedEval(const EvalCache& cache, bool restartFade) {
    (void)restartFade;
    m_snapshot = cache.snapshot;
    if (cache.hasCentipawns) m_lastCentipawns = cache.centipawns;
    if (cache.hasWdl) {
        m_lastWdl[0] = cache.wdl[0];
        m_lastWdl[1] = cache.wdl[1];
        m_lastWdl[2] = cache.wdl[2];
    }
    m_lastEvalHash = cache.hash;
    m_lastEvalArch = cache.arch;
    m_lastEvalSideToMove = cache.sideToMove;
    m_haveEval = true;
    refreshActivationOverlay();
    m_activationFade = 1.0f;
}

void App::waitForEvalJob() {
    if (m_evalFuture.valid()) {
        m_evalFuture.wait();
    }
    m_evalInFlight = false;
}

bool App::evalJobTargetsCurrent(std::uint64_t hash, Arch arch) const noexcept {
    if (m_editor.active()) return false;
    return hash == currentEvalHash() && arch == m_arch;
}

bool App::evalInFlightForCurrent() const noexcept {
    return m_evalInFlight && evalJobTargetsCurrent(m_evalInFlightHash,
                                                   m_evalInFlightArch);
}

void App::refreshActivationOverlay() {
    // Pick the snapshot slot to drive the per-square heatmap. NNUE uses the
    // side-to-move's per-piece contribution magnitudes; CNN uses the
    // channel-mean of the final residual block. BT4 publishes positive,
    // board-aware token salience. CNN and BT4 tensors are in LC0 side-to-move
    // orientation; black-to-move maps back to absolute board squares with a
    // rank mirror.
    const char* key = nullptr;
    bool lc0Perspective = false;
    if (m_arch == Arch::Off || m_arch == Arch::Classical) {
        m_board->clearActivationOverlay();
        return;
    } else if (m_arch == Arch::Cnn && m_cnnLoaded) {
        m_board->setDimInactivePieces(false);
        m_board->setDimInactiveSquares(false);
        key = cnnv::nn::lc0_cnn::snapshot_keys::kFinalActivation;
        lc0Perspective = true;
    } else if (m_arch == Arch::Bt4) {
        m_board->setDimInactivePieces(true);
        m_board->setDimInactiveSquares(true);
        key = cnnv::nn::lc0_bt4::snapshot_keys::kBoardSalience;
        lc0Perspective = true;
    } else {
        m_board->setDimInactivePieces(true);
        m_board->setDimInactiveSquares(true);
        key = (m_lastEvalSideToMove == cnnv::chess::Color::White)
            ? cnnv::nn::nnue::snapshot_keys::kPieceContributionWhite
            : cnnv::nn::nnue::snapshot_keys::kPieceContributionBlack;
    }
    const float* data = m_snapshot.data(key);
    const std::size_t n = m_snapshot.size(key);
    if (data == nullptr || n != 64) {
        m_board->clearActivationOverlay();
        return;
    }
    // BT4 salience is already positive and board-aware. More generally, these
    // overlay tensors are meant to be read as magnitudes, so centering them
    // would make quiet squares look artificially important.
    constexpr float centre = 0.0f;

    float maxAbs = 0.0f;
    for (std::size_t i = 0; i < 64; ++i) {
        float v = m_arch == Arch::Bt4
            ? std::max(0.0f, data[i])
            : std::fabs(data[i] - centre);
        if (v > maxAbs) maxAbs = v;
    }
    if (maxAbs < 1e-6f) {
        m_board->clearActivationOverlay();
        return;
    }
    std::array<float, 64> overlay{};
    for (std::size_t i = 0; i < 64; ++i) {
        std::size_t boardIndex = i;
        if (lc0Perspective &&
            m_lastEvalSideToMove == cnnv::chess::Color::Black) {
            boardIndex = i ^ 56U;
        }
        float t = (m_arch == Arch::Bt4
            ? std::max(0.0f, data[i])
            : std::fabs(data[i] - centre)) / maxAbs;
        if (m_arch == Arch::Bt4) {
            constexpr float kNoiseFloor = 0.22f;
            t = t <= kNoiseFloor ? 0.0f : (t - kNoiseFloor) / (1.0f - kNoiseFloor);
            t = std::clamp(t, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t);
        }
        overlay[boardIndex] = t;
    }
    m_board->setActivationOverlay(overlay);
}

void App::render() {
    if (m_treeMode) {
        BeginDrawing();
        ClearBackground(m_theme.background);
        drawTreeScreen();
        EndDrawing();
        return;
    }

    pollSearchJob();
    const SearchPreview pendingSearch = currentSearchPreview();
    bool searchFrameReadyForAck =
        m_searchActive && pendingSearch.have && !m_editor.active() &&
        searchPreviewReadyForDisplayAck(pendingSearch);
    if (searchFrameReadyForAck) {
        searchFrameReadyForAck =
            promoteSearchPreviewForDisplay(pendingSearch.serial);
    }
    evaluateIfStale();
    const SearchPreview search = currentDisplayedSearchPreview();
    const bool showSearch = m_searchActive && search.have && !m_editor.active();

    m_activationFade = m_haveEval ? 1.0f : 0.0f;

    BeginDrawing();
    ClearBackground(m_theme.background);

    // Status banner across the top.
    Rectangle status{
        0.0f, 0.0f,
        static_cast<float>(m_width), 44.0f,
    };
    DrawRectangleRec(status, m_theme.panelBackground);
    std::string banner;
    if (m_editor.active()) {
        banner = "Editor mode \xE2\x80\x94 click squares to place pieces, right-click to erase";
    } else if (showSearch) {
        std::string line = search.line.empty() ? "..." : search.line;
        if (line.size() > 72) line = line.substr(0, 69) + "...";
        const std::string nodes = formatNodeCount(search.nodes);
        char text[224];
        std::snprintf(text, sizeof(text),
                      "%s search %s  depth %d  eval %+0.2f  nodes %s  node %s",
                      archName(m_arch),
                      search.complete ? "done" : "live",
                      search.depth,
                      search.score,
                      nodes.c_str(),
                      line.c_str());
        banner = text;
    } else {
        banner = statusText();
    }
    drawText(m_theme, banner.c_str(), 16, 11, 22, m_theme.textPrimary);
    drawClockArea(status);

    // Right-aligned eval readout (when not in editor).
    if (!m_editor.active()) {
        char eval[96];
        if (m_arch == Arch::Off) {
            std::snprintf(eval, sizeof(eval), "NN view off");
        } else if (m_arch == Arch::Classical) {
            const int cp =
                chess::eval::evaluateWhiteBreakdown(currentEvalPosition()).stmTotal();
            std::snprintf(eval, sizeof(eval), "Classical %+.2f  (cp %+d)",
                          static_cast<double>(cp) / 100.0, cp);
        } else if (m_haveEval && m_lastEvalArch == Arch::Bt4) {
            const float scalar = m_lastWdl[0] - m_lastWdl[2];
            std::snprintf(eval, sizeof(eval),
                          "BT4  W %.2f  D %.2f  L %.2f  v %+.2f",
                          m_lastWdl[0], m_lastWdl[1], m_lastWdl[2], scalar);
        } else if (m_arch == Arch::Cnn && !m_cnnLoaded) {
            std::snprintf(eval, sizeof(eval), "CNN no weights");
        } else if (m_haveEval && m_lastEvalArch == Arch::Cnn) {
            const float scalar = m_lastWdl[0] - m_lastWdl[2];
            std::snprintf(eval, sizeof(eval),
                          "CNN  W %.2f  D %.2f  L %.2f  v %+.2f",
                          m_lastWdl[0], m_lastWdl[1], m_lastWdl[2], scalar);
        } else if (m_haveEval) {
            std::snprintf(eval, sizeof(eval), "NNUE %+.2f  (cp %+.0f)",
                          m_lastCentipawns / 100.0f, m_lastCentipawns);
        } else if (evalInFlightForCurrent()) {
            std::snprintf(eval, sizeof(eval), "%s evaluating...", archName(m_arch));
        } else if (m_evalInFlight) {
            std::snprintf(eval, sizeof(eval), "%s queued...", archName(m_arch));
        } else if (m_evalFailed && evalJobTargetsCurrent(m_evalFailedHash, m_evalFailedArch)) {
            std::snprintf(eval, sizeof(eval), "%s failed", archName(m_arch));
        } else {
            std::snprintf(eval, sizeof(eval), "%s", archName(m_arch));
        }
        const int sz = 19;
        const int width = measureText(m_theme, eval, sz);
        drawText(m_theme, eval, m_width - width - 16, 14, sz,
                 m_theme.textMuted);
    }

    // Re-aim the BoardView at whichever position is "live" right now.
    const chess::Position& livePos = showSearch
        ? search.position
        : (m_editor.active() ? m_editor.position() : m_game.position());
    m_board->setPosition(livePos);
    m_board->setSelection((m_editor.active() || showSearch) ? std::nullopt : m_selected);
    m_board->setLegalDestinations((m_editor.active() || showSearch) ? 0ULL : m_legalTargets);
    std::optional<chess::Move> lastMove;
    if (showSearch && !search.lastMove.isNone()) {
        lastMove = search.lastMove;
    } else if (!m_editor.active() && m_game.history().current()) {
        lastMove = m_game.history().current()->move;
    }
    m_board->setLastMove(lastMove);
    if (m_dragging && m_dragFrom.has_value()) {
        m_board->setDragging(*m_dragFrom, GetMousePosition());
    } else {
        m_board->clearDragging();
    }
    const float fade = m_activationFade * m_activationFade * (3.0f - 2.0f * m_activationFade);
    m_board->setActivationOverlayOpacity(fade);
    m_board->draw(m_theme);

    // Right panel background.
    Rectangle panel{
        m_board->bounds().x + m_board->bounds().width, 44.0f,
        static_cast<float>(m_width) - (m_board->bounds().x + m_board->bounds().width),
        static_cast<float>(m_height) - 44.0f,
    };
    DrawRectangleRec(panel, m_theme.panelBackground);
    m_controls.draw(m_theme);
    if (m_editor.active()) {
        if (m_editorPalette) m_editorPalette->draw(m_theme);
        if (m_editorPanel)   m_editorPanel->draw(m_theme);
    } else {
        drawArchitectureSelector();
        if (IActivationView* view = activeActivationView()) {
            if (m_arch == Arch::Bt4 && m_bt4View) {
                m_bt4View->setPosition(&livePos);
            }
            if (m_arch == Arch::Classical && m_classicalView) {
                m_classicalView->setPosition(currentEvalPosition());
            }
            view->update(m_snapshot);
            // Draw the view through the panel camera so it can be zoomed/panned;
            // clip to the panel so zoomed content does not overflow onto the
            // board or move list. Identity camera renders exactly as before.
            const Camera2D cam = activationPanelCamera();
            BeginScissorMode(static_cast<int>(m_activationPanelBounds.x),
                             static_cast<int>(m_activationPanelBounds.y),
                             static_cast<int>(m_activationPanelBounds.width),
                             static_cast<int>(m_activationPanelBounds.height));
            BeginMode2D(cam);
            tensorviz::setActiveCamera(cam);
            view->draw(m_theme);
            tensorviz::clearActiveCamera();
            EndMode2D();
            EndScissorMode();
        }
        if (m_moveList) m_moveList->draw(m_theme);
    }

    constexpr float kFooterHeight = 44.0f;
    const Rectangle footer{
        panel.x,
        panel.y + panel.height - kFooterHeight,
        panel.width,
        kFooterHeight,
    };
    DrawRectangleRec(footer, tensorviz::blend(m_theme.panelBackground,
                                              m_theme.background, 0.22f));
    DrawRectangleRec(Rectangle{footer.x, footer.y, footer.width, 1.0f},
                     m_theme.panelBorder);
    auto footerLine = [&](const char* text, float y, Color color) {
        int fontSize = 13;
        const int maxWidth = static_cast<int>(footer.width - 24.0f);
        while (fontSize > 10 && measureText(m_theme, text, fontSize) > maxWidth) {
            --fontSize;
        }
        drawText(m_theme, text, static_cast<int>(footer.x + 12.0f),
                 static_cast<int>(y), fontSize, color);
    };
    footerLine(
        "Board: F flip  R reset  L load  S save  T setup  A view  G tree  E search",
        footer.y + 8.0f, m_theme.textMuted);
    const char* secondary =
        (!m_editor.active() && m_arch != Arch::Off)
            ? "Panel: wheel zoom  right/middle pan  +/-/0 reset    History: Ctrl+Z/Y  Left/Right  Home/End  Esc"
            : "F11 fullscreen  Ctrl+Z/Y undo/redo  Left/Right step  Home/End  Esc";
    footerLine(secondary, footer.y + 25.0f, m_theme.textDim);

    m_promotionPicker->draw(m_theme);
    m_fenDialog.draw(m_theme);

    EndDrawing();

    if (searchFrameReadyForAck) {
        markSearchPreviewDisplayed(pendingSearch.serial);
    }
}

int App::run() {
    while (!WindowShouldClose()) {
        handleInput();
        render();
    }
    return 0;
}

}  // namespace cnnv::viz
