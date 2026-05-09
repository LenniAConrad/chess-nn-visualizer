#include "game/Game.h"

#include "chess/Fen.h"
#include "chess/MoveGenerator.h"
#include "chess/MoveList.h"
#include "chess/San.h"

namespace cnnv::game {

Game::Game() {
    reset();
}

void Game::reset() {
    m_position.setStartpos();
    m_history.clear();
}

bool Game::loadFen(const std::string& fen) {
    auto opt = cnnv::chess::Fen::parse(fen);
    if (!opt.has_value()) return false;
    m_position = *opt;
    m_history.clear();
    return true;
}

void Game::generateLegalMoves(cnnv::chess::MoveList& out) {
    cnnv::chess::MoveGenerator::generateLegal(m_position, out);
}

bool Game::tryMove(cnnv::chess::Move m) {
    cnnv::chess::MoveList legal;
    cnnv::chess::MoveGenerator::generateLegal(m_position, legal);
    bool ok = false;
    for (cnnv::chess::Move legalMove : legal) {
        if (legalMove == m) { ok = true; break; }
    }
    if (!ok) return false;

    MoveRecord rec;
    rec.move = m;
    rec.fenBefore = cnnv::chess::Fen::format(m_position);
    rec.san = cnnv::chess::San::toSan(m_position, m);
    m_position.make(m);
    rec.fenAfter = cnnv::chess::Fen::format(m_position);
    m_history.pushMove(std::move(rec));
    return true;
}

bool Game::undo() {
    if (!m_history.canUndo()) return false;
    m_position.unmake();
    m_history.undo();
    return true;
}

bool Game::redo() {
    if (!m_history.canRedo()) return false;
    const MoveRecord* node = m_history.redo();
    if (!node) return false;
    m_position.make(node->move);
    return true;
}

bool Game::jumpToPly(std::size_t targetPly) {
    if (targetPly > m_history.size()) return false;
    while (m_history.plyCount() > targetPly) {
        if (!undo()) return false;
    }
    while (m_history.plyCount() < targetPly) {
        if (!redo()) return false;
    }
    return true;
}

GameStatus Game::status() const {
    if (m_position.isCheckmate()) {
        return m_position.sideToMove() == cnnv::chess::Color::White
            ? GameStatus::BlackWins : GameStatus::WhiteWins;
    }
    if (m_position.isStalemate())            return GameStatus::DrawStalemate;
    if (m_position.isInsufficientMaterial()) return GameStatus::DrawInsufficientMaterial;
    if (m_position.isFiftyMoveDraw())        return GameStatus::DrawFiftyMove;
    if (m_position.isThreefoldRepetition())  return GameStatus::DrawThreefold;
    return GameStatus::Ongoing;
}

std::string Game::statusText() const {
    switch (status()) {
        case GameStatus::WhiteWins:                return "Checkmate — White wins";
        case GameStatus::BlackWins:                return "Checkmate — Black wins";
        case GameStatus::DrawStalemate:            return "Stalemate — Draw";
        case GameStatus::DrawInsufficientMaterial: return "Draw — Insufficient material";
        case GameStatus::DrawFiftyMove:            return "Draw — Fifty-move rule";
        case GameStatus::DrawThreefold:            return "Draw — Threefold repetition";
        case GameStatus::Ongoing:
            return m_position.sideToMove() == cnnv::chess::Color::White
                ? "White to move" : "Black to move";
    }
    return {};
}

}  // namespace cnnv::game
