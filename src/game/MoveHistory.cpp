#include "game/MoveHistory.h"

namespace cnnv::game {

MoveHistory::MoveHistory() = default;

MoveHistory::~MoveHistory() {
    clear();
}

void MoveHistory::clear() {
    MoveRecord* node = m_head;
    while (node) {
        MoveRecord* next = node->next;
        delete node;
        node = next;
    }
    m_head = nullptr;
    m_current = nullptr;
}

void MoveHistory::truncateAfter(MoveRecord* node) noexcept {
    MoveRecord* victim = node ? node->next : m_head;
    if (node) node->next = nullptr;
    while (victim) {
        MoveRecord* next = victim->next;
        delete victim;
        victim = next;
    }
    if (!node) m_head = nullptr;
}

void MoveHistory::pushMove(MoveRecord rec) {
    truncateAfter(m_current);

    MoveRecord* node = new MoveRecord(std::move(rec));
    node->prev = m_current;
    node->next = nullptr;

    if (m_current) {
        m_current->next = node;
    } else {
        m_head = node;
    }
    m_current = node;
}

const MoveRecord* MoveHistory::undo() noexcept {
    if (!m_current) return nullptr;
    const MoveRecord* prevCurrent = m_current;
    m_current = m_current->prev;
    return prevCurrent;
}

const MoveRecord* MoveHistory::redo() noexcept {
    MoveRecord* next = m_current ? m_current->next : m_head;
    if (!next) return nullptr;
    m_current = next;
    return m_current;
}

std::size_t MoveHistory::size() const noexcept {
    std::size_t n = 0;
    for (const MoveRecord* node = m_head; node; node = node->next) ++n;
    return n;
}

std::size_t MoveHistory::plyCount() const noexcept {
    std::size_t n = 0;
    for (const MoveRecord* node = m_current; node; node = node->prev) ++n;
    return n;
}

}  // namespace cnnv::game
