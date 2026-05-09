#include "ActivationSnapshot.h"

#include <stdexcept>

namespace cnnv::nn {

namespace {

std::size_t product(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

const std::vector<std::size_t>& empty_shape() {
    static const std::vector<std::size_t> kEmpty;
    return kEmpty;
}

}  // namespace

void ActivationSnapshot::clear() { m_entries.clear(); }

float* ActivationSnapshot::allocate(const std::string& key,
                                    std::vector<std::size_t> shape) {
    Entry e;
    e.data.assign(product(shape), 0.0f);
    e.shape = std::move(shape);
    auto& slot = m_entries[key] = std::move(e);
    return slot.data.data();
}

void ActivationSnapshot::store(const std::string& key,
                               std::vector<std::size_t> shape,
                               const float* data) {
    const std::size_t n = product(shape);
    Entry e;
    if (n > 0 && data != nullptr) {
        e.data.assign(data, data + n);
    } else {
        e.data.assign(n, 0.0f);
    }
    e.shape = std::move(shape);
    m_entries[key] = std::move(e);
}

bool ActivationSnapshot::has(const std::string& key) const {
    return m_entries.find(key) != m_entries.end();
}

const ActivationSnapshot::Entry* ActivationSnapshot::find(
    const std::string& key) const {
    auto it = m_entries.find(key);
    return it == m_entries.end() ? nullptr : &it->second;
}

const float* ActivationSnapshot::data(const std::string& key) const {
    auto it = m_entries.find(key);
    return it == m_entries.end() ? nullptr : it->second.data.data();
}

std::size_t ActivationSnapshot::size(const std::string& key) const {
    auto it = m_entries.find(key);
    return it == m_entries.end() ? 0 : it->second.data.size();
}

const std::vector<std::size_t>& ActivationSnapshot::shape(
    const std::string& key) const {
    auto it = m_entries.find(key);
    return it == m_entries.end() ? empty_shape() : it->second.shape;
}

}  // namespace cnnv::nn
