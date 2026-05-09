#pragma once

/**
 * @file ActivationSnapshot.h
 * @brief Network-agnostic activation capture buffer for visualization panels.
 */

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cnnv::nn {

/**
 * @brief Keyed collection of tensors emitted by a network forward pass.
 *
 * Entries are flat float buffers keyed by stable strings such as
 * `nnue.accumulator.white`, `cnn.block3.relu`, or `bt4.block5.attention.heads`.
 * Each entry also stores its logical shape so views can reshape the data on
 * read without knowing the producer's internal storage type.
 */
class ActivationSnapshot {
   public:
    /**
     * @brief One captured activation tensor.
     */
    struct Entry {
        /** @brief Tensor shape in row-major logical order. */
        std::vector<std::size_t> shape;

        /** @brief Flat tensor values. Product of `shape` equals `data.size()`. */
        std::vector<float> data;
    };

    /**
     * @brief Removes every captured tensor.
     */
    void clear();

    /**
     * @brief Allocates or replaces a zero-filled entry.
     * @param key Stable activation key.
     * @param shape Logical tensor shape.
     * @return Pointer to the entry's writable data buffer.
     */
    float* allocate(const std::string& key,
                    std::vector<std::size_t> shape);

    /**
     * @brief Copies an activation tensor into the snapshot.
     * @param key Stable activation key.
     * @param shape Logical tensor shape.
     * @param data Source pointer. Length must equal the product of `shape`.
     */
    void store(const std::string& key, std::vector<std::size_t> shape,
               const float* data);

    /**
     * @brief Tests whether a key is present.
     */
    bool has(const std::string& key) const;

    /**
     * @brief Finds an entry by key.
     * @return Pointer to the entry, or null when missing.
     */
    const Entry* find(const std::string& key) const;

    /**
     * @brief Returns the raw data pointer for an entry.
     * @return Data pointer, or null when the key is missing.
     */
    const float* data(const std::string& key) const;

    /**
     * @brief Returns the number of floats stored for a key.
     * @return Entry size, or zero when the key is missing.
     */
    std::size_t size(const std::string& key) const;

    /**
     * @brief Returns the logical shape for a key.
     * @return Stored shape, or an empty shape when the key is missing.
     */
    const std::vector<std::size_t>& shape(const std::string& key) const;

    /**
     * @brief Exposes all entries for generic inspection and debug views.
     */
    const std::unordered_map<std::string, Entry>& entries() const {
        return m_entries;
    }

   private:
    std::unordered_map<std::string, Entry> m_entries;
};

}  // namespace cnnv::nn
