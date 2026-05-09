#pragma once

/**
 * @file Tensor.h
 * @brief Small fixed-rank tensor container used by native NN operations.
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnnv::nn {

/**
 * @brief Row-major owning tensor with compile-time rank.
 * @tparam T Element type.
 * @tparam Rank Number of dimensions.
 *
 * The tensor stores values in a flat `std::vector` and computes row-major
 * strides from the shape. It is intentionally lightweight: no expression
 * templates, broadcasting, or views.
 */
template <typename T, std::size_t Rank>
class Tensor {
   public:
    /** @brief Element type alias. */
    using value_type = T;

    /** @brief Fixed-size shape array. */
    using Shape = std::array<std::size_t, Rank>;

    /**
     * @brief Creates an empty tensor with all shape/stride entries zero.
     */
    Tensor() {
        m_shape.fill(0);
        m_strides.fill(0);
    }

    /**
     * @brief Creates a tensor with the supplied shape and zero-filled data.
     */
    explicit Tensor(const Shape& shape) { reshape(shape); }

    /**
     * @brief Creates a tensor from an initializer-list shape.
     * @throws std::invalid_argument if the number of dimensions does not equal
     * `Rank`.
     */
    Tensor(std::initializer_list<std::size_t> dims) {
        if (dims.size() != Rank) {
            throw std::invalid_argument("Tensor: dim count != Rank");
        }
        Shape s{};
        std::size_t i = 0;
        for (auto d : dims) s[i++] = d;
        reshape(s);
    }

    /**
     * @brief Replaces the shape and zero-fills the resized storage.
     * @param shape New tensor shape.
     */
    void reshape(const Shape& shape) {
        m_shape = shape;
        recompute_strides();
        m_data.assign(total_size(), T{});
    }

    /**
     * @brief Fills every tensor element with one value.
     */
    void fill(T value) { std::fill(m_data.begin(), m_data.end(), value); }

    /**
     * @brief Mutable element access by multi-dimensional index.
     * @param idx Exactly `Rank` indices.
     * @return Reference to the indexed element.
     */
    template <typename... Idx>
    T& at(Idx... idx) {
        static_assert(sizeof...(Idx) == Rank, "at(): wrong arity");
        return m_data[flat_index(std::array<std::size_t, Rank>{
            static_cast<std::size_t>(idx)...})];
    }

    /**
     * @brief Const element access by multi-dimensional index.
     * @param idx Exactly `Rank` indices.
     * @return Const reference to the indexed element.
     */
    template <typename... Idx>
    const T& at(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "at(): wrong arity");
        return m_data[flat_index(std::array<std::size_t, Rank>{
            static_cast<std::size_t>(idx)...})];
    }

    /** @brief Mutable pointer to flat row-major storage. */
    T* data() { return m_data.data(); }

    /** @brief Const pointer to flat row-major storage. */
    const T* data() const { return m_data.data(); }

    /** @brief Number of elements stored. */
    std::size_t size() const { return m_data.size(); }

    /** @brief Full tensor shape. */
    const Shape& shape() const { return m_shape; }

    /** @brief Size of one dimension. */
    std::size_t dim(std::size_t i) const { return m_shape[i]; }

   private:
    /**
     * @brief Computes the product of all dimensions.
     */
    std::size_t total_size() const {
        std::size_t n = 1;
        for (auto d : m_shape) n *= d;
        return n;
    }

    /**
     * @brief Recomputes row-major strides from the current shape.
     */
    void recompute_strides() {
        std::size_t s = 1;
        for (std::size_t i = Rank; i-- > 0;) {
            m_strides[i] = s;
            s *= m_shape[i];
        }
    }

    /**
     * @brief Converts a multi-dimensional index to a flat offset.
     */
    std::size_t flat_index(const std::array<std::size_t, Rank>& idx) const {
        std::size_t flat = 0;
        for (std::size_t i = 0; i < Rank; ++i) {
            assert(idx[i] < m_shape[i] && "Tensor index out of range");
            flat += idx[i] * m_strides[i];
        }
        return flat;
    }

    std::vector<T> m_data;
    Shape m_shape{};
    Shape m_strides{};
};

}  // namespace cnnv::nn
