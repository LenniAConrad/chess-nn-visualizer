#pragma once

/**
 * @file WeightFileReader.h
 * @brief Little-endian binary reader for neural-network weight files.
 */

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace cnnv::io {

/**
 * @brief Little-endian reader for flat neural-network weight blobs.
 *
 * The class owns one `std::ifstream`, optionally validates a magic header when
 * opened, and exposes typed scalar and array reads that throw on EOF or short
 * reads. All multi-byte values are interpreted as little-endian.
 */
class WeightFileReader {
   public:
    WeightFileReader() = default;

    /**
     * @brief Opens a weight file and optionally validates a magic header.
     * @param path File to open.
     * @param expected_magic Optional byte prefix that must match.
     *
     * Throws `std::runtime_error` if opening fails or the magic does not match.
     */
    void open(const std::string& path,
              const std::string& expected_magic = {});

    /**
     * @brief Closes the current file, if any.
     */
    void close();

    /**
     * @brief Tests whether a file is currently open.
     */
    bool is_open() const { return m_stream.is_open(); }

    /** @brief Reads an unsigned 32-bit integer. */
    std::uint32_t read_u32();

    /** @brief Reads a signed 32-bit integer. */
    std::int32_t read_i32();

    /** @brief Reads an unsigned 16-bit integer. */
    std::uint16_t read_u16();

    /** @brief Reads a signed 16-bit integer. */
    std::int16_t read_i16();

    /** @brief Reads an unsigned 8-bit integer. */
    std::uint8_t read_u8();

    /** @brief Reads a signed 8-bit integer. */
    std::int8_t read_i8();

    /** @brief Reads a 32-bit IEEE float. */
    float read_f32();

    /**
     * @brief Reads raw bytes into a caller-provided buffer.
     * @param dst Destination buffer.
     * @param count Number of bytes to read.
     */
    void read_bytes(void* dst, std::size_t count);

    /** @brief Reads `count` float32 values into a pre-sized buffer. */
    void read_f32_array(float* dst, std::size_t count);

    /** @brief Reads `count` signed 16-bit values into a pre-sized buffer. */
    void read_i16_array(std::int16_t* dst, std::size_t count);

    /** @brief Reads `count` signed 8-bit values into a pre-sized buffer. */
    void read_i8_array(std::int8_t* dst, std::size_t count);

    /**
     * @brief Reads a fixed-length string.
     * @param length Number of bytes to consume.
     * @return String containing exactly `length` bytes.
     */
    std::string read_fixed_string(std::size_t length);

    /**
     * @brief Current stream offset in bytes.
     */
    std::size_t tell();

    /**
     * @brief Moves the stream to an absolute byte offset.
     */
    void seek(std::size_t offset);

    /**
     * @brief Total file size in bytes.
     */
    std::size_t size() const { return m_size; }

    /**
     * @brief Path of the currently opened file.
     */
    const std::string& path() const { return m_path; }

   private:
    void ensure_open() const;

    std::ifstream m_stream;
    std::string m_path;
    std::size_t m_size = 0;
};

}  // namespace cnnv::io
