#include "WeightFileReader.h"

#include <stdexcept>

namespace cnnv::io {

namespace {

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "WeightFileReader assumes a little-endian host"
#endif

}  // namespace

void WeightFileReader::open(const std::string& path,
                            const std::string& expected_magic) {
    m_stream.open(path, std::ios::binary);
    if (!m_stream) {
        throw std::runtime_error("WeightFileReader: cannot open " + path);
    }
    m_path = path;

    m_stream.seekg(0, std::ios::end);
    m_size = static_cast<std::size_t>(m_stream.tellg());
    m_stream.seekg(0, std::ios::beg);

    if (!expected_magic.empty()) {
        std::string actual = read_fixed_string(expected_magic.size());
        if (actual != expected_magic) {
            throw std::runtime_error("WeightFileReader: magic mismatch in " +
                                     path + " (expected '" + expected_magic +
                                     "', got '" + actual + "')");
        }
    }
}

void WeightFileReader::close() {
    if (m_stream.is_open()) m_stream.close();
    m_path.clear();
    m_size = 0;
}

void WeightFileReader::ensure_open() const {
    if (!m_stream.is_open()) {
        throw std::runtime_error("WeightFileReader: not open");
    }
}

void WeightFileReader::read_bytes(void* dst, std::size_t count) {
    ensure_open();
    m_stream.read(static_cast<char*>(dst), static_cast<std::streamsize>(count));
    if (m_stream.gcount() != static_cast<std::streamsize>(count)) {
        throw std::runtime_error("WeightFileReader: short read in " + m_path);
    }
}

std::uint32_t WeightFileReader::read_u32() {
    std::uint32_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

std::int32_t WeightFileReader::read_i32() {
    std::int32_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

std::uint16_t WeightFileReader::read_u16() {
    std::uint16_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

std::int16_t WeightFileReader::read_i16() {
    std::int16_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

std::uint8_t WeightFileReader::read_u8() {
    std::uint8_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

std::int8_t WeightFileReader::read_i8() {
    std::int8_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

float WeightFileReader::read_f32() {
    float v;
    read_bytes(&v, sizeof(v));
    return v;
}

void WeightFileReader::read_f32_array(float* dst, std::size_t count) {
    read_bytes(dst, count * sizeof(float));
}

void WeightFileReader::read_i16_array(std::int16_t* dst, std::size_t count) {
    read_bytes(dst, count * sizeof(std::int16_t));
}

void WeightFileReader::read_i8_array(std::int8_t* dst, std::size_t count) {
    read_bytes(dst, count * sizeof(std::int8_t));
}

std::string WeightFileReader::read_fixed_string(std::size_t length) {
    std::string s(length, '\0');
    read_bytes(s.data(), length);
    return s;
}

std::size_t WeightFileReader::tell() {
    ensure_open();
    return static_cast<std::size_t>(m_stream.tellg());
}

void WeightFileReader::seek(std::size_t offset) {
    ensure_open();
    m_stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
}

}  // namespace cnnv::io
