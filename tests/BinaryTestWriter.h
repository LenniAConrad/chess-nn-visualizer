#pragma once

/**
 * @file BinaryTestWriter.h
 * @brief Kleine Schreibhilfen fuer binaere Test-Modelldateien.
 *
 * Die Loader-Tests erzeugen mehrere Mini-Netze im jeweiligen Dateiformat.
 * Diese Helfer halten die primitiven Little-Endian-Schreiboperationen an
 * einer Stelle, waehrend formatspezifische Strukturen in den Tests bleiben.
 */

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace cnnv_test::bin {

/** @brief Schreibt einen 32-Bit-Wert im nativen Little-Endian-Testformat. */
inline void writeU32(std::ofstream& os, std::uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/** @brief Schreibt einen signierten 32-Bit-Wert fuer CRTK-Laengenfelder. */
inline void writeI32(std::ofstream& os, std::int32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/** @brief Schreibt einen float32-Wert ohne zusaetzliche Metadaten. */
inline void writeF32(std::ofstream& os, float value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/** @brief Schreibt ein einzelnes Byte, meistens fuer boolsche Flags. */
inline void writeU8(std::ofstream& os, std::uint8_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/**
 * @brief Schreibt einen Laengen-prefixten ASCII-String mit int32-Laenge.
 *
 * BT4J nutzt diese Form fuer Enum- und Namensfelder.
 */
inline void writeStringI32(std::ofstream& os, const std::string& value) {
    writeI32(os, static_cast<std::int32_t>(value.size()));
    os.write(value.data(), static_cast<std::streamsize>(value.size()));
}

/**
 * @brief Schreibt ein float32-Array mit uint32-Laenge.
 *
 * Wird von den LC0J- und NNUE-Testdateien genutzt.
 */
inline void writeFloatArrayU32(std::ofstream& os,
                               const std::vector<float>& values) {
    writeU32(os, static_cast<std::uint32_t>(values.size()));
    if (!values.empty()) {
        os.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}

/**
 * @brief Schreibt ein float32-Array mit int32-Laenge.
 *
 * Wird vom BT4J-Testformat genutzt, das Java-ByteBuffer-Int-Laengen spiegelt.
 */
inline void writeFloatArrayI32(std::ofstream& os,
                               const std::vector<float>& values) {
    writeI32(os, static_cast<std::int32_t>(values.size()));
    if (!values.empty()) {
        os.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}

}  // namespace cnnv_test::bin
