#pragma once

/**
 * @file ConfigIo.h
 * @brief Minimal key/value config loader and saver.
 */

#include <map>
#include <string>

namespace cnnv::io {

/**
 * @brief Minimal INI-like flat key/value configuration.
 *
 * Supported lines are `key = value`. Section headers are ignored, keys are
 * flat, and comments begin with `#` or `;`. The app uses this for window size,
 * default FEN, asset directories, model paths, and other runtime settings.
 */
class Config {
public:
    Config();

    /**
     * @brief Loads values from a config file.
     * @param path File to read.
     * @return True if opened and parsed; false if the file could not be opened.
     *
     * Missing keys keep their current values.
     */
    bool load(const std::string& path);

    /**
     * @brief Saves all key/value pairs sorted by key.
     * @param path File to overwrite.
     * @return True on success; false on I/O error.
     */
    bool save(const std::string& path) const;

    /**
     * @brief Reads a string value.
     * @param key Config key.
     * @param fallback Value returned when the key is absent.
     */
    std::string getString(const std::string& key, const std::string& fallback) const;

    /**
     * @brief Reads an integer value.
     * @return Parsed integer, or `fallback` when missing or malformed.
     */
    int         getInt   (const std::string& key, int fallback) const;

    /**
     * @brief Reads a boolean value.
     * @return Parsed boolean, or `fallback` when missing or malformed.
     */
    bool        getBool  (const std::string& key, bool fallback) const;

    /** @brief Sets a string value. */
    void setString(const std::string& key, const std::string& value);

    /** @brief Sets an integer value. */
    void setInt   (const std::string& key, int value);

    /** @brief Sets a boolean value. */
    void setBool  (const std::string& key, bool value);

private:
    std::map<std::string, std::string> m_values;
};

}  // namespace cnnv::io
