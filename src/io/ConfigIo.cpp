#include "io/ConfigIo.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace cnnv::io {

namespace {

std::string trim(const std::string& s) {
    std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

}  // namespace

Config::Config() = default;

bool Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        if (t[0] == '#' || t[0] == ';' || t[0] == '[') continue;
        std::size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(t.substr(0, eq));
        std::string value = trim(t.substr(eq + 1));
        if (!key.empty()) m_values[key] = value;
    }
    return true;
}

bool Config::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    out << "# cpp-nn-visualizer configuration\n";
    for (const auto& [k, v] : m_values) {
        out << k << " = " << v << '\n';
    }
    return static_cast<bool>(out);
}

std::string Config::getString(const std::string& key, const std::string& fallback) const {
    auto it = m_values.find(key);
    return it == m_values.end() ? fallback : it->second;
}

int Config::getInt(const std::string& key, int fallback) const {
    auto it = m_values.find(key);
    if (it == m_values.end()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

bool Config::getBool(const std::string& key, bool fallback) const {
    auto it = m_values.find(key);
    if (it == m_values.end()) return fallback;
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (v == "true" || v == "1" || v == "yes" || v == "on")  return true;
    if (v == "false" || v == "0" || v == "no" || v == "off") return false;
    return fallback;
}

void Config::setString(const std::string& key, const std::string& value) {
    m_values[key] = value;
}

void Config::setInt(const std::string& key, int value) {
    m_values[key] = std::to_string(value);
}

void Config::setBool(const std::string& key, bool value) {
    m_values[key] = value ? "true" : "false";
}

}  // namespace cnnv::io
