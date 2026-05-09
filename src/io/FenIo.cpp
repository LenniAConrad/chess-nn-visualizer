#include "io/FenIo.h"

#include "chess/Fen.h"

#include <fstream>
#include <sstream>
#include <string>

namespace cnnv::io {

std::optional<cnnv::chess::Position> loadPositionFromFenFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;

    std::string line;
    while (std::getline(in, line)) {
        // Skip leading whitespace and entirely-empty lines.
        std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        std::string trimmed = line.substr(first);
        // Drop comments after a '#' character.
        std::size_t hash = trimmed.find('#');
        if (hash != std::string::npos) trimmed = trimmed.substr(0, hash);
        // Trim trailing whitespace.
        std::size_t last = trimmed.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) trimmed = trimmed.substr(0, last + 1);
        if (trimmed.empty()) continue;
        return cnnv::chess::Fen::parse(trimmed);
    }
    return std::nullopt;
}

bool savePositionToFenFile(const std::string& path,
                           const cnnv::chess::Position& pos) {
    std::ofstream out(path);
    if (!out) return false;
    out << cnnv::chess::Fen::format(pos) << '\n';
    return static_cast<bool>(out);
}

}  // namespace cnnv::io
