#include "game/Pgn.h"

#include "chess/Fen.h"

#include <ctime>
#include <fstream>
#include <sstream>

namespace cnnv::game {

namespace {

const char* resultTag(GameStatus s) {
    switch (s) {
        case GameStatus::WhiteWins: return "1-0";
        case GameStatus::BlackWins: return "0-1";
        case GameStatus::DrawStalemate:
        case GameStatus::DrawInsufficientMaterial:
        case GameStatus::DrawFiftyMove:
        case GameStatus::DrawThreefold:
            return "1/2-1/2";
        case GameStatus::Ongoing:
        default:
            return "*";
    }
}

std::string todayTagDate() {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y.%m.%d", &tm);
    return std::string(buf);
}

}  // namespace

std::string exportPgn(const Game& g, const std::string& whiteName,
                      const std::string& blackName) {
    const char* result = resultTag(g.status());

    std::ostringstream out;
    out << "[Event \"Casual Game\"]\n"
        << "[Site \"cpp-nn-visualizer\"]\n"
        << "[Date \"" << todayTagDate() << "\"]\n"
        << "[Round \"-\"]\n"
        << "[White \"" << whiteName << "\"]\n"
        << "[Black \"" << blackName << "\"]\n"
        << "[Result \"" << result << "\"]\n"
        << "\n";

    int ply = 0;
    int charsOnLine = 0;
    auto append = [&](const std::string& token) {
        if (charsOnLine + static_cast<int>(token.size()) + 1 > 80) {
            out << '\n';
            charsOnLine = 0;
        } else if (charsOnLine > 0) {
            out << ' ';
            ++charsOnLine;
        }
        out << token;
        charsOnLine += static_cast<int>(token.size());
    };

    for (const MoveRecord* node = g.history().head(); node; node = node->next) {
        if (ply % 2 == 0) {
            std::string num = std::to_string(ply / 2 + 1) + ".";
            append(num);
        }
        append(node->san);
        ++ply;
    }

    if (charsOnLine > 0) out << ' ';
    out << result << '\n';
    return out.str();
}

bool writePgnFile(const std::string& path, const Game& g,
                  const std::string& whiteName,
                  const std::string& blackName) {
    std::ofstream f(path);
    if (!f) return false;
    f << exportPgn(g, whiteName, blackName);
    return static_cast<bool>(f);
}

}  // namespace cnnv::game
