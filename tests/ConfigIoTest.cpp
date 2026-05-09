#include "TestMain.h"
#include "io/ConfigIo.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace cnnv::io;

namespace {

std::string tempPath(const char* leaf) {
    std::string p = "/tmp/cnnv_test_";
    p += leaf;
    return p;
}

}  // namespace

TEST(config_round_trip_save_and_load) {
    std::string path = tempPath("config.ini");
    {
        Config c;
        c.setString("window.title", "test");
        c.setInt   ("window.width", 1024);
        c.setBool  ("debug.verbose", true);
        CHECK(c.save(path));
    }
    {
        Config c;
        CHECK(c.load(path));
        CHECK_EQ(c.getString("window.title", "fallback"), std::string("test"));
        CHECK_EQ(c.getInt("window.width", 0), 1024);
        CHECK_EQ(c.getBool("debug.verbose", false), true);
    }
    std::remove(path.c_str());
}

TEST(config_returns_fallback_for_missing_keys) {
    Config c;
    CHECK_EQ(c.getString("absent.key", "default"), std::string("default"));
    CHECK_EQ(c.getInt("absent.int", 42), 42);
    CHECK_EQ(c.getBool("absent.bool", true), true);
}

TEST(config_load_skips_comments_and_blank_lines) {
    std::string path = tempPath("commented.ini");
    {
        std::ofstream out(path);
        out << "# header\n"
            << "\n"
            << "  ; another comment\n"
            << "[section] # ignored\n"
            << "key = value\n"
            << " indented = trim me \n";
    }
    Config c;
    CHECK(c.load(path));
    CHECK_EQ(c.getString("key", ""), std::string("value"));
    CHECK_EQ(c.getString("indented", ""), std::string("trim me"));
    std::remove(path.c_str());
}

TEST(config_int_parse_failure_falls_back) {
    std::string path = tempPath("bad-int.ini");
    {
        std::ofstream out(path);
        out << "n = not-a-number\n";
    }
    Config c;
    CHECK(c.load(path));
    CHECK_EQ(c.getInt("n", 7), 7);
    std::remove(path.c_str());
}

TEST(config_bool_accepts_common_spellings) {
    Config c;
    c.setString("a", "True");
    c.setString("b", "yes");
    c.setString("c", "1");
    c.setString("d", "off");
    CHECK_EQ(c.getBool("a", false), true);
    CHECK_EQ(c.getBool("b", false), true);
    CHECK_EQ(c.getBool("c", false), true);
    CHECK_EQ(c.getBool("d", true),  false);
}
