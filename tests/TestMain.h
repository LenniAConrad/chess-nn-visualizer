#pragma once

// Tiny hand-rolled test framework. Avoids a third-party test dependency
// while still giving us TEST(name) {...} blocks that auto-register and a
// CHECK(expr) macro that fails loudly. Build a single executable that
// includes this header + every test .cpp, then run it via ctest.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace cnnv_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct CheckFailure : std::exception {
    std::string message;
    explicit CheckFailure(std::string m) : message(std::move(m)) {}
    const char* what() const noexcept override { return message.c_str(); }
};

inline int run_all() {
    int failed = 0;
    int passed = 0;
    for (const auto& test : registry()) {
        try {
            test.fn();
            std::printf("[ OK  ] %s\n", test.name.c_str());
            ++passed;
        } catch (const CheckFailure& e) {
            std::printf("[FAIL ] %s: %s\n", test.name.c_str(), e.what());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("[THROW] %s: %s\n", test.name.c_str(), e.what());
            ++failed;
        } catch (...) {
            std::printf("[THROW] %s: unknown exception\n", test.name.c_str());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed (%zu total)\n",
                passed, failed, registry().size());
    return failed == 0 ? 0 : 1;
}

}  // namespace cnnv_test

#define TEST(name)                                                         \
    static void test_##name();                                             \
    static ::cnnv_test::Registrar registrar_##name(#name, &test_##name);   \
    static void test_##name()

#define CHECK(expr)                                                        \
    do {                                                                   \
        if (!(expr)) {                                                     \
            throw ::cnnv_test::CheckFailure(                               \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +   \
                ": CHECK(" #expr ") failed");                              \
        }                                                                  \
    } while (0)

#define CHECK_EQ(a, b)                                                     \
    do {                                                                   \
        auto _a = (a);                                                     \
        auto _b = (b);                                                     \
        if (!(_a == _b)) {                                                 \
            throw ::cnnv_test::CheckFailure(                               \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +   \
                ": CHECK_EQ(" #a ", " #b ") failed");                      \
        }                                                                  \
    } while (0)
