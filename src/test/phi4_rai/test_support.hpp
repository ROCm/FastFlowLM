#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#define TEST_REQUIRE(condition)                                                   \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +       \
                                     #condition);                                  \
        }                                                                         \
    } while (false)

inline void RequireContains(std::string_view text, std::string_view expected) {
    if (text.find(expected) == std::string_view::npos) {
        throw std::runtime_error("expected '" + std::string(text) +
                                 "' to contain '" + std::string(expected) + "'");
    }
}

template <typename Exception = std::exception, typename Callable>
std::string RequireThrows(Callable&& callable) {
    try {
        callable();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

/// \brief how many tests have failed so far in this binary
/// \note Exiting on the first failure hides every later one, so a suite with
///       three broken tests looks like a suite with one and each fix uncovers
///       the next. Keep going and fail the process at the end instead.
inline int& FailureCount() {
    static int failures = 0;
    return failures;
}

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        // Every main here ends with an unconditional "PASS" and return 0, so
        // the process exit code has to be forced from here. atexit runs after
        // main returns, which is late enough to have counted every test.
        if (++FailureCount() == 1) {
            std::atexit([] {
                std::cerr << FailureCount() << " test(s) FAILED\n";
                std::cerr.flush();
                std::_Exit(1);
            });
        }
    }
}
