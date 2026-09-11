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

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}
