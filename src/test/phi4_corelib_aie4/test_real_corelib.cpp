#include "corelib/corelib_api.hpp"
#include "corelib/corelib_runtime.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* configured = std::getenv("FLM_AIE4_CORELIB_PATH");
    if (configured == nullptr || *configured == '\0') {
        std::cout << "SKIP: FLM_AIE4_CORELIB_PATH is unset\n";
        return 77;
    }

    try {
        const auto api = flm::corelib::CorelibApi::Load(
            flm::corelib::CorelibApi::ResolveLibraryPath(
                std::filesystem::current_path()));
        const auto version = api->runtime_version();
        TEST_REQUIRE(version.major == 0 && version.minor == 3 && version.patch == 0);
#define FLM_ASSERT_CORELIB_SYMBOL(member, symbol) TEST_REQUIRE(api->functions().member != nullptr);
        FLM_CORELIB_FUNCTIONS(FLM_ASSERT_CORELIB_SYMBOL)
#undef FLM_ASSERT_CORELIB_SYMBOL
        auto runtime = flm::corelib::CorelibRuntime::CreateForTest(api);
        runtime.reset();
        flm::corelib::CorelibRuntime::ShutdownProcess();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
