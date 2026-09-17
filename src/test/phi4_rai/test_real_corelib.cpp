#include "rai/corelib_api.hpp"
#include "rai/corelib_runtime.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* configured = std::getenv("FLM_RAI_CORELIB_PATH");
    if (configured == nullptr || *configured == '\0') {
        std::cout << "SKIP: FLM_RAI_CORELIB_PATH is unset\n";
        return 77;
    }

    try {
        const auto api = flm::corelib::CorelibApi::Load(
            flm::corelib::CorelibApi::ResolveLibraryPath(
                std::filesystem::current_path()));
        const auto version = api->runtime_version();
        // The pin this build was compiled against, not a literal: the
        // point is that the DLL on this box is the one FastFlowLM was
        // built for, whichever that is.
        TEST_REQUIRE(version.major == RYZENAI_CORELIB_VERSION_MAJOR &&
                     version.minor == RYZENAI_CORELIB_VERSION_MINOR &&
                     version.patch == RYZENAI_CORELIB_VERSION_PATCH);
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
