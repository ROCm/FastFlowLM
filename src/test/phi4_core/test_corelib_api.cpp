#include "core/corelib_object.hpp"
#include "core/corelib_runtime.hpp"
#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
template <typename>
struct ReturnsStatus : std::false_type {};
template <typename... Args>
struct ReturnsStatus<ryzenai_corelib_status (*)(Args...)> : std::true_type {};

#define FLM_COUNT_SYMBOL(member, symbol) +1
constexpr std::size_t kSymbolCount = 0 FLM_CORELIB_FUNCTIONS(FLM_COUNT_SYMBOL);
#undef FLM_COUNT_SYMBOL
#define FLM_COUNT_STATUS_SYMBOL(member, symbol) +(ReturnsStatus<decltype(&::symbol)>::value ? 1 : 0)
constexpr std::size_t kStatusSymbolCount = 0 FLM_CORELIB_FUNCTIONS(FLM_COUNT_STATUS_SYMBOL);
#undef FLM_COUNT_STATUS_SYMBOL

using flm::corelib::CorelibApi;
using flm::corelib::CorelibError;
using flm::corelib::CorelibRuntime;
using flm::corelib::UniqueHostView;
using flm::corelib::UniqueMatMulWeights;
using flm::corelib::UniqueSsMlpWeights;
using flm::corelib::UniqueStream;
using flm::corelib::UniqueTensor;
using flm::corelib::UniqueTensorWindow;

void SetCorelibPath(const char* value) {
#ifdef _WIN32
    _putenv_s("FLM_CORE_CORELIB_PATH", value ? value : "");
#else
    if (value) setenv("FLM_CORE_CORELIB_PATH", value, 1);
    else unsetenv("FLM_CORE_CORELIB_PATH");
#endif
}

std::shared_ptr<CorelibApi> ValidApi() {
    return CorelibApi::ResolveForTest(fake_corelib::Resolver());
}

void TestVersionIsResolvedBeforeEveryOtherSymbol() {
    fake_corelib::Reset();
    ValidApi();
    const auto& order = fake_corelib::GetState().resolution_order;
    TEST_REQUIRE(order.size() == kSymbolCount);
    TEST_REQUIRE(order.front() == "ryzenai_corelib_get_version");
}

void TestExactlyThePinnedVersionIsAccepted() {
    fake_corelib::Reset();
    const auto api = ValidApi();
    const auto version = api->runtime_version();
    TEST_REQUIRE(version.major == RYZENAI_CORELIB_VERSION_MAJOR);
    TEST_REQUIRE(version.minor == RYZENAI_CORELIB_VERSION_MINOR);
    TEST_REQUIRE(version.patch == RYZENAI_CORELIB_VERSION_PATCH);
}

void TestMajorMinorAndPatchMismatchesAreRejectedWithBothVersions() {
    for (const auto version : {flm::corelib::CorelibVersion{1, 3, 0},
                               flm::corelib::CorelibVersion{0, 4, 0},
                               flm::corelib::CorelibVersion{0, 3, 1}}) {
        fake_corelib::Reset();
        fake_corelib::GetState().version = version;
        const std::string error = RequireThrows([&] { ValidApi(); });
        RequireContains(error, std::to_string(RYZENAI_CORELIB_VERSION_MAJOR) + "." +
                                   std::to_string(RYZENAI_CORELIB_VERSION_MINOR) + "." +
                                   std::to_string(RYZENAI_CORELIB_VERSION_PATCH));
        RequireContains(error, std::to_string(version.major) + "." +
                                   std::to_string(version.minor) + "." +
                                   std::to_string(version.patch));
        TEST_REQUIRE(fake_corelib::GetState().resolution_order.size() == 1);
    }
}

void TestEveryRequiredSymbolIsResolvedExactlyOnce() {
    fake_corelib::Reset();
    ValidApi();
    TEST_REQUIRE(fake_corelib::GetState().resolution_counts.size() == kSymbolCount);
    for (const auto& [name, count] : fake_corelib::GetState().resolution_counts) {
        (void)name;
        TEST_REQUIRE(count == 1);
    }
}

void TestEveryResolvedFakeFunctionUsesItsExactAbi() {
    fake_corelib::Reset();
    const auto api = ValidApi();
    fake_corelib::GetState().call_counts.clear();
    fake_corelib::GetState().default_status = ryzenai_corelib_status_bad_argument;
    fake_corelib::GetState().selftest_status = ryzenai_corelib_status_bad_argument;
    const auto statuses = fake_corelib::CallEveryResolvedFunction(api->functions());
    TEST_REQUIRE(statuses.size() == kStatusSymbolCount);
    TEST_REQUIRE(std::all_of(statuses.begin(), statuses.end(), [](auto status) {
        return status == ryzenai_corelib_status_bad_argument;
    }));
    TEST_REQUIRE(fake_corelib::GetState().call_counts.size() == kSymbolCount);
    for (const auto& [name, count] : fake_corelib::GetState().call_counts) {
        (void)name;
        TEST_REQUIRE(count == 1);
    }

    fake_corelib::GetState().statuses["ryzenai_corelib_tensor_write"] =
        ryzenai_corelib_status_unsupported;
    TEST_REQUIRE(api->functions().tensor_write(
                     nullptr, ryzenai_corelib_data_type_bf16, nullptr, 0, 0) ==
                 ryzenai_corelib_status_unsupported);
    TEST_REQUIRE(fake_corelib::GetState()
                     .call_counts["ryzenai_corelib_tensor_write"] == 2);
}

void TestUnusedRmsNormSymbolsAreNotRequired() {
    for (const auto* symbol : {
             "ryzenai_corelib_rmsnorm_bf16_weights_create_scale",
             "ryzenai_corelib_rmsnorm_bf16_pad_rows"}) {
        fake_corelib::Reset();
        fake_corelib::GetState().missing_symbol = symbol;
        (void)ValidApi();
        TEST_REQUIRE(!fake_corelib::GetState().resolution_counts.contains(symbol));
    }
}

void TestMissingSymbolNamesTheSymbolAndUnloadsTheDll() {
    fake_corelib::Reset();
    fake_corelib::GetState().missing_symbol = "ryzenai_corelib_create_stream";
    std::weak_ptr<int> unloaded;
    std::string error;
    {
        auto module_lifetime = std::make_shared<int>(1);
        unloaded = module_lifetime;
        auto base = fake_corelib::Resolver();
        CorelibApi::Resolver resolver =
            [module_lifetime, base](std::string_view name) { return base(name); };
        module_lifetime.reset();
        error = RequireThrows([&] { CorelibApi::ResolveForTest(std::move(resolver)); });
    }
    RequireContains(error, "ryzenai_corelib_create_stream");
    TEST_REQUIRE(unloaded.expired());
}

void TestCorelibErrorCopiesStatusCallAndThreadLocalDetail() {
    fake_corelib::Reset();
    fake_corelib::GetState().detail = "invalid tensor row count";
    fake_corelib::GetState().status_text = "bad argument";
    const auto api = ValidApi();
    try {
        api->Check(ryzenai_corelib_status_bad_argument, "tensor_write");
        TEST_REQUIRE(false);
    } catch (const CorelibError& error) {
        TEST_REQUIRE(error.status() == ryzenai_corelib_status_bad_argument);
        TEST_REQUIRE(error.call() == "tensor_write");
        TEST_REQUIRE(error.detail() == "invalid tensor row count");
        RequireContains(error.what(), "bad argument");
    }
}

void TestEnvironmentPathMustBeAnAbsoluteDllPath() {
    SetCorelibPath("relative/ryzenai_corelib.dll");
    RequireContains(RequireThrows([] {
                        CorelibApi::ResolveLibraryPath("C:/apps/flm");
                    }),
                    "absolute");
    SetCorelibPath("C:/apps/flm/core");
    RequireContains(RequireThrows([] {
                        CorelibApi::ResolveLibraryPath("C:/apps/flm");
                    }),
                    ".dll");
    SetCorelibPath(nullptr);
}

void TestEnvironmentPathWinsOverExecutableRelativePath() {
    SetCorelibPath("C:/corelib/custom.dll");
    TEST_REQUIRE(CorelibApi::ResolveLibraryPath("C:/apps/flm") ==
                 std::filesystem::path("C:/corelib/custom.dll"));
    SetCorelibPath(nullptr);
}

void TestFallbackIsExeDirectoryCoreDllNotCurrentDirectory() {
    SetCorelibPath(nullptr);
    const auto expected = std::filesystem::absolute(
        std::filesystem::path("C:/apps/flm") / "core" / "ryzenai_corelib.dll");
    TEST_REQUIRE(CorelibApi::ResolveLibraryPath("C:/apps/flm") == expected);
}

void TestEveryUniqueObjectReleasesExactlyOnceAfterMoves() {
    fake_corelib::Reset();
    const auto api = ValidApi();
    {
        UniqueTensor first(api, fake_corelib::MakeObject());
        UniqueTensor moved(std::move(first));
        UniqueTensor assigned;
        assigned = std::move(moved);
        UniqueStream stream(api, fake_corelib::MakeObject());
        UniqueTensorWindow window(api, fake_corelib::MakeObject());
        UniqueHostView host_view(api, fake_corelib::MakeObject());
        UniqueMatMulWeights matmul(api, fake_corelib::MakeObject());
        UniqueSsMlpWeights ssmlp(api, fake_corelib::MakeObject());
        TEST_REQUIRE(!first && !moved && assigned);
        TEST_REQUIRE(api->live_object_count() == 6);
        TEST_REQUIRE(fake_corelib::GetState().releases == 0);
    }
    TEST_REQUIRE(fake_corelib::GetState().releases == 6);
    TEST_REQUIRE(api->live_object_count() == 0);
}

void TestRuntimeRunsDependencySelftestAndRequiresAUsableNpu() {
    fake_corelib::Reset();
    fake_corelib::GetState().selftest_status = ryzenai_corelib_status_failure;
    RequireContains(RequireThrows([] {
                        CorelibRuntime::CreateForTest(ValidApi());
                    }),
                    "selftest_dependencies");

    fake_corelib::Reset();
    fake_corelib::GetState().npu_usable = false;
    RequireContains(RequireThrows([] {
                        CorelibRuntime::CreateForTest(ValidApi());
                    }),
                    "no usable NPU");

    fake_corelib::Reset();
    const auto runtime = CorelibRuntime::CreateForTest(ValidApi());
    TEST_REQUIRE(runtime->api() != nullptr);
    CorelibRuntime::ShutdownProcess();
}

void TestNpuArchIsCorelibsAnswerAndEmptyWhenItCannotLoad() {
    fake_corelib::Reset();
    {
        const auto runtime = CorelibRuntime::CreateForTest(ValidApi());
        fake_corelib::GetState().npu_arch = "AIE2P";
        TEST_REQUIRE(CorelibRuntime::NpuArch({}) == "AIE2P");
        fake_corelib::GetState().npu_arch = "AIEM";
        TEST_REQUIRE(CorelibRuntime::NpuArch({}) == "AIEM");
    }
    CorelibRuntime::ShutdownProcess();

    const auto empty = std::filesystem::temp_directory_path() / "flm-no-corelib-here";
    std::filesystem::create_directories(empty);
    TEST_REQUIRE(CorelibRuntime::NpuArch(empty).empty());
    std::filesystem::remove_all(empty);
}

void TestExecutionLeaseSerializesTwoThreads() {
    fake_corelib::Reset();
    const auto runtime = CorelibRuntime::CreateForTest(ValidApi());
    std::atomic<int> ready{0};
    auto worker = [&] {
        ++ready;
        while (ready.load() != 2) std::this_thread::yield();
        auto lease = runtime->AcquireExecution();
        fake_corelib::EnterLease();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        fake_corelib::LeaveLease();
    };
    std::thread first(worker);
    std::thread second(worker);
    first.join();
    second.join();
    TEST_REQUIRE(fake_corelib::GetState().maximum_active_leases == 1);
    CorelibRuntime::ShutdownProcess();
}

void TestShutdownReleasesExecutionLockBeforeDestroyingRuntimeOwner() {
    fake_corelib::Reset();
    bool destroyed = false;
    bool destroyed_while_locked = false;
    CorelibRuntime::SetDestructionObserverForTest([&](bool execution_lock_held) {
        destroyed = true;
        destroyed_while_locked = execution_lock_held;
    });
    auto runtime = CorelibRuntime::CreateForTest(ValidApi());
    runtime.reset();
    CorelibRuntime::ShutdownProcess();
    CorelibRuntime::SetDestructionObserverForTest({});
    TEST_REQUIRE(destroyed);
    TEST_REQUIRE(!destroyed_while_locked);
}

void TestCleanupRunsAfterTheLastObjectAndOnlyOnce() {
    fake_corelib::Reset();
    const auto runtime = CorelibRuntime::CreateForTest(ValidApi());
    auto object = std::make_unique<UniqueTensor>(runtime->api(),
                                                 fake_corelib::MakeObject());
    RequireContains(RequireThrows([] { CorelibRuntime::ShutdownProcess(); }),
                    "live corelib object");
    TEST_REQUIRE(fake_corelib::GetState().cleanup_calls == 0);
    object.reset();

    std::atomic<bool> lease_acquired{false};
    std::thread holder([&] {
        auto lease = runtime->AcquireExecution();
        fake_corelib::EnterLease();
        lease_acquired = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        fake_corelib::LeaveLease();
    });
    while (!lease_acquired.load()) std::this_thread::yield();
    CorelibRuntime::ShutdownProcess();
    holder.join();
    CorelibRuntime::ShutdownProcess();
    TEST_REQUIRE(fake_corelib::GetState().cleanup_calls == 1);
    TEST_REQUIRE(fake_corelib::GetState().releases == 1);
    TEST_REQUIRE(fake_corelib::GetState().lifetime_events ==
                 std::vector<std::string>({"release", "lease_leave", "cleanup"}));
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestVersionIsResolvedBeforeEveryOtherSymbol);
    RUN_TEST(TestExactlyThePinnedVersionIsAccepted);
    RUN_TEST(TestMajorMinorAndPatchMismatchesAreRejectedWithBothVersions);
    RUN_TEST(TestEveryRequiredSymbolIsResolvedExactlyOnce);
    RUN_TEST(TestEveryResolvedFakeFunctionUsesItsExactAbi);
    RUN_TEST(TestUnusedRmsNormSymbolsAreNotRequired);
    RUN_TEST(TestMissingSymbolNamesTheSymbolAndUnloadsTheDll);
    RUN_TEST(TestCorelibErrorCopiesStatusCallAndThreadLocalDetail);
    RUN_TEST(TestEnvironmentPathMustBeAnAbsoluteDllPath);
    RUN_TEST(TestEnvironmentPathWinsOverExecutableRelativePath);
    RUN_TEST(TestFallbackIsExeDirectoryCoreDllNotCurrentDirectory);
    RUN_TEST(TestEveryUniqueObjectReleasesExactlyOnceAfterMoves);
    RUN_TEST(TestRuntimeRunsDependencySelftestAndRequiresAUsableNpu);
    RUN_TEST(TestNpuArchIsCorelibsAnswerAndEmptyWhenItCannotLoad);
    RUN_TEST(TestExecutionLeaseSerializesTwoThreads);
    RUN_TEST(TestShutdownReleasesExecutionLockBeforeDestroyingRuntimeOwner);
    RUN_TEST(TestCleanupRunsAfterTheLastObjectAndOnlyOnce);
#undef RUN_TEST
    return 0;
}
