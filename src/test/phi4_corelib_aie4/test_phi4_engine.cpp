#include "models/phi4/phi4_corelib_aie4.hpp"
#include "models/phi4/phi4_corelib_host.hpp"
#include "fake_corelib.hpp"
#include "gguf_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::phi4::Phi4GgufPackage;
using flm::phi4::phi4_corelib_aie4;

const std::filesystem::path& FullPackagePath() {
    static auto file = gguf_fixture::Builder().AddFullContractTensors(false).Write("engine");
    return file.path;
}

struct Harness {
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<Phi4GgufPackage> package;
    std::unique_ptr<phi4_corelib_aie4> engine;

    Harness() {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        package = Phi4GgufPackage::Open(FullPackagePath());
        engine = std::make_unique<phi4_corelib_aie4>(LM_Config{}, package, runtime);
    }
    ~Harness() {
        engine.reset();
        package.reset();
        runtime.reset();
        CorelibRuntime::ShutdownProcess();
    }
};

void TestEngineCreatesOneStreamAndPersistentHelperSizedTensors() {
    Harness h;
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.call_counts.at("ryzenai_corelib_create_stream") == 1);
    TEST_REQUIRE(state.tensor_creates.size() == 74);
    TEST_REQUIRE(state.tensor_creates[0].shape == std::vector<std::int64_t>({4096, 3072}));
    TEST_REQUIRE(state.tensor_creates[3].shape == std::vector<std::int64_t>({4096, 3072}));
    TEST_REQUIRE(state.tensor_creates[4].shape == std::vector<std::int64_t>({4096, 1024}));
    TEST_REQUIRE(state.tensor_creates[6].shape == std::vector<std::int64_t>({1, 3072}));
    TEST_REQUIRE(state.tensor_creates[7].shape == std::vector<std::int64_t>({1, 200064}));
}

void TestEngineCreates129Matmul32SsmlpAndOneRmsNormWeight() {
    Harness h;
    const auto& records = fake_corelib::GetState().weight_creates;
    TEST_REQUIRE(records.size() == 162);
    TEST_REQUIRE(std::count_if(records.begin(), records.end(), [](const auto& r) { return r.kind == "matmul"; }) == 129);
    TEST_REQUIRE(std::count_if(records.begin(), records.end(), [](const auto& r) { return r.kind == "ssmlp"; }) == 32);
    TEST_REQUIRE(std::count_if(records.begin(), records.end(), [](const auto& r) { return r.kind == "rmsnorm"; }) == 1);
}

void TestEveryProjectionUsesQ8RequantizedGroup64Threads0() {
    Harness h;
    for (const auto& record : fake_corelib::GetState().weight_creates) {
        if (record.kind == "rmsnorm") continue;
        TEST_REQUIRE(record.group_size == 64);
        TEST_REQUIRE(record.threads == 0);
    }
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_matmul_bf16_weights_create_gguf"] == 0);
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_ssmlp_bf16_weights_create_gguf"] == 0);
}

void TestWeightCreationIsSerialAndNeverExceedsOneInFlightCreate() {
    Harness h;
    TEST_REQUIRE(fake_corelib::GetState().maximum_active_weight_creates == 1);
    const auto& records = fake_corelib::GetState().weight_creates;
    TEST_REQUIRE(records.front().kind == "rmsnorm");
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto base = 1 + layer * 5;
        TEST_REQUIRE(records[base + 0].kind == "matmul");
        TEST_REQUIRE(records[base + 1].kind == "matmul");
        TEST_REQUIRE(records[base + 2].kind == "matmul");
        TEST_REQUIRE(records[base + 3].kind == "matmul");
        TEST_REQUIRE(records[base + 4].kind == "ssmlp");
    }
    TEST_REQUIRE(records.back().kind == "matmul");
}

void TestQkvAndGateUpPointersMatchExactMappedSubranges() {
    Harness h;
    const auto qkv = h.package->AttentionQkv(0);
    const auto gate_up = h.package->GateUp(0);
    const auto& records = fake_corelib::GetState().weight_creates;
    TEST_REQUIRE(records[1].pointers[0] == qkv.values[0].bytes.data());
    TEST_REQUIRE(records[2].pointers[0] == qkv.values[1].bytes.data());
    TEST_REQUIRE(records[3].pointers[0] == qkv.values[2].bytes.data());
    TEST_REQUIRE(records[5].pointers[0] == gate_up.values[0].bytes.data());
    TEST_REQUIRE(records[5].pointers[1] == gate_up.values[1].bytes.data());
}

void TestNormsAndEpsilonReachCorelibAsBf16() {
    Harness h;
    const auto expected = flm::phi4::ConvertF32ToBf16(std::array<float, 1>{1.0e-5f})[0];
    const auto& records = fake_corelib::GetState().weight_creates;
    TEST_REQUIRE(records.front().epsilon == expected);
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto& record = records[1 + layer * 5 + 4];
        TEST_REQUIRE(record.epsilon == expected);
        TEST_REQUIRE(record.norm0.size() == 3072);
        TEST_REQUIRE(record.norm1.size() == 3072);
    }
}

void TestEmbeddingMappingOutlivesAllLazyRowReads() {
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    auto package = Phi4GgufPackage::Open(FullPackagePath());
    std::weak_ptr<Phi4GgufPackage> lifetime = package;
    auto engine = std::make_unique<phi4_corelib_aie4>(LM_Config{}, package, runtime);
    package.reset();
    TEST_REQUIRE(!lifetime.expired());
    (void)engine->forward(0);
    engine.reset();
    TEST_REQUIRE(lifetime.expired());
    runtime.reset();
    CorelibRuntime::ShutdownProcess();
}

void TestNoDeviceObjectExistsWhenPackageValidationFails() {
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    auto file = gguf_fixture::Builder().AddExactFixtureTensors().Write("invalid-engine");
    auto package = Phi4GgufPackage::Open(file.path);
    RequireContains(RequireThrows([&] {
        phi4_corelib_aie4 engine(LM_Config{}, package, runtime);
    }), "blk.1");
    TEST_REQUIRE(fake_corelib::GetState().live_objects == 0);
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_create_stream"] == 0);
    package.reset(); runtime.reset(); CorelibRuntime::ShutdownProcess();
}

void TestPrefillDecodesEmbeddingRowsAndAdvancesPosition() {
    Harness h;
    std::vector<int> ids{2, 1, 2};
    const auto logits = h.engine->prefill(ids);
    TEST_REQUIRE(logits.size() == 200064);
    TEST_REQUIRE(h.engine->get_current_context_length() == 3);
    TEST_REQUIRE(fake_corelib::GetState().tensor_writes[2].source_type == ryzenai_corelib_data_type_fp32);
}

void TestDecodeUsesOneRowAndAdvancesPosition() {
    Harness h;
    (void)h.engine->forward(4);
    TEST_REQUIRE(h.engine->get_current_context_length() == 1);
    TEST_REQUIRE(fake_corelib::GetState().dispatches.front().rows == 1);
}

void TestVProjectionWritesWindowAtPositionTimes128() {
    Harness h;
    h.engine->set_context_length(7);
    (void)h.engine->forward(1);
    const auto& windows = fake_corelib::GetState().tensor_windows;
    TEST_REQUIRE(windows.size() == 32);
    TEST_REQUIRE(windows.front().shape == std::vector<std::int64_t>({8, 4089, 128}));
    TEST_REQUIRE(windows.front().offset == 7 * 128);
    TEST_REQUIRE(fake_corelib::GetState().dispatches[3].window_offset == 7 * 128);
}

void TestEachLayerOrdersQKVThenMhaThenOThenSsmlpOnOneStream() {
    Harness h;
    (void)h.engine->forward(1);
    const auto& calls = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(calls.size() == 194);
    const void* stream = calls.front().stream;
    TEST_REQUIRE(calls.front().kind == "rmsnorm");
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const std::size_t base = 1 + layer * 6;
        TEST_REQUIRE(calls[base + 0].kind == "matmul");
        TEST_REQUIRE(calls[base + 1].kind == "matmul");
        TEST_REQUIRE(calls[base + 2].kind == "matmul");
        TEST_REQUIRE(calls[base + 3].kind == "mha");
        TEST_REQUIRE(calls[base + 4].kind == "matmul");
        TEST_REQUIRE(calls[base + 5].kind == "ssmlp");
    }
    TEST_REQUIRE(std::all_of(calls.begin(), calls.end(), [&](const auto& c) { return c.stream == stream; }));
}

void TestBuffersAreZeroPaddedBeforeSubmissionForEachRowBucket() {
    Harness h;
    fake_corelib::GetState().tensor_writes.clear();
    std::vector<int> ids{1, 2};
    (void)h.engine->prefill(ids);
    const auto& writes = fake_corelib::GetState().tensor_writes;
    TEST_REQUIRE(writes[0].count == 64 * 3072);
    TEST_REQUIRE(writes[1].count == 64 * 3072);
    TEST_REQUIRE(writes[1].all_zero);
}

void TestForwardSynchronizesBeforeHostReadAndLmHeadRead() {
    Harness h;
    fake_corelib::GetState().call_log.clear();
    (void)h.engine->forward(1);
    const auto& log = fake_corelib::GetState().call_log;
    const auto first_sync = std::find(log.begin(), log.end(), "ryzenai_corelib_stream_synchronize");
    const auto first_read = std::find(log.begin(), log.end(), "ryzenai_corelib_tensor_read");
    TEST_REQUIRE(first_sync < first_read);
    const auto lm_submit = std::find(first_read, log.end(), "ryzenai_corelib_matmul_bf16");
    const auto second_sync = std::find(lm_submit, log.end(), "ryzenai_corelib_stream_synchronize");
    const auto logits_read = std::find(second_sync, log.end(), "ryzenai_corelib_tensor_read");
    TEST_REQUIRE(lm_submit < second_sync && second_sync < logits_read);
}

void TestKVCachesRemainFixedAt8By4096By128() {
    Harness h;
    const auto& creates = fake_corelib::GetState().tensor_creates;
    const auto count = std::count_if(creates.begin(), creates.end(), [](const auto& record) {
        return record.shape == std::vector<std::int64_t>({8, 4096, 128});
    });
    TEST_REQUIRE(count == 64);
}

void TestPrompt4096IsAcceptedOnlyWithoutARequestedDecodeToken() {
    Harness h;
    std::vector<int> ids(4096, 0);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->get_current_context_length() == 4096);
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "capacity");
}

void TestTotalDecodeWindowStopsAt4095() {
    Harness h;
    h.engine->set_context_length(4094);
    (void)h.engine->forward(0);
    TEST_REQUIRE(h.engine->get_current_context_length() == 4095);
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "4095");
}

void TestClearContextResetsLogicalPositionWithoutRecreatingWeights() {
    Harness h;
    (void)h.engine->forward(0);
    const auto creates = fake_corelib::GetState().weight_creates.size();
    h.engine->clear_context();
    TEST_REQUIRE(h.engine->get_current_context_length() == 0);
    TEST_REQUIRE(fake_corelib::GetState().weight_creates.size() == creates);
}

void TestCheckpointRestoreChangesOnlyLogicalPosition() {
    Harness h;
    (void)h.engine->forward(0);
    TEST_REQUIRE(h.engine->checkpoint() == 1);
    (void)h.engine->forward(0);
    const auto creates = fake_corelib::GetState().weight_creates.size();
    TEST_REQUIRE(h.engine->restore() == 1);
    TEST_REQUIRE(h.engine->get_current_context_length() == 1);
    TEST_REQUIRE(fake_corelib::GetState().weight_creates.size() == creates);
}

void TestPreSubmitFailureIsRecoverable() {
    Harness h;
    fake_corelib::GetState().statuses["ryzenai_corelib_rmsnorm_bf16"] = ryzenai_corelib_status_failure;
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "rmsnorm");
    TEST_REQUIRE(!h.engine->poisoned());
    fake_corelib::GetState().statuses.erase("ryzenai_corelib_rmsnorm_bf16");
    (void)h.engine->forward(0);
}

void TestPostSubmitFailureSynchronizesThenPoisonsAndClearsState() {
    Harness h;
    h.engine->set_context_length(3);
    h.engine->checkpoint();
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_matmul_bf16";
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "matmul");
    TEST_REQUIRE(h.engine->poisoned());
    TEST_REQUIRE(!fake_corelib::GetState().work_in_flight);
}

void TestSynchronizeFailurePoisonsAndClearsState() {
    Harness h;
    fake_corelib::GetState().statuses["ryzenai_corelib_stream_synchronize"] = ryzenai_corelib_status_failure;
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "synchronize");
    TEST_REQUIRE(h.engine->poisoned());
    TEST_REQUIRE(!fake_corelib::GetState().work_in_flight);
}

void TestPoisonedInstanceRejectsEveryLaterEntryPoint() {
    Harness h;
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_matmul_bf16";
    (void)RequireThrows([&] { (void)h.engine->forward(0); });
    RequireContains(RequireThrows([&] { h.engine->clear_context(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->get_current_context_length(); }), "poisoned");
    std::vector<int> ids{0};
    RequireContains(RequireThrows([&] { (void)h.engine->prefill(ids); }), "poisoned");
}

void TestCancellationBoundaryLeavesNoOutstandingFakeWork() {
    Harness h;
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_ssmlp_bf16";
    (void)RequireThrows([&] { (void)h.engine->forward(0); });
    TEST_REQUIRE(!fake_corelib::GetState().work_in_flight);
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestEngineCreatesOneStreamAndPersistentHelperSizedTensors);
    RUN_TEST(TestEngineCreates129Matmul32SsmlpAndOneRmsNormWeight);
    RUN_TEST(TestEveryProjectionUsesQ8RequantizedGroup64Threads0);
    RUN_TEST(TestWeightCreationIsSerialAndNeverExceedsOneInFlightCreate);
    RUN_TEST(TestQkvAndGateUpPointersMatchExactMappedSubranges);
    RUN_TEST(TestNormsAndEpsilonReachCorelibAsBf16);
    RUN_TEST(TestEmbeddingMappingOutlivesAllLazyRowReads);
    RUN_TEST(TestNoDeviceObjectExistsWhenPackageValidationFails);
    RUN_TEST(TestPrefillDecodesEmbeddingRowsAndAdvancesPosition);
    RUN_TEST(TestDecodeUsesOneRowAndAdvancesPosition);
    RUN_TEST(TestVProjectionWritesWindowAtPositionTimes128);
    RUN_TEST(TestEachLayerOrdersQKVThenMhaThenOThenSsmlpOnOneStream);
    RUN_TEST(TestBuffersAreZeroPaddedBeforeSubmissionForEachRowBucket);
    RUN_TEST(TestForwardSynchronizesBeforeHostReadAndLmHeadRead);
    RUN_TEST(TestKVCachesRemainFixedAt8By4096By128);
    RUN_TEST(TestPrompt4096IsAcceptedOnlyWithoutARequestedDecodeToken);
    RUN_TEST(TestTotalDecodeWindowStopsAt4095);
    RUN_TEST(TestClearContextResetsLogicalPositionWithoutRecreatingWeights);
    RUN_TEST(TestCheckpointRestoreChangesOnlyLogicalPosition);
    RUN_TEST(TestPreSubmitFailureIsRecoverable);
    RUN_TEST(TestPostSubmitFailureSynchronizesThenPoisonsAndClearsState);
    RUN_TEST(TestSynchronizeFailurePoisonsAndClearsState);
    RUN_TEST(TestPoisonedInstanceRejectsEveryLaterEntryPoint);
    RUN_TEST(TestCancellationBoundaryLeavesNoOutstandingFakeWork);
#undef RUN_TEST
}
