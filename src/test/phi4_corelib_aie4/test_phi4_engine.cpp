#include "models/phi4/phi4_corelib_aie4.hpp"
#include "models/phi4/phi4_corelib_host.hpp"
#include "fake_corelib.hpp"
#include "gguf_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
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

    explicit Harness(std::function<void(fake_corelib::State&)> configure = {}) {
        fake_corelib::Reset();
        if (configure) configure(fake_corelib::GetState());
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

void TestEngineAllocatesMaximaAcrossAllRowsAndConsumers() {
    Harness h([](auto& state) {
        state.pad_row_overrides["matmul-3072"][2048] = 5000;
        state.pad_row_overrides["matmul-1024"][2048] = 6000;
        state.pad_row_overrides["ssmlp"][2048] = 7000;
        state.pad_row_overrides["mha"][2048] = 9000;
    });
    const auto& tensors = fake_corelib::GetState().tensor_creates;
    TEST_REQUIRE(tensors[0].shape == std::vector<std::int64_t>({7000, 3072}));
    TEST_REQUIRE(tensors[1].shape == std::vector<std::int64_t>({7000, 3072}));
    TEST_REQUIRE(tensors[2].shape == std::vector<std::int64_t>({7000, 3072}));
    TEST_REQUIRE(tensors[3].shape == std::vector<std::int64_t>({9000, 3072}));
    TEST_REQUIRE(tensors[4].shape == std::vector<std::int64_t>({9000, 1024}));
    TEST_REQUIRE(tensors[5].shape == std::vector<std::int64_t>({9000, 3072}));
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

void TestValidatedPackageFlowsDirectlyIntoAllRequantizedCreates() {
    Harness h;
    const auto& records = fake_corelib::GetState().weight_creates;
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto base = 1 + layer * 5;
        const auto qkv = h.package->AttentionQkv(layer);
        const auto gate_up = h.package->GateUp(layer);
        TEST_REQUIRE(records[base + 0].pointers ==
                     std::vector<const void*>{qkv.values[0].bytes.data()});
        TEST_REQUIRE(records[base + 1].pointers ==
                     std::vector<const void*>{qkv.values[1].bytes.data()});
        TEST_REQUIRE(records[base + 2].pointers ==
                     std::vector<const void*>{qkv.values[2].bytes.data()});
        const std::vector<const void*> output_pointer{
            h.package->RequireQ8("blk." + std::to_string(layer) +
                ".attn_output.weight", std::array<std::int64_t, 2>{3072, 3072})
                .bytes.data()};
        TEST_REQUIRE(records[base + 3].pointers == output_pointer);
        const std::vector<const void*> mlp_pointers{
            gate_up.values[0].bytes.data(), gate_up.values[1].bytes.data(),
            h.package->RequireQ8("blk." + std::to_string(layer) +
                ".ffn_down.weight", std::array<std::int64_t, 2>{3072, 8192})
                .bytes.data()};
        TEST_REQUIRE(records[base + 4].pointers == mlp_pointers);
    }
    const std::vector<const void*> embedding_pointer{
        h.package->RequireQ8("token_embd.weight",
            std::array<std::int64_t, 2>{200064, 3072}).bytes.data()};
    TEST_REQUIRE(records.back().pointers == embedding_pointer);
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
    Harness h([](auto& state) {
        state.pad_row_overrides["matmul-1024"][2] = 96;
    });
    fake_corelib::GetState().tensor_writes.clear();
    std::vector<int> ids{1, 2};
    (void)h.engine->prefill(ids);
    const auto& writes = fake_corelib::GetState().tensor_writes;
    TEST_REQUIRE(writes[0].count == 96 * 3072);
    TEST_REQUIRE(writes[1].count == 96 * 3072);
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
    fake_corelib::GetState().statuses["ryzenai_corelib_rmsnorm_bf16"] = ryzenai_corelib_status_bad_argument;
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "rmsnorm");
    TEST_REQUIRE(!h.engine->poisoned());
    fake_corelib::GetState().statuses.erase("ryzenai_corelib_rmsnorm_bf16");
    (void)h.engine->forward(0);
}

void TestInitialRmsNormPostSubmitFailureSynchronizesAndPoisons() {
    Harness h;
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_rmsnorm_bf16";
    RequireContains(RequireThrows([&] { (void)h.engine->forward(0); }), "rmsnorm");
    TEST_REQUIRE(h.engine->poisoned());
    TEST_REQUIRE(!fake_corelib::GetState().work_in_flight);
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_stream_synchronize"] == 1);
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

void TestFakeTensorWindowRetainsAndPropagatesParentStorage() {
    fake_corelib::Reset();
    auto api = CorelibApi::ResolveForTest(fake_corelib::Resolver());
    const std::array<std::int64_t, 1> parent_shape{16};
    void* parent = nullptr;
    api->Check(api->functions().create_device_tensor(
        ryzenai_corelib_data_type_bf16, parent_shape.data(), parent_shape.size(), &parent),
        "create parent");
    const std::array<std::uint16_t, 4> original{11, 22, 33, 44};
    api->Check(api->functions().tensor_write(parent, ryzenai_corelib_data_type_bf16,
                                              original.data(), original.size(), 4),
               "write parent");
    const std::array<std::int64_t, 1> window_shape{4};
    void* window = nullptr;
    api->Check(api->functions().create_tensor_window(
        parent, window_shape.data(), window_shape.size(), 4, &window), "create window");
    std::array<std::uint16_t, 4> read{};
    api->Check(api->functions().tensor_read(window, ryzenai_corelib_data_type_bf16,
                                             read.data(), read.size(), 0), "read window");
    TEST_REQUIRE(read == original);
    const std::array<std::uint16_t, 2> replacement{77, 88};
    api->Check(api->functions().tensor_write(window, ryzenai_corelib_data_type_bf16,
                                              replacement.data(), replacement.size(), 1),
               "write window");
    std::array<std::uint16_t, 4> reread{};
    api->Check(api->functions().tensor_read(parent, ryzenai_corelib_data_type_bf16,
                                             reread.data(), reread.size(), 4), "read parent");
    TEST_REQUIRE((reread == std::array<std::uint16_t, 4>{11, 77, 88, 44}));
    api->Release(parent);
    reread.fill(0);
    api->Check(api->functions().tensor_read(window, ryzenai_corelib_data_type_bf16,
                                             reread.data(), reread.size(), 0), "reread retained window");
    TEST_REQUIRE((reread == std::array<std::uint16_t, 4>{11, 77, 88, 44}));
    api->Release(window);
    TEST_REQUIRE(fake_corelib::GetState().live_objects == 0);
}

void WriteCacheRow(Harness& h, std::size_t tensor_index, int position,
                   std::uint16_t base) {
    auto& record = fake_corelib::GetState().tensor_creates[tensor_index];
    for (std::size_t head = 0; head < 8; ++head) {
        std::array<std::uint16_t, 128> values{};
        values.fill(static_cast<std::uint16_t>(base + head));
        h.runtime->api()->Check(h.runtime->api()->functions().tensor_write(
            record.object, ryzenai_corelib_data_type_bf16, values.data(), values.size(),
            (head * 4096 + position) * 128), "seed cache row");
    }
}

void TestGetKCacheGathersHeadMajorPosition() {
    Harness h;
    WriteCacheRow(h, 10, 7, 100);
    const auto result = h.engine->get_k_cache(0, 7);
    const auto* bits = reinterpret_cast<const std::uint16_t*>(result.data());
    for (std::size_t head = 0; head < 8; ++head)
        for (std::size_t i = 0; i < 128; ++i)
            TEST_REQUIRE(bits[head * 128 + i] == 100 + head);
}

void TestGetVCacheGathersHeadMajorPosition() {
    Harness h;
    WriteCacheRow(h, 11, 9, 200);
    const auto result = h.engine->get_v_cache(0, 9);
    const auto* bits = reinterpret_cast<const std::uint16_t*>(result.data());
    for (std::size_t head = 0; head < 8; ++head)
        for (std::size_t i = 0; i < 128; ++i)
            TEST_REQUIRE(bits[head * 128 + i] == 200 + head);
}

void TestCancellationBoundaryLeavesNoOutstandingFakeWork() {
    Harness h;
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_ssmlp_bf16";
    (void)RequireThrows([&] { (void)h.engine->forward(0); });
    TEST_REQUIRE(!fake_corelib::GetState().work_in_flight);
}

void TestTwoConcurrentAie4RequestsNeverOverlapDispatch() {
    Harness h;
    auto second_engine = std::make_unique<phi4_corelib_aie4>(
        LM_Config{}, h.package, h.runtime);
    fake_corelib::GetState().maximum_active_leases = 0;
    fake_corelib::GetState().statuses["test_observe_dispatch_concurrency"] =
        ryzenai_corelib_status_success;
    fake_corelib::GetState().dispatches.clear();
    std::barrier start(3);
    std::thread first([&] { start.arrive_and_wait(); (void)h.engine->forward(1); });
    std::thread second([&] { start.arrive_and_wait(); (void)second_engine->forward(2); });
    start.arrive_and_wait();
    first.join();
    second.join();

    const auto& dispatches = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(fake_corelib::GetState().maximum_active_leases == 1);
    TEST_REQUIRE(dispatches.size() == 388);
    const auto first_request = dispatches.front().thread_id;
    TEST_REQUIRE(first_request != dispatches.back().thread_id);
    TEST_REQUIRE(std::all_of(dispatches.begin(), dispatches.begin() + 194,
                             [&](const auto& call) {
                                 return call.thread_id == first_request;
                             }));
    TEST_REQUIRE(std::all_of(dispatches.begin() + 194, dispatches.end(),
                             [&](const auto& call) {
                                 return call.thread_id != first_request;
                             }));

    // Prove the fake itself does not serialize or race when the runtime lease is
    // intentionally bypassed: the overlap detector must report both calls.
    fake_corelib::GetState().dispatches.clear();
    fake_corelib::GetState().maximum_active_leases = 0;
    std::barrier unsafe_start(3);
    std::atomic<bool> unsafe_calls_succeeded{true};
    const auto invoke_without_lease = [&] {
        unsafe_start.arrive_and_wait();
        if (h.runtime->api()->functions().rmsnorm(
                nullptr, nullptr, 1, nullptr, nullptr) !=
            ryzenai_corelib_status_success)
            unsafe_calls_succeeded = false;
    };
    std::thread unsafe_first(invoke_without_lease);
    std::thread unsafe_second(invoke_without_lease);
    unsafe_start.arrive_and_wait();
    unsafe_first.join();
    unsafe_second.join();
    TEST_REQUIRE(unsafe_calls_succeeded);
    TEST_REQUIRE(fake_corelib::GetState().maximum_active_leases == 2);
    TEST_REQUIRE(fake_corelib::GetState().dispatches.size() == 2);
}

void TestTenSequentialLoadsReleaseEveryObjectAndNeverEmitAllZeroLogits() {
    for (int cycle = 0; cycle < 10; ++cycle) {
        {
            Harness h;
            const auto logits = h.engine->forward(cycle);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(logits.data());
            TEST_REQUIRE(std::any_of(bits, bits + logits.size(),
                                     [](std::uint16_t value) { return value != 0; }));
        }
        TEST_REQUIRE(fake_corelib::GetState().live_objects == 0);
    }
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestEngineCreatesOneStreamAndPersistentHelperSizedTensors);
    RUN_TEST(TestEngineAllocatesMaximaAcrossAllRowsAndConsumers);
    RUN_TEST(TestEngineCreates129Matmul32SsmlpAndOneRmsNormWeight);
    RUN_TEST(TestEveryProjectionUsesQ8RequantizedGroup64Threads0);
    RUN_TEST(TestWeightCreationIsSerialAndNeverExceedsOneInFlightCreate);
    RUN_TEST(TestQkvAndGateUpPointersMatchExactMappedSubranges);
    RUN_TEST(TestValidatedPackageFlowsDirectlyIntoAllRequantizedCreates);
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
    RUN_TEST(TestInitialRmsNormPostSubmitFailureSynchronizesAndPoisons);
    RUN_TEST(TestPostSubmitFailureSynchronizesThenPoisonsAndClearsState);
    RUN_TEST(TestSynchronizeFailurePoisonsAndClearsState);
    RUN_TEST(TestPoisonedInstanceRejectsEveryLaterEntryPoint);
    RUN_TEST(TestFakeTensorWindowRetainsAndPropagatesParentStorage);
    RUN_TEST(TestGetKCacheGathersHeadMajorPosition);
    RUN_TEST(TestGetVCacheGathersHeadMajorPosition);
    RUN_TEST(TestCancellationBoundaryLeavesNoOutstandingFakeWork);
    RUN_TEST(TestTwoConcurrentAie4RequestsNeverOverlapDispatch);
    RUN_TEST(TestTenSequentialLoadsReleaseEveryObjectAndNeverEmitAllZeroLogits);
#undef RUN_TEST
}
