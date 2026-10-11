/// \file test_qwen3_rai.cpp
/// \brief Qwen3 on rai, hardware-free: size selection, the GGUF contract, and
///        the engine's dispatch sequence against the fake corelib
#include "models/qwen3/rai/aie_next/qwen3_rai.hpp"
#include "models/qwen3/rai/aie_next/qwen3_rai_config.hpp"
#include "models/qwen3/rai/aie_next/qwen3_rai_gguf.hpp"
#include "../phi4_rai/fake_corelib.hpp"
#include "../phi4_rai/gguf_fixture.hpp"
#include "../phi4_rai/test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::qwen3::kQwen3Rows;
using flm::qwen3::Qwen3Config;
using flm::qwen3::Qwen3GgufPackage;
using flm::qwen3::qwen3_rai;
using gguf_fixture::kF32;
using gguf_fixture::kQ8_0;

const Qwen3Config& Row(std::string_view size) {
    for (const auto& row : kQwen3Rows)
        if (row.size == size) return row;
    throw std::runtime_error("no row " + std::string(size));
}

/// \brief a Qwen3 GGUF of one row's exact shape, payload all zeros
gguf_fixture::Builder Qwen3Builder(const Qwen3Config& c, bool untied = false,
                                   bool with_layers = true) {
    gguf_fixture::Builder builder;
    for (const auto* key : {"general.architecture", "phi3.block_count", "phi3.context_length",
                            "phi3.embedding_length", "phi3.feed_forward_length",
                            "phi3.attention.head_count", "phi3.attention.head_count_kv",
                            "phi3.attention.layer_norm_rms_epsilon", "phi3.rope.dimension_count",
                            "phi3.rope.freq_base", "phi3.rope.scaling.attn_factor",
                            "phi3.rope.scaling.original_context_length", "tokenizer.ggml.tokens",
                            "tokenizer.ggml.add_bos_token", "tokenizer.ggml.eos_token_id"})
        builder.RemoveMetadata(key);
    builder.AddMetadata("general.architecture", std::string("qwen3"))
        .AddMetadata("qwen3.block_count", static_cast<std::uint32_t>(c.layers))
        .AddMetadata("qwen3.context_length", std::uint32_t{40960})
        .AddMetadata("qwen3.embedding_length", static_cast<std::uint32_t>(c.hidden))
        .AddMetadata("qwen3.feed_forward_length", static_cast<std::uint32_t>(c.intermediate))
        .AddMetadata("qwen3.attention.head_count", static_cast<std::uint32_t>(c.q_heads))
        .AddMetadata("qwen3.attention.head_count_kv", static_cast<std::uint32_t>(c.kv_heads))
        .AddMetadata("qwen3.attention.key_length", static_cast<std::uint32_t>(c.head_dim))
        .AddMetadata("qwen3.attention.value_length", static_cast<std::uint32_t>(c.head_dim))
        .AddMetadata("qwen3.attention.layer_norm_rms_epsilon", 1.0e-6f)
        .AddMetadata("qwen3.rope.freq_base", 1.0e6f)
        .AddMetadata("tokenizer.ggml.eos_token_id", std::uint32_t{151645});
    const auto h = static_cast<std::uint64_t>(c.hidden);
    const auto q = static_cast<std::uint64_t>(c.q_dim());
    const auto kv = static_cast<std::uint64_t>(c.kv_dim());
    const auto inter = static_cast<std::uint64_t>(c.intermediate);
    builder.AddTensor("token_embd.weight", {static_cast<std::uint64_t>(c.vocab), h}, kQ8_0);
    if (untied) builder.AddTensor("output.weight", {static_cast<std::uint64_t>(c.vocab), h}, kQ8_0);
    builder.AddTensor("output_norm.weight", {h}, kF32);
    if (!with_layers) return builder;
    for (std::int64_t layer = 0; layer < c.layers; ++layer) {
        const auto p = "blk." + std::to_string(layer);
        builder.AddTensor(p + ".attn_norm.weight", {h}, kF32)
            .AddTensor(p + ".ffn_norm.weight", {h}, kF32)
            .AddTensor(p + ".attn_q_norm.weight", {128}, kF32)
            .AddTensor(p + ".attn_k_norm.weight", {128}, kF32)
            .AddTensor(p + ".attn_q.weight", {q, h}, kQ8_0)
            .AddTensor(p + ".attn_k.weight", {kv, h}, kQ8_0)
            .AddTensor(p + ".attn_v.weight", {kv, h}, kQ8_0)
            .AddTensor(p + ".attn_output.weight", {h, q}, kQ8_0)
            .AddTensor(p + ".ffn_gate.weight", {inter, h}, kQ8_0)
            .AddTensor(p + ".ffn_up.weight", {inter, h}, kQ8_0)
            .AddTensor(p + ".ffn_down.weight", {h, inter}, kQ8_0);
    }
    return builder;
}

const std::filesystem::path& TiedPath() {
    static auto file = Qwen3Builder(Row("0.6B")).Write("qwen3_tied");
    return file.path;
}

nlohmann::json Config(const Qwen3Config& c, std::optional<bool> tied = true) {
    nlohmann::json config = {{"model_type", "qwen3"}, {"num_hidden_layers", c.layers},
                             {"hidden_size", c.hidden}, {"intermediate_size", c.intermediate},
                             {"num_attention_heads", c.q_heads}, {"num_key_value_heads", c.kv_heads},
                             {"head_dim", c.head_dim}, {"vocab_size", c.vocab},
                             {"rms_norm_eps", 1e-06}, {"rope_theta", 1000000}};
    if (tied) config["tie_word_embeddings"] = *tied;
    return config;
}

nlohmann::json Tokenizer() {
    return {{"added_tokens", nlohmann::json::array({
                {{"id", 151643}, {"content", "<|endoftext|>"}},
                {{"id", 151644}, {"content", "<|im_start|>"}},
                {{"id", 151645}, {"content", "<|im_end|>"}},
                {{"id", 151667}, {"content", "<think>"}},
                {{"id", 151668}, {"content", "</think>"}}})}};
}

nlohmann::json TokenizerConfig() {
    return {{"eos_token", "<|im_end|>"},
            {"chat_template", "<|im_start|>{{ m }}<|im_end|>"}};
}

// ---------------------------------------------------------------- config

void TestEveryRowSelectsItself() {
    for (const auto& row : kQwen3Rows) {
        const flm::qwen3::Qwen3Shape shape{row.layers, row.hidden, row.q_heads, row.kv_heads,
                                           row.head_dim, row.intermediate, 1.0e6,
                                           static_cast<double>(1.0e-6f)};
        TEST_REQUIRE(&flm::qwen3::SelectQwen3Row(shape) == &row);
    }
    TEST_REQUIRE(Row("0.6B").head_group == 32);
    TEST_REQUIRE(Row("1.7B").head_group == 64);
    TEST_REQUIRE(Row("0.6B").q_dim() == 2048);
    TEST_REQUIRE(Row("4B").q_dim() == 4096);
}

void TestQwen3_4B_2507IsRefusedOnItsRopeTheta() {
    const auto& row = Row("4B");
    const flm::qwen3::Qwen3Shape shape{row.layers, row.hidden, row.q_heads, row.kv_heads,
                                       row.head_dim, row.intermediate, 5.0e6, 1.0e-6};
    RequireContains(RequireThrows([&] { flm::qwen3::SelectQwen3Row(shape); }), "rope theta");
}

void TestUnknownShapeIsRefused() {
    const flm::qwen3::Qwen3Shape shape{24, 896, 14, 2, 64, 4864, 1.0e6, 1.0e-6};
    RequireContains(RequireThrows([&] { flm::qwen3::SelectQwen3Row(shape); }),
                    "no supported Qwen3 size");
}

// ---------------------------------------------------------------- GGUF

void TestTiedPackageResolvesToTokenEmbedding() {
    const auto package = Qwen3GgufPackage::Open(TiedPath());
    TEST_REQUIRE(package->Config().size == "0.6B");
    TEST_REQUIRE(package->TiedLmHead());
    TEST_REQUIRE(package->LmHead().name == "token_embd.weight");
    TEST_REQUIRE(package->Layer(27).q.shape == std::vector<std::int64_t>({2048, 1024}));
    TEST_REQUIRE(package->Layer(27).o.shape == std::vector<std::int64_t>({1024, 2048}));
    package->ValidateContract(Config(Row("0.6B")), Tokenizer(), TokenizerConfig());
    // A config that does not state tying is checked on everything else.
    package->ValidateContract(Config(Row("0.6B"), std::nullopt), Tokenizer(), TokenizerConfig());
}

void TestUntiedPackageReadsOutputWeight() {
    const auto file = Qwen3Builder(Row("0.6B"), true).Write("qwen3_untied");
    const auto package = Qwen3GgufPackage::Open(file.path);
    TEST_REQUIRE(!package->TiedLmHead());
    TEST_REQUIRE(package->LmHead().name == "output.weight");
    RequireContains(RequireThrows([&] {
                        package->ValidateContract(Config(Row("0.6B"), true), Tokenizer(),
                                                  TokenizerConfig());
                    }),
                    "tie_word_embeddings");
}

void TestContractMismatchesAreNamed() {
    const auto package = Qwen3GgufPackage::Open(TiedPath());
    auto config = Config(Row("0.6B"));
    config["vocab_size"] = 152064;
    RequireContains(RequireThrows([&] { package->ValidateContract(config, Tokenizer(), TokenizerConfig()); }),
                    "vocab_size");
    auto tokenizer = Tokenizer();
    tokenizer["added_tokens"][2]["id"] = 7;
    RequireContains(RequireThrows([&] { package->ValidateContract(Config(Row("0.6B")), tokenizer, TokenizerConfig()); }),
                    "<|im_end|>");
    auto tokenizer_config = TokenizerConfig();
    tokenizer_config["eos_token"] = "<|endoftext|>";
    RequireContains(RequireThrows([&] { package->ValidateContract(Config(Row("0.6B")), Tokenizer(), tokenizer_config); }),
                    "eos_token");
}

void TestWrongArchitectureAndMissingTensorsAreRefused() {
    auto qwen2 = Qwen3Builder(Row("0.6B"), false, false);
    qwen2.SetMetadata("general.architecture", std::string("qwen2"));
    const auto qwen2_file = qwen2.Write("qwen3_arch");
    RequireContains(RequireThrows([&] { Qwen3GgufPackage::Open(qwen2_file.path); }),
                    "general.architecture");

    const auto no_layers = Qwen3Builder(Row("0.6B"), false, false).Write("qwen3_nolayers");
    RequireContains(RequireThrows([&] { Qwen3GgufPackage::Open(no_layers.path); }),
                    "blk.0.attn_norm.weight");

    auto no_key_length = Qwen3Builder(Row("0.6B"), false, false);
    no_key_length.RemoveMetadata("qwen3.attention.key_length");
    const auto no_key_file = no_key_length.Write("qwen3_nokey");
    RequireContains(RequireThrows([&] { Qwen3GgufPackage::Open(no_key_file.path); }),
                    "qwen3.attention.key_length");
}

// ---------------------------------------------------------------- engine

struct ScopedEnv {
    std::string name, previous;
    bool had{};
    ScopedEnv(std::string key, const std::string& value) : name(std::move(key)) {
        if (const char* existing = std::getenv(name.c_str())) { previous = existing; had = true; }
        _putenv_s(name.c_str(), value.c_str());
    }
    ~ScopedEnv() { _putenv_s(name.c_str(), had ? previous.c_str() : ""); }
};

struct Harness {
    ScopedEnv cache;
    std::shared_ptr<CorelibRuntime> runtime;
    std::unique_ptr<qwen3_rai> engine;

    explicit Harness(const std::string& cache_setting = "0") : cache("FLM_RAI_WEIGHT_CACHE", cache_setting) {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        engine = std::make_unique<qwen3_rai>(LM_Config{}, Qwen3GgufPackage::Open(TiedPath()), runtime);
    }
    ~Harness() {
        engine.reset();
        runtime.reset();
        CorelibRuntime::ShutdownProcess();
    }
};

std::vector<std::string> Kinds(std::size_t from = 0) {
    std::vector<std::string> kinds;
    const auto& dispatches = fake_corelib::GetState().dispatches;
    for (std::size_t i = from; i < dispatches.size(); ++i) kinds.push_back(dispatches[i].kind);
    return kinds;
}

void TestStreamEnablesTheQwen3BundleBeforeAnyKernelQuery() {
    Harness h;
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.kernel_bundles == std::vector<std::string>({"qwen3"}));
    const auto& log = state.call_log;
    const auto enabled = std::find(log.begin(), log.end(), "ryzenai_corelib_stream_enable_kernel_bundle");
    const auto queried = std::find(log.begin(), log.end(), "ryzenai_corelib_matmul_bf16_enum_kernels");
    TEST_REQUIRE(queried != log.end());
    TEST_REQUIRE(enabled < queried);
}

void TestLoadPacksEveryWeightAtItsGroup() {
    Harness h;
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.stream_prefill_pdi == 1 && state.stream_token_pdi == 16);
    std::size_t matmuls = 0, ssmlps = 0, norms = 0;
    for (const auto& create : state.weight_creates) {
        if (create.kind == "matmul") {
            ++matmuls;
            const bool head = create.n == 151936;
            TEST_REQUIRE(create.group_size == (head ? 32u : 64u));
        } else if (create.kind == "ssmlp") {
            ++ssmlps;
            TEST_REQUIRE(create.group_size == 64u && create.k == 1024 && create.n == 3072);
        } else if (create.kind == "rmsnorm") {
            ++norms;
            TEST_REQUIRE(create.k == 128 && create.norm0.size() == 128);
        }
    }
    TEST_REQUIRE(matmuls == 28 * 4 + 1);
    TEST_REQUIRE(ssmlps == 28);
    TEST_REQUIRE(norms == 28 * 2);
    TEST_REQUIRE(state.maximum_active_weight_creates == 1);
    const auto& views = state.host_view_creates;
    TEST_REQUIRE(views.size() == 2);
    for (const auto& view : views) TEST_REQUIRE(view.shape == std::vector<std::int64_t>({4096, 64}));
}

void TestPrefillRunsEightDispatchesALayerAtTheBucket() {
    Harness h;
    std::vector<int> ids(70, 5);
    (void)h.engine->prefill(ids);
    const auto kinds = Kinds();
    TEST_REQUIRE(kinds.size() == 28 * 8 + 1);
    const std::vector<std::string> layer{"matmul", "matmul", "matmul", "rmsnorm",
                                         "rmsnorm", "mha", "matmul", "ssmlp"};
    TEST_REQUIRE(std::equal(layer.begin(), layer.end(), kinds.begin()));
    const auto& dispatches = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(dispatches[0].rows == 128);
    // QK-Norm runs per head: 128 rows x 16 heads, and 128 x 8 for K.
    TEST_REQUIRE(dispatches[3].rows == 128 * 16);
    TEST_REQUIRE(dispatches[4].rows == 128 * 8);
    TEST_REQUIRE(dispatches[5].position == 0);
    TEST_REQUIRE(dispatches.back().rows == 1);
    TEST_REQUIRE(h.engine->get_current_context_length() == 70);
}

void TestDecodeRunsOneRowAtThePosition() {
    Harness h;
    std::vector<int> ids(3, 9);
    (void)h.engine->prefill(ids);
    const auto before = fake_corelib::GetState().dispatches.size();
    (void)h.engine->forward(11);
    const auto& dispatches = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(dispatches[before].rows == 1);
    TEST_REQUIRE(dispatches[before + 3].rows == 16);
    TEST_REQUIRE(dispatches[before + 5].position == 3);
    // v_proj writes the V cache at row 3 of every head.
    TEST_REQUIRE(dispatches[before + 2].window_offset == 3 * 128);
    TEST_REQUIRE(h.engine->get_current_context_length() == 4);
}

void TestTheVProjectionWritesTheCacheThroughAStridedWindow() {
    Harness h;
    (void)h.engine->forward(1);
    const auto& windows = fake_corelib::GetState().tensor_windows;
    const auto v = std::find_if(windows.begin(), windows.end(),
                                [](const auto& w) { return w.shape.size() == 3; });
    TEST_REQUIRE(v != windows.end());
    TEST_REQUIRE(v->strides.size() == 3 && v->strides[2] == 1);
    TEST_REQUIRE(v->strides[1] == v->shape[2]);
    TEST_REQUIRE(v->strides[0] == 4096 * v->strides[1]);
}

void TestContinuingPrefillReplaysTheConversationFromZero() {
    Harness h;
    std::vector<int> first(10, 3);
    (void)h.engine->prefill(first);
    (void)h.engine->forward(4);
    std::vector<int> next(60, 6);
    const auto before = fake_corelib::GetState().dispatches.size();
    (void)h.engine->prefill(next);
    const auto& dispatches = fake_corelib::GetState().dispatches;
    // 71 tokens from position 0: the 128 bucket, attention at position 0.
    TEST_REQUIRE(dispatches[before].rows == 128);
    TEST_REQUIRE(dispatches[before + 5].position == 0);
    TEST_REQUIRE(h.engine->get_current_context_length() == 71);
}

void TestRestoreRewindsTheReplayHistory() {
    Harness h;
    std::vector<int> prompt(5, 3);
    (void)h.engine->prefill(prompt);
    TEST_REQUIRE(h.engine->checkpoint() == 5);
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    TEST_REQUIRE(h.engine->restore() == 5);
    std::vector<int> next(2, 8);
    const auto before = fake_corelib::GetState().dispatches.size();
    (void)h.engine->prefill(next);
    TEST_REQUIRE(fake_corelib::GetState().dispatches[before].rows == 64);
    TEST_REQUIRE(h.engine->get_current_context_length() == 7);
}

void TestUnknownHistoryCannotBeContinued() {
    Harness h;
    h.engine->set_context_length(12);
    std::vector<int> ids(2, 1);
    RequireContains(RequireThrows([&] { (void)h.engine->prefill(ids); }), "without the tokens");
    TEST_REQUIRE(!h.engine->poisoned());
    TEST_REQUIRE(h.engine->get_current_context_length() == 12);
}

void TestDecodeStopsAtTheWindow() {
    Harness h;
    h.engine->clear_context();
    h.engine->set_context_length(4095);
    RequireContains(RequireThrows([&] { (void)h.engine->forward(1); }), "4095");
    TEST_REQUIRE(!h.engine->poisoned());
}

void TestFailureAfterSubmitPoisons() {
    Harness h;
    fake_corelib::GetState().fail_after_submit = "ryzenai_corelib_rmsnorm_bf16";
    std::vector<int> ids(4, 2);
    (void)RequireThrows([&] { (void)h.engine->prefill(ids); });
    TEST_REQUIRE(h.engine->poisoned());
    RequireContains(RequireThrows([&] { (void)h.engine->forward(1); }), "poisoned");
}

void TestWeightCacheRoundTrip() {
    const auto directory = std::filesystem::temp_directory_path() / "flm_qwen3_rai_cache";
    std::filesystem::remove_all(directory);
    {
        Harness packed(directory.string());
        TEST_REQUIRE(fake_corelib::GetState().weight_from_file.empty());
    }
    TEST_REQUIRE(std::filesystem::exists(directory / (TiedPath().stem().string() + ".rai-weights.json")));
    {
        Harness cached(directory.string());
        const auto& state = fake_corelib::GetState();
        TEST_REQUIRE(state.weight_from_file.size() == 28 * 5 + 1);
        // Only the QK-Norm scales still go through a packer.
        TEST_REQUIRE(std::all_of(state.weight_creates.begin(), state.weight_creates.end(),
                                 [](const auto& create) { return create.kind == "rmsnorm"; }));
    }
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    RunTest(TestEveryRowSelectsItself, "every row selects itself");
    RunTest(TestQwen3_4B_2507IsRefusedOnItsRopeTheta, "Qwen3-4B-2507 is refused on its rope theta");
    RunTest(TestUnknownShapeIsRefused, "unknown shape is refused");
    RunTest(TestTiedPackageResolvesToTokenEmbedding, "tied package resolves to token_embd");
    RunTest(TestUntiedPackageReadsOutputWeight, "untied package reads output.weight");
    RunTest(TestContractMismatchesAreNamed, "contract mismatches are named");
    RunTest(TestWrongArchitectureAndMissingTensorsAreRefused, "wrong architecture and missing tensors are refused");
    RunTest(TestStreamEnablesTheQwen3BundleBeforeAnyKernelQuery,
            "stream enables the qwen3 bundle before any kernel query");
    RunTest(TestLoadPacksEveryWeightAtItsGroup, "load packs every weight at its group");
    RunTest(TestPrefillRunsEightDispatchesALayerAtTheBucket, "prefill runs eight dispatches a layer at the bucket");
    RunTest(TestDecodeRunsOneRowAtThePosition, "decode runs one row at the position");
    RunTest(TestTheVProjectionWritesTheCacheThroughAStridedWindow,
            "the V projection writes the cache through a strided window");
    RunTest(TestContinuingPrefillReplaysTheConversationFromZero, "continuing prefill replays from zero");
    RunTest(TestRestoreRewindsTheReplayHistory, "restore rewinds the replay history");
    RunTest(TestUnknownHistoryCannotBeContinued, "unknown history cannot be continued");
    RunTest(TestDecodeStopsAtTheWindow, "decode stops at the window");
    RunTest(TestFailureAfterSubmitPoisons, "failure after submit poisons");
    RunTest(TestWeightCacheRoundTrip, "weight cache round trip");
    std::cout << "All Qwen3 rai tests passed\n";
    return 0;
}
