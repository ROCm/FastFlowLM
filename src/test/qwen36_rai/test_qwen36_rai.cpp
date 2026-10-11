/// \file test_qwen36_rai.cpp
/// \brief Qwen3.6 on rai, hardware-free: capacity, prefill shape, rewind and
///        snapshots against the fake corelib
#include "models/qwen36/rai/aie_next/qwen36_rai.hpp"
#include "../phi4_rai/fake_corelib.hpp"
#include "../phi4_rai/test_support.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::qwen36::qwen36_rai;

struct Harness {
    std::shared_ptr<CorelibRuntime> runtime;
    std::unique_ptr<qwen36_rai> engine;

    explicit Harness(std::uint32_t maximum) {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        try {
            engine = std::make_unique<qwen36_rai>("C:/fake/qwen36.gguf", runtime, maximum);
        } catch (...) {
            Shutdown();
            throw;
        }
    }
    ~Harness() { Shutdown(); }

    void Shutdown() {
        engine.reset();
        runtime.reset();
        CorelibRuntime::ShutdownProcess();
    }
};

std::vector<int> Ids(std::size_t count) {
    std::vector<int> ids(count);
    std::iota(ids.begin(), ids.end(), 1);
    return ids;
}

void TestTheModelIsCreatedAtTheConfiguredCapacity() {
    Harness h(32768);
    TEST_REQUIRE(fake_corelib::GetState().model_max_sequence_length == 32768);
}

void TestTheModelsOwnLimitIsAccepted() {
    Harness h(262144);
    TEST_REQUIRE(fake_corelib::GetState().model_max_sequence_length == 262144);
}

void TestAMaximumPastTheModelsOwnIsRefused() {
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    (void)RequireThrows([&] { qwen36_rai engine("C:/fake/qwen36.gguf", runtime, 262145); });
    runtime.reset();
    CorelibRuntime::ShutdownProcess();
}

void TestAPromptIsOneForward() {
    Harness h(8192);
    auto ids = Ids(5000);
    (void)h.engine->prefill(ids);
    const auto& forwards = fake_corelib::GetState().model_forwards;
    TEST_REQUIRE(forwards.size() == 1);
    TEST_REQUIRE(forwards[0].rows == 5000 && forwards[0].logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5000);
}

void TestRestoreLoadsTheSnapshotWithoutReplaying() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto forwards_before = fake_corelib::GetState().model_forwards.size();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(fake_corelib::GetState().state_saves == 1);
    TEST_REQUIRE(fake_corelib::GetState().state_restores == 1);
    TEST_REQUIRE(fake_corelib::GetState().model_forwards.size() == forwards_before);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestRewindingElsewhereReplaysInOnePass() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->forward(1);
    h.engine->set_context_length(5);
    const auto& forwards = fake_corelib::GetState().model_forwards;
    TEST_REQUIRE(forwards.back().rows == 5 && !forwards.back().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5);
}

void TestRewindingOntoTheCheckpointLoadsItsSnapshot() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto forwards_before = fake_corelib::GetState().model_forwards.size();
    h.engine->set_context_length(10);
    TEST_REQUIRE(fake_corelib::GetState().state_restores == 1);
    TEST_REQUIRE(fake_corelib::GetState().model_forwards.size() == forwards_before);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestRewindingOntoADivergedCheckpointReplays() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    h.engine->set_context_length(5);
    std::vector<int> other{101, 102, 103, 104, 105};
    (void)h.engine->prefill(other);
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    auto& state = fake_corelib::GetState();
    const auto resets_before = state.call_counts["ryzenai_corelib_model_reset"];
    const auto forwards_before = state.model_forwards.size();
    h.engine->set_context_length(10);
    TEST_REQUIRE(state.state_restores == 0);
    TEST_REQUIRE(state.call_counts["ryzenai_corelib_model_reset"] == resets_before + 1);
    TEST_REQUIRE(state.model_forwards.size() == forwards_before + 1);
    TEST_REQUIRE(state.model_forwards.back().rows == 10 && !state.model_forwards.back().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestAFailedRestorePoisonsTheEngine() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    fake_corelib::GetState().statuses["ryzenai_corelib_model_state_restore"] =
        ryzenai_corelib_status_failure;
    (void)RequireThrows([&] { (void)h.engine->restore(); });
    TEST_REQUIRE(h.engine->poisoned());
    RequireContains(RequireThrows([&] { (void)h.engine->forward(2); }), "poisoned");
}

void TestAFailedForwardPoisonsTheEngineAndDropsTheCheckpoint() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    auto& state = fake_corelib::GetState();
    state.statuses["ryzenai_corelib_model_forward"] = ryzenai_corelib_status_failure;
    (void)RequireThrows([&] { (void)h.engine->forward(1); });
    TEST_REQUIRE(h.engine->poisoned());
    state.statuses.erase("ryzenai_corelib_model_forward");
    const auto forwards = state.model_forwards.size();
    RequireContains(RequireThrows([&] { (void)h.engine->forward(2); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->prefill(ids); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->restore(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->checkpoint(); }), "poisoned");
    TEST_REQUIRE(state.model_forwards.size() == forwards);
    TEST_REQUIRE(state.state_restores == 0);
}

void TestCheckpointingTheSameStateTwiceSavesOnce() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    TEST_REQUIRE(fake_corelib::GetState().state_saves == 1);
}

void TestRestoringRightAfterACheckpointLoadsNothing() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(fake_corelib::GetState().state_restores == 0);
}

void TestShrinkingBelowTheCheckpointDropsIt() {
    Harness h(16384);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    h.engine->set_context_length(2);
    h.engine->update_max_length(5);
    TEST_REQUIRE(h.engine->restore() == -1);
    TEST_REQUIRE(fake_corelib::GetState().state_restores == 0);
}

void TestGrowingTheMaximumRecreatesTheModelAndKeepsTheSequence() {
    Harness h(4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    h.engine->update_max_length(16384);
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.model_creates == 2);
    TEST_REQUIRE(state.model_max_sequence_length == 16384);
    TEST_REQUIRE(state.state_saves == 1 && state.state_restores == 1);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestShrinkingWithinTheCapacityKeepsTheModel() {
    Harness h(16384);
    h.engine->update_max_length(8192);
    TEST_REQUIRE(fake_corelib::GetState().model_creates == 1);
}

}  // namespace

int main() {
    RunTest(TestTheModelIsCreatedAtTheConfiguredCapacity, "the model is created at the configured capacity");
    RunTest(TestTheModelsOwnLimitIsAccepted, "the model's own limit is accepted");
    RunTest(TestAMaximumPastTheModelsOwnIsRefused, "a maximum past the model's own is refused");
    RunTest(TestAPromptIsOneForward, "a prompt is one forward");
    RunTest(TestRestoreLoadsTheSnapshotWithoutReplaying, "restore loads the snapshot without replaying");
    RunTest(TestRewindingElsewhereReplaysInOnePass, "rewinding elsewhere replays in one pass");
    RunTest(TestRewindingOntoTheCheckpointLoadsItsSnapshot, "rewinding onto the checkpoint loads its snapshot");
    RunTest(TestRewindingOntoADivergedCheckpointReplays, "rewinding onto a diverged checkpoint replays");
    RunTest(TestAFailedRestorePoisonsTheEngine, "a failed restore poisons the engine");
    RunTest(TestAFailedForwardPoisonsTheEngineAndDropsTheCheckpoint,
            "a failed forward poisons the engine and drops the checkpoint");
    RunTest(TestCheckpointingTheSameStateTwiceSavesOnce, "checkpointing the same state twice saves once");
    RunTest(TestRestoringRightAfterACheckpointLoadsNothing, "restoring right after a checkpoint loads nothing");
    RunTest(TestShrinkingBelowTheCheckpointDropsIt, "shrinking below the checkpoint drops it");
    RunTest(TestGrowingTheMaximumRecreatesTheModelAndKeepsTheSequence, "growing the maximum keeps the sequence");
    RunTest(TestShrinkingWithinTheCapacityKeepsTheModel, "shrinking within the capacity keeps the model");
    std::cout << "All Qwen3.6 rai tests passed\n";
    return 0;
}
