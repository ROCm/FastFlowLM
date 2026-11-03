/// \file test_corelib_model.cpp
/// \brief core::CorelibModel against the fake corelib: capacity, where a pass
///        starts, rewind, checkpoints and poisoning, for every Rewind mode
#include "core/corelib_model.hpp"
#include "../phi4_core/fake_corelib.hpp"
#include "../phi4_core/test_support.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::core::Continuation;
using flm::core::CorelibModel;
using flm::core::ModelSpec;
using flm::core::Rewind;

const ModelSpec kSnapshot{"qwen36", "Qwen3.6", 131072, Continuation::kAnywhere, Rewind::kSnapshot};
const ModelSpec kReplay{"qwen35", "Qwen3.5", 32768, Continuation::kAnywhere, Rewind::kReplay};
const ModelSpec kFromStart{"qwen3", "Qwen3", 4096, Continuation::kFromStart, Rewind::kSeek};
const ModelSpec kFromPage{"gemma4", "Gemma 4", 16384, Continuation::kFromPage, Rewind::kSeek};

constexpr std::int64_t kVocab = 300000;
constexpr std::size_t kContinue = RYZENAI_CORELIB_POSITION_CONTINUE;

struct Harness {
    std::shared_ptr<CorelibRuntime> runtime;
    std::unique_ptr<CorelibModel> engine;

    Harness(const ModelSpec& spec, std::uint32_t maximum) {
        fake_corelib::Reset();
        fake_corelib::GetState().model_vocab = kVocab;
        runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        try {
            engine = std::make_unique<CorelibModel>(spec, "C:/fake/model.gguf", runtime, maximum);
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

fake_corelib::State& S() { return fake_corelib::GetState(); }
const fake_corelib::ModelForwardRecord& Last() { return S().model_forwards.back(); }
int Resets() { return S().call_counts["ryzenai_corelib_model_reset"]; }

// --- construction -----------------------------------------------------------

void TestTheModelIsCorelibsOwnByNameWithOneSession() {
    Harness h(kFromStart, 4096);
    TEST_REQUIRE((S().model_names == std::vector<std::string>{"qwen3"}));
    TEST_REQUIRE(S().model_path == "C:/fake/model.gguf");
    TEST_REQUIRE(S().session_creates == 1);
}

void TestTheModelIsCreatedAtTheConfiguredCapacity() {
    Harness h(kSnapshot, 32768);
    TEST_REQUIRE(S().model_max_sequence_length == 32768);
}

void TestTheSpecsLimitIsAccepted() {
    Harness h(kSnapshot, 131072);
    TEST_REQUIRE(S().model_max_sequence_length == 131072);
}

void TestAMaximumOutsideTheSpecIsRefused() {
    (void)RequireThrows([] { Harness h(kSnapshot, 131073); });
    (void)RequireThrows([] { Harness h(kSnapshot, 0); });
}

void TestAModelThatCannotSeekMustTakeAPassAnywhere() {
    const ModelSpec bad{"x", "X", 4096, Continuation::kFromStart, Rewind::kReplay};
    RequireContains(RequireThrows<std::invalid_argument>([&] { Harness h(bad, 4096); }), "seek");
}

// --- one pass ----------------------------------------------------------------

void TestAPromptIsOneForward() {
    Harness h(kSnapshot, 8192);
    auto ids = Ids(5000);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(S().model_forwards.size() == 1);
    TEST_REQUIRE(Last().rows == 5000 && Last().logits && Last().position == kContinue);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5000);
}

void TestATokenOutsideTheVocabularyIsRefusedBeforeCorelibSeesIt() {
    Harness h(kFromStart, 4096);
    std::vector<int> ids{1, static_cast<int>(kVocab)};
    (void)RequireThrows<std::out_of_range>([&] { (void)h.engine->prefill(ids); });
    (void)RequireThrows<std::out_of_range>([&] { (void)h.engine->forward(-1); });
    TEST_REQUIRE(S().model_forwards.empty());
    TEST_REQUIRE(!h.engine->poisoned());
}

void TestARequestPastTheMaximumIsRefusedBeforeCorelibSeesIt() {
    Harness h(kFromStart, 8);
    auto ids = Ids(9);
    (void)RequireThrows<std::out_of_range>([&] { (void)h.engine->prefill(ids); });
    TEST_REQUIRE(S().model_forwards.empty());
    TEST_REQUIRE(!h.engine->poisoned());
}

void TestAVisionPayloadIsRefused() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(3);
    int payload = 0;
    (void)RequireThrows<std::invalid_argument>([&] { (void)h.engine->prefill(ids, &payload); });
}

void TestTheKvCacheIsNotExposed() {
    Harness h(kFromStart, 4096);
    RequireContains(RequireThrows([&] { (void)h.engine->get_k_cache(0, 0); }), "does not expose");
    RequireContains(RequireThrows([&] { (void)h.engine->get_v_cache(0, 0); }), "does not expose");
}

// --- Rewind::kSnapshot (the Qwen3.6 behaviour, carried over) -----------------

void TestRestoreLoadsTheSnapshotWithoutReplaying() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto forwards_before = S().model_forwards.size();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(S().state_saves == 1);
    TEST_REQUIRE(S().state_restores == 1);
    TEST_REQUIRE(S().model_forwards.size() == forwards_before);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestRewindingElsewhereReplaysInOnePass() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->forward(1);
    h.engine->set_context_length(5);
    TEST_REQUIRE(Last().rows == 5 && !Last().logits && Last().position == kContinue);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5);
}

void TestRewindingOntoTheCheckpointLoadsItsSnapshot() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto forwards_before = S().model_forwards.size();
    h.engine->set_context_length(10);
    TEST_REQUIRE(S().state_restores == 1);
    TEST_REQUIRE(S().model_forwards.size() == forwards_before);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestRewindingOntoADivergedCheckpointReplays() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    h.engine->set_context_length(5);
    std::vector<int> other{101, 102, 103, 104, 105};
    (void)h.engine->prefill(other);
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto resets_before = Resets();
    const auto forwards_before = S().model_forwards.size();
    h.engine->set_context_length(10);
    TEST_REQUIRE(S().state_restores == 0);
    TEST_REQUIRE(Resets() == resets_before + 1);
    TEST_REQUIRE(S().model_forwards.size() == forwards_before + 1);
    TEST_REQUIRE(Last().rows == 10 && !Last().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestAFailedRestorePoisonsTheEngine() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    S().statuses["ryzenai_corelib_model_state_restore"] = ryzenai_corelib_status_failure;
    (void)RequireThrows([&] { (void)h.engine->restore(); });
    TEST_REQUIRE(h.engine->poisoned());
    RequireContains(RequireThrows([&] { (void)h.engine->forward(2); }), "poisoned");
}

void TestAFailedForwardPoisonsTheEngineAndDropsTheCheckpoint() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    S().statuses["ryzenai_corelib_model_forward"] = ryzenai_corelib_status_failure;
    (void)RequireThrows([&] { (void)h.engine->forward(1); });
    TEST_REQUIRE(h.engine->poisoned());
    S().statuses.erase("ryzenai_corelib_model_forward");
    const auto forwards = S().model_forwards.size();
    RequireContains(RequireThrows([&] { (void)h.engine->forward(2); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->prefill(ids); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->restore(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)h.engine->checkpoint(); }), "poisoned");
    TEST_REQUIRE(S().model_forwards.size() == forwards);
    TEST_REQUIRE(S().state_restores == 0);
}

void TestAForwardThatFailsOnceRunsAgainFromTheStart() {
    Harness h(kFromPage, 16384);
    auto ids = Ids(5000);
    (void)h.engine->prefill(ids);
    const int resets = Resets();
    S().fail_counts["ryzenai_corelib_model_forward"] = 1;
    std::vector<int> more{1, 2, 3};
    (void)h.engine->prefill(more);
    TEST_REQUIRE(!h.engine->poisoned());
    TEST_REQUIRE(Resets() == resets + 1);
    TEST_REQUIRE(Last().position == 0 && Last().rows == 5003 && Last().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5003);
    (void)h.engine->forward(4);
    TEST_REQUIRE(Last().position == 5003 && Last().rows == 1);
}

void TestARecurrentModelRunsAgainFromTheStartAfterAFailure() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    S().fail_counts["ryzenai_corelib_model_forward"] = 1;
    (void)h.engine->forward(1);
    TEST_REQUIRE(!h.engine->poisoned());
    TEST_REQUIRE(Last().rows == 11 && Last().position == RYZENAI_CORELIB_POSITION_CONTINUE);
    TEST_REQUIRE(h.engine->get_current_context_length() == 11);
}

void TestAForwardThatFailsTwicePoisonsTheEngine() {
    Harness h(kFromPage, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    S().fail_counts["ryzenai_corelib_model_forward"] = 2;
    (void)RequireThrows([&] { (void)h.engine->forward(1); });
    TEST_REQUIRE(h.engine->poisoned());
}

void TestCheckpointingTheSameStateTwiceSavesOnce() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    TEST_REQUIRE(S().state_saves == 1);
}

void TestRestoringRightAfterACheckpointLoadsNothing() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(S().state_restores == 0);
}

void TestShrinkingBelowTheCheckpointDropsIt() {
    Harness h(kSnapshot, 16384);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    h.engine->set_context_length(2);
    h.engine->update_max_length(5);
    TEST_REQUIRE(h.engine->restore() == -1);
    TEST_REQUIRE(S().state_restores == 0);
}

void TestGrowingASnapshotModelSavesAndRestoresAcrossTheNewSession() {
    Harness h(kSnapshot, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    h.engine->update_max_length(16384);
    TEST_REQUIRE(S().session_creates == 2 && S().model_names.size() == 1);
    TEST_REQUIRE(S().model_max_sequence_length == 16384);
    TEST_REQUIRE(S().state_saves == 1 && S().state_restores == 1);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

void TestShrinkingWithinTheCapacityKeepsTheSession() {
    Harness h(kSnapshot, 16384);
    h.engine->update_max_length(8192);
    TEST_REQUIRE(S().session_creates == 1);
}

// --- Rewind::kSeek -------------------------------------------------------------

void TestASeekModelStatesWhereEveryPassStarts() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(5);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(Last().rows == 5 && Last().position == 0);
    (void)h.engine->forward(9);
    TEST_REQUIRE(Last().rows == 1 && Last().position == 5);
}

void TestAFromStartContinuationRerunsTheWholeSequence() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(5);
    (void)h.engine->prefill(ids);
    std::vector<int> more{6, 7};
    (void)h.engine->prefill(more);
    TEST_REQUIRE(Last().rows == 7 && Last().position == 0 && Last().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 7);
}

void TestASeekRewindMovesOnlyThePosition() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    const auto forwards = S().model_forwards.size();
    const auto resets = Resets();
    h.engine->set_context_length(4);
    TEST_REQUIRE(S().model_forwards.size() == forwards && Resets() == resets);
    TEST_REQUIRE(h.engine->get_current_context_length() == 4);
    (void)h.engine->forward(1);
    TEST_REQUIRE(Last().rows == 1 && Last().position == 4);
}

void TestASeekCheckpointKeepsNoSnapshot() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    TEST_REQUIRE(h.engine->checkpoint() == 10);
    TEST_REQUIRE(S().state_saves == 0);
}

void TestASeekRestoreOntoAnUnchangedPrefixMovesOnlyThePosition() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    (void)h.engine->forward(2);
    const auto forwards = S().model_forwards.size();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(S().model_forwards.size() == forwards);
    (void)h.engine->forward(3);
    TEST_REQUIRE(Last().position == 10);
}

void TestASeekRestoreAfterTheCacheWasOverwrittenReplays() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    h.engine->set_context_length(5);
    (void)h.engine->forward(101);
    const auto resets = Resets();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(Resets() == resets + 1);
    TEST_REQUIRE(Last().rows == 10 && Last().position == 0 && !Last().logits);
}

void TestAnUnpagedFromPageContinuationRerunsFromZero() {
    Harness h(kFromPage, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    std::vector<int> more{1, 2};
    (void)h.engine->prefill(more);
    TEST_REQUIRE(Last().rows == 12 && Last().position == 0);
}

void TestAPagedContinuationRerunsOnlyItsPage() {
    Harness h(kFromPage, 16384);
    auto ids = Ids(5000);
    (void)h.engine->prefill(ids);
    std::vector<int> more{1, 2};
    (void)h.engine->prefill(more);
    TEST_REQUIRE(Last().position == 4096 && Last().rows == 5000 - 4096 + 2);
    TEST_REQUIRE(h.engine->get_current_context_length() == 5002);
}

void TestAPagedContinuationOnABoundaryRunsOnlyTheNewRows() {
    Harness h(kFromPage, 16384);
    auto ids = Ids(4096);
    (void)h.engine->prefill(ids);
    std::vector<int> more{1, 2};
    (void)h.engine->prefill(more);
    TEST_REQUIRE(Last().position == 4096 && Last().rows == 2);
}

void TestGrowingASeekModelReplaysIntoTheNewSession() {
    Harness h(kFromPage, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    h.engine->update_max_length(16384);
    TEST_REQUIRE(S().session_creates == 2 && S().model_names.size() == 1 &&
                 S().state_saves == 0);
    TEST_REQUIRE(Last().rows == 10 && Last().position == 0 && !Last().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 10);
}

// --- Rewind::kReplay -----------------------------------------------------------

void TestAReplayCheckpointKeepsNoSnapshot() {
    Harness h(kReplay, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    TEST_REQUIRE(S().state_saves == 0);
}

void TestAReplayRestoreResetsAndReplays() {
    Harness h(kReplay, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    (void)h.engine->forward(1);
    const auto resets = Resets();
    TEST_REQUIRE(h.engine->restore() == 10);
    TEST_REQUIRE(Resets() == resets + 1);
    TEST_REQUIRE(Last().rows == 10 && Last().position == kContinue && !Last().logits);
    TEST_REQUIRE(S().state_restores == 0);
}

void TestAReplayRewindResetsAndReplays() {
    Harness h(kReplay, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    h.engine->set_context_length(3);
    TEST_REQUIRE(Last().rows == 3 && Last().position == kContinue && !Last().logits);
    TEST_REQUIRE(h.engine->get_current_context_length() == 3);
}

// --- clear ---------------------------------------------------------------------

void TestClearingASeekModelDropsTheSequenceWithoutAReset() {
    Harness h(kFromStart, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    const auto resets = Resets();
    h.engine->clear_context();
    TEST_REQUIRE(Resets() == resets);
    TEST_REQUIRE(h.engine->get_current_context_length() == 0);
    TEST_REQUIRE(h.engine->restore() == -1);
    const auto forwards = S().model_forwards.size();
    auto next = Ids(3);
    (void)h.engine->prefill(next);
    TEST_REQUIRE(S().model_forwards.size() == forwards + 1);
    TEST_REQUIRE(Last().rows == 3 && Last().position == 0);
}

void TestClearingAReplayModelResets() {
    Harness h(kReplay, 4096);
    auto ids = Ids(10);
    (void)h.engine->prefill(ids);
    (void)h.engine->checkpoint();
    const auto resets = Resets();
    h.engine->clear_context();
    TEST_REQUIRE(Resets() == resets + 1);
    TEST_REQUIRE(h.engine->get_current_context_length() == 0);
    TEST_REQUIRE(h.engine->restore() == -1);
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestTheModelIsCorelibsOwnByNameWithOneSession);
    RUN(TestTheModelIsCreatedAtTheConfiguredCapacity);
    RUN(TestTheSpecsLimitIsAccepted);
    RUN(TestAMaximumOutsideTheSpecIsRefused);
    RUN(TestAModelThatCannotSeekMustTakeAPassAnywhere);
    RUN(TestAPromptIsOneForward);
    RUN(TestATokenOutsideTheVocabularyIsRefusedBeforeCorelibSeesIt);
    RUN(TestARequestPastTheMaximumIsRefusedBeforeCorelibSeesIt);
    RUN(TestAVisionPayloadIsRefused);
    RUN(TestTheKvCacheIsNotExposed);
    RUN(TestRestoreLoadsTheSnapshotWithoutReplaying);
    RUN(TestRewindingElsewhereReplaysInOnePass);
    RUN(TestRewindingOntoTheCheckpointLoadsItsSnapshot);
    RUN(TestRewindingOntoADivergedCheckpointReplays);
    RUN(TestAFailedRestorePoisonsTheEngine);
    RUN(TestAFailedForwardPoisonsTheEngineAndDropsTheCheckpoint);
    RUN(TestAForwardThatFailsOnceRunsAgainFromTheStart);
    RUN(TestARecurrentModelRunsAgainFromTheStartAfterAFailure);
    RUN(TestAForwardThatFailsTwicePoisonsTheEngine);
    RUN(TestCheckpointingTheSameStateTwiceSavesOnce);
    RUN(TestRestoringRightAfterACheckpointLoadsNothing);
    RUN(TestShrinkingBelowTheCheckpointDropsIt);
    RUN(TestGrowingASnapshotModelSavesAndRestoresAcrossTheNewSession);
    RUN(TestShrinkingWithinTheCapacityKeepsTheSession);
    RUN(TestASeekModelStatesWhereEveryPassStarts);
    RUN(TestAFromStartContinuationRerunsTheWholeSequence);
    RUN(TestASeekRewindMovesOnlyThePosition);
    RUN(TestASeekCheckpointKeepsNoSnapshot);
    RUN(TestASeekRestoreOntoAnUnchangedPrefixMovesOnlyThePosition);
    RUN(TestASeekRestoreAfterTheCacheWasOverwrittenReplays);
    RUN(TestAnUnpagedFromPageContinuationRerunsFromZero);
    RUN(TestAPagedContinuationRerunsOnlyItsPage);
    RUN(TestAPagedContinuationOnABoundaryRunsOnlyTheNewRows);
    RUN(TestGrowingASeekModelReplaysIntoTheNewSession);
    RUN(TestAReplayCheckpointKeepsNoSnapshot);
    RUN(TestAReplayRestoreResetsAndReplays);
    RUN(TestAReplayRewindResetsAndReplays);
    RUN(TestClearingASeekModelDropsTheSequenceWithoutAReset);
    RUN(TestClearingAReplayModelResets);
#undef RUN
    std::cout << "All CorelibModel tests passed\n";
    return 0;
}
