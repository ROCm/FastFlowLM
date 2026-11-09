/// \file test_prompt_reuse.cpp
/// \brief plan_prompt_reuse: how much of a prompt the engine keeps and which images / audio clips the model drops
///        from its payload. Every vision model and _shared_insert ask it, so they cannot disagree. No NPU needed.
#include "prompt_reuse.hpp"

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

constexpr int IMG = 900;   // image placeholder
constexpr int BOA = 901;   // audio open / close
constexpr int EOA = 902;
constexpr int PAD = 903;   // audio rows

/// \brief text, then `n` image rows, then text
std::vector<int> with_image(std::vector<int> head, size_t n, std::vector<int> tail) {
    head.insert(head.end(), n, IMG);
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

std::vector<prompt_media_t> images(const std::vector<int>& tokens, const std::vector<size_t>& counts) {
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, IMG, counts, prompt_media_t::kImage, media);
    return media;
}

void test_text_only() {
    std::printf("text only\n");
    const std::vector<int> prompt = {1, 2, 3, 4, 5};
    auto plan = plan_prompt_reuse({}, prompt);
    CHECK(plan.reason == prompt_reuse_t::kNoHistory && plan.skip == 0 && !plan.clear);

    plan = plan_prompt_reuse({1, 2, 3}, prompt);
    CHECK(plan.reason == prompt_reuse_t::kPrefix && plan.skip == 3 && !plan.clear);

    plan = plan_prompt_reuse({1, 2, 9}, prompt);
    CHECK(plan.reason == prompt_reuse_t::kDiverged && plan.skip == 0 && plan.clear);

    // a history longer than the prompt (prompt + reply of the last turn) is a divergence too
    plan = plan_prompt_reuse({1, 2, 3, 4, 5, 6}, prompt);
    CHECK(plan.reason == prompt_reuse_t::kDiverged && plan.skip == 0 && plan.clear);

    // the d49e4c3e case: the history is the whole prompt; the last token has to be read again for its logits
    plan = plan_prompt_reuse(prompt, prompt);
    CHECK(plan.reason == prompt_reuse_t::kWholePrompt && plan.skip == 0 && plan.clear);
}

void test_same_size_image_again() {
    std::printf("the same question about another image of the same size (CI red / blue)\n");
    // the tokens only know the image's size, so red and blue give the same prompt
    const std::vector<int> prompt = with_image({1, 2}, 4, {3, 4});
    const auto media = images(prompt, {4});
    const auto plan = plan_prompt_reuse(prompt, prompt, media);
    CHECK(plan.clear && plan.skip == 0);
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 0);   // keep the image: it is prefilled again
    CHECK(plan.media_end == 2 + 4);
}

void test_follow_up_turn() {
    std::printf("a follow-up turn keeps the image of the turn before\n");
    const std::vector<int> turn1 = with_image({1, 2}, 4, {3});
    std::vector<int> turn2 = turn1;
    turn2.insert(turn2.end(), {7, 8});
    const auto media = images(turn2, {4});
    const auto plan = plan_prompt_reuse(turn1, turn2, media);
    CHECK(plan.reason == prompt_reuse_t::kPrefix && plan.skip == turn1.size());
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 1);
    CHECK(plan.media_end == 0);

    // a new image in turn 2: the first is cached, the second is prefilled, ending 3 rows into the new tokens
    std::vector<int> turn2b = turn1;
    turn2b.insert(turn2b.end(), {7, IMG, IMG, 8});
    const auto media_b = images(turn2b, {4, 2});
    const auto plan_b = plan_prompt_reuse(turn1, turn2b, media_b);
    CHECK(plan_b.skip == turn1.size());
    CHECK(plan_b.skipped_media[prompt_media_t::kImage] == 1);
    CHECK(plan_b.media_end == 3);
}

void test_diverged_keeps_everything() {
    std::printf("a history that diverges keeps every item\n");
    const std::vector<int> prompt = with_image({1, 2}, 4, {3, 4});
    const auto media = images(prompt, {4});
    const auto plan = plan_prompt_reuse(with_image({1, 2}, 4, {5}), prompt, media);
    CHECK(plan.reason == prompt_reuse_t::kDiverged && plan.clear && plan.skip == 0);
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 0);
}

void test_inside_image() {
    std::printf("a history that ends inside an image is not reused\n");
    const std::vector<int> prompt = with_image({1, 2}, 4, {3, 4});
    const auto media = images(prompt, {4});
    const std::vector<int> history(prompt.begin(), prompt.begin() + 4);   // 2 text + 2 of the 4 rows
    const auto plan = plan_prompt_reuse(history, prompt, media);
    CHECK(plan.reason == prompt_reuse_t::kInsideMedia && plan.clear && plan.skip == 0);
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 0);
    CHECK(plan.media_end == 6);

    // ending right after the image is fine
    const std::vector<int> after(prompt.begin(), prompt.begin() + 6);
    const auto plan_after = plan_prompt_reuse(after, prompt, media);
    CHECK(plan_after.reason == prompt_reuse_t::kPrefix && plan_after.skip == 6);
    CHECK(plan_after.skipped_media[prompt_media_t::kImage] == 1);
}

void test_counts() {
    std::printf("items from token counts\n");
    // two adjacent images, then a picture split over two runs (view + slice)
    const std::vector<int> prompt = {1, IMG, IMG, IMG, IMG, IMG, 2, IMG, IMG, 3, IMG, 4};
    const auto media = images(prompt, {2, 3, 3});
    CHECK(media.size() == 3);
    CHECK(media[0].begin == 1 && media[0].end == 3);
    CHECK(media[1].begin == 3 && media[1].end == 6);
    CHECK(media[2].begin == 7 && media[2].end == 11);   // both runs, and the text between them
    // a history ending between the picture's two runs ends inside it
    const std::vector<int> between(prompt.begin(), prompt.begin() + 10);
    CHECK(plan_prompt_reuse(between, prompt, media).reason == prompt_reuse_t::kInsideMedia);

    // a per-image count that does not match how the prompt was expanded is caught as an item split in two:
    // with /4 and 16 rows per image, the first image would look like 4 rows, ending inside the real image
    const std::vector<int> two = with_image({1}, 16, {2});
    const auto wrong = images(two, {4});
    CHECK(wrong[0].end == 5);
}

void test_audio_brackets() {
    std::printf("audio items between open / close tokens, next to images\n");
    std::vector<int> prompt = {1, BOA, PAD, PAD, EOA, 2, IMG, IMG, 3, BOA, PAD, EOA, 4};
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(prompt, IMG, {2}, prompt_media_t::kImage, media);
    prompt_media_from_brackets(prompt, BOA, EOA, prompt_media_t::kAudio, media);
    CHECK(media.size() == 3);
    const std::vector<int> history(prompt.begin(), prompt.begin() + 9);
    const auto plan = plan_prompt_reuse(history, prompt, media);
    CHECK(plan.skip == 9);
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 1);
    CHECK(plan.skipped_media[prompt_media_t::kAudio] == 1);
    CHECK(plan.media_end == 12 - 9);
    CHECK(plan.media_end_by_kind[prompt_media_t::kImage] == 0);
    CHECK(plan.media_end_by_kind[prompt_media_t::kAudio] == 12 - 9);

    // an item with no tokens (a clip too short for one row) adds no rows to the first chunk
    std::vector<prompt_media_t> with_empty = media;
    prompt_media_t empty;
    empty.kind = prompt_media_t::kAudio;
    empty.begin = empty.end = prompt.size();
    with_empty.push_back(empty);
    CHECK(plan_prompt_reuse(history, prompt, with_empty).media_end == 12 - 9);

    const std::vector<int> inside(prompt.begin(), prompt.begin() + 3);
    CHECK(plan_prompt_reuse(inside, prompt, media).reason == prompt_reuse_t::kInsideMedia);
}
void test_hashes() {
    std::printf("an earlier image replaced by another of the same size is not reused\n");
    // turn 1 asked about red; the client resends the conversation with turn 1's image replaced by blue
    const std::vector<int> turn1 = with_image({1, 2}, 4, {3});
    std::vector<int> turn2 = turn1;
    turn2.insert(turn2.end(), {7, 8});
    auto red = images(turn1, {4});
    red[0].hash = 0x7ed;
    auto blue = images(turn2, {4});
    blue[0].hash = 0xb1e;

    auto plan = plan_prompt_reuse(turn1, turn2, blue, &red);
    CHECK(plan.reason == prompt_reuse_t::kMediaChanged && plan.clear && plan.skip == 0);
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 0);   // blue is read in
    CHECK(plan.media_end == 6);

    // the same picture again is reused
    auto red2 = images(turn2, {4});
    red2[0].hash = 0x7ed;
    plan = plan_prompt_reuse(turn1, turn2, red2, &red);
    CHECK(plan.reason == prompt_reuse_t::kPrefix && plan.skip == turn1.size());
    CHECK(plan.skipped_media[prompt_media_t::kImage] == 1);

    // a history whose items were checked elsewhere (a prefix slot), and items the model does not hash, are not compared
    CHECK(plan_prompt_reuse(turn1, turn2, blue, nullptr).reason == prompt_reuse_t::kPrefix);
    auto unhashed = images(turn2, {4});
    CHECK(plan_prompt_reuse(turn1, turn2, unhashed, &red).reason == prompt_reuse_t::kPrefix);

    // a history that does not know the item (read in without a plan) is not trusted with it
    const std::vector<prompt_media_t> nothing;
    CHECK(plan_prompt_reuse(turn1, turn2, red2, &nothing).reason == prompt_reuse_t::kMediaChanged);

    // an image after the history is new either way: nothing to compare
    std::vector<int> turn2b = turn1;
    turn2b.insert(turn2b.end(), {7, IMG, IMG, 8});
    auto media_b = images(turn2b, {4, 2});
    media_b[0].hash = 0x7ed;
    media_b[1].hash = 0xb1e;
    plan = plan_prompt_reuse(turn1, turn2b, media_b, &red);
    CHECK(plan.reason == prompt_reuse_t::kPrefix && plan.skipped_media[prompt_media_t::kImage] == 1);
}
}  // namespace

int main() {
    test_text_only();
    test_same_size_image_again();
    test_follow_up_turn();
    test_diverged_keeps_everything();
    test_inside_image();
    test_counts();
    test_audio_brackets();
    test_hashes();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
