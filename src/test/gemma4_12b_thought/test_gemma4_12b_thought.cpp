/// \file test_gemma4_12b_thought.cpp
/// \brief A prompt is trimmed by the empty thought channel only when it ends
///        with that channel. Thinking off still asks generate() to open it.
///        No NPU.
#include "AutoModel/gemma4_12b_thought.hpp"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::printf("FAIL: %s\n", what);
    failures++;
}

std::vector<int> Channel() {
    return {flm::kGemma4ChannelBegin, flm::kGemma4ThoughtToken, flm::kGemma4NewlineToken,
            flm::kGemma4ChannelEnd};
}

std::vector<int> With(std::vector<int> prefix) {
    const std::vector<int> channel = Channel();
    prefix.insert(prefix.end(), channel.begin(), channel.end());
    return prefix;
}

}  // namespace

int main() {
    Expect(flm::gemma4_ends_with_empty_thought(With({1, 2})),
           "a prompt that ends with the channel was not recognized");
    Expect(flm::gemma4_ends_with_empty_thought(Channel()), "the channel alone was not recognized");
    Expect(!flm::gemma4_ends_with_empty_thought({}), "an empty prompt was treated as the channel");
    Expect(!flm::gemma4_ends_with_empty_thought({100, 45518, 107}), "three tokens were treated as the channel");

    std::vector<int> shifted = With({1});
    shifted.push_back(9);
    Expect(!flm::gemma4_ends_with_empty_thought(shifted), "a channel that is not at the end was trimmed");
    Expect(!flm::gemma4_ends_with_empty_thought({100, 1, 107, 101}), "a different second token was accepted");

    const std::vector<int> normal = With({11, 12});
    Expect(flm::gemma4_feed_empty_thought(false, false), "thinking off did not feed the channel");
    Expect(flm::gemma4_empty_thought_trim(false, false, normal) == 4, "a normal turn did not trim four tokens");

    // A tool-result turn ends at the tool-response close, not the thought channel.
    const std::vector<int> tool_result{11, 12, 50, 51, 52, 53};
    Expect(!flm::gemma4_ends_with_empty_thought(tool_result), "a tool result was treated as the channel");
    Expect(flm::gemma4_feed_empty_thought(false, false), "a tool result did not still feed the channel");
    Expect(flm::gemma4_empty_thought_trim(false, false, tool_result) == 0, "a tool result was trimmed");

    Expect(!flm::gemma4_feed_empty_thought(true, false), "thinking on fed the channel");
    Expect(flm::gemma4_empty_thought_trim(true, false, normal) == 0, "thinking on trimmed the prompt");
    Expect(!flm::gemma4_feed_empty_thought(false, true), "a continuation fed the channel");
    Expect(flm::gemma4_empty_thought_trim(false, true, normal) == 0, "a continuation trimmed the prompt");

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
