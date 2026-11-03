/// \file test_chat_continue.cpp
/// \brief continue_final_message without a model: where the rendered prompt is cut, on the template shapes FLM ships,
///        and how the output parser treats the continued text. No NPU needed.
#include "chat/chat_continue.hpp"
#include "chat/chat_format.hpp"
#include "chat/chat_parser.hpp"
#include "chat/chat_stream.hpp"
#include "minja/chat-template.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
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

#define CHECK_EQ(actual, expected)                                                  \
    do {                                                                            \
        const std::string a_ = (actual), e_ = (expected);                           \
        if (a_ != e_) {                                                             \
            std::printf("  FAIL %s:%d: %s\n    got:      [%s]\n    expected: [%s]\n", \
                        __FILE__, __LINE__, #actual, a_.c_str(), e_.c_str());       \
            failures++;                                                             \
        }                                                                           \
    } while (0)

using json = nlohmann::ordered_json;

json chat(const std::string& user, const json& assistant) {
    return json::array({{{"role", "user"}, {"content", user}}, {{"role", "assistant"}, {"content", assistant}}});
}

/// \brief What AutoModel::_render_chat_template does when continuing
std::string continued(const std::string& source, const json& messages, const std::string& bos = "") {
    minja::chat_template tmpl(source, bos, "");
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = false;
    return chat_continue_final_message(tmpl.apply(inputs), inputs.messages.back());
}

bool rejects(const std::string& rendered, const json& message) {
    try {
        chat_continue_final_message(rendered, message);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

// Shapes of the templates in the model packages, cut down to the parts that decide the cut.
const std::string LLAMA3 =
    "{{- bos_token }}{%- for m in messages %}"
    "{{- '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' + m['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

const std::string CHATML =
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

// Qwen3: the last assistant turn after the last user query always gets a think block
const std::string QWEN3 =
    "{%- for m in messages %}"
    "{%- if m['role'] == 'assistant' and loop.last %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m['content'].lstrip('\\n') + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{%- endif %}";

// Gemma: assistant is "model", content trimmed
const std::string GEMMA =
    "{{- bos_token }}{%- for m in messages %}"
    "{%- set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "{{- '<start_of_turn>' + role + '\\n' + m['content'] | trim + '<end_of_turn>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<start_of_turn>model\\n' }}{%- endif %}";

// DeepSeek-R1: an assistant turn keeps only what follows </think>
const std::string DEEPSEEK =
    "{{- bos_token }}{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{- '<\xEF\xBD\x9C" "User\xEF\xBD\x9C>' + m['content'] }}"
    "{%- else %}{{- '<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>' + m['content'].split('</think>')[-1] + '<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>' }}{%- endif %}";

// Qwen2-VL: typed content, images before the text
const std::string QWEN2VL =
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' }}"
    "{%- if m['content'] is string %}{{- m['content'] }}{%- else %}{%- for c in m['content'] %}"
    "{%- if c['type'] == 'image' %}{{- '<|vision_start|><|image_pad|><|vision_end|>' }}{%- else %}{{- c['text'] }}{%- endif %}"
    "{%- endfor %}{%- endif %}{{- '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

// Qwen3.5: a reply opens with an empty think block, a finished message has none
const std::string QWEN35 =
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] | trim + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{%- endif %}";

// LFM2.5-Thinking: a reply opens inside a think block, a finished message is the answer alone
const std::string THINKING =
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}";

/// \brief What AutoModel::_render_chat_template does when continuing
chat_continued continued_prompt(const std::string& source, const json& messages) {
    minja::chat_template tmpl(source, "", "");
    return chat_continue_prompt(
        [&tmpl](const json& chat, bool add_generation_prompt) {
            minja::chat_template_inputs inputs;
            inputs.messages = chat;
            inputs.add_generation_prompt = add_generation_prompt;
            return tmpl.apply(inputs);
        },
        messages);
}

void test_prompt() {
    std::printf("prompt\n");
    const std::string q = "Mind the <think> tag. Is Paris the capital of France?";
    const std::string user_turn = "<|im_start|>user\n" + q + "<|im_end|>\n";

    // where the template opens a reply the way it renders a finished message, the cut is transformers'
    chat_continued chatml = continued_prompt(CHATML, chat(q, "The answer is:"));
    CHECK_EQ(chatml.prompt, continued(CHATML, chat(q, "The answer is:")));
    CHECK_EQ(chatml.turn, "<|im_start|>assistant\nThe answer is:");
    chat_continued qwen3 = continued_prompt(QWEN3, chat(q, "The answer is:"));
    CHECK_EQ(qwen3.prompt, continued(QWEN3, chat(q, "The answer is:")));
    CHECK_EQ(qwen3.turn, "<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is:");

    // where it does not, the message's text follows the reply's opening, as the model writes a reply
    chat_continued qwen35 = continued_prompt(QWEN35, chat(q, "The answer is: "));
    CHECK_EQ(qwen35.prompt, user_turn + "<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is:");
    CHECK_EQ(qwen35.turn, "<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is:");
    chat_continued thinking = continued_prompt(THINKING, chat(q, "The answer is:"));
    CHECK_EQ(thinking.prompt, user_turn + "<|im_start|>assistant\n<think>\nThe answer is:");

    // the turn is what the output parser reads: the user's <think> is not in it, the template's are
    const chat_format_spec spec = chat_format_make(chat_format_kind::HERMES_JSON, true);
    CHECK_EQ(chat_format_prefill(spec, chatml.turn, true), "");
    CHECK_EQ(chat_format_prefill(spec, qwen35.turn, true), "");
    CHECK_EQ(chat_format_prefill(spec, thinking.turn, true), "<think>");
}

void test_template_shapes() {
    std::printf("template shapes\n");
    const std::string q = "Is Paris the capital of France? Answer yes or no.";

    CHECK_EQ(continued(LLAMA3, chat(q, "The answer is:"), "<|begin_of_text|>"),
             "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + q +
             "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:");
    // the template trims, so the trailing space goes, as in transformers
    CHECK_EQ(continued(LLAMA3, chat(q, "The answer is: "), "<|begin_of_text|>"),
             "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + q +
             "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:");

    CHECK_EQ(continued(CHATML, chat(q, "The answer is:")),
             "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\nThe answer is:");
    // the template keeps the text as it is, trailing space included
    CHECK_EQ(continued(CHATML, chat(q, "The answer is: ")),
             "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\nThe answer is: ");

    CHECK_EQ(continued(QWEN3, chat(q, "The answer is:")),
             "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is:");
    // lstrip('\n') in the template: matched without the leading newline
    CHECK_EQ(continued(QWEN3, chat(q, "\nThe answer is:")),
             "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is:");

    CHECK_EQ(continued(GEMMA, chat(q, " The answer is: "), "<bos>"),
             "<bos><start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\nThe answer is:");

    CHECK_EQ(continued(DEEPSEEK, chat(q, "The answer is:"), "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>"),
             "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C><\xEF\xBD\x9C" "User\xEF\xBD\x9C>" + q +
             "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>The answer is:");
    // an unclosed think block is left alone, so reasoning can be continued
    CHECK_EQ(continued(DEEPSEEK, chat(q, "<think>\nParis is"), ""),
             "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>" + q + "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C><think>\nParis is");
    // the template moves the closed block out of the message: nothing to continue from
    bool threw = false;
    try {
        continued(DEEPSEEK, chat(q, "<think>Paris.</think>The answer is:"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    json vl = chat(q, json::array({{{"type", "image"}}, {{"type", "text"}, {"text", "This picture shows"}}}));
    CHECK_EQ(continued(QWEN2VL, vl),
             "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|><|image_pad|><|vision_end|>This picture shows");
}

void test_cut() {
    std::printf("cut\n");
    const json msg = {{"role", "assistant"}, {"content", "The answer is:"}};
    // the last occurrence: the user may have written the same words
    CHECK_EQ(chat_continue_final_message("U: The answer is: A: The answer is:<end>", msg), "U: The answer is: A: The answer is:");
    // leading whitespace in the message, kept by the template: cut after the text
    CHECK_EQ(chat_continue_final_message("A:  The answer is:<end>", {{"content", "  The answer is:"}}), "A:  The answer is:");
    // trailing whitespace the template kept is kept; trailing whitespace it dropped is not looked for
    CHECK_EQ(chat_continue_final_message("A: x \n<end>", {{"content", "x \n"}}), "A: x \n");
    CHECK_EQ(chat_continue_final_message("A: x<end>", {{"content", "x \n"}}), "A: x");
    // whitespace on both sides: transformers compares against the unstripped text, so the
    // trailing space goes even when the template kept it; kept that way to cut where vLLM cuts
    CHECK_EQ(chat_continue_final_message("A:  x <end>", {{"content", "  x "}}), "A:  x");

    CHECK(rejects("A: <end>", {{"content", ""}}));
    CHECK(rejects("A:  \n<end>", {{"content", " \n"}}));
    CHECK(rejects("A: <end>", {{"content", nullptr}}));
    CHECK(rejects("A: <end>", {{"role", "assistant"}}));
    CHECK(rejects("A: <image><end>", {{"content", json::array({{{"type", "image"}}})}}));
    CHECK(rejects("A: something else<end>", msg));
}

void test_prefill() {
    std::printf("prefill\n");
    chat_format_spec r1 = chat_format_make(chat_format_kind::CONTENT_ONLY, true);
    r1.reasoning_forced = true;
    const std::string tail = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>The answer is:";
    // a fresh R1 turn reasons first; a continued answer is not reasoning
    CHECK_EQ(chat_format_prefill(r1, tail), "<think>");
    CHECK_EQ(chat_format_prefill(r1, tail, true), "");
    // a continued message that opened reasoning is still in it
    CHECK_EQ(chat_format_prefill(r1, "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C><think>\nParis is", true), "<think>");

    chat_format_spec qwen = chat_format_make(chat_format_kind::HERMES_JSON, true);
    CHECK_EQ(chat_format_prefill(qwen, "assistant\n<think>\n\n</think>\n\nThe answer is:", true), "");
    CHECK_EQ(chat_format_prefill(qwen, "assistant\n<think>\n\n</think>\n\n<think>\nLet me", true), "<think>");
}

void test_parse() {
    std::printf("parse\n");
    const chat_format_spec qwen = chat_format_make(chat_format_kind::HERMES_JSON, true);
    chat_parse_options plain;
    chat_parse_options keep;
    keep.keep_whitespace = true;

    CHECK_EQ(chat_parse(qwen, " yes", false, plain).content, "yes");
    CHECK_EQ(chat_parse(qwen, " yes", false, keep).content, " yes");
    // the whole text, once final: prefix + content is what the model wrote
    CHECK_EQ(chat_parse(qwen, " yes \n", false, keep).content, " yes \n");
    CHECK_EQ(chat_parse(qwen, " \n", false, keep).content, " \n");
    // while generating, trailing whitespace waits for what follows it
    CHECK_EQ(chat_parse(qwen, " yes \n", true, keep).content, " yes");
    CHECK_EQ(chat_parse(qwen, " \n", true, keep).content, "");
    // continuing inside reasoning: the reasoning text is exact, the answer after it is laid out as usual
    chat_msg in_reasoning = chat_parse(qwen, "<think> the capital</think>\n\nParis.\n", false, keep);
    CHECK_EQ(in_reasoning.reasoning, " the capital");
    CHECK_EQ(in_reasoning.content, "Paris.");
    // a newline before a tool call is layout, not an answer
    CHECK_EQ(chat_parse(qwen, "\n<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>", false, keep).content, "");

    // a model that reasons first reads the continued message as reasoning, and closes it
    chat_parse_options closes = keep;
    closes.reasoning_may_close = true;
    chat_msg closed = chat_parse(qwen, " Paris.\n</think>\n\nThe capital of France is Paris.", false, closes);
    CHECK_EQ(closed.reasoning, " Paris.\n");
    CHECK_EQ(closed.content, "The capital of France is Paris.");
    // no marker: the continuation is the answer, as written
    CHECK_EQ(chat_parse(qwen, " Yes", false, closes).content, " Yes");
    CHECK_EQ(chat_parse(qwen, " Yes", false, closes).reasoning, "");
    // while neither marker has shown, nothing is released
    CHECK_EQ(chat_parse(qwen, " Paris.\n</thi", true, closes).content, "");
    CHECK_EQ(chat_parse(qwen, " Paris.\n</thi", true, closes).reasoning, "");
    // a block the model opens itself is parsed as usual
    chat_msg opened = chat_parse(qwen, "<think>hm</think>Paris.", false, closes);
    CHECK_EQ(opened.reasoning, "hm");
    CHECK_EQ(opened.content, "Paris.");
    {
        chat_stream stream(qwen, "", closes);
        std::string r, c;
        for (const char* piece : {" Paris", ".\n</", "think>\n\nThe", " capital."}) {
            for (const chat_msg_delta& d : stream.update(piece, false)) { r += d.reasoning; c += d.content; }
        }
        for (const chat_msg_delta& d : stream.update("", true)) { r += d.reasoning; c += d.content; }
        CHECK_EQ(r, " Paris.\n");
        CHECK_EQ(c, "The capital.");
    }

    // the stream sends the same text, and never takes back what it sent
    for (const std::vector<std::string>& pieces : std::vector<std::vector<std::string>>{
             {" ", "yes", ",", " it", " is.", "\n"}, {"\n", "\n"}}) {
        chat_stream stream(qwen, "", keep);
        std::string streamed, whole;
        for (const std::string& piece : pieces) {
            whole += piece;
            for (const chat_msg_delta& d : stream.update(piece, false)) {
                streamed += d.content;
            }
        }
        for (const chat_msg_delta& d : stream.update("", true)) {
            streamed += d.content;
        }
        CHECK_EQ(streamed, whole);
        CHECK_EQ(streamed, chat_parse(qwen, whole, false, keep).content);
    }
    // text after a tool call or a think block settles the whitespace before it, so a partial
    // parse is still a prefix of the final one and no delta gets dropped
    for (const std::vector<std::string>& pieces : std::vector<std::vector<std::string>>{
             {" Paris. ", "<tool_call>{\"name\": \"f\", \"arguments\": {}}", "</tool_call>", "\nDo", "ne"},
             {"foo ", "<think>x</think>", " b", "ar"}}) {
        chat_stream stream(qwen, "", keep);
        std::string streamed, whole;
        for (const std::string& piece : pieces) {
            whole += piece;
            for (const chat_msg_delta& d : stream.update(piece, false)) {
                streamed += d.content;
            }
        }
        for (const chat_msg_delta& d : stream.update("", true)) {
            streamed += d.content;
        }
        CHECK_EQ(streamed, chat_parse(qwen, whole, false, keep).content);
    }
    // an empty block the model closes is layout: the answer after it is the content
    chat_msg empty_block = chat_parse(qwen, "\n</think>\n\nParis.", false, closes);
    CHECK_EQ(empty_block.reasoning, "");
    CHECK_EQ(empty_block.content, "Paris.");
}

} // namespace

int main() {
    test_template_shapes();
    test_prompt();
    test_cut();
    test_prefill();
    test_parse();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
