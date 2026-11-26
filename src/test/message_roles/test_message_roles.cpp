/// \file test_message_roles.cpp
/// \brief `developer` and later `system` messages without a model: what the probe finds on the Qwen3 and Qwen3.5
///        templates the models ship (templates/, copied from the packages) and on gpt-oss and ChatML shapes, and what
///        the normalization makes of each role order. No NPU needed.
#include "chat/message_roles.hpp"
#include "minja/chat-template.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
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
using flm::chat::message_role_support;

std::string read_template(const char* name) {
    std::ifstream in(std::string(MESSAGE_ROLES_TEMPLATE_DIR) + "/" + name, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open template ") + name);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// \brief The text parts of each message joined, as the server's normalize_template does before the template
json flatten(const json& messages) {
    json out = json::array();
    for (auto message : messages) {
        if (message.contains("content") && message["content"].is_array()) {
            std::string text;
            for (const auto& part : message["content"]) {
                if (part.value("type", "") == "text") text += part["text"].get<std::string>();
            }
            message["content"] = text;
        }
        out.push_back(message);
    }
    return out;
}

/// \brief What AutoModel renders: the template with the extra context every model gets
struct model_template {
    minja::chat_template tmpl;
    explicit model_template(const std::string& source) : tmpl(source, "", "<|im_end|>") {}

    std::string render(const json& messages, bool add_generation_prompt = true) const {
        minja::chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.extra_context = {{"user_system_prompt", ""}};
        inputs.add_generation_prompt = add_generation_prompt;
        return tmpl.apply(inputs);
    }
    message_role_support probe() const {
        return flm::chat::probe_message_roles([this](const json& messages) { return render(messages); });
    }
    /// \brief The request path: normalize, flatten, render
    std::string prompt(const json& messages, bool add_generation_prompt = true) const {
        return render(flatten(flm::chat::normalize_messages(messages, probe())), add_generation_prompt);
    }
};

json msg(const char* role, const json& content) {
    return {{"role", role}, {"content", content}};
}

const std::string RULE = "The secret word is PINEAPPLE.";

// gpt-oss: the first message, system or developer, becomes the developer message; later instructions have no branch.
const std::string GPT_OSS_SHAPE =
    "{%- if messages[0].role == 'developer' or messages[0].role == 'system' -%}"
    "<|start|>developer<|message|>{{ messages[0].content }}<|end|>"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}{%- set loop_messages = messages -%}{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- if message.role == 'user' or message.role == 'assistant' -%}"
    "<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}";

// generic ChatML: any role name is printed as written
const std::string CHATML_SHAPE =
    "{%- for message in messages -%}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

void test_probe() {
    std::printf("probe\n");
    const auto qwen3 = model_template(read_template("qwen3.jinja")).probe();
    CHECK(!qwen3.renders_developer);
    CHECK(qwen3.accepts_later_system);

    const auto qwen3_5 = model_template(read_template("qwen3_5.jinja")).probe();
    CHECK(!qwen3_5.renders_developer);
    CHECK(!qwen3_5.accepts_later_system);

    const auto gpt_oss = model_template(GPT_OSS_SHAPE).probe();
    CHECK(gpt_oss.renders_developer);
    CHECK(!gpt_oss.accepts_later_system);

    const auto chatml = model_template(CHATML_SHAPE).probe();
    CHECK(!chatml.renders_developer);  // it prints any role, so `developer` is not a role it knows
    CHECK(chatml.accepts_later_system);

    // a template that moves every system message to the top does not take a later one in place
    const std::string hoisting =
        "{%- for message in messages -%}{%- if message.role == 'system' -%}"
        "<|im_start|>system\n{{ message.content }}<|im_end|>\n{%- endif -%}{%- endfor -%}"
        "{%- for message in messages -%}{%- if message.role != 'system' -%}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endif -%}{%- endfor -%}";
    CHECK(!model_template(hoisting).probe().accepts_later_system);

    const auto broken = flm::chat::probe_message_roles([](const json&) -> std::string {
        throw std::runtime_error("template error");
    });
    CHECK(!broken.renders_developer);
    CHECK(!broken.accepts_later_system);
}

void test_normalize() {
    std::printf("normalize\n");
    const message_role_support plain{false, true};
    const message_role_support strict{false, false};
    const message_role_support harmony{true, false};

    // a leading developer message becomes system
    auto out = flm::chat::normalize_messages(json::array({msg("developer", RULE), msg("user", "Hi")}), plain);
    CHECK_EQ(out[0]["role"].get<std::string>(), "system");
    CHECK_EQ(flatten(out)[0]["content"].get<std::string>(), RULE);

    // developer then system at the start: one system message, in order
    out = flm::chat::normalize_messages(json::array({msg("developer", "A"), msg("system", "B"), msg("user", "Hi")}), plain);
    CHECK(out.size() == 2);
    CHECK_EQ(out[0]["role"].get<std::string>(), "system");
    CHECK_EQ(flatten(out)[0]["content"].get<std::string>(), "AB");

    // gpt-oss keeps a leading developer message, and the run stays one message
    out = flm::chat::normalize_messages(json::array({msg("system", "A"), msg("developer", "B"), msg("user", "Hi")}), harmony);
    CHECK(out.size() == 2);
    CHECK_EQ(out[0]["role"].get<std::string>(), "developer");
    out = flm::chat::normalize_messages(json::array({msg("system", "A"), msg("user", "Hi")}), harmony);
    CHECK_EQ(out[0]["role"].get<std::string>(), "system");

    const json later = json::array({msg("system", "S"), msg("user", "Q1"), msg("assistant", "A1"),
                                    msg("developer", RULE), msg("user", "Q2")});
    // a template that takes a later system message: it stays system, in place
    out = flm::chat::normalize_messages(later, plain);
    CHECK(out.size() == 5);
    CHECK_EQ(out[3]["role"].get<std::string>(), "system");
    CHECK_EQ(flatten(out)[3]["content"].get<std::string>(), RULE);

    // a template that rejects it: a user turn in place, wrapped, joined with the user message after it
    out = flm::chat::normalize_messages(later, strict);
    CHECK(out.size() == 4);
    CHECK_EQ(out[3]["role"].get<std::string>(), "user");
    CHECK_EQ(flatten(out)[3]["content"].get<std::string>(),
             "<system-reminder>\n" + RULE + "\n</system-reminder>\nQ2");

    // not wrapped twice
    out = flm::chat::normalize_messages(
        json::array({msg("user", "Q1"), msg("system", " <system-reminder>\nX\n</system-reminder>\n")}), strict);
    CHECK_EQ(flatten(out)[0]["content"].get<std::string>(), "Q1\n <system-reminder>\nX\n</system-reminder>\n");

    // after a user message it starts on a new line; an empty instruction adds nothing
    out = flm::chat::normalize_messages(json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2"),
                                                     msg("system", RULE), msg("developer", nullptr),
                                                     msg("system", ""), msg("user", "Q3")}),
                                        strict);
    CHECK(out.size() == 3);
    CHECK_EQ(flatten(out)[2]["content"].get<std::string>(),
             "Q2\n<system-reminder>\n" + RULE + "\n</system-reminder>\nQ3");

    // an empty instruction between an assistant turn and a later one is not a user turn: the later one starts
    // its own turn, with no newline in front of it
    out = flm::chat::normalize_messages(json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("system", ""),
                                                     msg("system", RULE), msg("user", "Q2")}),
                                        strict);
    CHECK(out.size() == 3);
    CHECK_EQ(flatten(out)[2]["content"].get<std::string>(), "<system-reminder>\n" + RULE + "\n</system-reminder>\nQ2");

    // a text-part array is joined as the server joins it
    out = flm::chat::normalize_messages(
        json::array({msg("user", "Q1"), msg("assistant", "A1"),
                     msg("system", json::array({{{"type", "text"}, {"text", "lo"}}, {{"type", "text"}, {"text", "ok"}}}))}),
        strict);
    CHECK_EQ(flatten(out)[2]["content"].get<std::string>(), "<system-reminder>\nlook\n</system-reminder>\n");

    // a later instruction sent as a user turn holds text only: an image there is a 400 naming the part (as an image
    // in any message but a user message is), and content that cannot be read is a 400, not an empty turn
    const json image = {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AA=="}}}};
    const auto error_of = [&](const json& messages) {
        try {
            flm::chat::normalize_messages(messages, strict);
        } catch (const flm::chat::message_error& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    CHECK_EQ(error_of(json::array({msg("user", "Q1"), msg("assistant", "A1"),
                                   msg("system", json::array({{{"type", "text"}, {"text", "look"}}, image}))})),
             "messages[2].content[1]: a 'system' message holds text parts only, not 'image_url'");
    CHECK_EQ(error_of(json::array({msg("user", "Q1"), msg("assistant", "A1"),
                                   msg("developer", {{"type", "text"}, {"text", RULE}})})),
             "messages[2].content: must be a string or an array of content parts");
    CHECK_EQ(error_of(json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("system", 7)})),
             "messages[2].content: must be a string or an array of content parts");

    // an instruction with no text whose role changes adds no turn: not an empty user turn, not an empty system turn
    for (const json& empty : {json(""), json(nullptr), json::array(), json::array({{{"type", "text"}, {"text", ""}}})}) {
        out = flm::chat::normalize_messages(
            json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("system", empty), msg("user", "Q2")}), strict);
        CHECK(out.size() == 3);
        CHECK_EQ(flatten(out)[2]["content"].get<std::string>(), "Q2");
        out = flm::chat::normalize_messages(json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("system", empty)}), strict);
        CHECK(out.size() == 2);
    }
    out = flm::chat::normalize_messages(json::array({msg("developer", ""), msg("user", "Hi")}), plain);
    CHECK(out.size() == 1);
    CHECK_EQ(out[0]["role"].get<std::string>(), "user");
    out = flm::chat::normalize_messages(json::array({msg("user", "Q1"), msg("assistant", "A1"), msg("developer", nullptr),
                                                     msg("user", "Q2")}), plain);
    CHECK(out.size() == 3);
    // an empty system message that stays system is passed on as before
    out = flm::chat::normalize_messages(json::array({msg("system", ""), msg("user", "Hi")}), plain);
    CHECK(out.size() == 2);

    // each normalized message keeps the request index it starts at
    std::vector<std::size_t> indices;
    flm::chat::normalize_messages(later, strict, &indices);
    CHECK(indices.size() == 4);
    CHECK(indices.size() == 4 && indices[3] == 3);
    indices.clear();
    flm::chat::normalize_messages(json::array({msg("developer", ""), msg("user", "Hi"), msg("assistant", "A"),
                                               msg("tool", "T")}), plain, &indices);
    CHECK(indices.size() == 3 && indices[0] == 1 && indices[2] == 3);

    // reasoning of earlier replies is still removed
    out = flm::chat::normalize_messages(
        json::array({msg("user", "Q"), {{"role", "assistant"}, {"content", "A"}, {"reasoning_content", "R"}}}), plain);
    CHECK(!out[1].contains("reasoning_content"));
}

void test_active_tool_reasoning() {
    // AMD Qwen3.5-9B_gpuep_1.0_gpu_128 @ f3a9b4aeb60a42542cc4fec561e1f1095e427483.
    const model_template gpu(read_template("qwen3_5_9b_oga.jinja"));
    const auto roles = gpu.probe();
    const auto prepare = [&](const json& history) {
        std::vector<std::size_t> indices;
        auto messages = flatten(flm::chat::normalize_messages(history, roles, &indices));
        return messages;
    };
    auto call = msg("assistant", "");
    call["tool_calls"] = json::array({{{"type", "function"}, {"id", "a"},
        {"function", {{"name", "read"}, {"arguments", json::object()}}}}});
    call["reasoning_content"] = "synthetic note";
    auto history = json::array({msg("user", "Q"), call, msg("tool", "42")});
    CHECK_EQ(gpu.render(prepare(history)), gpu.render(history));
    // An entire wrapped tool result is a continuation, including when sent as text parts.
    history[2] = msg("user", json::array({{{"type", "text"}, {"text", " <tool_response>"}},
                                          {{"type", "text"}, {"text", "42</tool_response>\n"}}}));
    history.push_back(call);
    history.push_back(msg("tool", "43"));
    CHECK(prepare(history)[1].contains("reasoning_content") && prepare(history)[3].contains("reasoning_content"));
    history.push_back(msg("user", "New question"));
    CHECK(!prepare(history)[1].contains("reasoning_content") && !prepare(history)[3].contains("reasoning_content"));
    history.push_back(call);
    history.back().erase("tool_calls");
    CHECK(!prepare(history).back().contains("reasoning_content"));
    history.back() = call;
    history.back()["reasoning_content"] = 17;
    history.back()["thinking"] = "alias";
    history.back()["reasoning"] = "alias";
    const auto invalid = prepare(history).back();
    CHECK(!invalid.contains("reasoning_content") && !invalid.contains("thinking") && !invalid.contains("reasoning"));
}

void test_models() {
    std::printf("models\n");
    const model_template qwen3(read_template("qwen3.jinja"));
    const model_template qwen3_5(read_template("qwen3_5.jinja"));

    // the instruction reaches the prompt, and as developer it renders exactly as system
    for (const model_template* model : {&qwen3, &qwen3_5}) {
        const std::string as_system = model->prompt(json::array({msg("system", RULE), msg("user", "What is the secret word?")}));
        const std::string as_developer = model->prompt(json::array({msg("developer", RULE), msg("user", "What is the secret word?")}));
        CHECK(as_system.find(RULE) != std::string::npos);
        CHECK_EQ(as_developer, as_system);
    }

    // a later instruction renders, in place, on both
    const json history = json::array({msg("system", "You are terse."), msg("user", "Hi"), msg("assistant", "Hello.")});
    for (const char* role : {"system", "developer"}) {
        json messages = history;
        messages.push_back(msg(role, RULE));
        messages.push_back(msg("user", "What is the secret word?"));
        const std::string p3 = qwen3.prompt(messages);
        CHECK(p3.find("<|im_start|>system\n" + RULE + "<|im_end|>") != std::string::npos);
        std::string p35;
        try {
            p35 = qwen3_5.prompt(messages);
        } catch (const std::exception& e) {
            std::printf("  qwen3.5 threw: %s\n", e.what());
        }
        CHECK(p35.find("<|im_start|>user\n<system-reminder>\n" + RULE + "\n</system-reminder>\nWhat is the secret word?") !=
              std::string::npos);
        CHECK(p35.find("Hello.") < p35.find(RULE));
    }

    // the instruction text itself reaches the prompt, not only its tags
    for (const model_template* model : {&qwen3, &qwen3_5}) {
        json messages = history;
        messages.push_back(msg("developer", RULE));
        messages.push_back(msg("user", "What is the secret word?"));
        CHECK(model->prompt(messages).find("The secret word is PINEAPPLE.") != std::string::npos);
    }

    // a conversation that only grows renders a growing prompt: the earlier turns render the same each time,
    // including one with a later instruction (nothing is moved to the system message at the top)
    json turns = json::array({msg("system", "You are terse."), msg("user", "Q1")});
    const json appended[] = {msg("assistant", "A1"), msg("system", RULE), msg("user", "Q2"),
                             msg("assistant", "A2"), msg("developer", "Answer in English."), msg("user", "Q3")};
    for (const model_template* model : {&qwen3, &qwen3_5}) {
        json messages = turns;
        std::string before = model->prompt(messages, false);
        for (const auto& next : appended) {
            messages.push_back(next);
            if (next["role"] != "user") continue;  // a request ends with a user turn
            const std::string now = model->prompt(messages, false);
            if (now.compare(0, before.size(), before) != 0) {
                std::printf("  not a prefix:\n    before: [%s]\n    now:    [%s]\n", before.c_str(), now.c_str());
                failures++;
            }
            before = now;
        }
    }
}
}  // namespace

int main() {
    try {
        test_probe();
        test_normalize();
        test_active_tool_reasoning();
        test_models();
    } catch (const std::exception& e) {
        std::printf("  FAIL: exception %s\n", e.what());
        failures++;
    }
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all message role checks passed\n");
    return 0;
}
