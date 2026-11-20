/// \file test_chat_protocols.cpp
/// \brief What a chat turn looks like on the wire, without a model: the Chat Completions sink is
///        fed the same turns the handler was, and must write the bytes recorded in golden/.
/// \note golden/chat_completions.txt was recorded from RestHandler::handle_openai_chat_completion
///       as it was before the chat core and its sinks existed (origin/main 1dfa10bf), driven by an
///       engine that writes a fixed text in fixed pieces. Record it again only when the shape of
///       the Chat Completions response is changed on purpose. No NPU needed.
#include "chat_core.hpp"
#include "protocol_openai_chat.hpp"

#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;

/// \brief The values that differ between two runs of the same code, replaced as the recording has them
std::string normalize(std::string s) {
    s = std::regex_replace(s, std::regex("chatcmpl-[0-9a-f]{24}"), "chatcmpl-ID");
    s = std::regex_replace(s, std::regex("\"created\":[0-9]+"), "\"created\":0");
    s = std::regex_replace(s, std::regex("fp_[0-9a-f]{16}"), "fp_ID");
    s = std::regex_replace(s, std::regex("call_[A-Za-z0-9]{24}"), "call_ID");
    s = std::regex_replace(s, std::regex("\"load_duration\":[-+.eE0-9]+"), "\"load_duration\":0");
    return s;
}

/// \brief What the scripted model of the recording did in one case
struct case_t {
    std::string name;
    bool tools = false;                  // the request offered get_weather
    std::vector<std::string> chunks;     // what the model wrote, one flush each
    chat_finish stop = chat_finish::STOP;  // why it stopped, before a tool call is taken into account
    bool cancel_in_prefill = false;
    bool prefix_cache = false;
    int fail_at = -1;                    // the model failed before writing this chunk
    bool fail_in_prefill = false;        // ... or the prompt could not be fed
    chat_error error;                    // what it failed with
    bool logprobs = false;               // the request asked for logprobs
};

const json TOOLS = json::array({{{"type", "function"}, {"function", {{"name", "get_weather"},
    {"parameters", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}}}}}}});

std::vector<case_t> cases() {
    std::vector<case_t> all;
    case_t c;

    c = {}; c.name = "text"; c.chunks = {"Hel", "lo", " wor", "ld", "!"};
    all.push_back(c);

    c = {}; c.name = "reasoning_text";
    c.chunks = {"<think>", "\nLet me", " think.", "\n</think>", "\n\n", "The answer", " is 4."};
    all.push_back(c);

    c = {}; c.name = "tool_call"; c.tools = true;
    c.chunks = {"<tool_call>", "\n{\"name\": \"get_", "weather\", \"arguments\": {\"city\":", " \"Paris\"}}", "\n</tool_call>"};
    all.push_back(c);

    c = {}; c.name = "two_tool_calls"; c.tools = true;
    c.chunks = {"I will look.\n", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
                "\n<tool_call>\n{\"name\": \"get_weather\", ", "\"arguments\": {\"city\": \"Rome\"}}\n</tool_call>"};
    all.push_back(c);

    // characters that arrive in pieces, and one the model never finishes
    c = {}; c.name = "utf8_split"; c.chunks = {"caf", "\xC3", "\xA9", " \xE2\x82", "\xAC", " \xE2"};
    all.push_back(c);

    c = {}; c.name = "length"; c.chunks = {"Hel", "lo"}; c.stop = chat_finish::LENGTH;
    all.push_back(c);

    // a call the token limit cut off is not reported as a finished call
    c = {}; c.name = "tool_call_length"; c.tools = true; c.stop = chat_finish::LENGTH;
    c.chunks = {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"};
    all.push_back(c);

    c = {}; c.name = "cancel_in_generation"; c.chunks = {"Hel", "lo"}; c.stop = chat_finish::CANCEL;
    all.push_back(c);

    c = {}; c.name = "prefix_cache"; c.chunks = {"Hel", "lo"}; c.prefix_cache = true;
    all.push_back(c);

    // tool markup is content when the request offered no tools
    // only content has logprobs: the reasoning's tokens are left out
    c = {}; c.name = "logprobs"; c.logprobs = true; c.chunks = {"<think>", "hm", "</think>", "Hel", "lo"};
    all.push_back(c);

    c = {}; c.name = "tools_not_offered";
    c.chunks = {"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"};
    all.push_back(c);

    c = {}; c.name = "cancel_in_prefill"; c.cancel_in_prefill = true; c.stop = chat_finish::CANCEL;
    all.push_back(c);

    c = {}; c.name = "prefill_failed_400"; c.fail_in_prefill = true;
    c.error = {400, "invalid_request_error", "the prompt is longer than the context", "", std::nullopt};
    all.push_back(c);

    // an error after chunks went out is one more event, and the stream is still ended
    c = {}; c.name = "generate_throw_mid"; c.chunks = {"Hel", "lo", " wor"}; c.fail_at = 2;
    c.error = {500, "server_error", "the scripted model threw", "", std::nullopt};
    all.push_back(c);

    c = {}; c.name = "generate_model_error_mid"; c.chunks = {"Hel", "lo", " wor"}; c.fail_at = 2;
    c.error = {503, "model_error", "the scripted model failed", "", true};
    all.push_back(c);
    return all;
}

/// \brief One case through the sink, the way RestHandler::run_chat_turn drives it; every send becomes a line
std::string run(const case_t& c, bool stream, bool include_usage) {
    std::ostringstream out;
    auto send_response = [&](const json& response) {
        out << "response: " << normalize(response.dump()) << "\n";
    };
    auto send_streaming_response = [&](const json& data, bool is_final) {
        const std::string text = data.get<std::string>();
        out << (is_final ? "final:    " : "chunk:    ") << normalize(text);
        if (text.empty() || text.back() != '\n') out << "\n";
    };
    openai_chat_sink sink(send_response, send_streaming_response, include_usage);

    const chat_format_spec spec = chat_format_make(chat_format_kind::HERMES_JSON, true);
    const json tools = c.tools ? TOOLS : json::array();
    chat_parse_options options;
    options.tools = &tools;
    options.parse_tool_calls = c.tools;
    const std::string prefill = chat_format_prefill(spec, "<|im_start|>assistant\n", false, options.parse_tool_calls);

    prefix_cache_report_t prefix_cache;
    prefix_cache.valid = true;
    prefix_cache.hit = true;
    prefix_cache.slot = 3;
    prefix_cache.matched_tokens = 8;
    prefix_cache.prompt_tokens = 37;
    prefix_cache.saved.push_back({4, false, 40, 8, 4096});
    prefix_cache.slots_used = 2;
    prefix_cache.slots_max = 8;
    prefix_cache.bytes_used = 8192;
    prefix_cache.bytes_budget = 1 << 20;

    auto turn_end = [&](const chat_msg& message, bool generated) {
        chat_turn_end end;
        end.finish = !message.tool_calls.empty() && c.stop == chat_finish::STOP ? chat_finish::TOOL_CALLS : c.stop;
        end.generated = generated;
        end.message = &message;
        if (generated) {
            end.prompt_tokens = 40;
            end.cached_prompt_tokens = 8;
            end.generated_tokens = static_cast<int>(c.chunks.size());
            end.prefill_duration = 250'000'000;
            end.decoding_duration = 500'000'000;
        }
        end.context_tokens = 123;
        end.context_capacity = 4096;
        end.prefix_cache = c.prefix_cache ? &prefix_cache : nullptr;
        return end;
    };

    sink.on_begin({"model-faker", stream});
    chat_turn_ostream ostream(spec, options, sink);
    if (c.cancel_in_prefill) {
        sink.on_end(turn_end(stream ? ostream.finish() : chat_msg(), false));
        return out.str();
    }
    if (c.fail_in_prefill) {
        sink.on_error(c.error);
        return out.str();
    }
    chat_turn_start start;
    start.prompt_tokens = 40;
    start.cached_prompt_tokens = 8;
    if (c.logprobs) {
        // what chat_logprobs_aligner hands out for this text: chunk i is token i, logprob -0.25 * (i + 1)
        start.content_logprobs = [&c](const std::string& content) {
            json entries = json::array();
            size_t at = 0;
            for (size_t i = 3; i < c.chunks.size() && at < content.size(); i++) {
                if (content.compare(at, c.chunks[i].size(), c.chunks[i]) != 0) continue;
                json bytes = json::array();
                for (unsigned char byte : c.chunks[i]) bytes.push_back(byte);
                entries.push_back({{"token", c.chunks[i]}, {"logprob", -0.25 * static_cast<double>(i + 1)},
                                   {"bytes", bytes}, {"top_logprobs", json::array()}});
                at += c.chunks[i].size();
            }
            return entries;
        };
    }
    ostream.start(prefill);
    sink.on_start(start);

    std::string text;
    for (size_t i = 0; i < c.chunks.size(); i++) {
        if (static_cast<int>(i) == c.fail_at) {
            sink.on_error(c.error);
            return out.str();
        }
        if (stream) ostream << c.chunks[i] << std::flush;
        text += c.chunks[i];
    }
    if (stream) {
        sink.on_end(turn_end(ostream.finish(), true));
    }
    else {
        chat_msg message = chat_parse(spec, prefill + sanitize_utf8(text), false, options);
        for (chat_tool_call& call : message.tool_calls) {
            call.id = chat_new_tool_call_id();
        }
        sink.on_end(turn_end(message, true));
    }
    return out.str();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str(), out;
    // a checkout may have turned the line ends into CRLF; no recorded line holds a bare CR
    for (char ch : text) {
        if (ch != '\r') out += ch;
    }
    return out;
}

/// \brief The recording, block by block: "### name [mode]" to the next one
std::vector<std::pair<std::string, std::string>> blocks(const std::string& text) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t pos = text.find("### ");
    while (pos != std::string::npos) {
        const size_t next = text.find("\n### ", pos);
        const std::string block = text.substr(pos, next == std::string::npos ? std::string::npos : next + 1 - pos);
        out.push_back({block.substr(0, block.find('\n')), block});
        pos = next == std::string::npos ? next : next + 1;
    }
    return out;
}
}  // namespace

int main(int argc, char* argv[]) {
    const std::string golden_path = argc > 1 ? argv[1] : std::string(GOLDEN_DIR) + "/chat_completions.txt";
    int failures = 0;
    try {
        const auto golden = blocks(read_file(golden_path));
        std::vector<std::pair<std::string, std::string>> actual;
        for (const case_t& c : cases()) {
            actual.push_back({"### " + c.name + " [nstream]", "### " + c.name + " [nstream]\n" + run(c, false, false) + "\n"});
            actual.push_back({"### " + c.name + " [stream]", "### " + c.name + " [stream]\n" + run(c, true, false) + "\n"});
            actual.push_back({"### " + c.name + " [stream+usage]", "### " + c.name + " [stream+usage]\n" + run(c, true, true) + "\n"});
        }
        if (golden.size() != actual.size()) {
            std::printf("  FAIL: the recording has %zu blocks, the test wrote %zu\n", golden.size(), actual.size());
            failures++;
        }
        for (size_t i = 0; i < golden.size() && i < actual.size(); i++) {
            if (golden[i].second != actual[i].second) {
                std::printf("  FAIL %s\n    got:      [%s]\n    expected: [%s]\n", actual[i].first.c_str(),
                            actual[i].second.c_str(), golden[i].second.c_str());
                failures++;
            }
        }
        std::printf("%zu blocks compared with %s\n", actual.size(), golden_path.c_str());

        // an error names its field, and a model error says whether the conversation survived
        const json with_param = openai_error_json({404, "invalid_request_error", "no such model", "model", std::nullopt});
        if (with_param.dump() != R"({"error":{"message":"no such model","type":"invalid_request_error","param":"model","code":404}})") {
            std::printf("  FAIL error with param: %s\n", with_param.dump().c_str());
            failures++;
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        failures++;
    }
    if (failures == 0) {
        std::printf("chat protocol tests passed\n");
        return 0;
    }
    std::printf("%d chat protocol check(s) failed\n", failures);
    return 1;
}
