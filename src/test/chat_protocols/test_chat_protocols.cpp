/// \file test_chat_protocols.cpp
/// \brief What a chat turn looks like on the wire, without a model: the Chat Completions sink is
///        fed the same turns the handler was, and must write the bytes recorded in golden/.
/// \note golden/chat_completions.txt was recorded from RestHandler::handle_openai_chat_completion
///       as it was before the chat core and its sinks existed (origin/main 1dfa10bf), driven by an
///       engine that writes a fixed text in fixed pieces. Record it again only when the shape of
///       the Chat Completions response is changed on purpose. No NPU needed.
#include "chat_core.hpp"
#include "protocol_openai_chat.hpp"
#include "protocol_responses.hpp"

#include <cstdio>
#include <fstream>
#include <optional>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

    // only content has logprobs: the reasoning's tokens are left out
    c = {}; c.name = "logprobs"; c.logprobs = true; c.chunks = {"<think>", "hm", "</think>", "Hel", "lo"};
    all.push_back(c);

    // tool markup is content when the request offered no tools
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

/// \brief Fails the test with what was expected and what came
int check(const std::string& what, const std::string& got, const std::string& expected) {
    if (got == expected) return 0;
    std::printf("  FAIL %s\n    got:      [%s]\n    expected: [%s]\n", what.c_str(), got.c_str(), expected.c_str());
    return 1;
}

/// \brief Where an error goes once the turn is under way. Not in the recording, because the
///        scripted model cannot make cleaning up fail; the expected lines are those of
///        send_stream_error and the outer catch of handle_openai_chat_completion at 1dfa10bf.
int error_routing() {
    int failures = 0;
    std::string out;
    bool throw_on_error_event = false;
    auto send_response = [&](const json& response) { out += "response " + response.dump() + "\n"; };
    auto send_streaming_response = [&](const json& data, bool is_final) {
        const std::string text = data.get<std::string>();
        if (throw_on_error_event && text.find("\"error\"") != std::string::npos) {
            throw std::runtime_error("the client is gone");
        }
        out += (is_final ? "final " : "chunk ") + text.substr(0, text.find('\n')) + "\n";
    };
    const chat_error first{500, "server_error", "first", "", std::nullopt};
    const chat_error second{500, "server_error", "second", "", std::nullopt};
    const std::string first_event = "chunk data: " + openai_error_json(first).dump() + "\nfinal data: [DONE]\n";
    const std::string second_object = "response " + openai_error_json(second).dump() + "\n";
    chat_msg_delta hi;
    hi.content = "Hi";
    chat_msg message;
    message.content = "Hi";
    chat_turn_end end;
    end.message = &message;

    // no chunk out yet: the error object alone
    {
        out.clear();
        openai_chat_sink sink(send_response, send_streaming_response, false);
        sink.on_begin({"m", true});
        sink.on_start({});
        sink.on_error(first);
        failures += check("error before any chunk", out, "response " + openai_error_json(first).dump() + "\n");
    }
    // the stream was ended and cleaning up failed: one more event and [DONE], as before
    {
        out.clear();
        openai_chat_sink sink(send_response, send_streaming_response, false);
        sink.on_begin({"m", true});
        sink.on_start({});
        sink.on_end(end);
        const size_t ended = out.size();
        sink.on_error(first);
        failures += check("error after the stream ended", out.substr(ended), first_event);
    }
    // an error after an error is the error object alone
    {
        out.clear();
        openai_chat_sink sink(send_response, send_streaming_response, false);
        sink.on_begin({"m", true});
        sink.on_start({});
        sink.on_delta(hi);
        const size_t streamed = out.size();
        sink.on_error(first);
        sink.on_error(second);
        failures += check("error after an error", out.substr(streamed), first_event + second_object);
    }
    // writing the error failed: the error about that is the error object alone
    {
        out.clear();
        openai_chat_sink sink(send_response, send_streaming_response, false);
        sink.on_begin({"m", true});
        sink.on_start({});
        sink.on_delta(hi);
        const size_t streamed = out.size();
        throw_on_error_event = true;
        try {
            sink.on_error(first);
        } catch (const std::exception&) {
        }
        throw_on_error_event = false;
        sink.on_error(second);
        failures += check("error while writing an error", out.substr(streamed), second_object);
    }
    return failures;
}

/// \brief The flush hook sees each flush's bytes, the incomplete ones too, before the parser does
int flush_hook() {
    struct log_sink : chat_sink {
        std::vector<std::string>& log;
        explicit log_sink(std::vector<std::string>& log) : log(log) {}
        void on_begin(const chat_turn_begin&) override {}
        void on_start(const chat_turn_start&) override {}
        void on_delta(const chat_msg_delta& d) override { log.push_back("delta " + d.content); }
        void on_end(const chat_turn_end&) override {}
        void on_error(const chat_error&) override {}
    };
    std::vector<std::string> log;
    log_sink sink(log);
    const chat_format_spec spec = chat_format_make(chat_format_kind::HERMES_JSON, true);
    const json tools = json::array();
    chat_parse_options options;
    options.tools = &tools;
    chat_turn_ostream ostream(spec, options, sink);
    ostream.start(chat_format_prefill(spec, "<|im_start|>assistant\n", false, false),
                  [&log](const std::string& written) { log.push_back("hook " + written); });
    ostream << "Hel" << std::flush;
    ostream << "lo \xE2" << std::flush;
    ostream.finish();

    int failures = 0;
    failures += check("flush hook, first flush", log.size() >= 2 ? log[0] + "|" + log[1] : "", "hook Hel|delta Hel");
    std::string hooked;
    for (const std::string& line : log) {
        if (line.rfind("hook ", 0) == 0) hooked += line.substr(5);
    }
    failures += check("flush hook, every byte", hooked, "Hello \xE2");
    return failures;
}
}  // namespace

int test_anthropic();
int main(int argc, char* argv[]) {
    const std::string golden_path = argc > 1 ? argv[1] : std::string(GOLDEN_DIR) + "/chat_completions.txt";
    int failures = test_anthropic();
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
        failures += check("model error that kept the conversation",
                          openai_error_json({503, "model_error", "busy", "", false}).dump(),
                          R"({"error":{"message":"busy","type":"service_unavailable_error","param":null,"code":503,"session_cleared":false}})");
        {
            // Invalid UTF-8 bytes in streamed delta must replace cleanly instead of throwing type_error(316).
            std::string raw_delta_out;
            openai_chat_sink sink([](const json&) {},
                                  [&](const json& d, bool) { raw_delta_out += d.get<std::string>(); }, false);
            sink.on_begin({"m", true});
            sink.on_start({});
            chat_msg_delta bad_utf8;
            bad_utf8.content = std::string("ok\xFF");
            sink.on_delta(bad_utf8);
            if (raw_delta_out.find("ok\xEF\xBF\xBD") == std::string::npos) {
                std::printf("  FAIL invalid UTF-8 delta replacement: %s\n", raw_delta_out.c_str());
                failures++;
            }
        }
        failures += error_routing();
        failures += flush_hook();

        // Feed the same ordered text and tool fragments to the streaming sink, then
        // compare the terminal object with the one-shot output (IDs vary per request).
        {
            json request = {{"model","m"},{"stream",true}}, final;
            std::vector<json> frames;
            int last_count = 0;
            auto stream = [&](const json& raw, bool last) {
                const auto wire = raw.get<std::string>();
                const auto start = wire.find("\ndata: ");
                if (start == std::string::npos) throw std::runtime_error("missing SSE data");
                auto e = json::parse(wire.substr(start + 7));
                if (wire.substr(0, start) != "event: " + e.at("type").get<std::string>() ||
                    e.at("sequence_number") != frames.size()) throw std::runtime_error("bad SSE envelope");
                frames.push_back(std::move(e));
                if (last) ++last_count;
            };
            chat_msg m;
            m.reasoning = "why"; m.content = "answer"; m.tool_calls = {{"call_A","read","{\"x\":1}"}};
            chat_turn_end end; end.message = &m; end.finish = chat_finish::TOOL_CALLS;
            flm::responses::sink streamed(request, [&](const json& j) { final = j; }, stream);
            auto generate = [&](flm::responses::sink& target) {
                target.on_begin({"m",true}); target.on_start({});
                chat_msg_delta d; d.reasoning = m.reasoning; target.on_delta(d);
                d = {}; d.content = m.content; target.on_delta(d);
                d = {}; d.tool_index = 0; d.tool_id = "call_A"; d.tool_name = "read"; d.tool_arguments = "{\"x\":";
                target.on_delta(d);
                d = {}; d.tool_index = 0; d.tool_arguments = "1}"; target.on_delta(d);
                d = {}; d.content = "after call"; target.on_delta(d);
                target.on_end(end);
            };
            generate(streamed); streamed.on_error({}); // no second terminal frame
            auto items = frames.back().at("response").at("output");
            failures += check("responses terminal", frames.back()["type"].dump(), "\"response.completed\"");
            failures += check("one final frame", std::to_string(last_count), "1");
            // Reasoning, text and a call in the order the model wrote them; each item is
            // closed only at the end, so a call cannot run before the turn is known to finish.
            std::string types;
            for (const auto& frame : frames) types += frame.at("type").get<std::string>().substr(9) + " ";
            failures += check("responses event order", types,
                "created in_progress "
                "output_item.added reasoning_summary_part.added reasoning_summary_text.delta "
                "output_item.added content_part.added output_text.delta "
                "output_item.added function_call_arguments.delta function_call_arguments.delta "
                "output_item.added content_part.added output_text.delta "
                "reasoning_summary_text.done reasoning_summary_part.done output_item.done "
                "output_text.done content_part.done output_item.done "
                "function_call_arguments.done output_item.done "
                "output_text.done content_part.done output_item.done completed ");
            auto first = [&](const std::string& type, size_t skip = 0) {
                for (const auto& frame : frames)
                    if (frame["type"] == "response." + type && skip-- == 0) return frame;
                throw std::runtime_error("no " + type + " event");
            };
            auto keys = [](const json& j) {
                std::string out;
                for (auto it = j.begin(); it != j.end(); ++it) out += it.key() + ",";
                return out;
            };
            auto without_id = [](json item) { item.erase("id"); return item; };
            failures += check("reasoning delta shape", keys(first("reasoning_summary_text.delta")),
                              "item_id,output_index,summary_index,delta,type,sequence_number,");
            failures += check("reasoning part", first("reasoning_summary_part.added").at("part").dump(),
                              R"({"type":"summary_text","text":""})");
            failures += check("reasoning item", without_id(first("output_item.done").at("item")).dump(),
                              R"({"type":"reasoning","summary":[{"type":"summary_text","text":"why"}],"status":"completed"})");
            failures += check("text delta shape", keys(first("output_text.delta")),
                              "item_id,output_index,content_index,delta,logprobs,type,sequence_number,");
            failures += check("text part", first("content_part.done").at("part").dump(),
                              R"({"type":"output_text","text":"answer","annotations":[]})");
            failures += check("call added", without_id(first("output_item.added", 2).at("item")).dump(),
                              R"({"type":"function_call","name":"read","call_id":"call_A","arguments":"","status":"in_progress"})");
            failures += check("call delta shape", keys(first("function_call_arguments.delta")),
                              "item_id,output_index,delta,type,sequence_number,");
            const auto call_done = first("function_call_arguments.done");
            failures += check("call done shape", keys(call_done) + call_done.at("arguments").get<std::string>(),
                              "item_id,output_index,name,arguments,type,sequence_number,{\"x\":1}");
            failures += check("call item", without_id(first("output_item.done", 2).at("item")).dump(),
                              R"({"type":"function_call","name":"read","call_id":"call_A","arguments":"{\"x\":1}","status":"completed"})");
            failures += check("terminal usage shape", keys(frames.back().at("response").at("usage")),
                              "input_tokens,output_tokens,total_tokens,input_tokens_details,output_tokens_details,");
            std::string arguments;
            for (const auto& frame : frames) {
                if (frame["type"] == "response.function_call_arguments.delta") {
                    const auto i = frame.at("output_index").get<size_t>();
                    failures += check("stable tool item",frame.at("item_id").dump(),items.at(i)["id"].dump());
                    arguments += frame.at("delta").get<std::string>();
                }
            }
            failures += check("reconstructed arguments", arguments, m.tool_calls[0].arguments);
            json nonstream = {{"model","m"}};
            flm::responses::sink one(nonstream, [&](const json& j) { final = j; }, stream);
            generate(one);
            failures += check("text after call", items.back().at("content")[0].at("text").dump(), "\"after call\"");
            for (auto& it : items) it.erase("id");
            for (auto& it : final.at("output")) it.erase("id");
            failures += check("stream/one-shot item parity", items.dump(), final.at("output").dump());
            for (const bool error : {false, true}) {
                frames.clear(); last_count = 0;
                flm::responses::sink partial(request, [&](const json& j) { final = j; }, stream);
                partial.on_begin({"m",true}); partial.on_start({});
                chat_msg_delta d; d.content = "partial"; partial.on_delta(d);
                d = {}; d.tool_index = 0; d.tool_name = "read"; d.tool_id = "call_A";
                d.tool_arguments = "{\"x\":"; partial.on_delta(d);
                if (error) partial.on_error({503,"model_error","busy","",true});
                else { end.finish = chat_finish::LENGTH; partial.on_end(end); }
                partial.on_error({});
                failures += check("only one terminal after error/limit", std::to_string(last_count), "1");
                for (const auto& e : frames)
                    if (e["type"] == "response.function_call_arguments.done" ||
                        (e["type"] == "response.output_item.done" && e["item"]["type"] == "function_call"))
                        throw std::runtime_error("partial function call advertised as done");
                failures += check("incomplete/failed terminal", frames.back()["type"].get<std::string>(),
                                  error ? "response.failed" : "response.incomplete");
                failures += check("partial item",frames.back()["response"]["output"][0]["status"].dump(),"\"incomplete\"");
            }
            frames.clear();
            flm::responses::sink cancelled(request, [&](const json& j) { final = j; }, stream);
            cancelled.on_begin({"m",true}); end.finish = chat_finish::CANCEL; end.generated = false;
            cancelled.on_end(end);
            failures += check("cancel start clean", frames[0]["response"]["error"].dump(), "null");
            failures += check("cancel start no usage", frames[0]["response"]["usage"].dump(), "null");
            failures += check("cancel terminal", frames.back()["type"].dump(), "\"response.failed\"");
        }

        const auto simple = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":"hello","reasoning":{"effort":"low"},"max_output_tokens":42,
            "text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"}}}
        })"));
        failures += check("string input", simple.at("messages").dump(), R"([{"role":"user","content":"hello"}])");
        failures += check("effort", simple.at("reasoning_effort").dump(), "\"low\"");
        failures += check("output budget", simple.at("max_completion_tokens").dump(), "42");
        failures += check("response schema", simple.at("response_format").at("json_schema").dump(),
                          R"({"name":"answer","schema":{"type":"object"}})");
        const auto defaults = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":"x","store":null,"temperature":null,"parallel_tool_calls":null,
            "top_p":null,"max_output_tokens":null,"reasoning":{"effort":null}
        })"));
        failures += check("nullable generation controls", defaults.dump(),
                          R"({"model":"m","stream":true,"messages":[{"role":"user","content":"x"}]})");
        // null is unset for every optional field, and so is the default of a field FLM cannot
        // honor; fields clients send by default (include, metadata, ...) are ignored.
        const auto unset = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":[{"type":null,"role":"user","content":"x"}],"stream":null,"instructions":null,
            "text":null,"tools":null,"tool_choice":null,"reasoning":null,"previous_response_id":null,
            "conversation":null,"background":false,"truncation":"disabled","top_logprobs":0,
            "max_tool_calls":null,"prompt":null,"include":["reasoning.encrypted_content"],"metadata":{},
            "service_tier":"auto","prompt_cache_key":"k","user":"u","stream_options":null
        })"));
        failures += check("null means unset", unset.dump(),
                          R"({"model":"m","stream":true,"messages":[{"role":"user","content":"x"}]})");
        const auto unset_parts = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":"x","text":{"format":null,"verbosity":"low"},
            "tools":[{"type":"function","name":"read","parameters":null}]
        })"));
        failures += check("null format and parameters", unset_parts.contains("response_format") ? "format" :
                          unset_parts.at("tools")[0].at("function").dump(), R"({"name":"read","parameters":null})");
        {
            const auto request = json::parse(R"({"model":"m","input":"x","tools":null,"tool_choice":null,
                "parallel_tool_calls":null})");
            json out;
            flm::responses::sink sink(request, [&](const json& j) { out = j; }, [](const json&, bool) {});
            chat_msg m; chat_turn_end end; end.message = &m;
            sink.on_begin({"m",true}); sink.on_start({}); sink.on_end(end);
            failures += check("null echoes the defaults", out.at("tools").dump() + out.at("tool_choice").dump() +
                              out.at("parallel_tool_calls").dump(), "[]\"auto\"true");
        }
        // A field that would change the answer is refused, naming it, before anything streams.
        for (const auto& [field, value] : std::vector<std::pair<std::string, json>>{
                 {"previous_response_id", "resp_1"}, {"conversation", "conv_1"}, {"background", true},
                 {"truncation", "auto"}, {"top_logprobs", 2}, {"max_tool_calls", 1},
                 {"prompt", json{{"id", "pmpt_1"}}}}) {
            const json request = {{"model","m"},{"input","x"},{"stream",true},{field,value}};
            json sent;
            int frames_sent = 0;
            flm::responses::sink sink(request, [&](const json& j) { sent = j; },
                                      [&](const json&, bool) { ++frames_sent; });
            try {
                flm::responses::to_chat_request(request);
            } catch (const flm::responses::request_error& e) { // as handle_openai_responses does
                sink.on_error({400, "invalid_request_error", e.what(), e.param, std::nullopt});
            }
            const auto err = flm::error_model::extract_error(sent);
            failures += check("refused " + field, err ? std::to_string(err->status) + " " + err->type + " " +
                              err->param + " " + std::to_string(frames_sent) : "accepted",
                              "400 invalid_request_error " + field + " 0");
        }

        // A byte-fallback token can leave invalid UTF-8 in text or arguments: the stream and the
        // one-shot object (written by the server with dump_json) stay valid JSON, with U+FFFD instead.
        for (const bool streamed : {true, false}) {
            const std::string bad = "ok \xff\xfe";
            const json request = {{"model","m"},{"stream",streamed}};
            std::string wire;
            json one;
            flm::responses::sink sink(request, [&](const json& j) { one = j; },
                                      [&](const json& raw, bool) { wire += raw.get<std::string>(); });
            chat_msg m; chat_turn_end end; end.message = &m; end.finish = chat_finish::TOOL_CALLS;
            sink.on_begin({"m",true}); sink.on_start({});
            chat_msg_delta d; d.reasoning = bad; sink.on_delta(d);
            d = {}; d.content = bad; sink.on_delta(d);
            d = {}; d.tool_index = 0; d.tool_id = "call_A"; d.tool_name = "read";
            d.tool_arguments = "{\"x\":\"" + bad + "\"}"; sink.on_delta(d);
            sink.on_end(end);
            json terminal;
            if (streamed) {
                size_t count = 0;
                for (size_t pos = 0; pos < wire.size(); ++count) {
                    const auto data = wire.find("\ndata: ", pos), next = wire.find("\n\n", pos);
                    if (data == std::string::npos || next == std::string::npos) throw std::runtime_error("bad SSE frame");
                    auto e = json::parse(wire.substr(data + 7, next - data - 7));
                    if (e.contains("response")) terminal = std::move(e["response"]);
                    pos = next + 2;
                }
                failures += check("utf-8 stream frames", std::to_string(count), "19");
            } else {
                bool threw = false;
                try { (void)one.dump(); } catch (const json::type_error&) { threw = true; }
                failures += check("plain dump would throw", threw ? "throws" : "no throw", "throws");
                terminal = json::parse(flm::error_model::dump_json(one));
            }
            const auto& output = terminal.at("output");
            failures += check(std::string("utf-8 replaced, stream=") + (streamed ? "1" : "0"),
                              output.at(0).at("summary")[0].at("text").get<std::string>() + "|" +
                              output.at(1).at("content")[0].at("text").get<std::string>() + "|" +
                              output.at(2).at("arguments").get<std::string>() + "|" + terminal.at("status").get<std::string>(),
                              "ok \xEF\xBF\xBD\xEF\xBF\xBD|ok \xEF\xBF\xBD\xEF\xBF\xBD|{\"x\":\"ok \xEF\xBF\xBD\xEF\xBF\xBD\"}|completed");
        }
        const auto replay = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"why"}]},
            {"role":"assistant","content":[{"type":"input_text","text":"answer"}]},
            {"role":"user","content":"next"}]})")).at("messages");
        failures += check("explicit assistant history", replay[0].dump(),
                          R"({"role":"assistant","content":"answer","reasoning_content":"why"})");
        const auto turns = flm::responses::to_chat_request(json::parse(R"({
            "model":"m","input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"r1"}]},
            {"role":"assistant","content":"first"},{"type":"reasoning","summary":[{"type":"summary_text","text":"r2"}]},
            {"role":"assistant","content":"second"}]})")).at("messages");
        failures += check("distinct reasoning turns", turns[0].at("reasoning_content").dump() +
                          turns[1].at("reasoning_content").dump(), "\"r1\"\"r2\"");

        // Function result IDs, not the order of the results, bind them to calls.
        const auto history = json::parse(R"({
            "model":"m", "instructions":"Be brief.", "input":[
                {"role":"user","content":"Both places?"},
                {"type":"function_call","name":"weather","call_id":"call_A","arguments":"{\"city\":\"Paris\"}"},
                {"type":"reasoning","summary":[]},
                {"type":"function_call","name":"weather","call_id":"call_B","arguments":"{\"city\":\"Rome\"}"},
                {"type":"function_call_output","call_id":"call_B","output":"rain"},
                {"type":"function_call_output","call_id":"call_A","output":"sun"}
            ]})");
        const auto messages = flm::responses::to_chat_request(history).at("messages");
        failures += check("instructions", messages.at(0).at("content").get<std::string>(), "Be brief.");
        failures += check("user input", messages.at(1).at("content").get<std::string>(), "Both places?");
        std::vector<std::string> calls, results;
        for (const auto& message : messages) {
            if (message.contains("tool_calls")) {
                for (const auto& call : message.at("tool_calls"))
                    calls.push_back(call.at("id").get<std::string>() + ":" +
                                    call.at("function").at("arguments").get<std::string>());
            }
            if (message.at("role") == "tool")
                results.push_back(message.at("tool_call_id").get<std::string>() + ":" +
                                  message.at("content").get<std::string>());
        }
        failures += check("parallel calls", json(calls).dump(),
                          json::array({"call_A:{\"city\":\"Paris\"}", "call_B:{\"city\":\"Rome\"}"}).dump());
        failures += check("reversed tool outputs", json(results).dump(),
                          json::array({"call_A:sun", "call_B:rain"}).dump());
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
