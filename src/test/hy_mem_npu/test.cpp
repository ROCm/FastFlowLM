/// \file test.cpp
/// \brief Hy-Memory-30B-A3B end to end: text in, JSON out
/// \note Hy-Memory is the LLM behind the hy_memory engine: it gets one-shot
///       prompts and must answer with JSON. Three modes:
///         --messages <file>  one conversation (the OpenAI shape:
///                            [{"role": ..., "content": ...}, ...]) -> its JSON
///         --prompt-tokens N[,N...]  the same extraction task built to exactly
///                            that many prompt tokens, one run per length, then a
///                            summary table (--repeat N for the median of N runs;
///                            prefill speed moves ~20% run to run). The default
///                            messages file is ~100 tokens, too short to measure
///                            prefill with.
///         --chain            the engine's real two-call chain, as steps 1-2 of
///                            models_simple/verify_hy_mem_30A3B.py:
///                            EXTRACT (conversation -> new memories), then
///                            RECONCILE (new + existing memories -> ADD / UPDATE /
///                            SUPERSEDE ops). The prompt templates come from
///                            prompts/hy_memory_prompts.json
///                            (prompts/dump_hy_memory_prompts.py).
///       Each reply streams, then the JSON is pulled out of it the way
///       verify_hy_mem_30A3B.py does (a ```json fence if there is one, otherwise
///       the outermost {...} or [...]).
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <set>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hy_memory.hpp"
#include "AutoModel/flm_backend.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \note builtin_backends.cpp names every engine and so links every prebuilt
///       library; this harness registers just the one it drives.
namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("hy-memory", kFlmBackendId, flm_factory<hy_mem_npu>());
}
}  // namespace flm::backend

/// \brief HyMemory with its tokenizer in reach (it is protected), so --prompt-tokens
///        can measure a prompt the way the engine will see it
class HyMemoryProbe : public HyMemory {
public:
    using HyMemory::HyMemory;
    std::vector<int> encode(const std::string& text) { return this->tokenizer->encode(text); }
};

/// \brief the JSON in a reply, or nothing if none parses
static std::optional<nlohmann::ordered_json> parse_json_block(const std::string& text) {
    std::string body;
    const size_t fence = text.find("```");
    if (fence != std::string::npos) {
        // ```json\n ... ``` (the language tag is optional)
        size_t start = fence + 3;
        if (text.compare(start, 4, "json") == 0) start += 4;
        const size_t end = text.find("```", start);
        body = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    } else {
        const size_t obj = text.find('{'), arr = text.find('[');
        const size_t start = std::min(obj, arr);                  // npos is the largest size_t
        const size_t end_obj = text.rfind('}'), end_arr = text.rfind(']');
        // the later of the two closers, ignoring a missing one
        size_t end = std::string::npos;
        if (end_obj != std::string::npos) end = end_obj;
        if (end_arr != std::string::npos && (end == std::string::npos || end_arr > end)) end = end_arr;
        if (start == std::string::npos || end == std::string::npos || end < start) return std::nullopt;
        body = text.substr(start, end - start + 1);
    }
    nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(body, nullptr, false);   // no throw
    if (!parsed.is_discarded()) return parsed;

    // Fallback for one known shape: objects separated by commas with no [ ] around
    // them ({...},\n{...}), which the model gives for RECONCILE at times. The
    // hy_memory engine recovers such replies with json_repair; wrapping them in
    // [ ] is the narrow version of that. Said out loud, so it never hides anything.
    if (!body.empty() && body.front() == '{') {
        parsed = nlohmann::ordered_json::parse("[" + body + "]", nullptr, false);
        if (!parsed.is_discarded()) {
            std::cout << "NOTE: the reply is a list of objects without [ ]; parsed it wrapped in [ ] "
                         "(hy_memory's json_repair fallback would recover it too)" << std::endl;
            return parsed;
        }
    }
    return std::nullopt;
}

/// \brief a messages file: a bare array, or an object with a "messages" array
static nlohmann::ordered_json read_messages(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }
    nlohmann::ordered_json j = nlohmann::ordered_json::parse(f);
    if (j.is_object() && j.contains("messages")) j = j["messages"];
    if (!j.is_array()) {
        throw std::runtime_error(path + ": expected a list of {role, content} messages");
    }
    return j;
}

/// \brief Python's str.format for the prompt templates: {name} takes values[name],
///        {{ and }} are a literal { and }
static std::string py_format(const std::string& tmpl, const std::map<std::string, std::string>& values) {
    std::string out;
    for (size_t i = 0; i < tmpl.size(); i++) {
        const char c = tmpl[i];
        const bool doubled = i + 1 < tmpl.size() && tmpl[i + 1] == c;
        if ((c == '{' || c == '}') && doubled) {          // escaped brace
            out += c;
            i++;
        } else if (c == '{') {                            // a {name} field
            const size_t close = tmpl.find('}', i);
            const std::string name = tmpl.substr(i + 1, close - i - 1);
            if (close == std::string::npos || values.count(name) == 0) {
                throw std::runtime_error("prompt template field {" + name + "} has no value");
            }
            out += values.at(name);
            i = close;
        } else {
            out += c;
        }
    }
    return out;
}

/// \brief Python's json.dumps(x, ensure_ascii=False) without indent: ", " and ": "
///        separators (nlohmann's compact dump uses "," and ":")
static std::string py_json_dumps(const nlohmann::ordered_json& j) {
    if (j.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            s += (first ? "" : ", ") + nlohmann::ordered_json(it.key()).dump() + ": " + py_json_dumps(it.value());
            first = false;
        }
        return s + "}";
    }
    if (j.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < j.size(); i++) {
            s += (i ? ", " : "") + py_json_dumps(j[i]);
        }
        return s + "]";
    }
    return j.dump();   // scalars print the same; UTF-8 stays raw like ensure_ascii=False
}

/// \brief local time as strftime(fmt)
static std::string now_string(const char* fmt) {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), fmt, std::localtime(&t));
    return buf;
}

/// \brief --save: where each request's messages and reply go (empty = nowhere)
static std::string g_save_dir;

/// \brief one request through the model: prints the reply as it streams, returns it
/// \param step file name stem for --save: <step>_messages.json, <step>_reply.txt
///        (models_simple/ref_hy_mem_messages.py replays them through transformers)
/// \param out_meta if given, the request's token counts and timings land here
static std::string run_chat(AutoModel& chat, const std::string& step, const std::string& title,
                            const nlohmann::ordered_json& messages, int max_new_tokens,
                            chat_meta_info_t* out_meta = nullptr) {
    std::cout << "\n" << std::string(100, '=') << "\n" << title << "\n" << std::string(100, '=') << std::endl;
    if (!g_save_dir.empty()) {
        std::ofstream(g_save_dir + "/" + step + "_messages.json") << messages.dump(2) << std::endl;
    }
    chat_meta_info_t meta_info;
    lm_uniform_input_t input;
    input.messages = messages;
    chat.start_total_timer();
    if (!chat.insert(meta_info, input)) {
        throw std::runtime_error("insert failed");
    }
    std::string reply = chat.generate(meta_info, max_new_tokens, std::cout);
    chat.stop_total_timer();
    std::cout << "\n\n" << chat.show_profile() << std::endl;
    if (meta_info.generated_tokens >= max_new_tokens) {
        header_print("WARNING", "hit the " << max_new_tokens << "-token limit; the reply is probably cut off");
    }
    if (!g_save_dir.empty()) {
        std::ofstream(g_save_dir + "/" + step + "_reply.txt") << reply;
    }
    if (out_meta) *out_meta = meta_info;
    return reply;
}

// =============================================================================
// --prompt-tokens: the same task at a chosen prompt length
// =============================================================================
// The default messages file is ~100 tokens, which is too short to say anything
// about prefill speed: at that length almost all of the time is fixed overhead,
// while a 4K prompt reaches ~200 tok/s. Every prefill stage runs on the NPU at
// every length, so the curve here is the padding cost, not a change of path: a
// run is a whole number of 256-token rounds and an expert only sees about L/16
// of a chunk's tokens, so short prompts spend most of the array on padding.
// HY_MEM_MM_NPU_MIN (0 = off) can put short chunks' experts back on the host to
// compare.

/// \brief the system prompt of messages/model_card_demo.json
static const char* kSizedSystem =
    "Extract explicit user facts and preferences from the conversation. Write each memory as a "
    "concise, self-contained statement. Do not infer or invent information. Return only a JSON "
    "object with a \"memories\" list of strings.";

/// \brief the facts to extract -- the same ones the short demo uses, so a long
///        prompt should produce the same JSON as a short one
static const char* kSizedFacts =
    "My name is Alex. I live in Seattle and work as a software engineer. I mainly use Python and "
    "prefer concise answers with code examples. I'm vegetarian, so please keep that in mind when "
    "suggesting meals.";

/// \brief one earlier exchange with nothing worth remembering in it
/// \note Deliberately factless: it makes the prompt longer without changing the
///       answer, so the JSON stays comparable across lengths. The index keeps the
///       turns from being an exact repetition.
static std::string filler_turn(int i) {
    static const char* topics[] = {"the weather this week", "the bus timetable", "a film we mentioned",
                                   "the office coffee machine", "a crossword clue", "the lunch queue"};
    const std::string topic = topics[i % 6];
    return "User: Quick one, what did we say about " + topic + " earlier (note " + std::to_string(i) + ")?\n"
           "Assistant: Nothing worth writing down, we only touched on " + topic + " in passing.\n";
}

/// \brief a conversation whose templated form is exactly `target` tokens
/// \param tag makes the prompt unique per run, so the prefix cache cannot serve
///        part of it and leave the engine prefilling fewer tokens than asked
/// \return the messages, and the token count actually reached in *got
static nlohmann::ordered_json build_sized_prompt(HyMemoryProbe& chat, int target, const std::string& tag, int* got) {
    // The filler line is always here, even with no words on it: if it appeared only
    // once pad > 0 then the first word would also bring the line itself, the count
    // would jump by several tokens at once, and the search below would step over
    // its target and oscillate. With the line always present each word is one step.
    auto make = [&](int turns, int pad) {
        std::string convo = "[session " + tag + ", earlier conversation]\n";
        for (int i = 0; i < turns; i++) convo += filler_turn(i);
        convo += "User: Also, ignore this filler:";
        for (int i = 0; i < pad; i++) convo += " again";
        convo += "\n";
        convo += "\n[latest messages]\n" + std::string(kSizedFacts) + "\n";
        nlohmann::ordered_json m = nlohmann::ordered_json::array();
        m.push_back({{"role", "system"}, {"content", kSizedSystem}});
        m.push_back({{"role", "user"}, {"content", convo}});
        return m;
    };
    auto measure = [&](nlohmann::ordered_json m) { return (int)chat.encode(chat.apply_chat_template(m)).size(); };

    const int floor_tokens = measure(make(0, 0));
    if (target < floor_tokens) {
        throw std::runtime_error("--prompt-tokens " + std::to_string(target) + " is below the " +
                                 std::to_string(floor_tokens) + " tokens the task itself needs");
    }
    // coarse: whole exchanges (~20 tokens each) until one more would overshoot
    int turns = 0;
    while (measure(make(turns + 1, 0)) <= target) turns++;
    // fine: single filler words, each one token. Keep the closest attempt rather
    // than the last one, so an odd tokenization cannot leave us further away than
    // where we started.
    int pad = 0, cur = measure(make(turns, 0));
    int best_pad = pad, best_cur = cur;
    for (int step = 0; step < 100 && cur != target; step++) {
        const int want = pad + (target - cur);
        pad = want < 0 ? 0 : want;
        cur = measure(make(turns, pad));
        if (std::abs(cur - target) < std::abs(best_cur - target)) {
            best_pad = pad;
            best_cur = cur;
        }
    }
    *got = best_cur;
    return make(turns, best_pad);
}

/// \brief one run per requested prompt length, then a summary table
/// \param repeat runs per length; the table reports the median. Prefill speed here
///        varies by about +-20% run to run (the host stages share the machine with
///        everything else), which is wider than the differences this is usually
///        used to look at -- one run per length is not enough to compare two.
static int run_sized(HyMemoryProbe& chat, const std::vector<int>& sizes, int max_new_tokens, int repeat) {
    struct row {
        int want, got, prompt_tokens, cached, generated;
        double prefill_ms, prefill_tps, decode_tps, e2e_s;
        bool json_ok;
    };
    std::vector<row> rows;
    int bad = 0;

    for (int want : sizes) {
        int got = 0;
        const std::string tag = std::to_string(want);
        nlohmann::ordered_json messages = build_sized_prompt(chat, want, tag, &got);
        if (got != want) {
            header_print("WARNING", "asked for " << want << " prompt tokens, built " << got
                         << " (the filler could not land on it exactly)");
        }
        std::vector<double> prefill_ms_runs, prefill_tps_runs, decode_tps_runs, e2e_s_runs;
        chat_meta_info_t meta{};
        for (int r = 0; r < repeat; r++) {
            const std::string title = "prompt of " + std::to_string(got) + " tokens"
                                    + (repeat > 1 ? "  [run " + std::to_string(r + 1) + " of " + std::to_string(repeat) + "]" : "");
            // end to end as the caller sees it: prefill, decode, sampling and the
            // glue in between, for this one request (the model is already loaded)
            const auto t_start = std::chrono::steady_clock::now();
            const std::string reply = run_chat(chat, "size_" + tag, title, messages, max_new_tokens, &meta);
            const double e2e_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

            std::optional<nlohmann::ordered_json> parsed = parse_json_block(reply);
            if (!parsed) {
                std::cout << "NOT VALID JSON" << std::endl;
                bad++;
            } else if (r == 0) {
                std::cout << parsed->dump(2) << std::endl;
            }
            const double prefill_ms = meta.prefill_duration / 1e6;
            const int fresh = meta.prompt_tokens - meta.cached_prompt_tokens;   // what prefill really ran
            prefill_ms_runs.push_back(prefill_ms);
            prefill_tps_runs.push_back(prefill_ms > 0 ? fresh / (prefill_ms / 1000.0) : 0.0);
            decode_tps_runs.push_back(meta.decoding_duration > 0 ? meta.generated_tokens / (meta.decoding_duration / 1e9) : 0.0);
            e2e_s_runs.push_back(e2e_s);
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        rows.push_back({want, got, meta.prompt_tokens, meta.cached_prompt_tokens, meta.generated_tokens,
                        median(prefill_ms_runs), median(prefill_tps_runs), median(decode_tps_runs),
                        median(e2e_s_runs), bad == 0});
    }

    std::cout << "\n" << std::string(100, '=') << "\nprompt size sweep\n" << std::string(100, '=') << std::endl;
    std::cout << std::left << std::setw(8) << "want" << std::setw(8) << "built" << std::setw(9) << "prompt"
              << std::setw(8) << "cached" << std::setw(12) << "prefill ms" << std::setw(14) << "prefill tok/s"
              << std::setw(13) << "decode tok/s" << std::setw(7) << "gen" << std::setw(10) << "e2e s" << "JSON" << std::endl;
    for (const row& r : rows) {
        std::cout << std::left << std::setw(8) << r.want << std::setw(8) << r.got << std::setw(9) << r.prompt_tokens
                  << std::setw(8) << r.cached << std::setw(12) << std::fixed << std::setprecision(1) << r.prefill_ms
                  << std::setw(14) << std::setprecision(1) << r.prefill_tps
                  << std::setw(13) << std::setprecision(2) << r.decode_tps
                  << std::setw(7) << r.generated
                  << std::setw(10) << std::setprecision(2) << r.e2e_s << (r.json_ok ? "ok" : "BAD") << std::endl;
    }
    // A non-zero "cached" means the engine reused part of an earlier prompt, so it
    // prefilled fewer tokens than "prompt" and the row is not the length it claims.
    for (const row& r : rows) {
        if (r.cached != 0) {
            header_print("WARNING", "prompt of " << r.got << " reused " << r.cached
                         << " cached tokens, so prefill ran on " << (r.prompt_tokens - r.cached));
        }
    }
    std::cout << "\ne2e s = one whole request, prefill + decode + sampling, model already loaded;"
                 "\nit therefore moves with 'gen' as well as with the prompt length."
                 "\n(set HY_TIME_HOST=1 for the engine's own per-stage prefill breakdown)" << std::endl;
    return bad == 0 ? 0 : 1;
}

/// \brief steps 1-2 of verify_hy_mem_30A3B.py: the hy_memory engine's EXTRACT then
///        RECONCILE calls, with the engine defaults (English, few-shot off)
/// \return 0 if both replies parsed into the shape the engine accepts
static int run_chain(AutoModel& chat, const std::string& prompts_path) {
    std::ifstream pf(prompts_path);
    if (!pf.is_open()) {
        header_print("error", "cannot open " << prompts_path << " (run prompts/dump_hy_memory_prompts.py)");
        return 1;
    }
    const nlohmann::ordered_json prompts = nlohmann::ordered_json::parse(pf);
    const std::string today = now_string("%Y-%m-%d");
    const std::string now_minute = now_string("%Y-%m-%d %H:%M");

    // ---- STEP 1: EXTRACT ----
    // The conversation being written into memory; the writer dumps request.messages
    // as a JSON list of {role, content} -- that string is "New Messages".
    nlohmann::ordered_json new_messages = nlohmann::ordered_json::array();
    new_messages.push_back({{"role", "user"}, {"content",
        "My name is Alex. I moved from Portland to Seattle last month and now work as a "
        "software engineer. I mainly use Python and prefer concise answers with code "
        "examples. I'm vegetarian. I plan to run the Seattle marathon in November."}});
    const std::string extract_user_prompt = py_format(prompts["extract"].get<std::string>(), {
        {"content", py_json_dumps(new_messages)},
        {"current_date", today},       // system date
        {"memory_at", today},          // add() without memory_at uses "now"
        {"last_messages", "(none)"},   // no earlier turns in this session
    });
    nlohmann::ordered_json extract_messages = nlohmann::ordered_json::array();
    extract_messages.push_back({{"role", "system"}, {"content", prompts["extract_system"]}});
    extract_messages.push_back({{"role", "user"}, {"content", extract_user_prompt}});
    // MEMORY_AGENT_MAX_TOKENS = 2000
    const std::string extract_reply = run_chat(chat, "extract", "STEP 1: EXTRACT (hy_memory real prompt)", extract_messages, 2000);

    std::optional<nlohmann::ordered_json> extracted = parse_json_block(extract_reply);
    if (!extracted || !extracted->is_object()) {
        std::cout << "EXTRACT reply is not a JSON object" << std::endl;
        return 1;
    }
    nlohmann::ordered_json new_memories = extracted->value("memory", nlohmann::ordered_json::array());
    if (!new_memories.is_array()) new_memories = nlohmann::ordered_json::array();
    std::cout << "extracted memory (-> goes to reconcile):" << std::endl;
    for (const auto& m : new_memories) {
        std::cout << "  - " << m.value("content", "") << "   tags=" << m.value("tags", nlohmann::ordered_json::array()).dump()
                  << " owner=" << m.value("owner", nlohmann::ordered_json()).dump() << std::endl;
    }
    // intentions are stored on their own and never pass through reconcile
    std::cout << "extracted intentions (stored separately, not reconciled):" << std::endl;
    for (const auto& it : extracted->value("intentions", nlohmann::ordered_json::array())) {
        std::cout << "  - " << it.value("content", "") << "   valid_until=" << it.value("valid_until", nlohmann::ordered_json()).dump() << std::endl;
    }
    std::cout << "basic_info: " << extracted->value("basic_info", nlohmann::ordered_json::object()).dump() << std::endl;

    // ---- STEP 2: RECONCILE ----
    // Fake existing memories, each chosen to provoke one kind of op:
    //   mem-001 city -> SUPERSEDE, mem-002 Python -> UPDATE, mem-003 dog -> untouched,
    //   (none) vegetarian -> ADD
    const nlohmann::ordered_json existing_memories = nlohmann::ordered_json::parse(R"([
        {"memory_id": "mem-001", "content": "The user lives in Portland.",
         "owner": "user", "memory_at": "2026-03-02 10:15", "tags": ["location", "home"]},
        {"memory_id": "mem-002", "content": "The user likes programming in Python.",
         "owner": "user", "memory_at": "2026-05-20 18:40", "tags": ["python", "programming"]},
        {"memory_id": "mem-003", "content": "The user has a dog named Max.",
         "owner": "user", "memory_at": "2026-07-11 09:05", "tags": ["pet", "dog"]}
    ])");
    // same fields and key order as the engine: new memories carry memory_at, no id
    nlohmann::ordered_json new_block = nlohmann::ordered_json::array();
    for (const auto& m : new_memories) {
        nlohmann::ordered_json owner = m.value("owner", nlohmann::ordered_json());
        new_block.push_back({
            {"content", m.value("content", "")},
            {"owner", owner.is_string() && !owner.get<std::string>().empty() ? owner : nlohmann::ordered_json("user")},
            {"memory_at", now_minute},
            {"tags", m.value("tags", nlohmann::ordered_json::array())},
        });
    }
    const std::string reconcile_prompt = py_format(prompts["reconcile"].get<std::string>(), {
        {"current_date", now_minute},
        {"existing_memories", existing_memories.dump(2)},   // json.dumps(indent=2) prints the same
        {"new_memories", new_block.dump(2)},
        {"few_shot_section", ""},                           // few-shot off by default
    });
    // llm.complete(prompt) sends one user message; the provider prepends an empty system one
    nlohmann::ordered_json reconcile_messages = nlohmann::ordered_json::array();
    reconcile_messages.push_back({{"role", "system"}, {"content", ""}});
    reconcile_messages.push_back({{"role", "user"}, {"content", reconcile_prompt}});
    // engine: max(MAX_TOKENS 2000, n_new * PER_OP_TOKENS 300 + BASE_OVERHEAD 500), capped at 8000
    const int max_tokens = std::min(8000, std::max(2000, (int)new_memories.size() * 300 + 500));
    const std::string reconcile_reply = run_chat(chat, "reconcile", "STEP 2: RECONCILE (hy_memory real prompt)", reconcile_messages, max_tokens);

    std::optional<nlohmann::ordered_json> ops = parse_json_block(reconcile_reply);
    // The prompt asks for a bare array; the engine (reconciler._parse_ops) accepts
    // {"ops": [...]} too, but any other wrapper falls back to ADD_ALL.
    if (ops && ops->is_object()) {
        std::string wrapper;
        for (auto it = ops->begin(); it != ops->end(); ++it) {
            if (it.value().is_array()) { wrapper = it.key(); break; }
        }
        if (wrapper == "ops") {
            std::cout << "NOTE: reply is wrapped as {\"ops\": [...]}; hy_memory accepts this (group format)." << std::endl;
        } else if (!wrapper.empty()) {
            std::cout << "WARNING: reply is wrapped as {\"" << wrapper << "\": [...]}; hy_memory's _parse_ops would skip it and fall back to ADD_ALL." << std::endl;
        }
        if (!wrapper.empty()) ops = (*ops)[wrapper];
    }
    if (!ops || !ops->is_array()) {
        std::cout << "RECONCILE reply is not a JSON array" << std::endl;
        return 1;
    }
    std::cout << "reconcile ops:" << std::endl;
    std::set<std::string> touched;
    for (const auto& op : *ops) {
        nlohmann::ordered_json target = op.contains("memory_id") ? op["memory_id"]
                                       : op.value("memory_ids", nlohmann::ordered_json("-"));
        std::cout << "  " << std::left << std::setw(10) << op.value("op", "?") << " target=" << target.dump() << std::endl;
        std::cout << "             content: " << op.value("content", nlohmann::ordered_json()).dump() << std::endl;
        if (op.contains("supersede_reason")) std::cout << "             reason : " << op["supersede_reason"].dump() << std::endl;
        if (op.contains("related")) std::cout << "             related: " << op["related"].dump() << std::endl;
        if (op.value("memory_id", nlohmann::ordered_json()).is_string()) touched.insert(op["memory_id"].get<std::string>());
        for (const auto& id : op.value("memory_ids", nlohmann::ordered_json::array())) {
            if (id.is_string()) touched.insert(id.get<std::string>());
        }
    }
    for (const auto& mem : existing_memories) {
        if (!touched.count(mem["memory_id"].get<std::string>())) {
            std::cout << "  (untouched) " << mem["memory_id"].get<std::string>() << ": " << mem["content"].get<std::string>() << std::endl;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hy-memory:30b-a3b"), "Model tag");
    desc.add_options()("messages", arg_utils::po::value<std::string>()->default_value("../../../test/hy_mem_npu/messages/model_card_demo.json"),
        "conversation to send: [{\"role\": ..., \"content\": ...}, ...]");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(2000), "max new tokens (hy_memory's MEMORY_AGENT_MAX_TOKENS)");
    desc.add_options()("reasoning", arg_utils::po::value<std::string>()->default_value("no_think"), "reasoning_effort: no_think | low | high");
    desc.add_options()("out,o", arg_utils::po::value<std::string>()->default_value(""), "also write the parsed JSON to this file");
    desc.add_options()("chain", arg_utils::po::bool_switch()->default_value(false), "run hy_memory's EXTRACT -> RECONCILE chain instead of --messages");
    desc.add_options()("prompt-tokens", arg_utils::po::value<std::string>()->default_value(""),
        "run the same extraction task at these prompt lengths instead of --messages, "
        "comma separated (e.g. \"128,512,2048\")");
    desc.add_options()("repeat", arg_utils::po::value<int>()->default_value(1),
        "runs per --prompt-tokens length; the table reports the median (prefill speed "
        "varies ~20% run to run, so use 3 or more to compare two settings)");
    desc.add_options()("prompts", arg_utils::po::value<std::string>()->default_value("../../../test/hy_mem_npu/prompts/hy_memory_prompts.json"),
        "the prompt templates for --chain");
    desc.add_options()("save", arg_utils::po::value<std::string>()->default_value(""),
        "write each request's <step>_messages.json and <step>_reply.txt to this directory (for models_simple/ref_hy_mem_messages.py)");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    const std::string tag = vm["model"].as<std::string>();
    std::string list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(list_path, models_dir);
    const std::string model_path = models.get_model_path(tag);
    auto model_info = models.get_model_info(tag).second;
    std::cout << "Model: " << tag << " (" << model_path << ")" << std::endl;

    HyMemoryProbe chat(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    auto t0 = std::chrono::steady_clock::now();
    chat.load_model(model_path, model_info, -1, false);
    header_print("info", "model loaded in "
                 << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s");
    if (!chat.configure_parameter("reasoning_effort", vm["reasoning"].as<std::string>())) {
        header_print("error", "reasoning must be no_think, low or high");
        return 1;
    }

    g_save_dir = vm["save"].as<std::string>();
    if (!g_save_dir.empty()) {
        std::filesystem::create_directories(g_save_dir);
    }
    if (vm["chain"].as<bool>()) {
        return run_chain(chat, vm["prompts"].as<std::string>());
    }

    const std::string sizes_arg = vm["prompt-tokens"].as<std::string>();
    if (!sizes_arg.empty()) {
        std::vector<int> sizes;
        std::stringstream ss(sizes_arg);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) sizes.push_back(std::stoi(item));
        }
        if (sizes.empty()) {
            header_print("error", "--prompt-tokens needs at least one length");
            return 1;
        }
        return run_sized(chat, sizes, vm["Length"].as<int>(), std::max(1, vm["repeat"].as<int>()));
    }

    const nlohmann::ordered_json messages = read_messages(vm["messages"].as<std::string>());
    const std::string reply = run_chat(chat, "messages", "messages: " + vm["messages"].as<std::string>(), messages,
                                       vm["Length"].as<int>());

    std::cout << std::string(100, '=') << "\nJSON\n" << std::string(100, '=') << std::endl;
    std::optional<nlohmann::ordered_json> parsed = parse_json_block(reply);
    if (!parsed) {
        std::cout << "NOT VALID JSON" << std::endl;
        return 1;
    }
    std::cout << parsed->dump(2) << std::endl;
    if (!vm["out"].as<std::string>().empty()) {
        std::ofstream(vm["out"].as<std::string>()) << parsed->dump(2) << std::endl;
    }
    return 0;
}
