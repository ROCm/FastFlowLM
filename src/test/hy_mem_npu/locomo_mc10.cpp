/// \file locomo_mc10.cpp
/// \brief LoCoMo-MC10 multiple-choice scoring on the hy_mem_npu engine
/// \note Reads the token ids made by Projects/hy_locomo_eval/make_prompts.py.
///       Each conversation is prefilled once; every question restores to that
///       point and prefills only its own tail. See Projects/hy_locomo_eval/NOTES.md.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hy_memory.hpp"
#include "AutoModel/flm_backend.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("hy-memory", kFlmBackendId, flm_factory<hy_mem_npu>());
}
}  // namespace flm::backend

class HyMemoryProbe : public HyMemory {
public:
    using HyMemory::HyMemory;
    causal_lm* engine() { return this->lm_engine; }
    std::string decode(const std::vector<int>& ids) { return this->tokenizer->decode(ids); }
};

using json = nlohmann::ordered_json;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::set<std::string> done_question_ids(const std::string& out_path) {
    std::set<std::string> done;
    std::ifstream f(out_path);
    std::string line;
    while (std::getline(f, line)) {
        json row = json::parse(line, nullptr, false);
        if (!row.is_discarded() && row.contains("question_id")) done.insert(row["question_id"].get<std::string>());
    }
    return done;
}

static std::vector<float> to_float(const buffer<bf16>& logits) {
    std::vector<float> f(logits.size());
    for (size_t i = 0; i < f.size(); i++) f[i] = (float)logits[i];
    return f;
}

static void dump_logits(const std::string& dir, const std::string& question_id, const std::vector<float>& logits) {
    std::ofstream f(dir + "/" + question_id + ".f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
}

struct Scored {
    std::vector<float> letter_logits;
    int pick;
    int top_token;
    bool correct;
};

static Scored score(const std::vector<float>& logits, const std::vector<int>& letter_ids, const json& question) {
    Scored s;
    for (int id : letter_ids) s.letter_logits.push_back(logits[id]);
    s.pick = (int)(std::max_element(s.letter_logits.begin(), s.letter_logits.end()) - s.letter_logits.begin());
    s.top_token = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    s.correct = false;
    for (const auto& i : question["accepted_indices"]) {
        if (i.get<int>() == s.pick) s.correct = true;
    }
    return s;
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hy-memory:30b-a3b"), "Model tag");
    desc.add_options()("prompts", arg_utils::po::value<std::string>()->default_value(
        "/scratch/michyu/Projects/hy_locomo_eval/out/prompts.json"), "make_prompts.py output");
    desc.add_options()("out", arg_utils::po::value<std::string>()->required(), "results, one JSON line per question (appended)");
    desc.add_options()("set", arg_utils::po::value<std::string>()->default_value("pilot"), "pilot or all");
    desc.add_options()("convs", arg_utils::po::value<std::string>()->default_value(""), "only these conversations, e.g. conv-41,conv-30");
    desc.add_options()("limit", arg_utils::po::value<int>()->default_value(0), "at most N questions per conversation (0 = no limit)");
    desc.add_options()("nochat", arg_utils::po::bool_switch()->default_value(false), "question and choices only, no conversation");
    desc.add_options()("ctx", arg_utils::po::value<int>()->default_value(32768), "context length (KV cache size)");
    desc.add_options()("dump-logits", arg_utils::po::value<std::string>()->default_value(""), "write each question's full logits (float32) here");
    desc.add_options()("resume", arg_utils::po::bool_switch()->default_value(false), "skip questions already in --out");
    desc.add_options()("fresh", arg_utils::po::bool_switch()->default_value(false), "prefill the conversation again for every question (checks the reuse)");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);
    arg_utils::po::notify(vm);

    const std::string out_path = vm["out"].as<std::string>();
    const bool nochat = vm["nochat"].as<bool>();
    const bool fresh = vm["fresh"].as<bool>();
    const bool pilot_only = vm["set"].as<std::string>() == "pilot";
    const int limit = vm["limit"].as<int>();
    const std::string dump_dir = vm["dump-logits"].as<std::string>();
    const std::vector<std::string> conv_filter = split_commas(vm["convs"].as<std::string>());

    std::ifstream pf(vm["prompts"].as<std::string>());
    if (!pf.is_open()) {
        header_print("error", "cannot open " << vm["prompts"].as<std::string>());
        return 1;
    }
    const json prompts = json::parse(pf);
    const std::vector<int> letter_ids = prompts["letter_ids"].get<std::vector<int>>();
    const std::string letters = prompts["letters"].get<std::string>();

    std::vector<std::string> conv_order;
    std::map<std::string, std::vector<json>> by_conv;
    for (const json& q : prompts["questions"]) {
        const std::string conv = q["conv"].get<std::string>();
        if (pilot_only && !q["pilot"].get<bool>()) continue;
        if (!conv_filter.empty() && std::find(conv_filter.begin(), conv_filter.end(), conv) == conv_filter.end()) continue;
        if (limit > 0 && (int)by_conv[conv].size() >= limit) continue;
        if (by_conv[conv].empty()) conv_order.push_back(conv);
        by_conv[conv].push_back(q);
    }

    const std::set<std::string> done = vm["resume"].as<bool>() ? done_question_ids(out_path) : std::set<std::string>{};

    const std::string tag = vm["model"].as<std::string>();
    std::string list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(list_path, models_dir);
    HyMemoryProbe chat(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    chat.load_model(models.get_model_path(tag), models.get_model_info(tag).second, vm["ctx"].as<int>(), false);
    causal_lm* engine = chat.engine();

    std::ofstream out(out_path, std::ios::app);
    int total = 0, total_correct = 0;
    for (const std::string& conv : conv_order) {
        std::vector<json> todo;
        for (const json& q : by_conv[conv]) {
            if (!done.count(q["question_id"].get<std::string>())) todo.push_back(q);
        }
        if (todo.empty()) continue;

        std::vector<int> prefix = prompts["prefixes"][nochat ? "nochat" : conv].get<std::vector<int>>();
        engine->clear_context();
        auto t0 = std::chrono::steady_clock::now();
        engine->prefill(prefix);
        const double prefix_ms = ms_since(t0);
        engine->checkpoint();
        header_print("info", conv << ": prefix " << prefix.size() << " tokens in " << prefix_ms / 1000.0 << " s ("
                     << 1000.0 * prefix.size() / prefix_ms << " tok/s), " << todo.size() << " questions");

        int correct = 0;
        for (const json& q : todo) {
            t0 = std::chrono::steady_clock::now();
            if (fresh) {
                engine->clear_context();
                engine->prefill(prefix);
            } else {
                engine->restore();
            }
            const double restore_ms = ms_since(t0);
            std::vector<int> tail = q[nochat ? "tail_nochat" : "tail"].get<std::vector<int>>();
            t0 = std::chrono::steady_clock::now();
            const std::vector<float> logits = to_float(engine->prefill(tail));
            const double tail_ms = ms_since(t0);
            const Scored s = score(logits, letter_ids, q);
            if (!dump_dir.empty()) dump_logits(dump_dir, q["question_id"].get<std::string>(), logits);

            json row;
            row["question_id"] = q["question_id"];
            row["conv"] = conv;
            row["question_type"] = q["question_type"];
            row["nochat"] = nochat;
            row["fresh"] = fresh;
            row["pick"] = std::string(1, letters[s.pick]);
            row["correct_index"] = q["correct_index"];
            row["correct"] = s.correct;
            row["letter_logits"] = s.letter_logits;
            row["top_token"] = s.top_token;
            row["top_token_text"] = chat.decode({s.top_token});
            row["prefix_tokens"] = prefix.size();
            row["tail_tokens"] = tail.size();
            row["restore_ms"] = restore_ms;
            row["tail_ms"] = tail_ms;
            out << row.dump() << std::endl;

            correct += s.correct;
            std::cout << "  " << q["question_id"].get<std::string>() << " " << q["question_type"].get<std::string>()
                      << ": pick " << letters[s.pick] << ", answer " << letters[q["correct_index"].get<int>()]
                      << (s.correct ? "  ok" : "  WRONG") << "  (" << tail.size() << " tokens, restore " << restore_ms
                      << " ms, prefill " << tail_ms << " ms)" << std::endl;
        }
        total += (int)todo.size();
        total_correct += correct;
        header_print("info", conv << ": " << correct << "/" << todo.size() << " correct");
    }
    header_print("info", "this run: " << total_correct << "/" << total << " correct"
                 << (total ? " (" + std::to_string(100.0 * total_correct / total) + "%)" : ""));
    return 0;
}
