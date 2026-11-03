/// \file test.cpp
/// \brief speed harness for Qwen3.5 (qwen3_5vl_npu): TTFT, prefill and decode speed per prompt length
/// \note See README.md in this folder for how it measures and why.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "AutoModel/flm_backend.hpp"
#include "models/qwen3_5vl/flm/aie2p/qwen3_5vl_npu.hpp"
#if defined(FLM_HAS_QWEN3_5VL_FLASH)
#include "models/qwen3_5vl_flash/flm/aie2p/qwen3_5vl_flash.hpp"
#endif
#include "tokenizer/tokenizer.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("qwen3.5", kFlmBackendId, flm_factory<qwen3_5vl_npu>());
#if defined(FLM_HAS_QWEN3_5VL_FLASH)
    registry.register_backend("qwen3.5-flash", kFlmBackendId, flm_factory<qwen3_5vl_flash>());
#endif
}
}  // namespace flm::backend

namespace fs = std::filesystem;
using steady = std::chrono::steady_clock;

static bool g_engine_log = false;

struct bench_args_t {
    std::string model_tag;
    std::vector<int> lengths;
    int focus_len;
    int gen_tokens;
    int iterations;
    int warmup;
    std::string prompt_file;
    std::string csv_file;
    bool show_response;
    bool think;
};

struct run_result_t {
    int target_len;
    int iteration;
    int prompt_tokens;
    int cached_tokens;
    int generated_tokens;
    double ttft_s;
    double prefill_tps;
    double decode_tps;
    std::string stop_reason;
};

struct stats_t {
    double mean = 0, std = 0, min = 0, max = 0;
};

static std::vector<int> parse_lengths(const std::string& text) {
    std::vector<int> lengths;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            lengths.push_back(std::stoi(item));
        }
    }
    return lengths;
}

static bool parse_args(int argc, char* argv[], bench_args_t& args) {
    namespace po = arg_utils::po;
    std::string lengths_text;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help")
        ("model,m", po::value<std::string>(&args.model_tag)->default_value("qwen3.5:2b"), "Model tag in model_list.json")
        ("len,l", po::value<std::string>(&lengths_text)->default_value("128,1024,4096,10240"), "Prompt lengths in tokens, comma separated")
        ("focus", po::value<int>(&args.focus_len)->default_value(10240), "Length reported as the headline FOCUS result")
        ("gen,g", po::value<int>(&args.gen_tokens)->default_value(128), "Tokens to decode per run")
        ("iters,i", po::value<int>(&args.iterations)->default_value(3), "Measured runs per length")
        ("warmup,w", po::value<int>(&args.warmup)->default_value(1), "Unmeasured runs before measuring, per length")
        ("prompt-file,f", po::value<std::string>(&args.prompt_file)->default_value(""), "Base text; default bench_text.txt next to the binary")
        ("csv,c", po::value<std::string>(&args.csv_file)->default_value(""), "Write one row per run to this CSV file")
        ("show,s", po::value<bool>(&args.show_response)->default_value(true), "Print the response of the first run at each length")
        ("think,t", po::value<bool>(&args.think)->default_value(false), "Enable thinking in the chat template")
        ("engine-log,e", po::value<bool>(&g_engine_log)->default_value(false), "Let the engine print to stdout during runs");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
    }
    args.lengths = parse_lengths(lengths_text);
    if (args.prompt_file.empty()) {
        args.prompt_file = utils::get_executable_directory() + "/bench_text.txt";
    }
    args.prompt_file = fs::absolute(args.prompt_file).string();
    if (!args.csv_file.empty()) {
        args.csv_file = fs::absolute(args.csv_file).string();
    }
    return true;
}

static bool check_paths(const std::string& model_path, const std::string& xclbin_dir, const std::string& prompt_file) {
    bool ok = true;
    for (const char* name : {"model.q4nx", "config.json", "tokenizer.json"}) {
        if (!fs::exists(fs::path(model_path) / name)) {
            header_print("ERROR", "Missing " << name << " in " << model_path
                         << " -- set FLM_MODEL_PATH to the folder that contains models/<name> (make test FLM_MODEL_PATH=...)");
            ok = false;
        }
    }
    if (!fs::exists(fs::path(xclbin_dir) / "layer.xclbin")) {
        header_print("ERROR", "No layer.xclbin in " << xclbin_dir
                     << " -- run through `make test`: this DEV_BUILD finds xclbins at ../../../xclbins relative to the current folder");
        ok = false;
    }
    if (!fs::exists(prompt_file)) {
        header_print("ERROR", "Prompt file not found: " << prompt_file);
        ok = false;
    }
    return ok;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open prompt file: " + path);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static std::string make_prompt(Tokenizer& tokenizer, const std::vector<int>& base_tokens, int length) {
    std::vector<int> tokens;
    tokens.reserve(length);
    while ((int)tokens.size() < length) {
        int take = std::min((int)base_tokens.size(), length - (int)tokens.size());
        tokens.insert(tokens.end(), base_tokens.begin(), base_tokens.begin() + take);
    }
    return tokenizer.decode(tokens);
}

static int pick_context_length(const bench_args_t& args) {
    int longest = *std::max_element(args.lengths.begin(), args.lengths.end());
    int needed = longest + args.gen_tokens + 256;
    int context = 8192;
    while (context < needed) {
        context *= 2;
    }
    return context;
}

static double load_average() {
    std::ifstream file("/proc/loadavg");
    double load = -1;
    file >> load;
    return load;
}

struct cout_muter_t {
    std::streambuf* saved = nullptr;
    explicit cout_muter_t(bool mute) {
        if (mute) {
            saved = std::cout.rdbuf(nullptr);
        }
    }
    ~cout_muter_t() {
        if (saved) {
            std::cout.rdbuf(saved);
            std::cout.clear();
        }
    }
};

static run_result_t run_once(AutoModel& chat, const std::string& prompt, int gen_tokens, std::ostream& os) {
    cout_muter_t muter(!g_engine_log);
    chat.clear_context();
    chat_meta_info_t meta_info;
    lm_uniform_input_t input;
    input.prompt = prompt;

    auto t0 = steady::now();
    chat.insert(meta_info, input);
    auto t1 = steady::now();
    chat.generate(meta_info, gen_tokens, os);

    run_result_t r{};
    r.prompt_tokens = meta_info.prompt_tokens;
    r.cached_tokens = meta_info.cached_prompt_tokens;
    r.generated_tokens = meta_info.generated_tokens;
    r.ttft_s = std::chrono::duration<double>(t1 - t0).count();
    int prefilled = meta_info.prompt_tokens - meta_info.cached_prompt_tokens;
    r.prefill_tps = prefilled / (meta_info.prefill_duration / 1e9);
    r.decode_tps = meta_info.generated_tokens / (meta_info.decoding_duration / 1e9);
    r.stop_reason = stop_reason_to_string(meta_info.stop_reason);
    return r;
}

static stats_t compute_stats(const std::vector<double>& values) {
    stats_t s;
    if (values.empty()) {
        return s;
    }
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    for (double v : values) {
        s.mean += v;
    }
    s.mean /= values.size();
    for (double v : values) {
        s.std += (v - s.mean) * (v - s.mean);
    }
    s.std = std::sqrt(s.std / values.size());
    return s;
}

static void print_run(const run_result_t& r) {
    std::cout << std::fixed << std::setprecision(2)
              << "  len " << std::setw(6) << r.target_len
              << "  iter " << r.iteration
              << "  prompt " << std::setw(6) << r.prompt_tokens
              << "  gen " << std::setw(4) << r.generated_tokens << " (" << r.stop_reason << ")"
              << "  TTFT " << std::setprecision(3) << r.ttft_s << " s"
              << "  prefill " << std::setprecision(1) << r.prefill_tps << " tok/s"
              << "  decode " << std::setprecision(2) << r.decode_tps << " tok/s" << std::endl;
    if (r.cached_tokens > 0) {
        header_print("WARN", r.cached_tokens << " prompt tokens came from the cache; prefill speed is not a clean measurement");
    }
}

static std::string format_stats(const stats_t& s, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision)
       << s.mean << " +- " << s.std << " [" << s.min << ", " << s.max << "]";
    return ss.str();
}

struct length_summary_t {
    int prompt_tokens = 0;
    stats_t ttft, prefill, decode;
};

static length_summary_t summarize_length(const std::vector<run_result_t>& results, int len) {
    std::vector<double> ttft, prefill, decode;
    length_summary_t s;
    for (const auto& r : results) {
        if (r.target_len != len) {
            continue;
        }
        ttft.push_back(r.ttft_s);
        prefill.push_back(r.prefill_tps);
        decode.push_back(r.decode_tps);
        s.prompt_tokens = r.prompt_tokens;
    }
    s.ttft = compute_stats(ttft);
    s.prefill = compute_stats(prefill);
    s.decode = compute_stats(decode);
    return s;
}

static void print_summary(const bench_args_t& args, const std::vector<run_result_t>& results) {
    std::cout << std::endl << std::left
              << std::setw(8) << "len" << std::setw(8) << "prompt"
              << std::setw(34) << "TTFT s (mean +- std [min, max])"
              << std::setw(34) << "prefill tok/s"
              << "decode tok/s" << std::endl;
    std::cout << std::string(118, '-') << std::endl;
    for (int len : args.lengths) {
        length_summary_t s = summarize_length(results, len);
        std::cout << std::setw(8) << len << std::setw(8) << s.prompt_tokens
                  << std::setw(34) << format_stats(s.ttft, 3)
                  << std::setw(34) << format_stats(s.prefill, 1)
                  << format_stats(s.decode, 2) << std::endl;
    }
    std::cout << std::right << std::endl;
}

static void print_focus(const bench_args_t& args, const std::vector<run_result_t>& results) {
    bool measured = std::find(args.lengths.begin(), args.lengths.end(), args.focus_len) != args.lengths.end();
    if (!measured) {
        header_print("info", "FOCUS length " << args.focus_len << " was not in --len, no FOCUS line");
        return;
    }
    length_summary_t s = summarize_length(results, args.focus_len);
    std::cout << std::fixed
              << "FOCUS " << args.focus_len << " (prompt " << s.prompt_tokens << "): "
              << "TTFT " << std::setprecision(3) << s.ttft.mean << " s | "
              << "prefill " << std::setprecision(1) << s.prefill.mean << " tok/s | "
              << "decode " << std::setprecision(2) << s.decode.mean << " tok/s" << std::endl << std::endl;
}

static void write_csv(const std::string& path, const std::vector<run_result_t>& results) {
    std::ofstream out(path, std::ios::trunc);
    out << "target_len,iteration,prompt_tokens,cached_tokens,generated_tokens,stop_reason,ttft_s,prefill_tok_s,decode_tok_s\n";
    for (const auto& r : results) {
        out << r.target_len << "," << r.iteration << "," << r.prompt_tokens << "," << r.cached_tokens << ","
            << r.generated_tokens << "," << r.stop_reason << "," << r.ttft_s << ","
            << r.prefill_tps << "," << r.decode_tps << "\n";
    }
    header_print("info", "Wrote " << results.size() << " rows to " << path);
}

int main(int argc, char* argv[]) {
    bench_args_t args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }

    std::string model_list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(model_list_path, models_dir);
    std::string model_path = models.get_model_path(args.model_tag);
    nlohmann::json model_info = models.get_model_info(args.model_tag).second;
    int context_length = pick_context_length(args);

    const std::string xclbin_folder = model_info.value("xclbin_folder", fs::path(model_path).filename().string());
    std::string xclbin_dir = fs::absolute("../../../xclbins/" + xclbin_folder).lexically_normal().string();
    header_print("info", "Model:   " << args.model_tag << " -> " << model_path);
    header_print("info", "Xclbins: " << xclbin_dir);
    header_print("info", "Prompt:  " << args.prompt_file);
    if (!check_paths(model_path, xclbin_dir, args.prompt_file)) {
        return 1;
    }
    std::string lengths_text;
    for (int len : args.lengths) {
        lengths_text += std::to_string(len) + " ";
    }
    header_print("info", "Lengths: " << lengths_text << "| gen " << args.gen_tokens
                 << " | iters " << args.iterations << " | warmup " << args.warmup
                 << " | ctx " << context_length << " | think " << args.think);
    header_print("info", "CPU load average (1 min): " << load_average());

    Tokenizer tokenizer(model_path);
    std::vector<int> base_tokens = tokenizer.encode(read_text_file(args.prompt_file));

    std::unique_ptr<AutoModel> chat;
#if defined(FLM_HAS_QWEN3_5VL_FLASH)
    if (model_info["details"]["family"] == "qwen3.5-flash") {
        chat = std::make_unique<Qwen3_5VL_Flash>(&npu_device_global);
    }
#endif
    if (!chat) {
        chat = std::make_unique<Qwen3_5VL>(&npu_device_global);
    }
    npu_device_global = flm_rt::device(0);
    chat->set_prefix_limits(-1, 0);
    auto load_start = steady::now();
    chat->load_model(model_path, model_info, context_length, false);
    double load_s = std::chrono::duration<double>(steady::now() - load_start).count();
    header_print("info", "Model loaded in " << std::fixed << std::setprecision(2) << load_s << " s");
    chat->set_topk(1);
    chat->configure_parameter("enable_think", args.think);

    std::ostream null_stream(nullptr);
    std::vector<run_result_t> results;
    for (int len : args.lengths) {
        std::string prompt = make_prompt(tokenizer, base_tokens, len);
        for (int w = 0; w < args.warmup; w++) {
            run_once(*chat, prompt, args.gen_tokens, null_stream);
        }
        for (int it = 0; it < args.iterations; it++) {
            std::ostringstream response;
            bool show = args.show_response && it == 0;
            run_result_t r = run_once(*chat, prompt, args.gen_tokens, show ? (std::ostream&)response : null_stream);
            r.target_len = len;
            r.iteration = it;
            print_run(r);
            if (show) {
                std::cout << "    response: " << response.str() << std::endl;
            }
            results.push_back(r);
        }
    }

    print_summary(args, results);
    print_focus(args, results);
    if (!args.csv_file.empty()) {
        write_csv(args.csv_file, results);
    }
    return 0;
}
