/// \file test_long.cpp
/// \brief long-context check for the hunyuan-dense engine: prefill and decode
///        time per prompt length, plus logits dumps for compare_logits.py
/// \author FastFlowLM Team
/// \date 2026-11-03
/// \note See README.md in this folder.

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hunyuan.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

class HunyuanProbe : public Hunyuan {
public:
    using Hunyuan::Hunyuan;
    causal_lm* engine() { return this->lm_engine; }
    Tokenizer* tokens() { return this->tokenizer.get(); }
};

static std::vector<int> parse_lengths(const std::string& text) {
    std::vector<int> lengths;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        lengths.push_back(std::stoi(item));
    }
    return lengths;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static int argmax(buffer<bf16>& logits) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); i++) {
        if ((float)logits[i] > (float)logits[best]) {
            best = (int)i;
        }
    }
    return best;
}

/// \brief FNV-1a of the logits bytes: equal hashes = byte-identical logits
static unsigned long long logits_hash(buffer<bf16>& logits) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(logits.data());
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < logits.size() * sizeof(bf16); i++) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    return h;
}

static void dump(const std::string& dir, const std::string& name, buffer<bf16>& logits) {
    if (dir.empty()) {
        return;
    }
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(bf16));
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hy-mt2-flash:1.8b"), "Model tag");
    desc.add_options()("text,t", arg_utils::po::value<std::string>()->required(), "Plain-text file the prompts are cut from");
    desc.add_options()("lengths,l", arg_utils::po::value<std::string>()->default_value("512,1024,2048,4096,8192,16384,32768"), "Prompt lengths in tokens");
    desc.add_options()("decode,d", arg_utils::po::value<int>()->default_value(32), "Decode steps after each prompt");
    desc.add_options()("context,L", arg_utils::po::value<int>()->default_value(0), "Engine context, 0 = longest prompt + decode");
    desc.add_options()("dump", arg_utils::po::value<std::string>()->default_value(""), "Folder for logits dumps, empty = none");
    desc.add_options()("grow-to", arg_utils::po::value<int>()->default_value(0), "Grow the engine context to this length (update_max_length) halfway through each decode, 0 = never");
    desc.add_options()("hash", arg_utils::po::bool_switch()->default_value(false), "Print a hash of each prompt's prefill logits (last column)");
    desc.add_options()("help,h", "Show this help");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    arg_utils::po::notify(vm);

    std::string tag = vm["model"].as<std::string>();
    std::vector<int> lengths = parse_lengths(vm["lengths"].as<std::string>());
    int decode_steps = vm["decode"].as<int>();
    std::string dump_dir = vm["dump"].as<std::string>();
    int grow_to = vm["grow-to"].as<int>();
    bool print_hash = vm["hash"].as<bool>();

    int longest = 0;
    for (int n : lengths) {
        longest = std::max(longest, n);
    }
    int context_length = vm["context"].as<int>();
    if (context_length == 0) {
        context_length = longest + decode_steps;
    }

    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);
    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag).second;
    // the catalog's single-turn label pins the context at the 1k default
    model_info.erase("label");

    std::unique_ptr<HunyuanProbe> chat = std::make_unique<HunyuanProbe>(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    chat->load_model(model_path, model_info, context_length, false);
    causal_lm* engine = chat->engine();

    std::vector<int> text_tokens = chat->tokens()->encode(read_file(vm["text"].as<std::string>()));
    if ((int)text_tokens.size() < longest + decode_steps) {
        std::cout << "Text has only " << text_tokens.size() << " tokens, need " << longest + decode_steps << std::endl;
        return 1;
    }
    std::cout << "Context " << context_length << ", text " << text_tokens.size() << " tokens" << std::endl;

    // first call pays for sequence building and bo allocation
    {
        std::vector<int> warm(text_tokens.begin(), text_tokens.begin() + std::min(600, longest));
        engine->clear_context();
        engine->prefill(warm);
    }

    std::cout << std::endl
              << std::setw(8) << "tokens" << std::setw(13) << "prefill(ms)" << std::setw(14) << "prefill tok/s"
              << std::setw(13) << "decode(ms)" << std::setw(13) << "decode tok/s" << std::setw(10) << "top1" << std::endl;

    for (int n : lengths) {
        std::vector<int> prompt(text_tokens.begin(), text_tokens.begin() + n);
        engine->clear_context();

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        buffer<bf16> logits = engine->prefill(prompt);
        double prefill_ms = ms_since(t0);
        int top1 = argmax(logits);
        unsigned long long hash = print_hash ? logits_hash(logits) : 0;
        dump(dump_dir, "L" + std::to_string(n) + "_prefill.bin", logits);

        // teacher forcing: every build sees the same tokens, so the logits compare step by step
        double decode_ms = 0.0;
        for (int i = 0; i < decode_steps; i++) {
            if (grow_to > 0 && i == decode_steps / 2) {
                // the path a client's larger context request takes (AutoModel::set_max_length)
                engine->update_max_length((uint32_t)grow_to);
            }
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            buffer<bf16> step = engine->forward(text_tokens[n + i]);
            decode_ms += ms_since(t1);
            dump(dump_dir, "L" + std::to_string(n) + "_decode" + std::to_string(i) + ".bin", step);
        }

        double decode_rate = decode_steps > 0 ? decode_steps * 1000.0 / decode_ms : 0.0;
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << n << std::setw(13) << prefill_ms << std::setw(14) << n * 1000.0 / prefill_ms
                  << std::setw(13) << decode_ms << std::setw(13) << decode_rate << std::setw(10) << top1;
        if (print_hash) {
            std::cout << "  " << std::hex << hash << std::dec;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
