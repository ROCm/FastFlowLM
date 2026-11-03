// Long-context accuracy probe for the Qwen3.5 engine: prefill the first N tokens of a long text and read the
// model's log-probability of the true token N, for many N. Run it with the old and the new engine and compare
// (mean NLL, top-1 agreement, and the KL between the two runs' full next-token distributions via --dump).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
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

namespace fs = std::filesystem;
namespace po = boost::program_options;

flm_rt::device npu_device_global;

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("qwen3.5", kFlmBackendId, flm_factory<qwen3_5vl_npu>());
#if defined(FLM_HAS_QWEN3_5VL_FLASH)
    registry.register_backend("qwen3.5-flash", kFlmBackendId, flm_factory<qwen3_5vl_flash>());
#endif
}
}  // namespace flm::backend

template <class Base>
class nll_model_t : public Base {
public:
    using Base::Base;
    causal_lm* engine() { return this->lm_engine; }
};

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    std::string model_tag;
    std::string text_files;
    std::string dump_path;
    int cuts = 0;
    int min_len = 0;
    int max_len = 0;
    po::options_description desc("nll_test options");
    desc.add_options()
        ("help,h", "help")
        ("model,m", po::value<std::string>(&model_tag)->default_value("qwen3.5:2b"), "model tag")
        ("text,t", po::value<std::string>(&text_files)->required(), "comma separated text files, joined with a blank line")
        ("cuts,n", po::value<int>(&cuts)->default_value(40), "number of cut points")
        ("min-len", po::value<int>(&min_len)->default_value(512), "shortest prefix")
        ("max-len", po::value<int>(&max_len)->default_value(16000), "longest prefix")
        ("dump,d", po::value<std::string>(&dump_path)->default_value(""), "write every cut's fp32 logits to this file")
        ("lengths", po::value<std::string>()->default_value(""), "comma separated exact prefix lengths instead of -n cuts")
        ("multiturn", po::bool_switch()->default_value(false), "instead of cuts: prefill / decode / prefill again / checkpoint-restore / clear, dumping the logits of every step");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    std::string model_list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(model_list_path, models_dir);
    std::string model_path = models.get_model_path(model_tag);
    nlohmann::json model_info = models.get_model_info(model_tag).second;

    std::string text;
    std::stringstream files(text_files);
    std::string file;
    while (std::getline(files, file, ',')) {
        if (!text.empty()) text += "\n\n";
        text += read_file(file);
    }
    Tokenizer tokenizer(model_path);
    std::vector<int> tokens = tokenizer.encode(text);
    max_len = std::min(max_len, (int)tokens.size() - 1);
    header_print("nll", "text tokens " << tokens.size() << ", cuts " << cuts << " in [" << min_len << ", " << max_len << "]");

    std::vector<int> cut_points;
    if (!vm["lengths"].as<std::string>().empty()) {
        std::stringstream ls(vm["lengths"].as<std::string>());
        std::string item;
        while (std::getline(ls, item, ',')) cut_points.push_back(std::stoi(item));
        cuts = 0;
    }
    for (int i = 0; i < cuts; i++) {
        double f = cuts == 1 ? 1.0 : (double)i / (cuts - 1);
        cut_points.push_back((int)std::lround(min_len * std::pow((double)max_len / min_len, f)));
    }

    std::unique_ptr<AutoModel> chat;
    std::function<causal_lm*()> engine;
#if defined(FLM_HAS_QWEN3_5VL_FLASH)
    if (model_info["details"]["family"] == "qwen3.5-flash") {
        auto model = std::make_unique<nll_model_t<Qwen3_5VL_Flash>>(&npu_device_global);
        engine = [m = model.get()] { return m->engine(); };
        chat = std::move(model);
    }
#endif
    if (!chat) {
        auto model = std::make_unique<nll_model_t<Qwen3_5VL>>(&npu_device_global);
        engine = [m = model.get()] { return m->engine(); };
        chat = std::move(model);
    }
    npu_device_global = flm_rt::device(0);
    chat->set_prefix_limits(-1, 0);
    chat->load_model(model_path, model_info, 32768, false);

    std::ofstream dump;
    if (!dump_path.empty()) dump.open(dump_path, std::ios::binary | std::ios::trunc);

    if (vm["multiturn"].as<bool>()) {
        // a chat-like sequence that leaves K/V rows written by decode, by a restore and by a clear in the cache
        auto argmax = [](buffer<bf16>& logits) {
            size_t best = 0;
            for (size_t v = 1; v < logits.size(); v++) if ((float)logits[v] > (float)logits[best]) best = v;
            return (int)best;
        };
        auto save = [&](const char* step, buffer<bf16>& logits) {
            std::vector<float> l(logits.size());
            for (size_t v = 0; v < l.size(); v++) l[v] = (float)logits[v];
            if (dump.is_open()) dump.write((const char*)l.data(), l.size() * sizeof(float));
            std::cout << step << ": context " << engine()->get_current_context_length() << ", top1 " << argmax(logits) << std::endl;
        };
        auto prefill = [&](const char* step, int from, int to) {
            std::vector<int> ids(tokens.begin() + from, tokens.begin() + to);
            buffer<bf16> logits = engine()->prefill(ids);
            save(step, logits);
            return logits;
        };
        auto decode = [&](const char* step, buffer<bf16> logits, int n) {
            for (int t = 0; t < n; t++) logits = engine()->forward(argmax(logits));
            save(step, logits);
            return logits;
        };
        chat->clear_context();
        buffer<bf16> logits = prefill("prefill 3000", 0, 3000);
        logits = decode("decode 37", logits, 37);
        logits = prefill("prefill +1500", 3000, 4500);
        engine()->checkpoint();
        decode("decode 8 (dropped)", logits, 8);
        engine()->restore();
        prefill("restore, prefill +700", 4500, 5200);
        chat->clear_context();
        prefill("clear, prefill 2333", 100, 2433);
        return 0;
    }

    double nll_sum = 0.0;
    int top1_hits = 0;
    std::cout << "cut,true_token,logprob,top1_token,top1_hit" << std::endl;
    for (int n : cut_points) {
        chat->clear_context();
        std::vector<int> ids(tokens.begin(), tokens.begin() + n);
        buffer<bf16> logits = engine()->prefill(ids);

        const size_t vocab = logits.size();
        std::vector<float> l(vocab);
        float max_logit = -INFINITY;
        size_t top1 = 0;
        for (size_t v = 0; v < vocab; v++) {
            l[v] = (float)logits[v];
            if (l[v] > max_logit) {
                max_logit = l[v];
                top1 = v;
            }
        }
        double sum = 0.0;
        for (size_t v = 0; v < vocab; v++) sum += std::exp((double)l[v] - max_logit);
        const int truth = tokens[n];
        const double logprob = (double)l[truth] - max_logit - std::log(sum);
        nll_sum -= logprob;
        top1_hits += (int)top1 == truth;
        std::cout << n << "," << truth << "," << std::setprecision(6) << logprob << "," << top1 << "," << ((int)top1 == truth) << std::endl;
        if (dump.is_open()) dump.write((const char*)l.data(), vocab * sizeof(float));
    }
    std::cout << "SUMMARY cuts=" << cut_points.size() << " mean_nll=" << std::setprecision(5) << nll_sum / cut_points.size()
              << " top1=" << top1_hits << "/" << cut_points.size() << std::endl;
    return 0;
}
