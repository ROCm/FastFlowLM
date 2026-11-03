/// \file vision_bench.cpp
/// \brief Time the Qwen3.5 vision encoder at several image sizes. See README.md.
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "utils/utils.hpp"
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "AutoModel/flm_backend.hpp"
#include "models/qwen3_5vl/flm/aie2p/qwen3_5vl_npu.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("qwen3.5", kFlmBackendId, flm_factory<qwen3_5vl_npu>());
}
}  // namespace flm::backend

namespace fs = std::filesystem;
using steady = std::chrono::steady_clock;

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

struct run_result_t {
    int prompt_tokens;
    double ttft_s;
    std::string answer;
};

static run_result_t run_once(AutoModel& chat, const std::string& prompt, const std::string& image, int gen_tokens) {
    std::ostringstream answer;
    run_result_t r{};
    {
        cout_muter_t muter(true);
        chat.clear_context();
        chat_meta_info_t meta_info;
        lm_uniform_input_t input;
        input.prompt = prompt;
        if (!image.empty()) {
            input.images.push_back(image);
        }
        auto t0 = steady::now();
        chat.insert(meta_info, input);
        auto t1 = steady::now();
        chat.generate(meta_info, gen_tokens, answer);
        r.prompt_tokens = meta_info.prompt_tokens;
        r.ttft_s = std::chrono::duration<double>(t1 - t0).count();
    }
    r.answer = answer.str();
    return r;
}

static std::vector<std::string> split_commas(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "usage: vision_bench <img1,img2,...> [iters 3] [warmup 1] [gen 24]" << std::endl;
        return 1;
    }
    std::vector<std::string> images = split_commas(argv[1]);
    int iterations = argc > 2 ? std::stoi(argv[2]) : 3;
    int warmup = argc > 3 ? std::stoi(argv[3]) : 1;
    int gen_tokens = argc > 4 ? std::stoi(argv[4]) : 24;
    for (auto& image : images) {
        image = fs::absolute(image).string();
    }

    std::string model_tag = "qwen3.5:2b";
    std::string model_list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(model_list_path, models_dir);
    std::string model_path = models.get_model_path(model_tag);
    nlohmann::json model_info = models.get_model_info(model_tag).second;

    std::unique_ptr<AutoModel> chat = std::make_unique<Qwen3_5VL>(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    chat->set_prefix_limits(-1, 0);
    {
        cout_muter_t muter(true);
        chat->load_model(model_path, model_info, 8192, false);
    }
    chat->set_topk(1);
    chat->configure_parameter("enable_think", false);

    const std::string prompt = "Describe this image in one sentence.";
    std::vector<std::string> cases = {""};
    cases.insert(cases.end(), images.begin(), images.end());
    for (const auto& image : cases) {
        std::string label = image.empty() ? "text only" : fs::path(image).filename().string();
        for (int w = 0; w < warmup; w++) {
            std::cerr << "VISION warmup " << label << std::endl;
            run_once(*chat, prompt, image, gen_tokens);
        }
        for (int it = 0; it < iterations; it++) {
            std::cerr << "VISION measured " << label << " iter " << it << std::endl;
            run_result_t r = run_once(*chat, prompt, image, gen_tokens);
            std::cout << std::fixed << std::setprecision(3)
                      << "RESULT " << std::setw(22) << label << "  iter " << it
                      << "  prompt " << std::setw(5) << r.prompt_tokens
                      << "  TTFT " << r.ttft_s << " s" << std::endl;
            std::cout << "  answer: " << r.answer << std::endl;
        }
    }
    return 0;
}
