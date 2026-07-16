#include <iostream>
#include <cmath>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_qwen3_5_omni.hpp"
#include "model_list.hpp"

xrt::device npu_device_global;

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(8192), "Max generation length");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    int length_limit = vm["Length"].as<int>();
    std::cout << "Model: " << tag << std::endl;

    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);

    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    std::pair<std::string, nlohmann::json> model_info_pair = model_list.get_model_info(tag);
    nlohmann::json model_info = model_info_pair.second;
    std::cout << "Model path: " << model_path << std::endl;

    // Qwen3_5_Omni now inherits AutoModel, so we can drive it through the base
    // interface. The engine (qwen3_5_omni) is not a causal_lm, but that only
    // affects the wrapper's internals -- the AutoModel surface is fully honored.
    std::unique_ptr<AutoModel> chat = std::make_unique<Qwen3_5_Omni>(&npu_device_global);
    npu_device_global = xrt::device(0);
    std::cout << "NPU Device initialized: " << npu_device_global.get_info<xrt::info::device::name>() << std::endl;

    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");


    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    chat->set_topk(1); // deterministic (greedy) for a smoke test

    constexpr bool use_image = false;

    if (short_prompt) {
        if constexpr (use_image) {
            uniformed_input.prompt = "What are these?";
            uniformed_input.images.push_back("../../../tb_files/panda.png");
            uniformed_input.audios.push_back("../../../tb_files/nvidia.mp3");
        }
        else {
            uniformed_input.prompt = "Is Alibaba a good company?";
            uniformed_input.images.push_back("../../../tb_files/mj_icon.jpg");
            uniformed_input.images.push_back("../../../tb_files/google_icon.png");
            uniformed_input.images.push_back("../../../tb_files/pcb.jpg");
            uniformed_input.images.push_back("../../../tb_files/panda.png");

        }
    } else {
        uniformed_input.prompt = "Hello, introduce yourself briefly.";
    }

    std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
    std::cout << "Response: " << std::endl;

    chat->start_total_timer();
    // The engine's prefill/forward compute paths may still be stubs; guard the
    // call so the driver-side wiring (tokenization, image/audio preprocessing,
    // soft-token expansion) can be exercised without a hard crash.
    try {
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
    } catch (const std::exception& e) {
        header_print("FLM", "Engine path not available yet: " << e.what());
    }
    chat->stop_total_timer();
    std::cout << std::endl << std::endl;

    // Speech output (talker path) is omni-specific -- not part of the AutoModel
    // interface -- so reach it via the concrete type. Not implemented in the
    // engine yet; surfaces a clear message rather than crashing.
    static_cast<Qwen3_5_Omni*>(chat.get())->say("output.wav");

    std::cout << chat->show_profile() << std::endl;

    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;
    std::cout << std::endl;

    return 0;
}
