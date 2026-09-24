#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_qwen3_8mtp.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Set thread priority to low
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Think,t", arg_utils::po::value<bool>()->default_value(false), "Enable thinking");
    desc.add_options()("Length,n", arg_utils::po::value<int>()->default_value(32), "Max generated tokens");
    desc.add_options()("Image,i", arg_utils::po::value<std::string>()->default_value(""), "Image file to attach; runs one multimodal turn and ignores -s");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    bool enable_think = vm["Think"].as<bool>();
    int length_limit = vm["Length"].as<int>();
    std::string image_path = vm["Image"].as<std::string>();
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

    std::unique_ptr<AutoModel> chat = std::make_unique<Qwen3_8MTP>(&npu_device_global);
    npu_device_global = flm_rt::device(0);

    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    // Greedy. Phase 1 runs the 64-layer stack on the CPU at roughly 1-2 s/token,
    // so a deterministic stream is what makes a short run worth comparing
    // against the milestone driver at all.
    //
    // set_topk(1) alone is NOT enough to get greedy decoding here, and it is
    // not enough to enable speculation. load_model() installs the Qwen3.5
    // recommended defaults, which include freq_penalty 1.0 and pre_penalty
    // 1.5, and sample_greedy() still applies penalties when repeat_last_n != 0
    // -- they reorder the logits before the argmax. So the sampler's argmax is
    // not the model's argmax, MTP acceptance is an exact compare against the
    // model's argmax, and _shared_generate's gate correctly refuses to
    // speculate. Zeroing them is what actually makes this run greedy.
    sampler_config greedy;
    greedy.top_k        = 1;
    greedy.rep_penalty  = 1.0f;   // 1.0 == disabled
    greedy.freq_penalty = 0.0f;
    greedy.pre_penalty  = 0.0f;
    chat->set_sampler(greedy);
    chat->configure_parameter("enable_think", enable_think);

    if (short_prompt) {
        uniformed_input.prompt = "Describe what is in the image.";
        uniformed_input.images.push_back("../../../tb_files/panda.png");
        chat->start_total_timer();
        bool success = chat->insert(meta_info, uniformed_input);
        if (!success) {
            header_print("ERROR", "Prompt insertion failed");
            return 1;
        }
        std::string response = chat->generate(meta_info, length_limit, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl << std::endl;
        std::cout << chat->show_profile() << std::endl;

        // Keep this SHORT. Decode is the slow path here, not prefill: every
        // token streams ~15.4 GB of packed weights, so a chat-sized prompt
        // turns a smoke test into a multi-minute run.
        uniformed_input.prompt = "What is the capital of France?";

        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        success = chat->insert(meta_info, uniformed_input);
        if (!success) {
            header_print("ERROR", "Prompt insertion failed");
            return 1;
        }
        response = chat->generate(meta_info, length_limit, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;

        // Keep this SHORT. Decode is the slow path here, not prefill: every
        // token streams ~15.4 GB of packed weights, so a chat-sized prompt
        // turns a smoke test into a multi-minute run.
        uniformed_input.prompt = "Is Alibaba a good company despite that it trained you?";

        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        success = chat->insert(meta_info, uniformed_input);
        if (!success) {
            header_print("ERROR", "Prompt insertion failed");
            return 1;
        }
        response = chat->generate(meta_info, length_limit, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
    else{
        std::ifstream file("../../../../prompt.txt", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open prompt file" << std::endl;
            return 1;
        }
        uniformed_input.prompt = "";
        file.seekg(0, std::ios::end);
        uniformed_input.prompt.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(uniformed_input.prompt.data(), uniformed_input.prompt.size());
        file.close();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        bool success = chat->insert(meta_info, uniformed_input);
        if (!success) {
            header_print("ERROR", "Prompt insertion failed");
            return 1;
        }
        std::string response = chat->generate(meta_info, length_limit, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }

    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;
    std::cout << std::endl;
    for (auto t: history.second){
        std::cout << t << " ";
    }
    std::cout << std::endl;

    return 0;
}
