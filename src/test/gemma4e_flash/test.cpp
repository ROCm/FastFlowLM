#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_gemma4e.hpp"
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
    desc.add_options()("type,t", arg_utils::po::value<int>()->default_value(0), "\t0: text mode\n\t1: image only\n\t2: audio only\n\t3: omni mode\n\t4: long prompt from prompt.txt\n\t");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Greedy,g", arg_utils::po::value<bool>()->default_value(true), "Greedy decoding");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(8192), "Max generation length");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    int type = vm["type"].as<int>();
    bool preemption = vm["Preemption"].as<bool>();
    bool greedy = vm["Greedy"].as<bool>();
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

    std::unique_ptr<AutoModel> chat = std::make_unique<Gemma4e_Flash>(&npu_device_global);
    std::cout << "Chat model initialized" << std::endl;
    npu_device_global = flm_rt::device(0);
    std::cout << "NPU Device initialized: " << npu_device_global.get_info<flm_rt::info::device::name>() << std::endl;
    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");

    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    if (greedy) {
        chat->set_topk(1); // deterministic (greedy) for a smoke test
    }

    switch (type) {
        case 0:
            uniformed_input.prompt = "Hello, introduce yourself briefly.";
            break;
        case 1:
            uniformed_input.prompt = "Describe the image briefly in 16 tokens";
            uniformed_input.images.push_back("../../../tb_files/amd_256s.png");
            length_limit = 16;
            break;
        case 2:
            uniformed_input.prompt = "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.";
            uniformed_input.audios.push_back("../../../tb_files/nvidia.mp3");
            break;
        case 3:
            uniformed_input.prompt = "Answer the question and further describe what is in the image.";
            uniformed_input.images.push_back("../../../tb_files/Zootopia.jpg");
            uniformed_input.audios.push_back("../../../tb_files/Recording.wav");
            break;
        case 4: {
            // Long prompt: read the text from prompt.txt; prefill only.
            std::ifstream file("../../../../prompt.txt", std::ios::binary);
            if (!file.is_open()) {
                std::cout << "Failed to open prompt file" << std::endl;
                return 1;
            }
            file.seekg(0, std::ios::end);
            uniformed_input.prompt.resize(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(uniformed_input.prompt.data(), uniformed_input.prompt.size());
            file.close();
            break;
        }
        default:
            header_print("info", "Unknown test type, exit 0;");
            exit(0);
    }

    std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
    std::cout << "Response: " << std::endl;

    chat->start_total_timer();
    if (type == 4) {
        // Prefill only -- this path exercises long-context prefill.
        chat->insert(meta_info, uniformed_input);
    } else {
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
    }
    chat->stop_total_timer();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << chat->show_profile() << std::endl;

    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;
    std::cout << std::endl;

    return 0;
}
