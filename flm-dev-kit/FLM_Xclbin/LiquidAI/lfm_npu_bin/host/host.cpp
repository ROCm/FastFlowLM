// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "llama/llama_npu.hpp"
#include "chat/chat_bot.hpp"
#include "model_list.hpp"

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();

    std::cout << "Model: " << tag << std::endl;
    std::string model_list_path = "../../../../../model_list.json";
    std::string exe_dir = ".";
    model_list model_list(model_list_path, exe_dir);
   
    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag);

    auto chat = std::make_shared<chat_bot>(0);
   
    chat->load_model(model_path, model_info);
    chat_meta_info meta_info;

    if (short_prompt) {
        std::string prompt = "Hello! What is the capital of USA?";
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::vector<int> tokens = chat->tokenize(prompt, true, "user", true);
        std::string response = chat->generate_with_prompt(meta_info, tokens, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
        prompt = "!@#$%^&*";
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        tokens = chat->tokenize(prompt, true, "user", true);
        response = chat->generate_with_prompt(meta_info, tokens, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
        prompt = "How far is it from Beijing?";
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        tokens = chat->tokenize(prompt, true, "user", true);
        response = chat->generate_with_prompt(meta_info, tokens, 1024, std::cout);
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
        std::string prompt;
        file.seekg(0, std::ios::end);
        prompt.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(prompt.data(), prompt.size());
        file.close();
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::vector<int> tokens = chat->tokenize(prompt, true, "user", true);
        std::string response = chat->generate_with_prompt(meta_info, tokens, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
    std::string prompt = "!@#$%^&*";
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Response: ";
    chat->start_total_timer();
    std::vector<int> tokens = chat->tokenize(prompt, true, "user", true);
    std::string response = chat->generate_with_prompt(meta_info, tokens, 1024, std::cout);
    chat->stop_total_timer();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << chat->show_profile() << std::endl;
    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;
    std::cout << std::endl;
    std::cout << "History: " << history.first << std::endl;
    std::cout << "History tokens: "  << std::endl;
    for (int token : history.second){
        std::cout << token << " ";
    }

    return 0;
}