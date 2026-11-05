#include <iostream>
#include <cmath>
#include <sstream>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "AutoModel/flm_backend.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \note builtin_backends.cpp names every engine and so links every prebuilt library; this harness registers just
///       the one it drives.
namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("qwen3.5-flash", kFlmBackendId, flm_factory<qwen3_5vl_flash>());
}
}  // namespace flm::backend

// What one request gave back.
struct turn_result {
    std::string text;
    chat_meta_info_t meta;
    bool ok = false;
};

// One single-turn request: an optional system prompt, a user text, an optional image. Greedy decoding.
static turn_result run_turn(AutoModel& chat, const std::string& system_text, const std::string& user_text,
                            const std::string& image_path, int max_tokens) {
    turn_result r;
    lm_uniform_input_t input;
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    if (!system_text.empty()) {
        messages.push_back({{"role", "system"}, {"content", system_text}});
    }
    messages.push_back({{"role", "user"}, {"content", user_text}});
    input.messages = messages;
    if (!image_path.empty()) {
        input.images.push_back(image_path);
    }
    r.meta.max_prefill_len = 1024;
    std::ostringstream os;
    r.ok = chat.insert(r.meta, input);
    if (r.ok) {
        r.text = chat.generate(r.meta, max_tokens, os);
    }
    return r;
}

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) {
        failures++;
    }
}

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("qwen3.5-flash:2b"), "Model tag");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short run (one request) or the full check list");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_run = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
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

    std::unique_ptr<AutoModel> chat = std::make_unique<Qwen3_5VL_Flash>(&npu_device_global);
    npu_device_global = flm_rt::device(0);

    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");
    chat->set_topk(1);

    const std::string system_a = "You are a terse assistant. Answer in one short sentence.";
    const std::string system_b = "Answer in French, in one short sentence.";
    const std::string user_1 = "What is the capital of Japan?";
    const std::string user_2 = "Name one primary color.";
    const std::string image = "../../../tb_files/amd_256s.png";

    if (short_run) {
        turn_result r = run_turn(*chat, system_a, user_1, "", 32);
        std::cout << "Prompt: " << user_1 << std::endl << "Response: " << r.text << std::endl;
        std::cout << chat->show_profile() << std::endl;
        return r.ok ? 0 : 1;
    }

    // first request: the system prompt is pinned while it is served
    turn_result t1 = run_turn(*chat, system_a, user_1, "", 32);
    check(t1.ok && !t1.text.empty(), "first request answers");
    // the same request again: rewinds to the pin, same answer, the pinned tokens are reported as cached
    turn_result t2 = run_turn(*chat, system_a, user_1, "", 32);
    check(t2.ok && t2.text == t1.text, "same request again, same answer");
    check(t2.meta.cached_prompt_tokens > 0, "the pinned system prefix is served from the cache");
    // another user text under the same system prompt
    turn_result t3 = run_turn(*chat, system_a, user_2, "", 32);
    check(t3.ok && t3.meta.cached_prompt_tokens == t2.meta.cached_prompt_tokens, "other user text, same pin");
    // another system prompt: pinned afresh
    turn_result t4 = run_turn(*chat, system_b, user_1, "", 32);
    check(t4.ok && !t4.text.empty(), "other system prompt answers");
    // no system prompt: nothing pinned
    turn_result t5 = run_turn(*chat, "", user_1, "", 32);
    check(t5.ok && t5.meta.cached_prompt_tokens == 0, "no system prompt, nothing cached");
    // back to the first system prompt: same answer as the first time
    turn_result t6 = run_turn(*chat, system_a, user_1, "", 32);
    check(t6.ok && t6.text == t1.text, "first system prompt again, same answer as the first time");
    // a user text that merges with the end of the pinned prefix must not reuse a stale pin
    turn_result t7 = run_turn(*chat, system_a, "\n" + user_1, "", 32);
    check(t7.ok && !t7.text.empty(), "user text starting with a newline answers");
    turn_result t8 = run_turn(*chat, system_a, user_1, "", 32);
    check(t8.ok && t8.text == t1.text, "the request after it is still right");
    // clear, then the first request again
    chat->clear_context();
    turn_result t9 = run_turn(*chat, system_a, user_1, "", 32);
    check(t9.ok && t9.text == t1.text, "after clear_context, same answer as the first time");
    // an image
    turn_result i1 = run_turn(*chat, system_a, "Describe this image in brief.", image, 32);
    check(i1.ok && !i1.text.empty(), "image request answers");
    turn_result i2 = run_turn(*chat, system_a, "Describe this image in brief.", image, 32);
    check(i2.ok && i2.text == i1.text, "image request again, same answer");
    turn_result t10 = run_turn(*chat, system_a, user_1, "", 32);
    check(t10.ok && t10.text == t1.text, "text request after an image is not affected");

    std::cout << std::endl << chat->show_profile() << std::endl;
    std::cout << (failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
