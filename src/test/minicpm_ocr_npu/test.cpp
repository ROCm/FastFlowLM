/// \file test.cpp
/// \brief standalone harness for MiniCPM_OCR
/// \note Four turns: a CLI-style prompt, a REST-style follow-up that restores
///       the prompt-cache checkpoint, then clear_context() and a fresh prompt
///       with thinking on, then an image. Each turn reports the token that
///       stopped it.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_minicpm_ocr.hpp"
#include "AutoModel/flm_backend.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \note builtin_backends.cpp names every engine and so links every prebuilt
///       library; this harness registers just the one it drives.
namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("minicpm-ocr", kFlmBackendId, flm_factory<minicpm_ocr_npu>());
}
}  // namespace flm::backend

/// \brief Name a picture the way a person would: "720p", "1080p", ...
/// \note  The label keys off the SHORT side, which is what the p-numbers mean --
///        960x720 and 1280x720 are both 720p. It is a label, not a measurement:
///        anything that is not within 5% of a standard height is reported by its
///        pixel count instead of being rounded into the nearest bucket.
static std::string resolution_label(int w, int h) {
    const int shortest = std::min(w, h);
    static const int kStandard[] = {480, 576, 720, 1080, 1440, 2160, 4320};
    for (int s : kStandard) {
        if (std::abs(shortest - s) * 20 <= s) return std::to_string(s) + "p";
    }
    std::ostringstream mp;
    mp.precision(1);
    mp << std::fixed << (w * (double)h / 1e6) << " MP";
    return mp.str();
}

/// \brief Decode the image header and say how big it is.
/// \note  MiniCPM's TTFT is driven by how many VIEWS the slicer makes, which is a
///        function of both the pixel count and the aspect ratio (get_sliced_grid
///        picks a grid, capped by max_slice_nums), so print both. The view count
///        itself is private to MiniCPM_OCR -- read it off the "vision:" line that
///        FLM_MINICPM_OCR_VISION_PROFILE=1 prints.
static void report_image(const std::string& path) {
    ImageReader reader;
    image_data_t decoded;
    if (!reader.load_image(path, decoded)) {
        header_print("WARNING", "could not decode " << path << " to report its size");
        return;
    }
    const int w = decoded.width, h = decoded.height;
    reader.recycle(decoded);
    std::ostringstream aspect;
    aspect.precision(2);
    aspect << std::fixed << (w / (double)h);
    header_print("info", "image: " << w << "x" << h << " (" << resolution_label(w, h)
                 << ", aspect " << aspect.str() << ")");
}

static void report_turn(AutoModel& chat, const chat_meta_info_t& meta_info) {
    std::vector<int> history = chat.get_history().second;
    std::cout << std::endl;
    std::cout << "prompt_tokens: " << meta_info.prompt_tokens
              << ", cached_prompt_tokens: " << meta_info.cached_prompt_tokens
              << ", generated_tokens: " << meta_info.generated_tokens
              << ", stop_reason: " << stop_reason_to_string(meta_info.stop_reason) << std::endl;
    std::cout << "last token: " << (history.empty() ? -1 : history.back())
              << ", context length: " << chat.get_current_context_length() << std::endl;
    std::cout << chat.show_profile() << std::endl;
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("minicpm-ocr:1b"), "Model tag");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(256), "Max generated tokens per turn");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    // A single timed turn on a prompt from a file, for measuring prefill against length
    // (models_simple/test_input/{1k,2k,8k}.txt). Skips the four scripted turns entirely.
    desc.add_options()("prompt-file", arg_utils::po::value<std::string>()->default_value(""),
                       "Run one turn on this file's contents and exit");
    // Default is the repo's own test image, found relative to the executable the
    // same way model_list.json is. Pass --image "" to skip turn 4 outright.
    desc.add_options()("image,i", arg_utils::po::value<std::string>()->default_value("<default>"),
                       "Image for turn 4; \"\" skips it");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    int length = vm["Length"].as<int>();
    bool preemption = vm["Preemption"].as<bool>();
    std::cout << "Model: " << tag << std::endl;
    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);

    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag).second;
    std::cout << "Model path: " << model_path << std::endl;

    std::unique_ptr<AutoModel> chat = std::make_unique<MiniCPM_OCR>(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");
    chat->set_topk(1);
    chat->configure_parameter("enable_think", false);

    // --prompt-file: one turn, nothing else, so the reported prefill is this prompt alone.
    const std::string prompt_file = vm["prompt-file"].as<std::string>();
    if (!prompt_file.empty()) {
        std::ifstream pf(prompt_file);
        if (!pf) { header_print("ERROR", "cannot open " << prompt_file); return 1; }
        std::stringstream buf; buf << pf.rdbuf();
        lm_uniform_input_t one;
        one.prompt = buf.str();
        std::cout << "Prompt file: " << prompt_file << " (" << one.prompt.size() << " bytes)" << std::endl;
        chat_meta_info_t m;
        chat->start_total_timer();
        chat->insert(m, one);
        chat->generate(m, length, std::cout);
        chat->stop_total_timer();
        report_turn(*chat, m);
        return 0;
    }

    // Turn 1: a pure prompt, as `flm run` sends it.
    lm_uniform_input_t input;
    input.prompt = "My name is Ada and I live in Lisbon. In one sentence, what is Lisbon known for?";
    std::cout << "Prompt: " << input.prompt << std::endl << "Response: " << std::endl;
    chat_meta_info_t meta_info;
    chat->start_total_timer();
    chat->insert(meta_info, input);
    std::string answer = chat->generate(meta_info, length, std::cout);
    chat->stop_total_timer();
    report_turn(*chat, meta_info);

    // Turn 2: the whole conversation, as `flm serve` sends it on a prompt-cache
    // hit. restore() rewinds to the end of turn 1's prompt, so only the
    // assistant reply and the new user turn should be prefilled.
    lm_uniform_input_t follow_up;
    follow_up.messages = nlohmann::ordered_json::array();
    follow_up.messages.push_back({ {"role", "user"}, {"content", input.prompt} });
    follow_up.messages.push_back({ {"role", "assistant"}, {"content", answer} });
    follow_up.messages.push_back({ {"role", "user"}, {"content", "What is my name, and which city did I mention?"} });
    std::cout << "Prompt: " << follow_up.messages.back()["content"].get<std::string>() << std::endl << "Response: " << std::endl;
    chat_meta_info_t meta_info2;
    meta_info2.restore_allowed = true;
    chat->start_total_timer();
    chat->insert(meta_info2, follow_up);
    chat->generate(meta_info2, length, std::cout);
    chat->stop_total_timer();
    report_turn(*chat, meta_info2);

    // Turn 3: a fresh conversation with thinking on.
    chat->clear_context();
    chat->configure_parameter("enable_think", true);
    lm_uniform_input_t fresh;
    fresh.prompt = "What is 17 * 23? Answer briefly.";
    std::cout << "Prompt: " << fresh.prompt << std::endl << "Response: " << std::endl;
    chat_meta_info_t meta_info3;
    chat->start_total_timer();
    chat->insert(meta_info3, fresh);
    chat->generate(meta_info3, 4 * length, std::cout);
    chat->stop_total_timer();
    report_turn(*chat, meta_info3);

    // Turn 4: an image, as `flm run` sends it after /input. The vision tower only
    // runs if the package has one -- config.json needs vision_model_weight, which
    // is what sets is_vlm. Without it MiniCPM_OCR warns and drops the image, and
    // this turn degenerates to a text question about nothing, so say so rather
    // than let a 25-token prefill read as success.
    std::string image = vm["image"].as<std::string>();
    if (image == "<default>")
        image = (std::filesystem::path(exe_dir) / ".." / ".." / ".." / ".."
                 / ".github" / "script" / "assets" / "test_image1.jpg").lexically_normal().string();

    if (image.empty()) {
        header_print("info", "turn 4 (image) skipped: --image \"\"");
    } else if (!std::filesystem::exists(image)) {
        header_print("WARNING", "turn 4 (image) skipped: no such file: " << image);
    } else {
        chat->clear_context();
        chat->configure_parameter("enable_think", false);
        lm_uniform_input_t with_image;
        with_image.prompt = "Describe this image.";
        with_image.images.push_back(image);
        std::cout << "Image:  " << image << std::endl;
        report_image(image);
        std::cout << "Prompt: " << with_image.prompt << std::endl << "Response: " << std::endl;
        chat_meta_info_t meta_info4;
        chat->start_total_timer();
        chat->insert(meta_info4, with_image);
        chat->generate(meta_info4, length, std::cout);
        chat->stop_total_timer();
        report_turn(*chat, meta_info4);

        // The tower turns the repo's 960x720 test image into 5 views / 5040
        // patches / 315 rows, so a real image prefill is ~330 tokens against ~15
        // for the question alone. Anything near the latter means the image never
        // reached the tower -- a missing vision_model_weight, or a path the
        // reader could not decode -- which is otherwise easy to miss, because the
        // model still answers, just from the question alone.
        constexpr int kMinImagePromptTokens = 100;
        if (meta_info4.prompt_tokens < kMinImagePromptTokens) {
            header_print("WARNING", "prompt_tokens=" << meta_info4.prompt_tokens
                         << " is too low for an image turn; the image was NOT consumed"
                            " (is_vlm false, or the file failed to decode)");
        } else {
            header_print("info", "image consumed: " << meta_info4.prompt_tokens
                         << " prompt tokens (question alone would be ~15)");
        }
    }

    return 0;
}
