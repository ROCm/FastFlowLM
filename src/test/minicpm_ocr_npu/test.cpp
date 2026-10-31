/// \file test.cpp
/// \brief standalone harness for MiniCPM_OCR
/// \note Four turns: a CLI-style prompt, a REST-style follow-up that restores
///       the prompt-cache checkpoint, then clear_context() and a fresh prompt
///       with thinking on, then an image. Each turn reports the token that
///       stopped it.
/// \note --multi-slot instead runs three interleaved conversations through the
///       prefix slot cache and checks each one gets its own state back (run_multi_slot).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "base64.hpp"
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

// ---------------------------------------------------------------------------------------------
// --multi-slot
// ---------------------------------------------------------------------------------------------

/// \brief MiniCPM_OCR plus the prefix-slot controls the test needs (all protected in AutoModel)
class MiniCPMOCRProbe : public MiniCPM_OCR {
public:
    using MiniCPM_OCR::MiniCPM_OCR;
    /// \brief turn the prefix slots on (-1 = the default count) or off (0), as --prefix-slots does
    void set_slots(int slots) { this->set_prefix_limits(-1, slots); }
    bool slots_enabled() const { return this->prefix_cache_.enabled(); }
    /// \brief back to a freshly loaded model: no live context, no saved snapshots
    void reset_everything() {
        this->prefix_cache_.forget_all();
        this->clear_context();
    }
    /// \brief what the slot cache did for the last request (valid only if it went through the slots)
    prefix_cache_report_t last_report() const { return this->prefix_cache_.report(); }
    size_t count_tokens(const std::string& text) { return this->tokenizer->encode(text).size(); }
};

/// \brief one conversation of the test: a first user message, then one follow-up question
struct ms_conversation_t {
    std::string name;
    nlohmann::ordered_json first_user;   //!< {"role": "user", "content": ..., maybe "images": [base64]}
    std::string question2;
};

/// \brief what one turn did
struct ms_turn_t {
    std::string reply;
    int prompt_tokens = 0;
    int cached_tokens = 0;      //!< served from a saved state instead of prefilled
    double prefill_ms = 0.0;
    bool slot_hit = false;      //!< resumed from a slot (only when the slots are on)
    int slots_used = 0;         //!< slots held after the turn (only when the slots are on)
};

/// \brief one request, sent the way `flm serve` sends every request
static ms_turn_t ms_run_turn(MiniCPMOCRProbe& chat, const std::string& title,
                             const nlohmann::ordered_json& messages, int length) {
    std::cout << "\n---- " << title << " ----" << std::endl;
    chat_meta_info_t m;
    // the server sets this on every request: the model may reuse what it already holds.
    // Without it neither the slots nor the single checkpoint are ever consulted.
    m.restore_allowed = true;
    lm_uniform_input_t in;
    in.messages = messages;
    chat.start_total_timer();
    chat.insert(m, in);
    ms_turn_t t;
    t.reply = chat.generate(m, length, std::cout);
    chat.stop_total_timer();
    t.prompt_tokens = m.prompt_tokens;
    t.cached_tokens = m.cached_prompt_tokens;
    t.prefill_ms = m.prefill_duration / 1e6;
    if (chat.slots_enabled()) {
        const prefix_cache_report_t r = chat.last_report();
        t.slot_hit = r.valid && r.hit;
        t.slots_used = r.slots_used;
    }
    std::cout << "\n    prompt " << t.prompt_tokens << " tokens, " << t.cached_tokens
              << " reused, " << t.prompt_tokens - t.cached_tokens << " prefilled in "
              << t.prefill_ms << " ms";
    if (chat.slots_enabled()) {
        std::cout << "; slot " << (t.slot_hit ? "HIT" : "miss") << ", " << t.slots_used << " slots held";
    }
    std::cout << std::endl;
    return t;
}

/// \brief turn 2 of a conversation: turn 1, its reply, and the follow-up question
static nlohmann::ordered_json ms_turn2_messages(const ms_conversation_t& c, const std::string& reply1) {
    nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
    msgs.push_back(c.first_user);
    msgs.push_back({ {"role", "assistant"}, {"content", reply1} });
    msgs.push_back({ {"role", "user"}, {"content", c.question2} });
    return msgs;
}

/// \brief a made-up inventory list of at least `min_tokens` tokens. Long enough (>= 1536) that its
///        prefill runs the gated delta rule on the NPU, while the short follow-up runs it on the
///        host -- so a resume has to hand the recurrent state from one backend to the other.
static std::string ms_long_document(MiniCPMOCRProbe& chat, size_t min_tokens) {
    static const char* kCities[] = {"Lisbon", "Oslo", "Lima", "Cairo", "Hanoi", "Quito", "Perth", "Riga"};
    static const char* kGoods[] = {"copper wire", "rice", "glass jars", "wool", "tea", "steel pipes", "paper", "salt"};
    std::string doc = "Warehouse inventory report.\n";
    int item = 1;
    while (chat.count_tokens(doc) < min_tokens) {
        // ten lines between token counts: the encode is over the whole document each time
        for (int k = 0; k < 10; k++, item++) {
            const int crates = 100 + (37 * item) % 900;
            doc += "Item " + std::to_string(item) + ": the warehouse in " + kCities[item % 8] + " holds "
                 + std::to_string(crates) + " crates of " + kGoods[(item * 3) % 8] + ".\n";
        }
    }
    return doc + "\nRead the report. How many items does it list? Answer in one sentence.";
}

/// \brief --multi-slot: do interleaved conversations each get their own state back from the slots?
/// \note  Three conversations, each a first turn plus a short follow-up:
///          A  a ~2k-token document (GDN on the NPU for turn 1, on the host for the follow-up)
///          B  a short prompt (GDN on the host throughout)
///          C  an image and a question about it (skipped with --image "")
///        Phase 1, reference: slots OFF, each conversation alone from a clean model, `rounds`
///          times. The single-checkpoint path rewinds turn 2 to the end of turn 1's prompt and
///          prefills the rest. Repeating it measures whether the engine repeats itself at all.
///        Phase 2, multi-slot: slots ON, all first turns, then all follow-ups (A1 B1 C1 A2 B2 C2).
///          Every follow-up must resume from ITS conversation's snapshot, at the same point the
///          reference resumed (same cached-token count, so the same tokens are prefilled), and
///          every reply must equal the reference's byte for byte.
///        Phase 3, timing only: slots OFF, same interleaved order -- what the slots save. Each
///          follow-up there has to prefill its whole conversation again.
///        Sampling is greedy with the penalties off, so a reply depends on the model state alone.
static int run_multi_slot(MiniCPMOCRProbe& chat, const std::vector<ms_conversation_t>& convs,
                          int length, int rounds) {
    chat.set_topk(1);
    chat.set_penalty_window(0);   // penalties off: resuming resets their history, the legacy path does not
    chat.configure_parameter("enable_think", false);
    const size_t n = convs.size();

    // ---- phase 1: reference, slots off, one conversation at a time ----
    std::vector<std::vector<ms_turn_t>> ref1(n), ref2(n);   // [conversation][round]
    chat.set_slots(0);
    for (int r = 0; r < rounds; r++) {
        for (size_t i = 0; i < n; i++) {
            chat.reset_everything();
            const std::string tag = "reference round " + std::to_string(r + 1) + ", " + convs[i].name;
            ref1[i].push_back(ms_run_turn(chat, tag + " turn 1", nlohmann::ordered_json::array({convs[i].first_user}), length));
            ref2[i].push_back(ms_run_turn(chat, tag + " turn 2", ms_turn2_messages(convs[i], ref1[i].back().reply), length));
        }
    }

    // ---- phase 2: slots on, interleaved ----
    chat.set_slots(-1);
    if (!chat.slots_enabled()) {
        header_print("ERROR", "the prefix slots did not turn on (FLM_DISABLE_PREFIX_SLOTS set, or no state api)");
        return 1;
    }
    chat.reset_everything();
    std::vector<ms_turn_t> ms1(n), ms2(n);
    for (size_t i = 0; i < n; i++) {
        ms1[i] = ms_run_turn(chat, "multi-slot " + convs[i].name + " turn 1", nlohmann::ordered_json::array({convs[i].first_user}), length);
    }
    for (size_t i = 0; i < n; i++) {
        // the SAME conversation as the reference's turn 2, so the prompts are byte identical
        ms2[i] = ms_run_turn(chat, "multi-slot " + convs[i].name + " turn 2", ms_turn2_messages(convs[i], ref1[i][0].reply), length);
    }

    // ---- phase 3: slots off, interleaved, timing only ----
    chat.set_slots(0);
    chat.reset_everything();
    std::vector<ms_turn_t> no1(n), no2(n);
    for (size_t i = 0; i < n; i++) {
        no1[i] = ms_run_turn(chat, "no slots " + convs[i].name + " turn 1", nlohmann::ordered_json::array({convs[i].first_user}), length);
    }
    for (size_t i = 0; i < n; i++) {
        no2[i] = ms_run_turn(chat, "no slots " + convs[i].name + " turn 2", ms_turn2_messages(convs[i], ref1[i][0].reply), length);
    }
    chat.set_slots(-1);   // back to the default for anything that runs after
    chat.reset_everything();

    // ---- verdict ----
    std::cout << "\n" << std::string(100, '=') << "\nmulti-slot summary\n" << std::string(100, '=') << std::endl;
    std::cout << "follow-up turn: tokens reused + prefill ms\n";
    std::printf("%-6s | %7s | %-22s | %-22s | %-22s | %s\n", "conv", "prompt", "reference (alone)",
                "slots on, interleaved", "slots off, interleaved", "replies vs reference");
    bool all_pass = true, all_repeat = true;
    for (size_t i = 0; i < n; i++) {
        // does the engine repeat itself? every reference round must agree
        bool repeats = true;
        for (int r = 1; r < rounds; r++) {
            repeats = repeats && ref1[i][r].reply == ref1[i][0].reply && ref2[i][r].reply == ref2[i][0].reply;
        }
        all_repeat = all_repeat && repeats;
        const ms_turn_t& ref = ref2[i][0];
        const bool hit = ms2[i].slot_hit && ms2[i].cached_tokens > 0;
        const bool same_split = ms2[i].cached_tokens == ref.cached_tokens;
        const bool same1 = ms1[i].reply == ref1[i][0].reply;
        const bool same2 = ms2[i].reply == ref.reply;
        const bool pass = hit && same_split && same1 && same2;
        all_pass = all_pass && pass;

        char a[64], b[64], c[64];
        std::snprintf(a, sizeof(a), "%5d + %7.1f ms", ref.cached_tokens, ref.prefill_ms);
        std::snprintf(b, sizeof(b), "%5d + %7.1f ms", ms2[i].cached_tokens, ms2[i].prefill_ms);
        std::snprintf(c, sizeof(c), "%5d + %7.1f ms", no2[i].cached_tokens, no2[i].prefill_ms);
        std::string verdict = std::string("turn 1 ") + (same1 ? "same" : "DIFFERS") + ", turn 2 " + (same2 ? "same" : "DIFFERS");
        if (!hit) verdict += ", NO SLOT HIT";
        else if (!same_split) verdict += ", resumed at a different point";
        if (!repeats) verdict += " (engine does not repeat itself here)";
        std::printf("%-6s | %7d | %-22s | %-22s | %-22s | %s\n", convs[i].name.c_str(), ref.prompt_tokens, a, b, c, verdict.c_str());
    }
    if (all_pass) {
        std::cout << "\nPASS: every interleaved follow-up resumed from its own conversation's snapshot, at the same"
                     " point as the reference, and every reply matches it byte for byte" << std::endl;
        return 0;
    }
    std::cout << (all_repeat
        ? "\nFAIL: the engine repeats itself without slots, but the interleaved run with slots differs"
        : "\nINCONCLUSIVE: the engine did not repeat itself across reference rounds, so an exact comparison"
          " cannot separate the slots from its own spread") << std::endl;
    return 1;
}

/// \brief read a whole file as bytes, base64-encoded (a message's "images" entry)
static std::string ms_file_base64(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream buf;
    buf << f.rdbuf();
    return base64::to_base64(buf.str());
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
    // With --prompt-file: a second turn, sent the way `flm serve` sends a follow-up (whole
    // conversation, restore_allowed). Turn one is sent CLI-style (no restore_allowed), so with
    // the prefix slots on (the default) it saves no snapshot, and turn two prefills the whole
    // conversation in two calls -- the head up to its last user turn, then the tail. Their
    // lengths pick the GDN backend of each, which is what makes it the test for handing the
    // recurrent state between the two inside one request.
    desc.add_options()("follow-up", arg_utils::po::value<std::string>()->default_value(""),
                       "With --prompt-file: file holding a follow-up user turn");
    // With --follow-up: send turn one with restore_allowed as well, so it saves a snapshot and
    // turn two resumes from it, prefilling only the reply and the new user turn -- the short
    // cached follow-up of a real chat.
    desc.add_options()("resume", arg_utils::po::bool_switch()->default_value(false),
                       "With --follow-up: turn two resumes from turn one's saved state");
    // Default is the repo's own test image, found relative to the executable the
    // same way model_list.json is. Pass --image "" to skip turn 4 (and --multi-slot's
    // image conversation) outright.
    desc.add_options()("image,i", arg_utils::po::value<std::string>()->default_value("<default>"),
                       "Image for turn 4 (and for --multi-slot's image conversation); \"\" skips it");
    desc.add_options()("multi-slot", arg_utils::po::bool_switch()->default_value(false),
                       "Run three interleaved conversations with the prefix slots off and on and compare (run_multi_slot)");
    desc.add_options()("rounds", arg_utils::po::value<int>()->default_value(2),
                       "With --multi-slot: reference rounds (>= 2 tells whether the engine repeats itself)");
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

    std::unique_ptr<MiniCPMOCRProbe> chat = std::make_unique<MiniCPMOCRProbe>(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");
    chat->set_topk(1);
    chat->configure_parameter("enable_think", false);

    // Default is the repo's own test image, found relative to the executable the
    // same way model_list.json is.
    std::string image = vm["image"].as<std::string>();
    if (image == "<default>")
        image = (std::filesystem::path(exe_dir) / ".." / ".." / ".." / ".."
                 / ".github" / "script" / "assets" / "test_image1.jpg").lexically_normal().string();

    if (vm["multi-slot"].as<bool>()) {
        std::vector<ms_conversation_t> convs;
        // A: long enough that turn 1 runs the GDN on the NPU (>= 1536 tokens), or --prompt-file
        std::string doc;
        const std::string doc_file = vm["prompt-file"].as<std::string>();
        if (!doc_file.empty()) {
            std::ifstream df(doc_file);
            if (!df) { header_print("ERROR", "cannot open " << doc_file); return 1; }
            std::stringstream buf; buf << df.rdbuf();
            doc = buf.str();
        } else {
            doc = ms_long_document(*chat, 2000);
        }
        convs.push_back({"A", { {"role", "user"}, {"content", doc} },
                         "Which city holds the crates listed as item 7, and what is in them? Answer in one sentence."});
        // B: short, the GDN on the host throughout
        convs.push_back({"B", { {"role", "user"}, {"content", "My name is Ada and I live in Lisbon. In one sentence, what is Lisbon known for?"} },
                         "What is my name, and which city did I mention?"});
        // C: a picture; the slot must only be reused for the same picture
        if (image.empty()) {
            header_print("info", "--multi-slot: image conversation skipped (--image \"\")");
        } else if (!std::filesystem::exists(image)) {
            header_print("WARNING", "--multi-slot: image conversation skipped, no such file: " << image);
        } else {
            nlohmann::ordered_json user = { {"role", "user"}, {"content", "Describe this image."} };
            user["images"] = nlohmann::ordered_json::array({ ms_file_base64(image) });
            convs.push_back({"C", user, "What is the main colour in it? Answer in a few words."});
        }
        return run_multi_slot(*chat, convs, length, std::max(1, vm["rounds"].as<int>()));
    }

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
        m.restore_allowed = vm["resume"].as<bool>();   // see --resume
        chat->start_total_timer();
        chat->insert(m, one);
        const std::string answer = chat->generate(m, length, std::cout);
        chat->stop_total_timer();
        report_turn(*chat, m);

        const std::string follow_file = vm["follow-up"].as<std::string>();
        if (!follow_file.empty()) {
            std::ifstream ff(follow_file);
            if (!ff) { header_print("ERROR", "cannot open " << follow_file); return 1; }
            std::stringstream fbuf; fbuf << ff.rdbuf();
            lm_uniform_input_t two;
            two.messages = nlohmann::ordered_json::array();
            two.messages.push_back({ {"role", "user"}, {"content", one.prompt} });
            two.messages.push_back({ {"role", "assistant"}, {"content", answer} });
            two.messages.push_back({ {"role", "user"}, {"content", fbuf.str()} });
            std::cout << "Follow-up file: " << follow_file << " (" << fbuf.str().size() << " bytes)" << std::endl;
            chat_meta_info_t m2;
            m2.restore_allowed = true;
            chat->start_total_timer();
            chat->insert(m2, two);
            chat->generate(m2, length, std::cout);
            chat->stop_total_timer();
            report_turn(*chat, m2);
        }
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
    // than let a 25-token prefill read as success. (`image` is resolved above.)
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
