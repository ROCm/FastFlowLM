/// \file auto_decision_model.cpp
/// \brief AutoDecisionModel: the d3 contract around the d3_npu engine (see the header).
#include "AutoDecisionModel/auto_decision_model.hpp"

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "AutoDecisionModel/d3_engine_adapter.hpp"
#include "image_process_utils/imageproc.hpp"
#include "minja/chat-template.hpp"
#include "models/d3/flm/aie2p/d3_npu.hpp"   // d3_image_payload (the d3_npu class only with FLM_HAS_D3_NPU)
#include "utils/utils.hpp"
#ifdef FLM_D3_VIDEO
#include "video/video_input.hpp"
#endif

std::unordered_set<std::string> decisionModelTags = {"d3:edge", "d3:lite", "d3:nano", "d3:mini", "d3:flash", "d3:turbo", "d3:ultra"};

/// Decision models whose warmup passes are skipped: one 2,048-row pass of a 27B backbone is minutes of
/// NPU time, far more than the first-request latency it would hide. D3_WARMUP=1 / 0 overrides per run.
static const std::unordered_set<std::string> decisionNoWarmupTags = {"d3:turbo", "d3:ultra"};

namespace {

std::string read_text(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// readout.safetensors: {"weight": F32 [255, hidden]}
std::vector<float> load_readout(const std::filesystem::path& p, int& hidden) {
    const std::string blob = read_text(p);
    if (blob.size() < 8) throw std::runtime_error("readout.safetensors is truncated");
    uint64_t n = 0;
    std::memcpy(&n, blob.data(), 8);
    if (8 + n > blob.size()) throw std::runtime_error("readout.safetensors header is truncated");
    const auto header = nlohmann::json::parse(blob.substr(8, n));
    if (!header.contains("weight")) throw std::runtime_error("readout.safetensors has no weight tensor");
    const auto& w = header["weight"];
    const auto shape = w["shape"].get<std::vector<int64_t>>();
    const auto off = w["data_offsets"].get<std::vector<uint64_t>>();
    if (w["dtype"] != "F32" || shape.size() != 2 || shape[0] != flm::d3::MAX_OPTIONS) {
        throw std::runtime_error("readout weight must be F32 [255, hidden]");
    }
    hidden = static_cast<int>(shape[1]);
    const size_t bytes = off[1] - off[0];
    if (bytes != static_cast<size_t>(shape[0] * shape[1]) * sizeof(float) || 8 + n + off[1] > blob.size()) {
        throw std::runtime_error("readout weight data does not match its shape");
    }
    std::vector<float> out(static_cast<size_t>(shape[0] * shape[1]));
    std::memcpy(out.data(), blob.data() + 8 + n + off[0], bytes);
    return out;
}

// The Qwen2-VL image processor as d3-lite's processor configures it (preprocessor_config.json): RGB
// resized (bicubic, antialiased) to the grid smart_resize chose, rescaled by 1/255, normalized with
// mean = std = 0.5, the frame repeated for the 2-frame temporal patch, and cut into 16 x 16 patches in
// 2 x 2 merge order.
constexpr int TEMPORAL_PATCH = 2;
constexpr float RESCALE = 1.0f / 255.0f;
constexpr float IMAGE_MEAN = 0.5f;
constexpr float IMAGE_STD = 0.5f;

/// Pillow's resample coefficients for one axis (Resample.c precompute_coeffs, bicubic a = -0.5,
/// antialiased: the kernel widens by the downscale factor), as 22-bit fixed point
/// (normalize_coeffs_8bpc). bounds: per output pixel, first input index and tap count.
void resample_coeffs(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& coeffs, int& ksize) {
    constexpr int PRECISION_BITS = 32 - 8 - 2;
    const auto bicubic = [](double x) {
        const double a = -0.5;
        x = std::fabs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
    ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    bounds.assign(static_cast<size_t>(out_size) * 2, 0);
    coeffs.assign(static_cast<size_t>(out_size) * ksize, 0);
    std::vector<double> k(ksize);
    for (int xx = 0; xx < out_size; xx++) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; x++) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) {
            const double w = ww != 0.0 ? k[x] / ww : k[x];
            coeffs[static_cast<size_t>(xx) * ksize + x] =
                static_cast<int32_t>(w < 0 ? w * (1 << PRECISION_BITS) - 0.5 : w * (1 << PRECISION_BITS) + 0.5);
        }
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
}

/// Pillow's Image.resize((w, h), BICUBIC) of 8-bit RGB, which the processor's resize (torchvision,
/// uint8, bicubic, antialias) reproduces exactly: horizontal pass, then vertical, each rounded and
/// clipped to 8 bits. In and out are RGB, interleaved.
std::vector<uint8_t> pil_resize_bicubic(const uint8_t* src, int in_w, int in_h, int out_w, int out_h) {
    constexpr int PRECISION_BITS = 32 - 8 - 2;
    const auto clip8 = [](int64_t v) -> uint8_t {
        v >>= PRECISION_BITS;
        return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    };
    // horizontal (skipped when the width stays: Pillow resamples only the axes that change)
    std::vector<uint8_t> mid;
    const uint8_t* rows = src;
    if (out_w != in_w) {
        std::vector<int> b;
        std::vector<int32_t> k;
        int ks = 0;
        resample_coeffs(in_w, out_w, b, k, ks);
        mid.resize(static_cast<size_t>(out_w) * in_h * 3);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < in_h; y++) {
            const uint8_t* line = src + static_cast<size_t>(y) * in_w * 3;
            for (int xx = 0; xx < out_w; xx++) {
                const int xmin = b[xx * 2], n = b[xx * 2 + 1];
                const int32_t* kk = k.data() + static_cast<size_t>(xx) * ks;
                for (int c = 0; c < 3; c++) {
                    int64_t ss = int64_t(1) << (PRECISION_BITS - 1);
                    for (int x = 0; x < n; x++) ss += int64_t(line[(xmin + x) * 3 + c]) * kk[x];
                    mid[(static_cast<size_t>(y) * out_w + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        rows = mid.data();
    }
    if (out_h == in_h) return std::vector<uint8_t>(rows, rows + static_cast<size_t>(out_w) * in_h * 3);
    std::vector<int> b;
    std::vector<int32_t> k;
    int ks = 0;
    resample_coeffs(in_h, out_h, b, k, ks);
    std::vector<uint8_t> out(static_cast<size_t>(out_w) * out_h * 3);
    #pragma omp parallel for schedule(static)
    for (int yy = 0; yy < out_h; yy++) {
        const int ymin = b[yy * 2], n = b[yy * 2 + 1];
        const int32_t* kk = k.data() + static_cast<size_t>(yy) * ks;
        for (int x = 0; x < out_w * 3; x++) {
            int64_t ss = int64_t(1) << (PRECISION_BITS - 1);
            for (int y = 0; y < n; y++) ss += int64_t(rows[static_cast<size_t>(ymin + y) * out_w * 3 + x]) * kk[y];
            out[static_cast<size_t>(yy) * out_w * 3 + x] = clip8(ss);
        }
    }
    return out;
}

/// Append one decoded image's patches (RGB HWC, already decoded) to `out`, resized to grid_h x grid_w patches.
bool preprocess_image(const image_data_t& rgb, int grid_h, int grid_w, std::vector<bf16>& out) {
    const int w = grid_w * flm::d3::IMAGE_PATCH_SIZE, h = grid_h * flm::d3::IMAGE_PATCH_SIZE;
    const std::vector<uint8_t> resized = pil_resize_bicubic(rgb.pixels.data(), rgb.width, rgb.height, w, h);
    const size_t plane = static_cast<size_t>(w) * h, frame = plane * 3;
    std::vector<float> scratch(frame * TEMPORAL_PATCH);
    // rescale and normalize, planar (CHW): (x / 255 - mean) / std
    for (size_t i = 0; i < plane; i++) {
        for (int c = 0; c < 3; c++) {
            scratch[c * plane + i] = (static_cast<float>(resized[i * 3 + c]) * RESCALE - IMAGE_MEAN) / IMAGE_STD;
        }
    }
    for (int t = 1; t < TEMPORAL_PATCH; t++) std::memcpy(scratch.data() + t * frame, scratch.data(), frame * sizeof(float));
    const size_t at = out.size();
    out.resize(at + frame * TEMPORAL_PATCH);
    imgproc::reorder_patches_inplace(scratch.data(), out.data() + at, 1, 1, TEMPORAL_PATCH, 3, grid_h, grid_w,
                                     flm::d3::IMAGE_MERGE_SIZE, flm::d3::IMAGE_PATCH_SIZE);
    return true;
}

}  // namespace

AutoDecisionModel::AutoDecisionModel(flm_rt::device* npu_device_inst, std::string model_tag)
    : npu_device_inst_(npu_device_inst), tag_(std::move(model_tag)) {}

AutoDecisionModel::~AutoDecisionModel() = default;

std::vector<int> AutoDecisionModel::encode(const std::string& text) {
    return tokenizer_->encode(text);
}

void AutoDecisionModel::load_model(const std::string& model_path, const nlohmann::json& model_info,
                                   bool enable_preemption) {
    (void)model_info;
    namespace fs = std::filesystem;
    const fs::path root(model_path);
    header_print("FLM", "Loading decision model: " << model_path);

    // decision_config.json: the same checks as d3_runtime.D3.__init__
    const auto dc = nlohmann::json::parse(read_text(root / "decision_config.json"));
    if (dc.value("format_version", 0) != 1) throw std::runtime_error("decision_config.json: format_version must be 1");
    if (dc.value("format_id", std::string()) != flm::d3::FORMAT_ID) throw std::runtime_error("decision_config.json: unsupported format_id");
    if (dc.value("prompt", std::string("d3")) != "d3") throw std::runtime_error("decision_config.json: prompt must be d3");
    // noncausal_full_attention: every token sees every token (d3:edge, d3:lite, d3:nano, d3:mini, d3:flash);
    // prefix_bidir: the state block sees only itself, each question block sees the state and itself (d3:turbo)
    const std::string attention_mode = dc.value("attention_mode", std::string("causal"));
    if (attention_mode != "noncausal_full_attention" && attention_mode != "prefix_bidir") {
        throw std::runtime_error("decision_config.json: this engine runs noncausal_full_attention and prefix_bidir checkpoints only");
    }
    prefix_bidir_ = attention_mode == "prefix_bidir";
    {   // D3_STATE_ONCE=0 counts whole prompts in usage.input_tokens, as upstream's runtime does
        const char* once = std::getenv("D3_STATE_ONCE");
        state_once_ = !(once && std::string(once) == "0");
    }
    if (dc.value("pooling", std::string("last")) != "last") throw std::runtime_error("decision_config.json: pooling must be last");
    if (dc.value("readout_dtype", std::string("float32")) != "float32") throw std::runtime_error("decision_config.json: readout_dtype must be float32");
    temperature_ = dc.value("temperature", 1.0f);
    if (!(std::isfinite(temperature_) && temperature_ > 0)) throw std::runtime_error("decision_config.json: temperature must be finite and positive");
    temperature_by_questions_ = flm::d3::temperature_table(dc.contains("temperature_by_questions") ? dc["temperature_by_questions"] : nlohmann::json(nullptr),
                                                           "decision_config.json");
    {   // prompt_layout: absent / null (the plain layout) or "header-v1"; anything else is refused, as upstream does
        const auto layout = dc.contains("prompt_layout") ? dc["prompt_layout"] : nlohmann::json(nullptr);
        if (!layout.is_null() && !(layout.is_string() && layout.get<std::string>() == "header-v1")) {
            throw std::runtime_error("decision_config.json: unknown prompt_layout " + layout.dump());
        }
        header_v1_ = !layout.is_null();
    }
    // validated_length is the longest prompt the checkpoint was validated on upstream (d3-ultra: 131,072); this build's cap
    // is MAX_INPUT_TOKENS (what the NPU engine was measured to run) and stays there whatever the package says.
    if (dc.contains("validated_length") && dc["validated_length"].is_number_integer() &&
        dc["validated_length"].get<long long>() > MAX_INPUT_TOKENS) {
        header_print("FLM", "decision_config.json validated_length " << dc["validated_length"].get<long long>()
                                                                      << " is above this build's cap; max input stays " << MAX_INPUT_TOKENS << " tokens");
    }
    const auto codes = dc["codes"].get<std::vector<std::string>>();
    const auto token_ids = dc["token_ids"].get<std::vector<int>>();

    // the tokenizer must give exactly the package's answer codes (upstream refuses otherwise)
    tokenizer_ = std::make_unique<Tokenizer>(model_path);
    std::vector<std::string> got_codes;
    std::vector<int> got_ids;
    flm::d3::answer_codes([this](const std::string& s) { return encode(s); }, got_codes, got_ids);
    if (got_codes != codes || got_ids != token_ids) {
        throw std::runtime_error("the answer codes or token ids in decision_config.json differ from the ones this tokenizer gives");
    }
    if (encode("State:\nA") != encode("State:\nA")) throw std::runtime_error("tokenizer is not deterministic");
    codes_ = codes;

    // the package's chat template must render the d3 messages exactly as the contract does
    {
        std::string tmpl;
        if (fs::exists(root / "chat_template.jinja")) {
            tmpl = read_text(root / "chat_template.jinja");
        } else {
            tmpl = nlohmann::json::parse(read_text(root / "tokenizer_config.json")).value("chat_template", std::string());
        }
        minja::chat_template chat(tmpl, "", "");
        minja::chat_template_inputs in;
        in.add_generation_prompt = true;
        in.extra_context = nlohmann::ordered_json{{"enable_thinking", false}};
        const std::string user = "State:\n(empty)\n\nQuestion:\nprobe\n\nOptions:\nA: x\n\nReply with only the code of the best option.";
        in.messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", flm::d3::SYSTEM_PROMPT}},
            {{"role", "user"}, {"content", user}},
        });
        if (chat.apply(in) != flm::d3::chat_text(flm::d3::SYSTEM_PROMPT, user)) {
            throw std::runtime_error("chat_template.jinja renders the d3 prompt differently from the d3 contract");
        }
        // and the image prompt (image_messages: the images, then the text)
        in.messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", flm::d3::SYSTEM_PROMPT}},
            {{"role", "user"}, {"content", nlohmann::ordered_json::array({
                {{"type", "image"}}, {{"type", "image"}}, {{"type", "text"}, {"text", user}}})}},
        });
        const std::string pair = std::string("<|vision_start|>") + flm::d3::IMAGE_TOKEN + "<|vision_end|>";
        if (chat.apply(in) != flm::d3::chat_text(flm::d3::SYSTEM_PROMPT, pair + pair + user)) {
            throw std::runtime_error("chat_template.jinja renders the d3 image prompt differently from the d3 contract");
        }
    }

    readout_ = load_readout(root / "readout.safetensors", hidden_);

    // composed requests (assets, box): a prefix_bidir checkpoint declares what it answers in `systemone`
    if (prefix_bidir_) {
        composed_ = true;
        const auto so = dc.value("systemone", nlohmann::json::object());
        composed_config_.tag = tag_;
        composed_config_.max_length = MAX_INPUT_TOKENS;
        composed_config_.temperature = temperature_;
        composed_config_.temperature_by_questions = temperature_by_questions_;
        composed_config_.question_types = {"choice", "noul", "score"};
        for (const char* kind : {"box", "track"}) {
            for (const auto& declared : so.value("question_types", nlohmann::json::array())) {
                if (declared == kind) composed_config_.question_types.push_back(kind);
            }
        }
        {   // A2: upstream refuses `score` questions in any composed request; D3_COMPOSED_SCORE=1 answers them instead
            const char* score = std::getenv("D3_COMPOSED_SCORE");
            composed_config_.refuse_score = !(score && std::string(score) == "1");
        }
        composed_config_.image_placements = {"state"};
        for (const char* place : {"instructions", "criteria"}) {
            for (const auto& declared : so.value("image_placements", nlohmann::json::array())) {
                if (declared == place) composed_config_.image_placements.push_back(place);
            }
        }
        if (so.contains("box_calibration") && so["box_calibration"].is_array()) {
            for (const auto& knot : so["box_calibration"]) {
                composed_config_.box_calibration.emplace_back(knot.at(0).get<double>(), knot.at(1).get<double>());
            }
        }
    }

#ifdef FLM_D3_VIDEO
    // videos: decoded on the host (FFmpeg, loaded at run time). Plain questions read them in the state (every temporal
    // group goes to the engine as a visual run; see system_one for the plain path), a box at a moment / a track reads the
    // clip window by window (one state-once call per window, system_one_composed).
    if (prefix_bidir_) {
        const char* env = std::getenv("D3_VIDEO");
        video_ = !(env && std::string(env) == "0");
    }
    if (video_) {
        flm::d3::video::VideoLoadOptions& load = video_options_;
        load.backend = flm::d3::video::make_ffmpeg_backend();
        load.decode_image = [this](const flm::d3::ImageFile& f, flm::d3::video::RgbFrame& out, std::string* why) {
            image_data_t rgb;
            if (!image_reader_.load_image_bytes(f.data, rgb)) {
                if (why) *why = "the file is truncated or corrupt";
                return false;
            }
            out.width = rgb.width;
            out.height = rgb.height;
            out.rgb.assign(rgb.pixels.data(), rgb.pixels.data() + static_cast<size_t>(rgb.width) * rgb.height * 3);
            image_reader_.recycle(rgb);
            return true;
        };
        video_loader_ = flm::d3::video::make_video_loader(load);
    }
#endif

    // the engine
    config_ = std::make_unique<LM_Config>();
    config_->from_pretrained(model_path);
#ifdef FLM_HAS_D3_NPU
    // the header this build mirrors and the library it loaded must be the same revision (d3_npu.hpp); a build
    // against a revision 1 library (FLM_D3_NPU_REV2 undefined, see CMakeLists.txt) runs the per-row path only
    unsigned engine_caps = 0;
    std::string engine_rev = "revision 1";
#ifdef FLM_D3_NPU_REV2
    if (d3_npu_abi_version() != D3_NPU_ABI_VERSION) {
        throw std::runtime_error("libd3_npu is revision " + std::to_string(d3_npu_abi_version()) +
                                 " but this FastFlowLM was built for revision " + std::to_string(D3_NPU_ABI_VERSION));
    }
    engine_caps = d3_npu_capabilities();
    engine_rev = "revision " + std::to_string(d3_npu_abi_version()) + ", capabilities " + std::to_string(engine_caps);
#endif
    header_print("FLM", "d3 engine: " << engine_rev
                 << ((engine_caps & 1u) && state_once_ ? " (state once per request)" : " (per-row prefix_bidir)"));
    npu_ = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, npu_device_inst_, enable_preemption);
    auto engine = std::make_unique<d3_npu>(*config_, npu_.get(), MAX_INPUT_TOKENS);
    {
        Q4NX q4nx(model_path);
        engine->load_weights(q4nx);
    }
    if (engine->hidden_size() != hidden_) throw std::runtime_error("readout width differs from the model's hidden size");
    d3_npu* engine_raw = engine.get();
    engine_ = std::move(engine);
    readout_engine_ = make_d3_engine_adapter(engine_.get(), engine_raw, engine_caps, state_once_, encode(flm::d3::IMAGE_TOKEN).at(0));
    if (composed_) {
        // a box reads four hidden rows of one block (readout_from_end): only the state-once engine returns them
        // (D3_CAP_STATE_ONCE | D3_CAP_READOUT_ROWS); without it box / track are answered per question as unsupported
        const bool rows = state_once_ && (engine_caps & (D3_CAP_STATE_ONCE | D3_CAP_READOUT_ROWS)) == (D3_CAP_STATE_ONCE | D3_CAP_READOUT_ROWS);
        auto& types = composed_config_.question_types;
        composed_config_.box_question_images = (engine_caps & D3_CAP_QUESTION_IMAGES) != 0;
        if (!rows) {
            types.erase(std::remove_if(types.begin(), types.end(), [](const std::string& t) { return t == "box" || t == "track"; }), types.end());
            header_print("FLM", "d3 engine without state-once readout rows: box questions are not answered");
        } else if (!video_) {
            // a track reads a video: without the decoder it is not declared (the models listing, the refusal)
            types.erase(std::remove(types.begin(), types.end(), std::string("track")), types.end());
        }
        composed_config_.video = video_;
        composed_config_.refuse_video_questions = !video_;
        if (video_ && rows) {
            if (const char* limit = std::getenv("D3_MOMENT_ROWS")) composed_config_.moment_row_limit = std::max(1, std::atoi(limit));
        }
#ifdef FLM_D3_NPU_REV2
        header_print("FLM", "d3 engine capabilities: state-once " << ((engine_caps & D3_CAP_STATE_ONCE) ? "yes" : "no")
                     << ", readout rows " << ((engine_caps & D3_CAP_READOUT_ROWS) ? "yes" : "no")
                     << ", images in questions " << ((engine_caps & D3_CAP_QUESTION_IMAGES) ? "yes" : "no")
                     << ", video " << ((engine_caps & D3_CAP_VIDEO) ? "yes" : "no (video pads are rewritten to image pads)"));
#endif
    }
#else
    (void)enable_preemption;
    throw std::runtime_error("this FastFlowLM build has no d3 engine (libd3_npu)");
#endif

    // replies name the upstream model, as d3_server does ("model": "d3-edge", "d3-lite", "d3-nano", ...);
    // without a marker in config.json the name is the tag with its colon turned into a dash
    std::string tag_name = tag_;
    std::replace(tag_name.begin(), tag_name.end(), ':', '-');
    const auto& marker = config_->sub("decision");
    name_ = marker.is_object() ? marker.value("model_name", tag_name) : tag_name;
    composed_config_.model_name = name_;
    header_print("FLM", "Decision model ready: " << name_ << " (" << tag_ << "), max input " << MAX_INPUT_TOKENS << " tokens");
}

std::map<std::string, std::vector<double>> AutoDecisionModel::run(const flm::d3::Prepared& prepared, void* payload) {
    const auto keys = prepared.runnable();
    std::map<std::string, std::vector<double>> out;
    if (keys.empty()) return out;
    if (prefix_bidir_) return run_state_once(prepared, static_cast<const d3_image_payload*>(payload));
    std::vector<std::vector<int>> ids;
    ids.reserve(keys.size());
    for (const auto& k : keys) ids.push_back(prepared.sequences.at(k));
    // moved, never copied: XRT buffer copies do not own their memory
    std::vector<buffer<bf16>> hidden = engine_->encode_batch(ids, payload);
    std::vector<float> h(hidden_);
    // temperature_by_questions: the request's number of (valid) choice / noul / score questions; the plain path has no box / track
    const float request_temperature =
        flm::d3::temperature_for(temperature_by_questions_, temperature_, static_cast<int>(prepared.questions.size()));
    for (size_t i = 0; i < keys.size(); i++) {
        const bf16* row = hidden[i].data();
        for (int d = 0; d < hidden_; d++) h[d] = static_cast<float>(row[d]);
        const int count = static_cast<int>(prepared.questions.at(keys[i]).keys.size());
        auto p = flm::d3::readout_probabilities(h.data(), readout_.data(), hidden_, count, request_temperature);
        out[keys[i]] = std::vector<double>(p.begin(), p.end());
    }
    return out;
}

/// prefix_bidir (d3:turbo) plain request: every runnable question is `[state | block]` with the same state block
/// (upstream's _run_state_once refuses anything else), so the request is ONE state and the questions are its blocks.
/// The adapter runs it as d3_npu::encode_state_once when the library reports the capability (D3_CAP_STATE_ONCE; the
/// state's attention K/V and GDN state are computed once), else as the M1 per-row batch (every row [state | block]
/// with payload.state_len; D3_STATE_ONCE=0 forces that). Both give the same hidden rows.
std::map<std::string, std::vector<double>> AutoDecisionModel::run_state_once(const flm::d3::Prepared& prepared,
                                                                            const d3_image_payload* payload) {
    const auto keys = prepared.runnable();
    std::map<std::string, std::vector<double>> out;
    if (keys.empty()) return out;
    // the request's images all lie inside the state block (plain questions carry none of their own); the payload
    // holds their patches back to back, 3 x 2 x 16 x 16 values per patch
    std::vector<flm::d3::EngineImage> images;
    if (payload) {
        constexpr size_t PER_PATCH = 3 * TEMPORAL_PATCH * flm::d3::IMAGE_PATCH_SIZE * flm::d3::IMAGE_PATCH_SIZE;
        size_t at = 0;
        for (const auto& image : payload->images) {
            flm::d3::EngineImage im;
            im.grid_h = image.grid_h;
            im.grid_w = image.grid_w;
            const size_t n = static_cast<size_t>(im.grid_h) * im.grid_w * PER_PATCH;
            if (at + n > payload->pixels.size()) throw std::runtime_error("image payload is shorter than its grids");
            im.pixels.resize(n);
            static_assert(sizeof(bf16) == sizeof(uint16_t), "bf16 is 16 bits");
            std::memcpy(im.pixels.data(), payload->pixels.data() + at, n * sizeof(uint16_t));
            at += n;
            images.push_back(std::move(im));
        }
    }
    const flm::d3::EngineStateRequest request =
        flm::d3::plain_state_request(prepared, std::move(images), payload ? payload->image_token_id : -1);
    const std::vector<std::vector<float>> hidden = readout_engine_->encode_state_once(request);
    return flm::d3::plain_probabilities(prepared, hidden, readout_.data(), hidden_,
                                         flm::d3::temperature_for(temperature_by_questions_, temperature_, static_cast<int>(prepared.questions.size())));
}

DecisionReply AutoDecisionModel::system_one(const std::string& raw_body) {
    DecisionReply reply;
    flm::d3::PyValue body;
    try {
        body = flm::d3::parse_json(raw_body);
    } catch (const flm::d3::JsonError& e) {
        reply.status = 422;
        reply.body = {{"detail", std::string("could not read the body as JSON (") + e.what() + ")"}};
        return reply;
    }
    // assets and box / track questions: the composed path (upstream _system_one's dispatch)
    if (composed_ && flm::d3::is_composed(body)) return system_one_composed(body);
    flm::d3::Prepared prepared;
    // each image is decoded right after its header passes (upstream loads them one by one)
    std::vector<image_data_t> decoded;
    const auto decode = [&](size_t, const flm::d3::ImageFile& f) {
        image_data_t rgb;
        if (!image_reader_.load_image_bytes(f.data, rgb)) {
            throw flm::d3::ContractError("the image could not be decoded (the file is truncated or corrupt)");
        }
        if (rgb.width != f.width || rgb.height != f.height) {
            image_reader_.recycle(rgb);
            throw flm::d3::ContractError("the image could not be decoded (its pixels do not match its header)");
        }
        decoded.push_back(std::move(rgb));
    };
    const auto release = [&]() {
        for (auto& d : decoded) image_reader_.recycle(d);
        decoded.clear();
    };
    try {
        flm::d3::PrepareOptions options;
        options.header_v1 = header_v1_;
        if (prefix_bidir_) {
            options.unsupported_for = tag_;   // long_context / videos (and track): not in this build
            options.composed = composed_;
            options.reasons = true;           // d3-turbo's runtime adds `reason` to every error answer
            const auto& types = composed_config_.question_types;
            if (video_ && std::find(types.begin(), types.end(), std::string("track")) != types.end()) {
                options.answers = "choice, noul, score, box and track questions about text, images and video";
            }
            if (video_) options.load_video = video_loader_;
        }
        prepared = flm::d3::prepare(body, codes_, decode, options);
    } catch (const flm::d3::RequestError& e) {
        release();
        reply.status = 422;
        reply.body = {{"detail", e.detail}};
        return reply;
    }
    // prefix_bidir: the state block length of every question (the tokenizer has no offsets API, so
    // from token byte lengths: d3_contract state_block_length)
    std::function<size_t(int)> token_bytes;
    if (prefix_bidir_) token_bytes = [this](int id) { return tokenizer_->token_byte_length(id); };
    flm::d3::tokenize(prepared, [this](const std::string& s) { return encode(s); }, MAX_INPUT_TOKENS, token_bytes);
    const auto invalid = flm::d3::invalid_questions(prepared);
    if (!invalid.empty()) {
        release();
        reply.status = 422;
        reply.body = {{"detail", {{"invalid_questions", invalid}}}};
        return reply;
    }
    // the processor's pixels, once per request (every question shares the images)
    d3_image_payload payload;
    for (size_t i = 0; i < decoded.size(); i++) {
        const auto [gh, gw] = prepared.image_grids[i];
        if (!preprocess_image(decoded[i], gh, gw, payload.pixels)) {
            release();
            throw std::runtime_error("could not prepare image " + std::to_string(i));
        }
        payload.images.push_back({gh, gw});
    }
    release();
#ifdef FLM_D3_VIDEO
    if (!prepared.videos.empty()) {
        // The engine (revision 1 and 2) knows image runs only. Every temporal group of a video is one run of
        // grid_h/2 x grid_w/2 tokens whose positions are (s, s + row, s + col) and whose pixels are one
        // [3][2][16][16] patch grid: exactly an image's (Qwen3.5 get_rope_index splits a video per group; the vision
        // tower attends within a group), so the groups follow the request's images in the payload and the video pad
        // ids become the image pad id (the rows are replaced by the vision features anyway).
        flm::d3::video::VideoPayload videos;
        for (size_t v = 0; v < prepared.videos.size(); v++) {
            flm::d3::video::append_video(videos, *prepared.videos[v], prepared.video_grids[v]);
        }
        const size_t at = payload.pixels.size();
        payload.pixels.resize(at + videos.pixels.size());
        static_assert(sizeof(bf16) == sizeof(uint16_t), "bf16 is 16 bits");
        std::memcpy(payload.pixels.data() + at, videos.pixels.data(), videos.pixels.size() * sizeof(uint16_t));
        for (const auto& g : videos.groups) payload.images.push_back({g.first, g.second});
        const int image_pad = encode(flm::d3::IMAGE_TOKEN).at(0), video_pad = encode(flm::d3::VIDEO_TOKEN).at(0);
        for (auto& entry : prepared.sequences) {
            for (int& id : entry.second) if (id == video_pad) id = image_pad;
        }
    }
#endif
    if (!payload.images.empty()) {
        const std::vector<int> pad = encode(flm::d3::IMAGE_TOKEN);
        payload.image_token_id = pad.at(0);
        if (const char* dir = std::getenv("D3_DUMP_PIXELS")) {
            // rig: the processor's output, to compare with the upstream processor's pixel_values
            const std::string stem = std::string(dir) + "/request_" + std::to_string(dump_count_++);
            std::ofstream(stem + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(payload.pixels.data()), payload.pixels.size() * sizeof(bf16));
            nlohmann::json grids = nlohmann::json::array();
            for (const auto& im : payload.images) grids.push_back({1, im.grid_h, im.grid_w});
            std::ofstream(stem + ".json") << grids.dump();
        }
    }
    std::lock_guard<std::mutex> lock(turn_);
    // old engines read a non-null payload as "has images": text of lite / nano / mini passes nullptr
    const auto probabilities = run(prepared, (payload.images.empty() && !prefix_bidir_) ? nullptr : &payload);
    reply.body = flm::d3::respond(prepared, probabilities, name_, state_once_);
    return reply;
}

nlohmann::json AutoDecisionModel::modalities() const {
    // what this build reads: text and images; video in the state when the build has the decoder (FLM_VIDEO_FFMPEG)
    return video_ ? nlohmann::json::array({"text", "image", "video"}) : nlohmann::json::array({"text", "image"});
}

std::vector<std::string> AutoDecisionModel::question_types() const {
    std::vector<std::string> out;
    if (!composed_) return out;
    for (const auto& t : composed_config_.question_types) out.push_back(t);
    return out;
}

nlohmann::json AutoDecisionModel::video_limits() const {
    if (!video_) return nullptr;
    // d3_server.models(): video_contract() and MAX_VIDEO_BYTES
    return {{"fps", flm::d3::video::VIDEO_FPS},
            {"max_frames", flm::d3::video::VIDEO_MAX_FRAMES},
            {"max_pixels_per_frame", flm::d3::video::VIDEO_MAX_PIXELS},
            {"max_tokens_per_request", flm::d3::video::VIDEO_MAX_TOKENS},
            {"max_bytes", flm::d3::video::MAX_VIDEO_BYTES}};
}

DecisionReply AutoDecisionModel::system_one_composed(const flm::d3::PyValue& body) {
    DecisionReply reply;
    // images by table index: decoded when their header passes (the table also holds video groups, so by the index the
    // contract passes), preprocessed when an engine request needs them
    std::map<size_t, image_data_t> decoded;
    const auto release = [&]() {
        for (auto& d : decoded) image_reader_.recycle(d.second);
        decoded.clear();
    };
    flm::d3::ComposedHooks hooks;
    hooks.encode = [this](const std::string& s) { return encode(s); };
    hooks.token_bytes = [this](int id) { return tokenizer_->token_byte_length(id); };
    hooks.decode = [&](size_t index, const flm::d3::ImageFile& f) {
        image_data_t rgb;
        if (!image_reader_.load_image_bytes(f.data, rgb)) {
            throw flm::d3::ContractError("the image could not be decoded (the file is truncated or corrupt)");
        }
        if (rgb.width != f.width || rgb.height != f.height) {
            image_reader_.recycle(rgb);
            throw flm::d3::ContractError("the image could not be decoded (its pixels do not match its header)");
        }
        decoded.emplace(index, std::move(rgb));
    };
#ifdef FLM_D3_VIDEO
    if (video_) {
        hooks.load_video = video_loader_;
        hooks.load_windows = [this](const flm::d3::PyValue& value, int rate) {
            return flm::d3::video::load_video_windows(value, rate, video_options_, composed_config_.video_config);
        };
    }
#endif
    flm::d3::ComposedRequest request;
    try {
        request = flm::d3::prepare_composed(body, codes_, composed_config_, hooks);
    } catch (const flm::d3::RequestError& e) {
        release();
        reply.status = 422;
        reply.body = {{"detail", e.detail}};
        return reply;
    } catch (...) {
        release();
        throw;
    }
    // the processor's pixels of one table entry, once per request (every use shares them)
    std::map<size_t, flm::d3::EngineImage> prepared_images;
    std::map<int, flm::d3::video::VideoPayload> clip_pixels;   // the request's videos for plain questions, all groups
    const auto image = [&](size_t index) -> flm::d3::EngineImage {
        const auto& im = request.images.at(index);
        flm::d3::EngineImage out;
        out.grid_h = im.grid_h;
        out.grid_w = im.grid_w;
        if (im.clip >= 0) {
            auto have = clip_pixels.find(im.clip);
            if (have == clip_pixels.end()) {
                flm::d3::video::VideoPayload payload;
                flm::d3::video::append_video(payload, *request.clips.at(static_cast<size_t>(im.clip)),
                                             request.clip_grids.at(static_cast<size_t>(im.clip)), composed_config_.video_config);
                have = clip_pixels.emplace(im.clip, std::move(payload)).first;
            }
            const size_t n = static_cast<size_t>(im.grid_h) * im.grid_w * flm::d3::video::PATCH_VALUES;
            const auto& px = have->second.pixels;
            if ((static_cast<size_t>(im.group) + 1) * n > px.size()) throw std::runtime_error("video payload is shorter than its groups");
            out.pixels.assign(px.begin() + static_cast<long>(n * im.group), px.begin() + static_cast<long>(n * (im.group + 1)));
            return out;
        }
        auto hit = prepared_images.find(index);
        if (hit != prepared_images.end()) return hit->second;
        std::vector<bf16> pixels;
        if (!preprocess_image(decoded.at(index), im.grid_h, im.grid_w, pixels)) {
            throw std::runtime_error("could not prepare image " + im.source);
        }
        out.pixels.resize(pixels.size());
        static_assert(sizeof(bf16) == sizeof(uint16_t), "bf16 is 16 bits");
        std::memcpy(out.pixels.data(), pixels.data(), pixels.size() * sizeof(uint16_t));
        return prepared_images.emplace(index, std::move(out)).first->second;
    };
    const auto progress = [](size_t done, size_t calls, const std::string& state, double seconds) {
        if (calls > 1 || state != "request") {
            header_print("FLM", "d3 state " << state << " (" << done << "/" << calls << ") " << std::fixed << std::setprecision(1) << seconds << " s");
        }
    };
    try {
        std::lock_guard<std::mutex> lock(turn_);
        reply.body = flm::d3::run_composed(request, *readout_engine_, readout_.data(), composed_config_, image, progress);
    } catch (...) {
        release();
        throw;
    }
    release();
    return reply;
}

void AutoDecisionModel::warmup() {
    // One pass per GEMM row bucket typical requests land in: 2,048 rows (several questions packed),
    // 1,024 and 512 (one question). The engine keeps each bucket's instruction sequences for its
    // lifetime and the scratch of buckets that fit one 4,096-row pass together (these three do), so
    // the first requests in each bucket run at steady-state speed.
    bool enabled = !decisionNoWarmupTags.count(tag_);
    if (const char* env = std::getenv("D3_WARMUP")) enabled = std::string(env) != "0";
    if (!enabled) {
        header_print("FLM", "Decision model warmup skipped for " << tag_);
        return;
    }
    std::lock_guard<std::mutex> lock(turn_);
    const std::vector<std::vector<std::vector<int>>> passes = {
        std::vector<std::vector<int>>(4, std::vector<int>(500, 32)),   // 2,000 rows -> 2,048
        {std::vector<int>(700, 32)},                                   // 1,024
        {std::vector<int>(330, 32)},                                   // 512
    };
    for (const auto& ids : passes) {
        auto warm = engine_->encode_batch(ids);
        (void)warm;
    }
}
