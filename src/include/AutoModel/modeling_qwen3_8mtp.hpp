/// \file modeling_qwen3_8mtp.hpp
/// \brief Qwen3_8MTP class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note AutoModel wrapper for Qwen3.8-27B (model_type qwen3_5): 64 layers,
///       48 GatedDeltaNet + 16 full attention, plus a 1-layer MTP draft head.
/// \note Multimodal when the checkpoint ships vision_weight.q4nx alongside
///       model.q4nx; text-only otherwise. Images are dropped loudly rather
///       than templated into placeholders nothing fills -- that failure mode
///       prefills garbage and still produces fluent text, so it never flags
///       itself. Ask the engine with has_vision_tower(), do not assume.
/// \note The think ids are 248068/248069, NOT the 151667/151668 that
///       modeling_qwen3.hpp hardcodes. Copying those gives a model whose
///       reasoning block never closes.

#pragma once
#include "AutoModel/automodel.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "base64.hpp"


/// \brief one decoded image, host side, on its way to the engine
/// \note Deliberately NOT qwen3_8mtp_image_t: that struct is the engine's ABI
///       and carries only a pointer and a grid. Everything here -- the uint8
///       CHW pixels, the pre- and post-resize extents -- is preprocessing
///       state the engine must never see, and keeping it out of the shared
///       header is what lets the engine struct stay a POD across the .so
///       boundary.
typedef struct {
    int width = 0;
    int height = 0;
    int width_resized = 0;   ///< assigned by preprocess_image
    int height_resized = 0;
    int grid_t = 1;          ///< a still image is one temporal frame
    int grid_h = 0;          ///< PATCHES, not pixels
    int grid_w = 0;

    bytes _data;             ///< uint8 (3, H, W); freed by preprocess_image
} qwen3_8mtp_host_image_t;


/************              Qwen3_8MTP            **************/
class Qwen3_8MTP : public AutoModel {
private:

    bool enable_think = false;
    bool enable_tool = true;

    /// \note Qwen3.8 vocabulary, verified against tokenizer.json:
    ///       <think> 248068, </think> 248069. The chat template emits
    ///       "<think>\n" when thinking is on and "<think>\n\n</think>\n\n"
    ///       when it is off, exactly as Qwen3.5 does.
    int think_start_id = 248068;
    int think_end_id = 248069;

    /// \brief the generation prompt's trailing ids that generate() must echo
    /// \note insert() prefills the whole templated prompt, think preamble
    ///       included, so by the time generate() runs those tokens are already
    ///       in the cache and in token_history. What is not done is putting
    ///       them on screen, and only thinking-on has anything to show: it
    ///       opens a reasoning block the user reads. Thinking-off leaves this
    ///       empty.
    /// \note Recorded at insert() time rather than rebuilt from
    ///       think_start_id/198 in generate(), so a caller that flips
    ///       enable_think between the two calls echoes what was actually
    ///       prefilled instead of what the flag says now.
    std::vector<int> preamble_to_stream;

    void setup_tokenizer(std::string model_path);

    // ---- image pipeline ---------------------------------------------------
    //
    // Modelled on Qwen3_5VL's, with one deliberate difference: this engine's
    // encode_image() takes `const float*`, not bf16. See preprocess_image().

    ImageReader image_reader_;

    /// \brief cap the decoded height before preprocessing; 0 = no cap
    /// \note Purely a cost knob. smart_resize below still runs and still has
    ///       the final say over the grid, so this only changes how many
    ///       patches the tower is asked to encode -- which at ~70 ms per
    ///       merged token on the CPU backend is the whole encode time.
    int image_pre_resize = 0;

    /// Image-processor geometry. Read from config.json's vision_config in
    /// load_model() where the checkpoint states it, and defaulted to
    /// Qwen3.8-27B's preprocessor_config.json where it does not -- the NPU2
    /// checkpoint ships no preprocessor_config.json, and the converter does
    /// not inject the QWEN3_5_* keys that Qwen3_5VL reads.
    ///
    /// image_mean and image_std are 0.5, NOT the OPENAI_CLIP constants. Using
    /// CLIP's costs no error anyone would notice at the tower's output and
    /// every bit of it at the model's, which is why it is written down.
    unsigned int vision_patch_size = 16;
    unsigned int vision_merge_size = 2;
    unsigned int vision_temporal_patch_size = 2;
    unsigned int vision_shortest_edge = 65536;
    unsigned int vision_longest_edge = 16777216;
    float vision_rescale_factor = 1.0f / 255.0f;
    float vision_image_mean = 0.5f;
    float vision_image_std = 0.5f;

    qwen3_8mtp_host_image_t load_image(const std::string& filename);
    qwen3_8mtp_host_image_t load_image_base64(const std::string& base64_string);

    /// Shared tails of the two loaders. Factored out because the Qwen3_5VL
    /// pair they are modelled on is a verbatim copy that has already drifted.
    void _apply_pre_resize(image_data_t& decoded);
    qwen3_8mtp_host_image_t _finish_load(image_data_t& decoded);

    void smart_resize(
        int height, int width,
        int& h_bar, int& w_bar,
        int factor,
        int min_pixels,
        int max_pixels);

    /// \brief resize, normalise and reorder one image; appends to `pixel_values`
    /// \note Appends in MERGE-BLOCK order, which is what the engine's
    ///       qwen3_8mtp_image_t contract requires. Raster order is not a
    ///       crash -- the tower runs and the image is silently scrambled.
    void preprocess_image(qwen3_8mtp_host_image_t& image,
                          std::vector<float>& pixel_values);

public:
    Qwen3_8MTP(flm_rt::device* npu_device_inst);

    /// \brief decode + preprocess one image file into the engine's pixel_values
    /// \return false if the image could not be decoded or preprocessed
    ///
    /// The whole image pipeline with no model behind it. It exists so the host
    /// preprocessing can be scored against a captured `pixel_values` without
    /// loading 17 GB of weights first -- and it needs scoring separately,
    /// because the tower's own verification feeds it a reference pixel_values
    /// and therefore says nothing about how this code produces one. A wrong
    /// patch order here passes every tower check and still scrambles the image.
    ///
    /// Safe to call on a default-constructed Qwen3_8MTP: it touches only
    /// image_reader_ and the vision_* geometry, which is defaulted to this
    /// checkpoint's values and only refined by load_model().
    bool preprocess_image_file(const std::string& path,
                               std::vector<float>& pixel_values,
                               int& grid_t, int& grid_h, int& grid_w);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;

    /// \brief the base profile plus the MTP draft/verify/replay breakdown
    /// \note Overridden because "Decoding time" is one number for a model with
    ///       three distinct decode phases. The runtime profiler cannot split
    ///       them -- from its side a cycle is one opaque speculate() call --
    ///       so the breakdown is measured in the engine and appended here.
    /// \note Calls the base implementation rather than reproducing its rows,
    ///       so the shared statistics cannot drift out of step with it.
    /// \note Appends nothing when speculation never ran, which keeps the
    ///       non-greedy and no-MTP-head paths byte-identical to before.
    std::string show_profile() override;

private:
    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

public:

    /// \brief Configure a parameter with type-erased value
    /// \param parameter_name the name of the parameter
    /// \param value the value to set (can be any type)
    /// \return true if the parameter was configured successfully, false otherwise
    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "reasoning_effort") {
            std::string reasoning_effort;
            try {
                reasoning_effort = std::any_cast<std::string>(value);
                if (reasoning_effort == "high" || reasoning_effort == "medium" || reasoning_effort == "low")
                    this->enable_think = true;
                else if (reasoning_effort == "none")
                    this->enable_think = false;
                else
                    header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
        else if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "img_pre_resize") {
            // Same ladder and the same key name as Qwen3_5VL, so a caller that
            // drives both models does not need to special-case this one.
            try {
                this->image_pre_resize = std::any_cast<int>(value);
                int target_size;
                if (this->image_pre_resize <= 0)      target_size = 0;
                else if (this->image_pre_resize == 1) target_size = 480;
                else if (this->image_pre_resize <= 2) target_size = 720;
                else if (this->image_pre_resize <= 3) target_size = 1080;
                else if (this->image_pre_resize <= 4) target_size = 1440;
                else if (this->image_pre_resize <= 5) target_size = 2160;
                else if (this->image_pre_resize <= 6) target_size = 2880;
                else if (this->image_pre_resize <= 7) target_size = 3240;
                else if (this->image_pre_resize <= 8) target_size = 4320;
                else { this->image_pre_resize = 0; target_size = 0; }
                if (this->image_pre_resize > 0)
                    header_print_r("FLM", "Qwen3.8 pre-resize image height to " +
                                          std::to_string(target_size) +
                                          " pixels if larger than that");
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        return AutoModel::configure_parameter(parameter_name, value);
    }
};
