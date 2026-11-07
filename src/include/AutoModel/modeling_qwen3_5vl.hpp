/// \file Qwen3_5VL.hpp
/// \brief Qwen3_5VL class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the Qwen3_5VL class

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"


#include "typedef.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

/// \brief how the vision model wants its images preprocessed
/// \note From the NPU or the Flash engine's config, or from the OGA package's
///       processor_config.json. All zero when the engine runs no images.
struct qwen3_5vl_vision_geometry_t {
    unsigned patch_size = 0;
    unsigned merge_size = 0;
    unsigned temporal_patch_size = 0;
    unsigned min_pixels = 0;
    unsigned max_pixels = 0;
    float rescale_factor = 0.0f;
    float image_mean = 0.0f;
    float image_std = 0.0f;

    bool valid() const { return patch_size && merge_size && temporal_patch_size && max_pixels; }
};

/************              Qwen3VL_4b            **************/
class Qwen3_5VL : public AutoModel {
protected:
    // the flash model (Qwen3_5VL_Flash) reads these
    bool enable_think = false;
    bool enable_tool = false;

    /// \brief render a request into the prompt tokens and the image payload (what insert() prefills)
    /// \param meta_info the request's meta info
    /// \param input the request (messages or prompt, images)
    /// \param image_payload receives the decoded, preprocessed images
    /// \param tokens receives the prompt tokens, image soft tokens expanded, the generation prompt included
    /// \return false if the request has nothing to render
    bool _build_tokens(chat_meta_info_t& meta_info, lm_uniform_input_t& input, qwen3_5vl_image_payload_t& image_payload, std::vector<int>& tokens);

private:
    int think_start_id = 248068;
    int think_end_id = 248069;
    
    void setup_tokenizer(std::string model_path);
    
    // Image processing functionality
    ImageReader image_reader_;
    qwen3_5vl_image_t load_image(const std::string& filename);
    qwen3_5vl_image_t load_image_base64(const std::string& base64_string);
    
    int image_pre_resize = 0;

    int debug_count= 0;
    void smart_resize(
    int height, int width,
    int& h_bar,int& w_bar,
    int factor,
    int min_pixels,
    int max_pixels);
    
    /// \param max_pixels the most pixels the resized image may have
    void preprocess_image(qwen3_5vl_image_t& image,  std::vector<bf16> &pixel_values, unsigned max_pixels);

    /// \brief hand the loaded engine to the prefix cache (when it is the XRT qwen3_5vl one that has slots)
    void _attach_prefix_cache();

    /// \brief read the loaded engine's image preprocessing into vision_geometry
    void _load_vision_geometry();

    /// \brief continue the history the engine ran instead of restoring the checkpoint
    /// \note The rendered prompt carries the last reply as the client sent it
    ///       back. When that is the reply the engine generated, the prompt is
    ///       rewritten to the engine's history (its reasoning included) plus what
    ///       follows the reply, so nothing before it runs again.
    /// \return whether `tokens` now continues token_history
    bool _continue_verbatim(std::vector<int>& tokens, std::optional<int> requested_max_new_tokens);

    /// \brief the content of each image in checkpoint_his, kept on OGA
    std::vector<uint64_t> checkpoint_image_hashes_;

protected:
    qwen3_5vl_vision_geometry_t vision_geometry;

public:
    Qwen3_5VL(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool supports_continue_final_message() const override { return true; }

private:

public:

    bool thinking_enabled() const override {
        return this->enable_think;
    }
    bool tools_enabled() const override {
        return this->enable_tool;
    }

    /// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	bool configure_parameter(std::string parameter_name, const std::any& value) override{
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
            try {
                this->image_pre_resize = std::any_cast<int>(value);
                int target_size;
                if (this->image_pre_resize <= 0) {
                    target_size = 0;
                } else if (this->image_pre_resize == 1) {
                    target_size = 480;
                } else if (this->image_pre_resize <= 2) {
                    target_size = 720;
                } else if (this->image_pre_resize <= 3) {
                    target_size = 1080;
                } else if (this->image_pre_resize <= 4) {
                    target_size = 1440;
                } else if (this->image_pre_resize <= 5) {
                    target_size = 2160;
                } else if (this->image_pre_resize <= 6) {
                    target_size = 2880;
                } else if (this->image_pre_resize <= 7) {
                    target_size = 3240;
                } else if (this->image_pre_resize <= 8) {
                    target_size = 4320;
                } else {
                    this->image_pre_resize = 0;
                    target_size = 0;
                }
                if (this->image_pre_resize > 0) {
                    header_print_r("FLM", "Qwen3.5 pre-resize image height to " + std::to_string(target_size) + " pixels if larger than that");
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
		return false;
	}
};


/************              Qwen3_5VL_Flash            **************/
/// Same checkpoint and the same wrapper as Qwen3_5VL, backed by the qwen3_5vl_flash engine: the 2B-only prefill
/// kernels (bfp16 matmul and attention, faster GateDeltaNet and conv), lm_head inside the decode layer, one vision
/// overlay. Tuned for short contexts; Qwen3_5VL stays the general-purpose path.
///
/// Logically single-turn, like every flash model: every insert() starts from a clean state. The one exception is the
/// system prompt: on the first request that carries one, insert() prefills its shared prefix and checkpoints it (the
/// engine's checkpoint holds the whole hybrid state, recurrent and conv state as well as the KV rows), then rewinds to
/// that checkpoint on later requests with the same system text.
class Qwen3_5VL_Flash : public Qwen3_5VL {
private:
    /// \brief system text the current checkpoint was built from, empty when none
    std::string pinned_system_text;

    /// \brief token history at the time of the checkpoint
    std::vector<int> system_his;

    /// \brief number of tokens covered by the checkpoint, 0 when nothing is pinned
    int system_tokens = 0;

    /// \brief prefill the system prefix of `system_text` and checkpoint it
    /// \return number of tokens pinned, 0 if the prefix could not be isolated
    int _pin_system_prefix(const std::string& system_text);

    /// \brief forget the pin, so the next request with a system prompt pins it again
    /// \note  Leaves the engine alone: callers clear it themselves.
    void _forget_pin();

    /// \brief Drop everything the previous turn left behind, back to the pin.
    /// \note  Deliberately not AutoModel::clear_context(): that also resets the TTFT and TOTAL profilers, which the
    ///        caller starts around the whole turn.
    void _reset_turn();

public:
    Qwen3_5VL_Flash(flm_rt::device* npu_device_inst) : Qwen3_5VL(npu_device_inst) { single_turn = true; }

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;

    /// \brief Clear the context, the pinned system prefix included
    /// \note  The engine zeroes its state but keeps the checkpoint, so a pin kept past this would restore zeroed rows
    ///        on the next request. This also covers _shared_insert clearing a prompt that diverges from the pinned
    ///        tokens: it prefills from row 0, over the pinned rows.
    void clear_context() override;

    /// \brief Apply the chat template without tools: the flash model does not support tool calling
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override {
        nlohmann::ordered_json no_tools = nlohmann::ordered_json::object();
        return Qwen3_5VL::apply_chat_template(messages, no_tools, continue_final_message);
    }

    /// \brief Single-turn: insert() rebuilds the prompt and never continues a message
    bool supports_continue_final_message() const override { return false; }
};
