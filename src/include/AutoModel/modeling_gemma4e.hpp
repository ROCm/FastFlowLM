/// \file Gemma4e.hpp
/// \brief Gemma4e class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the Gemma4e class

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"


#include "typedef.hpp"
#include "image/image_reader.hpp"
#include "audio/audio_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

/************              Qwen3VL_4b            **************/
/// \brief Engine constants the Gemma4e wrapper needs from its backing engine.
/// \note  gemma4e_npu and gemma4e_flash are unrelated sibling types (both derive
///        straight from causal_lm), so the wrapper cannot downcast to one concrete
///        engine. Each subclass reads these through its own engine_config()
///        override instead. Field names mirror the engine members exactly.
struct gemma4e_engine_config_t {
    unsigned int GEMMA4E_VISION_PATCH_SIZE;
    unsigned int GEMMA4E_POOLING_KERNEL_SIZE;
    float        GEMMA4E_VISION_RESCALE_FACTOR;
    float        GEMMA4E_VISION_IMAGE_MEAN;
    float        GEMMA4E_VISION_IMAGE_STD;
    int          Gemma4E_Audio_resample_rate;
    unsigned int Gemma4E_Audio_conv2d_kernel_size;
    unsigned int Gemma4E_Audio_conv2d_Stride;
    unsigned int Gemma4e_Audio_conv2d_Padding;
};

class Gemma4e : public AutoModel {
private:
    // some model specific template variables
    static constexpr int boi_token_id = 255999; // begin of image token id
    static constexpr int image_token_id = 258880; // image token id
    static constexpr int eoi_token_id = 258882; // end of image token id

    static constexpr int boa_token_id = 256000; // begin of audio token id
    static constexpr int audio_token_id = 258881; // audio token id
    static constexpr int eoa_token_id = 258883; // end of audio token id

    static constexpr int think_start_id = 100;
    static constexpr int think_end_id = 101;

    bool enable_think = false;
    bool enable_tool = false;
    void setup_tokenizer(std::string model_path);
    
    // Image processing functionality
    ImageReader image_reader_;
    gemma4e_image_t load_image(const std::string& filename);
    gemma4e_image_t load_image_base64(const std::string& base64_string);

    // Audio processing functionality
    AudioReader audio_reader_;
    audio_data_t load_audio(const std::string &filename, int resample_rate, MonoDownmixMode downmix = MonoDownmixMode::NONE);
    audio_data_t load_audio_base64(const std::string &base64_str, int resample_rate, MonoDownmixMode downmix);
    std::vector<audio_data_t> clip_audio_length(audio_data_t& audio, double max_duration_second);
    void extract_spectrogram(std::vector<audio_data_t>& audio_inputs, gemma4e_audio_payload_t& audio_payload);

protected:
    /// Protected so Gemma4e_Flash can lower the default; see its constructor.
    int image_softtoken_budget = 280; // set a default value

    /// Max number of max_support_audio_length_seconds chunks kept per audio input.
    /// -1 keeps every chunk, i.e. the whole clip. Gemma4e_Flash sets it to 1 so a
    /// long clip is cut off after the first 30 s instead of overrunning its context.
    int max_audio_chunks = -1;
private:

    int debug_count= 0;

    
    void preprocess_image(
      gemma4e_image_t &image,
      std::pair<int, int> & patch_element_per_patch,
      uint32_t & valid_patch_size, // the unpadded size per image
      std::vector<bf16> &pixel_values,
      std::vector<int> &image_grid_pairs, // [num_of_position_id][x, y]      
      uint32_t &num_soft_tokens
    );


    std::vector<uint8_t>  aspect_ratio_preserving_resize( 
        gemma4e_image_t& image,
        int patch_size,
        int max_patches,
        int pooling_kernel_size
    );

    StreamResult parse_stream_content_impl(const std::string content, bool is_final);




protected:
    /// \brief Reads the engine constants the wrapper needs.
    /// \note  Overridden by Gemma4e_Flash to read them off the flash engine.
    virtual gemma4e_engine_config_t engine_config() const;

    /// \brief Builds the engine behind this wrapper; load_model() calls it.
    /// \note  Overridden by Gemma4e_Flash to swap in the flash engine. Everything
    ///        else -- weights, tokenizer, sampler, chat template -- is identical.
    virtual void create_engine();

public:
    Gemma4e(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;
    chat_template_type_t get_chat_template_type() {
        return chat_template_type_t::gemma4;
    }
    bool check_using_checkpint() {
		return this->enable_think;
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
        else if (parameter_name == "image_max_tokens") {
            try {
                this->image_softtoken_budget = std::any_cast<int>(value);
                
                if(image_softtoken_budget != 70 && image_softtoken_budget != 140 &&
                    image_softtoken_budget != 280 && image_softtoken_budget != 560 && image_softtoken_budget != 1120) {
                    header_print("WARNING", "Invalid image budget value: " << image_softtoken_budget << ". Supported values are 70, 140, 280, 560, 1120. Using 280...");
                    this->image_softtoken_budget = 280;
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }            
        }

		return false;
	}
};


/************              Gemma4e_Flash            **************/
/// Same checkpoint and same wrapper as Gemma4e, backed by the gemma4e_flash
/// engine. gemma4e_flash is currently a byte-for-byte copy of gemma4e_npu and
/// is the engine being tuned for short input prompts; Gemma4e stays the
/// general-purpose path.
class Gemma4e_Flash : public Gemma4e {
protected:
    void create_engine() override;
    gemma4e_engine_config_t engine_config() const override;

public:
    Gemma4e_Flash(flm_rt::device* npu_device_inst) : Gemma4e(npu_device_inst) {
        // Flash targets short prompts, so it defaults to the smallest supported
        // image budget instead of the 280 Gemma4e uses. Still overridable via
        // the "image_max_tokens" setting, same as the base wrapper.
        this->image_softtoken_budget = 70;
        // Flash's context is short, so keep only the first 30 s chunk of any audio.
        this->max_audio_chunks = 1;
    }
};
