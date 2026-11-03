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

/************              Qwen3VL_4b            **************/
class Qwen3_5VL : public AutoModel {
private:

    bool enable_think = false;
    bool enable_tool = false;
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
    
    void preprocess_image(qwen3_5vl_image_t& image,  std::vector<bf16> &pixel_values);

    /// \brief hand the loaded engine to the prefix cache (when it is the XRT qwen3_5vl one that has slots)
    void _attach_prefix_cache();

protected:
    /// \brief image settings the engine read from vision_config
    struct vision_params_t {
        unsigned patch_size;
        unsigned image_merge_size;
        unsigned merge_size;
        unsigned temporal_patch_size;
        unsigned shortest_edge;
        unsigned longest_edge;
        float rescale_factor;
        float image_mean;
        float image_std;
    };

    template <class Engine>
    static vision_params_t _read_vision_params(const Engine* eng) {
        return {eng->QWEN3_5_PATCH_SIZE, eng->QWEN3_5_IMAGE_MERGE_SIZE, eng->QWEN3_5_MERGE_SIZE,
                eng->QWEN3_5_TEMPORAL_PATCH_SIZE, eng->QWEN3_5_SHORTEST_EDGE, eng->QWEN3_5_LONGEST_EDGE,
                eng->QWEN3_5_VISION_RESCALE_FACTOR, eng->QWEN3_5_VISION_RESCALE_IMAGE_MEAN, eng->QWEN3_5_VISION_RESCALE_IMAGE_STD};
    }

    /// \brief the three places that know the engine type; Qwen3_5VL_Flash points them at qwen3_5vl_flash
    virtual vision_params_t _vision_params() const {
        return _read_vision_params(reinterpret_cast<const qwen3_5vl_npu*>(this->lm_engine));
    }
    virtual const flm_state_api* _state_api() {
        return qwen3_5vl_state_api(this->lm_engine);
    }
    /// \brief true when lm_engine is this wrapper's XRT engine (the one with saved-state slots)
    virtual bool _is_own_engine() const {
        return dynamic_cast<qwen3_5vl_npu*>(this->lm_engine) != nullptr;
    }

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

#if defined(FLM_HAS_QWEN3_5VL_FLASH)
/************              Qwen3_5VL_Flash            **************/
/// Same checkpoint and same wrapper as Qwen3_5VL (multi-turn, full context),
/// backed by the qwen3_5vl_flash engine: the Qwen3.5-2B kernels from
/// FLM_Xclbin/Qwen3_5_2B, loaded from the model's own xclbin folder.
class Qwen3_5VL_Flash : public Qwen3_5VL {
protected:
    vision_params_t _vision_params() const override {
        return _read_vision_params(static_cast<const qwen3_5vl_flash*>(this->lm_engine));
    }
    const flm_state_api* _state_api() override {
        return qwen3_5vl_flash_state_api(this->lm_engine);
    }
    bool _is_own_engine() const override {
        return dynamic_cast<qwen3_5vl_flash*>(this->lm_engine) != nullptr;
    }

public:
    Qwen3_5VL_Flash(flm_rt::device* npu_device_inst) : Qwen3_5VL(npu_device_inst) {}
};
#endif
