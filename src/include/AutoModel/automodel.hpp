/// \file automodel.hpp
/// \brief automodel class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a header file for the auto_model class
#pragma once

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <type_traits>
#include <any>
#include <optional>
#include <stdexcept>
#include "typedef.hpp"
#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "AutoModel/model_backend.hpp"
#include "models/llama/flm/aie2p/llama_npu.hpp"
#include "models/qwen2/flm/aie2p/qwen2_npu.hpp"
#include "models/qwen3/flm/aie2p/qwen3_npu.hpp"
#include "models/qwen2vl/flm/aie2p/qwen2vl_npu.hpp"
#include "models/qwen3vl/flm/aie2p/qwen3vl_npu.hpp"
#include "models/qwen3vl_flash/flm/aie2p/qwen3vl_flash.hpp"
#include "models/qwen3_5vl/flm/aie2p/qwen3_5vl_npu.hpp"
#include "models/qwen3_6_moe/flm/aie2p/qwen3_6_moe_npu.hpp"
#include "models/gemma/flm/aie2p/gemma_npu.hpp"
#include "models/gemma_text/flm/aie2p/gemma_text_npu.hpp"
#include "models/gemma4e/flm/aie2p/gemma4e_npu.hpp"
#include "models/gemma4e_flash/flm/aie2p/gemma4e_flash.hpp"
#include "models/gemma4_12b/flm/aie2p/gemma4_12b_npu.hpp"
#include "models/lfm2/flm/aie2p/lfm2_npu.hpp"
#include "models/phi4/flm/aie2p/phi4_npu.hpp"
#include "models/gpt_oss/flm/aie2p/gpt_oss_npu.hpp"
#include "models/nanbeige/flm/aie2p/nanbeige_npu.hpp"
#include "models/hunyuan/flm/aie2p/hunyuan_npu.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "utils/utils.hpp"
#include "utils/profiler.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "npu_utils/npu_utils.hpp"
#include "minja/chat-template.hpp"
#include <nlohmann/json.hpp>

// Forward declaration
struct CancellationToken;

enum class StreamEventType {
	WAITING,        
	CONTENT,        
	REASONING,      
	TOOL_DONE,
};

struct StreamResult {
	StreamEventType type;
	std::string content;       
	std::string tool_id;       
	std::string tool_name;     
	std::string tool_args_str; 
};

struct NonStreamResult {
	std::string content;
	std::string reasoning_content;
	std::string tool_name;
	std::string tool_args;
	std::vector<std::pair<std::string, std::string>> tool_calls_list; // (name, args) for multiple tool calls
};

typedef enum {
    EOT_DETECTED, 
    MAX_LENGTH_REACHED,
    ERROR_DETECTED,
	CANCEL_DETECTED,
	TOOL_DETECTED
} stop_reason_t;

inline std::string stop_reason_to_string(stop_reason_t reason){
    switch (reason){
        case EOT_DETECTED:
            return "stop";
        case MAX_LENGTH_REACHED:
            return "length";
		case CANCEL_DETECTED:
			return "cancel";
        case ERROR_DETECTED:
            return "error";
		case TOOL_DETECTED:
			return "tool_calls";
		default:
            return "UNKNOWN";
    }
}

/// \brief What the request allows the model to do about tool calls.
/// \note Only the two modes the server parses today; "required" and the
///       per-function object form are reported as unsupported and fall back
///       to TOOL_CHOICE_AUTO.
typedef enum {
	TOOL_CHOICE_AUTO,   // the model may emit tool calls (default)
	TOOL_CHOICE_NONE    // tool call tokens are masked out of the logits
} tool_choice_t;

struct chat_meta_info_t {
	int max_prefill_len;
    int prompt_tokens;        // whole prompt, cached prefix included
    int cached_prompt_tokens; // subset of prompt_tokens served from the KV cache
    int generated_tokens;
    uint64_t total_duration; // in nanoseconds
    uint64_t load_duration; // in nanoseconds
    uint64_t prefill_duration; // in nanoseconds
    uint64_t decoding_duration; // in nanoseconds
    stop_reason_t stop_reason;
	bool restore_allowed;
	tool_choice_t tool_choice;

	chat_meta_info_t() : max_prefill_len(0), prompt_tokens(0), cached_prompt_tokens(0), generated_tokens(0), total_duration(0), load_duration(0), prefill_duration(0), decoding_duration(0), stop_reason(EOT_DETECTED), restore_allowed(false), tool_choice(TOOL_CHOICE_AUTO) {}
};

typedef enum {
	FILE_NAME,
	BASE64_ENCODED,
	URL_LINK
} input_payload_type_t; // so far not used, leave for future use

typedef enum {
	chat_ml,
	harmony,
	gemma4,
	unknown_template
} chat_template_type_t;

struct lm_uniform_input_t {
	std::string prompt;
	nlohmann::ordered_json messages;
	std::vector<std::string> images;
	std::vector<input_payload_type_t> image_payload_types;
	std::vector<std::string> audios;
	std::vector<input_payload_type_t> audio_payload_types;
	nlohmann::ordered_json tools;
	std::optional<int> requested_max_new_tokens;
};

inline std::optional<int> normalize_requested_max_new_tokens(
	std::optional<int> requested) {
	return requested.has_value() && *requested > 0 ? requested : std::nullopt;
}

using json = nlohmann::ordered_json;

class ModelRequestError final : public std::runtime_error {
public:
	ModelRequestError(int http_code, bool session_cleared, std::string message);
	int http_code() const noexcept;
	bool session_cleared() const noexcept;
private:
	int http_code_;
	bool session_cleared_;
};

class AutoModel {
protected:
	std::string model_path = "";
	/// \brief the execution backend, which owns the engine
	std::unique_ptr<flm::backend::ModelBackend> backend_ = nullptr;
	/// \brief a non-owning view of backend_'s engine, or null when none is loaded
	causal_lm* lm_engine = nullptr;
	std::unique_ptr<Tokenizer> tokenizer = nullptr;
	std::unique_ptr<Sampler> sampler = nullptr;
	bool is_model_loaded = false;
	std::string current_model = "";
	std::vector<int> token_history;
	flm_rt::device* npu_device_inst = nullptr;
	std::unique_ptr<npu_xclbin_manager> npu = nullptr;
	bool enable_preemption = false;
    std::vector<int> checkpoint_his;
	/// \brief dump the undecorated model output to stdout once a turn ends
	/// \note on by default; models whose turns are short and driven in bulk (the
	///       hunyuan translator) turn it off so the log is not doubled.
	bool log_raw_output = true;
	/// \brief run one more forward on the eos token once a turn ends
	/// \note this keeps the kv cache aligned with token_history so a following
	///       turn can append to it. Models that rewind or clear between turns
	///       throw that state away anyway, so for them it is a wasted step.
	bool forward_on_eos = true;
	/// \brief whether this instance is running under `flm serve`
	/// \note off by default (`flm run`); the server sets it via set_server_mode()
	///       so that serve-only diagnostics don't show up in the CLI.
	bool is_server_mode = false;


	uint32_t MAX_L = 0;
	int last_token = -1;
	uint32_t total_tokens = 0;
	std::unique_ptr<LM_Config> lm_config = nullptr;
	std::unique_ptr<minja::chat_template> chat_tmpl = nullptr;

	std::string bos_token;
    std::string eos_token;
    std::string boi_token;
    std::string eoi_token;
    bool has_bos_token;
    int bos_token_id;
    std::vector<int> eos_token_ids;

	std::string user_system_prompt = "";

    nlohmann::json extra_context;

	typedef enum {
		PREFILL_TIME,
		DECODING_TIME,
		SAMPLING_TIME,
		TKOEN_ENCODE_TIME,
		TKOEN_DECODE_TIME,
		TTFT_TIME,
		TOTAL_TIME,
		PROFILER_TYPE_NUM
	} profiler_type;
	std::vector<profiler> profiler_list;
	time_utils::time_with_unit last_prefill_time;

	std::string tool_name_; 
	bool is_in_tool_block_ = false;
	std::string buffer_;
	StreamEventType current_mode_ = StreamEventType::CONTENT;
	bool waiting_for_header_ = true;



	void _shared_load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false);
	void _shared_initialize_model_state(std::string model_path, json model_info, int context_length);
	void _shared_initialize_legacy_npu(bool enable_preemption);
	nlohmann::json _shared_setup_tokenizer(std::string model_path);

	/// \brief Load a model onto its chosen backend
	/// \param model_path the model directory
	/// \param model_info the resolved model_list.json entry
	/// \param default_context_length the requested context length, or -1 for the catalog default
	/// \param enable_preemption whether preemption was asked for
	/// \param requested_backend the --backend value, empty when it was not given
	/// \note Resolves the backend, initializes the shared state, builds the
	///       backend through the registry and points lm_engine at its engine.
	///       Every frontend calls this instead of doing it by hand; what is left
	///       for the frontend is the tokenizer, the chat template and the sampler.
	/// \throws std::runtime_error if the backend is unknown, unavailable for the
	///         model, or cannot honour the requested preemption/context length
	void _shared_load_backend(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false, const std::string& requested_backend = "", const nlohmann::json* tokenizer_config = nullptr);

	/// \brief Drop the conversation after a failed inference
	/// \param poisoned whether the engine can no longer be driven at all
	/// \note A poisoned engine is not asked to clear its context; only a reload
	///       can recover it.
	void _shared_after_inference_failure(bool poisoned);

	/// \brief Refuse to use an engine that needs a reload
	/// \throws ModelRequestError 500 when the backend reports itself poisoned
	void _shared_guard_poisoned() const;

	/// \brief The most tokens the loaded model may hold
	/// \return MAX_L, lowered to the backend's own decode limit when it has one
	uint32_t decode_cap() const {
		const uint32_t backend_cap =
			backend_ ? backend_->max_decode_length() : 0;
		return backend_cap == 0 ? MAX_L : std::min(MAX_L, backend_cap);
	}

	/// \brief Insert tokens into the model
	/// \param meta_info the meta information of the chat
	/// \param tokens the tokens to insert
	/// \param payload the payload, it shall not be used as this function is only used for chunkwised insertion, no image allowed
	/// \return true if the tokens were inserted successfully, false otherwise
	bool _shared_insert(chat_meta_info_t& meta_info, std::vector<int>& tokens, std::function<bool()> is_cancelled = [] { return false; }, void* payload = nullptr, int first_len_run = 0, std::optional<int> requested_max_new_tokens = std::nullopt);

	/// \brief Reject a request that cannot fit in the backend's decode limit
	/// \param rendered_tokens the length of the rendered prompt
	/// \param requested the caller's max_new_tokens, if any
	/// \note Only backends that declare a hard decode limit of their own are
	///       checked; the FastFlowLM engines are bounded by MAX_L alone and keep
	///       their existing behaviour of truncating rather than refusing.
	/// \throws ModelRequestError 400 when prompt plus output cannot fit
	void _shared_validate_capacity(std::size_t rendered_tokens, std::optional<int> requested) const;
	buffer<bf16> _chunked_insert(chat_meta_info_t& meta_info, std::vector<int>& tokens, std::function<bool()> is_cancelled = [] { return false; }, void* payload = nullptr, int first_len_run = 0);
	std::string _shared_generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; });

	/// \brief Push the tool start token out of contention before sampling.
	/// \param y the logits of the token about to be sampled
	/// \param meta_info the meta information of the chat, carrying the tool choice
	/// \note No-op unless the request asked for tool_choice=none and this model
	///       reports a tool start token, so the default path is untouched.
	void _apply_tool_choice_mask(buffer<bf16>& y, const chat_meta_info_t& meta_info);

	StreamResult _shared_think_tool_calling_pasrsed(const std::string content);

public:
	//************ Shared by all models *************/

	/// \brief true if this model clears the kv cache on every insert() and
	///        does not preserve context across turns (e.g. qwen3vl_flash).
	bool single_turn = false;

	virtual ~AutoModel() = default;

	AutoModel(flm_rt::device* npu_device_inst, std::string current_model = "");

	void reset_parser() {
		buffer_.clear();
		is_in_tool_block_ = false;
		current_mode_ = StreamEventType::CONTENT;
	}

	/// \brief Clear the context
	virtual void clear_context();

	/// \brief Mark this instance as running under `flm serve` (vs. `flm run`)
	/// \param value true if running under `flm serve`
	void set_server_mode(bool value) { is_server_mode = value; }

	/// \brief Get the current model
	/// \return the current model
	std::string get_current_model();

	/// \brief Get the id of the backend the model is loaded on
	/// \return the backend id, or empty when no model is loaded
	virtual std::string backend_id() const noexcept {
		return backend_ ? backend_->id() : std::string();
	}

	/// \brief Get the current context length
	/// \return the current context length
	virtual int get_current_context_length();

	/// \brief Set the sampler
	/// \param sampler_config the sampler config
	virtual void set_sampler(sampler_config& sampler_config);

	/// \brief Set the max length
	/// \param MAX_L the max length
	virtual void set_max_length(unsigned int MAX_L);

	/// \brief Get the max length
	/// \return the max length
	unsigned int get_max_length() const { return MAX_L; }

	/// \brief Show the model info
	/// \return the model info
	virtual std::string show_model_info();

	/// \brief Show the profile
	/// \return the profile
	virtual std::string show_profile();

	/// \brief Get the history
	/// \return the history
	virtual std::pair<std::string, std::vector<int>> get_history();

	/// \brief Verbose
	void verbose();

	/// \brief total time held by one profiler slot, in ms
	/// \param slot index into profiler_list, see profiler_type
	float get_profiler_ms(int slot){
		time_utils::time_with_unit t = this->profiler_list[slot].get_total_time();
		return time_utils::cast_to_s(t).first * 1000.0f;
	}

	/// \brief profiler slot ids, so a harness can break a turn down by phase
	enum profiler_slot_t {
		SLOT_PREFILL = PREFILL_TIME,
		SLOT_DECODING = DECODING_TIME,
		SLOT_SAMPLING = SAMPLING_TIME,
		SLOT_ENCODE = TKOEN_ENCODE_TIME,
		SLOT_DECODE = TKOEN_DECODE_TIME,
	};

	float get_ttft(){
    	time_utils::time_with_unit ttft_time = this->profiler_list[TTFT_TIME].get_total_time();
		time_utils::time_with_unit ttft_in_second = time_utils::cast_to_s(ttft_time);
		return ttft_in_second.first;
	}

	/// \brief Set the topk
	/// \param topk the topk
	void set_topk(int topk);

	/// \brief Set the topp
	/// \param topp the topp
	void set_topp(float topp);

	/// \brief Set the minp
	/// \param topp the minp
	void set_minp(float minp);

	/// \brief Set the temperature
	/// \param temperature the temperature
	void set_temperature(float temperature);

	/// \brief Set the presencepenalty
	/// \param presence_penalty the presence penalty
	void set_presence_penalty(float presence_penalty);

	/// \brief Set the repetition penalty
	/// \param repetition_penalty the repetition penalty
	void set_repetition_penalty(float repetition_penalty);

	/// \brief Set the frequency penalty
	/// \param frequency_penalty the frequency penalty
	void set_frequency_penalty(float frequency_penalty);

	/// \brief Set the frequency penalty window
	/// \param frequency_penalty_window the frequency penalty window
	void set_frequency_penalty_window(int frequency_penalty_window);

	/// \brief Set the penalty window
	/// \param penalty_window the penalty window
	void set_penalty_window(int penalty_window);

	/// \brief Start the ttft timer
	/// \return the ttft timer
	void start_ttft_timer();

	/// \brief Stop the ttft timer
	/// \return the ttft timer
	void stop_ttft_timer();

	/// \brief Reset the total timer
	/// \return the total timer
	void reset_total_timer();

	/// \brief Start the total timer
	/// \return the total timer
	void start_total_timer();

	/// \brief Stop the total timer
	/// \return the total timer
	void stop_total_timer();


	inline bool is_normal_token(int token) {
		if (token == this->bos_token_id){
			return false;
		}
		else {
			for (auto& id : this->eos_token_ids){
				if (token == id){
					return false;
				}
			}
		}
		return true;
	}

	inline bool is_eos(int token) {
		return std::find(this->eos_token_ids.begin(), this->eos_token_ids.end(), token) != this->eos_token_ids.end();
	}

	/// \brief Prepare the benchmark
	/// \param text the text to benchmark
	/// \return a pair of the number of tokens and the text that produce 1000 tokens
	std::pair<int, std::string> prepare_benchmark(std::string& text){
		std::vector<int> tokens = this->tokenizer->encode(text);
		int num_tokens = tokens.size();
		std::string benchmark_text = text;
		if (num_tokens > 1000 - 32) {
			tokens.resize(1000 - 32); // considering that the chat template always add some tokens to it
			benchmark_text = this->tokenizer->decode(tokens);
		}
		return { num_tokens, benchmark_text };
	}

	//************ Unique for each model *************/
	
	virtual void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") {}
	virtual std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) = 0;
	virtual chat_template_type_t get_chat_template_type() {
		return chat_template_type_t::chat_ml;
	}
	virtual bool check_using_checkpint() {
		return true;
	}

	/// \brief The token that opens a tool call for this model
	/// \return the token id, or -1 if this model has no single such token
	/// \note Models that report an id can have tool calling suppressed through
	///       tool_choice=none; the rest keep emitting tool calls either way.
	virtual int get_tool_start_token_id() const {
		return -1;
	}
	/// \brief Insert the tokens
	/// \param tokens the tokens
	/// \param is_system_prompt the is system prompt
	/// \return true if the tokens are inserted successfully, false otherwise
	virtual bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) = 0;

	/// \brief Generate the tokens with prompt
	virtual std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) = 0;
	std::string generate_with_prompt(
		chat_meta_info_t& meta_info,
		lm_uniform_input_t& input,
		int length_limit,
		std::ostream& os,
		std::function<bool()> is_cancelled);

	/// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	virtual bool configure_parameter(std::string parameter_name, const std::any& value) {
		if (parameter_name == "system_prompt") {
			try {
				this->user_system_prompt = std::any_cast<std::string>(value);
				this->extra_context["user_system_prompt"] = this->user_system_prompt;
				return true;
			} catch (const std::bad_any_cast&) {
				return false;
			}
		}
		return false;
	}

	/// \brief Convenience template wrapper for type-safe parameter configuration
	template<typename T>
	bool configure_parameter(std::string parameter_name, const T& value) {
		return configure_parameter(parameter_name, std::any(value));
	}

	virtual std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) = 0;

	virtual NonStreamResult parse_nstream_content(const std::string response_text) {
		NonStreamResult result;
		result.content = response_text;
		return result;
	}

	virtual StreamResult parse_stream_content(const std::string content) {
		//header_print("AUTOMODEL PARSING", content);

		StreamResult result;
		result.type = StreamEventType::CONTENT; 
		result.content = content;

		return result;
	}

	virtual StreamResult parse_stream_content_final(const std::string content) {
		if (!content.empty()) {
			return parse_stream_content(content);
		}

		StreamResult result;
		result.type = StreamEventType::WAITING;
		return result;
	}
};


