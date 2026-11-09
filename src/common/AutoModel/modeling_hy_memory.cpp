/// \file modeling_hy_memory.cpp
/// \brief HyMemory class
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note AutoModel wrapper for the Hy-Memory-30B-A3B engine (hy_mem_npu).

#include "AutoModel/modeling_hy_memory.hpp"
#include "models/hy_mem/flm/aie2p/hy_mem_npu.hpp"

#include <chrono>
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

#if defined(FLM_USE_HRX)
constexpr bool kHrxRuntime = true;
#else
constexpr bool kHrxRuntime = false;
#endif

/// Longest prefill chunk handed to the engine on the HRX runtime.
///
/// On HRX a chunk of about 2k rows or more makes an NPU dispatch of the prefill overlay time out
/// ("amdxdna dispatch timed out: ert state 8", DEADLINE_EXCEEDED), the engine goes on with a failed
/// result and the request hangs or returns garbage (measured: 1.4k-token prompt fine, 2.3k-token
/// prompt hangs; with the chunk capped at 1024 the same prompt runs clean). XRT is not affected and
/// is faster with big chunks (the MoE expert matmuls pad to 256 rows), so only HRX is capped.
constexpr int kHrxMaxPrefillChunk = 1024;

/// Hand the heap's free memory back to the OS once a request is done.
///
/// A prefill allocates and frees many MB-sized host buffers (expert gather/scatter blocks,
/// router and GLU scratch). glibc keeps each thread's freed chunks in that thread's arena
/// and only gives back the top of an arena, so a server that runs requests on several
/// threads ends up holding about 5 GB of free memory in ~10 arenas after a few long
/// prompts (measured: 25.1 GB RSS, of which only 0.75 GB in use; malloc_trim(0) -> 20.4 GB).
/// Trimming after the request keeps RSS at what is live. FLM_HEAP_TRIM_LOG=1 prints the time it takes.
void release_idle_heap() {
#if defined(__GLIBC__)
    const auto t0 = std::chrono::steady_clock::now();
    malloc_trim(0);
    if (std::getenv("FLM_HEAP_TRIM_LOG")) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        header_print("FLM", "heap trim took " << ms << " ms");
    }
#endif
}

}  // namespace

/************              Hy-Memory family            **************/
HyMemory::HyMemory(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "HyMemory") {}

void HyMemory::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // Greedy, as the hy_memory engine calls it (temperature 0): its replies are
    // JSON that a parser reads, not prose. A REST request can still override this.
    sampler_config config;
    config.top_k = 1;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    config.freq_penalty = 0.0f;
    config.pre_penalty = 0.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, hy_mem_npu_state_api(this->lm_engine));
}

void HyMemory::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

/// \note The template reads `reasoning_effort` ("high" | "low" | "no_think", default
///       "no_think") and, when there are no tools, appends
///       <｜reasoning_mode｜>reasoning_effort:<value> to the system prompt.
std::string HyMemory::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    if (tools.is_array() && !tools.empty()) {
        inputs.tools = tools;
    }
    inputs.extra_context = this->extra_context;
    inputs.extra_context["reasoning_effort"] = this->reasoning_effort;
    return this->chat_tmpl->apply(inputs);
}

bool HyMemory::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "reasoning_effort") {
        try {
            const std::string effort = std::any_cast<std::string>(value);
            if (effort != "no_think" && effort != "low" && effort != "high") {
                return false;
            }
            this->reasoning_effort = effort;
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    return this->AutoModel::configure_parameter(parameter_name, value);
}

bool HyMemory::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Language only: drop any multimodal payload rather than failing the request.
    if (!input.images.empty()) {
        header_print("WARNING", "Hy-Memory-30B-A3B is a text-only model, ignoring " << input.images.size() << " image(s)");
        input.images.clear();
        input.image_payload_types.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Hy-Memory-30B-A3B is a text-only model, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    if (!input.messages.empty()) {   // already formatted messages, usually from the REST API
        for (auto& message : input.messages) {
            message.erase("images");
            message.erase("audios");
        }
        messages = input.messages;
    }
    else {                           // plain text, usually from the cli
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
    }

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text = this->apply_chat_template(messages, input.tools);
    // the template already starts with <bos>; encode() adds no second one
    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    if (kHrxRuntime && (meta_info.max_prefill_len <= 0 || meta_info.max_prefill_len > kHrxMaxPrefillChunk)) {
        meta_info.max_prefill_len = kHrxMaxPrefillChunk;
    }

    // Prefix slots: resume from the longest saved prefix of this prompt. Trim is 0 --
    // the whole rendered prompt is prefilled here, nothing is left for generate() to
    // feed. The marker opens the first user turn, so everything before it (system +
    // tools) is the shared head that gets its own snapshot.
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        return this->_prefix_text_insert(meta_info, tokens, 0, "<｜hy_User｜>", is_cancelled);
    }
    this->_prefix_legacy_path();

    return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
}

std::string HyMemory::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string reply = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    release_idle_heap();
    return reply;
}

std::string HyMemory::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
