/// \file modeling_qwen3_asr.cpp
/// \brief Qwen3_ASR class: Confucius4-R2T2 speech recognition (see modeling_qwen3_asr.hpp)

#include "AutoModel/modeling_qwen3_asr.hpp"
#include "metrices.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <streambuf>

namespace {

/// Streams the model's answer to the caller without the "language X<asr_text>" header: everything up to and including
/// "<asr_text>" is held back; if the marker never comes (the language was forced in the prompt, or the model skipped
/// it) the held-back text is released after `max_header` characters.
class answer_filter_buf : public std::streambuf {
public:
    answer_filter_buf(std::ostream& sink, bool header_expected) : sink_(sink), passthrough_(!header_expected) {}
    ~answer_filter_buf() override { release(); }

protected:
    // Whole chunks, not single bytes: the CLI / SSE writers behind `sink_` treat every write as a token and hold back an
    // incomplete UTF-8 tail themselves, which they cannot do for one-byte writes of a multi-byte character.
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (passthrough_) { sink_.write(s, n); return n; }
        held_.append(s, (size_t)n);
        static const std::string marker = "<asr_text>";
        const size_t at = held_.find(marker);
        if (at != std::string::npos) {
            passthrough_ = true;
            const std::string rest = held_.substr(at + marker.size());
            held_.clear();
            if (!rest.empty()) sink_.write(rest.data(), (std::streamsize)rest.size());
        } else if (held_.size() > kMaxHeader) {   // no header after all: stop holding text back
            release();
        }
        return n;
    }
    int_type overflow(int_type c) override {
        if (c == traits_type::eof()) return c;
        const char ch = traits_type::to_char_type(c);
        xsputn(&ch, 1);
        return c;
    }
    int sync() override { sink_.flush(); return 0; }

private:
    static constexpr size_t kMaxHeader = 64;
    void release() {
        if (!held_.empty()) { sink_.write(held_.data(), (std::streamsize)held_.size()); held_.clear(); }
        passthrough_ = true;
    }
    std::ostream& sink_;
    std::string held_;
    bool passthrough_;
};

}  // namespace

bool qwen3_asr_split_answer(const std::string& answer, std::string& language, std::string& text) {
    static const std::string marker = "<asr_text>";
    const size_t at = answer.find(marker);
    if (at == std::string::npos) {
        language.clear();
        text = answer;
        const size_t bar0 = text.find('|');
        if (bar0 != std::string::npos) text.resize(bar0);
        return false;
    }
    const std::string header = answer.substr(0, at);        // "language Chinese" / "language None"
    text = answer.substr(at + marker.size());
    // a streaming-trained model may end its text with a "|" marker (the reference keeps the part before it)
    const size_t bar = text.find('|');
    if (bar != std::string::npos) text.resize(bar);
    language.clear();
    const std::string key = "language ";
    const size_t k = header.find(key);
    if (k != std::string::npos) language = header.substr(k + key.size());
    if (language == "None") language.clear();               // the model's word for "no language": not a language
    return true;
}

/************              Qwen3_ASR family            **************/
Qwen3_ASR::Qwen3_ASR(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_ASR") {}

void Qwen3_ASR::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // greedy: the model is trained and benchmarked with temperature ~0
    sampler_config config;
    config.top_k = 1;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    config.freq_penalty = 0.0f;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3_ASR::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_ASR::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    if (!tools.empty()) inputs.tools = tools;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

bool Qwen3_ASR::configure_parameter(std::string parameter_name, const std::any& value) {
    try {
        if (parameter_name == "system_prompt") {
            this->user_system_prompt = std::any_cast<std::string>(value);
            this->extra_context["user_system_prompt"] = this->user_system_prompt;
            return true;
        }
        if (parameter_name == "language") {
            this->forced_language = std::any_cast<std::string>(value);
            return true;
        }
    } catch (const std::bad_any_cast&) {
        return false;
    }
    return false;
}

bool Qwen3_ASR::_build_prompt(const std::vector<float>& samples, const std::string& context, const std::string& language,
                              const std::string& assistant_text, std::vector<int>& tokens, void*& payload_out, int& first_len_run) {
    // log-mel and the number of audio tokens it makes
    auto payload = std::make_shared<qwen3_asr_payload_t>();
    audio_data_t audio;
    audio.samples = samples;
    audio.num_samples = audio.num_frames = samples.size();
    audio.channels = 1;
    audio.sample_rate = SAMPLE_RATE;
    std::vector<float> mel;
    int frames = 0;
    this->compute_log_mel(audio, mel, frames);
    if (frames <= 0) {
        header_print("ERROR", "The audio is too short to transcribe");
        return false;
    }
    payload->mel_frames = frames;
    payload->mel = std::move(mel);
    payload->n_audio = qwen3_asr_audio_tokens(frames);
    payload->audio_pad_id = AUDIO_PAD_ID;

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string prompt = "<|im_start|>system\n" + context + "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                         "<|im_end|>\n<|im_start|>assistant\n";
    if (!language.empty()) prompt += "language " + language + "<asr_text>";
    prompt += assistant_text;
    std::vector<int> init = this->tokenizer->encode(prompt);
    tokens.clear();
    tokens.reserve(init.size() + payload->n_audio);
    first_len_run = 0;
    for (int id : init) {
        if (id == AUDIO_PAD_ID) {
            for (int j = 0; j < payload->n_audio; j++) tokens.push_back(AUDIO_PAD_ID);
            first_len_run = (int)tokens.size() + 1;   // the audio tokens and <|audio_end|> must be in the first prefill chunk
        } else {
            tokens.push_back(id);
        }
    }
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    this->payload_holder_ = payload;
    payload_out = payload.get();
    return true;
}

bool Qwen3_ASR::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // 1. the audio: CLI paths in input.audios, REST base64 strings in messages[*]["audios"]; several are joined in time
    std::vector<float> samples;
    auto append = [&](audio_data_t&& a) {
        if (a.channels > 1 || a.num_frames == 0) {
            header_print("ERROR", "Skipping audio that is not mono or is empty");
            return;
        }
        samples.insert(samples.end(), a.samples.begin(), a.samples.end());
    };
    for (const std::string& path : input.audios) append(this->load_audio(path));
    std::string text_context;
    if (!input.messages.empty()) {
        for (const auto& message : input.messages) {
            if (message.contains("audios")) {
                for (const auto& aud : message["audios"]) append(this->load_audio_base64(aud.get<std::string>()));
            }
            if (message.value("role", "") == "user" && message.contains("content") && message["content"].is_string())
                text_context = message["content"].get<std::string>();
        }
    } else {
        text_context = input.prompt;
    }
    if (samples.empty()) {
        header_print("WARNING", "No audio to transcribe: give an audio file (/input \"file.wav\") or an audio message");
        return false;
    }
    // text next to the audio is context / hot words for the model, unless a system prompt was configured
    const std::string context = this->user_system_prompt.empty() ? text_context : this->user_system_prompt;

    // 2. prompt + payload
    std::vector<int> tokens;
    void* payload = nullptr;
    int first_len_run = 0;
    if (!this->_build_prompt(samples, context, this->forced_language, "", tokens, payload, first_len_run)) return false;

    // 3. every request is its own transcription: nothing of the previous one is reused
    this->clear_context();
    return this->_shared_insert(meta_info, tokens, is_cancelled, payload, first_len_run);
}

std::string Qwen3_ASR::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    answer_filter_buf filter(os, /*header_expected=*/this->forced_language.empty());
    std::ostream filtered(&filter);
    std::string answer = this->_shared_generate(meta_info, length_limit, filtered, is_cancelled);
    filtered.flush();
    std::string language, text;
    qwen3_asr_split_answer(answer, language, text);
    return text;
}

std::string Qwen3_ASR::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
