/// \file modeling_qwen3_asr.hpp
/// \brief Qwen3_ASR class: Confucius4-R2T2 (NetEase Youdao), Qwen3-ASR-1.7B with streaming training
/// \note Speech in, text out. The engine (qwen3_asr_npu) encodes the audio on the host, prefills the prompt on the
///       host and decodes on the NPU; this class owns audio loading, the log-mel front end, the prompt and the output
///       parsing. Prompt: <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>{pads}<|audio_end|>
///       <|im_end|>\n<|im_start|>assistant\n[language X<asr_text>]; the model answers "language X<asr_text>text".
#pragma once
#include "AutoModel/automodel.hpp"
#include "audio/audio_reader.hpp"
#include "base64.hpp"

#include <string>
#include <vector>

/************              Qwen3_ASR            **************/
class Qwen3_ASR : public AutoModel {
public:
    explicit Qwen3_ASR(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;

    chat_template_type_t get_chat_template_type() override { return chat_template_type_t::chat_ml; }
    bool thinking_enabled() const override { return false; }
    bool tools_enabled() const override { return false; }

    /// \brief "system_prompt": context / hot words for the model; "language": force a language ("Chinese", "English", ...; "" = auto)
    bool configure_parameter(std::string parameter_name, const std::any& value) override;

    /// \brief the model can transcribe an audio file given as raw bytes (wav / mp3 / flac ...): /v1/audio/transcriptions
    bool supports_transcription() const override { return true; }
    /// \brief Streaming transcription of a whole file, the way the model is meant to run live: the audio is fed in chunks, after
    ///        each one the text that can no longer change (all but the last `unfixed_token_num` tokens) is committed and reported
    ///        through `on_commit` (append-only). Same algorithm and defaults as the reference implementation
    ///        (netease-youdao/Confucius4-R2T2 example.py). Cost per step: encoder + prefill of ALL audio so far + a few decode steps.
    /// \param chunk_ms audio per step (80 ms ... 2 s allowed by the model); 160 ms is the reference default but needs a fast host
    /// \param on_commit called with each newly committed piece of text
    /// \return the final text (the last, complete hypothesis)
    std::string transcribe_streaming(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language,
                                     const std::string& context, int chunk_ms, int lookahead_ms, int unfixed_token_num,
                                     const std::function<void(const std::string&)>& on_commit, std::string& detected_language,
                                     double& duration_seconds, std::function<bool()> is_cancelled = [] { return false; });

    /// \brief Transcribe one audio file. The text is also streamed to `os`.
    /// \param meta_info the request's meta info (max_prefill_len set by the caller; filled with the usual statistics)
    /// \param language forced language, or "" for automatic detection
    /// \param context optional context / hot words
    /// \param detected_language set to the language the model reported ("" if it reported none)
    /// \param duration_seconds set to the audio length
    std::string transcribe(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language, const std::string& context,
                           int length_limit, std::ostream& os, std::string& detected_language, double& duration_seconds,
                           std::function<bool()> is_cancelled = [] { return false; });

private:
    void setup_tokenizer(std::string model_path);

    // ---- token ids (tokenizer.json of the package) ----
    static constexpr int AUDIO_START_ID = 151669;
    static constexpr int AUDIO_END_ID = 151670;
    static constexpr int AUDIO_PAD_ID = 151676;
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int HOP_LENGTH = 160;
    static constexpr int NUM_MEL_BINS = 128;

    std::string forced_language;                 ///< "" = let the model detect it

    // ---- audio (modeling_qwen3_asr_audio.cpp) ----
    AudioReader audio_reader_;
    audio_data_t load_audio(const std::string& filename);
    audio_data_t load_audio_base64(const std::string& base64_str);
    audio_data_t load_audio_bytes(const std::vector<uint8_t>& bytes);
    /// \brief log-mel [frames][128] fp32 (Whisper feature extractor: n_fft 400, hop 160, Slaney mels, log10 clamp (max-8), (x+4)/4)
    void compute_log_mel(const audio_data_t& audio, std::vector<float>& mel, int& frames);

    /// \brief Build the prompt tokens (audio pads expanded) and the engine payload from mono 16 kHz samples.
    /// \param text_after_assistant text that continues the assistant turn (forced "language X<asr_text>", streaming prefix)
    bool _build_prompt(const std::vector<float>& samples, const std::string& context, const std::string& language,
                       const std::string& assistant_text, std::vector<int>& tokens, void*& payload_out, int& first_len_run);
    /// \brief one prefill + a bounded decode: the text the model generates after `assistant_text`, at most max_new_tokens tokens
    std::string _asr_step(chat_meta_info_t& meta_info, const std::vector<float>& samples, const std::string& context,
                          const std::string& language, const std::string& assistant_text, int max_new_tokens,
                          const std::function<bool()>& is_cancelled);
    /// \brief the engine payload of the request being built (kept alive until the prefill)
    std::shared_ptr<void> payload_holder_;
};

/// \brief Split the model's answer "language X<asr_text>text" into the language and the text.
/// \return false if there was no "<asr_text>" marker (then `text` is the whole answer and `language` is empty)
bool qwen3_asr_split_answer(const std::string& answer, std::string& language, std::string& text);
