/// \file modeling_qwen3_asr_audio.cpp
/// \brief Qwen3_ASR audio front end (loading, log-mel) and the transcribe() entry point
/// \note The log-mel is the one of Whisper's / Qwen3-ASR's WhisperFeatureExtractor, as ported for the omni and gemma4e
///       models (audioproc); its output is compared with the HF feature extractor in src/test/qwen3_asr.

#include "AutoModel/modeling_qwen3_asr.hpp"
#include "audio_process_utils/audioproc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

audio_data_t Qwen3_ASR::load_audio(const std::string& filename) {
    audio_data_t result;
    if (!audio_reader_.load_audio(filename, result, SAMPLE_RATE, MonoDownmixMode::MEAN)) {
        header_print("ERROR", "Qwen3_ASR failed to load audio: " << filename);
        return audio_data_t();
    }
    return result;
}

audio_data_t Qwen3_ASR::load_audio_base64(const std::string& base64_str) {
    audio_data_t result;
    const std::string bytes = base64::from_base64(base64_str);
    if (!audio_reader_.load_audio_from_memory(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), result, SAMPLE_RATE,
                                              MonoDownmixMode::MEAN)) {
        header_print("ERROR", "Qwen3_ASR failed to load audio from a base64 string");
        return audio_data_t();
    }
    return result;
}

audio_data_t Qwen3_ASR::load_audio_bytes(const std::vector<uint8_t>& bytes) {
    audio_data_t result;
    if (!audio_reader_.load_audio_from_memory(bytes.data(), bytes.size(), result, SAMPLE_RATE, MonoDownmixMode::MEAN)) {
        header_print("ERROR", "Qwen3_ASR failed to decode the audio");
        return audio_data_t();
    }
    return result;
}

void Qwen3_ASR::compute_log_mel(const audio_data_t& audio, std::vector<float>& mel, int& frames) {
    constexpr int n_fft = 400;
    constexpr float min_frequency = 0.0f;
    constexpr float max_frequency = 8000.0f;
    constexpr float clamp_min = 1e-10f;
    constexpr float dynamic_range = 8.0f;
    constexpr float log_offset = 4.0f;
    constexpr float log_scale = 4.0f;
    const int num_frequency_bins = n_fft / 2 + 1;   // 201

    frames = 0;
    mel.clear();
    const int n_samples = static_cast<int>(audio.num_frames);
    if (n_samples <= 0) return;

    // torch.hann_window(400) is periodic; mel filters: Slaney scale and norm
    std::vector<float> window = audioproc::window_function_optimized(n_fft, "hann", /*periodic=*/true);
    std::vector<float> mel_filters = audioproc::mel_filter_bank_optimized(num_frequency_bins, NUM_MEL_BINS, min_frequency, max_frequency,
                                                                          SAMPLE_RATE, /*apply_slaney_norm=*/true, /*slaney_mel_scale=*/true);

    const int num_frames_alloc = audioproc::stft_num_frames(n_samples, n_fft, HOP_LENGTH, /*center=*/true);
    if (num_frames_alloc <= 0) return;
    std::vector<float> power_spec(static_cast<size_t>(num_frames_alloc) * num_frequency_bins);
    const int num_frames = audioproc::stft_power_optimized(audio.samples.data(), n_samples, window.data(), n_fft, HOP_LENGTH,
                                                           /*center=*/true, audioproc::StftPadMode::reflect, power_spec.data());
    frames = num_frames - 1;   // the feature extractor drops the last STFT frame
    if (frames <= 0) { frames = 0; return; }

    mel.resize(static_cast<size_t>(frames) * NUM_MEL_BINS);
    audioproc::mel_spectrogram_optimized(power_spec.data(), mel_filters.data(), mel.data(), frames, num_frequency_bins, NUM_MEL_BINS);

    const int total = frames * NUM_MEL_BINS;
    audioproc::log_mel_floor_optimized</*UseClamp=*/true, /*Base=*/10>(mel.data(), mel.data(), total, clamp_min);   // log10(max(x, 1e-10))
    const float max_val = audioproc::reduce_max(mel.data(), total);
    audioproc::clamp_below_max(mel.data(), total, max_val, dynamic_range);                                      // max(x, max - 8)
    audioproc::affine_scale(mel.data(), total, log_offset, log_scale);                                          // (x + 4) / 4
}

std::string Qwen3_ASR::transcribe(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language, const std::string& context,
                                  int length_limit, std::ostream& os, std::string& detected_language, double& duration_seconds,
                                  std::function<bool()> is_cancelled) {
    detected_language.clear();
    duration_seconds = 0.0;
    audio_data_t audio = this->load_audio_bytes(audio_bytes);
    if (audio.channels != 1 || audio.num_frames == 0) throw std::invalid_argument("the audio could not be decoded (wav, flac, mp3 or ogg expected)");
    duration_seconds = audio.duration_seconds;

    std::vector<int> tokens;
    void* payload = nullptr;
    int first_len_run = 0;
    if (!this->_build_prompt(audio.samples, context, language, "", tokens, payload, first_len_run))
        throw std::invalid_argument("the audio is too short to transcribe");

    meta_info.restore_allowed = false;
    this->clear_context();
    if (!this->_shared_insert(meta_info, tokens, is_cancelled, payload, first_len_run)) throw std::invalid_argument("the audio is too long for the context window (about 10 minutes at the default 8192 tokens)");

    std::ostringstream answer;
    {
        // keep the model's own header for the language, stream only the text
        struct tee : std::streambuf {
            std::ostringstream& a;
            explicit tee(std::ostringstream& s) : a(s) {}
            int_type overflow(int_type c) override { if (c != traits_type::eof()) a.put(traits_type::to_char_type(c)); return c; }
        } raw(answer);
        std::ostream raw_os(&raw);
        this->_shared_generate(meta_info, length_limit, raw_os, is_cancelled);
    }
    std::string text;
    qwen3_asr_split_answer(answer.str(), detected_language, text);
    os << text << std::flush;
    return text;
}
