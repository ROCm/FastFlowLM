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

namespace {
constexpr int N_FFT = 400;
constexpr int STFT_PAD = N_FFT / 2;               // center=True pads half a window on each side
constexpr int FREQ_BINS = N_FFT / 2 + 1;          // 201
}  // namespace

void Qwen3_ASR::_finish_log_mel(std::vector<float>& mel) {
    const int total = (int)mel.size();
    if (total == 0) return;
    const float max_val = audioproc::reduce_max(mel.data(), total);
    audioproc::clamp_below_max(mel.data(), total, max_val, 8.0f);
    audioproc::affine_scale(mel.data(), total, 4.0f, 4.0f);
}

void Qwen3_ASR::_log10_mel_frames(const float* x, int n, int first, std::vector<float>& out, int& end) {
    // torch.hann_window(400) is periodic; mel filters: Slaney scale and norm
    static const std::vector<float> window = audioproc::window_function_optimized(N_FFT, "hann", /*periodic=*/true);
    static const std::vector<float> mel_filters = audioproc::mel_filter_bank_optimized(FREQ_BINS, NUM_MEL_BINS, 0.0f, 8000.0f, SAMPLE_RATE,
                                                                                       /*apply_slaney_norm=*/true, /*slaney_mel_scale=*/true);
    out.clear();
    end = n > 0 ? audioproc::stft_num_frames(n, N_FFT, HOP_LENGTH, /*center=*/true) - 1 : 0;   // the feature extractor drops the last frame
    if (end <= 0) { end = 0; return; }
    first = std::max(first, 0);
    if (first >= end) return;

    std::vector<float> power;
    int skip = 0;   // frames of `power` before `first`
    if (first * HOP_LENGTH < STFT_PAD || n < STFT_PAD + 2) {
        // frames near the start: the whole clip, reflect-padded at both ends
        power.resize((size_t)(end + 1) * FREQ_BINS);
        audioproc::stft_power_optimized(x, n, window.data(), N_FFT, HOP_LENGTH, /*center=*/true, audioproc::StftPadMode::reflect, power.data());
        skip = first;
    } else {
        // from frame `first`: its samples, then the clip's right reflect padding as stft(center=True) adds it (x[n-2], x[n-3], ...)
        const int s0 = first * HOP_LENGTH - STFT_PAD;
        std::vector<float> buf((size_t)(n - s0 + STFT_PAD));
        std::memcpy(buf.data(), x + s0, sizeof(float) * (size_t)(n - s0));
        for (int i = 0; i < STFT_PAD; i++) buf[(size_t)(n - s0 + i)] = x[n - 2 - i];
        const int frames = audioproc::stft_num_frames((int)buf.size(), N_FFT, HOP_LENGTH, /*center=*/false);
        power.resize((size_t)frames * FREQ_BINS);
        audioproc::stft_power_optimized(buf.data(), (int)buf.size(), window.data(), N_FFT, HOP_LENGTH, /*center=*/false,
                                        audioproc::StftPadMode::reflect, power.data());
    }
    const int count = end - first;
    out.resize((size_t)count * NUM_MEL_BINS);
    audioproc::mel_spectrogram_optimized(power.data() + (size_t)skip * FREQ_BINS, mel_filters.data(), out.data(), count, FREQ_BINS, NUM_MEL_BINS);
    audioproc::log_mel_floor_optimized</*UseClamp=*/true, /*Base=*/10>(out.data(), out.data(), count * NUM_MEL_BINS, 1e-10f);   // log10(max(x, 1e-10))
}

void Qwen3_ASR::compute_log_mel(const audio_data_t& audio, std::vector<float>& mel, int& frames) {
    this->_log10_mel_frames(audio.samples.data(), static_cast<int>(audio.num_frames), 0, mel, frames);
    _finish_log_mel(mel);
}

void Qwen3_ASR::_check_audio_fits(size_t samples) const {
    const size_t audio_tokens = (size_t)qwen3_asr_audio_tokens((int)(samples / 160));
    const size_t reserve = std::max<size_t>(256, audio_tokens / 2);
    const size_t cap = this->decode_cap();
    if (cap > 0 && audio_tokens + 64 + reserve > cap) {
        const double max_s = (double)(cap - 64) / 1.5 / 12.5;
        throw std::invalid_argument("the audio is too long for the context window (at most about " + std::to_string((int)max_s) + " seconds)");
    }
}

std::string Qwen3_ASR::transcribe(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language, const std::string& context,
                                  int length_limit, std::ostream& os, std::string& detected_language, double& duration_seconds,
                                  std::function<bool()> is_cancelled) {
    detected_language.clear();
    duration_seconds = 0.0;
    audio_data_t audio = this->load_audio_bytes(audio_bytes);
    if (audio.channels != 1 || audio.num_frames == 0) throw std::invalid_argument("the audio could not be decoded (wav, flac, mp3 or ogg expected)");
    duration_seconds = audio.duration_seconds;
    this->_check_audio_fits(audio.samples.size());

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
