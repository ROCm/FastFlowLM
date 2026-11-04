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

/// whether the UTF-8 code point starting at s[i] is a CJK character / CJK punctuation (no space between such texts)
bool cjk_at(const std::string& s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    return c >= 0xE3 && c <= 0xEF;   // U+3000..U+FFFF: CJK punctuation, kana, ideographs, full-width forms
}

/// the start of the last UTF-8 code point of s
size_t last_cp(const std::string& s) {
    size_t i = s.size() - 1;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    return i;
}
}  // namespace

bool Qwen3_ASR::_needs_space(const std::string& before, const std::string& after) {
    return !before.empty() && !after.empty() && !cjk_at(before, last_cp(before)) && !cjk_at(after, 0) && before.back() != ' ' && after.front() != ' ';
}

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

namespace {
/// Offline audio longer than this is transcribed in pieces of about this length (see the file comment of the split below)
constexpr double OFFLINE_PIECE_SECONDS = 60.0;

/// qwen_asr.inference.utils.split_audio_into_chunks: pieces of at most max_sec (+ the search window), each cut at the quietest 100 ms
/// window within +-5 s of the target; the pieces cover the audio exactly (no gap, no overlap). Returns the piece boundaries.
std::vector<std::pair<size_t, size_t>> split_audio_pieces(const std::vector<float>& wav, int sr, double max_sec) {
    const size_t total = wav.size();
    const size_t max_len = (size_t)(max_sec * sr), expand = (size_t)(5.0 * sr), win = std::max<size_t>(4, (size_t)(0.1 * sr));
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t start = 0;
    while (total - start > max_len) {
        const size_t cut = start + max_len;
        const size_t left = std::max(start, cut - std::min(cut, expand)), right = std::min(total, cut + expand);
        size_t boundary = cut;
        if (right - left > win) {
            // the window of `win` samples with the smallest sum of |x| (first one on ties), then its quietest sample
            double sum = 0.0;
            for (size_t i = left; i < left + win; i++) sum += std::fabs(wav[i]);
            double best = sum;
            size_t best_pos = left;
            for (size_t i = left + 1; i + win <= right; i++) {
                sum += std::fabs(wav[i + win - 1]) - std::fabs(wav[i - 1]);
                if (sum < best) { best = sum; best_pos = i; }
            }
            size_t inner = best_pos;
            for (size_t i = best_pos; i < best_pos + win; i++)
                if (std::fabs(wav[i]) < std::fabs(wav[inner])) inner = i;
            boundary = inner;
        }
        boundary = std::min(std::max(boundary, start + 1), total);
        pieces.emplace_back(start, boundary);
        start = boundary;
    }
    pieces.emplace_back(start, total);
    return pieces;
}

}  // namespace

std::string Qwen3_ASR::transcribe(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language, const std::string& context,
                                  int length_limit, std::ostream& os, std::string& detected_language, double& duration_seconds,
                                  std::function<bool()> is_cancelled) {
    detected_language.clear();
    duration_seconds = 0.0;
    audio_data_t audio = this->load_audio_bytes(audio_bytes);
    if (audio.channels != 1 || audio.num_frames == 0) throw std::invalid_argument("the audio could not be decoded (wav, flac, mp3 or ogg expected)");
    duration_seconds = audio.duration_seconds;

    // long audio: pieces of about a minute, transcribed one by one (as qwen_asr does above its own, longer, limit)
    const auto pieces = split_audio_pieces(audio.samples, SAMPLE_RATE, OFFLINE_PIECE_SECONDS);
    std::string text, last_language;
    // a piece's prompt is built one piece ahead and goes with the previous piece, whose decode on the NPU the engine uses to prefill
    // it on the CPU (the prompt of a piece does not depend on the text of the one before)
    struct piece_prompt {
        bool ok = false;
        std::vector<int> tokens;
        void* payload = nullptr;
        std::shared_ptr<void> holder;   // keeps the payload alive (payload_holder_ only holds the last one built)
        int first_len_run = 0;
    };
    auto prepare = [&](size_t p) {
        piece_prompt r;
        std::vector<float> piece(audio.samples.begin() + pieces[p].first, audio.samples.begin() + pieces[p].second);
        const size_t min_len = SAMPLE_RATE / 2;   // qwen_asr pads pieces shorter than 0.5 s with silence
        if (piece.size() < min_len) piece.resize(min_len, 0.0f);
        r.ok = this->_build_prompt(piece, context, language, "", r.tokens, r.payload, r.first_len_run);
        r.holder = this->payload_holder_;
        return r;
    };
    piece_prompt next = prepare(0);
    for (size_t p = 0; p < pieces.size(); p++) {
        if (p > 0 && is_cancelled()) break;
        piece_prompt cur = std::move(next);
        if (p + 1 < pieces.size()) {
            next = prepare(p + 1);
            if (cur.ok && next.ok) {
                auto* pl = static_cast<qwen3_asr_payload_t*>(cur.payload);
                const auto* np = static_cast<const qwen3_asr_payload_t*>(next.payload);
                pl->next_ids = next.tokens;
                pl->next_mel = np->mel;
                pl->next_mel_frames = np->mel_frames;
            }
        }
        if (!cur.ok) {
            if (pieces.size() == 1) throw std::invalid_argument("the audio is too short to transcribe");
            continue;
        }
        std::vector<int>& tokens = cur.tokens;
        void* payload = cur.payload;
        const int first_len_run = cur.first_len_run;
        meta_info.restore_allowed = false;
        this->clear_context();
        if (!this->_shared_insert(meta_info, tokens, is_cancelled, payload, first_len_run)) {
            if (is_cancelled()) break;
            throw std::invalid_argument("the audio could not be transcribed (prompt too long for the context window)");
        }

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
        std::string piece_language, piece_text;
        qwen3_asr_split_answer(answer.str(), piece_language, piece_text);
        // languages: qwen_asr merge_languages (in order, consecutive duplicates and empty ones dropped)
        if (!piece_language.empty() && piece_language != last_language) {
            detected_language += (detected_language.empty() ? "" : ",") + piece_language;
            last_language = piece_language;
        }
        if (piece_text.empty()) continue;
        // qwen_asr joins the pieces without a separator; a space keeps two Latin-script pieces from running together
        const std::string joined = (_needs_space(text, piece_text) ? " " : "") + piece_text;
        text += joined;
        os << joined << std::flush;
    }
    return text;
}
