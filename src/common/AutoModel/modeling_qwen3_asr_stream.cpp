/// \file modeling_qwen3_asr_stream.cpp
/// \brief Qwen3_ASR streaming transcription: a port of the reference loop of netease-youdao/Confucius4-R2T2
///        (example.py run_streaming + r2t2_asr.py streaming_transcribe / finish_streaming_transcribe) and of the
///        text post-processing it uses (qwen_asr parse_asr_output with detect_and_fix_repetitions, punctuation normalisation).
///
/// How it works: the audio comes in chunks. After every chunk the model is asked again, with ALL audio so far, to continue the
/// text already written minus its last `unfixed_token_num` tokens, for a few tokens only. The text without those last tokens is
/// "fixed": it never changes again, so it can be shown at once (append-only). The tail is rewritten by the next step.

#include "AutoModel/modeling_qwen3_asr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// UTF-8 helpers: the reference works on Python str (code points), so the text rules here work on code points too
// ---------------------------------------------------------------------------------------------------------------------
std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1f; len = 2; }
        else if ((c >> 4) == 0xe) { cp = c & 0x0f; len = 3; }
        else if ((c >> 3) == 0x1e) { cp = c & 0x07; len = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + len > s.size()) { out.push_back(0xFFFD); break; }
        for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3f);
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char32_t cp : s) {
        if (cp < 0x80) out.push_back((char)cp);
        else if (cp < 0x800) { out.push_back((char)(0xc0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3f))); }
        else if (cp < 0x10000) { out.push_back((char)(0xe0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3f))); out.push_back((char)(0x80 | (cp & 0x3f))); }
        else { out.push_back((char)(0xf0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3f))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3f))); out.push_back((char)(0x80 | (cp & 0x3f))); }
    }
    return out;
}

bool is_cjk(char32_t c) { return c >= 0x4e00 && c <= 0x9fff; }
bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || c == 0x3000 || c == 0xa0 || c == 0x2028 || c == 0x2029; }

std::string strip(const std::string& s) {
    const std::u32string u = utf8_to_u32(s);
    size_t b = 0, e = u.size();
    while (b < e && is_space(u[b])) ++b;
    while (e > b && is_space(u[e - 1])) --e;
    return u32_to_utf8(u.substr(b, e - b));
}

std::string before_bar(const std::string& s) {   // python: s.split("|")[0]
    const size_t at = s.find('|');
    return at == std::string::npos ? s : s.substr(0, at);
}

bool contains_replacement(const std::string& s) { return s.find("\xEF\xBF\xBD") != std::string::npos; }   // U+FFFD

std::string remove_replacement(std::string s) {
    size_t at;
    while ((at = s.find("\xEF\xBF\xBD")) != std::string::npos) s.erase(at, 3);
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// text rules of the reference
// ---------------------------------------------------------------------------------------------------------------------

/// r2t2_asr._normalize_punct_by_context: punctuation follows the script of the character before it
std::string normalize_punct_by_context(const std::string& text) {
    static const std::map<char32_t, char32_t> en2zh = {{U',', 0xff0c}, {U'.', 0x3002}, {U'!', 0xff01}, {U'?', 0xff1f},
                                                       {U';', 0xff1b}, {U':', 0xff1a}, {U'(', 0xff08}, {U')', 0xff09}};
    std::map<char32_t, char32_t> zh2en;
    for (auto& kv : en2zh) zh2en[kv.second] = kv.first;
    const std::u32string u = utf8_to_u32(text);
    std::u32string out;
    for (size_t i = 0; i < u.size(); ++i) {
        const char32_t c = u[i];
        if (!en2zh.count(c) && !zh2en.count(c)) { out.push_back(c); continue; }
        char32_t prev = 0;
        for (size_t j = i; j-- > 0;) { if (!is_space(u[j])) { prev = u[j]; break; } }    // the original text, not the output
        if (prev == 0) { out.push_back(c); continue; }
        if (is_cjk(prev)) {
            auto it = en2zh.find(c);
            out.push_back(it != en2zh.end() ? it->second : c);
        } else if (prev < 0x80 && (std::isalnum((int)prev) || prev == '"' || prev == '\'')) {
            auto it = zh2en.find(c);
            out.push_back(it != zh2en.end() ? it->second : c);
        } else {
            out.push_back(c);
        }
    }
    return u32_to_utf8(out);
}

/// qwen_asr.inference.utils.detect_and_fix_repetitions (threshold 20): a character repeated more than 20 times, or a pattern of up to 20
/// characters repeated at least 20 times, is cut down to one copy
std::u32string fix_char_repeats(const std::u32string& s, size_t thresh) {
    std::u32string res;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i]) ++count;
        if (count > thresh) { res.push_back(s[i]); }
        else { res.append(s, i, count); }
        i += count;
    }
    return res;
}

std::u32string fix_pattern_repeats(const std::u32string& s, size_t thresh, size_t max_len = 20) {
    const size_t n = s.size();
    const size_t min_repeat_chars = thresh * 2;
    if (n < min_repeat_chars) return s;
    size_t i = 0;
    std::u32string result;
    bool found = false;
    while (i + min_repeat_chars <= n) {
        found = false;
        for (size_t k = 1; k <= max_len; ++k) {
            if (i + k * thresh > n) break;
            const std::u32string pattern = s.substr(i, k);
            bool valid = true;
            for (size_t rep = 1; rep < thresh; ++rep) {
                if (s.compare(i + rep * k, k, pattern) != 0) { valid = false; break; }
            }
            if (valid) {
                size_t end_index = i + thresh * k;
                while (end_index + k <= n && s.compare(end_index, k, pattern) == 0) end_index += k;
                result += pattern;
                result += fix_pattern_repeats(s.substr(end_index), thresh, max_len);
                i = n;
                found = true;
                break;
            }
        }
        if (found) break;
        result.push_back(s[i]);
        ++i;
    }
    if (!found) result += s.substr(std::min(i, n));
    return result;
}

std::string detect_and_fix_repetitions(const std::string& text) {
    return u32_to_utf8(fix_pattern_repeats(fix_char_repeats(utf8_to_u32(text), 20), 20));
}

std::string normalize_language_name(std::string s) {
    s = strip(s);
    if (s.empty()) return s;
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    s[0] = (char)std::toupper((unsigned char)s[0]);
    return s;
}

/// qwen_asr parse_asr_output: (language, text) of a raw answer; "" language = unknown
std::pair<std::string, std::string> parse_asr_output(const std::string& raw, const std::string& user_language) {
    std::string s = strip(raw);
    if (s.empty()) return {"", ""};
    s = detect_and_fix_repetitions(s);
    if (!user_language.empty()) return {user_language, s};
    static const std::string tag = "<asr_text>";
    const size_t at = s.find(tag);
    if (at == std::string::npos) return {"", strip(s)};
    const std::string meta = s.substr(0, at);
    const std::string text = s.substr(at + tag.size());
    std::string meta_lower = meta;
    for (auto& c : meta_lower) c = (char)std::tolower((unsigned char)c);
    if (meta_lower.find("language none") != std::string::npos) {
        const std::string t = strip(text);
        return {"", t};
    }
    std::string lang;
    std::istringstream lines(meta);
    std::string line;
    while (std::getline(lines, line)) {
        line = strip(line);
        if (line.empty()) continue;
        std::string low = line;
        for (auto& c : low) c = (char)std::tolower((unsigned char)c);
        if (low.rfind("language ", 0) == 0) {
            const std::string val = strip(line.substr(9));
            if (!val.empty()) lang = normalize_language_name(val);
        }
        break;
    }
    return {lang, strip(text)};
}

/// the reference's Chinese clean-up: no space between two Chinese characters
std::string drop_spaces_between_cjk(const std::string& s) {
    const std::u32string u = utf8_to_u32(s);
    std::u32string out;
    for (size_t i = 0; i < u.size(); ++i) {
        if (is_space(u[i]) && !out.empty() && is_cjk(out.back())) {
            size_t j = i;
            while (j < u.size() && is_space(u[j])) ++j;
            if (j < u.size() && is_cjk(u[j])) { i = j - 1; continue; }
        }
        out.push_back(u[i]);
    }
    return u32_to_utf8(out);
}

/// the text up to a rollback of `k` tokens of `ids`, without a broken last character: back off one more token while the decode ends in U+FFFD
std::string decode_with_rollback(Tokenizer& tok, const std::vector<int>& ids, int k, bool avoid_replacement_loop) {
    if (!avoid_replacement_loop) {
        const int end_idx = std::min((int)ids.size(), std::max(1, (int)ids.size() - k));   // python ids[:max(1, len - k)]: no text, no prefix
        return end_idx > 0 ? tok.decode(std::vector<int>(ids.begin(), ids.begin() + end_idx)) : std::string();
    }
    while (true) {
        const int end_idx = std::max(0, (int)ids.size() - k);
        const std::string s = end_idx > 0 ? tok.decode(std::vector<int>(ids.begin(), ids.begin() + end_idx)) : std::string();
        if (!contains_replacement(s)) return s;
        if (end_idx == 0) return "";
        ++k;
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// one prefill + bounded decode
// ---------------------------------------------------------------------------------------------------------------------
std::string Qwen3_ASR::_asr_step(chat_meta_info_t& meta_info, const std::vector<float>& samples, const std::string& context,
                                 const std::string& language, const std::string& assistant_text, int max_new_tokens,
                                 const std::function<bool()>& is_cancelled) {
    std::vector<int> tokens;
    void* payload = nullptr;
    int first_len_run = 0;
    if (!this->_build_prompt(samples, context, language, assistant_text, tokens, payload, first_len_run,
                             this->streaming_caches_ ? &this->stream_mel_ : nullptr)) return "";
    if (this->streaming_caches_) {   // the engine reuses what it kept of the previous step of this recording
        auto* p = static_cast<qwen3_asr_payload_t*>(payload);
        p->streaming = true;
        p->stream_reset = this->stream_reset_pending_;
        this->stream_reset_pending_ = false;
        // the next step's audio is this one plus a chunk (or the rest of the recording): its mel lets the engine encode it during the decode
        const size_t now = this->stream_seg_start_ + samples.size();
        if (this->stream_wav_ && now < this->stream_wav_->size()) {
            const size_t next = std::min(now + this->stream_step_, this->stream_wav_->size()) - this->stream_seg_start_;
            this->_stream_log_mel(this->stream_wav_->data() + this->stream_seg_start_, (int)next, this->stream_mel_, p->next_mel, p->next_mel_frames);
        }
    }
    chat_meta_info_t step_meta;
    step_meta.max_prefill_len = meta_info.max_prefill_len;
    step_meta.restore_allowed = false;
    this->clear_context();
    if (!this->_shared_insert(step_meta, tokens, is_cancelled, payload, first_len_run)) return "";
    std::vector<int> ids;
    int tok = this->last_token;   // sampled from the prefill
    while (!this->is_eos(tok)) {
        ids.push_back(tok);
        if ((int)ids.size() >= max_new_tokens || is_cancelled()) break;
        buffer<bf16> y = this->lm_engine->forward(tok);
        this->total_tokens++;
        tok = this->_sample(y, step_meta);
    }
    return this->tokenizer->decode(ids);
}

// ---------------------------------------------------------------------------------------------------------------------
// the streaming loop
// ---------------------------------------------------------------------------------------------------------------------
std::string Qwen3_ASR::transcribe_streaming(chat_meta_info_t& meta_info, const std::vector<uint8_t>& audio_bytes, const std::string& language,
                                            const std::string& context, int chunk_ms, int lookahead_ms, int unfixed_token_num,
                                            const std::function<void(const std::string&)>& on_commit, std::string& detected_language,
                                            double& duration_seconds, std::function<bool()> is_cancelled) {
    detected_language.clear();
    duration_seconds = 0.0;
    audio_data_t audio = this->load_audio_bytes(audio_bytes);
    if (audio.channels != 1 || audio.num_frames == 0) throw std::invalid_argument("the audio could not be decoded (wav, flac, mp3 or ogg expected)");
    duration_seconds = audio.duration_seconds;
    if (chunk_ms < 80 || chunk_ms > 2000) throw std::invalid_argument("chunk_size_ms must be between 80 and 2000");
    unfixed_token_num = std::max(0, unfixed_token_num);
    const std::vector<float>& wav = audio.samples;
    static const bool caches_off = std::getenv("QWEN3_ASR_STREAM_CACHE") && std::string(std::getenv("QWEN3_ASR_STREAM_CACHE")) == "0";
    this->streaming_caches_ = !caches_off;
    this->stream_reset_pending_ = true;
    this->stream_mel_ = stream_mel_cache();
    this->stream_wav_ = this->streaming_caches_ ? &wav : nullptr;
    this->stream_seg_start_ = 0;

    // ---- state (ASRStreamingState) ----
    const size_t step = (size_t)std::llround(chunk_ms / 1000.0 * SAMPLE_RATE);
    const size_t lookahead = (size_t)std::llround(lookahead_ms / 1000.0 * SAMPLE_RATE);
    this->stream_step_ = step;
    size_t chunk_samples = step;
    int chunk_id = 0;
    std::vector<float> buffer_pcm, audio_accum;
    std::string raw_decoded, state_text;
    std::string committed;   // what on_commit has been given so far
    std::string base;        // the final text of the finished contexts (see the 30-45 s cut below)
    std::string languages_done, last_language;   // qwen_asr merge_languages over the contexts
    auto languages = [&](const std::string& lang) {
        std::string out = languages_done;
        if (!lang.empty() && lang != last_language) out += (out.empty() ? "" : ",") + lang;
        return out;
    };
    auto full_text = [&](const std::string& seg) { return base + (_needs_space(base, seg) ? " " : "") + seg; };

    auto report = [&](const std::string& fixed) {   // append-only: only a growth of what was reported is passed on
        const std::string full = full_text(fixed);
        if (full.size() > committed.size() && full.compare(0, committed.size(), committed) == 0) {
            on_commit(full.substr(committed.size()));
            committed = full;
        }
    };

    // One context covers 30-45 s of audio: every step re-reads all of its audio and text, so a longer one gets slower and slower (and the
    // NPU decode can end a transcript early at long contexts). After 30 s, at the first chunk end whose last 100 ms is quiet (mean |x| at
    // most a quarter of the median 100 ms window of the context), or at 45 s, a final step finishes the text (as at the end of the
    // recording), and the next chunk starts a fresh context. Not in the reference streaming loop, which keeps one context throughout.
    constexpr double CONTEXT_MIN_SECONDS = 30.0, CONTEXT_MAX_SECONDS = 45.0;
    const int first_max_new_tokens_ = std::max(1, (int)((step + lookahead) / 1280));
    std::string last_text_tmp;
    auto maybe_cut = [&]() -> bool {
        const double secs = (double)audio_accum.size() / SAMPLE_RATE;
        if (secs < CONTEXT_MIN_SECONDS || is_cancelled()) return false;
        if (secs < CONTEXT_MAX_SECONDS) {
            const size_t win = SAMPLE_RATE / 10;
            std::vector<float> means;
            means.reserve(audio_accum.size() / win);
            for (size_t i = 0; i + win <= audio_accum.size(); i += win) {
                double s = 0.0;
                for (size_t j = i; j < i + win; j++) s += std::fabs(audio_accum[j]);
                means.push_back((float)(s / win));
            }
            const float last = means.back();
            std::nth_element(means.begin(), means.begin() + means.size() / 2, means.end());
            if (last > 0.25f * means[means.size() / 2]) return false;
        }
        // the final step of this context: the text so far minus its unfixed tokens, continued on all its audio
        const std::vector<int> cur_ids = this->tokenizer->encode(raw_decoded);
        const std::string prefix = before_bar(decode_with_rollback(*this->tokenizer, cur_ids, unfixed_token_num, /*avoid_replacement_loop=*/false));
        std::string gen_text = this->_asr_step(meta_info, audio_accum, context, language, prefix, first_max_new_tokens_, is_cancelled);
        gen_text = remove_replacement(normalize_punct_by_context(gen_text));
        auto [lang, txt] = parse_asr_output(before_bar(prefix + gen_text), language);
        report(before_bar(txt));
        detected_language = languages(lang);
        languages_done = detected_language;
        if (!lang.empty()) last_language = lang;
        base = committed;   // what the client has (a final text that does not extend it is not sent)
        // a fresh context for the next chunk
        this->stream_seg_start_ += audio_accum.size();
        audio_accum.clear();
        raw_decoded.clear();
        state_text.clear();
        last_text_tmp.clear();
        this->stream_reset_pending_ = true;
        this->stream_mel_ = stream_mel_cache();
        return true;
    };

    // streaming_transcribe: feed `seg`, run one decode step per full chunk, return the committed text of the last step
    auto streaming_transcribe = [&](const std::vector<float>& seg, int max_new_tokens) -> std::string {
        buffer_pcm.insert(buffer_pcm.end(), seg.begin(), seg.end());
        std::string fixed_text;
        while (buffer_pcm.size() >= chunk_samples) {
            audio_accum.insert(audio_accum.end(), buffer_pcm.begin(), buffer_pcm.begin() + chunk_samples);
            buffer_pcm.erase(buffer_pcm.begin(), buffer_pcm.begin() + chunk_samples);

            // prefix: the text so far minus its last `unfixed_token_num` tokens (unfixed_chunk_num is 0 in the reference example)
            std::string prefix;
            raw_decoded = before_bar(raw_decoded);
            {
                const std::vector<int> cur_ids = this->tokenizer->encode(raw_decoded);
                prefix = decode_with_rollback(*this->tokenizer, cur_ids, unfixed_token_num, /*avoid_replacement_loop=*/true);
            }
            prefix = before_bar(prefix);

            std::string gen_text = this->_asr_step(meta_info, audio_accum, context, language, prefix, max_new_tokens, is_cancelled);
            gen_text = remove_replacement(normalize_punct_by_context(gen_text));
            raw_decoded = prefix + gen_text;

            std::string lang;
            if (language.empty()) lang = parse_asr_output(raw_decoded, language).first;
            if (language == "Chinese" || lang == "Chinese") raw_decoded = drop_spaces_between_cjk(raw_decoded);
            auto [lang2, txt] = parse_asr_output(raw_decoded, language);

            static const std::string tag = "<asr_text>";
            bool has_tag = raw_decoded.find(tag) != std::string::npos;
            if (has_tag) raw_decoded = raw_decoded.substr(0, raw_decoded.find(tag)) + tag + txt;
            else raw_decoded = txt;
            raw_decoded = before_bar(raw_decoded);

            int k = unfixed_token_num;
            const std::vector<int> cur_ids = this->tokenizer->encode(raw_decoded);
            has_tag = raw_decoded.find(tag) != std::string::npos;
            if (has_tag && raw_decoded.substr(raw_decoded.find(tag) + tag.size()).empty()) k = 0;
            fixed_text = decode_with_rollback(*this->tokenizer, cur_ids, k, /*avoid_replacement_loop=*/true);
            if (fixed_text.find(tag) != std::string::npos) fixed_text = fixed_text.substr(fixed_text.find(tag) + tag.size());
            fixed_text = before_bar(fixed_text);

            has_tag = raw_decoded.find(tag) != std::string::npos;
            if (!has_tag && language.empty()) {      // the model has not started the answer yet
                state_text.clear();
                fixed_text.clear();
                maybe_cut();
                continue;
            }
            detected_language = languages(lang2);
            state_text = before_bar(txt);
            ++chunk_id;
            report(fixed_text);
            if (maybe_cut()) fixed_text.clear();
        }
        return fixed_text;
    };

    // finish_streaming_transcribe: the tail shorter than a chunk
    auto finish = [&](int max_new_tokens) {
        if (buffer_pcm.empty()) return;
        audio_accum.insert(audio_accum.end(), buffer_pcm.begin(), buffer_pcm.end());
        buffer_pcm.clear();
        const std::vector<int> cur_ids = this->tokenizer->encode(raw_decoded);
        std::string prefix = before_bar(decode_with_rollback(*this->tokenizer, cur_ids, unfixed_token_num, /*avoid_replacement_loop=*/false));
        std::string gen_text = this->_asr_step(meta_info, audio_accum, context, language, prefix, max_new_tokens, is_cancelled);
        gen_text = remove_replacement(normalize_punct_by_context(gen_text));
        raw_decoded = before_bar(prefix + gen_text);
        auto [lang, txt] = parse_asr_output(raw_decoded, language);
        detected_language = languages(lang);
        state_text = before_bar(txt);
        ++chunk_id;
    };

    // ---- run_streaming (example.py) ----
    size_t pos = 0;
    bool is_first = true;
    int max_new_tokens = std::max(1, (int)((step + lookahead) / 1280));      // 1280 samples = 80 ms = one audio token
    const int first_max_new_tokens = max_new_tokens;
    const int max_new_tokens_floor = std::min(32, std::max(4, 2 * (int)(step / 1280)));
    while (pos < wav.size()) {
        size_t n;
        if (is_first) {                                   // the first call carries the look-ahead
            n = std::min(step + lookahead, wav.size() - pos);
            chunk_samples = std::max<size_t>(1, step + lookahead);
            is_first = false;
        } else {
            n = std::min(step, wav.size() - pos);
            chunk_samples = std::max<size_t>(1, step);
        }
        const std::vector<float> seg(wav.begin() + pos, wav.begin() + pos + n);
        pos += n;

        std::string fixed = before_bar(streaming_transcribe(seg, max_new_tokens));
        if (fixed.size() > last_text_tmp.size()) {
            last_text_tmp = fixed;
            max_new_tokens = std::max(1, (int)(step / 1280));
        } else {
            max_new_tokens += 1;                          // nothing new was committed: allow one more token next time
        }
        max_new_tokens = std::min(max_new_tokens_floor, max_new_tokens);
        if (is_cancelled()) break;
        if ((double)n / SAMPLE_RATE < chunk_ms / 1000.0) break;   // the last, short segment
    }
    finish(first_max_new_tokens);
    this->stream_wav_ = nullptr;
    const std::string full = full_text(before_bar(state_text));
    if (full.size() > committed.size() && full.compare(0, committed.size(), committed) == 0)
        on_commit(full.substr(committed.size()));
    else if (committed.empty() && !full.empty())
        on_commit(full);
    return full;
}
