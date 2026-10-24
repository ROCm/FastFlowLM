/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file chat_logprobs.hpp
 * \brief Matches the logprobs of the sampled tokens to the content the client sees
 * \author FastFlowLM Team
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <streambuf>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "AutoModel/automodel.hpp"

using json = nlohmann::ordered_json;

/// \brief Whether text is well formed UTF-8, as json::dump() requires
inline bool is_valid_utf8(const std::string& text) {
    static const uint32_t min_code_point[] = {0, 0x80, 0x800, 0x10000};
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = text[i];
        size_t more;
        uint32_t cp;
        if (lead < 0x80) { i++; continue; }
        else if ((lead & 0xE0) == 0xC0) { more = 1; cp = lead & 0x1F; }
        else if ((lead & 0xF0) == 0xE0) { more = 2; cp = lead & 0x0F; }
        else if ((lead & 0xF8) == 0xF0) { more = 3; cp = lead & 0x07; }
        else { return false; }
        if (i + more >= text.size()) {
            return false;
        }
        for (size_t k = 1; k <= more; k++) {
            const unsigned char next = text[i + k];
            if ((next & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (next & 0x3F);
        }
        if (cp < min_code_point[more] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += more + 1;
    }
    return true;
}

/// \brief A float as the double that prints as its shortest round-tripping decimal
/// \note json stores numbers as double, and the float's exact widening prints
///       as -0.3172530531883240 where -0.31725305 means the same float.
inline double shortest_double(float value) {
    char text[32];
    for (int digits = 6; digits < 9; digits++) {
        std::snprintf(text, sizeof(text), "%.*g", digits, value);
        if (std::strtof(text, nullptr) == value) {
            return std::strtod(text, nullptr);
        }
    }
    std::snprintf(text, sizeof(text), "%.9g", value);
    return std::strtod(text, nullptr);
}

/// \brief Places each sampled token in the generated text, then hands every
///        content delta the logprobs of the tokens it was made from
/// \note The engine records an entry for every token it samples, but the client
///       only sees what the output parser makes of the text: reasoning, markup
///       and tool calls are not content, so their tokens are left out of
///       logprobs.content. Each flush of the output stream is matched to the
///       entries recorded since the previous one, which gives every token its
///       byte range in the text; a content delta is then found in the text and
///       takes the tokens it overlaps.
class chat_logprobs_aligner {
public:
    ///@param engine decodes the tokens; must outlive the aligner
    ///@param entries the request's meta_info.logprobs, which the engine appends to
    chat_logprobs_aligner(AutoModel* engine, const std::vector<chat_token_logprob>& entries)
        : engine_(engine), entries_(entries) {}

    ///@brief Record what the model wrote since the previous flush
    ///@note Call before the parser sees the text, so a delta it yields can be placed.
    void on_flush(const std::string& written) {
        const size_t window_start = text_.size();
        text_ += written;
        // a sample taken back before anything was written leaves fewer entries
        seen_ = std::min(seen_, entries_.size());
        // Usually one token per flush. Match newest first, each one ending where
        // the next begins; a token whose text is not there was sampled but never
        // written (a discarded sample), and has no place in the output.
        size_t end = text_.size();
        std::vector<placed_token> found;
        for (size_t i = entries_.size(); i-- > seen_;) {
            std::string piece = engine_->token_text(entries_[i].token_id);
            if (piece.size() <= end - window_start &&
                text_.compare(end - piece.size(), piece.size(), piece) == 0) {
                found.push_back({i, end - piece.size(), end});
                end -= piece.size();
            }
        }
        placed_.insert(placed_.end(), found.rbegin(), found.rend());
        seen_ = entries_.size();
    }

    ///@brief The logprobs of the tokens a content delta was made from
    ///@param delta new content the parser produced, in order
    ///@return the logprobs.content entries for it, possibly none
    json take_content(const std::string& delta) {
        json out = json::array();
        if (delta.empty()) {
            return out;
        }
        // The first piece of content is the newest text when the parser releases
        // it, which skips any earlier reasoning that happens to read the same.
        // Later pieces continue from where the content got to.
        size_t pos = content_started_ ? text_.find(delta, content_end_) : text_.rfind(delta);
        if (pos == std::string::npos) {
            // The parser rewrote the text: credit the newest tokens with it
            size_t first = placed_.size();
            size_t covered = 0;
            while (first > next_ && covered < delta.size()) {
                first--;
                covered += placed_[first].end - placed_[first].start;
            }
            for (size_t i = first; i < placed_.size(); i++) {
                out.push_back(entry_json(entries_[placed_[i].entry]));
            }
            next_ = placed_.size();
            content_started_ = true;
            content_end_ = text_.size();
            return out;
        }
        const size_t end = pos + delta.size();
        for (; next_ < placed_.size(); next_++) {
            const placed_token& p = placed_[next_];
            if (p.start >= end) {
                break;  // held back, or not content at all; a later delta decides
            }
            if (p.end <= pos) {
                continue;  // before the content: reasoning or markup
            }
            out.push_back(entry_json(entries_[p.entry]));
        }
        content_started_ = true;
        content_end_ = std::max(content_end_, end);
        return out;
    }

private:
    struct placed_token {
        size_t entry;  // index into entries_
        size_t start;  // byte range in text_
        size_t end;
    };

    ///@brief OpenAI's {token, logprob, bytes} for one token
    json token_json(int token_id, float logprob) {
        const std::string text = engine_->token_text(token_id);
        json token = text;
        if (!is_valid_utf8(text)) {
            // part of a character: name it by its bytes, the way OpenAI does
            std::string name = "bytes:";
            char hex[5];
            for (unsigned char c : text) {
                std::snprintf(hex, sizeof(hex), "\\x%02x", c);
                name += hex;
            }
            token = name;
        }
        json bytes = nullptr;
        if (engine_->is_normal_token(token_id)) {
            bytes = json::array();
            for (unsigned char c : text) {
                bytes.push_back(static_cast<int>(c));
            }
        }
        // -inf (or a broken NaN) does not survive JSON; OpenAI's floor is -9999
        if (!std::isfinite(logprob) || logprob < -9999.0f) {
            logprob = -9999.0f;
        }
        return {{"token", token}, {"logprob", shortest_double(logprob)}, {"bytes", bytes}};
    }

    json entry_json(const chat_token_logprob& entry) {
        json out = token_json(entry.token_id, entry.logprob);
        json top = json::array();
        for (const token_logprob_t& alt : entry.top) {
            top.push_back(token_json(alt.token_id, alt.logprob));
        }
        out["top_logprobs"] = top;
        return out;
    }

    AutoModel* engine_;
    const std::vector<chat_token_logprob>& entries_;
    ///@brief Entries already matched to a flush
    size_t seen_ = 0;
    ///@brief Everything the model wrote, as the parser sees it
    std::string text_;
    ///@brief Entries found in text_, in order
    std::vector<placed_token> placed_;
    ///@brief First entry of placed_ no content delta has claimed or passed
    size_t next_ = 0;
    bool content_started_ = false;
    ///@brief Where in text_ the content so far ends
    size_t content_end_ = 0;
};

///@brief Output stream for a non-streamed response that wants logprobs
///@note Discards the text like nullstream, but places every token as it is
///      written; the response's content then takes its logprobs from aligner().
class chat_logprobs_ostream : public std::ostream {
public:
    chat_logprobs_ostream(AutoModel* engine, const std::vector<chat_token_logprob>& entries)
        : std::ostream(&buf_), buf_(engine, entries) {}

    chat_logprobs_aligner& aligner() {
        return buf_.aligner;
    }

private:
    struct recording_buf : public std::streambuf {
        recording_buf(AutoModel* engine, const std::vector<chat_token_logprob>& entries)
            : aligner(engine, entries) {}

        int_type overflow(int_type ch) override {
            if (ch != traits_type::eof()) {
                written += static_cast<char>(ch);
            }
            return ch;
        }

        int sync() override {
            aligner.on_flush(written);
            written.clear();
            return 0;
        }

        chat_logprobs_aligner aligner;
        std::string written;
    };

    recording_buf buf_;
};
