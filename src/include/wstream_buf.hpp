/// \file wstream_buf.hpp
/// \brief wstream_buf class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to write UTF-8 characters to a stream
#pragma once

#include <iostream>
#include <string>
#include <vector>

/// The number of bytes at the end of text that begin a UTF-8 character it does not finish.
/// \note Tokens are bytes, so a generation cut off by the length limit or a cancel
///       can stop partway through a character. Those bytes are invalid UTF-8, and
///       json::dump() throws on them.
inline size_t incomplete_utf8_tail(const std::string& text) {
    for (size_t back = 1; back <= 4 && back <= text.size(); back++) {
        unsigned char c = text[text.size() - back];
        if ((c & 0xC0) == 0x80) {
            continue;  // a continuation byte
        }
        size_t width = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return width > back ? back : 0;
    }
    return 0;
}

/// The length (1-4) of the well-formed UTF-8 character that starts at p, given the avail bytes there;
/// 0 when those bytes cannot start one (drop the first byte and look again at the next);
/// -1 when they are a well-formed start that needs more bytes.
/// \note Well-formed is Unicode's Table 3-7, which json::dump() enforces. A lead byte alone is not enough:
///       the model can emit the byte tokens of a character and then stop it with another token
///       (e.g. 0xE2 followed by "a"), and such bytes reached dump() and failed the request with a 500.
inline int utf8_char_len(const unsigned char* p, size_t avail) {
    const unsigned char c = p[0];
    int len;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (c < 0x80) return 1;
    else if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; else if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; else if (c == 0xF4) hi = 0x8F; }
    else return 0;
    for (int k = 1; k < len; k++) {
        if ((size_t)k >= avail) return -1;
        const unsigned char b = p[k];
        if (k == 1 ? (b < lo || b > hi) : (b & 0xC0) != 0x80) return 0;
    }
    return len;
}

/// text without the bytes that are not part of a well-formed UTF-8 character (see utf8_char_len),
/// including a character the end of text cuts off
inline std::string sanitize_utf8(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    const unsigned char* p = reinterpret_cast<const unsigned char*>(text.data());
    size_t pos = 0;
    while (pos < text.size()) {
        const int len = utf8_char_len(p + pos, text.size() - pos);
        if (len > 0) {
            out.append(text, pos, len);
            pos += len;
        } else {
            pos++;
        }
    }
    return out;
}

/// \brief wstream_buf class
/// \note This class is used to avoid partial UTF-8 characters from the decoder.
struct wstream_buf : std::streambuf {
    std::vector<char> utf8_buffer;  // Buffer for incomplete UTF-8 sequences
    std::ostream& os;
    wstream_buf(std::ostream& os) : os(os) {}

    /// \brief called when a sequence of chars is written
    /// \param s the string to write
    /// \param n the number of characters to write
    /// \return the number of characters written
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        // Append new data to buffer
        utf8_buffer.insert(utf8_buffer.end(), s, s + n);
        
        // Process complete UTF-8 sequences
        size_t pos = 0;
        while (pos < utf8_buffer.size()) {
            const int len = utf8_char_len(reinterpret_cast<const unsigned char*>(utf8_buffer.data()) + pos, utf8_buffer.size() - pos);
            if (len == 0) {
                // Not part of a well-formed character: skip it
                pos++;
                continue;
            }
            if (len < 0) {
                // Incomplete sequence, keep it in buffer
                break;
            }
            const size_t seq_len = (size_t)len;
            os.write(&utf8_buffer[pos], seq_len);
            pos += seq_len;
        }
     
        // Remove processed bytes from buffer
        if (pos > 0) {
            utf8_buffer.erase(utf8_buffer.begin(), utf8_buffer.begin() + pos);
        }
        
        return n;
    }

    /// \brief fallback for single-char writes
    /// \param ch the character to write
    /// \return the character written
    int_type overflow(int_type ch) override {
        if (ch == traits_type::eof()) return ch;
        char c = traits_type::to_char_type(ch);
        
        // Process the single byte (xsputn appends it to the buffer)
        xsputn(&c, 1);
        return ch;
    }
};

class nullbuf : public std::streambuf {
    protected:
        int overflow(int c) override { return c; }  // Simply ignore input
    };
    
    class nullstream : public std::ostream {
    public:
        nullstream() : std::ostream(&m_sb) {}
    private:
        nullbuf m_sb;
};