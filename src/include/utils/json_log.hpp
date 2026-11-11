/// \file json_log.hpp
/// \brief keeps request bodies short in the server log
#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <nlohmann/json.hpp>

namespace json_log {

/// longest string a logged body keeps whole; longer ones (base64 images, raw frames, audio) are cut in the middle
constexpr size_t kMaxString = 256;

/// `s` if it is at most kMaxString bytes, else its first ~`head` and last ~`tail` bytes, both moved back to a UTF-8
/// character boundary, around a note of the full length
inline std::string cut(const std::string& s, size_t head = 32, size_t tail = 16) {
	if (s.size() <= kMaxString) return s;
	auto boundary = [&](size_t i) {
		while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) --i;
		return i;
	};
	const size_t h = boundary(head);
	const size_t t = boundary(s.size() - tail);
	return s.substr(0, h) + "...(" + std::to_string(s.size()) + " bytes)..." + s.substr(t);
}

/// Parses a request body for the log only: every string longer than kMaxString is cut as it is read, so a request
/// carrying megabytes of image data is neither printed whole nor held in memory a second time. Throws as
/// nlohmann::json::parse does.
inline nlohmann::json parse(const std::string& body) {
	return nlohmann::json::parse(body, [](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
		if (event == nlohmann::json::parse_event_t::value && parsed.is_string()) {
			auto& s = parsed.get_ref<std::string&>();
			if (s.size() > kMaxString) s = cut(s);
		}
		return true;
	});
}

/// an exception message for the log or an error reply: a JSON parse error quotes the text it last read, which for
/// a truncated body is everything after the last opening quote (up to the whole body); keep the start, which names
/// the error and its position
inline std::string what(const std::exception& e) {
	return cut(e.what(), 200);
}

/// `body` as indented JSON; never throws on a string that is not valid UTF-8
inline std::string dump(const nlohmann::json& body) {
	return body.dump(4, ' ', false, nlohmann::json::error_handler_t::replace);
}

}  // namespace json_log
