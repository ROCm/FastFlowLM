/// \file json_log.hpp
/// \brief keeps request bodies short, and by default free of user content, in the server log
#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <vector>
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

/// whether a string under `key` is a setting (a model tag, a role, an enum), not text a user wrote; only these
/// strings stay readable in a body logged without its content
inline bool is_setting_key(const std::string& key) {
	static const char* const kSettings[] = {"model", "role", "type", "object", "encoding_format", "tool_choice",
	                                        "reasoning_effort", "service_tier", "detail", "keep_alive", "truncation"};
	for (const char* setting : kSettings)
		if (key == setting) return true;
	return false;
}

/// what a hidden string is logged as: its length in characters (UTF-8 code points)
inline std::string hidden(const std::string& s) {
	size_t chars = 0;
	for (unsigned char c : s)
		if ((c & 0xC0) != 0x80) ++chars;
	return "<" + std::to_string(chars) + " chars>";
}

/// Parses a request body for the log. With `keep_content`, as parse(body). Without it, every string is replaced by
/// its length as it is read, except the strings of setting keys (is_setting_key; an array's strings count as its
/// key's): the log keeps the request's shape, numbers, flags, model and roles, and no text a user sent. Object keys
/// stay as they are. Throws as nlohmann::json::parse does.
inline nlohmann::json parse(const std::string& body, bool keep_content) {
	if (keep_content) return parse(body);
	using event_t = nlohmann::json::parse_event_t;
	struct Open {
		bool array;
		std::string key;  // the key this object or array is the value of ("" at the top)
	};
	std::vector<Open> open;
	std::string key;  // the last key read in the innermost object
	auto key_of_value = [&]() -> const std::string& {
		static const std::string top;
		if (open.empty()) return top;
		return open.back().array ? open.back().key : key;
	};
	return nlohmann::json::parse(body, [&](int, event_t event, nlohmann::json& parsed) {
		switch (event) {
		case event_t::key:
			key = parsed.get<std::string>();
			break;
		case event_t::object_start:
		case event_t::array_start:
			open.push_back({event == event_t::array_start, key_of_value()});
			break;
		case event_t::object_end:
		case event_t::array_end:
			if (!open.empty()) open.pop_back();
			break;
		case event_t::value:
			if (parsed.is_string() && !is_setting_key(key_of_value())) parsed = hidden(parsed.get_ref<const std::string&>());
			break;
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

/// what(e) for the log: without `keep_content`, a parse error's "; last read: '...'" tail, which quotes the
/// request text, is left out
inline std::string what(const std::exception& e, bool keep_content) {
	if (keep_content) return what(e);
	std::string message = e.what();
	if (const size_t at = message.find("; last read: "); at != std::string::npos) message.resize(at);
	return cut(message, 200);
}

/// `body` as indented JSON; never throws on a string that is not valid UTF-8
inline std::string dump(const nlohmann::json& body) {
	return body.dump(4, ' ', false, nlohmann::json::error_handler_t::replace);
}

}  // namespace json_log
