/// \file json_log.hpp
/// \brief keeps request bodies short, and by default free of user content, in the server log
#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <string_view>
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

/// what a hidden string is logged as: its length in characters (UTF-8 code points)
inline std::string hidden(const std::string& s) {
	size_t chars = 0;
	for (unsigned char c : s)
		if ((c & 0xC0) != 0x80) ++chars;
	return "<" + std::to_string(chars) + " chars>";
}

/// whether `s` reads as a name (a model tag, a parameter name), not as text: 1 to `max` printable ASCII
/// characters with no space
inline bool is_name(const std::string& s, size_t max) {
	if (s.empty() || s.size() > max) return false;
	for (unsigned char c : s)
		if (c <= ' ' || c >= 0x7f) return false;
	return true;
}

/// The one line a request body is logged as when its content must stay out of the log: its top-level parameters
/// only. Numbers, flags and null are shown as they are, and the model tag when it reads as a name (is_name).
/// Every other string is its length ("<42 chars>"), an array its item count, an object its key count. Nothing
/// below the top level is shown or kept, so no key or value inside messages, tools, a schema or metadata reaches
/// the log; a parameter whose name does not read as a name is counted, not shown. Throws as
/// nlohmann::json::parse does.
inline std::string summary(const std::string& body) {
	using event_t = nlohmann::json::parse_event_t;
	using json = nlohmann::json;
	bool is_model = false;  // the top-level key read last is "model"
	const json parsed = json::parse(body, [&](int depth, event_t event, json& value) {
		if (event == event_t::key && depth == 1) is_model = value == "model";
		if (event != event_t::value) return true;
		if (depth > 1) {
			value = nullptr;  // only counted: dropped as it is read
		} else if (value.is_string()) {
			const auto& text = value.get_ref<const std::string&>();
			if (!(depth == 1 && is_model && is_name(text, 64))) value = hidden(text);
		}
		return true;
	});
	const auto brief = [](const json& value) -> json {
		if (value.is_array()) return "<" + std::to_string(value.size()) + " items>";
		if (value.is_object()) return "<" + std::to_string(value.size()) + " keys>";
		return value;
	};
	if (!parsed.is_object()) return brief(parsed).dump();
	json line = json::object();
	size_t unnamed = 0;
	for (const auto& [key, value] : parsed.items()) {
		if (is_name(key, 40)) line[key] = brief(value);
		else ++unnamed;
	}
	if (unnamed) line["<other parameters>"] = unnamed;
	return line.dump();
}

/// a request target for the log: without `keep_content` its query ("?" and what follows) is left out, as a
/// client can put text there
inline std::string target(std::string_view request_target, bool keep_content) {
	const size_t query = keep_content ? std::string_view::npos : request_target.find('?');
	if (query == std::string_view::npos) return std::string(request_target);
	return std::string(request_target.substr(0, query)) + "?<query hidden>";
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
