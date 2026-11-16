/// \file log_privacy.hpp
/// \brief whether the log may carry user content (request text, model replies, transcripts)
/// \note By default it may not: the server log keeps metadata only, as vLLM and SGLang do
///       (request logging is opt-in there too). Two environment variables turn content on:
///         FLM_LOG_CONTENT=1      request bodies with their text (cut as before: message text to its first
///                                and last 10 characters, other long strings to their ends), the
///                                "Model RAW Output:" dump, ASR transcripts
///         FLM_LOG_RAW_OUTPUT=1   the "Model RAW Output:" dump and transcripts only (the older switch;
///                                diagnostics that parse per-turn output on that marker set it)
///       Unset, empty or "0" leave a switch off. These functions read the environment on every call;
///       the reply dump is decided when a model object is created (AutoModel::log_raw_output).
#pragma once

#include <cstdlib>
#include <cstring>

namespace flm::log_privacy {

/// whether environment variable `name` is set to something other than "" or "0"
inline bool env_switch_on(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// FLM_LOG_CONTENT: request bodies, replies and transcripts go to the log
inline bool content_enabled() {
	return env_switch_on("FLM_LOG_CONTENT");
}

/// the "Model RAW Output:" dump of every reply (and ASR transcripts): FLM_LOG_CONTENT or FLM_LOG_RAW_OUTPUT
inline bool raw_output_enabled() {
	return content_enabled() || env_switch_on("FLM_LOG_RAW_OUTPUT");
}

}  // namespace flm::log_privacy
