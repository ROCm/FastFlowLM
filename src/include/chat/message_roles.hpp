/// \file message_roles.hpp
/// \brief Instruction messages (`system`, `developer`) put where the model's chat template renders them, and the
///        merge of consecutive messages that FLM does before every chat template.
/// \note OpenAI defines `developer` as `system` under a new name and allows either role anywhere in the conversation.
///       Templates differ: Qwen3 drops a `developer` message, Qwen3.5 raises on it and on a `system` message after the
///       first turn, gpt-oss renders a leading `developer` message itself. What a template does is found once, by
///       rendering (probe_message_roles), never from the model name. A later instruction is never moved: an agent
///       appends one on many turns, and moving it would change the prompt prefix that prefix reuse resumes from.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace flm::chat {

/// \brief What a chat template does with instruction messages, found by rendering (probe_message_roles)
struct message_role_support {
    /// \brief a leading `developer` message is rendered as a role of its own (gpt-oss); otherwise it is sent as `system`
    bool renders_developer = false;
    /// \brief a `system` message after the first turn is rendered in place; otherwise it is sent as a user turn,
    ///        wrapped in <system-reminder> tags, at the same position
    bool accepts_later_system = true;
};

inline constexpr const char* system_reminder_open = "<system-reminder>";
inline constexpr const char* system_reminder_close = "</system-reminder>";

/// \brief Render `messages` with the model's template; may throw
using message_renderer = std::function<std::string(const nlohmann::ordered_json& messages)>;

/// \brief Find what a template does with `developer` and with a `system` message after the first turn
/// \note A render that throws, or leaves the needle out, counts as not supported. A template that renders any role
///       name as written (generic ChatML) renders `developer` too, but the model was not trained on it, so
///       `renders_developer` also needs the template to leave out a role it cannot know. A later `system` message
///       counts as accepted only when it is rendered after the turns before it: a template that moves every system
///       message to the top would change the prompt prefix on every turn.
///       The probe renders without tools and with the model's current options; templates seen so far (Qwen2.5, Qwen3,
///       Qwen3.5, Gemma3, gpt-oss) give the same answer with tools.
inline message_role_support probe_message_roles(const message_renderer& render) {
    using json = nlohmann::ordered_json;
    const std::string needle = "<FLM Role Needle>";
    const std::string answer = "<FLM Probe Answer>";
    const auto rendered = [&](const json& messages) {
        try {
            return render(messages);
        } catch (...) {
            return std::string();
        }
    };
    const auto renders = [&](const json& messages) { return rendered(messages).find(needle) != std::string::npos; };
    const auto leading = [&](const char* role) {
        return json::array({{{"role", role}, {"content", needle}}, {{"role", "user"}, {"content", "Hello."}}});
    };
    message_role_support support;
    support.renders_developer = renders(leading("developer")) && !renders(leading("flm-probe-unknown-role"));
    const std::string later = rendered(json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "Hello."}},
        {{"role", "assistant"}, {"content", answer}},
        {{"role", "system"}, {"content", needle}},
        {{"role", "user"}, {"content", "Go on."}},
    }));
    const auto at_answer = later.find(answer), at_needle = later.find(needle);
    support.accepts_later_system =
        at_needle != std::string::npos && at_answer != std::string::npos && at_needle > at_answer;
    return support;
}

/// \brief A request whose messages cannot be normalized; the server answers 400 with what()
class message_error : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

namespace detail {

inline bool is_instruction_role(const std::string& role) {
    return role == "system" || role == "developer";
}

inline bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/// \brief Whether `text`, without surrounding whitespace, is already one <system-reminder> block
inline bool is_system_reminder(const std::string& text) {
    std::size_t begin = 0, end = text.size();
    while (begin < end && is_space(text[begin])) ++begin;
    while (end > begin && is_space(text[end - 1])) --end;
    const std::string open = system_reminder_open, close = system_reminder_close;
    return end - begin >= open.size() + close.size() && text.compare(begin, open.size(), open) == 0 &&
           text.compare(end - close.size(), close.size(), close) == 0;
}

/// \brief `text` as one <system-reminder> block
/// \param after_user the block joins the user message before it, so it starts on a new line: the server joins the
///        text parts of a merged user turn with nothing between them (for the same reason the block ends with one)
inline std::string wrap_system_reminder(const std::string& text, bool after_user) {
    const std::string lead = after_user ? "\n" : "";
    if (is_system_reminder(text)) return lead + text;
    return lead + system_reminder_open + "\n" + text + "\n" + system_reminder_close + "\n";
}

/// \brief The text of an instruction message's content, its text parts joined as the server joins them
///        (normalize_template); a missing or null content is empty
/// \throws message_error for content that is not a string, an array of text parts, or null: a part that is not
///         text would bypass the server's rule that only a user message holds images, and content that cannot be
///         read would be dropped without a word
inline std::string instruction_text(const nlohmann::ordered_json& message, std::size_t index) {
    const std::string where = "messages[" + std::to_string(index) + "]";
    const std::string role = message.value("role", "");
    const auto content = message.find("content");
    if (content == message.end() || content->is_null()) return std::string();
    if (content->is_string()) return content->get<std::string>();
    if (!content->is_array()) throw message_error(where + ".content: must be a string or an array of content parts");
    std::string text;
    for (std::size_t c = 0; c < content->size(); ++c) {
        const auto& part = (*content)[c];
        const auto t = part.is_object() ? part.find("text") : part.end();
        const bool is_text = part.is_object() && (!part.contains("type") || part["type"] == "text");
        if (!is_text || t == part.end() || !t->is_string()) {
            const std::string type = part.is_object() && part.contains("type") && part["type"].is_string()
                                         ? part["type"].get<std::string>() : std::string("?");
            throw message_error(where + ".content[" + std::to_string(c) + "]: a '" + role +
                                "' message holds text parts only, not '" + type + "'");
        }
        text += t->get<std::string>();
    }
    return text;
}

/// \brief Whether `message` has nothing to render: no content, null, an empty string or no parts
inline bool has_no_content(const nlohmann::ordered_json& message) {
    const auto content = message.find("content");
    return content == message.end() || content->is_null() || (content->is_string() && content->get<std::string>().empty()) ||
           (content->is_array() && content->empty());
}

/// \brief Append the content of `message` to `parts` as typed parts; an empty string adds nothing
inline void append_content_parts(nlohmann::ordered_json& parts, const nlohmann::ordered_json& message) {
    const auto content = message.find("content");
    if (content == message.end()) return;
    if (content->is_array()) {
        for (const auto& item : *content) parts.push_back(item);
    } else if (content->is_string()) {
        const std::string text = content->get<std::string>();
        if (!text.empty()) parts.push_back({{"type", "text"}, {"text", text}});
    }
}

}  // namespace detail

/// \brief Give every `system` and `developer` message a role and a position the template renders
/// \return for each message, whether it is left out: an instruction message with no text whose role had to change
///         (a `developer` sent as `system`, or a later instruction sent as a user turn) would only add an empty turn
/// \throws message_error for a later instruction that becomes a user turn and holds anything but text
/// \note The leading run of instruction messages becomes one role: `developer` when the template renders it and the
///       run has a `developer` message, else `system`. A later instruction message stays where it is: as `system` when
///       the template accepts that, else as a user turn wrapped in <system-reminder> tags. No instruction with text is
///       dropped or moved, so a conversation that only grows keeps rendering the same prefix.
inline std::vector<bool> place_instruction_messages(nlohmann::ordered_json& messages, const message_role_support& support) {
    std::vector<bool> left_out(messages.is_array() ? messages.size() : 0, false);
    if (!messages.is_array()) return left_out;
    std::size_t lead = 0;
    bool lead_has_developer = false;
    for (; lead < messages.size(); ++lead) {
        if (!messages[lead].is_object()) break;
        const std::string role = messages[lead].value("role", "");
        if (!detail::is_instruction_role(role)) break;
        lead_has_developer = lead_has_developer || role == "developer";
    }
    const char* lead_role = support.renders_developer && lead_has_developer ? "developer" : "system";
    // Whether the last message kept so far is a user turn: an instruction left out is not a message, so the one
    // after it joins whatever came before it, not the (emptied) turn it was changed into
    bool after_user = false;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        auto& message = messages[i];
        if (!message.is_object()) {
            after_user = false;
            continue;
        }
        const std::string role = message.value("role", "");
        if (!detail::is_instruction_role(role)) {
            after_user = role == "user";
            continue;
        }
        if (i >= lead && !support.accepts_later_system) {
            const std::string text = detail::instruction_text(message, i);
            if (text.empty()) {
                left_out[i] = true;
            } else {
                message["content"] = detail::wrap_system_reminder(text, after_user);
                after_user = true;
            }
            message["role"] = "user";
            continue;
        }
        const std::string placed = i < lead ? lead_role : "system";
        if (placed != role && detail::has_no_content(message)) left_out[i] = true;
        else after_user = false;
        message["role"] = placed;
    }
    return left_out;
}

/// \brief Normalize a chat request's messages before the chat template
/// \param messages the request's messages
/// \param support what the model's template does with instruction messages
/// \param original_indices if set, receives the request index of each normalized message
/// \return the messages with instruction messages placed (place_instruction_messages), consecutive user messages and
///         consecutive instruction messages of the same role merged into one message with typed content (as Ollama
///         does), and earlier reasoning removed from assistant messages
/// \throws message_error when a message cannot be placed (place_instruction_messages)
inline nlohmann::ordered_json normalize_messages(nlohmann::ordered_json messages, const message_role_support& support,
                                                 std::vector<std::size_t>* original_indices = nullptr) {
    using json = nlohmann::ordered_json;
    if (messages.empty()) return messages;
    const std::vector<bool> left_out = place_instruction_messages(messages, support);
    std::vector<std::size_t> kept;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (i >= left_out.size() || !left_out[i]) kept.push_back(i);
    }

    json normalized = json::array();
    for (std::size_t k = 0; k < kept.size(); k++) {
        const std::size_t first = kept[k];
        if (original_indices) original_indices->push_back(first);
        auto current_msg = messages[first];
        const std::string role = current_msg.value("role", "");

        if (role == "user" || detail::is_instruction_role(role)) {
            json merged_content_array = json::array();
            while (true) {
                detail::append_content_parts(merged_content_array, messages[kept[k]]);
                if (k + 1 < kept.size() && messages[kept[k + 1]].value("role", "") == role) k++;
                else break;
            }
            current_msg["content"] = merged_content_array;
        }
        else if (role == "assistant") {
            // History drops reasoning by default. The Qwen GPU tool-call policy is applied
            // separately, once the final user-turn boundaries are visible to the template.
            current_msg.erase("thinking");
            current_msg.erase("reasoning");
            current_msg.erase("reasoning_content");
        }
        normalized.push_back(current_msg);
    }
    return normalized;
}

}  // namespace flm::chat
