/// \file chat_continue.hpp
/// \brief Continue the final assistant message instead of starting a new turn
/// \note The request field continue_final_message (vLLM and Hugging Face): the
///       chat is rendered without a generation prompt and cut right after the
///       final message's text, so the model writes on from there. The cut follows
///       transformers' apply_chat_template(continue_final_message=True), so the
///       same template gives the same prompt here as there, except where the
///       template writes a finished message differently from how it opens a reply
///       (see chat_continue_prompt).
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include <nlohmann/json.hpp>

/// \brief Cut a rendered chat right after the text of its final message
/// \param rendered the chat rendered without a generation prompt
/// \param final_message the last message as the template was given it; its
///        content is a string or an array of typed parts, whose last text part counts
/// \param text_at set to where the message's text starts in the result, when not null
/// \return the rendered chat up to the end of the final message's text
/// \note Whitespace around the text is matched loosely, as transformers does:
///       templates often trim the content. Trailing whitespace is kept only when
///       the template rendered it.
/// \throws std::invalid_argument when the final message has no text, or the
///         template did not render that text as it is (some templates rewrite the
///         last assistant message, e.g. by moving a <think> block out of it)
std::string chat_continue_final_message(const std::string& rendered, const nlohmann::ordered_json& final_message,
                                        size_t* text_at = nullptr);

/// \brief Renders a chat: the messages, and whether to end on a generation prompt
using chat_render_fn = std::function<std::string(const nlohmann::ordered_json& messages, bool add_generation_prompt)>;

/// \brief A prompt that continues the final message
struct chat_continued {
    std::string prompt; ///< what the model is fed
    std::string turn;   ///< the continued assistant turn: the template's opening of it and the message's text
};

/// \brief Build the prompt that continues the final assistant message of a chat
/// \param render renders the chat with the model's template
/// \param messages the chat, ending with the assistant message to continue
/// \note The prompt is transformers' cut wherever the template opens a reply the
///       way it renders a finished message. Where it does not, the cut reads as a
///       finished turn and the model ends it (Qwen3.5 leaves the empty think block
///       out of a finished message, Gemma4 with thinking off its empty thought
///       channel), so the prompt is then the reply's opening followed by the text.
/// \throws std::invalid_argument as chat_continue_final_message
chat_continued chat_continue_prompt(const chat_render_fn& render, const nlohmann::ordered_json& messages);
