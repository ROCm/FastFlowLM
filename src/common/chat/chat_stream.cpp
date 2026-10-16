/// \file chat_stream.cpp
/// \brief Per request streaming state

#include "chat/chat_stream.hpp"

#include <random>

std::string chat_new_tool_call_id() {
    static const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<size_t> pick(0, sizeof(ALPHABET) - 2);
    std::string id = "call_";
    for (int i = 0; i < 24; i++) {
        id += ALPHABET[pick(gen)];
    }
    return id;
}

chat_stream::chat_stream(const chat_format_spec& spec, std::string prefill, chat_parse_options options)
    : spec_(spec), options_(options), text_(std::move(prefill)) {}

std::vector<chat_msg_delta> chat_stream::update(const std::string& text, bool is_final) {
    text_ += text;
    chat_msg next = chat_parse(spec_, text_, !is_final, options_);

    // ids are ours, not the model's: keep each call's id for the whole stream
    for (size_t i = 0; i < next.tool_calls.size(); i++) {
        if (i >= tool_ids_.size()) {
            tool_ids_.push_back(chat_new_tool_call_id());
        }
        next.tool_calls[i].id = tool_ids_[i];
    }

    std::vector<chat_msg_delta> deltas = chat_compute_deltas(msg_, next);
    msg_ = std::move(next);
    return deltas;
}
