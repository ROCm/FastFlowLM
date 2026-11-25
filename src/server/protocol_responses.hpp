/// \file protocol_responses.hpp
/// \brief Write a shared chat turn as a Responses object or named SSE events.
#pragma once

#include <ctime>
#include <map>
#include "protocol_openai_chat.hpp"
#include "responses_request.hpp"

namespace flm::responses {
class sink final : public chat_sink {
public:
    sink(const json& request, std::function<void(const json&)> send,
         std::function<void(const json&, bool)> send_stream)
        : request_(request), send_(std::move(send)), send_stream_(std::move(send_stream)) {}

    void on_begin(const chat_turn_begin& begin) override {
        stream_ = begin.stream;
        const auto suffix = chat_new_tool_call_id().substr(5);
        response_ = {{"id","resp_" + suffix},{"object","response"},{"created_at",std::time(nullptr)},
            {"model",begin.model},{"status","in_progress"},{"output",json::array()},
            {"error",nullptr},{"incomplete_details",nullptr},{"usage",nullptr},{"store",false},
            {"tools",request_.value("tools", json::array())},{"tool_choice",request_.value("tool_choice", json("auto"))},
            {"parallel_tool_calls",!provided(request_,"parallel_tool_calls") || request_["parallel_tool_calls"].get<bool>()}};
        for (auto key : {"instructions", "max_output_tokens", "reasoning", "text", "temperature", "top_p"})
            if (request_.contains(key)) response_[key] = request_[key];
    }

    void on_start(const chat_turn_start&) override {
        if (stream_) {
            started_ = true;
            emit("response.created", {{"response",response_}});
            emit("response.in_progress", {{"response",response_}});
        }
    }

    void on_delta(const chat_msg_delta& d) override {
        if (d.is_tool_call()) {
            auto it = calls_.find(d.tool_index);
            if (it == calls_.end()) {
                const auto i = add("fc", {{"type","function_call"},{"name",d.tool_name},
                    {"call_id",d.tool_id},{"arguments",""}});
                it = calls_.emplace(d.tool_index, i).first;
            }
            const auto i = it->second;
            output()[i]["arguments"].get_ref<std::string&>() += d.tool_arguments;
            if (stream_ && !d.tool_arguments.empty()) {
                auto fields = address(i);
                fields["delta"] = d.tool_arguments;
                emit("response.function_call_arguments.delta", std::move(fields));
            }
            return;
        }
        if (!d.reasoning.empty()) append_text(true, d.reasoning);
        if (!d.content.empty()) append_text(false, d.content);
    }

    void on_end(const chat_turn_end& end) override {
        if (terminal_) return;
        if (!stream_ && end.generated && end.message) {
            const auto& m = *end.message;
            if (!m.reasoning.empty()) append_text(true, m.reasoning);
            if (!m.content.empty()) append_text(false, m.content);
            for (const auto& c : m.tool_calls)
                add("fc", {{"type","function_call"},{"name",c.name},{"call_id",c.id},{"arguments",c.arguments}});
        }
        response_["usage"] = {{"input_tokens",end.prompt_tokens},{"output_tokens",end.generated_tokens},
            {"total_tokens",end.prompt_tokens + end.generated_tokens},
            {"input_tokens_details",{{"cached_tokens",end.cached_prompt_tokens}}},
            {"output_tokens_details",{{"reasoning_tokens",0}}}};
        if (end.finish == chat_finish::FAILED || end.finish == chat_finish::CANCEL) {
            fail(end.finish == chat_finish::CANCEL ? "Generation cancelled" : "Generation failed");
        } else if (end.finish == chat_finish::LENGTH) {
            response_["incomplete_details"] = {{"reason","max_output_tokens"}};
            finish("incomplete");
        } else {
            finish("completed");
        }
    }

    void on_error(const chat_error& error) override {
        if (terminal_) return;
        if (started_) fail(error.message);
        else {
            terminal_ = true;
            send_(openai_error_json(error));
        }
    }

private:
    json& output() { return response_.at("output"); }

    void emit(const std::string& type, json fields, bool last = false) {
        fields["type"] = type;
        fields["sequence_number"] = sequence_++;
        send_stream_("event: " + type + "\ndata: " + fields.dump() + "\n\n", last);
    }

    json address(size_t i) {
        return {{"item_id",output()[i]["id"]},{"output_index",i}};
    }

    size_t add(const std::string& prefix, json item) {
        const auto i = output().size();
        item["id"] = prefix + "_" + response_.at("id").get_ref<const std::string&>().substr(5) + "_" + std::to_string(i);
        item["status"] = "in_progress";
        output().push_back(std::move(item));
        if (stream_) emit("response.output_item.added", {{"output_index",i},{"item",output()[i]}});
        return i;
    }

    void append_text(bool reasoning, const std::string& text) {
        const char* type = reasoning ? "reasoning" : "message";
        const char* parts = reasoning ? "summary" : "content";
        const char* index = reasoning ? "summary_index" : "content_index";
        if (output().empty() || output().back()["type"] != type) {
            json item = {{"type",type},{parts,json::array()}};
            if (!reasoning) item["role"] = "assistant";
            const auto i = add(reasoning ? "rs" : "msg", std::move(item));
            json part = {{"type",reasoning ? "summary_text" : "output_text"},{"text",""}};
            if (!reasoning) part["annotations"] = json::array();
            output()[i][parts].push_back(part);
            if (stream_) {
                auto fields = address(i);
                fields[index] = 0;
                fields["part"] = std::move(part);
                emit(reasoning ? "response.reasoning_summary_part.added" : "response.content_part.added", std::move(fields));
            }
        }
        output().back()[parts][0]["text"].get_ref<std::string&>() += text;
        if (stream_) {
            auto fields = address(output().size() - 1);
            fields[index] = 0;
            fields["delta"] = text;
            if (!reasoning) fields["logprobs"] = json::array();
            emit(reasoning ? "response.reasoning_summary_text.delta" : "response.output_text.delta", std::move(fields));
        }
    }

    void done(size_t i, const std::string& status) {
        auto& item = output()[i];
        item["status"] = status == "completed" ? "completed" : "incomplete";
        if (!stream_) return;
        auto fields = address(i);
        if (item["type"] == "function_call") {
            // Some clients execute as soon as output_item.done arrives. A partial
            // call belongs in the terminal snapshot, never in a callable done event.
            if (status != "completed") return;
            fields["name"] = item["name"];
            fields["arguments"] = item["arguments"];
            emit("response.function_call_arguments.done", std::move(fields));
        } else {
            const bool reasoning = item["type"] == "reasoning";
            auto& part = item[reasoning ? "summary" : "content"][0];
            fields[reasoning ? "summary_index" : "content_index"] = 0;
            auto text = fields;
            text["text"] = part["text"];
            if (!reasoning) text["logprobs"] = json::array();
            emit(reasoning ? "response.reasoning_summary_text.done" : "response.output_text.done", std::move(text));
            fields["part"] = part;
            emit(reasoning ? "response.reasoning_summary_part.done" : "response.content_part.done", std::move(fields));
        }
        emit("response.output_item.done", {{"output_index",i},{"item",item}});
    }

    void fail(const std::string& why) {
        response_["error"] = {{"code","server_error"},{"message",why}};
        finish("failed");
    }

    void finish(const std::string& status) {
        if (stream_ && !started_) on_start({}); // cancelled during prefill
        for (size_t i = 0; i < output().size(); ++i) done(i, status);
        response_["status"] = status;
        terminal_ = true;
        if (stream_) emit("response." + status, {{"response",response_}}, true);
        else send_(response_);
    }

    const json& request_;
    std::function<void(const json&)> send_;
    std::function<void(const json&, bool)> send_stream_;
    bool stream_ = false, started_ = false, terminal_ = false;
    size_t sequence_ = 0;
    std::map<int, size_t> calls_;
    json response_;
};
} // namespace flm::responses
