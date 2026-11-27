/// \file protocol_anthropic.hpp
/// \brief Write a single chat turn as an Anthropic Message or a sequence of named SSE events.
#pragma once
#include "protocol_openai_chat.hpp"
#include "anthropic_request.hpp"

namespace flm::anthropic {
class sink final : public chat_sink {
    using json = nlohmann::ordered_json;
    enum class block { none, thinking, text, tool };
public:
    sink(std::function<void(const json&)> send, std::function<void(const json&,bool)> stream,
         display thinking_display = display::local)
        : send_(std::move(send)), stream_(std::move(stream)), display_(thinking_display) {}
    void on_begin(const chat_turn_begin& begin) override {
        streaming_ = begin.stream;
        message_ = {{"type","message"},{"id","msg_"+chat_new_tool_call_id().substr(5)},
                    {"role","assistant"},{"model",begin.model},{"content",json::array()},
                    {"stop_reason",nullptr},{"stop_sequence",nullptr}};
    }
    void on_start(const chat_turn_start& s) override {
        if (!streaming_ || started_) return;
        message_["usage"] = usage(s.prompt_tokens,s.cached_prompt_tokens,0);
        started_ = true;
        emit("message_start",{{"message",message_}});
    }
    void on_delta(const chat_msg_delta& d) override {
        if (terminal_) return;
        if (d.is_tool_call()) {
            if (active_!=block::tool || tool_index_!=d.tool_index) {
                open(block::tool,{{"type","tool_use"},{"id",d.tool_id},{"name",d.tool_name},{"input",json::object()}});
                tool_index_ = d.tool_index;
            }
            if (!d.tool_arguments.empty()) delta({{"type","input_json_delta"},{"partial_json",d.tool_arguments}});
        } else if (!d.reasoning.empty()) {
            if (active_!=block::thinking) open(block::thinking,{{"type","thinking"},{"thinking",""},{"signature",""}});
            thinking_bytes_ += d.reasoning.size();
            if (display_==display::local) delta({{"type","thinking_delta"},{"thinking",d.reasoning}});
        } else if (!d.content.empty()) {
            if (active_!=block::text) open(block::text,{{"type","text"},{"text",""}});
            delta({{"type","text_delta"},{"text",d.content}});
        }
    }
    void on_end(const chat_turn_end& end) override {
        if (terminal_) return;
        if (end.finish==chat_finish::FAILED || end.finish==chat_finish::CANCEL) {
            on_error({500,"server_error","Generation did not complete",{},std::nullopt});
            return;
        }
        const char* reason = end.finish==chat_finish::LENGTH ? "max_tokens" :
                             end.finish==chat_finish::TOOL_CALLS ? "tool_use" : "end_turn";
        auto counts = usage(end.prompt_tokens,end.cached_prompt_tokens,end.generated_tokens);
        if (streaming_) {
            close();
            emit("message_delta",{{"delta",{{"stop_reason",reason},{"stop_sequence",nullptr}}},{"usage",counts}});
            terminal_ = true;
            emit("message_stop",json::object(),true);
        } else {
            auto& content = message_["content"];
            const auto& msg = *end.message;
            if (!msg.reasoning.empty()) content.push_back({{"type","thinking"},
                {"thinking",display_==display::local ? msg.reasoning : summary(msg.reasoning.size())},{"signature",""}});
            if (!msg.content.empty()) content.push_back({{"type","text"},{"text",msg.content}});
            for (const auto& call : msg.tool_calls) {
                auto args = json::parse(call.arguments,nullptr,false);
                if (!args.is_object()) {
                    if (end.finish==chat_finish::LENGTH) continue; // incomplete call is not executable
                    on_error({500,"server_error","Model produced invalid tool input",{},std::nullopt});
                    return;
                }
                content.push_back({{"type","tool_use"},{"id",call.id},{"name",call.name},{"input",std::move(args)}});
            }
            message_["usage"] = std::move(counts);
            message_["stop_reason"] = reason;
            terminal_ = true;
            send_(message_);
        }
    }
    void on_error(const chat_error& e) override {
        if (terminal_) return;
        terminal_ = true;
        if (started_) {
            stream_(flm::error_model::stream_error_frame(flm::error_model::Protocol::Anthropic,
                    {e.status,e.type,e.message,e.param,e.session_cleared}),true);
        } else send_(openai_error_json(e)); // canonical value; HTTP layer selects the route writer
    }
private:
    static json usage(int prompt, int cached, int output) {
        const int read = std::clamp(cached,0,std::max(0,prompt));
        return {{"input_tokens",std::max(0,prompt)-read},{"cache_creation_input_tokens",0},
                {"cache_read_input_tokens",read},{"output_tokens",output}};
    }
    void emit(const char* type, json fields, bool last=false) {
        fields["type"] = type;
        stream_(std::string("event: ")+type+"\ndata: "+flm::error_model::dump_json(fields)+"\n\n",last);
    }
    void delta(json data) { emit("content_block_delta",{{"index",index_},{"delta",std::move(data)}}); }
    void close() {
        if (active_==block::none) return;
        if (active_==block::thinking) {
            if (display_==display::summarized) delta({{"type","thinking_delta"},{"thinking",summary(thinking_bytes_)}});
            thinking_bytes_=0;
            delta({{"type","signature_delta"},{"signature",""}});
        }
        emit("content_block_stop",{{"index",index_}});
        active_ = block::none;
        ++index_;
    }
    void open(block kind, json value) {
        close();
        active_ = kind;
        emit("content_block_start",{{"index",index_},{"content_block",std::move(value)}});
    }
    std::function<void(const json&)> send_;
    std::function<void(const json&,bool)> stream_;
    std::string summary(size_t bytes) const {
        return display_==display::omitted ? "" : "Local reasoning: "+std::to_string(bytes)+" bytes.";
    }
    display display_;
    size_t thinking_bytes_=0;
    json message_;
    bool streaming_ = false, started_ = false, terminal_ = false;
    block active_ = block::none;
    int index_ = 0, tool_index_ = -1;
};
} // namespace flm::anthropic
