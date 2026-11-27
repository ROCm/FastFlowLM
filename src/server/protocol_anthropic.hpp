/// \file protocol_anthropic.hpp
/// \brief Assemble one ordered sequence of Messages blocks for both wire modes.
#pragma once
#include "protocol_openai_chat.hpp"

namespace flm::anthropic {
class sink final : public chat_sink {
    using json = nlohmann::ordered_json;
    enum class block { none, thinking, text, tool };
public:
    sink(std::function<void(const json&)> send, std::function<void(const json&,bool)> stream,
         bool streaming, bool show_thinking=true)
        : send_(std::move(send)), stream_(std::move(stream)), streaming_(streaming), show_thinking_(show_thinking) {}
    void on_begin(const chat_turn_begin& begin) override {
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
                if (!close()) return;
                if (d.tool_index<=tool_index_ || d.tool_id.empty() || d.tool_name.empty()) {
                    on_error({500,"server_error","Cannot represent interleaved tool input",{},std::nullopt});
                    return;
                }
                active_=block::tool;
                tool_index_=d.tool_index;
                current_={{"type","tool_use"},{"id",d.tool_id},{"name",d.tool_name},{"input",json::object()}};
            }
            if (!d.tool_arguments.empty()) fragments_.push_back(d.tool_arguments);
            return;
        }
        if (!d.reasoning.empty()) append(block::thinking,d.reasoning);
        if (!d.content.empty() && !terminal_) append(block::text,d.content);
    }
    void on_end(const chat_turn_end& end) override {
        if (terminal_) return;
        if (end.finish==chat_finish::FAILED || end.finish==chat_finish::CANCEL) {
            on_error({500,"server_error","Generation did not complete",{},std::nullopt});
            return;
        }
        if (!close(end.finish==chat_finish::LENGTH)) return;
        const char* reason = end.finish==chat_finish::LENGTH ? "max_tokens" :
                             end.finish==chat_finish::TOOL_CALLS ? "tool_use" : "end_turn";
        auto counts = usage(end.prompt_tokens,end.cached_prompt_tokens,end.generated_tokens);
        if (streaming_) {
            emit("message_delta",{{"delta",{{"stop_reason",reason},{"stop_sequence",nullptr}}},{"usage",counts}});
            terminal_=true;
            emit("message_stop",json::object(),true);
        } else {
            message_["usage"]=std::move(counts);
            message_["stop_reason"]=reason;
            terminal_=true;
            send_(message_);
        }
    }
    void on_error(const chat_error& e) override {
        if (terminal_) return;
        terminal_=true;
        auto error=e;
        if (error.param=="reasoning_effort") {
            error.param="thinking.type";
            error.message="This model does not support the requested thinking mode";
        }
        if (started_) stream_(flm::error_model::stream_error_frame(flm::error_model::Protocol::Anthropic,
            {error.status,error.type,error.message,error.param,error.session_cleared}),true);
        else send_(openai_error_json(error)); // canonical value; HTTP selects the route's dialect
    }
private:
    static json usage(int prompt, int cached, int output) {
        const int read=std::clamp(cached,0,std::max(0,prompt));
        return {{"input_tokens",std::max(0,prompt)-read},{"cache_creation_input_tokens",0},
                {"cache_read_input_tokens",read},{"output_tokens",output}};
    }
    void emit(const char* type,json fields,bool last=false) {
        fields["type"]=type;
        stream_(std::string("event: ")+type+"\ndata: "+flm::error_model::dump_json(fields)+"\n\n",last);
    }
    void start_block() { emit("content_block_start",{{"index",index_},{"content_block",current_}}); }
    void delta(json value) { emit("content_block_delta",{{"index",index_},{"delta",std::move(value)}}); }
    void append(block kind,const std::string& value) {
        const bool thinking=kind==block::thinking;
        const char* key=thinking ? "thinking" : "text";
        if (active_!=kind) {
            if (!close()) return;
            active_=kind; current_={{"type",key},{key,""}};
            if (thinking) current_["signature"]="";
            if (streaming_) start_block();
        }
        if (thinking && !show_thinking_) return;
        if (streaming_) delta({{"type",thinking ? "thinking_delta" : "text_delta"},{key,value}});
        else current_[key].get_ref<std::string&>()+=value;
    }
    bool close(bool capped=false) {
        if (active_==block::none) return true;
        if (active_==block::tool) {
            // Once an SDK sees tool_use it cannot retract it. Hold only this call until
            // its input is known to be an object; never publish a truncated callable block.
            std::string input;
            for (const auto& f : fragments_) input+=f;
            auto args=json::parse(input,nullptr,false);
            if (!args.is_object()) {
                if (!capped) on_error({500,"server_error","Model produced invalid tool input",{},std::nullopt});
                active_=block::none; fragments_.clear();
                return capped;
            }
            if (streaming_) {
                start_block();
                for (const auto& f : fragments_) delta({{"type","input_json_delta"},{"partial_json",f}});
            } else current_["input"]=std::move(args);
            fragments_.clear();
        } else if (streaming_ && active_==block::thinking) delta({{"type","signature_delta"},{"signature",""}});
        if (streaming_) emit("content_block_stop",{{"index",index_}});
        else message_["content"].push_back(std::move(current_));
        active_=block::none; ++index_;
        return true;
    }
    std::function<void(const json&)> send_;
    std::function<void(const json&,bool)> stream_;
    bool streaming_, show_thinking_, started_=false, terminal_=false;
    json message_, current_;
    std::vector<std::string> fragments_;
    block active_=block::none;
    int index_=0, tool_index_=-1;
};
} // namespace flm::anthropic
