/// \file anthropic_request.hpp
/// \brief Convert Anthropic Messages input into the shared chat request.
#pragma once
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>
#include "chat/positional_tool_results.hpp"

namespace flm::anthropic {
using json = nlohmann::ordered_json;
using fields = std::initializer_list<std::string_view>;

struct request_error : std::invalid_argument {
    request_error(const std::string& path, const std::string& what)
        : std::invalid_argument(path + ": " + what) {}
};

inline void keys(const json& obj, const std::string& path, fields allowed) {
    if (!obj.is_object()) throw request_error(path.empty() ? "body" : path, "expected an object");
    for (auto it = obj.begin(); it != obj.end(); ++it)
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
            throw request_error(path.empty() ? it.key() : path + "." + it.key(), "unsupported field");
}

inline const json& member(const json& obj, const char* key, const std::string& path, json::value_t type) {
    auto it = obj.find(key);
    if (it == obj.end() || it->type() != type)
        throw request_error(path.empty() ? key : path + "." + key, "missing or incorrect type");
    return *it;
}
inline std::string text(const json& obj, const char* key, const std::string& path) {
    return member(obj, key, path, json::value_t::string).get<std::string>();
}
inline int integer(const json& obj, const char* key, const std::string& path, int min) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_number_integer() || *it < min || *it > std::numeric_limits<int>::max())
        throw request_error(path.empty() ? key : path + "." + key, "expected an integer in range");
    return it->get<int>();
}
inline std::string at_index(const std::string& p, size_t i) { return p + "[" + std::to_string(i) + "]"; }

inline std::string text_blocks(const json& content, const std::string& p) {
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array()) throw request_error(p, "expected text or text blocks");
    std::string result;
    for (size_t i = 0; i < content.size(); ++i) {
        const auto at = at_index(p,i);
        const auto& b = content[i];
        if (text(b,"type",at) != "text") throw request_error(at+".type","only text supported here");
        keys(b,at,{"type","text","cache_control"});
        if (i) result += "\n";
        result += text(b,"text",at);
    }
    return result;
}

inline json prompt_fields(const json& req, std::string& first_image) {
    json result = json::object(), out = json::array();
    if (req.contains("system")) out.push_back({{"role","system"},{"content",text_blocks(req.at("system"),"system")}});
    std::set<std::string> seen, pending;
    const auto& messages = member(req,"messages","",json::value_t::array);
    if (messages.empty()) throw request_error("messages","expected at least one turn");
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto at = at_index("messages",i);
        const auto& msg = messages[i];
        keys(msg,at,{"role","content"});
        const auto role = text(msg,"role",at);
        if (role != "user" && role != "assistant") throw request_error(at+".role","expected user or assistant");
        if (!msg.contains("content")) throw request_error(at+".content","required");
        const auto& c = msg["content"];
        if (c.is_string()) {
            if (!pending.empty()) throw request_error(at+".content","missing tool results");
            out.push_back(msg);
            continue;
        }
        if (!c.is_array() || c.empty()) throw request_error(at+".content","expected text or nonempty blocks");
        json parts = json::array(), calls = json::array();
        std::string reasoning, answer;
        if (role == "assistant" && !pending.empty()) throw request_error(at,"missing tool results");
        for (size_t j = 0; j < c.size(); ++j) {
            const auto p = at_index(at+".content",j);
            const auto& b = c[j];
            const auto type = text(b,"type",p);
            if (type == "tool_result" && role == "user") {
                keys(b,p,{"type","tool_use_id","content","is_error","cache_control"});
                const auto id = text(b,"tool_use_id",p);
                if (!parts.empty() || !pending.erase(id)) throw request_error(p+".tool_use_id","late, unknown, or duplicate tool result");
                if (b.contains("is_error") && member(b,"is_error",p,json::value_t::boolean).get<bool>())
                    throw request_error(p+".is_error","error results are not representable");
                out.push_back({{"role","tool"},{"tool_call_id",id},{"content",b.contains("content") ? text_blocks(b["content"],p+".content") : ""}});
            } else if (type == "tool_use" && role == "assistant") {
                keys(b,p,{"type","id","name","input","cache_control"});
                const auto id = text(b,"id",p), name = text(b,"name",p);
                if (id.empty() || !seen.insert(id).second) throw request_error(p+".id","empty or duplicate tool ID");
                if (name.empty()) throw request_error(p+".name","expected a tool name");
                pending.insert(id);
                calls.push_back({{"id",id},{"type","function"},{"function",{{"name",name},
                    {"arguments",member(b,"input",p,json::value_t::object).dump()}}}});
            } else if (type == "thinking" && role == "assistant") {
                keys(b,p,{"type","thinking","signature","cache_control"});
                if (!answer.empty() || !calls.empty() || !text(b,"signature",p).empty())
                    throw request_error(p,"only leading, unsigned local thinking can be replayed");
                reasoning += text(b,"thinking",p);
            } else if (type == "text") {
                keys(b,p,{"type","text","cache_control"});
                if (!pending.empty()) throw request_error(p,"text cannot precede outstanding tool results or follow a tool call");
                const auto value = text(b,"text",p);
                if (role == "assistant") answer += value;
                else parts.push_back({{"type","text"},{"text",value}});
            } else if (type == "image" && role == "user") {
                if (first_image.empty()) first_image = p;
                keys(b,p,{"type","source","cache_control"});
                if (!pending.empty()) throw request_error(p,"expected tool results first");
                const auto& src = member(b,"source",p,json::value_t::object);
                const auto sp = p+".source";
                if (text(src,"type",sp) != "base64") throw request_error(sp+".type","send an inline base64 image");
                keys(src,sp,{"type","media_type","data"});
                const auto mime = text(src,"media_type",sp);
                if (mime!="image/png" && mime!="image/jpeg" && mime!="image/webp" && mime!="image/gif")
                    throw request_error(sp+".media_type","unsupported image media type");
                parts.push_back({{"type","image_url"},{"image_url",{{"url","data:"+mime+";base64,"+text(src,"data",sp)}}}});
            } else throw request_error(p+".type","unsupported "+role+" block: "+type);
        }
        if (role == "assistant") {
            json a = {{"role",role},{"content",answer}};
            if (!reasoning.empty()) a["reasoning_content"] = std::move(reasoning);
            if (!calls.empty()) a["tool_calls"] = std::move(calls);
            out.push_back(std::move(a));
        } else {
            if (!pending.empty()) throw request_error(at+".content","missing tool results");
            if (!parts.empty()) {
                if (parts.size()==1 && parts[0]["type"]=="text") out.push_back({{"role",role},{"content",parts[0]["text"]}});
                else out.push_back({{"role",role},{"content",std::move(parts)}});
            }
        }
    }
    if (!pending.empty()) throw request_error("messages","missing tool results");
    if (auto e = flm::chat::order_parallel_tool_results(out)) throw request_error("messages",e->message);
    result["messages"] = std::move(out);
    if (req.contains("tools")) {
        const auto& tools = member(req,"tools","",json::value_t::array);
        result["tools"] = json::array();
        for (size_t i=0; i<tools.size(); ++i) {
            const auto at = at_index("tools",i);
            const auto& t = tools[i];
            if (t.contains("type") && t["type"] != "custom") throw request_error(at+".type","only client tools are supported");
            keys(t,at,{"type","name","description","input_schema","cache_control","eager_input_streaming"});
            if (t.contains("eager_input_streaming")) member(t,"eager_input_streaming",at,json::value_t::boolean);
            json fn = {{"name",text(t,"name",at)},{"parameters",member(t,"input_schema",at,json::value_t::object)}};
            if (t.contains("description")) fn["description"] = text(t,"description",at);
            result["tools"].push_back({{"type","function"},{"function",std::move(fn)}});
        }
    }
    return result;
}

struct generation_request { json chat; int thinking_budget = 0; bool show_local_thinking = true; std::string first_image; };
inline generation_request convert(const json& req) {
    keys(req,"",{"model","max_tokens","messages","system","tools","tool_choice","thinking","stream",
                 "temperature","top_p","top_k","stop_sequences","metadata","service_tier","cache_control"});
    generation_request out;
    out.chat = prompt_fields(req, out.first_image);
    auto& c = out.chat;
    c["model"] = text(req,"model","");
    if (c["model"]=="") throw request_error("model","expected a model name");
    const int total = integer(req,"max_tokens","",1);
    c["max_tokens"] = total;
    if (req.contains("stream")) c["stream"] = member(req,"stream","",json::value_t::boolean);
    for (const char* k : {"temperature","top_p"}) if (req.contains(k)) {
        if (!req[k].is_number() || req[k]<0 || req[k]>1) throw request_error(k,"expected a number from 0 to 1");
        c[k] = req[k];
    }
    if (req.contains("top_k")) c["top_k"] = integer(req,"top_k","",0);
    if (req.contains("stop_sequences") && !member(req,"stop_sequences","",json::value_t::array).empty())
        throw request_error("stop_sequences","matched stop sequences are not available");
    if (req.contains("tool_choice")) {
        const auto& choice = req["tool_choice"];
        const auto mode = text(choice,"type","tool_choice");
        if (mode == "tool") {
            keys(choice,"tool_choice",{"type","name","disable_parallel_tool_use"});
            c["tool_choice"] = {{"type","function"},{"function",{{"name",text(choice,"name","tool_choice")}}}};
        } else {
            keys(choice,"tool_choice",{"type","disable_parallel_tool_use"});
            if (mode!="auto" && mode!="any" && mode!="none") throw request_error("tool_choice.type","unsupported mode");
            c["tool_choice"] = mode=="any" ? "required" : mode;
        }
        if (choice.contains("disable_parallel_tool_use"))
            c["parallel_tool_calls"] = !member(choice,"disable_parallel_tool_use","tool_choice",json::value_t::boolean).get<bool>();
    }
    if (req.contains("thinking")) {
        const auto& t = req["thinking"];
        const auto mode = text(t,"type","thinking");
        if (mode=="enabled") {
            keys(t,"thinking",{"type","budget_tokens","display"});
            if (t.contains("display")) {
                const auto mode_display = text(t,"display","thinking");
                if (mode_display!="summarized" && mode_display!="omitted")
                    throw request_error("thinking.display","expected summarized or omitted");
                // Local models have no summarizer. Withhold raw reasoning rather than
                // mislabel it as a summary or replay fabricated text on the next turn.
                out.show_local_thinking = false;
            }
            out.thinking_budget = integer(t,"budget_tokens","thinking",1024);
            if (out.thinking_budget >= total) throw request_error("thinking.budget_tokens","must be below max_tokens");
            c["reasoning_effort"] = "medium";
        } else if (mode=="disabled") {
            keys(t,"thinking",{"type"});
            c["reasoning_effort"] = "none";
        } else throw request_error("thinking.type","expected enabled or disabled");
    }
    return out;
}
} // namespace flm::anthropic
