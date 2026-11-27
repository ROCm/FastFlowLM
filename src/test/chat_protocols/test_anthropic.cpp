#include "anthropic_request.hpp"
#include "protocol_anthropic.hpp"
#include <cstdio>

int test_anthropic() {
    using namespace flm::anthropic;
    int failures = 0;
    auto check = [&](bool ok, const char* label) {
        if (!ok) { std::printf("FAIL Messages: %s\n",label); ++failures; }
    };
    auto input = json::parse(R"({"model":"m","max_tokens":2048,"system":[{"type":"text","text":"guide"}],
      "thinking":{"type":"enabled","budget_tokens":1024},
      "tools":[{"name":"f","input_schema":{"type":"object"}}],
      "tool_choice":{"type":"any","disable_parallel_tool_use":true},
      "messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQ=="}}]},
        {"role":"assistant","content":[{"type":"thinking","thinking":"history","signature":""},
          {"type":"tool_use","id":"a","name":"f","input":{"x":1}},{"type":"tool_use","id":"b","name":"f","input":{"x":2}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"B"},{"type":"tool_result","tool_use_id":"a","content":"A"}]}]})");
    auto r = convert(input);
    const auto& msgs = r.chat.at("messages");
    check(r.thinking_budget==1024 && r.chat.at("max_tokens")==2048, "separate budgets");
    check(msgs[0]["content"]=="guide" && msgs[1]["content"][0]["image_url"]["url"]=="data:image/png;base64,AQ==","system and image");
    check(msgs[2]["reasoning_content"]=="history" && msgs[3]["tool_call_id"]=="a" && msgs[3]["content"]=="A" &&
          msgs[4]["tool_call_id"]=="b" && msgs[4]["content"]=="B","correlated reverse-order results");
    check(r.chat["tool_choice"]=="required" && r.chat["parallel_tool_calls"]==false,"tool choice");
    auto rejected = [&](json bad,const char* path) {
        try { (void)convert(bad); check(false,path); }
        catch (const request_error& e) { check(std::string(e.what()).find(path)!=std::string::npos,path); }
    };
    auto bad=input; bad["messages"][2]["content"][0]["tool_use_id"]="absent"; rejected(bad,"tool_use_id");
    bad=input; bad["messages"][0]["content"][0]["source"]["type"]="url"; rejected(bad,"source.type");
    bad=input; bad["messages"][1]["content"][0]["signature"]="foreign"; rejected(bad,"messages[1].content[0]");
    bad=input; bad["thinking"]["budget_tokens"]=1024.5; rejected(bad,"thinking.budget_tokens");
    bad=input; bad["max_tokens"]=1024; rejected(bad,"thinking.budget_tokens");
    bad=input; bad["tools"][0]["name"]=""; rejected(bad,"tools[0].name");
    auto nullable=input;
    for (const char* key : {"temperature","top_p","system","tool_choice","stream","thinking"}) nullable[key]=nullptr;
    check(convert(nullable).thinking_budget==0 && !convert(nullable).chat.contains("tool_choice"),"optional nulls are absent");

    std::vector<json> events;
    bool last=false;
    sink s([&](const json&){check(false,"unexpected HTTP body");},
           [&](const json& data,bool final) {
        check(!last,"event after terminal");
        auto wire=data.get<std::string>(); auto split=wire.find("\ndata: ");
        check(split!=std::string::npos,"named SSE");
        auto event=json::parse(wire.substr(split+7));
        check(wire.substr(0,split)=="event: "+event.at("type").get<std::string>(),"event name");
        events.push_back(std::move(event)); last=final;
    },true);
    s.on_begin({"model",true}); s.on_start({12,5,{}});
    chat_msg_delta d; d.reasoning="thinking"; s.on_delta(d);
    d={}; d.content="text"; s.on_delta(d);
    d={}; d.tool_index=0; d.tool_id="a"; d.tool_name="f"; d.tool_arguments="{\"x\":"; s.on_delta(d);
    d.tool_arguments="1}"; s.on_delta(d);
    d.tool_index=1; d.tool_id="b"; d.tool_arguments="{}"; s.on_delta(d);
    chat_msg m; chat_turn_end end; end.message=&m; end.finish=chat_finish::TOOL_CALLS;
    end.prompt_tokens=12; end.cached_prompt_tokens=5; end.generated_tokens=9;
    s.on_end(end); s.on_error({500,"server_error","late",{},std::nullopt});
    check(last && events.front()["type"]=="message_start" && events.back()["type"]=="message_stop","stream boundary");
    check(events.front()["message"]["usage"]["input_tokens"]==7 && events.front()["message"]["usage"]["cache_read_input_tokens"]==5,"cache split");
    int active=-1, next=0; std::string args; bool sig=false;
    for (const auto& e : events) {
        if (e["type"]=="content_block_start") { check(active<0 && e["index"]==next,"start index"); active=next++; }
        if (e["type"]=="content_block_delta") {
            check(e["index"]==active,"delta index");
            if (e["delta"]["type"]=="signature_delta") sig=true;
            if (e["delta"]["type"]=="input_json_delta" && active==2) args+=e["delta"]["partial_json"].get<std::string>();
        }
        if (e["type"]=="content_block_stop") { check(e["index"]==active,"stop index"); active=-1; }
    }
    check(next==4 && active<0 && sig && args=="{\"x\":1}" && events[events.size()-2]["delta"]["stop_reason"]=="tool_use","blocks and stop");
    auto proto=flm::error_model::protocol_for_path("/v1/messages/?k=v");
    auto abort=flm::error_model::stream_abort_frames(proto,{500,"server_error","failed"});
    check(abort && abort->starts_with("event: error\ndata: ") && abort->find("[DONE]")==std::string::npos,"transport abort dialect");
    std::vector<std::string> wire;
    sink interrupted([&](const json&){check(false,"HTTP error after stream began");},
        [&](const json& frame,bool){ wire.push_back(frame.get<std::string>()); },true);
    interrupted.on_begin({"m",true}); interrupted.on_start({});
    interrupted.on_error({500,"server_error","failed",{},std::nullopt});
    interrupted.on_end(end); interrupted.on_error({500,"server_error","repeated",{},std::nullopt});
    check(wire.size()==2 && wire.back()==*abort,"one error event, no later success");
    wire.clear();
    sink invalid_tool([&](const json&){check(false,"HTTP body after invalid streamed tool");},
        [&](const json& frame,bool){wire.push_back(frame.get<std::string>());},true);
    invalid_tool.on_begin({"m",true}); invalid_tool.on_start({});
    d={}; d.tool_index=0; d.tool_id="a"; d.tool_name="f"; d.tool_arguments="[]"; invalid_tool.on_delta(d);
    m.tool_calls.push_back({"a","f","[]"});
    invalid_tool.on_end(end);
    check(wire.back().starts_with("event: error\n") &&
          wire.back().find("message_stop")==std::string::npos,"invalid tool input cannot stream tool_use success");
    for (bool streaming : {false,true}) for (bool capped : {false,true}) {
        std::vector<std::string> order; json msg;
        sink ordered([&](const json& body){msg=body;},[&](const json& frame,bool){
            const auto w=frame.get<std::string>(); const auto e=json::parse(w.substr(w.find("\ndata: ")+7));
            if (e["type"]=="content_block_start") order.push_back(e["content_block"]["type"]);
            if (e["type"]=="message_delta") msg=e["delta"];
        },streaming);
        ordered.on_begin({"m",true}); ordered.on_start({});
        d={}; d.tool_index=0; d.tool_id="a"; d.tool_name="f"; d.tool_arguments="{}"; ordered.on_delta(d);
        d={}; d.content="after tool"; ordered.on_delta(d);
        if (capped) { d={}; d.tool_index=1; d.tool_id="b"; d.tool_name="f"; d.tool_arguments="{\"x\":"; ordered.on_delta(d); }
        auto e=end; e.finish=capped ? chat_finish::LENGTH : chat_finish::TOOL_CALLS; ordered.on_end(e);
        if (!streaming) for (auto& b : msg.at("content")) order.push_back(b["type"]);
        check(order==std::vector<std::string>{"tool_use","text"} &&
              msg.at("stop_reason")==(capped ? "max_tokens" : "tool_use"),"ordered and capped JSON/SSE blocks");
    }
    for (const char* mode : {"summarized","omitted"}) for (bool streaming : {false,true}) {
        auto req=input; req["thinking"]["display"]=mode;
        auto cfg=convert(req);
        std::string out;
        sink hidden([&](const json& body){out=body.dump();},
                    [&](const json& frame,bool){out+=frame.get<std::string>();},streaming,cfg.show_local_thinking);
        chat_msg done; done.reasoning="PRIVATE"; done.content="OK";
        chat_turn_end e; e.message=&done;
        hidden.on_begin({"m",streaming}); hidden.on_start({});
        {
            d={}; d.reasoning=done.reasoning; hidden.on_delta(d);
            d={}; d.content=done.content; hidden.on_delta(d);
        }
        hidden.on_end(e);
        check(out.find("PRIVATE")==std::string::npos && out.find("\"thinking\":\"\"")!=std::string::npos &&
              out.find("OK")!=std::string::npos, "display never leaks raw reasoning");
    }
    return failures;
}
