/// \file test_d3_compose.cpp
/// \brief Test of the composed-request port (src/common/decision/d3_compose, d3_composed) against upstream (no NPU).
///
///   test_d3_compose <fixtures dir> [<D3-Turbo-NPU2 package dir> | -] [<d3_contract fuzz.jsonl>]
///
/// The fixtures come from gen_compose_fixtures.py, which runs the d3-turbo package's own d3_compose.py and
/// d3_runtime.py (`_system_one_composed` with a stand-in encoder). The test requires byte-identical prompts and
/// parts, identical readout offsets, identical box decode / confidence / calibration (bit for bit), identical
/// per-question errors with their reasons, identical usage, and, from recorded hidden rows through a scripted CPU
/// engine, the same answers up to the float noise of the fp32 readout.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "decision/d3_compose.hpp"
#include "decision/d3_composed.hpp"
#include "decision/d3_engine_stub.hpp"
#include "decision/py_json.hpp"

#ifdef FLM_D3_TEST_TOKENIZER
#include "tokenizer/tokenizer.hpp"   // FastFlowLM's own Tokenizer (token ids only, no offsets) over tokenizers-cpp
#endif

using flm::d3::ContractError;
using flm::d3::ojson;
using flm::d3::PyValue;
namespace cp = flm::d3::compose;

namespace {

int failures = 0;
bool quiet = false;     // a request with combining marks under FastFlowLM's own tokenizer: mismatches are counted, not failures
int suppressed = 0;
std::map<std::string, int> checks_by, fails_by;

void fail(const std::string& group, const std::string& id, const std::string& what) {
    if (quiet) { suppressed++; return; }
    fails_by[group]++;
    static const int max_print = std::getenv("D3_TEST_MAXFAIL") ? std::atoi(std::getenv("D3_TEST_MAXFAIL")) : 60;
    if (failures++ < max_print) std::cerr << "FAIL [" << group << "] " << id << ": " << what.substr(0, 400) << "\n";
}
void count(const std::string& group) { checks_by[group]++; }

bool same(const ojson& a, const ojson& b, double tol, std::string& where) {
    if (a.is_number() && b.is_number()) {
        const double x = a.get<double>(), y = b.get<double>();
        if (x == y) return true;
        if (tol > 0 && std::fabs(x - y) <= tol * std::max(1.0, std::max(std::fabs(x), std::fabs(y)))) return true;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.17g != %.17g", x, y);
        where = buf;
        return false;
    }
    if (a.type() != b.type()) { where = "type " + a.dump() + " vs " + b.dump(); return false; }
    if (a.is_object()) {
        if (a.size() != b.size()) { where = "keys " + a.dump().substr(0, 200) + " vs " + b.dump().substr(0, 200); return false; }
        auto ia = a.begin();
        auto ib = b.begin();
        for (; ia != a.end(); ++ia, ++ib) {
            if (ia.key() != ib.key()) { where = "key order " + ia.key() + " vs " + ib.key(); return false; }
            if (!same(ia.value(), ib.value(), tol, where)) { where = ia.key() + "." + where; return false; }
        }
        return true;
    }
    if (a.is_array()) {
        if (a.size() != b.size()) { where = "length"; return false; }
        for (size_t i = 0; i < a.size(); i++)
            if (!same(a[i], b[i], tol, where)) { where = "[" + std::to_string(i) + "]" + where; return false; }
        return true;
    }
    if (a != b) { where = a.dump() + " vs " + b.dump(); return false; }
    return true;
}

std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string b64decode(const std::string& s) {
    static int table[256];
    static bool init = false;
    if (!init) {
        std::memset(table, -1, sizeof(table));
        const std::string a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < a.size(); i++) table[static_cast<unsigned char>(a[i])] = static_cast<int>(i);
        init = true;
    }
    std::string out;
    unsigned acc = 0;
    int bits = 0;
    for (unsigned char c : s) {
        if (table[c] < 0) continue;
        acc = (acc << 6) | static_cast<unsigned>(table[c]);
        bits += 6;
        if (bits >= 8) { bits -= 8; out += static_cast<char>((acc >> bits) & 0xFF); }
    }
    return out;
}

/// The edges of a decode record: base64 float32 (widened) or a list of doubles.
std::vector<std::vector<double>> edges_of(const ojson& rec) {
    std::vector<std::vector<double>> edges;
    if (rec.contains("edges_f32")) {
        const std::string raw = b64decode(rec["edges_f32"]);
        const size_t n = rec["n"];
        for (int d = 0; d < 4; d++) {
            std::vector<double> row(n);
            for (size_t j = 0; j < n; j++) {
                float f;
                std::memcpy(&f, raw.data() + (d * n + j) * 4, 4);
                row[j] = f;
            }
            edges.push_back(std::move(row));
        }
    } else {
        edges = rec["edges"].get<std::vector<std::vector<double>>>();
    }
    return edges;
}

std::vector<cp::Part> parts_of(const ojson& j) {
    std::vector<cp::Part> out;
    for (const auto& p : j) {
        const std::string k = p[0];
        if (k == "t") out.push_back(cp::text_part(p[1]));
        else if (k == "i") out.push_back(cp::image_part(p[1]));
        else if (k == "v") out.push_back(cp::video_part(p[1]));
        else out.push_back(cp::asset_part(p[1]));
    }
    return out;
}
ojson parts_json(const std::vector<cp::Part>& parts) {
    ojson out = ojson::array();
    for (const auto& p : parts) {
        using K = cp::Part::Kind;
        if (p.kind == K::Text) out.push_back({"t", p.text});
        else if (p.kind == K::Video) out.push_back({"v", p.index});
        else if (p.index >= 0) out.push_back({"i", p.index});
        else out.push_back({"a", p.text});
    }
    return out;
}

/// Runs `fn`; the outcome is "" (no error), the ContractError text, or "!<what>" for any other exception.
template <typename F>
std::string outcome(F&& fn) {
    try {
        fn();
    } catch (const ContractError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("!") + e.what();
    }
    return "";
}

/// Text with combining marks (Thai, Devanagari, U+0300 block): transformers' Qwen2Tokenizer rebuilds the pre-tokenizer
/// without \p{M}, so its ids differ from tokenizer.json's there (pre-existing for lite / nano / mini).
bool has_combining(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        const size_t w = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        uint32_t cp = w == 1 ? c : (c & (0xFF >> (w + 1)));
        for (size_t j = 1; j < w && i + j < s.size(); j++) cp = (cp << 6) | (s[i + j] & 0x3F);
        if ((cp >= 0x300 && cp <= 0x36F) || (cp >= 0x900 && cp <= 0x97F) || (cp >= 0xE00 && cp <= 0xE7F)) return true;
        i += w;
    }
    return false;
}

std::string error_expected(const ojson& rec) {
    // upstream's ValueError text, or "!" for another exception type (IndexError ...): not a ContractError here
    return rec["etype"] == "ValueError" ? rec["error"].get<std::string>() : std::string("!");
}
bool error_matches(const std::string& got, const ojson& rec) {
    const std::string want = error_expected(rec);
    return want == "!" ? (!got.empty() && got[0] == '!') : got == want;
}

}  // namespace

int main(int argc, char** argv) {
    std::function<std::vector<int>(const std::string&)> real_encode;   // FLM_D3_TEST_TOKENIZER + a package dir
    std::function<size_t(int)> real_bytes;
    int known_divergence = 0;
#ifdef FLM_D3_TEST_TOKENIZER
    std::unique_ptr<Tokenizer> real_tokenizer;
    if (argc > 2 && std::string(argv[2]) != "-") {
        real_tokenizer = std::make_unique<Tokenizer>(argv[2]);
        real_encode = [&](const std::string& s) { return real_tokenizer->encode(s); };
        real_bytes = [&](int id) { return real_tokenizer->token_byte_length(id); };
    }
#endif
    if (argc < 2) {
        std::cerr << "usage: test_d3_compose <fixtures dir> [<package dir>]\n";
        return 2;
    }
    const std::string dir = argv[1];
    const ojson config = ojson::parse(read_file(dir + "/decision_config.json"));
    const std::vector<std::string> codes = config["codes"].get<std::vector<std::string>>();
    cp::Calibration calibration;
    for (const auto& k : config["systemone"]["box_calibration"]) calibration.emplace_back(k[0].get<double>(), k[1].get<double>());

    // ---- unit fixtures -----------------------------------------------------------------------------------
    {
        std::ifstream f(dir + "/compose_units.jsonl");
        if (!f) { fail("setup", "compose_units.jsonl", "missing"); }
        std::string line;
        int n = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string kind = rec["kind"];
            const std::string id = kind + "#" + std::to_string(n++);
            count(kind);
            std::string where;
            try {
                if (kind == "check_references") {
                    const auto refs = rec["refs"].get<std::vector<std::string>>();
                    std::vector<std::string> got;
                    const std::string err = outcome([&] { got = cp::check_references(refs); });
                    if (rec.contains("error")) {
                        if (!error_matches(err, rec)) fail(kind, id, "error '" + err + "' vs '" + rec["error"].get<std::string>() + "'");
                    } else if (!err.empty() || got != rec["ok"].get<std::vector<std::string>>()) {
                        fail(kind, id, "refs " + err + " " + ojson(got).dump());
                    }
                } else if (kind == "split_string") {
                    const auto parts = cp::split_string(rec["text"], rec["refs"].get<std::vector<std::string>>());
                    if (!same(parts_json(parts), rec["parts"], 0, where)) fail(kind, id, where);
                } else if (kind == "describe_parts") {
                    const PyValue value = flm::d3::parse_json(rec["value"]);
                    std::vector<cp::Part> parts;
                    const std::string err = outcome([&] { parts = cp::describe_parts(value, rec["refs"].get<std::vector<std::string>>()); });
                    if (rec.contains("error")) {
                        if (!error_matches(err, rec)) fail(kind, id, "error '" + err + "' vs '" + rec["error"].get<std::string>() + "'");
                    } else if (!err.empty() || !same(parts_json(parts), rec["ok"], 0, where)) {
                        fail(kind, id, err + where);
                    }
                } else if (kind == "hoist_state") {
                    const PyValue state = flm::d3::parse_json(rec["state"]);
                    const auto h = cp::hoist_state(state, rec["refs"].get<std::vector<std::string>>(), rec["front"]);
                    if (flm::d3::dumps(h.state) != rec["hoisted"].get<std::string>() || h.uses != rec["uses"].get<std::vector<std::string>>()) {
                        fail(kind, id, flm::d3::dumps(h.state) + " vs " + rec["hoisted"].get<std::string>());
                    }
                } else if (kind == "user_parts") {
                    const PyValue state = flm::d3::parse_json(rec["state"]);
                    const PyValue question = flm::d3::parse_json(rec["question"]);
                    std::optional<double> at;
                    if (!rec["at"].is_null()) at = rec["at"].get<double>();
                    std::vector<cp::Part> parts;
                    std::string text;
                    const auto refs = rec["refs"].get<std::vector<std::string>>();
                    const std::string err = outcome([&] {
                        parts = cp::user_parts(state, question, codes, rec["n_front"], rec["n_videos"], refs, at);
                        text = cp::render_parts(parts, rec["suffix"]);
                    });
                    if (rec.contains("error")) {
                        if (!error_matches(err, rec)) fail(kind, id, "error '" + err + "' vs '" + rec["error"].get<std::string>() + "'");
                    } else if (!err.empty()) {
                        fail(kind, id, "unexpected " + err);
                    } else {
                        if (!same(parts_json(parts), rec["parts"], 0, where)) fail(kind, id, "parts " + where);
                        else if (text != rec["text"].get<std::string>()) fail(kind, id, "rendered text differs");
                    }
                } else if (kind == "suffix") {
                    const bool presence = rec["presence"];
                    if (cp::suffix(presence) != rec["text"].get<std::string>() || cp::markers(presence) != rec["markers"].get<std::vector<std::string>>())
                        fail(kind, id, "suffix");
                } else if (kind == "tail_text") {
                    std::optional<double> at;
                    if (!rec["at"].is_null()) at = rec["at"].get<double>();
                    if (cp::tail_text(codes, at, rec["presence"]) != rec["text"].get<std::string>()) fail(kind, id, "tail");
                } else if (kind == "format_seconds") {
                    if (cp::format_seconds(rec["value"]) != rec["text"].get<std::string>()) fail(kind, id, "%.1f " + cp::format_seconds(rec["value"]));
                } else if (kind == "round3") {
                    if (cp::py_round3(rec["value"]) != rec["rounded"].get<double>()) fail(kind, id, "round(x, 3)");
                } else if (kind == "points") {
                    const auto& pts = cp::r4_points();
                    const auto want = rec["points"].get<std::vector<std::vector<double>>>();
                    bool ok = pts.size() == want.size();
                    for (size_t i = 0; ok && i < pts.size(); i++)
                        for (int d = 0; d < 4; d++) ok = ok && pts[i][d] == want[i][d];
                    if (!ok) fail(kind, id, "r4 points differ");
                } else if (kind == "decode") {
                    const auto edges = edges_of(rec);
                    if (rec.contains("error")) {
                        const std::string err = outcome([&] { cp::decode(edges); });
                        if (err.empty()) fail(kind, id, "expected an error");
                    } else {
                        const cp::Box box = cp::decode(edges);
                        const ojson want = rec["box"];
                        if (box.top != want["top"].get<double>() || box.bottom != want["bottom"].get<double>() ||
                            box.left != want["left"].get<double>() || box.right != want["right"].get<double>()) {
                            fail(kind, id, "box " + ojson(cp::box_answer(box, 0)["box"]).dump() + " vs " + want.dump());
                        } else {
                            const double raw = cp::confidence(edges, box);
                            const ojson o = rec["other"];
                            const cp::Box other{o["top"], o["bottom"], o["left"], o["right"]};
                            if (raw != rec["raw"].get<double>()) fail(kind, id, "confidence " + std::to_string(raw) + " vs " + std::to_string(rec["raw"].get<double>()));
                            else if (cp::calibrate(raw, calibration) != rec["calibrated"].get<double>()) fail(kind, id, "calibrated");
                            else if (cp::confidence(edges, other) != rec["raw_other"].get<double>()) fail(kind, id, "confidence against another box");
                        }
                    }
                } else if (kind == "order_edges") {
                    const auto got = cp::order_edges(rec["a"], rec["b"]);
                    if (got.first != rec["out"][0].get<double>() || got.second != rec["out"][1].get<double>()) fail(kind, id, "order_edges");
                } else if (kind == "calibrate") {
                    cp::Calibration table;
                    if (!rec["table"].is_null())
                        for (const auto& k : rec["table"]) table.emplace_back(k[0].get<double>(), k[1].get<double>());
                    if (cp::calibrate(rec["value"], table) != rec["out"].get<double>()) fail(kind, id, "calibrate");
                } else if (kind == "visible") {
                    if (cp::visible_probability(rec["p"].get<std::vector<double>>()) != rec["out"].get<double>()) fail(kind, id, "visible");
                } else if (kind == "window_plan") {
                    const auto plan = cp::window_plan(rec["total"], rec["fps"], rec["rate"]);
                    if (plan != rec["windows"].get<std::vector<std::vector<int>>>()) {
                        fail(kind, id, "window_plan");
                    } else {
                        const auto want = rec["times"].get<std::vector<std::vector<double>>>();
                        for (size_t w = 0; w < plan.size(); w++)
                            if (cp::group_times(plan[w], rec["fps"]) != want[w]) fail(kind, id, "group_times");
                    }
                } else {
                    fail("setup", id, "unknown record kind");
                }
            } catch (const std::exception& e) {
                fail(kind, id, std::string("exception ") + e.what());
            }
        }
    }

    // ---- readout offsets ---------------------------------------------------------------------------------
    std::map<int, size_t> vocab_bytes;
    {
        const ojson v = ojson::parse(read_file(dir + "/vocab_bytes.json"));
        for (auto it = v.begin(); it != v.end(); ++it) vocab_bytes[std::stoi(it.key())] = it.value();
    }
    const auto token_bytes = [&](int id) -> size_t {
        if (id == 0x7FFF0000) return 1;   // the stand-in's one-byte token (see the request tests)
        auto it = vocab_bytes.find(id);
        if (it == vocab_bytes.end()) throw std::runtime_error("token id " + std::to_string(id) + " not in the fixture vocabulary");
        return it->second;
    };
    {
        std::ifstream f(dir + "/compose_offsets.jsonl");
        if (!f) fail("setup", "compose_offsets.jsonl", "missing");
        std::string line;
        int n = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = "offsets#" + std::to_string(n++);
            count("offsets");
            std::vector<int> got;
            std::vector<int> ids = rec["ids"].get<std::vector<int>>();
            std::function<size_t(int)> bytes = token_bytes;
            if (real_encode) {   // FastFlowLM's tokenizer and its byte lengths instead of the recorded ones
                const std::vector<int> mine = real_encode(rec["text"]);
                if (mine != ids) {
                    if (has_combining(rec["text"])) { known_divergence++; continue; }
                    fail("offsets", id, "FLM tokenizer ids differ from upstream's");
                    continue;
                }
                bytes = real_bytes;
            }
            const std::string err = outcome([&] { got = cp::readout_offsets(ids, bytes, rec["text"], rec["presence"]); });
            if (rec.contains("error")) {
                if (err != rec["error"].get<std::string>()) fail("offsets", id, "error '" + err + "' vs '" + rec["error"].get<std::string>() + "'");
            } else if (!err.empty() || got != rec["offsets"].get<std::vector<int>>()) {
                fail("offsets", id, err + " " + ojson(got).dump() + " vs " + rec["offsets"].dump());
            }
        }
        // synthetic tokenizations: the contract is "the colon is the end of a token" (A11)
        const auto synthetic = [&](const std::vector<std::string>& pieces, bool presence) -> std::string {
            std::string text;
            std::vector<int> ids;
            std::vector<size_t> lens;
            for (const auto& p : pieces) {
                ids.push_back(static_cast<int>(lens.size()));
                lens.push_back(p.size());
                text += p;
            }
            try {
                return ojson(cp::readout_offsets(ids, [&](int id) { return lens[id]; }, text, presence)).dump();
            } catch (const ContractError& e) {
                return e.what();
            }
        };
        struct Syn { std::vector<std::string> pieces; bool presence; std::string want; };
        const std::vector<Syn> syn = {
            // upstream's vocabulary: top : \n bottom : \n left : \n right :  -> 9, 6, 3, 0
            {{"abcd", "top", ":", "\n", "bottom", ":", "\n", "left", ":", "\n", "right", ":"}, false, "[9,6,3,0]"},
            {{"ab", "visible", ":", "\n", "top", ":", "\n", "bottom", ":", "\n", "left", ":", "\n", "right", ":"}, true, "[12,9,6,3,0]"},
            {{"ab", "visible", ":", "\n", "top", ":", "\n", "bottom", ":", "\n", "left", ":", "\n", "right", ":"}, false, "[9,6,3,0]"},   // the box markers end the track ones
            // one token per byte: distances in bytes
            {{"x", "t", "o", "p", ":", "\n", "b", "o", "t", "t", "o", "m", ":", "\n", "l", "e", "f", "t", ":", "\n", "r", "i", "g", "h", "t", ":"}, false, "[21,13,7,0]"},
            // a token that fuses a colon with the newline, or with the next word, hides that marker's end
            {{"abcd", "top", ":\n", "bottom", ":", "\n", "left", ":", "\n", "right", ":"}, false, "an edge marker's colon is not the end of a token"},
            {{"abcd", "top", ":", "\n", "bottom", ":", "\n", "left", ":", "\n", "right:"}, false, "[8,5,2,0]"},   // "right:" ends at a token end too
            {{"abcd", "top", ":", "\nbottom", ":", "\nleft", ":", "\nright", ":"}, false, "[6,4,2,0]"},
            {{"abcd", "top", ":\nbottom", ":", "\n", "left", ":", "\n", "right", ":"}, false, "an edge marker's colon is not the end of a token"},
            {{"a", "top:\nbottom:\nleft:\nright:x"}, false, "a Box prompt must end with its edge markers"},
        };
        for (size_t i = 0; i < syn.size(); i++) {
            count("offsets");
            const std::string got = synthetic(syn[i].pieces, syn[i].presence);
            const std::string want = syn[i].want;
            if (got != want) fail("offsets", "synthetic#" + std::to_string(i), "'" + got + "' vs '" + want + "'");
        }
        count("offsets");
        const auto plan = cp::readout_plan({12, 9, 6, 3, 0}, true);
        if (plan.counts != std::vector<int>{2, 128, 128, 128, 128}) fail("offsets", "plan", "counts");
    }

    // ---- composed requests -------------------------------------------------------------------------------
    int requests = 0, answered = 0, refused = 0, crashed = 0, rows_checked = 0, exact_checks = 0, encode_misses = 0, tolerant_flips = 0;
    double worst_prob = 0, worst_box = 0, worst_conf = 0;
    {
        const ojson rd = ojson::parse(read_file(dir + "/readout.json"));
        const int hidden = rd["hidden"];
        std::vector<float> readout;
        for (const auto& row : rd["weight"]) for (const auto& v : row) readout.push_back(v.get<float>());
        std::ifstream f(dir + "/compose_requests.jsonl");
        if (!f) fail("setup", "compose_requests.jsonl", "missing");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = rec["id"];
            requests++;
            std::map<std::string, std::vector<int>> toks;
            for (auto it = rec["tok"].begin(); it != rec["tok"].end(); ++it) toks[it.key()] = it.value().get<std::vector<int>>();
            toks[flm::d3::IMAGE_TOKEN] = {248056};
            flm::d3::ComposedHooks hooks;
            hooks.encode = [&](const std::string& s) {
                auto it = toks.find(s);
                if (it != toks.end()) return it->second;
                encode_misses++;
                return std::vector<int>(s.size(), 0x7FFF0000);   // the segment of a state: one one-byte token per byte (NFC text)
            };
            hooks.token_bytes = token_bytes;
            if (real_encode) {
                hooks.encode = real_encode;
                hooks.token_bytes = real_bytes;
            }
            flm::d3::ComposedConfig cfg;
            cfg.max_length = 32768;
            cfg.refuse_video_questions = false;   // upstream answers track / box-at questions itself (video_needed): compare that
            cfg.box_calibration = calibration;
            if (rec["refusal_model"]) {
                cfg.question_types = {"choice", "noul", "score"};
                cfg.image_placements = {"state"};
            }
            const PyValue body = flm::d3::parse_json(rec["body"]);
            count("request");
            flm::d3::ComposedRequest req;
            std::string request_error;
            try {
                req = flm::d3::prepare_composed(body, codes, cfg, hooks);
            } catch (const flm::d3::RequestError& e) {
                request_error = e.detail.is_string() ? e.detail.get<std::string>() : e.detail.dump();
            }
            if (rec.contains("detail")) {
                refused++;
                if (request_error != rec["detail"].get<std::string>()) fail("request", id, "request error '" + request_error + "' vs '" + rec["detail"].get<std::string>() + "'");
                continue;
            }
            if (!request_error.empty()) {
                // upstream crashed or answered: FLM must have answered too unless it is a 422 FLM defines itself
                fail("request", id, "unexpected request error: " + request_error);
                continue;
            }
            if (rec.contains("crash")) {
                crashed++;   // upstream raises (HTTP 500); FLM answers the question inline, nothing to compare
                continue;
            }
            answered++;
            // rows: one per fixture row, same prompt, same split, same readout plan
            // (rows are in request order on both sides; two questions may render the same text with different images)
            std::vector<const ojson*> fixture_rows;
            for (const auto& r : rec["rows"]) fixture_rows.push_back(&r);
            if (fixture_rows.size() != req.rows.size()) fail("rows", id, "row count " + std::to_string(req.rows.size()) + " vs " + std::to_string(fixture_rows.size()));
            const bool combining = real_encode && has_combining(rec["body"].get<std::string>());
            const int suppressed_before = suppressed;
            quiet = combining;
            std::vector<std::vector<std::vector<double>>> exact_values(req.rows.size());
            std::map<std::string, std::vector<float>> hidden_by_key;
            for (size_t i = 0; i < req.rows.size(); i++) {
                const auto& row = req.rows[i];
                count("rows");
                rows_checked++;
                if (i >= fixture_rows.size()) continue;
                const ojson& want = *fixture_rows[i];
                if (row.text != want["text"].get<std::string>()) { fail("rows", id + "/" + row.key, "prompt bytes differ"); continue; }
                uint32_t crc = 0;
                {   // zlib crc32 of the int64 ids
                    static uint32_t table[256];
                    static bool init = false;
                    if (!init) {
                        for (uint32_t n = 0; n < 256; n++) { uint32_t c = n; for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; table[n] = c; }
                        init = true;
                    }
                    crc = 0xFFFFFFFFu;
                    for (int tok : row.ids) {
                        const int64_t v = tok;
                        for (int b = 0; b < 8; b++) crc = table[(crc ^ static_cast<uint8_t>((v >> (8 * b)) & 0xFF)) & 0xFF] ^ (crc >> 8);
                    }
                    crc ^= 0xFFFFFFFFu;
                }
                if (static_cast<int>(row.ids.size()) != want["width"].get<int>() || crc != want["crc"].get<uint32_t>()) fail("rows", id + "/" + row.key, "token ids differ (width " + std::to_string(row.ids.size()) + " vs " + std::to_string(want["width"].get<int>()) + ")");
                else if (row.state_len != want["state"].get<int>() || static_cast<int>(row.ids.size()) - row.state_len != want["block"].get<int>()) fail("rows", id + "/" + row.key, "state split " + std::to_string(row.state_len) + " vs " + std::to_string(want["state"].get<int>()));
                else if (row.readout.offsets != want["offsets"].get<std::vector<int>>() || row.readout.counts != want["counts"].get<std::vector<int>>()) fail("rows", id + "/" + row.key, "readout plan");
                exact_values[i] = want["values"].get<std::vector<std::vector<double>>>();
                std::vector<float> h;
                for (const auto& r : want["hidden"]) for (const auto& v : r) h.push_back(v.get<float>());
                hidden_by_key[row.key] = std::move(h);
            }
            quiet = false;
            if (suppressed > suppressed_before) {   // known tokenizer divergence: not a port defect
                known_divergence++;
                continue;
            }
            // errors decided before the engine, and the exact path: upstream's own probabilities in, answers out
            {
                count("exact");
                exact_checks++;
                const ojson got = flm::d3::respond_composed(req, exact_values, cfg);
                std::string where;
                if (!same(got, rec["response"], 0, where)) fail("exact", id, where + "  got " + got.dump().substr(0, 300));
            }
            // the engine path: the same hidden rows through a scripted CPU engine, the fp32 readout on the C++ side
            {
                count("engine");
                flm::d3::ScriptedEngine engine(hidden, [&](const flm::d3::EngineStateRequest& call, size_t b) { return hidden_by_key.at(call.blocks[b].tag); });
                const ojson got = flm::d3::run_composed(req, engine, readout.data(), cfg, nullptr);
                std::string where;
                if (!same(got, rec["response"], 5e-4, where)) {
                    // exact structure, numbers off by more than readout noise
                    fail("engine", id, where);
                } else if (!same(got, rec["response"], 0, where)) {
                    tolerant_flips++;
                }
                // engine requests: one call, the state once, every block's readout positions
                count("engine");
                if (!req.rows.empty()) {
                    bool ok = engine.calls.size() == 1 && engine.calls[0].blocks.size() == req.rows.size();
                    for (size_t b = 0; ok && b < req.rows.size(); b++) {
                        const auto& blk = engine.calls[0].blocks[b];
                        ok = blk.readout_from_end == req.rows[b].readout.offsets &&
                             blk.ids.size() + engine.calls[0].state_ids.size() == req.rows[b].ids.size() &&
                             static_cast<int>(engine.calls[0].state_ids.size()) == req.rows[b].state_len &&
                             blk.image_ids.size() == req.rows[b].block_images.size();
                    }
                    if (!ok) fail("engine", id, "engine request shape");
                }
                // worst deviations (for the log)
                for (auto it = got["answers"].begin(); it != got["answers"].end(); ++it) {
                    const ojson& a = it.value();
                    const ojson& w = rec["response"]["answers"][it.key()];
                    if (a.contains("box") && w.contains("box")) {
                        for (const char* e : {"top", "bottom", "left", "right"}) worst_box = std::max(worst_box, std::fabs(a["box"][e].get<double>() - w["box"][e].get<double>()));
                        worst_conf = std::max(worst_conf, std::fabs(a["confidence"].get<double>() - w["confidence"].get<double>()));
                    } else if (a.contains("probabilities") && w.contains("probabilities")) {
                        for (auto p = a["probabilities"].begin(); p != a["probabilities"].end(); ++p) worst_prob = std::max(worst_prob, std::fabs(p.value().get<double>() - w["probabilities"][p.key()].get<double>()));
                    } else if (a.contains("noul") && w.contains("noul")) {
                        worst_prob = std::max(worst_prob, std::fabs(a["noul"].get<double>() - w["noul"].get<double>()));
                    }
                }
            }
        }
    }

    // ---- hand-written request-level cases (A1: request-level problems are 422, per-question ones inline) ----
    {
        flm::d3::ComposedHooks hooks;
        hooks.encode = [](const std::string& s) { return std::vector<int>(s.size(), 1); };
        hooks.token_bytes = [](int) -> size_t { return 1; };
        flm::d3::ComposedConfig cfg;
        cfg.box_calibration = calibration;
        const auto detail = [&](const std::string& body) -> std::string {
            try {
                flm::d3::prepare_composed(flm::d3::parse_json(body), codes, cfg, hooks);
            } catch (const flm::d3::RequestError& e) {
                return e.detail.is_string() ? e.detail.get<std::string>() : e.detail.dump();
            }
            return "";
        };
        const std::string tail = " by this build of d3:turbo (it answers choice, noul, score and box questions about text and images)";
        const std::string box = "\"questions\": {\"b\": {\"type\": \"box\", \"instructions\": \"x\"}}";
        struct Case { std::string body; std::string want; };
        const std::vector<Case> cases = {
            {"{" + box + ", \"long_context\": true}", "long_context is not supported" + tail},
            {"{" + box + ", \"long_context\": false}", ""},
            {"{" + box + ", \"assets\": [1]}", "assets takes an object of reference strings to base64 data URLs"},
            {"{" + box + ", \"assets\": []}", ""},
            {"{" + box + ", \"videos\": [\"data:video/mp4;base64,AAAA\"]}", "videos are not supported" + tail},
            {"{\"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\"}}}", "question 't': track questions are not supported" + tail},
            {"{\"questions\": {\"t\": {\"type\": \"box\", \"instructions\": \"x\", \"at\": 1.5}}}", "question 't': a box at a moment of a video is not supported" + tail},
            {"{" + box + ", \"zzz\": 1}", "'zzz' not understood; a request may carry state, questions, images, videos, assets, long_context, model"},
            {"{" + box + ", \"assets\": {\"\": \"data:image/png;base64,AAAA\"}}", "an asset reference must be a non-blank string"},
            {"{" + box + ", \"assets\": {\"x\": \"data:image/png;base64,AAAA\"}, \"state\": \"see x\"}", "images[0]: the image could not be identified (UnidentifiedImageError: cannot identify image file)"},
            {"{" + box + ", \"images\": [3], \"assets\": {\"x\": 1}}", "images[0]: strict mode: images must be given as a data: URL string, not a file path, web address or PIL image"},
            {"{\"questions\": {}}", "questions is required: an object of one or more question IDs, each to a question"},
        };
        for (const auto& c : cases) {
            count("request-level");
            std::string got = detail(c.body);
            // PIL's unidentified-image text carries an object address upstream; ours does not
            if (got != c.want) fail("request-level", c.body.substr(0, 80), "'" + got + "' vs '" + c.want + "'");
        }
        // routing: assets or a box / track question make a request composed
        count("routing");
        const auto composed = [](const std::string& b) { return flm::d3::is_composed(flm::d3::parse_json(b)); };
        if (composed("{\"questions\": {\"a\": {\"type\": \"noul\"}}}") || composed("{\"questions\": {\"a\": {\"type\": \"noul\"}}, \"assets\": {}}") ||
            !composed("{\"questions\": {\"a\": {\"type\": \"noul\"}}, \"assets\": {\"x\": 1}}") || !composed("{\"questions\": {\"a\": {\"type\": \"box\"}}}") ||
            !composed("{\"questions\": {\"a\": 5, \"t\": {\"type\": \"track\"}}}")) fail("routing", "is_composed", "dispatch");
        // refusal(): a checkpoint that declares no Box / image placements refuses per question, with a reason
        count("refusal");
        flm::d3::ComposedConfig bare = cfg;
        bare.question_types = {"choice", "noul", "score"};
        bare.image_placements = {"state"};
        const PyValue q = flm::d3::parse_json("{\"type\": \"choice\", \"instructions\": \"see <a>\", \"criteria\": {\"x\": null}}");
        const PyValue kind = PyValue::string("choice");
        const ojson r = flm::d3::refusal(&kind, q, {"<a>"}, bare);
        if (r.dump() != "{\"type\":\"choice\",\"error\":\"invalid_question\",\"reason\":\"question_images_unsupported\",\"message\":\"d3-turbo reads images in the state only, not in a question's instructions\"}")
            fail("refusal", "images", r.dump());
        const PyValue boxkind = PyValue::string("box");
        const ojson r2 = flm::d3::refusal(&boxkind, q, {}, bare);
        if (r2.dump() != "{\"type\":\"box\",\"error\":\"invalid_question\",\"reason\":\"box_unsupported\",\"message\":\"d3-turbo does not answer box questions\"}") fail("refusal", "box", r2.dump());
        // an engine without question-image support refuses a box whose own text holds an image (instead of failing the request)
        count("refusal");
        flm::d3::ComposedConfig noqi = cfg;
        noqi.box_question_images = false;
        const ojson r3 = flm::d3::refusal(&boxkind, q, {"<a>"}, noqi);
        if (r3.dump() != "{\"type\":\"box\",\"error\":\"invalid_question\",\"reason\":\"question_images_unsupported\",\"message\":\"d3-turbo reads images in the state only, not in a box question's text\"}")
            fail("refusal", "box with question images", r3.dump());
    }

    // ---- composed requests through the engine interface: duplicates, own images, flags, limits ----------------
    {
        // a self-contained tokenizer: one token per byte, `<|image_pad|>` one token (248056)
        const auto fake_encode = [](const std::string& s) {
            std::vector<int> ids;
            const std::string pad = flm::d3::IMAGE_TOKEN;
            for (size_t i = 0; i < s.size();) {
                if (s.compare(i, pad.size(), pad) == 0) { ids.push_back(248056); i += pad.size(); }
                else ids.push_back(static_cast<unsigned char>(s[i++]));
            }
            return ids;
        };
        flm::d3::ComposedHooks hooks;
        hooks.encode = fake_encode;
        hooks.token_bytes = [](int id) -> size_t { return id == 248056 ? 13 : 1; };
        flm::d3::ComposedConfig cfg;
        cfg.box_calibration = calibration;
        const std::string png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";
        const int hidden = 8;
        std::vector<float> readout(255 * hidden);
        for (size_t i = 0; i < readout.size(); i++) readout[i] = static_cast<float>(std::sin(0.37 * i));
        const auto engine_for = [&](flm::d3::ScriptedEngine*& out) {
            auto* e = new flm::d3::ScriptedEngine(hidden, [&](const flm::d3::EngineStateRequest& call, size_t b) {
                std::vector<float> rows;
                for (size_t r = 0; r < call.blocks[b].readout_from_end.size(); r++)
                    for (int d = 0; d < hidden; d++) rows.push_back(std::sin(0.11f * (b * 7 + r * 3 + d)) * 2.0f);
                return rows;
            });
            out = e;
            return std::unique_ptr<flm::d3::ScriptedEngine>(e);
        };
        const auto run = [&](const std::string& body, flm::d3::ComposedConfig c, flm::d3::ScriptedEngine*& engine_out, flm::d3::ComposedRequest* keep = nullptr) {
            const PyValue b = flm::d3::parse_json(body);
            flm::d3::ComposedRequest req = flm::d3::prepare_composed(b, codes, c, hooks);
            auto engine = engine_for(engine_out);
            const ojson r = flm::d3::run_composed(req, *engine, readout.data(), c, nullptr);
            if (keep) *keep = req;
            engine.release();   // the caller inspects engine_out (leaked on purpose in a test)
            return r;
        };
        // 1. duplicates: one asset used twice in the state is two pad runs (two images upstream), one table entry
        {
            count("interface");
            flm::d3::ScriptedEngine* eng = nullptr;
            flm::d3::ComposedRequest req;
            const ojson r = run("{\"state\": {\"k\": \"<a> and <a>\"}, \"assets\": {\"<a>\": \"" + png + "\"}, \"questions\": {"
                                "\"n\": {\"type\": \"noul\", \"instructions\": \"is it?\"}, \"b\": {\"type\": \"box\", \"instructions\": \"the <a>\"}}}", cfg, eng, &req);
            const auto& call = eng->calls.at(0);
            bool ok = eng->calls.size() == 1 && call.images.size() == 1 && call.state_image_ids == std::vector<int>{0, 0} &&
                      call.blocks.size() == 2 && call.blocks[0].image_ids.empty() && call.blocks[1].image_ids == std::vector<int>{0} &&
                      call.blocks[0].readout_from_end == std::vector<int>{0} && call.blocks[1].readout_from_end == std::vector<int>{21, 13, 7, 0} &&
                      call.images[0].grid_h == 16 && call.images[0].grid_w == 16 && call.image_token_id == 248056;
            int pads = 0;
            for (int t : call.state_ids) pads += t == 248056;
            ok = ok && pads == 2 * 64 && req.state.members[0].second.text == "[image 1] and [image 2]";
            if (!ok) fail("interface", "duplicates", r.dump().substr(0, 300));
            // usage: the state once + both blocks
            count("interface");
            const size_t S = req.rows[0].state_len;
            if (r["usage"]["input_tokens"].get<size_t>() != S + (req.rows[0].ids.size() - S) + (req.rows[1].ids.size() - S)) fail("interface", "usage", r["usage"].dump());
            count("interface");
            if (r["answers"]["b"]["type"] != "box" || !r["answers"]["b"]["box"].contains("top") || r["answers"]["n"]["type"] != "noul") fail("interface", "answers", r.dump());
            delete eng;
        }
        // 2. an option image shared by questions is one table entry; unused assets never load; flags
        {
            count("interface");
            flm::d3::ScriptedEngine* eng = nullptr;
            const std::string opt = "{\"type\": \"choice\", \"instructions\": \"which?\", \"criteria\": {\"x\": \"<o>\", \"y\": \"plain\"}}";
            const ojson r = run("{\"assets\": {\"<o>\": \"" + png + "\", \"<unused>\": \"not an image\"}, \"questions\": {\"q1\": " + opt + ", \"q2\": " + opt +
                                ", \"s\": {\"type\": \"score\", \"criteria\": [\"lo\", \"hi\"]}, \"bad\": {\"type\": \"nope\"}}}", cfg, eng);
            const auto& call = eng->calls.at(0);
            const bool ok = call.images.size() == 1 && call.blocks.size() == 2 && call.blocks[0].image_ids == std::vector<int>{0} &&
                            call.blocks[1].image_ids == std::vector<int>{0} && call.state_image_ids.empty();
            if (!ok) fail("interface", "shared option image", r.dump().substr(0, 300));
            count("interface");
            if (r["answers"]["s"].dump() != "{\"type\":\"score\",\"error\":\"invalid_question\",\"message\":\"score questions take no images here\",\"reason\":\"question_malformed\"}" ||
                r["answers"]["bad"]["reason"] != "question_malformed")
                fail("interface", "score flag (A2)", r["answers"].dump().substr(0, 300));
            delete eng;
            count("interface");
            flm::d3::ComposedConfig open = cfg;
            open.refuse_score = false;
            flm::d3::ScriptedEngine* eng2 = nullptr;
            const ojson r2 = run("{\"assets\": {\"<o>\": \"" + png + "\"}, \"questions\": {\"s\": {\"type\": \"score\", \"criteria\": [\"lo\", \"hi\"]}}}", open, eng2);
            if (r2["answers"]["s"]["type"] != "score" || !r2["answers"]["s"].contains("score")) fail("interface", "score flag off", r2.dump());
            delete eng2;
        }
        // 3. a bad own image is that question's inline error (index inside the question's own list); the others answer
        {
            count("interface");
            flm::d3::ScriptedEngine* eng = nullptr;
            const ojson r = run("{\"assets\": {\"<g>\": \"" + png + "\", \"<bad>\": \"data:image/gif;base64,AAAA\"}, \"questions\": {"
                                "\"ok\": {\"type\": \"noul\", \"instructions\": \"fine <g>\"}, \"nope\": {\"type\": \"box\", \"instructions\": \"<g> and <bad>\"}}}", cfg, eng);
            if (r["answers"]["ok"]["type"] != "noul" || r["answers"]["nope"].dump() !=
                    "{\"type\":\"box\",\"error\":\"invalid_question\",\"message\":\"images[1]: image data URLs take the types png, jpeg, jpg, webp, got 'gif'\",\"reason\":\"question_malformed\"}")
                fail("interface", "bad own image", r.dump().substr(0, 400));
            delete eng;
            // the same asset in the state is a request-level 422 with the hoisted index counted after the request's images
            count("interface");
            std::string err;
            try {
                flm::d3::ScriptedEngine* e2 = nullptr;
                run("{\"state\": \"see <bad>\", \"images\": [\"" + png + "\"], \"assets\": {\"<bad>\": \"data:image/gif;base64,AAAA\"}, \"questions\": {\"q\": {\"type\": \"box\", \"instructions\": \"x\"}}}", cfg, e2);
            } catch (const flm::d3::RequestError& e) {
                err = e.detail.get<std::string>();
            }
            if (err != "images[1]: image data URLs take the types png, jpeg, jpg, webp, got 'gif'") fail("interface", "hoisted index", err);
        }
        // 3b. temperature_by_questions (d3-ultra): choice / noul / score rows read the table at the request's number of such
        //     questions; box rows keep the base temperature
        {
            const std::string choice = "{\"type\": \"choice\", \"criteria\": {\"x\": \"one\", \"y\": \"two\", \"z\": null}}";
            const auto body = [&](int decisions) {
                std::string q = "{\"images\": [\"" + png + "\"], \"state\": \"s\", \"questions\": {\"b\": {\"type\": \"box\", \"instructions\": \"the thing\"}";
                for (int i = 0; i < decisions; i++) q += ", \"d" + std::to_string(i) + "\": " + (i % 2 ? "{\"type\": \"noul\"}" : choice);
                return q + "}}";
            };
            flm::d3::ComposedConfig table = cfg;
            table.temperature = 1.0f;
            table.temperature_by_questions = flm::d3::TemperatureTable{{2}, {3.0f, 0.5f}};
            const auto answers = [&](int decisions, flm::d3::ComposedConfig c) {
                flm::d3::ScriptedEngine* e = nullptr;
                const ojson r = run(body(decisions), c, e);
                delete e;
                return r["answers"];
            };
            const auto same_decisions = [&](const ojson& a, const ojson& b, int decisions) {
                for (int i = 0; i < decisions; i++) if (a["d" + std::to_string(i)] != b["d" + std::to_string(i)]) return false;
                return true;
            };
            flm::d3::ComposedConfig base = cfg, cold = cfg, warm = cfg;
            base.temperature = 1.0f;
            cold.temperature = 3.0f;
            warm.temperature = 0.5f;
            count("temperature");
            const ojson three = answers(3, table), two = answers(2, table);
            // 3 decisions: t_rest 0.5 for the decisions, base 1.0 for the box
            if (!same_decisions(three, answers(3, warm), 3) || three["b"] != answers(3, base)["b"]) fail("temperature", "3 decisions", three.dump().substr(0, 300));
            count("temperature");
            if (same_decisions(three, answers(3, base), 3)) fail("temperature", "the table changed nothing", "");
            // 2 decisions: the first bound (n <= 2) gives 3.0; the box still reads 1.0
            count("temperature");
            if (!same_decisions(two, answers(2, cold), 2) || two["b"] != answers(2, base)["b"]) fail("temperature", "2 decisions", two.dump().substr(0, 300));
            // an invalid question still counts (upstream counts every question of a decision type): 2 valid + 1 invalid = 3
            count("temperature");
            {
                std::string b3 = body(2);
                b3.insert(b3.size() - 2, ", \"bad\": {\"type\": \"choice\", \"criteria\": {}}");
                flm::d3::ScriptedEngine* e = nullptr;
                const ojson r = run(b3, table, e);
                delete e;
                if (!same_decisions(r["answers"], answers(2, warm), 2)) fail("temperature", "invalid question counted", r.dump().substr(0, 300));
            }
        }
        // 4. over the limit: the question's inline max_length_exceeded with its reason; the others answer
        {
            count("interface");
            flm::d3::ComposedConfig small = cfg;
            small.max_length = 700;
            flm::d3::ScriptedEngine* eng = nullptr;
            const ojson r = run("{\"state\": \"s\", \"questions\": {\"b\": {\"type\": \"box\", \"instructions\": \"" + std::string(400, 'x') + "\"}, \"n\": {\"type\": \"noul\"}}}", small, eng);
            const std::string want = "{\"type\":\"box\",\"error\":\"max_length_exceeded\",\"message\":\"the question prompt has";
            if (r["answers"]["b"].dump().compare(0, want.size(), want) != 0 || r["answers"]["b"]["reason"] != "question_tokens" ||
                r["answers"]["n"]["type"] != "noul" || eng->calls.at(0).blocks.size() != 1)
                fail("interface", "max length", r.dump().substr(0, 400));
            delete eng;
        }
        // 5. a track question or a box at a moment: 422 until M4 (the default), reasons per question otherwise
        {
            count("interface");
            const std::string body = "{\"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\"}}}";
            std::string err;
            try { flm::d3::ScriptedEngine* e = nullptr; run(body, cfg, e); } catch (const flm::d3::RequestError& e) { err = e.detail.get<std::string>(); }
            if (err.find("track questions are not supported by this build of d3:turbo") != 0 && err.find("question 't': track questions are not supported by this build of d3:turbo") != 0)
                fail("interface", "track 422", err);
            count("interface");
            flm::d3::ComposedConfig inline_cfg = cfg;
            inline_cfg.refuse_video_questions = false;
            flm::d3::ScriptedEngine* e2 = nullptr;
            const ojson r = run(body, inline_cfg, e2);
            if (r["answers"]["t"].dump() != "{\"type\":\"track\",\"error\":\"invalid_question\",\"message\":\"a track question needs a video\",\"reason\":\"video_needed\"}" || r["usage"]["input_tokens"] != 0)
                fail("interface", "track inline", r.dump());
            delete e2;
        }
    }

    // ---- robustness: the old fuzz corpus's bodies through the composed path must never throw anything but RequestError ----
    int fuzz_bodies = 0, fuzz_composed_ok = 0;
    if (argc > 3) {
        std::ifstream f(argv[3]);
        if (!f) fail("setup", argv[3], "missing");
        flm::d3::ComposedHooks hooks;
        hooks.encode = [](const std::string& s) { return std::vector<int>(s.begin(), s.end()); };
        hooks.token_bytes = [](int) -> size_t { return 1; };
        flm::d3::ComposedConfig cfg;
        cfg.refuse_video_questions = false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            PyValue body;
            try { body = flm::d3::parse_json(rec["body"].get<std::string>()); } catch (const flm::d3::JsonError&) { continue; }
            fuzz_bodies++;
            count("fuzz");
            try {
                const auto req = flm::d3::prepare_composed(body, codes, cfg, hooks);
                fuzz_composed_ok++;
                // answering needs only the shape of the rows
                std::vector<std::vector<std::vector<double>>> values(req.rows.size());
                for (size_t i = 0; i < req.rows.size(); i++) {
                    for (size_t p = 0; p < req.rows[i].readout.counts.size(); p++)
                        values[i].push_back(std::vector<double>(req.rows[i].readout.counts[p], 1.0 / req.rows[i].readout.counts[p]));
                }
                const ojson r = flm::d3::respond_composed(req, values, cfg);
                if (!r.contains("answers")) fail("fuzz", rec["id"], "no answers");
            } catch (const flm::d3::RequestError&) {
            } catch (const std::exception& e) {
                fail("fuzz", rec["id"], std::string("unexpected exception: ") + e.what());
            }
        }
    }

    int total = 0;
    for (const auto& c : checks_by) total += c.second;
    std::cout << "d3 compose: " << total << " checks, " << failures << " failures\n";
    for (const auto& c : checks_by) std::cout << "  " << c.first << ": " << c.second << " checks, " << fails_by[c.first] << " failures\n";
    std::cout << "requests: " << requests << " (" << answered << " answered, " << refused << " refused at request level, " << crashed
              << " upstream crashes), " << rows_checked << " rows, " << exact_checks << " exact response checks, " << tolerant_flips
              << " engine-path responses differing from upstream beyond bit equality (within 5e-4), " << encode_misses
              << " stand-in tokenizations (" << fuzz_bodies << " fuzz bodies, " << fuzz_composed_ok << " prepared), " << known_divergence << " skipped for the known combining-mark tokenizer divergence; worst |dprob| " << worst_prob << ", |dbox| " << worst_box << ", |dconfidence| " << worst_conf << "\n";
    return failures == 0 ? 0 : 1;
}
