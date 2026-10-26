/// \file d3_compose.cpp
/// \brief Port of upstream d3_compose.py (see d3_compose.hpp).
#include "decision/d3_compose.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

#include "decision/py_math.hpp"

// numpy rounds every operation (no a * b + c fusing): keep the compiler from contracting them (FMA targets)
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace flm::d3::compose {

const char* const DEFAULT_INSTRUCTIONS = "Choose the best matching option.";
const char* const REPLY = "Reply with only the code of the best option.";
const char* const HOISTED_LABEL_PREFIX = "[image ";

namespace {

// The private-use sentinels of `_mark` / `_unmark`: U+E000 (open), U+E001 (close), U+E002 (whole value)
const char* const OPEN = "\xEE\x80\x80";
const char* const CLOSE = "\xEE\x80\x81";
const char* const WHOLE = "\xEE\x80\x82";
const char* const NOUL_DESCRIPTIONS_[2] = {"No / false", "Yes / true"};

bool starts_at(const std::string& s, size_t at, const std::string& p) {
    return s.size() >= at + p.size() && s.compare(at, p.size(), p) == 0;
}

bool contains(const std::string& s, const char* p) { return s.find(p) != std::string::npos; }

std::u32string to_u32(const std::string& s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        const size_t w = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        char32_t cp = w == 1 ? c : (c & (0xFF >> (w + 1)));
        for (size_t j = 1; j < w && i + j < s.size(); j++) cp = (cp << 6) | (s[i + j] & 0x3F);
        out.push_back(cp);
        i += w;
    }
    return out;
}

void put_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

size_t code_points(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

/// `str.isspace()` of one code point (Python's definition: Unicode White_Space plus the C0 separators)
bool py_isspace(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/// Python's `int(text)` for the digits `_unmark` cuts out; ValueError text for anything else.
long long py_int(const std::string& s) {
    const std::u32string t = to_u32(s);
    size_t a = 0, b = t.size();
    while (a < b && py_isspace(t[a])) a++;
    while (b > a && py_isspace(t[b - 1])) b--;
    bool neg = false;
    size_t i = a;
    if (i < b && (t[i] == '+' || t[i] == '-')) neg = t[i++] == '-';
    long long v = 0;
    bool digit = false, last_underscore = false, ok = i < b;
    for (; i < b && ok; i++) {
        if (t[i] >= '0' && t[i] <= '9') {
            v = v * 10 + (t[i] - '0');
            digit = true;
            last_underscore = false;
            if (v > (1LL << 60)) ok = false;   // far beyond any index
        } else if (t[i] == '_' && digit && !last_underscore) {
            last_underscore = true;
        } else {
            ok = false;
        }
    }
    if (!ok || !digit || last_underscore) {
        throw ContractError("invalid literal for int() with base 10: " + repr(PyValue::string(s)));
    }
    return neg ? -v : v;
}

// ---- _mark / _unmark -----------------------------------------------------------------------------------------

PyValue mark(const PyValue& value, const std::vector<std::string>& refs, std::vector<std::string>& order) {
    if (value.is_string()) {
        const std::vector<Part> parts = split_string(value.text, refs);
        const bool any = std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind == Part::Kind::Image; });
        if (!any) return value;
        if (parts.size() == 1) {
            order.push_back(parts[0].text);
            return PyValue::string(std::string(WHOLE) + std::to_string(order.size() - 1) + WHOLE);
        }
        std::string out;
        for (const Part& p : parts) {
            if (p.kind == Part::Kind::Text) {
                out += p.text;
            } else {
                order.push_back(p.text);
                out += std::string(OPEN) + std::to_string(order.size() - 1) + CLOSE;
            }
        }
        return PyValue::string(std::move(out));
    }
    if (value.is_object()) {
        PyValue out = PyValue::object();
        for (const auto& m : value.members) out.members.emplace_back(m.first, mark(m.second, refs, order));
        return out;
    }
    if (value.is_array()) {
        PyValue out = PyValue::array();
        for (const auto& item : value.items) out.items.push_back(mark(item, refs, order));
        return out;
    }
    return value;
}

std::vector<Part> unmark(const std::string& utf8, const std::vector<std::string>& order) {
    const std::u32string text = to_u32(utf8);
    constexpr char32_t OPEN_C = 0xE000, CLOSE_C = 0xE001, WHOLE_C = 0xE002;
    constexpr char32_t EMPTIED = 0xFFFFFFFF;   // `buffer[-1] = ""` leaves an element that joins to nothing
    std::vector<Part> parts;
    std::vector<char32_t> buffer;
    const auto flush = [&]() {
        std::string s;
        for (char32_t c : buffer) {
            if (c != EMPTIED) put_utf8(s, c);
        }
        parts.push_back(text_part(std::move(s)));
        buffer.clear();
    };
    size_t at = 0;
    while (at < text.size()) {
        const char32_t c = text[at];
        if (c == OPEN_C || c == WHOLE_C) {
            const char32_t close = c == OPEN_C ? CLOSE_C : WHOLE_C;
            size_t end = at + 1;
            while (end < text.size() && text[end] != close) end++;
            if (end >= text.size()) throw ContractError("substring not found");
            std::string digits;
            for (size_t i = at + 1; i < end; i++) put_utf8(digits, text[i]);
            const long long index = py_int(digits);
            at = end + 1;
            if (c == WHOLE_C && !buffer.empty() && buffer.back() == U'"' && at < text.size() && text[at] == U'"') {
                buffer.back() = EMPTIED;
                at += 1;
            }
            if (!buffer.empty()) flush();
            const long long n = static_cast<long long>(order.size());
            const long long at_index = index < 0 ? index + n : index;
            if (at_index < 0 || at_index >= n) throw std::runtime_error("list index out of range");   // IndexError
            parts.push_back(asset_part(order[static_cast<size_t>(at_index)]));
            continue;
        }
        buffer.push_back(c);
        at++;
    }
    if (!buffer.empty()) flush();
    return parts;
}

}  // namespace

Part text_part(std::string s) {
    Part p;
    p.kind = Part::Kind::Text;
    p.text = std::move(s);
    return p;
}
Part image_part(int index) {
    Part p;
    p.kind = Part::Kind::Image;
    p.index = index;
    return p;
}
Part asset_part(std::string ref) {
    Part p;
    p.kind = Part::Kind::Image;
    p.text = std::move(ref);
    return p;
}
Part video_part(int index) {
    Part p;
    p.kind = Part::Kind::Video;
    p.index = index;
    return p;
}

std::vector<std::string> check_references(const std::vector<std::string>& references) {
    size_t total = 0;
    for (const std::string& ref : references) {
        const std::u32string cps = to_u32(ref);
        const bool blank = std::all_of(cps.begin(), cps.end(), [](char32_t c) { return py_isspace(c); });
        if (blank) throw ContractError("an asset reference must be a non-blank string");
        const size_t size = ref.size();
        if (size > static_cast<size_t>(MAX_REFERENCE_BYTES)) {
            throw ContractError("an asset reference has " + std::to_string(size) + " bytes, over " +
                                std::to_string(MAX_REFERENCE_BYTES));
        }
        if (contains(ref, OPEN) || contains(ref, CLOSE) || contains(ref, WHOLE)) {
            throw ContractError("an asset reference contains a reserved character");
        }
        total += size;
    }
    if (total > static_cast<size_t>(MAX_REFERENCES_BYTES)) {
        throw ContractError("asset references have " + std::to_string(total) + " bytes, over " +
                            std::to_string(MAX_REFERENCES_BYTES));
    }
    std::vector<std::pair<size_t, const std::string*>> keyed;
    for (const std::string& ref : references) keyed.emplace_back(code_points(ref), &ref);
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return *a.second < *b.second;   // UTF-8 byte order is code-point order
    });
    std::vector<std::string> out;
    for (const auto& k : keyed) out.push_back(*k.second);
    return out;
}

std::vector<Part> split_string(const std::string& text, const std::vector<std::string>& refs) {
    if (std::find(refs.begin(), refs.end(), text) != refs.end()) return {asset_part(text)};
    std::vector<Part> parts;
    size_t at = 0, start = 0;
    while (at < text.size()) {
        const std::string* hit = nullptr;
        for (const std::string& ref : refs) {
            if (starts_at(text, at, ref)) {
                hit = &ref;
                break;
            }
        }
        if (!hit) {
            at += 1;   // a reference starts with a UTF-8 lead byte, so a continuation byte never matches
            continue;
        }
        if (at > start) parts.push_back(text_part(text.substr(start, at - start)));
        parts.push_back(asset_part(*hit));
        at += hit->size();
        start = at;
    }
    if (start < text.size()) parts.push_back(text_part(text.substr(start)));
    return parts;
}

std::vector<Part> describe_parts(const PyValue& value, const std::vector<std::string>& refs) {
    if (value.is_string()) return split_string(value.text, refs);
    std::vector<std::string> order;
    const PyValue marked = mark(value, refs, order);
    if (order.empty()) return {text_part(describe(value))};
    return unmark(dumps(marked), order);
}

Hoisted hoist_state(const PyValue& state, const std::vector<std::string>& refs, int front) {
    Hoisted h;
    std::function<PyValue(const PyValue&)> walk = [&](const PyValue& value) -> PyValue {
        if (value.is_string()) {
            const std::vector<Part> parts = split_string(value.text, refs);
            const bool any = std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind == Part::Kind::Image; });
            if (!any) return value;
            std::string out;
            for (const Part& p : parts) {
                if (p.kind == Part::Kind::Text) {
                    out += p.text;
                } else {
                    h.uses.push_back(p.text);
                    out += std::string(HOISTED_LABEL_PREFIX) + std::to_string(front + static_cast<int>(h.uses.size())) + "]";
                }
            }
            return PyValue::string(std::move(out));
        }
        if (value.is_object()) {
            PyValue out = PyValue::object();
            for (const auto& m : value.members) out.members.emplace_back(m.first, walk(m.second));
            return out;
        }
        if (value.is_array()) {
            PyValue out = PyValue::array();
            for (const auto& item : value.items) out.items.push_back(walk(item));
            return out;
        }
        return value;
    };
    h.state = walk(state);
    return h;
}

std::vector<Part> merge_parts(const std::vector<Part>& parts) {
    std::vector<Part> out;
    for (const Part& p : parts) {
        if (p.kind == Part::Kind::Text) {
            if (p.text.empty()) continue;
            if (!out.empty() && out.back().kind == Part::Kind::Text) {
                out.back().text += p.text;
                continue;
            }
        }
        out.push_back(p);
    }
    return out;
}

std::string format_seconds(double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f", value);
    return buf;
}

std::vector<std::string> markers(bool presence) {
    std::vector<std::string> out;
    if (presence) out.push_back("visible");
    for (const char* e : {"top", "bottom", "left", "right"}) out.push_back(e);
    return out;
}

std::string suffix(bool presence) {
    std::string out;
    for (const std::string& name : markers(presence)) {
        if (!out.empty()) out += "\n";
        out += name + ":";
    }
    return out;
}

std::string tail_text(const std::vector<std::string>& codes, std::optional<double> at, bool presence) {
    const std::string when = at ? " in the video frame at " + format_seconds(*at) + " seconds" : std::string();
    std::string out = "\n\nBox:\nFind the target" + when + ". Give its top, bottom, left and right edges, each as a code on a " +
                      std::to_string(BINS) + "-step scale across the image: " + codes.at(0) +
                      " is the top or left side, " + codes.at(BINS - 1) + " the bottom or right side.";
    if (presence) {
        out += "\nAlso say whether the target is visible then: " + codes.at(0) + " no, " + codes.at(1) + " yes.";
    }
    out += "\n\nReply with the edge codes.";
    return out;
}

std::vector<Part> user_parts(const PyValue& state, const PyValue& question, const std::vector<std::string>& codes,
                             int n_front, int n_videos, const std::vector<std::string>& refs, std::optional<double> at) {
    const bool empty_state = state.is_null() || (state.is_string() && state.text.empty());
    const std::string body = empty_state ? std::string("(empty)") : describe(state);
    std::vector<Part> parts;
    for (int i = 0; i < n_front; i++) parts.push_back(image_part(i));
    for (int i = 0; i < n_videos; i++) parts.push_back(video_part(i));
    parts.push_back(text_part(std::string("State:\n") + body));
    parts.push_back(text_part("\n\nQuestion:\n"));
    const PyValue* kind_value = question.get("type");
    const std::string kind = kind_value && kind_value->is_string() ? kind_value->text : std::string();
    const PyValue* instructions = question.get("instructions");
    const auto add = [&](const std::vector<Part>& more) { parts.insert(parts.end(), more.begin(), more.end()); };
    if (instructions && instructions->truthy()) {
        add(describe_parts(*instructions, refs));
    } else {
        add(describe_parts(PyValue::string(DEFAULT_INSTRUCTIONS), refs));
    }
    if (kind == "box" || kind == "track") {
        parts.push_back(text_part(tail_text(codes, at, kind == "track")));
        return merge_parts(parts);
    }
    std::vector<std::vector<Part>> options;
    const PyValue* criteria = question.get("criteria");
    if (kind == "choice") {
        if (!criteria || !criteria->is_object()) throw ContractError("choice criteria must map option keys to descriptions");
        for (const auto& m : criteria->members) {
            std::vector<Part> option;
            if (m.second.is_null()) {
                option.push_back(text_part(m.first));
            } else {
                option.push_back(text_part(m.first + ": "));
                const auto more = describe_parts(m.second, refs);
                option.insert(option.end(), more.begin(), more.end());
            }
            options.push_back(std::move(option));
        }
    } else {
        const bool given = criteria && criteria->truthy() && criteria->is_object();
        for (int i = 0; i < 2; i++) {
            const PyValue* d = given ? criteria->get(i == 0 ? "false" : "true") : nullptr;
            if (d && d->truthy()) {
                options.push_back(describe_parts(*d, refs));
            } else {
                options.push_back(describe_parts(PyValue::string(NOUL_DESCRIPTIONS_[i]), refs));
            }
        }
    }
    parts.push_back(text_part("\n\nOptions:"));
    for (size_t i = 0; i < options.size() && i < codes.size(); i++) {
        parts.push_back(text_part("\n" + codes[i] + ": "));
        parts.insert(parts.end(), options[i].begin(), options[i].end());
    }
    parts.push_back(text_part(std::string("\n\n") + REPLY));
    return merge_parts(parts);
}

std::string render_parts(const std::vector<Part>& parts, const std::string& box_suffix) {
    std::string user;
    for (const Part& p : parts) {
        switch (p.kind) {
            case Part::Kind::Text: user += p.text; break;
            case Part::Kind::Image: user += std::string("<|vision_start|>") + IMAGE_TOKEN + "<|vision_end|>"; break;
            case Part::Kind::Video: user += std::string("<|vision_start|>") + VIDEO_TOKEN + "<|vision_end|>"; break;
        }
    }
    return chat_text(SYSTEM_PROMPT, user) + box_suffix;
}

std::vector<int> readout_offsets(const std::vector<int>& ids, const std::function<size_t(int)>& token_bytes,
                                 const std::string& text, bool presence) {
    const std::string tail = suffix(presence);
    if (text.size() < tail.size() || text.compare(text.size() - tail.size(), tail.size(), tail) != 0) {
        throw ContractError("a Box prompt must end with its edge markers");
    }
    // distance of each marker's colon from the end of the text, in bytes (the tail is ASCII)
    std::vector<size_t> distance;
    size_t at = 0;
    for (const std::string& name : markers(presence)) {
        at = tail.find(name + ":", at) + name.size() + 1;
        distance.push_back(tail.size() - at);
    }
    // a token ends at `distance` bytes before the end iff the tokens after it hold that many bytes
    const size_t furthest = *std::max_element(distance.begin(), distance.end());
    std::vector<int> found(distance.size(), -1);
    size_t cum = 0;
    for (size_t k = 0; k < ids.size() && cum <= furthest; k++) {
        for (size_t m = 0; m < distance.size(); m++) {
            if (found[m] < 0 && distance[m] == cum) found[m] = static_cast<int>(k);
        }
        cum += token_bytes(ids[ids.size() - 1 - k]);
    }
    for (int f : found) {
        if (f < 0) throw ContractError("an edge marker's colon is not the end of a token");
    }
    return found;
}

Readout readout_plan(const std::vector<int>& offsets, bool presence) {
    Readout r;
    r.offsets = offsets;
    for (size_t i = 0; i < offsets.size(); i++) r.counts.push_back(presence && i == 0 ? PRESENCE_BINS : BINS);
    return r;
}

// ---- the box readout -----------------------------------------------------------------------------------------

double expected(const std::vector<double>& probabilities) {
    const size_t count = probabilities.size();
    PySum total;
    for (double p : probabilities) total.add(p);
    PySum weighted;
    for (size_t k = 0; k < count; k++) {
        weighted.add(probabilities[k] * (static_cast<double>(k) + 0.5) / static_cast<double>(count));
    }
    if (total.value() == 0.0) throw std::runtime_error("float division by zero");
    return weighted.value() / total.value();
}

std::pair<double, double> order_edges(double a, double b, int bins) {
    if (a < b) return {a, b};
    const double mid = (a + b) / 2, half = 0.5 / bins;
    const auto pmax = [](double x, double y) { return y > x ? y : x; };   // Python's max(x, y)
    const auto pmin = [](double x, double y) { return y < x ? y : x; };   // Python's min(x, y)
    double lo = pmax(0.0, mid - half), hi = pmin(1.0, mid + half);
    if (lo >= hi) {
        if (hi >= 1.0) {
            lo = pmax(0.0, hi - 2 * half);
        } else {
            hi = lo + 2 * half;
        }
    }
    return {lo, hi};
}

Box decode(const std::vector<std::vector<double>>& edges) {
    if (edges.size() != 4) throw ContractError("a box takes four edge distributions");
    Box b;
    b.top = expected(edges[0]);
    b.bottom = expected(edges[1]);
    b.left = expected(edges[2]);
    b.right = expected(edges[3]);
    std::tie(b.top, b.bottom) = order_edges(b.top, b.bottom);
    std::tie(b.left, b.right) = order_edges(b.left, b.right);
    return b;
}

const std::vector<std::array<double, 4>>& r4_points() {
    static const std::vector<std::array<double, 4>> points = []() {
        double phi = 1.0;
        for (int i = 0; i < 64; i++) phi = std::pow(1 + phi, 1.0 / 5);
        double alphas[4];
        for (int d = 0; d < 4; d++) alphas[d] = py_mod1(std::pow(1 / phi, static_cast<double>(d + 1)));
        std::vector<std::array<double, 4>> out(CONFIDENCE_POINTS);
        for (int i = 0; i < CONFIDENCE_POINTS; i++) {
            for (int d = 0; d < 4; d++) out[i][d] = py_mod1(0.5 + alphas[d] * static_cast<double>(i + 1));
        }
        return out;
    }();
    return points;
}

double confidence(const std::vector<std::vector<double>>& edges, const Box& box) {
    if (edges.size() != 4) throw ContractError("a box takes four edge distributions");
    const size_t bins = edges[0].size();
    std::vector<std::vector<double>> cdf(4, std::vector<double>(bins));
    for (int d = 0; d < 4; d++) {
        if (edges[d].size() != bins) throw ContractError("the edge distributions differ in length");
        const double total = np_pairwise_sum(edges[d].data(), bins);   // probabilities.sum(axis=1)
        double run = 0.0;
        for (size_t j = 0; j < bins; j++) {
            run += edges[d][j] / total;   // np.cumsum: sequential
            cdf[d][j] = run;
        }
    }
    const auto& points = r4_points();
    const double reported = std::max(0.0, box.bottom - box.top) * std::max(0.0, box.right - box.left);
    long hits = 0;
    for (const auto& pt : points) {
        double draw[4];
        for (int d = 0; d < 4; d++) {
            size_t index = static_cast<size_t>(std::lower_bound(cdf[d].begin(), cdf[d].end(), pt[d]) - cdf[d].begin());
            if (index > bins - 1) index = bins - 1;
            draw[d] = (static_cast<double>(index) + 0.5) / static_cast<double>(bins);
        }
        const double top = draw[0], bottom = draw[1], left = draw[2], right = draw[3];
        if (!(top < bottom && left < right)) continue;
        const double inter_h = std::max(std::min(bottom, box.bottom) - std::max(top, box.top), 0.0);
        const double inter_w = std::max(std::min(right, box.right) - std::max(left, box.left), 0.0);
        const double inter = inter_h * inter_w;
        const double area = std::max(bottom - top, 0.0) * std::max(right - left, 0.0);
        const double uni = area + reported - inter;
        const double iou = uni > 0 ? inter / uni : 0.0;
        if (iou >= IOU_HIT) hits++;
    }
    return static_cast<double>(hits) / static_cast<double>(points.size());
}

double calibrate(double value, const Calibration& table) {
    if (table.empty()) return value;
    if (value <= table[0].first) return table[0].second;
    for (size_t i = 0; i + 1 < table.size(); i++) {
        const double x0 = table[i].first, y0 = table[i].second, x1 = table[i + 1].first, y1 = table[i + 1].second;
        if (value <= x1) return x1 == x0 ? y0 : y0 + (y1 - y0) * (value - x0) / (x1 - x0);
    }
    return table.back().second;
}

ojson box_answer(const Box& box, double confidence) {
    ojson b = ojson::object();
    b["top"] = box.top;
    b["bottom"] = box.bottom;
    b["left"] = box.left;
    b["right"] = box.right;
    ojson a = ojson::object();
    a["type"] = "box";
    a["box"] = b;
    a["confidence"] = confidence;
    return a;
}

double visible_probability(const std::vector<double>& two) {
    PySum total;
    for (double p : two) total.add(p);
    return two.at(1) / total.value();
}

double py_round3(double x) {
    char buf[400];
    std::snprintf(buf, sizeof(buf), "%.3f", x);
    return std::strtod(buf, nullptr);
}

std::vector<std::vector<int>> window_plan(int total, double fps, int rate) {
    const double step = fps / (2.0 * rate);
    const int count = std::max(1, static_cast<int>(static_cast<double>(total) / step));
    std::vector<int> indices;
    for (int k = 0; k < count; k++) {
        indices.push_back(std::min(total - 1, static_cast<int>(std::nearbyint(k * step))));
    }
    std::vector<std::vector<int>> windows;
    for (size_t i = 0; i < indices.size(); i += WINDOW_FRAMES) {
        windows.emplace_back(indices.begin() + i, indices.begin() + std::min(indices.size(), i + WINDOW_FRAMES));
    }
    for (auto& w : windows) {
        if (w.size() % 2) w.push_back(w.back());
        while (w.size() < 4) w.push_back(w.back());
    }
    return windows;
}

std::vector<double> group_times(const std::vector<int>& indices, double fps) {
    std::vector<double> seconds;
    for (int i : indices) seconds.push_back(i / fps);
    std::vector<double> out;
    for (size_t i = 0; i < seconds.size(); i += 2) {
        out.push_back((seconds.at(i) + seconds.at(i + 1)) / 2);
    }
    return out;
}

}  // namespace flm::d3::compose
