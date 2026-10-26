/// \file d3_composed.cpp
/// \brief Composed System One requests (see d3_composed.hpp): port of upstream `_system_one_composed`.
#include "decision/d3_composed.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <set>

namespace flm::d3 {

namespace {

using compose::Part;

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

/// `mentions`: whether a string inside `value` (strings, list items, object values; never keys) holds a reference.
bool mentions(const PyValue& value, const std::vector<std::string>& refs) {
    if (value.is_string()) {
        return std::any_of(refs.begin(), refs.end(), [&](const std::string& r) { return value.text.find(r) != std::string::npos; });
    }
    if (value.is_object()) {
        return std::any_of(value.members.begin(), value.members.end(), [&](const auto& m) { return mentions(m.second, refs); });
    }
    if (value.is_array()) {
        return std::any_of(value.items.begin(), value.items.end(), [&](const PyValue& v) { return mentions(v, refs); });
    }
    return false;
}

ojson type_json(const PyValue* kind) { return kind ? to_ojson(*kind) : ojson(nullptr); }

/// `{"type", "error", "message"}`: the shape of a caught ValueError; `with_reasons` appends the reason.
ojson caught(const ojson& type, const std::string& error, const std::string& message, const std::string& reason = std::string()) {
    ojson e = error_answer(type, error, message);
    if (!reason.empty()) e["reason"] = reason;
    return e;
}

std::string py_type_name(const PyValue& v) {
    switch (v.kind) {
        case PyValue::Kind::Null: return "NoneType";
        case PyValue::Kind::Bool: return "bool";
        case PyValue::Kind::Int: return "int";
        case PyValue::Kind::Float: return "float";
        case PyValue::Kind::String: return "str";
        case PyValue::Kind::Array: return "list";
        case PyValue::Kind::Object: return "dict";
    }
    return "object";
}

std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) b--;
    return s.substr(a, b - a);
}

/// Python's `float(value)` of a question's `at` (A3: the TypeError upstream does not catch is an invalid_question here).
double py_float(const PyValue& v) {
    switch (v.kind) {
        case PyValue::Kind::Bool: return v.b ? 1.0 : 0.0;
        case PyValue::Kind::Int: return std::strtod(v.text.c_str(), nullptr);
        case PyValue::Kind::Float: return v.f;
        case PyValue::Kind::String: {
            const std::string t = strip(v.text);
            const bool bad_chars = t.find_first_of("xXpP") != std::string::npos;
            char* end = nullptr;
            const double d = std::strtod(t.c_str(), &end);
            if (t.empty() || bad_chars || end != t.c_str() + t.size()) {
                throw ContractError("could not convert string to float: " + repr(v));
            }
            return d;
        }
        default:
            throw ContractError("float() argument must be a string or a real number, not '" + py_type_name(v) + "'");
    }
}

/// Python's `int(value)` of a track question's `rate`, clamped to the long long range (any big value is "not 1 or 2").
long long py_int(const PyValue& v) {
    const auto clamp = [](double d) { return d >= 9.2e18 ? LLONG_MAX : d <= -9.2e18 ? LLONG_MIN : static_cast<long long>(d); };
    switch (v.kind) {
        case PyValue::Kind::Bool: return v.b ? 1 : 0;
        case PyValue::Kind::Int: return clamp(std::strtod(v.text.c_str(), nullptr));
        case PyValue::Kind::Float:
            if (std::isnan(v.f)) throw ContractError("cannot convert float NaN to integer");
            if (std::isinf(v.f)) throw ContractError("cannot convert float infinity to integer");
            return clamp(std::trunc(v.f));
        case PyValue::Kind::String: {
            const std::string t = strip(v.text);
            size_t i = (!t.empty() && (t[0] == '+' || t[0] == '-')) ? 1 : 0;
            bool ok = i < t.size();
            for (size_t j = i; j < t.size(); j++) ok = ok && t[j] >= '0' && t[j] <= '9';
            if (!ok) throw ContractError("invalid literal for int() with base 10: " + repr(v));
            return clamp(std::strtod(t.c_str(), nullptr));
        }
        default:
            throw ContractError("int() argument must be a string, a bytes-like object or a real number, not '" +
                                py_type_name(v) + "'");
    }
}

std::string sorted_list_repr(std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    std::string out = "[";
    for (size_t i = 0; i < names.size(); i++) {
        if (i) out += ", ";
        out += repr(PyValue::string(names[i]));
    }
    return out + "]";
}

}  // namespace

std::vector<size_t> ComposedRequest::rows_of(const std::string& key) const {
    std::vector<size_t> out;
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].key == key) out.push_back(i);
    }
    return out;
}

bool is_composed(const PyValue& body) {
    const PyValue* assets = body.get("assets");
    if (assets && assets->truthy()) return true;
    const PyValue* questions = body.get("questions");
    if (!questions || !questions->is_object()) return false;
    for (const auto& m : questions->members) {
        const PyValue* kind = m.second.is_object() ? m.second.get("type") : nullptr;
        if (kind && kind->is_string() && (kind->text == "box" || kind->text == "track")) return true;
    }
    return false;
}

ojson refusal(const PyValue* kind, const PyValue& question, const std::vector<std::string>& refs, const ComposedConfig& config) {
    const std::string k = kind && kind->is_string() ? kind->text : std::string();
    if ((k == "box" || k == "track") && !contains(config.question_types, k)) {
        ojson e = ojson::object();
        e["type"] = type_json(kind);
        e["error"] = "invalid_question";
        e["reason"] = k + "_unsupported";
        e["message"] = config.model_name + " does not answer " + k + " questions";
        return e;
    }
    if (!question.is_object() || refs.empty()) return nullptr;
    if (k == "box" && !config.box_question_images && mentions(question, refs)) {
        ojson e = ojson::object();
        e["type"] = type_json(kind);
        e["error"] = "invalid_question";
        e["reason"] = "question_images_unsupported";
        e["message"] = config.model_name + " reads images in the state only, not in a box question's text";
        return e;
    }
    for (const char* place : {"instructions", "criteria"}) {
        const PyValue* value = question.get(place);
        if (!contains(config.image_placements, place) && value && mentions(*value, refs)) {
            ojson e = ojson::object();
            e["type"] = type_json(kind);
            e["error"] = "invalid_question";
            e["reason"] = "question_images_unsupported";
            e["message"] = config.model_name + " reads images in the state only, not in a question's " + place;
            return e;
        }
    }
    return nullptr;
}

namespace {

/// Everything one question's preparation reads and extends.
struct Context {
    const std::vector<std::string>& codes;
    const ComposedConfig& config;
    const ComposedHooks& hooks;
    const std::vector<std::string>& refs;
    const PyValue* assets = nullptr;
    ComposedRequest& request;
    std::map<std::string, int> asset_index;          //!< asset reference -> image table index
    std::map<std::string, std::string> asset_error;  //!< asset reference -> why it did not load
    std::map<std::string, size_t> segment_cache;
    // ---- videos (M4)
    const PyValue* videos = nullptr;                  //!< the body's `videos` list (when non-empty)
    int n_videos = 0;
    bool clips_loaded = false;
    std::vector<std::vector<int>> clip_ids;           //!< per request video: its expanded token ids (stamps, markers, pad runs)
    std::vector<std::vector<int>> clip_groups;        //!< per request video: image table index of each temporal group
    struct WindowSet {
        std::shared_ptr<std::vector<video::VideoWindow>> windows;
        std::vector<video::Moment> moments;           //!< every group of every window, in time order
        std::vector<std::vector<int>> entries;        //!< per window: image table index of each group
        std::vector<std::vector<int>> ids;            //!< per window: the expanded token ids of its video text
    };
    std::map<int, WindowSet> window_sets;             //!< by rate
    std::map<int, std::string> window_errors;         //!< rate -> why the clip could not be read (inline, per question)
    long long moment_rows = 0;                        //!< A5: rows of every box-at-a-moment / track question so far
    int pad = -1, video_pad = -1;
    int video_pad_id() {
        if (video_pad < 0) {
            const std::vector<int> one = hooks.encode(VIDEO_TOKEN);
            if (one.size() != 1) throw std::runtime_error("the tokenizer does not keep <|video_pad|> as one token");
            video_pad = one[0];
        }
        return video_pad;
    }
    int pad_id() {
        if (pad < 0) {
            const std::vector<int> one = hooks.encode(IMAGE_TOKEN);
            if (one.size() != 1) throw std::runtime_error("the tokenizer does not keep <|image_pad|> as one token");
            pad = one[0];
        }
        return pad;
    }

    /// One image into the table: header checks, the host's decoding, the processor's resize. Throws ContractError with
    /// the text of upstream's `load_image` (the caller adds "images[n]: ").
    int add_image(const std::string& source, const PyValue& value) {
        ImageFile f = inspect_image(value);
        if (hooks.decode) hooks.decode(request.images.size(), f);
        ComposedImage im;
        im.source = source;
        im.width = f.width;
        im.height = f.height;
        try {
            const auto hw = smart_resize(f.height, f.width);
            im.grid_h = hw.first / IMAGE_PATCH_SIZE;
            im.grid_w = hw.second / IMAGE_PATCH_SIZE;
            im.tokens = im.grid_h * im.grid_w / (IMAGE_MERGE_SIZE * IMAGE_MERGE_SIZE);
        } catch (const ContractError& e) {
            im.reject = e.what();   // the processor refuses it when a question's row is built
        }
        request.images.push_back(std::move(im));
        return static_cast<int>(request.images.size()) - 1;
    }

    int add_video_group(const std::string& source, int grid_h, int grid_w) {
        ComposedImage im;
        im.source = source;
        im.grid_h = grid_h;
        im.grid_w = grid_w;
        im.tokens = grid_h * grid_w / (IMAGE_MERGE_SIZE * IMAGE_MERGE_SIZE);
        request.images.push_back(std::move(im));
        return static_cast<int>(request.images.size()) - 1;
    }

    /// The expanded token ids of one video text: one pad per group is tokenized, then each pad becomes its group's run.
    std::vector<int> video_token_ids(const std::vector<double>& stamps, int tokens_per_group) {
        const std::string text = video::video_placeholder(stamps, tokens_per_group, /*expand=*/false);
        return video::expand_pad_runs(hooks.encode(text), video_pad_id(),
                                      std::vector<int>(stamps.size(), tokens_per_group));
    }

    /// The request's videos for plain questions (D3.load_videos + _prepare_videos): loaded when the first plain question
    /// needs them. Request-level problems (422) like the contract's plain path: "videos[i]: ...", the processor's refusal,
    /// the 16,384-token budget.
    void load_clips() {
        if (clips_loaded) return;
        if (!hooks.load_video) throw RequestError("videos are not supported by this build of " + config.tag);
        long long planned = 0;
        for (int i = 0; i < n_videos; i++) {
            std::shared_ptr<const video::VideoClip> clip;
            try {
                clip = hooks.load_video(static_cast<size_t>(i), videos->items[static_cast<size_t>(i)]);
            } catch (const video::VideoBackendUnavailable& e) {
                throw RequestError(e.what());
            } catch (const std::runtime_error& e) {
                throw RequestError("videos[" + std::to_string(i) + "]: " + e.what());
            }
            try {
                request.clip_grids.push_back(video::plan_video_grid(clip->n, clip->height, clip->width, config.video_config));
            } catch (const video::VideoError& e) {
                throw RequestError(std::string("video rejected by the processor: ") + e.what());
            }
            planned += request.clip_grids.back().tokens();
            request.clips.push_back(std::move(clip));
        }
        try {
            video::check_video_token_budget(planned);
        } catch (const video::VideoError& e) {
            throw RequestError(e.what());
        }
        for (int i = 0; i < n_videos; i++) {
            const video::VideoClip& clip = *request.clips[static_cast<size_t>(i)];
            const video::VideoGrid& g = request.clip_grids[static_cast<size_t>(i)];
            clip_ids.push_back(video_token_ids(video::group_timestamps(clip.indices, clip.fps), g.tokens_per_group()));
            clip_groups.emplace_back();
            for (int k = 0; k < g.grid_t; k++) {
                clip_groups.back().push_back(add_video_group("videos[" + std::to_string(i) + "] group " + std::to_string(k),
                                                             g.grid_h, g.grid_w));
                request.images.back().clip = i;
                request.images.back().group = k;
            }
        }
        clips_loaded = true;
    }

    /// `windows[rate] = load_video_windows(videos[0], rate)`, once per rate; a clip that cannot be read is the error of
    /// each question that asked (upstream's ValueError), a missing decoder is a 422.
    WindowSet& window_set(int rate) {
        auto have = window_sets.find(rate);
        if (have != window_sets.end()) return have->second;
        auto bad = window_errors.find(rate);
        if (bad != window_errors.end()) throw ContractError(bad->second);
        if (!hooks.load_windows) throw RequestError("videos are not supported by this build of " + config.tag);
        WindowSet ws;
        try {
            ws.windows = std::make_shared<std::vector<video::VideoWindow>>(hooks.load_windows(videos->items.at(0), rate));
            if (ws.windows->empty()) throw video::VideoError("no frame of the video could be decoded");
        } catch (const video::VideoBackendUnavailable& e) {
            throw RequestError(e.what());
        } catch (const std::runtime_error& e) {
            window_errors[rate] = e.what();
            throw ContractError(e.what());
        }
        for (size_t w = 0; w < ws.windows->size(); w++) {
            const video::VideoWindow& win = (*ws.windows)[w];
            for (size_t g = 0; g < win.times.size(); g++) ws.moments.push_back({static_cast<int>(w), static_cast<int>(g), win.times[g]});
            ws.ids.push_back(video_token_ids(video::group_timestamps(win.indices, win.fps), win.grid.tokens_per_group()));
            ws.entries.emplace_back();
            for (int g = 0; g < win.grid.grid_t; g++) {
                ws.entries.back().push_back(
                    add_video_group("video window " + std::to_string(rate) + ":" + std::to_string(w) + " group " + std::to_string(g),
                                    win.grid.grid_h, win.grid.grid_w));
                ComposedImage& im = request.images.back();
                im.rate = rate;
                im.window = static_cast<int>(w);
                im.group = g;
            }
        }
        request.windows[rate] = ws.windows;
        return window_sets.emplace(rate, std::move(ws)).first->second;
    }

    /// An asset used by one question: loaded once per request, failures remembered.
    int asset(const std::string& ref) {
        auto done = asset_index.find(ref);
        if (done != asset_index.end()) return done->second;
        auto bad = asset_error.find(ref);
        if (bad != asset_error.end()) throw ContractError(bad->second);
        try {
            const int idx = add_image(ref, *assets->get(ref));
            asset_index[ref] = idx;
            return idx;
        } catch (const ContractError& e) {
            asset_error[ref] = e.what();
            throw;
        }
    }
};

/// `_composed_question` + `_composed_row` up to the engine: the row of one image-path question.
ComposedRow compose_row(Context& cx, const std::string& key, const PyValue& question) {
    const PyValue* type = question.get("type");
    const std::string kind = type && type->is_string() ? type->text : std::string();
    ComposedRow row;
    row.key = key;
    row.type = type_json(type);
    row.kind = kind;
    if (kind == "box") {
        const PyValue* instructions = question.get("instructions");
        bool extra = false;
        for (const auto& m : question.members) extra = extra || (m.first != "type" && m.first != "instructions");
        if (extra || !instructions || !instructions->is_string()) throw ContractError("a box question takes type and instructions");
    } else {
        row.question = normalize_question(question);
        if (row.question.kind == "score" && cx.config.refuse_score) throw ContractError("score questions take no images here");
    }
    const int n_front = static_cast<int>(cx.request.front.size());
    if (cx.n_videos > 0) cx.load_clips();   // the request's videos are in the state of every plain row
    const std::vector<Part> parts = compose::user_parts(cx.request.state, question, cx.codes, n_front, cx.n_videos, cx.refs);
    const bool box = kind == "box";
    row.text = compose::render_parts(parts, box ? compose::suffix(false) : std::string());
    // the images of this question's own block, in reading order (a reference used twice is two images)
    std::vector<int> own;
    {
        int n = 0;
        for (const Part& p : parts) {
            if (p.kind != Part::Kind::Image || p.index >= 0) continue;
            try {
                own.push_back(cx.asset(p.text));
            } catch (const ContractError& e) {
                throw ContractError("images[" + std::to_string(n) + "]: " + e.what());
            }
            n++;
        }
    }
    const std::vector<int> plain = cx.hooks.encode(row.text);
    if (box) {
        row.readout = compose::readout_plan(compose::readout_offsets(plain, cx.hooks.token_bytes, row.text, false), false);
    } else {
        row.readout.offsets = {0};
        row.readout.counts = {static_cast<int>(row.question.keys.size())};
    }
    // the processor: every image (state images first, then this block's) must be accepted
    std::vector<int> in_order = cx.request.front;
    in_order.insert(in_order.end(), own.begin(), own.end());
    for (int idx : in_order) {
        if (!cx.request.images[idx].reject.empty()) throw ContractError(cx.request.images[idx].reject);
    }
    const auto count_of = [&](const char* token) {
        size_t n = 0;
        for (size_t at = row.text.find(token); at != std::string::npos; at = row.text.find(token, at + 1)) n++;
        return n;
    };
    if (count_of(IMAGE_TOKEN) != in_order.size() || count_of(VIDEO_TOKEN) != static_cast<size_t>(cx.n_videos)) {
        throw ContractError("the state or question contains a literal image or video placeholder token");
    }
    // each placeholder becomes its image's run of input tokens / its video's text, in order
    std::vector<int> expanded;
    expanded.reserve(plain.size());
    size_t next = 0, nvid = 0;
    const int pad = in_order.empty() ? -1 : cx.pad_id();
    const int video_pad = cx.n_videos > 0 ? cx.video_pad_id() : -1;
    for (int id : plain) {
        if (pad >= 0 && id == pad && next < in_order.size()) {
            expanded.insert(expanded.end(), cx.request.images[in_order[next]].tokens, pad);
            next++;
        } else if (video_pad >= 0 && id == video_pad && nvid < cx.clip_ids.size()) {
            expanded.insert(expanded.end(), cx.clip_ids[nvid].begin(), cx.clip_ids[nvid].end());
            nvid++;
        } else {
            expanded.push_back(id);
        }
    }
    if (next != in_order.size() || nvid != static_cast<size_t>(cx.n_videos)) {
        throw ContractError("the state or question contains a literal image or video placeholder token");
    }
    const int width = static_cast<int>(expanded.size());
    if (cx.config.max_length > 0 && width > cx.config.max_length) {
        throw MaxLengthExceeded("the question prompt has " + std::to_string(width) + " tokens, over the maximum context length of " +
                                std::to_string(cx.config.max_length) + " tokens; nothing was truncated");
    }
    int media = 0;
    for (int idx : cx.request.front) media += cx.request.images[idx].tokens - 1;
    for (const auto& ids : cx.clip_ids) media += static_cast<int>(ids.size()) - 1;
    row.state_len = state_block_length(row.text, cx.request.state, plain, cx.hooks.encode, cx.hooks.token_bytes, media, width,
                                       &cx.segment_cache);
    row.ids = std::move(expanded);
    row.state_images = cx.request.front;
    for (const auto& groups : cx.clip_groups) row.state_images.insert(row.state_images.end(), groups.begin(), groups.end());
    row.block_images = std::move(own);
    return row;
}

/// One moment of a box at a time / a track (`_video_box`'s loop body + `_composed_row`): the row `[window state | moment
/// question]`, the question carrying the moment's time (`at`), a track also asking whether the target is visible.
ComposedRow moment_row(Context& cx, const std::string& key, const PyValue& question, const std::string& kind,
                       const Context::WindowSet& ws, int rate, const video::Moment& m) {
    const bool track = kind == "track";
    ComposedRow row;
    row.key = key;
    row.type = type_json(question.get("type"));
    row.kind = kind;
    row.time = m.time;
    row.rate = rate;
    row.window = m.window;
    const int n_front = static_cast<int>(cx.request.front.size());
    const std::vector<Part> parts = compose::user_parts(cx.request.state, question, cx.codes, n_front, 1, cx.refs, m.time);
    row.text = compose::render_parts(parts, compose::suffix(track));
    for (const Part& p : parts) {
        if (p.kind == Part::Kind::Image && p.index < 0) throw ContractError("a video Box reads the video; images go in the state");
    }
    const std::vector<int> plain = cx.hooks.encode(row.text);
    row.readout = compose::readout_plan(compose::readout_offsets(plain, cx.hooks.token_bytes, row.text, track), track);
    for (int idx : cx.request.front) {
        if (!cx.request.images[idx].reject.empty()) throw ContractError(cx.request.images[idx].reject);
    }
    const auto count_of = [&](const char* token) {
        size_t n = 0;
        for (size_t at = row.text.find(token); at != std::string::npos; at = row.text.find(token, at + 1)) n++;
        return n;
    };
    if (count_of(IMAGE_TOKEN) != cx.request.front.size() || count_of(VIDEO_TOKEN) != 1) {
        throw ContractError("the state or question contains a literal image or video placeholder token");
    }
    const std::vector<int>& video_ids = ws.ids.at(static_cast<size_t>(m.window));
    std::vector<int> expanded;
    expanded.reserve(plain.size() + video_ids.size());
    size_t next = 0;
    const int pad = cx.request.front.empty() ? -1 : cx.pad_id();
    const int video_pad = cx.video_pad_id();
    bool video_done = false;
    for (int id : plain) {
        if (pad >= 0 && id == pad && next < cx.request.front.size()) {
            expanded.insert(expanded.end(), cx.request.images[cx.request.front[next]].tokens, pad);
            next++;
        } else if (id == video_pad && !video_done) {
            expanded.insert(expanded.end(), video_ids.begin(), video_ids.end());
            video_done = true;
        } else {
            expanded.push_back(id);
        }
    }
    if (next != cx.request.front.size() || !video_done) {
        throw ContractError("the state or question contains a literal image or video placeholder token");
    }
    const int width = static_cast<int>(expanded.size());
    if (cx.config.max_length > 0 && width > cx.config.max_length) {
        throw MaxLengthExceeded("the question prompt has " + std::to_string(width) + " tokens, over the maximum context length of " +
                                std::to_string(cx.config.max_length) + " tokens; nothing was truncated");
    }
    int media = static_cast<int>(video_ids.size()) - 1;
    for (int idx : cx.request.front) media += cx.request.images[idx].tokens - 1;
    row.state_len = state_block_length(row.text, cx.request.state, plain, cx.hooks.encode, cx.hooks.token_bytes, media, width,
                                       &cx.segment_cache);
    row.ids = std::move(expanded);
    row.state_images = cx.request.front;
    const auto& groups = ws.entries.at(static_cast<size_t>(m.window));
    row.state_images.insert(row.state_images.end(), groups.begin(), groups.end());
    row.state_key = "window:" + std::to_string(rate) + ":" + std::to_string(m.window);
    return row;
}

/// `_video_box`: the rows of a box at a time (the moment nearest `at`) or of a track (every moment, `rate` per second).
std::vector<ComposedRow> video_rows(Context& cx, const std::string& key, const PyValue& question, const std::string& kind) {
    std::vector<std::string> allowed = kind == "box" ? std::vector<std::string>{"type", "instructions", "at"}
                                                     : std::vector<std::string>{"type", "instructions", "rate"};
    bool extra = false;
    for (const auto& m : question.members) extra = extra || !contains(allowed, m.first);
    const PyValue* instructions = question.get("instructions");
    if (extra || !instructions || !instructions->is_string()) {
        throw ContractError("a " + kind + " question takes " + sorted_list_repr(allowed));
    }
    if (cx.n_videos == 0) throw QuestionRefused("video_needed", "a " + kind + " question needs a video");
    if (cx.n_videos > 1) {
        throw QuestionRefused("video_target_ambiguous", "a video " + kind + " needs exactly one video, got " + std::to_string(cx.n_videos));
    }
    long long rate = 1;
    if (kind == "track") {
        const PyValue* r = question.get("rate");
        if (r) rate = py_int(*r);
    }
    if (rate != 1 && rate != 2) throw ContractError("rate must be 1 or 2 updates per second");
    const Context::WindowSet& ws = cx.window_set(static_cast<int>(rate));
    std::vector<video::Moment> picked;
    if (kind == "box") {
        const double at = py_float(*question.get("at"));
        try {
            picked.push_back(video::nearest_moment(ws.moments, at));
        } catch (const video::VideoError& e) {
            throw ContractError(e.what());
        }
    } else {
        picked = ws.moments;
    }
    // A5: the budget of moment rows of the whole request (a request-level 422)
    cx.moment_rows += static_cast<long long>(picked.size());
    try {
        video::check_moment_rows(cx.moment_rows, cx.config.moment_row_limit);
    } catch (const video::VideoError& e) {
        throw RequestError(e.what());
    }
    std::vector<ComposedRow> rows;
    for (const video::Moment& m : picked) rows.push_back(moment_row(cx, key, question, kind, ws, static_cast<int>(rate), m));
    return rows;
}

}  // namespace

ComposedRequest prepare_composed(const PyValue& body, const std::vector<std::string>& codes, const ComposedConfig& config,
                                 const ComposedHooks& hooks) {
    PrepareOptions options;
    options.unsupported_for = config.tag;
    options.composed = true;
    options.reasons = true;
    options.answers = contains(config.question_types, "track") && config.video
                          ? "choice, noul, score, box and track questions about text, images and video"
                          : "choice, noul, score and box questions about text and images";
    options.refuse_video_questions = config.refuse_video_questions && contains(config.question_types, "track");
    const PyValue& questions = check_request(body, options);
    const std::string tail = " by this build of " + config.tag + " (it answers " + options.answers + ")";

    ComposedRequest req;
    const PyValue* assets = body.get("assets");
    const bool has_assets = assets && assets->truthy();
    if (has_assets && !assets->is_object()) {
        throw RequestError("assets takes an object of reference strings to base64 data URLs");
    }
    std::vector<std::string> refs;
    Context cx{codes, config, hooks, refs, has_assets ? assets : nullptr, req};

    // the request's images (strict, in order), then videos (not in this build)
    const PyValue* images = body.get("images");
    size_t n_images = 0;
    if (images && !images->is_null()) {
        if (!images->is_array()) throw RequestError("images takes a list of base64 data URLs");
        for (size_t i = 0; i < images->items.size(); i++) {
            try {
                req.front.push_back(cx.add_image("images[" + std::to_string(i) + "]", images->items[i]));
            } catch (const ContractError& e) {
                throw RequestError("images[" + std::to_string(i) + "]: " + e.what());
            }
        }
        n_images = images->items.size();
    }
    const PyValue* videos = body.get("videos");
    if (videos && !videos->is_null()) {
        if (!videos->is_array()) throw RequestError("videos takes a list of base64 data URLs");
        if (!videos->items.empty()) {
            if (!config.video || !hooks.load_video) throw RequestError("videos are not supported" + tail);
            cx.videos = videos;
            cx.n_videos = static_cast<int>(videos->items.size());
        }
    }
    for (const auto& m : questions.members) req.keys.push_back(m.first);
    const PyValue state = body.get("state") ? *body.get("state") : PyValue::null();
    try {
        check_json_value(state);
    } catch (const JsonError& e) {
        throw RequestError(std::string("not representable as JSON: ") + e.what());
    }

    // D3._system_one_composed: references, hoisting, the state's images
    if (has_assets) {
        std::vector<std::string> names;
        for (const auto& m : assets->members) names.push_back(m.first);
        try {
            refs = compose::check_references(names);
        } catch (const ContractError& e) {
            throw RequestError(e.what());
        }
    }
    compose::Hoisted hoisted = compose::hoist_state(state, refs, static_cast<int>(n_images));
    req.state = std::move(hoisted.state);
    for (size_t j = 0; j < hoisted.uses.size(); j++) {
        const size_t number = n_images + j;
        try {
            req.front.push_back(cx.asset(hoisted.uses[j]));
        } catch (const ContractError& e) {
            throw RequestError("images[" + std::to_string(number) + "]: " + e.what());
        }
    }

    for (const auto& m : questions.members) {
        const std::string& key = m.first;
        const PyValue& question = m.second;
        const PyValue* kind = question.is_object() ? question.get("type") : nullptr;
        const std::string k = kind && kind->is_string() ? kind->text : std::string();
        const ojson refused = refusal(kind, question, refs, config);
        if (!refused.is_null()) {
            req.errors[key] = refused;
            continue;
        }
        const ojson type = type_json(kind);
        try {
            if (!question.is_object()) throw ContractError("a question is an object with a type");
            const PyValue* at = question.get("at");
            if (k == "track" || (k == "box" && at && !at->is_null())) {
                for (ComposedRow& row : video_rows(cx, key, question, k)) req.rows.push_back(std::move(row));
            } else {
                req.rows.push_back(compose_row(cx, key, question));
            }
        } catch (const MaxLengthExceeded& e) {
            req.errors[key] = caught(type, "max_length_exceeded", e.what());
        } catch (const QuestionRefused& e) {
            req.errors[key] = caught(type, "invalid_question", e.what(), e.reason);
        } catch (const ContractError& e) {
            req.errors[key] = caught(type, "invalid_question", e.what());
        }
    }
    if (!req.images.empty()) req.image_token_id = cx.pad_id();
    if (cx.clips_loaded || !cx.window_sets.empty()) req.video_token_id = cx.video_pad_id();
    return req;
}

std::vector<std::vector<size_t>> state_groups(const ComposedRequest& request) {
    std::vector<std::string> order;
    std::map<std::string, std::vector<size_t>> by_state;
    for (size_t i = 0; i < request.rows.size(); i++) {
        const std::string& s = request.rows[i].state_key;
        if (!by_state.count(s)) order.push_back(s);
        by_state[s].push_back(i);
    }
    std::vector<std::vector<size_t>> out;
    for (const auto& s : order) out.push_back(by_state[s]);
    return out;
}

EngineStateRequest build_engine_request(const ComposedRequest& request, const std::vector<size_t>& group,
                                        const std::function<EngineImage(size_t)>& image) {
    EngineStateRequest out;
    out.image_token_id = request.image_token_id;
    out.video_token_id = request.video_token_id;
    std::map<int, int> compact;   // image table index -> position in out.images
    const auto use = [&](int table_index) {
        auto it = compact.find(table_index);
        if (it != compact.end()) return it->second;
        const ComposedImage& entry = request.images.at(static_cast<size_t>(table_index));
        EngineImage im;
        if (entry.window >= 0) {
            // a window's groups: the pixels the window's decoder kept (released after its call)
            const auto set = request.windows.find(entry.rate);
            if (set != request.windows.end()) {
                const auto px = set->second->at(static_cast<size_t>(entry.window)).group_pixels(entry.group);
                im.pixels = px;
                if (!px.empty()) { im.grid_h = entry.grid_h; im.grid_w = entry.grid_w; }
            }
        } else if (image) {
            im = image(static_cast<size_t>(table_index));
        }
        if (im.grid_h == 0) {
            im.grid_h = request.images[table_index].grid_h;
            im.grid_w = request.images[table_index].grid_w;
        }
        out.images.push_back(std::move(im));
        compact[table_index] = static_cast<int>(out.images.size()) - 1;
        return compact[table_index];
    };
    const ComposedRow& first = request.rows.at(group.at(0));
    out.state_ids.assign(first.ids.begin(), first.ids.begin() + first.state_len);
    for (int idx : first.state_images) out.state_image_ids.push_back(use(idx));
    for (size_t r : group) {
        const ComposedRow& row = request.rows.at(r);
        // upstream (_run_state_once) refuses requests whose questions do not share one state block
        if (row.state_len != first.state_len || !std::equal(out.state_ids.begin(), out.state_ids.end(), row.ids.begin())) {
            throw std::runtime_error("the questions of a request do not share the state block's tokens");
        }
        EngineBlock block;
        block.tag = row.key;
        block.ids.assign(row.ids.begin() + row.state_len, row.ids.end());
        block.readout_from_end = row.readout.offsets;
        for (int idx : row.block_images) block.image_ids.push_back(use(idx));
        out.blocks.push_back(std::move(block));
    }
    return out;
}

std::vector<std::vector<double>> row_probabilities(const ComposedRow& row, const std::vector<float>& hidden_rows, const float* readout,
                                                   int hidden_size, float temperature) {
    std::vector<std::vector<double>> values;
    const size_t positions = row.readout.offsets.size();
    if (hidden_rows.size() != positions * static_cast<size_t>(hidden_size)) {
        throw std::runtime_error("the engine returned " + std::to_string(hidden_rows.size() / std::max(hidden_size, 1)) +
                                 " hidden rows for " + std::to_string(positions) + " readout positions");
    }
    for (size_t p = 0; p < positions; p++) {
        const auto probs = readout_probabilities(hidden_rows.data() + p * hidden_size, readout, hidden_size, row.readout.counts[p],
                                                 temperature);
        values.emplace_back(probs.begin(), probs.end());
    }
    return values;
}

ojson row_answer(const ComposedRow& row, const std::vector<std::vector<double>>& values, const ComposedConfig& config) {
    if (row.kind == "box") {
        const std::vector<std::vector<double>> edges(values.end() - 4, values.end());
        const compose::Box box = compose::decode(edges);
        ojson answer = compose::box_answer(box, compose::calibrate(compose::confidence(edges, box), config.box_calibration));
        if (row.time) answer["time"] = compose::py_round3(*row.time);   // a box at a video moment
        return answer;
    }
    return product_answer(row.question, values.at(0));
}

namespace {

/// One moment of a track: `{"time", "box", "confidence", "visible"}` (`_video_box`).
ojson track_entry(const ComposedRow& row, const std::vector<std::vector<double>>& values, const ComposedConfig& config) {
    const std::vector<std::vector<double>> edges(values.end() - 4, values.end());
    const compose::Box box = compose::decode(edges);
    const ojson answer = compose::box_answer(box, compose::calibrate(compose::confidence(edges, box), config.box_calibration));
    ojson e = ojson::object();
    e["time"] = compose::py_round3(row.time.value_or(0.0));
    e["box"] = answer["box"];
    e["confidence"] = answer["confidence"];
    e["visible"] = compose::visible_probability(values.at(0));
    return e;
}

}  // namespace

ojson respond_composed(const ComposedRequest& request, const std::vector<std::vector<std::vector<double>>>& values,
                       const ComposedConfig& config) {
    ojson answers = ojson::object();
    size_t tokens = 0;
    std::set<std::string> counted;
    for (const auto& key : request.keys) {
        auto e = request.errors.find(key);
        if (e != request.errors.end()) {
            answers[key] = e->second;
            continue;
        }
        const auto rows = request.rows_of(key);
        if (rows.empty()) throw std::runtime_error("no row for question " + key);
        const ComposedRow& row = request.rows[rows[0]];
        try {
            if (row.kind == "track") {
                std::vector<ojson> entries;
                for (size_t r : rows) entries.push_back(track_entry(request.rows[r], values.at(r), config));
                std::stable_sort(entries.begin(), entries.end(),
                                 [](const ojson& a, const ojson& b) { return a["time"].get<double>() < b["time"].get<double>(); });
                ojson track = ojson::array();
                for (auto& entry : entries) track.push_back(std::move(entry));
                ojson answer = ojson::object();
                answer["type"] = "track";
                answer["track"] = std::move(track);
                answers[key] = std::move(answer);
            } else {
                answers[key] = row_answer(row, values.at(rows[0]), config);
            }
            // upstream counts a question's tokens once it has answered: each distinct state once, every block
            for (size_t r : rows) {
                const ComposedRow& used = request.rows[r];
                if (counted.insert(used.state_key).second) tokens += static_cast<size_t>(used.state_len);
                tokens += used.ids.size() - static_cast<size_t>(used.state_len);
            }
        } catch (const ContractError& ex) {
            answers[key] = caught(row.type, "invalid_question", ex.what());
        }
    }
    add_reasons(answers);
    ojson out = ojson::object();
    out["model"] = config.model_name;
    out["answers"] = answers;
    out["usage"] = {{"input_tokens", tokens}, {"output_tokens", 0}};
    return out;
}

ojson run_composed(const ComposedRequest& request, ReadoutEngine& engine, const float* readout, const ComposedConfig& config,
                   const std::function<EngineImage(size_t)>& image,
                   const std::function<void(size_t, size_t, const std::string&, double)>& progress) {
    std::vector<std::vector<std::vector<double>>> values(request.rows.size());
    const auto groups = state_groups(request);
    size_t done = 0;
    for (const auto& group : groups) {
        const auto began = std::chrono::steady_clock::now();
        const ComposedRow& first = request.rows[group[0]];
        {
            const EngineStateRequest call = build_engine_request(request, group, image);
            const auto hidden = engine.encode_state_once(call);
            if (hidden.size() != group.size()) throw std::runtime_error("the engine returned a different number of blocks than asked for");
            for (size_t i = 0; i < group.size(); i++) {
                values[group[i]] = row_probabilities(request.rows[group[i]], hidden[i], readout, engine.hidden_size(), config.temperature);
            }
        }
        // the window's frames are done: free its pixels before the next window is built
        if (first.window >= 0) {
            const auto set = request.windows.find(first.rate);
            if (set != request.windows.end()) set->second->at(static_cast<size_t>(first.window)).release_pixels();
        }
        done++;
        if (progress) {
            progress(done, groups.size(), first.state_key,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        }
    }
    return respond_composed(request, values, config);
}

}  // namespace flm::d3
