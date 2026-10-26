/// \file test_d3_video_composed.cpp
/// \brief M4 of the composed requests (box at a video moment, track, plain videos next to them), CPU only: the window loop through
///        a scripted engine (one state-once call per window, the moments as blocks, the readout rows), the question shapes and
///        refusals of upstream's `_video_box`, the row budget, the answers (track entries, box `time`), usage, the release of a
///        window's pixels, and `load_video_windows` over a fake decoder.
///
///   test_d3_video_composed <fixtures dir>      (reads decision_config.json for the codes and the box calibration)
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "decision/d3_compose.hpp"
#include "decision/d3_composed.hpp"
#include "decision/d3_engine_stub.hpp"
#include "video/video_decode.hpp"
#include "video/video_input.hpp"

using flm::d3::ojson;
using flm::d3::PyValue;
namespace cp = flm::d3::compose;
namespace v = flm::d3::video;

namespace {

int failures = 0, checks = 0;
void expect(bool ok, const std::string& what, const std::string& detail = std::string()) {
    checks++;
    if (!ok) {
        failures++;
        std::cerr << "FAIL " << what << (detail.empty() ? "" : ": " + detail.substr(0, 600)) << "\n";
    }
}

std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

constexpr int IMAGE_PAD = 248056, VIDEO_PAD = 248057;

/// One token per byte, the two pads one token each.
std::vector<int> fake_encode(const std::string& s) {
    std::vector<int> ids;
    const std::string image = flm::d3::IMAGE_TOKEN, video = flm::d3::VIDEO_TOKEN;
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, image.size(), image) == 0) { ids.push_back(IMAGE_PAD); i += image.size(); }
        else if (s.compare(i, video.size(), video) == 0) { ids.push_back(VIDEO_PAD); i += video.size(); }
        else ids.push_back(static_cast<unsigned char>(s[i++]));
    }
    return ids;
}

const std::string PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";
const std::string VIDEO_URL = "data:video/mp4;base64,AAAA";

/// A clip as the hook returns it: `total` source frames at `fps`, sampled like the plain `load_video` (even count).
struct Fake {
    int total = 600;
    double fps = 30.0;
    int h = 64, w = 64;
    int window_loads = 0;
    std::string window_error;         //!< non-empty: the hook throws VideoError(window_error)
    bool unavailable = false;         //!< the hook throws VideoBackendUnavailable

    std::vector<v::VideoWindow> windows(int rate) {
        window_loads++;
        if (unavailable) throw v::VideoBackendUnavailable("video inputs are not available: video decoding backend not available (fake)");
        if (!window_error.empty()) throw v::VideoError(window_error);
        std::vector<v::VideoWindow> out;
        int index = 0;
        for (const auto& plan : v::window_plan(total, fps, rate)) {
            v::VideoWindow win;
            win.indices = plan;
            win.fps = fps;
            win.total = total;
            win.grid = v::plan_video_grid(static_cast<int>(plan.size()), h, w);
            win.times = v::group_times(plan, fps);
            const size_t per_group = static_cast<size_t>(win.grid.grid_h) * win.grid.grid_w * v::PATCH_VALUES;
            win.pixels.resize(per_group * win.grid.grid_t);
            for (int g = 0; g < win.grid.grid_t; g++) {
                std::fill(win.pixels.begin() + static_cast<long>(per_group * g), win.pixels.begin() + static_cast<long>(per_group * (g + 1)),
                          static_cast<uint16_t>((index + 1) * 100 + g));   // marker of (window, group)
            }
            index++;
            out.push_back(std::move(win));
        }
        return out;
    }
    std::shared_ptr<const v::VideoClip> clip() {
        auto c = std::make_shared<v::VideoClip>();
        c->width = w;
        c->height = h;
        c->n = 8;
        c->frames.assign(static_cast<size_t>(c->n) * w * h * 3, 7);
        c->fps = 2.0;
        c->indices = {0, 1, 2, 3, 4, 5, 6, 7};
        c->total = 8;
        return c;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_d3_video_composed <fixtures dir>\n";
        return 2;
    }
    const ojson config = ojson::parse(read_file(std::string(argv[1]) + "/decision_config.json"));
    const std::vector<std::string> codes = config["codes"].get<std::vector<std::string>>();
    cp::Calibration calibration;
    for (const auto& k : config["systemone"]["box_calibration"]) calibration.emplace_back(k[0].get<double>(), k[1].get<double>());

    Fake fake;
    flm::d3::ComposedHooks hooks;
    hooks.encode = fake_encode;
    hooks.token_bytes = [](int id) -> size_t { return id == IMAGE_PAD || id == VIDEO_PAD ? 13 : 1; };
    hooks.load_video = [&](size_t, const PyValue&) { return fake.clip(); };
    hooks.load_windows = [&](const PyValue&, int rate) { return fake.windows(rate); };
    flm::d3::ComposedConfig cfg;
    cfg.box_calibration = calibration;
    cfg.video = true;
    cfg.refuse_video_questions = false;
    const int hidden = 8;
    std::vector<float> readout(255 * hidden);
    for (size_t i = 0; i < readout.size(); i++) readout[i] = static_cast<float>(std::sin(0.37 * i));
    int engine_calls_seen = 0;
    const auto make_engine = [&]() {
        return std::make_unique<flm::d3::ScriptedEngine>(hidden, [&](const flm::d3::EngineStateRequest& call, size_t b) {
            std::vector<float> rows;
            for (size_t r = 0; r < call.blocks[b].readout_from_end.size(); r++)
                for (int d = 0; d < hidden; d++) rows.push_back(std::sin(0.11f * (b * 7 + r * 3 + d)) * 2.0f);
            return rows;
        });
    };
    // the image callback of the host: video groups of the request's plain clip come from here (a marker per group)
    const auto image = [&](size_t) { return flm::d3::EngineImage(); };
    const auto body_of = [&](const std::string& s) { return flm::d3::parse_json(s); };
    const auto prepare = [&](const std::string& body, const flm::d3::ComposedConfig& c) {
        return flm::d3::prepare_composed(body_of(body), codes, c, hooks);
    };
    const auto request_error = [&](const std::string& body, const flm::d3::ComposedConfig& c) {
        try {
            prepare(body, c);
        } catch (const flm::d3::RequestError& e) {
            return e.detail.is_string() ? e.detail.get<std::string>() : e.detail.dump();
        }
        return std::string();
    };
    const auto answer_of = [&](const std::string& body, const flm::d3::ComposedConfig& c, const std::string& key) {
        auto req = prepare(body, c);
        auto engine = make_engine();
        const ojson r = flm::d3::run_composed(req, *engine, readout.data(), c, image);
        return r["answers"][key];
    };

    // ---- 1. a track at rate 2 over a 20 s clip: 3 windows (32, 32, 16 frames), one engine call each -----------------------
    {
        fake = Fake();
        auto req = prepare("{\"state\": \"s\", \"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"the bird\", \"rate\": 2}}}", cfg);
        expect(req.rows.size() == 40, "track rate 2: moments", std::to_string(req.rows.size()));
        expect(fake.window_loads == 1, "windows decoded once per rate");
        expect(req.clips.empty(), "plain clip not loaded when no plain question needs it");
        const auto groups = flm::d3::state_groups(req);
        expect(groups.size() == 3 && groups[0].size() == 16 && groups[1].size() == 16 && groups[2].size() == 8, "one state per window");
        auto engine = make_engine();
        std::vector<std::string> seen;
        const ojson r = flm::d3::run_composed(req, *engine, readout.data(), cfg, image,
                                              [&](size_t done, size_t of, const std::string& key, double) { seen.push_back(std::to_string(done) + "/" + std::to_string(of) + " " + key); });
        expect(engine->calls.size() == 3, "three engine calls");
        expect(seen == std::vector<std::string>{"1/3 window:2:0", "2/3 window:2:1", "3/3 window:2:2"}, "progress", seen.empty() ? "" : seen[0]);
        bool ok = engine->calls.size() == 3;
        size_t blocks_total = 0, state_total = 0;
        for (size_t w = 0; ok && w < 3; w++) {
            const auto& call = engine->calls[w];
            const size_t g = call.state_image_ids.size();
            ok = ok && call.video_token_id == VIDEO_PAD && (g == 16 || g == 8) && call.images.size() == g && call.blocks.size() == groups[w].size();
            // pad runs of the state: one per group, grid_h * grid_w / 4 each, all of the video pad id
            int pads = 0;
            for (int t : call.state_ids) pads += t == VIDEO_PAD;
            ok = ok && pads == static_cast<int>(g) * call.images[0].grid_h * call.images[0].grid_w / 4;
            for (size_t i = 0; ok && i < g; i++) ok = ok && call.images[i].pixels.size() == static_cast<size_t>(call.images[i].grid_h) * call.images[i].grid_w * v::PATCH_VALUES &&
                                                  call.images[i].pixels[0] == static_cast<uint16_t>((w + 1) * 100 + i);
            for (const auto& b : call.blocks) {
                ok = ok && b.readout_from_end.size() == 5 && b.readout_from_end.back() == 0 && b.image_ids.empty();
                for (size_t k = 1; k < b.readout_from_end.size(); k++) ok = ok && b.readout_from_end[k - 1] > b.readout_from_end[k];
                ok = ok && b.readout_from_end[0] < static_cast<int>(b.ids.size());
            }
            blocks_total += call.blocks.size();
            state_total += call.state_ids.size();
        }
        expect(ok, "window calls: video pad runs, the window's pixels, five readout rows per moment");
        expect(r["answers"]["t"]["type"] == "track" && r["answers"]["t"]["track"].size() == 40, "track answer", r.dump().substr(0, 300));
        bool shape = true;
        double last = -1;
        for (const auto& e : r["answers"]["t"]["track"]) {
            shape = shape && e.size() == 4 && e.contains("time") && e.contains("box") && e.contains("confidence") && e.contains("visible");
            std::vector<std::string> keys;
            for (auto it = e.begin(); it != e.end(); ++it) keys.push_back(it.key());
            shape = shape && keys == std::vector<std::string>{"time", "box", "confidence", "visible"};
            shape = shape && e["time"].get<double>() >= last && e["visible"].get<double>() >= 0 && e["visible"].get<double>() <= 1;
            last = e["time"].get<double>();
        }
        expect(shape, "track entries: time-sorted {time, box, confidence, visible}");
        // the times are the moments' times, round(t, 3)
        const auto& windows = *req.windows.at(2);
        expect(r["answers"]["t"]["track"][0]["time"].get<double>() == cp::py_round3(windows[0].times[0]), "first moment's time");
        size_t block_tokens = 0;
        for (const auto& row : req.rows) block_tokens += row.ids.size() - static_cast<size_t>(row.state_len);
        expect(r["usage"]["input_tokens"].get<size_t>() == state_total + block_tokens && block_tokens > 0 && blocks_total == 40,
               "usage: every window's state once + every moment's block", r["usage"].dump());
        bool released = true;
        for (const auto& w : windows) released = released && w.pixels.empty();
        expect(released, "a window's pixels are released after its call");
    }

    // ---- 2. a box at a time: the moment nearest `at`, one call, the answer carries `time` -----------------------------------
    {
        fake = Fake();
        auto req = prepare("{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"b\": {\"type\": \"box\", \"instructions\": \"the bird\", \"at\": 1.2}}}", cfg);
        expect(req.rows.size() == 1 && req.rows[0].time.has_value(), "box at: one moment row");
        // rate 1 over 20 s at 30 fps: a frame every 15 frames = 0.5 s; groups of two frames: times 0.25, 0.75, ...
        const double want = [&] {
            double best = 1e9, t = 0;
            for (const auto& w : *req.windows.at(1)) for (double m : w.times) if (std::fabs(m - 1.2) < best) { best = std::fabs(m - 1.2); t = m; }
            return t;
        }();
        expect(std::fabs(*req.rows[0].time - want) < 1e-12 && req.rows[0].state_key == "window:1:0" && req.rows[0].readout.offsets.size() == 4,
               "nearest moment", std::to_string(*req.rows[0].time));
        auto engine = make_engine();
        const ojson r = flm::d3::run_composed(req, *engine, readout.data(), cfg, image);
        const auto& a = r["answers"]["b"];
        expect(engine->calls.size() == 1 && engine->calls[0].blocks.size() == 1 && a["type"] == "box" && a["time"].get<double>() == cp::py_round3(want) &&
                   a.contains("box") && a.contains("confidence"),
               "box at a moment answer", r.dump().substr(0, 300));
        std::vector<std::string> keys;
        for (auto it = a.begin(); it != a.end(); ++it) keys.push_back(it.key());
        expect(keys == std::vector<std::string>{"type", "box", "confidence", "time"}, "box answer key order");
        // the text of the question block: the moment's time to one decimal
        const std::string t = cp::format_seconds(want);
        expect(req.rows[0].text.find("in the video frame at " + t + " seconds") != std::string::npos, "text carries the time", req.rows[0].text.substr(req.rows[0].text.size() > 400 ? req.rows[0].text.size() - 400 : 0));
    }

    // ---- 3. question shapes and refusals (inline answers) ---------------------------------------------------------------
    {
        fake = Fake();
        const std::string head = "{\"videos\": [\"" + VIDEO_URL + "\"], \"assets\": {\"<a>\": \"" + PNG + "\"}, \"questions\": {\"q\": ";
        const auto err = [&](const std::string& question, const std::string& videos = std::string()) {
            const std::string body = videos.empty() ? head + question + "}}" : "{\"videos\": " + videos + ", \"questions\": {\"q\": " + question + "}}";
            const ojson a = answer_of(body, cfg, "q");
            return a.dump();
        };
        const auto bad = [](const std::string& type, const std::string& message, const std::string& reason = "question_malformed") {
            return "{\"type\":\"" + type + "\",\"error\":\"invalid_question\",\"message\":\"" + message + "\",\"reason\":\"" + reason + "\"}";
        };
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": 99}") == bad("box", "at 99.0 s is outside the clip"), "at beyond the clip", err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": 99}"));
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": -0.5}") == bad("box", "at -0.5 s is outside the clip"), "at negative");
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": \"abc\"}") == bad("box", "could not convert string to float: 'abc'"), "at not a number");
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": [1]}") == bad("box", "float() argument must be a string or a real number, not 'list'"), "at a list");
        expect(err("{\"type\": \"track\", \"instructions\": \"x\", \"rate\": 3}") == bad("track", "rate must be 1 or 2 updates per second"), "rate 3");
        expect(err("{\"type\": \"track\", \"instructions\": \"x\", \"rate\": \"fast\"}") == bad("track", "invalid literal for int() with base 10: 'fast'"), "rate fast");
        expect(err("{\"type\": \"track\", \"instructions\": \"x\", \"rate\": null}") ==
                   bad("track", "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"), "rate null");
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": 1, \"rate\": 2}") == bad("box", "a box question takes ['at', 'instructions', 'type']"), "box with rate");
        expect(err("{\"type\": \"track\", \"at\": 1}") == bad("track", "a track question takes ['instructions', 'rate', 'type']"), "track with at, no instructions");
        expect(err("{\"type\": \"track\", \"instructions\": \"look at <a>\"}") == bad("track", "a video Box reads the video; images go in the state"), "asset in the instruction");
        expect(err("{\"type\": \"track\", \"instructions\": \"x\"}", "[]") == bad("track", "a track question needs a video", "video_needed"), "no video", err("{\"type\": \"track\", \"instructions\": \"x\"}", "[]"));
        expect(err("{\"type\": \"box\", \"instructions\": \"x\", \"at\": 1}", "[\"" + VIDEO_URL + "\", \"" + VIDEO_URL + "\"]") ==
                   bad("box", "a video box needs exactly one video, got 2", "video_target_ambiguous"), "two videos");
        // a rate given as "2" or 1.9 is read as Python's int(): 2 and 1
        auto r2 = prepare(head + "{\"type\": \"track\", \"instructions\": \"x\", \"rate\": \"2\"}}}", cfg);
        auto r19 = prepare(head + "{\"type\": \"track\", \"instructions\": \"x\", \"rate\": 1.9}}}", cfg);
        expect(r2.rows.size() == 40 && r19.rows.size() == 20, "rate \"2\" and 1.9", std::to_string(r2.rows.size()) + " " + std::to_string(r19.rows.size()));
        // an error on one question leaves the others answered
        const ojson mix = answer_of(head + "{\"type\": \"track\", \"instructions\": \"x\", \"rate\": 3}, \"ok\": {\"type\": \"box\", \"instructions\": \"x\", \"at\": 0.5}}}", cfg, "ok");
        expect(mix["type"] == "box" && mix.contains("time"), "other questions answered", mix.dump());
    }

    // ---- 4. the row budget (A5) is a request-level refusal; a clip that cannot be read is each question's error --------------
    {
        fake = Fake();
        flm::d3::ComposedConfig small = cfg;
        small.moment_row_limit = 30;
        const std::string track2 = "{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\", \"rate\": 2}}}";
        const std::string e = request_error(track2, small);
        expect(e.find("40 video moments, over the 30") != std::string::npos, "row budget 422", e);
        expect(request_error("{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\", \"rate\": 1}}}", small).empty(), "rate 1 fits (20 moments)");
        // the budget counts every question's moments
        const std::string two = "{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"a\": {\"type\": \"track\", \"instructions\": \"x\", \"rate\": 1}, \"b\": {\"type\": \"track\", \"instructions\": \"y\", \"rate\": 1}}}";
        expect(request_error(two, small).find("40 video moments") != std::string::npos, "two tracks add up");
        // an unreadable clip: the error of each question that asked; the failure is remembered per rate
        fake = Fake();
        fake.window_error = "each video must be at most 60.0 seconds long";
        auto req = prepare("{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"a\": {\"type\": \"track\", \"instructions\": \"x\"}, \"b\": {\"type\": \"box\", \"instructions\": \"x\", \"at\": 1}}}", cfg);
        expect(req.rows.empty() && req.errors.size() == 2 && req.errors.at("a")["message"] == fake.window_error && fake.window_loads == 1 && req.errors.at("a")["error"] == "invalid_question",
               "unreadable clip", std::to_string(fake.window_loads));
        fake = Fake();
        fake.unavailable = true;
        expect(request_error("{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\"}}}", cfg).find("video inputs are not available") == 0,
               "no decoder is a 422");
        expect(request_error("{\"long_context\": true, \"questions\": {\"c\": {\"type\": \"noul\"}}}", cfg) ==
                   "long_context is not supported by this build of d3:turbo (it answers choice, noul, score, box and track questions about text, images and video)",
               "the refusal names what the build answers");
        // a build without video: videos are refused, a track is the plain refusal
        fake = Fake();
        flm::d3::ComposedConfig off = cfg;
        off.video = false;
        expect(request_error("{\"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {\"b\": {\"type\": \"box\", \"instructions\": \"x\"}}}", off).find("videos are not supported") == 0, "videos refused without video");
        const ojson no_video = answer_of("{\"questions\": {\"t\": {\"type\": \"track\", \"instructions\": \"x\"}}}", off, "t");
        expect(no_video["reason"] == "video_needed", "track without videos", no_video.dump());
    }

    // ---- 5. a request with everything: a state image, a plain video question, a box on the image, a track -----------------------
    {
        fake = Fake();
        fake.total = 300;   // 10 s: rate 1: 20 frames, one window; the plain clip is 8 frames = 4 groups
        auto req = prepare("{\"state\": \"s\", \"images\": [\"" + PNG + "\"], \"videos\": [\"" + VIDEO_URL + "\"], \"questions\": {"
                           "\"c\": {\"type\": \"choice\", \"instructions\": \"which?\", \"criteria\": {\"x\": \"a\", \"y\": \"b\"}}, "
                           "\"b\": {\"type\": \"box\", \"instructions\": \"the bird\"}, "
                           "\"t\": {\"type\": \"track\", \"instructions\": \"the bird\", \"rate\": 1}}}", cfg);
        expect(req.clips.size() == 1 && req.rows.size() == 2 + 10, "rows: choice, box on the image, 10 moments", std::to_string(req.rows.size()));
        const auto groups = flm::d3::state_groups(req);
        expect(groups.size() == 2 && groups[0].size() == 2 && groups[1].size() == 10, "request state first, then the window", std::to_string(groups.size()));
        auto engine = make_engine();
        // the host supplies the plain clip's groups' pixels: a marker 7 per group
        const auto host_image = [&](size_t index) {
            flm::d3::EngineImage im;
            const auto& e = req.images.at(index);
            if (e.clip >= 0) {
                im.grid_h = e.grid_h;
                im.grid_w = e.grid_w;
                im.pixels.assign(static_cast<size_t>(e.grid_h) * e.grid_w * v::PATCH_VALUES, static_cast<uint16_t>(7000 + e.group));
            }
            return im;
        };
        const ojson r = flm::d3::run_composed(req, *engine, readout.data(), cfg, host_image);
        expect(engine->calls.size() == 2, "two calls");
        const auto& plain = engine->calls.at(0);
        const int clip_groups = req.clip_grids[0].grid_t;
        expect(plain.blocks.size() == 2 && plain.state_image_ids.size() == static_cast<size_t>(1 + clip_groups) && plain.images.size() == static_cast<size_t>(1 + clip_groups) &&
                   plain.video_token_id == VIDEO_PAD && plain.image_token_id == IMAGE_PAD,
               "the request state: the front image, then the clip's groups");
        int image_pads = 0, video_pads = 0;
        for (int t : plain.state_ids) { image_pads += t == IMAGE_PAD; video_pads += t == VIDEO_PAD; }
        expect(image_pads == 64 && video_pads == clip_groups * req.clip_grids[0].tokens_per_group(), "pad runs of the request state", std::to_string(image_pads) + " " + std::to_string(video_pads));
        expect(plain.images[1].pixels.size() > 0 && plain.images[1].pixels[0] == 7000 && plain.images[clip_groups].pixels[0] == 7000 + clip_groups - 1, "clip group pixels from the host");
        const auto& window = engine->calls.at(1);
        expect(window.blocks.size() == 10 && window.state_image_ids.size() == 1 + 10 && window.images[1].pixels[0] == 100,
               "the window state: the front image, then the window's groups");
        expect(r["answers"]["c"]["type"] == "choice" && r["answers"]["b"]["type"] == "box" && !r["answers"]["b"].contains("time") && r["answers"]["t"]["track"].size() == 10,
               "answers", r.dump().substr(0, 300));
        // the first call's state counts once: the window state is another one
        size_t blocks = 0;
        for (const auto& row : req.rows) blocks += row.ids.size() - static_cast<size_t>(row.state_len);
        expect(r["usage"]["input_tokens"].get<size_t>() == plain.state_ids.size() + window.state_ids.size() + blocks, "usage with two states", r["usage"].dump());
    }

    // ---- 6. load_video_windows over a fake decoder ----------------------------------------------------------------------------
    {
        struct Reader : v::FrameReader {
            v::StreamInfo si;
            int at = 0;
            explicit Reader(v::StreamInfo s) : si(s) {}
            const v::StreamInfo& info() const override { return si; }
            bool grab() override { return at++ < si.total; }
            bool retrieve(std::vector<uint8_t>& rgb) override {
                rgb.assign(static_cast<size_t>(si.width) * si.height * 3, static_cast<uint8_t>(at));
                return true;
            }
        };
        struct Backend : v::FrameBackend {
            v::StreamInfo si;
            explicit Backend(v::StreamInfo s) : si(s) {}
            std::string name() const override { return "fake"; }
            bool available(std::string*) const override { return true; }
            std::unique_ptr<v::FrameReader> open(const std::string&, std::string*) override { return std::make_unique<Reader>(si); }
        };
        v::VideoLoadOptions options;
        options.backend = std::make_shared<Backend>(v::StreamInfo{64, 48, 30.0, 600});
        const auto windows = v::load_video_windows(PyValue::string(VIDEO_URL), 2, options);
        const auto plan = v::window_plan(600, 30.0, 2);
        bool ok = windows.size() == plan.size() && windows.size() == 3;
        for (size_t w = 0; ok && w < windows.size(); w++) {
            const auto& win = windows[w];
            ok = ok && win.indices == plan[w] && win.times == v::group_times(plan[w], 30.0) && win.fps == 30.0 && win.total == 600 &&
                 win.grid.grid_t == static_cast<int>(plan[w].size()) / 2 &&
                 win.pixels.size() == static_cast<size_t>(win.grid.grid_t) * win.grid.grid_h * win.grid.grid_w * v::PATCH_VALUES &&
                 win.group_pixels(win.grid.grid_t - 1).size() == static_cast<size_t>(win.grid.grid_h) * win.grid.grid_w * v::PATCH_VALUES;
        }
        expect(ok, "load_video_windows: windows, times, pixels per group");
        std::string e1, e2, e3;
        try { v::load_video_windows(PyValue::integer("3"), 1, options); } catch (const v::VideoError& e) { e1 = e.what(); }
        expect(e1 == "a video Box needs a path, an http(s) URL or a data:video/...;base64 URL", "a frame list is not a window source", e1);
        try { v::load_video_windows(PyValue::string("/tmp/x.mp4"), 1, options); } catch (const v::VideoError& e) { e2 = e.what(); }
        expect(e2.find("strict=True") != std::string::npos, "a path is refused in strict mode", e2);
        v::VideoLoadOptions none;
        try { v::load_video_windows(PyValue::string(VIDEO_URL), 1, none); } catch (const v::VideoBackendUnavailable& e) { e3 = e.what(); }
        expect(e3.find("video inputs are not available") == 0, "no backend", e3);
        // over 60 s: 1,900 frames at 30 fps
        v::VideoLoadOptions longer;
        longer.backend = std::make_shared<Backend>(v::StreamInfo{64, 48, 30.0, 1900});
        std::string e4;
        try { v::load_video_windows(PyValue::string(VIDEO_URL), 1, longer); } catch (const v::VideoError& e) { e4 = e.what(); }
        expect(e4.find("at most 60") != std::string::npos, "the 60 s cap", e4);
    }

    std::cout << "d3 video composed: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
