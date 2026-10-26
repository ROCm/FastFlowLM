/// \file test_d3_video.cpp
/// \brief Tests of the d3 video input side (src/common/video + the video paths of src/common/decision/d3_contract.cpp):
///        no NPU, no model. Goldens come from the upstream Python processor (gen_fixtures.py).
///
///   test_d3_video <fixtures dir> [<big fixtures dir>] [<ffmpeg lib dir>]
///
/// Sections: sampling tables, per-case processor pixels / grid / text / prompts, pure helper checks (window decode
/// with a fake reader, frame lists, request validation), the contract (prepare + tokenize with a fake tokenizer),
/// M-RoPE positions, and the FFmpeg backend against OpenCV's frames (skipped when the libraries are missing).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "decision/d3_contract.hpp"
#include "decision/py_json.hpp"
#include "video/video_decode.hpp"
#include "video/video_input.hpp"
#include "video/video_preprocess.hpp"
#include "video/video_sampling.hpp"

using flm::d3::PyValue;
using json = nlohmann::ordered_json;
namespace v = flm::d3::video;

namespace {

int failures = 0, checks = 0, skipped = 0;

void fail(const std::string& id, const std::string& what) {
    if (failures < 60) std::cerr << "FAIL " << id << ": " << what << "\n";
    failures++;
}
#define CHECK(id, cond, what)                                           \
    do {                                                                \
        checks++;                                                       \
        if (!(cond)) fail((id), (what));                                \
    } while (0)

std::string read_file(const std::string& path, bool* ok = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (ok) *ok = false;
        return std::string();
    }
    if (ok) *ok = true;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---- md5 (RFC 1321) ---------------------------------------------------------------------------------------
std::string md5_hex(const uint8_t* data, size_t len) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8,
        0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
        0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87,
        0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039,
        0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
        0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391};
    static const int S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9,  14, 20, 5, 9,
                              14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; i++) msg.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t M[16];
        for (int i = 0; i < 16; i++) std::memcpy(&M[i], &msg[off + 4 * i], 4);
        uint32_t A = a0, B = b0, C = c0, D = d0;
        for (int i = 0; i < 64; i++) {
            uint32_t F;
            int g;
            if (i < 16) { F = (B & C) | (~B & D); g = i; }
            else if (i < 32) { F = (D & B) | (~D & C); g = (5 * i + 1) % 16; }
            else if (i < 48) { F = B ^ C ^ D; g = (3 * i + 5) % 16; }
            else { F = C ^ (B | ~D); g = (7 * i) % 16; }
            F = F + A + K[i] + M[g];
            A = D; D = C; C = B;
            B = B + ((F << S[i]) | (F >> (32 - S[i])));
        }
        a0 += A; b0 += B; c0 += C; d0 += D;
    }
    char out[33];
    const uint32_t r[4] = {a0, b0, c0, d0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) std::snprintf(out + 8 * i + 2 * j, 3, "%02x", (r[i] >> (8 * j)) & 0xFF);
    return std::string(out, 32);
}

uint64_t fnv_update(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

std::string base64(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t n = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
        out += T[n >> 18]; out += T[(n >> 12) & 63]; out += T[(n >> 6) & 63]; out += T[n & 63];
    }
    if (i + 1 == in.size()) {
        const uint32_t n = uint8_t(in[i]) << 16;
        out += T[n >> 18]; out += T[(n >> 12) & 63]; out += "==";
    } else if (i + 2 == in.size()) {
        const uint32_t n = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8);
        out += T[n >> 18]; out += T[(n >> 12) & 63]; out += T[(n >> 6) & 63]; out += '=';
    }
    return out;
}

/// The C++ twin of pattern() in gen_fixtures.py.
uint8_t synth(uint32_t seed, uint32_t noise, uint32_t f, uint32_t y, uint32_t x, uint32_t c) {
    uint32_t h = (seed * 0x9E3779B1u) ^ (f * 0x85EBCA6Bu) ^ (y * 0xC2B2AE35u) ^ (x * 0x27D4EB2Fu) ^ (c * 0x165667B1u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    const uint32_t n = ((h >> 24) * noise) >> 8;
    const uint32_t base = (x * 3 + y * 5 + f * 11 + c * 40) % 512;
    const uint32_t tri = base < 256 ? base : 511 - base;
    const uint32_t block = (((x / 16) + (y / 16)) & 1) * 40;
    return static_cast<uint8_t>(std::min<uint32_t>(tri + n + block, 255));
}

std::vector<uint8_t> synth_frame(uint32_t seed, uint32_t noise, int f, int h, int w) {
    std::vector<uint8_t> out(static_cast<size_t>(h) * w * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) out[(static_cast<size_t>(y) * w + x) * 3 + c] = synth(seed, noise, f, y, x, c);
    return out;
}

v::VideoClip synth_clip(const json& c, const std::vector<int>& indices, double fps, int total) {
    v::VideoClip clip;
    clip.width = c["w"];
    clip.height = c["h"];
    clip.n = static_cast<int>(indices.size());
    clip.fps = fps;
    clip.indices = indices;
    clip.total = total;
    for (int f : indices) {
        const auto fr = synth_frame(c["seed"], c["noise"], f, clip.height, clip.width);
        clip.frames.insert(clip.frames.end(), fr.begin(), fr.end());
    }
    return clip;
}

std::vector<int> frame_numbers(const v::VideoClip& clip, const json& c) {
    // the frame numbers the pattern is a function of: source indices for decoded clips, list positions for lists
    (void)c;
    return clip.indices;
}

// ---- 1. sampling tables -----------------------------------------------------------------------------------

void test_tables(const json& t) {
    for (const auto& e : t["explicit"]) {
        const int total = e["total"];
        const double fps = e["fps"];
        const std::string id = "table total=" + std::to_string(total) + " fps=" + std::to_string(fps);
        CHECK(id, v::sample_indices(total, fps) == e["sample"].get<std::vector<int>>(), "sample_indices");
        for (int rate : {1, 2}) {
            const auto got = v::window_plan(total, fps, rate);
            const auto& want = e["windows"][std::to_string(rate)];
            bool ok = got.size() == want.size();
            for (size_t i = 0; ok && i < got.size(); i++) {
                ok = got[i] == want[i]["indices"].get<std::vector<int>>();
                const auto times = v::group_times(got[i], fps);
                const auto wt = want[i]["times"].get<std::vector<double>>();
                ok = ok && times == wt;
            }
            CHECK(id + " rate=" + std::to_string(rate), ok, "window_plan / group_times");
        }
    }
    // exhaustive sweeps, hashed the way gen_fixtures.py does
    uint64_t hs = 0xCBF29CE484222325ULL, hw = 0xCBF29CE484222325ULL;
    for (double fps : t["fps_list"].get<std::vector<double>>()) {
        for (int total = 1; total < t["sweep_total_max"].get<int>(); total++) {
            const auto s = v::sample_indices(total, fps);
            const int32_t head[2] = {total, static_cast<int32_t>(s.size())};
            hs = fnv_update(hs, head, 8);
            hs = fnv_update(hs, s.data(), s.size() * 4);
            for (int rate : {1, 2}) {
                if (total > 400 && total % 7) continue;
                for (const auto& w : v::window_plan(total, fps, rate)) {
                    const std::vector<double> times = v::group_times(w, fps);
                    const int32_t h3[3] = {total, rate, static_cast<int32_t>(w.size())};
                    hw = fnv_update(hw, h3, 12);
                    hw = fnv_update(hw, w.data(), w.size() * 4);
                    hw = fnv_update(hw, times.data(), times.size() * 8);
                }
            }
        }
    }
    CHECK("sweep sample_indices", hs == t["sample_fnv"].get<uint64_t>(), "fnv of 33,000 sampled index lists differs");
    CHECK("sweep window_plan/group_times", hw == t["window_fnv"].get<uint64_t>(), "fnv of the window sweep differs");
}

// ---- 2. cases ---------------------------------------------------------------------------------------------

/// A tokenizer for tests: special tokens `<|name|>` are one id each, every other byte its own id.
std::vector<int> fake_encode(const std::string& text) {
    static const std::regex special("<\\|[a-z_]+\\|>");
    std::vector<int> out;
    size_t at = 0;
    auto bytes = [&](size_t from, size_t to) {
        for (size_t i = from; i < to; i++) out.push_back(static_cast<uint8_t>(text[i]));
    };
    for (auto it = std::sregex_iterator(text.begin(), text.end(), special); it != std::sregex_iterator(); ++it) {
        bytes(at, static_cast<size_t>(it->position()));
        const std::string name = it->str();
        int id = 100000;
        for (char ch : name) id = id * 31 % 999983 + ch;
        out.push_back(1000 + id % 100000);
        at = static_cast<size_t>(it->position() + it->length());
    }
    bytes(at, text.size());
    return out;
}

/// A PNG header only: enough for inspect_image (signature + IHDR); the decode callback is not run.
std::string fake_png_url(int w, int h) {
    std::string d = "\x89PNG\r\n\x1a\n";
    auto be = [&](uint32_t n) { for (int s = 24; s >= 0; s -= 8) d += static_cast<char>((n >> s) & 0xFF); };
    be(13);
    d += "IHDR";
    be(w);
    be(h);
    d += std::string("\x08\x02\x00\x00\x00", 5);
    d += std::string(4, '\0');
    return "data:image/png;base64," + base64(d);
}

PyValue to_py(const json& j) { return flm::d3::parse_json(j.dump()); }

struct CaseStats {
    long long px = 0, diff1 = 0, diffbig = 0;
    int maxdiff = 0;
};

void test_case(const std::string& dir, const json& c, int max_video_tokens, CaseStats& stats) {
    const std::string name = c["name"];
    const std::string mode = c["mode"];
    const int total = c["total"];
    const double fps = c["fps"];
    v::VideoProcessorConfig cfg;
    cfg.max_video_tokens = max_video_tokens;
    // sampling
    std::vector<int> indices = mode == "decode" ? v::sample_indices(total, fps) : v::frame_list_indices(total);
    v::pad_indices_to_even(indices);
    CHECK(name + " indices", indices == c["indices"].get<std::vector<int>>(), "sampled / padded indices differ");
    const double clip_fps = mode == "decode" ? fps : v::VIDEO_FPS;
    v::VideoClip clip = synth_clip(c, frame_numbers(v::VideoClip{.indices = indices}, c), clip_fps, total);
    CHECK(name + " clip meta", clip.n == c["n_frames"].get<int>() && clip.fps == c["video_fps"].get<double>() &&
                                   clip.total == c["video_total"].get<int>(),
          "clip frames / fps / total");
    // grid
    v::VideoGrid grid;
    try {
        grid = v::plan_video_grid(clip.n, clip.height, clip.width, cfg);
    } catch (const v::VideoError& e) {
        CHECK(name + " error", c.contains("error") && std::string(e.what()) == c["error"].get<std::string>(),
              std::string("error text: ") + e.what());
        return;
    }
    if (c.contains("error")) {
        fail(name + " error", "upstream refuses (" + c["error"].get<std::string>() + ") but the grid planned");
        return;
    }
    const auto g = c["grid"].get<std::vector<int>>();
    CHECK(name + " grid", grid.grid_t == g[0] && grid.grid_h == g[1] && grid.grid_w == g[2],
          "grid " + std::to_string(grid.grid_t) + "x" + std::to_string(grid.grid_h) + "x" + std::to_string(grid.grid_w));
    CHECK(name + " tokens", grid.tokens() == c["planned_tokens"].get<long long>(), "planned tokens");
    CHECK(name + " resized hw", grid.resized_h == c["resized"][0].get<int>() && grid.resized_w == c["resized"][1].get<int>(),
          "resized size");
    // pixels: the resize against torchvision's (+-1 LSB budget, A8), then the whole tensor
    const std::vector<uint8_t> resized = v::resize_clip(clip, grid);
    // The resized frames are committed only for the small cases (gen_fixtures.py regenerates all of them); for the
    // others the md5 of the whole pixel tensor below is the exactness check.
    bool have_gold = false;
    const std::string gold = read_file(dir + "/" + name + ".resized.u8", &have_gold);
    int maxdiff = 0;
    long long nd = 0;
    if (have_gold) {
        CHECK(name + " resized size bytes", gold.size() == resized.size(), "resized golden size " + std::to_string(gold.size()));
        if (gold.size() == resized.size()) {
            for (size_t i = 0; i < resized.size(); i++) {
                const int d = std::abs(static_cast<int>(resized[i]) - static_cast<int>(static_cast<uint8_t>(gold[i])));
                maxdiff = std::max(maxdiff, d);
                nd += d != 0;
            }
        }
        stats.px += static_cast<long long>(resized.size());
        stats.diff1 += nd;
        stats.maxdiff = std::max(stats.maxdiff, maxdiff);
        CHECK(name + " resize +-1 LSB", maxdiff <= 1, "max |diff| " + std::to_string(maxdiff));
    }
    if (have_gold) CHECK(name + " resize exact", maxdiff == 0, "resize differs from torchvision in " + std::to_string(nd) + " values");
    const std::vector<float> px = v::preprocess_video(clip, grid, cfg);
    CHECK(name + " pixel count", static_cast<long long>(px.size()) == c["pixel_count"].get<long long>(), "pixel count");
    if (maxdiff == 0) {
        CHECK(name + " pixel md5", md5_hex(reinterpret_cast<const uint8_t*>(px.data()), px.size() * 4) == c["pixel_md5"].get<std::string>(),
              "pixel_values_videos differ from the processor's (md5)");
    }
    // bf16 helper is the plain rounding of those floats
    std::vector<uint16_t> bf;
    v::preprocess_video_bf16(clip, grid, bf, cfg);
    bool same = bf.size() == px.size();
    for (size_t i = 0; same && i < px.size(); i += 997) same = bf[i] == v::float_to_bf16_bits(px[i]);
    CHECK(name + " bf16", same, "bf16 bits");
    // text
    const auto times = v::group_timestamps(clip.indices, clip.fps);
    const std::string vt = v::video_placeholder(times, grid.tokens_per_group(), true);
    CHECK(name + " video text", vt == c["video_text"].get<std::string>(), "video text differs from replace_video_token");
    // prompts through the contract: render, expansion, prepare + tokenize with a fake tokenizer
    for (const auto& p : c["prompts"]) {
        const int n_images = p["images"];
        std::vector<std::string> codes;
        for (int i = 0; i < 255; i++) codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string(1, 'A' + (i - 26) / 26) + std::string(1, 'A' + (i - 26) % 26));
        PyValue state = to_py(p["state"]);
        flm::d3::Question q = flm::d3::normalize_question(to_py(p["question"]));
        const std::string text = flm::d3::render_video(state, q, codes, n_images, 1);
        {
            const std::string gt = p["text"].get<std::string>();
            size_t at = 0;
            while (at < gt.size() && at < text.size() && gt[at] == text[at]) at++;
            CHECK(name + " prompt text", text == gt, "rendered prompt differs at byte " + std::to_string(at) + ": got [" + text.substr(at, 40) + "] want [" + gt.substr(at, 40) + "]");
        }
        std::string with_images = text;
        if (n_images) {
            std::string pad = flm::d3::IMAGE_TOKEN, run;
            for (int i = 0; i < 64; i++) run += pad;
            for (size_t at = with_images.find(pad); at != std::string::npos; at = with_images.find(pad, at + run.size()))
                with_images.replace(at, pad.size(), run);
        }
        const std::string expanded = v::expand_video_text(with_images, {vt});
        CHECK(name + " expanded prompt", expanded == p["expanded"].get<std::string>(), "expanded prompt differs from get_text_with_replacements");
        // prepare + tokenize
        json body = {{"state", p["state"]}, {"questions", {{"q", p["question"]}}}};
        if (n_images) body["images"] = json::array({fake_png_url(256, 256)});
        body["videos"] = json::array({"x"});
        flm::d3::PrepareOptions opt;
        opt.video_config = cfg;
        auto shared = std::make_shared<const v::VideoClip>(clip);
        opt.load_video = [&](size_t, const PyValue&) { return shared; };
        flm::d3::Prepared prep;
        try {
            prep = flm::d3::prepare(to_py(body), codes, nullptr, opt);
        } catch (const flm::d3::RequestError& e) {
            fail(name + " prepare", e.what());
            continue;
        }
        CHECK(name + " prepare text", prep.texts.count("q") && prep.texts["q"] == p["text"].get<std::string>(), "prepare text");
        flm::d3::tokenize(prep, fake_encode, 0);
        const auto want = fake_encode(p["expanded"].get<std::string>());
        CHECK(name + " tokenized sequence", prep.sequences.count("q") && prep.sequences["q"] == want,
              "tokenize() expansion differs from tokenizing the expanded prompt (" +
                  std::to_string(prep.sequences.count("q") ? prep.sequences["q"].size() : 0) + " vs " + std::to_string(want.size()) + ")");
        // pad runs and the engine's position plan on this sequence
        const int image_id = fake_encode(flm::d3::IMAGE_TOKEN)[0], video_id = fake_encode(flm::d3::VIDEO_TOKEN)[0];
        std::vector<std::pair<int, int>> groups;
        if (n_images) groups.emplace_back(16, 16);
        for (int gi = 0; gi < grid.grid_t; gi++) groups.emplace_back(grid.grid_h, grid.grid_w);
        std::vector<int> pos;
        try {
            const int next = v::mrope_positions(prep.sequences["q"], image_id, video_id, groups, pos);
            CHECK(name + " mrope next", next > 0 && pos.size() == prep.sequences["q"].size() * 3, "mrope output size");
        } catch (const v::VideoError& e) {
            fail(name + " mrope", e.what());
        }
    }
}

// ---- 3. pure helpers --------------------------------------------------------------------------------------

/// A fake reader: frame k is filled with the byte k (mod 256); `bad` frames fail to grab.
struct FakeReader : v::FrameReader {
    v::StreamInfo si;
    int frames_in_file;
    int next = 0, last = -1;
    std::vector<int> bad;
    FakeReader(v::StreamInfo s, int n, std::vector<int> b = {}) : si(s), frames_in_file(n), bad(std::move(b)) {}
    const v::StreamInfo& info() const override { return si; }
    bool grab() override {
        if (next >= frames_in_file) return false;
        last = next++;
        return std::find(bad.begin(), bad.end(), last) == bad.end();
    }
    bool retrieve(std::vector<uint8_t>& rgb) override {
        rgb.assign(static_cast<size_t>(si.width) * si.height * 3, static_cast<uint8_t>(last));
        return true;
    }
};

struct FakeBackend : v::FrameBackend {
    v::StreamInfo si;
    int frames_in_file;
    std::vector<int> bad;
    int opened = 0;
    bool up = true;
    FakeBackend(v::StreamInfo s, int n, std::vector<int> b = {}) : si(s), frames_in_file(n), bad(std::move(b)) {}
    std::string name() const override { return "fake"; }
    bool available(std::string* why) const override {
        if (!up && why) *why = "fake backend is down";
        return up;
    }
    std::unique_ptr<v::FrameReader> open(const std::string& bytes, std::string* error) override {
        if (bytes == "garbage") {
            *error = "the video could not be opened";
            return nullptr;
        }
        opened++;
        return std::make_unique<FakeReader>(si, frames_in_file, bad);
    }
};

void test_helpers() {
    // smart_resize ties: Python's round() is half to even
    {
        const auto hw = v::smart_resize_video(8, 48, 80, 2, 32, 4096, 200704LL * 8);   // 1.5 -> 2, 2.5 -> 2
        CHECK("smart_resize tie", hw.first == 64 && hw.second == 64, "round-half-even");
        const auto big = v::smart_resize_video(4, 1080, 1920, 2, 32, 4096, 200704LL * 4);
        CHECK("smart_resize cap", big.first == 320 && big.second == 576, "1080p -> " + std::to_string(big.first) + "x" + std::to_string(big.second));
        const auto small = v::smart_resize_video(4, 20, 30, 2, 32, 4096, 200704LL * 4);   // scaled up first (< 32)
        CHECK("smart_resize tiny", small.first % 32 == 0 && small.second % 32 == 0 && small.first >= 32, "tiny clip");
        bool threw = false;
        try { v::smart_resize_video(4, 8, 1700, 2, 32, 4096, 1 << 30); } catch (const v::VideoError& e) {
            threw = std::string(e.what()) == "absolute aspect ratio must be smaller than 200, got 212.5";
        }
        CHECK("smart_resize aspect", threw, "aspect ratio refusal");
    }
    // grid of the 360x640, 8 frame example the design verified with the real processor: (4, 20, 36), 180 tokens
    {
        const auto g = v::plan_video_grid(8, 360, 640);
        CHECK("grid 360x640", g.grid_t == 4 && g.grid_h == 20 && g.grid_w == 36 && g.tokens_per_group() == 180, "design 4.1 example");
    }
    // token budget
    {
        v::check_video_token_budget(16384);
        bool threw = false;
        try { v::check_video_token_budget(16385); } catch (const v::VideoError& e) {
            threw = std::string(e.what()) ==
                    "the videos take 16,385 input tokens, over the 16,384 tokens a request may spend on videos; send fewer or shorter videos";
        }
        CHECK("token budget text", threw, "16,384 tokens per request");
    }
    // timestamps text, banker's formatting
    {
        const auto t = v::group_timestamps({0, 12, 24, 36}, 24.0);
        CHECK("timestamps", t.size() == 2 && t[0] == 0.25 && t[1] == 1.25, "group timestamps");
        const std::string s = v::video_placeholder(t, 2, true);
        CHECK("placeholder text", s == "<0.2 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
                                       "<1.2 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>",
              "0.25 prints as 0.2, 1.25 as 1.2 (exact binary halves, round half even)");
        CHECK("odd indices repeat", v::group_timestamps({0, 12, 24}, 24.0).size() == 2, "odd tail pads");
    }
    // pad runs
    {
        const auto ids = v::expand_pad_runs({1, 9, 2, 9, 3}, 9, {2, 3});
        CHECK("pad runs", ids == std::vector<int>({1, 9, 9, 2, 9, 9, 9, 3}), "expand_pad_runs");
    }
    // window moments: track rate 1 and 2 on a 10.8 s, 30000/1001 fps clip (the hummingbird's shape)
    {
        const double fps = 30000.0 / 1001.0;
        const int total = 324;
        const auto w1 = v::window_plan(total, fps, 1), w2 = v::window_plan(total, fps, 2);
        const auto m1 = v::moments_of(w1, fps), m2 = v::moments_of(w2, fps);
        CHECK("window rate1", w1.size() == 1 && w1[0].size() == 22 && m1.size() == 11, "rate 1: 22 frames, 11 moments (" + std::to_string(w1[0].size()) + ")");
        size_t frames2 = 0;
        for (const auto& w : w2) frames2 += w.size();
        CHECK("window rate2 frames", frames2 >= 43 && frames2 <= 46, "rate 2 frames " + std::to_string(frames2));
        for (const auto& w : w2) CHECK("window <= 32", w.size() <= 32 && w.size() % 2 == 0 && w.size() >= 4, "window shape");
        CHECK("moments ordered", std::is_sorted(m2.begin(), m2.end(), [](auto& a, auto& b) { return a.time < b.time; }), "time order");
        // moments of rate 2 are twice as dense as rate 1
        CHECK("moments density", m2.size() >= 2 * m1.size() - 2 && m2.size() <= 2 * m1.size() + 2, "rate 2 has about twice the moments");
        const auto near = v::nearest_moment(m1, 3.1);
        CHECK("nearest moment", std::fabs(near.time - 3.1) <= 0.5, "nearest");
        bool threw = false;
        try { v::nearest_moment(m1, 12.0); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "at 12.0 s is outside the clip"; }
        CHECK("at outside", threw, "at beyond last moment + 1");
        threw = false;
        try { v::nearest_moment(m1, -0.5); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "at -0.5 s is outside the clip"; }
        CHECK("at negative", threw, "at < 0");
    }
    // a long clip: windows of 32 frames, each padded
    {
        const auto w = v::window_plan(60 * 30, 30.0, 2);   // 60 s, 30 fps, rate 2: 240 frames
        size_t frames = 0;
        for (const auto& x : w) frames += x.size();
        CHECK("60 s rate 2", w.size() == 8 && frames == 240, "8 windows of 30 frames at rate 2: " + std::to_string(w.size()));
        const auto w1 = v::window_plan(60 * 30, 30.0, 1);
        CHECK("60 s rate 1", w1.size() == 4, "4 windows at rate 1");
        const auto mm = v::moments_of(w, 30.0);
        CHECK("120 moments", mm.size() == 120, "60 s at rate 2 is 120 moments, got " + std::to_string(mm.size()));
        bool threw = false;
        try { v::check_moment_rows(257); } catch (const v::VideoError&) { threw = true; }
        CHECK("row budget", threw && !([] { try { v::check_moment_rows(256); return false; } catch (...) { return true; } })(), "256 rows run, 257 refused");
    }
    // question refusals
    {
        auto a = v::check_video_box("box", 0, 1);
        CHECK("video_needed", a && a->reason == "video_needed" && a->message == "a box question needs a video", "video_needed");
        auto b = v::check_video_box("track", 2, 1);
        CHECK("video_target_ambiguous", b && b->reason == "video_target_ambiguous" && b->message == "a video track needs exactly one video, got 2", "ambiguous");
        auto c = v::check_video_box("track", 1, 3);
        CHECK("rate", c && c->reason.empty() && c->message == "rate must be 1 or 2 updates per second", "rate refusal");
        CHECK("ok", !v::check_video_box("track", 1, 2) && !v::check_video_box("box", 1, 1), "valid questions");
    }
    // decode_clip over a fake reader
    {
        FakeBackend be({16, 8, 24.0, 100}, 100);
        const v::VideoClip c = v::decode_clip(be, "ok");
        CHECK("decode indices", c.indices == v::sample_indices(100, 24.0) || c.indices.size() == 8, "indices (8 frames of 100 at 24 fps)");
        CHECK("decode frames hold their number", c.n == static_cast<int>(c.indices.size()) && c.frame(1)[0] == c.indices[1], "pixels of frame k");
        CHECK("decode meta", c.fps == 24.0 && c.total == 100 && c.width == 16 && c.height == 8, "meta");
        // an odd count (5 of 5 frames at fps 1) is padded with its last frame
        FakeBackend odd({4, 4, 1.0, 5}, 5);
        const v::VideoClip o = v::decode_clip(odd, "ok");
        CHECK("decode odd padded", o.n == 6 && o.indices.size() == 6 && o.indices[5] == o.indices[4], "odd count repeats the last frame");
        // unknown count: counted, then read again
        FakeBackend unk({4, 4, 24.0, 0}, 30);
        const v::VideoClip u = v::decode_clip(unk, "ok");
        CHECK("decode unknown total", unk.opened == 2 && u.total == 30, "counts the frames, then reopens");
        // missing frames: the failed grabs skip their numbers
        FakeBackend holes({4, 4, 24.0, 40}, 40, {0});
        const v::VideoClip h = v::decode_clip(holes, "ok");
        CHECK("decode skips failed grabs", h.indices.front() != 0 || h.n >= 2, "frame 0 failed");
        // limits
        FakeBackend longc({4, 4, 24.0, 24 * 61}, 24 * 61);
        bool threw = false;
        try { v::decode_clip(longc, "ok"); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "each video must be at most 60 seconds long"; }
        CHECK("60 s limit", threw, "A4: a 61 s clip is refused");
        FakeBackend sixty({4, 4, 24.0, 24 * 60}, 24 * 60);
        CHECK("60 s ok", v::decode_clip(sixty, "ok").n == 32, "60 s exactly is accepted");
        FakeBackend huge({4000, 3000, 24.0, 100}, 100);
        threw = false;
        try { v::decode_clip(huge, "ok"); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "video frames must have at most 8,294,400 pixels"; }
        CHECK("source pixel limit", threw, "12 MP frames refused");
        FakeBackend nofrm({0, 0, 24.0, 10}, 10);
        threw = false;
        try { v::decode_clip(nofrm, "ok"); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "the video has no frame size"; }
        CHECK("no frame size", threw, "size 0");
        threw = false;
        try { v::decode_clip(be, "garbage"); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "the video could not be opened"; }
        CHECK("cannot open", threw, "unreadable bytes");
        FakeBackend down({4, 4, 24.0, 10}, 10);
        down.up = false;
        std::string msg;
        try { v::decode_clip(down, "ok"); } catch (const v::VideoBackendUnavailable& e) { msg = e.what(); }
        CHECK("backend unavailable", msg == "video inputs are not available: video decoding backend not available (fake backend is down)", msg);
        FakeBackend nofr({4, 4, 24.0, 0}, 0);
        threw = false;
        try { v::decode_clip(nofr, "ok"); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "the video has no frames"; }
        CHECK("empty video", threw, "unknown total and nothing to read");
        FakeBackend end({4, 4, 24.0, 40}, 40);
        CHECK("end_frame", v::decode_clip(end, "ok", {}, 10).total == 10, "end_frame clamps the total");
        // fps outside [0.1, 1000] counts as 24
        FakeBackend odd_fps({4, 4, 5000.0, 48}, 48);
        CHECK("unknown fps", v::decode_clip(odd_fps, "ok").fps == 24.0, "fps > 1000 -> 24");
    }
    // decode_windows over a fake reader: 10 s of 24 fps video, rate 2
    {
        FakeBackend be({4, 4, 24.0, 240}, 240);
        const auto windows = v::decode_windows(be, "ok", 2);
        const auto plan = v::window_plan(240, 24.0, 2);
        CHECK("windows count", windows.size() == plan.size(), "one clip per planned window");
        bool ok = windows.size() == plan.size();
        for (size_t i = 0; ok && i < plan.size(); i++) {
            ok = windows[i].indices == plan[i];
            for (int k = 0; ok && k < windows[i].n; k++) ok = windows[i].frame(k)[0] == static_cast<uint8_t>(plan[i][static_cast<size_t>(k)]);
        }
        CHECK("window frames", ok, "frames of each window are the planned source frames, duplicates repeated");
        FakeBackend cut({4, 4, 24.0, 240}, 100);   // the file ends early: later windows are empty and skipped
        const auto partial = v::decode_windows(cut, "ok", 1);
        CHECK("window short file", !partial.empty() && partial.size() <= v::window_plan(240, 24.0, 1).size(), "frames missing at the end");
        FakeBackend none({4, 4, 24.0, 0}, 10);
        bool threw = false;
        try { v::decode_windows(none, "ok", 1); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "the video has no frames"; }
        CHECK("windows no total", threw, "no frame count: no counting fallback (upstream)");
    }
    // frame lists
    {
        std::vector<v::RgbFrame> frames(2);
        for (auto& f : frames) { f.width = 3; f.height = 2; f.rgb.assign(18, 7); }
        const auto c = v::frames_clip(frames, {0, 1}, 2);
        CHECK("frames clip", c.n == 2 && c.fps == 2.0 && c.total == 2, "list clip");
        frames[1].width = 4;
        bool threw = false;
        try { v::frames_clip(frames, {0, 1}, 2); } catch (const v::VideoError& e) { threw = std::string(e.what()) == "all frames of a video must have the same size"; }
        CHECK("frames size", threw, "size mismatch");
        CHECK("thin list", v::frame_list_indices(33).size() == 32 && v::frame_list_indices(33).front() == 0 && v::frame_list_indices(33).back() == 32 &&
                               v::frame_list_indices(32).size() == 32 && v::frame_list_indices(1).size() == 1, "thinning to 32");
    }
}

// ---- 4. request values -------------------------------------------------------------------------------------

std::string error_of(const PyValue& value, const v::VideoLoadOptions& o) {
    try {
        v::load_video(value, o);
    } catch (const v::VideoBackendUnavailable& e) {
        return std::string("UNAVAILABLE: ") + e.what();
    } catch (const v::VideoError& e) {
        return e.what();
    }
    return "";
}

void test_request_values() {
    auto backend = std::make_shared<FakeBackend>(v::StreamInfo{8, 8, 24.0, 48}, 48);
    v::VideoLoadOptions o;
    o.backend = backend;
    const auto S = [](const std::string& s) { return PyValue::string(s); };
    CHECK("not inline", error_of(S("/tmp/clip.mp4"), o) == "cannot load this video with strict=True; pass the clip inline as a data:video/... URL", "path");
    CHECK("not inline url", error_of(S("https://x/y.mp4"), o) == "cannot load this video with strict=True; pass the clip inline as a data:video/... URL", "http");
    CHECK("empty", error_of(S(""), o) == "a video is a path, an http(s) URL, a data:video/...;base64 URL or a frame array", "empty string");
    CHECK("number", error_of(PyValue::number(3), o) == "cannot load this video with strict=True; pass the clip inline as a data:video/... URL", "number");
    CHECK("bad data url", error_of(S("data:video/mp4,abc"), o) == "expected a base64 data URL: data:video/<type>;base64,<data>", "no ;base64");
    CHECK("image data url", error_of(S("data:image/png;base64,AAAA"), o) == "expected a base64 data URL: data:video/<type>;base64,<data>", "png as a video");
    CHECK("type", error_of(S("data:video/avi;base64,AAAA"), o) == "video data URLs take the types mp4, webm, quicktime, x-matroska, got 'avi'", "type list");
    CHECK("type params", error_of(S("data:video/mp4;codecs=avc1;base64,!!!"), o) == "the data URL does not hold valid base64", "type ends at the first ';'");
    CHECK("base64", error_of(S("data:video/mp4;base64,A"), o) == "the data URL does not hold valid base64", "bad base64");
    const std::string big(42666700, 'A');
    CHECK("too long", error_of(S("data:video/webm;base64," + big), o) == "the encoded data exceeds the 32,000,000 byte limit", "byte limit");
    CHECK("mime case", error_of(S("data:VIDEO/MP4;BASE64," + base64("ok")), o) == "", "the head is matched lower-cased");
    CHECK("quicktime", error_of(S("data:video/quicktime;base64," + base64("ok")), o) == "", "quicktime");
    CHECK("mkv", error_of(S("data:video/x-matroska;base64," + base64("ok")), o) == "", "x-matroska");
    CHECK("gif ext", error_of(S("data:image/gif;base64," + base64("ok")), o) == "", "gif extension");
    o.allow_gif = false;
    CHECK("gif off", error_of(S("data:image/gif;base64," + base64("ok")), o) == "expected a base64 data URL: data:video/<type>;base64,<data>", "gif off");
    o.allow_gif = true;
    CHECK("unreadable", error_of(S("data:video/mp4;base64," + base64("garbage")), o) == "the video could not be opened", "decoder refuses");
    v::VideoLoadOptions none;
    CHECK("no decoder", error_of(S("data:video/mp4;base64," + base64("ok")), none).rfind("UNAVAILABLE: video inputs are not available: video decoding backend not available", 0) == 0, "no backend configured");
    backend->up = false;
    CHECK("backend down", error_of(S("data:video/mp4;base64," + base64("ok")), o) ==
                              "UNAVAILABLE: video inputs are not available: video decoding backend not available (fake backend is down)", "down");
    backend->up = true;

    // frame lists
    o.decode_image = [](const flm::d3::ImageFile& f, v::RgbFrame& out, std::string*) {
        out.width = f.width;
        out.height = f.height;
        out.rgb.assign(static_cast<size_t>(f.width) * f.height * 3, static_cast<uint8_t>(f.data.size()));
        return true;
    };
    json frames = json::array();
    for (int i = 0; i < 5; i++) frames.push_back(fake_png_url(40, 30));
    {
        const v::VideoClip c = v::load_video(to_py({{"frames", frames}}), o);
        CHECK("list clip", c.n == 6 && c.indices.size() == 6 && c.indices[5] == 4 && c.fps == 2.0 && c.total == 5 && c.width == 40, "5 frames pad to 6");
    }
    CHECK("list works without a decoder", ([&] { v::VideoLoadOptions n2 = o; n2.backend = nullptr; return error_of(to_py({{"frames", frames}}), n2); })() == "", "frame lists need no backend");
    CHECK("list key", error_of(to_py({{"frames", frames}, {"fps", 3}}), o).find("a video given as frames is {\"frames\"") == 0, "unknown key");
    CHECK("list empty", error_of(to_py({{"frames", json::array()}}), o) == "a video given as frames needs at least one frame", "empty");
    CHECK("list not array", error_of(to_py({{"frames", "x"}}), o).find("a video given as frames is") == 0, "not an array");
    {
        json bad = frames;
        bad[3] = "data:image/png;base64,AAAA";
        CHECK("list bad frame", error_of(to_py({{"frames", bad}}), o).rfind("frames[3]: ", 0) == 0, "index of the bad frame");
        json sized = frames;
        sized[2] = fake_png_url(41, 30);
        CHECK("list sizes", error_of(to_py({{"frames", sized}}), o) == "all frames of a video must have the same size", "size mismatch");
        json many = json::array();
        for (int i = 0; i < 40; i++) many.push_back(fake_png_url(16, 16));
        const v::VideoClip c = v::load_video(to_py({{"frames", many}}), o);
        CHECK("list thinned", c.n == 32 && c.indices.back() == 39 && c.indices.front() == 0, "40 frames thin to 32");
        json huge = json::array();
        for (int i = 0; i < 2049; i++) huge.push_back("x");
        CHECK("list too many", error_of(to_py({{"frames", huge}}), o) == "a video given as frames takes at most 2,048 frames", "frame cap");
    }
    flm::d3::PrepareOptions po;
    CHECK("object refused when frame lists are off", ([&] { v::VideoLoadOptions n3 = o; n3.allow_frame_lists = false; return error_of(to_py({{"frames", frames}}), n3); })() ==
                                                          "cannot load this video with strict=True; pass the clip inline as a data:video/... URL", "switch");
    (void)po;
}

// ---- 5. the contract: refusals unchanged, videos through prepare ------------------------------------------

void test_contract() {
    std::vector<std::string> codes;
    for (int i = 0; i < 255; i++) codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string(1, 'A' + (i - 26) / 26) + std::string(1, 'A' + (i - 26) % 26));
    const std::string req = R"({"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}},"videos":["data:video/mp4;base64,AAAA"]})";
    auto message = [&](const flm::d3::PrepareOptions& o, const std::string& body) {
        try {
            flm::d3::prepare(flm::d3::parse_json(body), codes, nullptr, o);
        } catch (const flm::d3::RequestError& e) {
            return std::string(e.what());
        }
        return std::string("OK");
    };
    flm::d3::PrepareOptions off;   // no loader: the M1 refusals, byte for byte
    CHECK("M1 refusal", message(off, req) == "video inputs are not available: this FastFlowLM build runs d3 on text and images", "default");
    flm::d3::PrepareOptions turbo = off;
    turbo.unsupported_for = "d3:turbo";
    CHECK("M1 refusal turbo", message(turbo, req) == "videos are not supported by this build of d3:turbo (it answers choice, noul and score questions about text and images)", "turbo");
    CHECK("empty list is the text path", message(off, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"videos":[]})") == "OK", "[]");

    // with a loader
    int calls = 0;
    flm::d3::PrepareOptions on;
    on.load_video = [&](size_t i, const PyValue&) -> std::shared_ptr<const v::VideoClip> {
        calls++;
        if (i == 1) throw v::VideoError("the video could not be opened");
        auto c = std::make_shared<v::VideoClip>();
        c->width = 64; c->height = 64; c->n = 4; c->fps = 24.0; c->total = 4; c->indices = {0, 1, 2, 3};
        c->frames.assign(static_cast<size_t>(4) * 64 * 64 * 3, 0);
        return c;
    };
    CHECK("loaded", message(on, req) == "OK" && calls == 1, "one video");
    calls = 0;
    CHECK("videos[1] error", message(on, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"videos":["a","b","c"]})") == "videos[1]: the video could not be opened" && calls == 2, "stops at the first bad video");
    flm::d3::PrepareOptions gone = off;
    gone.load_video = [](size_t, const PyValue&) -> std::shared_ptr<const v::VideoClip> { throw v::VideoBackendUnavailable(v::backend_unavailable_message("libs missing")); };
    CHECK("backend missing", message(gone, req) == "video inputs are not available: video decoding backend not available (libs missing)", "422 text without an index");

    // 16,384 tokens per request: 6 clips of 32 frames at 720p (2,880 tokens each) are refused, 5 run
    flm::d3::PrepareOptions big = off;
    big.load_video = [](size_t, const PyValue&) -> std::shared_ptr<const v::VideoClip> {
        auto c = std::make_shared<v::VideoClip>();
        c->width = 1280; c->height = 720; c->n = 32; c->fps = 2.0; c->total = 32;
        for (int i = 0; i < 32; i++) c->indices.push_back(i);
        c->frames.assign(1, 0);   // never read by prepare
        return c;
    };
    auto with_videos = [&](int n) {
        std::string b = R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"videos":[)";
        for (int i = 0; i < n; i++) b += std::string(i ? "," : "") + "\"v\"";
        return b + "]}";
    };
    CHECK("5 videos fit", message(big, with_videos(5)) == "OK", "5 x 2,880 tokens");
    CHECK("6 videos refused", message(big, with_videos(6)) ==
                                  "the videos take 17,280 input tokens, over the 16,384 tokens a request may spend on videos; send fewer or shorter videos",
          message(big, with_videos(6)));
    // end to end with the loader factory: data URL -> backend -> clip -> grid, texts, tokens
    {
        v::VideoLoadOptions lo;
        lo.backend = std::make_shared<FakeBackend>(v::StreamInfo{64, 48, 24.0, 96}, 96);
        flm::d3::PrepareOptions po;
        po.load_video = v::make_video_loader(lo);
        flm::d3::Prepared q = flm::d3::prepare(flm::d3::parse_json(R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"videos":["data:video/mp4;base64,)" + base64("ok") + R"("]})"), codes, nullptr, po);
        // 96 frames at 24 fps = 4 s: 8 frames, 4 groups; 64x48 stays small (min pixels) -> grid
        CHECK("loader end to end", q.videos.size() == 1 && q.videos[0]->n == 8 && q.video_runs.size() == 1 && q.video_runs[0].size() == 4 && q.has_videos,
              "prepared videos");
        flm::d3::tokenize(q, fake_encode, 0);
        CHECK("loader tokens", q.sequences.count("q") && !q.video_ids.empty() && q.video_ids[0].size() > 4 * q.video_runs[0][0], "video token ids");
    }
    // literal placeholder in the state
    flm::d3::PrepareOptions lit = on;
    const auto p = flm::d3::prepare(flm::d3::parse_json(R"({"state":"a <|video_pad|> b","questions":{"q":{"type":"noul","instructions":"x"}},"videos":["v"]})"), codes, nullptr, lit);
    CHECK("literal pad", p.errors.count("q") && p.errors.at("q")["message"] == "the state or question contains a literal image or video placeholder token", "placeholder in the state");
    // max_length message for videos
    flm::d3::Prepared pp = flm::d3::prepare(flm::d3::parse_json(req), codes, nullptr, on);
    flm::d3::tokenize(pp, fake_encode, 50);
    const std::string m = pp.errors.count("q") ? pp.errors.at("q")["message"].get<std::string>() : std::string();
    CHECK("max_length text", m.find(" for 0 image(s) and 1 video(s)), over the maximum context length of 50 tokens; nothing was truncated") != std::string::npos, m);
}

// ---- 6. positions -----------------------------------------------------------------------------------------

/// The engine's image_positions (d3_npu.cpp), copied: the reference the video plan must equal when both pad ids count.
void engine_positions(const std::vector<int>& ids, int token, const std::vector<std::pair<int, int>>& grids, int* pos) {
    int cur = 0;
    size_t image = 0, i = 0;
    while (i < ids.size()) {
        if (ids[i] == token && image < grids.size()) {
            const int gh = grids[image].first / 2, gw = grids[image].second / 2;
            for (int r = 0; r < gh; r++)
                for (int c = 0; c < gw; c++, i++) {
                    pos[i * 3 + 0] = cur;
                    pos[i * 3 + 1] = cur + r;
                    pos[i * 3 + 2] = cur + c;
                }
            cur += std::max(gh, gw);
            image++;
        } else {
            pos[i * 3 + 0] = pos[i * 3 + 1] = pos[i * 3 + 2] = cur++;
            i++;
        }
    }
}

void test_positions(const json* golden) {
    // a prompt: text, an image (4x6 grid -> 2x3 = 6 tokens), text, a video of 2 groups (4x2 -> 2 tokens each), text
    const int IMG = 900, VID = 901;
    std::vector<int> ids = {1, 2, 3};
    for (int i = 0; i < 6; i++) ids.push_back(IMG);
    ids.push_back(4);
    ids.push_back(5);
    for (int g = 0; g < 2; g++) {
        ids.push_back(6);
        for (int i = 0; i < 4; i++) ids.push_back(VID);
        ids.push_back(7);
    }
    ids.push_back(8);
    const std::vector<std::pair<int, int>> groups = {{4, 6}, {4, 4}, {4, 4}};
    std::vector<int> pos;
    const int next = v::mrope_positions(ids, IMG, VID, groups, pos);
    // reference: treat both ids as one token for the engine copy
    std::vector<int> unified = ids;
    for (int& x : unified) if (x == VID) x = IMG;
    std::vector<int> ref(ids.size() * 3);
    engine_positions(unified, IMG, groups, ref.data());
    CHECK("mrope == engine formula", pos == ref, "positions differ from the engine's image_positions");
    // by hand: 3 text (0,1,2), image run at 3: (3,3+r,3+c) r<2,c<3, next = 3 + 3 = 6
    CHECK("mrope hand", pos[3 * 3 + 0] == 3 && pos[3 * 3 + 1] == 3 && pos[3 * 3 + 2] == 3 && pos[8 * 3 + 1] == 4 && pos[8 * 3 + 2] == 5 &&
                            pos[9 * 3] == 6 && pos[10 * 3] == 7, "image run positions and the resume point");
    CHECK("mrope next", next == pos[(ids.size() - 1) * 3] + 1, "next position");
    bool threw = false;
    try { v::mrope_positions(ids, IMG, VID, {{4, 6}}, pos); } catch (const v::VideoError&) { threw = true; }
    CHECK("mrope too many runs", threw, "more runs than grids");
    threw = false;
    try { v::mrope_positions(ids, IMG, VID, {{4, 6}, {4, 4}, {4, 4}, {2, 2}}, pos); } catch (const v::VideoError&) { threw = true; }
    CHECK("mrope too few runs", threw, "fewer runs than grids");
    if (golden) {
        // golden from transformers' get_rope_index (Qwen3.5), gen_fixtures.py
        for (const auto& g : (*golden)["cases"]) {
            const std::vector<int> gids = g["ids"].get<std::vector<int>>();
            std::vector<std::pair<int, int>> gr;
            for (const auto& x : g["groups"]) gr.emplace_back(x[0].get<int>(), x[1].get<int>());
            std::vector<int> got;
            const int nx = v::mrope_positions(gids, g["image_id"], g["video_id"], gr, got);
            const auto want = g["positions"].get<std::vector<int>>();   // [3][n], axis-major
            bool ok = got.size() == want.size();
            for (size_t i = 0; ok && i < gids.size(); i++)
                for (int a = 0; a < 3; a++) ok = ok && got[i * 3 + a] == want[a * gids.size() + i];
            CHECK("mrope golden " + g["name"].get<std::string>(), ok && nx == g["next"].get<int>(), "positions differ from get_rope_index");
        }
    }
}

// ---- 7. decoder backend vs OpenCV frames --------------------------------------------------------------------

void test_backend(const std::string& dir, const json& decode, const std::string& lib_dir) {
    auto backend = v::make_ffmpeg_backend(lib_dir);
    std::string why;
    if (!backend->available(&why)) {
        std::cout << "SKIP ffmpeg backend tests: " << why << "\n";
        skipped++;
        // the refusal is the 422 text
        v::VideoLoadOptions o;
        o.backend = backend;
        const std::string e = error_of(PyValue::string("data:video/mp4;base64," + base64("x")), o);
        CHECK("missing libs refusal", e.rfind("UNAVAILABLE: video inputs are not available: video decoding backend not available", 0) == 0, e);
        return;
    }
    for (const auto& d : decode) {
        const std::string file = d["file"];
        if (d.contains("error")) continue;
        bool ok = false;
        const std::string bytes = read_file(dir + "/" + file, &ok);
        CHECK("backend " + file + " read", ok, "fixture missing");
        if (!ok) continue;
        v::VideoClip c;
        try {
            c = v::decode_clip(*backend, bytes);
        } catch (const v::VideoError& e) {
            fail("backend " + file, e.what());
            continue;
        }
        CHECK("backend " + file + " meta", c.width == d["w"] && c.height == d["h"] && c.indices == d["indices"].get<std::vector<int>>() &&
                                              std::fabs(c.fps - d["fps"].get<double>()) < 1e-9 && c.total == d["total"].get<int>(),
              "size/indices/fps/total: n=" + std::to_string(c.n) + " total=" + std::to_string(c.total) + " fps=" + std::to_string(c.fps));
        int maxd = 0;
        long long nd = 0;
        if (d.contains("frames_file")) {
            const std::string gold = read_file(dir + "/" + d["frames_file"].get<std::string>());
            CHECK("backend " + file + " frames size", gold.size() == c.frames.size(), "frame bytes");
            for (size_t i = 0; i < std::min(gold.size(), c.frames.size()); i++) {
                const int diff = std::abs(int(c.frames[i]) - int(uint8_t(gold[i])));
                maxd = std::max(maxd, diff);
                nd += diff != 0;
            }
            std::printf("  backend %-10s vs OpenCV frames: max |diff| %d, %lld of %zu values differ\n", file.c_str(), maxd, nd, gold.size());
            CHECK("backend " + file + " pixels +-1 LSB", maxd <= 1, "max |diff| " + std::to_string(maxd));
        }
        {   // the frames' md5 as upstream's decode gives them (rotation included)
            const std::string m = md5_hex(c.frames.data(), c.frames.size());
            std::printf("  backend %-10s frames md5 %s\n", file.c_str(), m == d["frames_md5"].get<std::string>() ? "equal" : "differs (see the +-1 LSB budget)");
            if (!d.contains("frames_file")) CHECK("backend " + file + " frames md5", m == d["frames_md5"].get<std::string>(), "rotated clip differs from OpenCV's");
        }
        // the same clip as a data URL through load_video
        v::VideoLoadOptions o;
        o.backend = backend;
        const std::string sub = file.find(".mp4") != std::string::npos ? "mp4" : file.find(".webm") != std::string::npos ? "webm" :
                                file.find(".mkv") != std::string::npos ? "x-matroska" : file.find(".mov") != std::string::npos ? "quicktime" : "gif";
        const std::string url = (sub == "gif" ? "data:image/gif;base64," : "data:video/" + sub + ";base64,") + base64(bytes);
        CHECK("backend " + file + " data URL", error_of(PyValue::string(url), o).empty(), error_of(PyValue::string(url), o));
        // windows at rates 1 and 2 against the plan
        for (int rate : {1, 2}) {
            if (d.contains("rotation")) continue;
            const auto ws = v::decode_windows(*backend, bytes, rate);
            const auto plan = v::window_plan(c.total, c.fps, rate);
            bool same = ws.size() == plan.size();
            for (size_t i = 0; same && i < plan.size(); i++) same = ws[i].indices == plan[i];
            CHECK("backend " + file + " windows rate " + std::to_string(rate), same, "window frames");
            // against upstream's _decode_windows: indices, fps, total and the frames themselves
            const auto& g = d["windows"][std::to_string(rate)];
            bool same_up = g.is_array() && g.size() == ws.size();
            for (size_t i = 0; same_up && i < ws.size(); i++) {
                same_up = ws[i].indices == g[i]["indices"].get<std::vector<int>>() && ws[i].fps == g[i]["fps"].get<double>() &&
                          ws[i].total == g[i]["total"].get<int>() &&
                          (md5_hex(ws[i].frames.data(), ws[i].frames.size()) == g[i]["md5"].get<std::string>() || maxd > 0);
            }
            CHECK("backend " + file + " windows vs upstream rate " + std::to_string(rate), same_up, "decode_windows differs from _decode_windows");
        }
    }
    // the whole request path with the real decoder: data URL -> prepare -> grid and texts
    {
        bool ok2 = false;
        const std::string bytes = read_file(dir + "/h264.mp4", &ok2);
        if (ok2) {
            v::VideoLoadOptions lo;
            lo.backend = backend;
            flm::d3::PrepareOptions po;
            po.load_video = v::make_video_loader(lo);
            std::vector<std::string> codes;
            for (int i = 0; i < 255; i++) codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string(1, 'A' + (i - 26) / 26) + std::string(1, 'A' + (i - 26) % 26));
            const std::string body = R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"videos":["data:video/mp4;base64,)" + base64(bytes) + R"("]})";
            flm::d3::Prepared q = flm::d3::prepare(flm::d3::parse_json(body), codes, nullptr, po);
            CHECK("e2e prepare", q.videos.size() == 1 && q.videos[0]->n == 8 && q.video_grids[0].grid_t == 4 && q.video_texts[0].find("<0.3 seconds><|vision_start|><|video_pad|>") == 0,
                  "mp4 through prepare");
            flm::d3::tokenize(q, fake_encode, 0);
            CHECK("e2e tokenize", q.sequences.count("q") == 1, "tokenized");
        }
    }
    // corrupt data
    std::string err;
    auto r = backend->open(std::string("this is not a video, just some text padded out to a few hundred bytes. ") + std::string(300, 'x'), &err);
    CHECK("backend rejects garbage", r == nullptr && !err.empty(), "garbage opened");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_d3_video <fixtures dir> [<big fixtures dir>] [<ffmpeg lib dir>]\n";
        return 2;
    }
    const std::string dir = argv[1];
    bool ok = false;
    const std::string text = read_file(dir + "/cases.json", &ok);
    if (!ok) {
        std::cerr << "no " << dir << "/cases.json: run gen_fixtures.py first\n";
        return 2;
    }
    const json doc = json::parse(text);
    test_helpers();
    test_request_values();
    test_contract();
    test_tables(doc["tables"]);
    CaseStats stats;
    for (const auto& c : doc["cases"]) test_case(dir, c, doc["max_video_tokens"], stats);
    std::printf("resize vs torchvision over %lld values: max |diff| %d, %lld differ\n", stats.px, stats.maxdiff, stats.diff1);
    {
        bool have = false;
        const std::string g = read_file(dir + "/mrope.json", &have);
        json gj;
        if (have) gj = json::parse(g);
        test_positions(have ? &gj : nullptr);
    }
    // the request limit table from the real processor
    {
        const auto& l = doc["limit"];
        const auto g = v::plan_video_grid(l["frames"], l["h"], l["w"]);
        CHECK("limit per video", g.tokens() == l["per_video_tokens"].get<long long>(), "tokens of a 32-frame 720p video");
    }
    std::string lib_dir = argc > 3 ? argv[3] : "";
    test_backend(dir, doc["decode"], lib_dir);
    if (argc > 2 && std::string(argv[2]) != "-") {
        const std::string big = argv[2];
        bool have = false;
        const std::string bt = read_file(big + "/cases.json", &have);
        if (have) {
            const json bd = json::parse(bt);
            CaseStats bs;
            for (const auto& c : bd["cases"]) test_case(big, c, bd["max_video_tokens"], bs);
            std::printf("big set: resize vs torchvision over %lld values: max |diff| %d, %lld differ\n", bs.px, bs.maxdiff, bs.diff1);
            if (bd.contains("hummingbird")) {   // the shipped example clip: sampling, OpenCV's frames, the plan
                const auto& hb = bd["hummingbird"];
                bool ex = false;
                const std::string bytes = read_file(hb["file"].get<std::string>(), &ex);
                auto backend = v::make_ffmpeg_backend(lib_dir);
                std::string why;
                if (ex && backend->available(&why)) {
                    const v::VideoClip c = v::decode_clip(*backend, bytes);
                    CHECK("hummingbird meta", c.indices == hb["indices"].get<std::vector<int>>() && std::fabs(c.fps - hb["fps"].get<double>()) < 1e-9 &&
                                                  c.total == hb["total"].get<int>() && c.width == hb["w"] && c.height == hb["h"],
                          "frames " + std::to_string(c.n) + " total " + std::to_string(c.total));
                    const std::string gold = read_file(big + "/hummingbird.frames.u8");
                    int md = 0;
                    long long nd = 0;
                    for (size_t i = 0; i < std::min(gold.size(), c.frames.size()); i++) {
                        const int d = std::abs(int(c.frames[i]) - int(uint8_t(gold[i])));
                        md = std::max(md, d);
                        nd += d != 0;
                    }
                    std::printf("hummingbird: %d frames of %dx%d, total %d at %.3f fps; vs OpenCV max |diff| %d, %lld of %zu differ\n", c.n, c.width,
                                c.height, c.total, c.fps, md, nd, gold.size());
                    CHECK("hummingbird pixels", gold.size() == c.frames.size() && md <= 1, "decoded frames vs OpenCV");
                    const v::VideoGrid g = v::plan_video_grid(c.n, c.height, c.width);
                    std::printf("hummingbird processor grid: %d x %d x %d (%lld tokens)\n", g.grid_t, g.grid_h, g.grid_w, g.tokens());
                    for (int rate : {1, 2}) {
                        const auto ws = v::decode_windows(*backend, bytes, rate);
                        const auto plan = v::window_plan(c.total, c.fps, rate);
                        size_t frames = 0;
                        for (const auto& w : ws) frames += w.indices.size();
                        bool same = ws.size() == plan.size();
                        for (size_t i = 0; same && i < plan.size(); i++) same = ws[i].indices == plan[i];
                        std::printf("hummingbird track rate %d: %zu window(s), %zu frames, %zu moments\n", rate, ws.size(), frames,
                                    v::moments_of(plan, c.fps).size());
                        CHECK("hummingbird windows rate " + std::to_string(rate), same, "windows");
                    }
                } else {
                    std::cout << "SKIP hummingbird: " << (ex ? why : "clip missing") << "\n";
                    skipped++;
                }
            }
        } else {
            std::cout << "SKIP big fixtures: " << big << "/cases.json missing\n";
            skipped++;
        }
    }
    std::printf("%d checks, %d failures, %d skipped sections\n", checks, failures, skipped);
    return failures ? 1 : 0;
}
