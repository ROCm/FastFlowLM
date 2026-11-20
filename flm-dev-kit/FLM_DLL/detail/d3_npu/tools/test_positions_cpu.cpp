// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Host-only test of the M-RoPE position planner (d3_positions.hpp) against transformers' Qwen3.5 get_rope_index.
//
//   test_positions_cpu [fixtures dir]      (default: tools/fixtures next to this file, or ./tools/fixtures)
//
// 1. mrope_video_agent.json: five synthetic prompts (images + videos, portrait grids) with the FULL positions of get_rope_index
//    (made by the FLM video agent's gen_fixtures.py): the planner must reproduce every position, with both pad ids as runs.
// 2. positions_goldens.json (gen_position_fixtures.py): 327 real d3-turbo prompts of the M3 / M4 golden sets: the visual runs'
//    first / last positions, the position after the last token, the text resume points, and the state-once split: the state
//    [0, s) planned alone (its own grids), the question block planned from the state's next position with ITS grids (images in
//    question blocks, start position) must equal the one-pass plan of the whole row, token by token.
// 3. hand cases: start position, several images per block, adjacent runs, portrait grids, a video's temporal groups, run checks.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "d3_positions.hpp"
#include "nlohmann/json.hpp"

struct img { int grid_h, grid_w; };

static int failures = 0, checks = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static nlohmann::json load(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::printf("cannot open %s\n", path.c_str()); std::exit(2); }
    return nlohmann::json::parse(f);
}

static void test_agent_fixtures(const std::string& dir) {
    const auto j = load(dir + "/mrope_video_agent.json");
    for (const auto& c : j["cases"]) {
        const std::vector<int> ids = c["ids"].get<std::vector<int>>();
        std::vector<img> g;
        for (const auto& x : c["groups"]) g.push_back({x[0].get<int>(), x[1].get<int>()});
        const int image_id = c["image_id"], video_id = c["video_id"];
        std::vector<int> pos(ids.size() * 3, -1);
        size_t used = 0;
        const int next = d3_plan_positions(ids, image_id, video_id, g.data(), g.size(), 0, pos.data(), &used);
        const auto want = c["positions"].get<std::vector<int>>();   // [3][n]
        bool ok = want.size() == pos.size();
        for (size_t i = 0; ok && i < ids.size(); i++)
            for (int a = 0; a < 3; a++) ok = ok && pos[i * 3 + a] == want[a * ids.size() + i];
        CHECK(ok && next == c["next"].get<int>() && used == g.size(), "agent fixture %s", c["name"].get<std::string>().c_str());
        d3_check_visual_runs(ids, image_id, video_id, g.data(), g.size(), "a prompt");
        // the same prompt as two pieces: split before the last text tokens' region (at the middle of the text before the first run)
        for (size_t cut : {ids.size() / 3, ids.size() / 2, ids.size() - 1}) {
            // a cut inside a run is not a state / block boundary; move it to the end of the run
            size_t k = cut;
            while (k < ids.size() && k > 0 && (ids[k] == image_id || ids[k] == video_id) && (ids[k - 1] == image_id || ids[k - 1] == video_id)) k++;
            std::vector<int> a(ids.begin(), ids.begin() + k), b(ids.begin() + k, ids.end());
            // grids of the runs in each part
            std::vector<img> ga, gb;
            size_t gi = 0;
            for (size_t i = 0; i < ids.size(); i++) {
                if (ids[i] == image_id || ids[i] == video_id) {
                    const size_t n = (size_t)g[gi].grid_h * g[gi].grid_w / 4;
                    (i < k ? ga : gb).push_back(g[gi]);
                    i += n - 1;
                    gi++;
                }
            }
            std::vector<int> pa(a.size() * 3), pb(b.size() * 3);
            const int mid = d3_plan_positions(a, image_id, video_id, ga.data(), ga.size(), 0, pa.data());
            const int end = d3_plan_positions(b, image_id, video_id, gb.data(), gb.size(), mid, pb.data());
            bool same = end == next;
            for (size_t i = 0; same && i < a.size() * 3; i++) same = pa[i] == pos[i];
            for (size_t i = 0; same && i < b.size() * 3; i++) same = pb[i] == pos[a.size() * 3 + i];
            CHECK(same, "agent fixture %s split at %zu", c["name"].get<std::string>().c_str(), k);
        }
    }
}

static void test_golden_rows(const std::string& dir) {
    const auto j = load(dir + "/positions_goldens.json");
    const int IMG = j["image_id"], VID = j["video_id"];
    size_t rows = 0, with_qimg = 0, with_video = 0, multi_qimg = 0;
    for (const auto& r : j["rows"]) {
        const int s = r["s"], n = r["n"];
        // rebuild the ids: text token 1 in the gaps, pads in the runs
        std::vector<int> ids;
        std::vector<img> grids, state_grids, block_grids;
        const auto& runs = r["runs"];
        const auto gaps = r["gaps"].get<std::vector<int>>();
        for (size_t k = 0; k < runs.size(); k++) {
            for (int t = 0; t < gaps[k]; t++) ids.push_back(1);
            const int gh = runs[k][2], gw = runs[k][3];
            for (int t = 0; t < gh * gw / 4; t++) ids.push_back(runs[k][1]);
            grids.push_back({gh, gw});
            (runs[k][0].get<int>() < s ? state_grids : block_grids).push_back({gh, gw});
        }
        for (int t = 0; t < gaps.back(); t++) ids.push_back(1);
        CHECK((int)ids.size() == n, "%s: rebuilt length %zu vs %d", r["id"].get<std::string>().c_str(), ids.size(), n);
        d3_check_visual_runs(ids, IMG, VID, grids.data(), grids.size(), "a prompt");
        std::vector<int> pos(ids.size() * 3);
        const int next = d3_plan_positions(ids, IMG, VID, grids.data(), grids.size(), 0, pos.data());
        bool ok = next == r["next"].get<int>();
        for (const auto& run : runs) {
            const size_t first = run[0], len = (size_t)run[2].get<int>() * run[3].get<int>() / 4;
            for (int a = 0; a < 3; a++)
                ok = ok && pos[first * 3 + a] == run[4][a].get<int>() && pos[(first + len - 1) * 3 + a] == run[5][a].get<int>();
        }
        const auto ta = r["text_at"].get<std::vector<int>>();
        const size_t at[4] = {0, (size_t)s - 1, (size_t)s, (size_t)n - 1};
        for (int k = 0; k < 4; k++) ok = ok && pos[at[k] * 3] == ta[k] && pos[at[k] * 3 + 1] == ta[k] && pos[at[k] * 3 + 2] == ta[k];
        CHECK(ok, "%s/%s#%s: one-pass plan differs from get_rope_index", r["set"].get<std::string>().c_str(), r["id"].get<std::string>().c_str(),
              r["key"].get<std::string>().c_str());
        // state-once: the state alone, then the block from the state's next position with its own grids
        std::vector<int> sid(ids.begin(), ids.begin() + s), bid(ids.begin() + s, ids.end());
        std::vector<int> ps(sid.size() * 3), pbk(bid.size() * 3);
        size_t used_s = 0, used_b = 0;
        const int mid = d3_plan_positions(sid, IMG, VID, state_grids.data(), state_grids.size(), 0, ps.data(), &used_s);
        const int end = d3_plan_positions(bid, IMG, VID, block_grids.data(), block_grids.size(), mid, pbk.data(), &used_b);
        bool same = end == next && used_s == state_grids.size() && used_b == block_grids.size();
        for (size_t i = 0; same && i < ps.size(); i++) same = ps[i] == pos[i];
        for (size_t i = 0; same && i < pbk.size(); i++) same = pbk[i] == pos[ps.size() + i];
        CHECK(same, "%s/%s: split plan (state, block from its next position) differs from the one-pass plan", r["set"].get<std::string>().c_str(),
              r["id"].get<std::string>().c_str());
        d3_check_visual_runs(bid, IMG, VID, block_grids.data(), block_grids.size(), "a question block");
        rows++;
        with_qimg += !block_grids.empty();
        multi_qimg += block_grids.size() > 1;
        with_video += r["set"].get<std::string>().find("video") != std::string::npos || r["set"] == "track";
    }
    std::printf("golden rows: %zu (%zu with images in the question block, %zu of them with several, %zu video rows)\n", rows, with_qimg, multi_qimg, with_video);
    CHECK(rows > 100 && with_qimg > 0 && with_video > 0, "fixture coverage");
}

static bool throws_adj(const std::vector<int>& v, const img* gs) {
    try { d3_check_visual_runs(v, 9, 8, gs, 2, "a prompt"); } catch (const std::invalid_argument&) { return true; }
    return false;
}

static void test_hand() {
    const int IMG = 9, VID = 8;
    // start position: a block of text, an image (4 x 6 grid -> 2 x 3 tokens), text, a portrait image (6 x 4 -> 3 x 2), text
    std::vector<int> ids = {1, 1};
    for (int i = 0; i < 6; i++) ids.push_back(IMG);
    ids.push_back(1);
    for (int i = 0; i < 6; i++) ids.push_back(IMG);
    ids.push_back(1);
    const img g[2] = {{4, 6}, {6, 4}};
    std::vector<int> pos(ids.size() * 3);
    const int next = d3_plan_positions(ids, IMG, VID, g, 2, 100, pos.data());
    // text 100, 101; image 1 at s = 102: (102, 102 + r, 102 + c), resume at 102 + 3 = 105; text 105; image 2 at 106: resume at 106 + 3 = 109
    CHECK(pos[0] == 100 && pos[3] == 101 && pos[6] == 102 && pos[7] == 102 && pos[8] == 102, "start position text / first run token");
    CHECK(pos[(2 + 5) * 3 + 1] == 103 && pos[(2 + 5) * 3 + 2] == 104, "run last token (row 1, col 2)");
    CHECK(pos[8 * 3] == 105 && pos[9 * 3] == 106 && pos[(9 + 5) * 3 + 1] == 108 && pos[(9 + 5) * 3 + 2] == 107, "second (portrait) run");
    CHECK(pos[15 * 3] == 109 && next == 110, "resume after the portrait run uses max(h, w) / 2: next %d", next);
    // adjacent runs of different pad ids (an image directly followed by a video group): told apart by the grids' lengths
    std::vector<int> adj;
    for (int i = 0; i < 4; i++) adj.push_back(IMG);
    for (int i = 0; i < 4; i++) adj.push_back(VID);
    const img ga[2] = {{4, 4}, {4, 4}};
    std::vector<int> pa(adj.size() * 3);
    size_t used = 0;
    const int nx = d3_plan_positions(adj, IMG, VID, ga, 2, 0, pa.data(), &used);
    CHECK(used == 2 && nx == 4 && pa[4 * 3] == 2 && pa[7 * 3 + 1] == 3 && pa[7 * 3 + 2] == 3, "adjacent image and video runs");
    CHECK(throws_adj(adj, ga), "adjacent runs are refused by the run check");
    // run checks
    auto throws = [&](const std::vector<int>& v, const img* gs, size_t n, int vid) {
        try { d3_check_visual_runs(v, IMG, vid, gs, n, "a prompt"); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(throws(std::vector<int>{9, 9, 9, 9, 1, 8, 8, 8, 8}, ga, 1, VID), "more runs than images");
    CHECK(throws(std::vector<int>{9, 9, 9, 9, 1, 8, 8, 8, 8}, ga, 2, -1), "video id off: the VID tokens are text, one run for two grids");
    CHECK(throws(std::vector<int>{9, 9, 9, 1}, ga, 1, VID), "short run");
    CHECK(throws(std::vector<int>{9, 9, 9, 9, 1, 8, 8, 8, 8}, ga, 3, VID), "fewer runs than images");
    std::vector<int> longer(5, IMG);
    CHECK(throws(longer, ga, 1, VID), "long run");
    // video temporal groups: three groups of an 8 x 6 grid (4 x 3 = 12 tokens each) with a text token in between; all groups at t advance by 4
    std::vector<int> vid;
    for (int gidx = 0; gidx < 3; gidx++) {
        vid.push_back(1);
        for (int i = 0; i < 12; i++) vid.push_back(VID);
        vid.push_back(1);
    }
    const img gv[3] = {{8, 6}, {8, 6}, {8, 6}};
    std::vector<int> pv(vid.size() * 3);
    const int nv = d3_plan_positions(vid, IMG, VID, gv, 3, 0, pv.data());
    // text 0; group 1 at 1 (resume 1 + 4 = 5); text 5; text 6 (the next group's leading token); group 2 at 7 -> 11; ...
    CHECK(pv[1 * 3] == 1 && pv[13 * 3] == 5 && pv[14 * 3] == 6 && pv[15 * 3] == 7 && pv[(15 + 11) * 3 + 1] == 7 + 3 && pv[(15 + 11) * 3 + 2] == 7 + 2, "video groups");
    CHECK(nv == 18 && pv[28 * 3] == 12 && pv[29 * 3] == 13 && pv[41 * 3] == 17, "video next %d", nv);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tools/fixtures";
    {
        std::ifstream f(dir + "/mrope_video_agent.json");
        if (!f) dir = std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/')) + "/fixtures";
    }
    test_hand();
    test_agent_fixtures(dir);
    test_golden_rows(dir);
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
