/// \file test_d3_adapter.cpp
/// \brief The engine adapter: its routing by capability bits, the d3_state_request it builds (question images, video pads) and the
///        per-row (M1) path against a mock `decision_model`: which prompts and payloads reach
///        `encode_batch` for blocks with and without own images, and what it refuses. CPU only. The state-once path
///        (`d3_npu::encode_state_once`) needs the real library and is only compile-checked here.
#include <iostream>

#include "AutoDecisionModel/d3_engine_adapter.hpp"
#include "AutoDecisionModel/d3_engine_adapter_call.hpp"
#include "decision/d3_engine.hpp"
#include "models/d3/flm/aie2p/d3_npu.hpp"

// The device runtime is not linked: buffers here live on the host, so the out-of-line destructor of a device
// buffer object (never created) only has to exist.
#if __has_include("xrt/xrt_bo.h")
namespace xrt { bo::~bo() = default; }
#endif

namespace {
int failures = 0, checks = 0;
void expect(bool ok, const char* what) {
    checks++;
    if (!ok) { failures++; std::cerr << "FAIL " << what << "\n"; }
}

struct Call {
    std::vector<std::vector<int>> ids;
    std::vector<int> state_len;
    std::vector<std::pair<int, int>> grids;
    size_t pixels = 0;
};

class MockEngine : public decision_model {
public:
    std::vector<Call> calls;
    void load_weights(Q4NX&) override {}
    std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, void* payload) override {
        Call c;
        c.ids = ids;
        auto* p = static_cast<d3_image_payload*>(payload);
        c.state_len = p->state_len;
        for (const auto& im : p->images) c.grids.emplace_back(im.grid_h, im.grid_w);
        c.pixels = p->pixels.size();
        calls.push_back(c);
        std::vector<buffer<bf16>> out;
        for (const auto& row : ids) {
            buffer<bf16> h(4);
            for (int d = 0; d < 4; d++) h.data()[d] = bf16(static_cast<float>(row.back() + d));   // last token id + d
            out.push_back(std::move(h));
        }
        return out;
    }
    buffer<bf16> encode(const std::vector<int>&, void*) override { return buffer<bf16>(4); }
    int max_length() override { return 32768; }
    int hidden_size() override { return 4; }
};
}  // namespace

int main() {
    MockEngine mock;
    auto adapter = make_d3_engine_adapter(&mock, nullptr, 0, true);
    flm::d3::EngineStateRequest req;
    req.image_token_id = 99;
    req.state_ids = {1, 2, 3};
    req.state_image_ids = {0};
    flm::d3::EngineImage a, b;
    a.grid_h = a.grid_w = 2; a.pixels.assign(4 * 1536, 1);
    b.grid_h = 2; b.grid_w = 4; b.pixels.assign(8 * 1536, 2);
    req.images = {a, b};
    flm::d3::EngineBlock q0{"q0", {10, 11}, {0}, {}};
    flm::d3::EngineBlock q1{"q1", {20, 99}, {0}, {1}};
    flm::d3::EngineBlock q2{"q2", {30, 31}, {0}, {}};
    req.blocks = {q0, q1, q2};
    const auto rows = adapter->encode_state_once(req);
    expect(rows.size() == 3 && rows[0].size() == 4 && rows[0][0] == 11 && rows[1][0] == 99 && rows[2][0] == 31, "rows in block order");
    // two batches: blocks without own images share the state image only; the block with an image has its own payload
    expect(mock.calls.size() == 2, "one encode_batch per distinct image set");
    bool plain = false, with_image = false;
    for (const auto& c : mock.calls) {
        if (c.ids.size() == 2 && c.grids.size() == 1 && c.grids[0] == std::make_pair(2, 2) && c.state_len == std::vector<int>{3, 3} &&
            c.ids[0] == std::vector<int>{1, 2, 3, 10, 11} && c.ids[1] == std::vector<int>{1, 2, 3, 30, 31} && c.pixels == 4 * 1536)
            plain = true;
        if (c.ids.size() == 1 && c.grids.size() == 2 && c.grids[1] == std::make_pair(2, 4) && c.state_len == std::vector<int>{3} && c.pixels == 12 * 1536)
            with_image = true;
    }
    expect(plain && with_image, "payloads: state image first, then the block's images; state_len = |state|");
    // a Box block (four readout rows) is refused by the per-row engine, with the reason in the message
    req.blocks = {flm::d3::EngineBlock{"box", {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}, {9, 6, 3, 0}, {}}};
    std::string what;
    try { adapter->encode_state_once(req); } catch (const std::runtime_error& e) { what = e.what(); }
    expect(what.find("Box question reads four") != std::string::npos, "box refused without state-once");
    // ---- routing by capability bits (not the revision): what one state-once call takes, what falls back to the per-row path
    {
        flm::d3::EngineStateRequest r;
        r.state_ids = {1, 2, 3};
        flm::d3::EngineImage im;
        im.grid_h = im.grid_w = 2;
        r.images = {im, im};
        r.blocks = {flm::d3::EngineBlock{"plain", {10, 11}, {0}, {}},
                    flm::d3::EngineBlock{"img", {20, 99}, {0}, {1}},
                    flm::d3::EngineBlock{"box", {30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, {9, 6, 3, 0}, {}},
                    flm::d3::EngineBlock{"box_img", {40, 99, 41, 42, 43, 44, 45, 46, 47, 48, 49}, {9, 6, 3, 0}, {1}}};
        const auto route = [&](unsigned caps, bool once) {
            std::vector<size_t> fused, per_row;
            route_d3_blocks(r, caps, once, fused, per_row);
            return std::make_pair(fused, per_row);
        };
        using V = std::vector<size_t>;
        expect(route(15, true) == std::make_pair(V{0, 1, 2, 3}, V{}), "caps 15: everything in one call");
        expect(route(3, true) == std::make_pair(V{0, 2}, V{1, 3}), "caps 3 (revision 2 as frozen): blocks with images fall back");
        expect(route(1, true) == std::make_pair(V{0}, V{1, 2, 3}), "state-once without readout rows: boxes fall back (and are refused)");
        expect(route(7, true) == std::make_pair(V{0, 1, 2, 3}, V{}), "question images without the video bit");
        expect(route(15, false) == std::make_pair(V{}, V{0, 1, 2, 3}), "D3_STATE_ONCE=0: all per row");
        expect(route(0, true) == std::make_pair(V{}, V{0, 1, 2, 3}), "no capabilities (a revision 1 library)");
    }
    // ---- the d3_state_request of a call: question image table, video pads
    {
        flm::d3::EngineStateRequest r;
        r.image_token_id = 99;
        r.video_token_id = 98;
        r.state_ids = {1, 98, 98, 2};
        r.state_image_ids = {0};
        flm::d3::EngineImage st, a, b;
        st.grid_h = 2; st.grid_w = 4; st.pixels.assign(8 * 1536, 5);
        a.grid_h = a.grid_w = 2; a.pixels.assign(4 * 1536, 6);
        b.grid_h = 2; b.grid_w = 2; b.pixels.assign(4 * 1536, 7);
        r.images = {st, a, b};
        r.blocks = {flm::d3::EngineBlock{"x", {10, 99, 11, 99, 12}, {0}, {1, 1}},   // the same table image twice
                    flm::d3::EngineBlock{"y", {20, 99, 21}, {0}, {2}},
                    flm::d3::EngineBlock{"z", {30, 31}, {1, 0}, {}}};
        auto call = make_d3_fused_call(r, {0, 1, 2}, 15, 248056);
        expect(call->call.questions.size() == 3 && call->call.questions[0].image_ids == std::vector<int>{0, 0} &&
                   call->call.questions[1].image_ids == std::vector<int>{1} && call->call.questions[2].image_ids.empty(),
               "question_images table: one entry per distinct image, image_ids in pad-run order");
        expect(call->call.question_images.images.size() == 2 && call->call.question_images.pixels.size() == 8 * 1536 &&
                   call->call.question_images.image_token_id == 99,
               "question_images payload");
        expect(call->call.media == &call->media && call->media.images.size() == 1 && call->media.images[0].grid_w == 4 && call->media.image_token_id == 99,
               "the state's images are the media");
        expect(call->call.video_token_id == 98 && call->call.state_ids == r.state_ids, "D3_CAP_VIDEO: the video pad id goes to the engine");
        expect(call->call.questions[2].readout_from_end == std::vector<int>{1, 0} && call->call.questions[0].ids == r.blocks[0].ids, "readout rows and ids kept");
        auto old = make_d3_fused_call(r, {2, 0}, 3, 248056);   // an engine without D3_CAP_VIDEO: pads become image pads
        expect(old->call.video_token_id == -1 && old->call.state_ids == std::vector<int>{1, 99, 99, 2} && old->call.questions.size() == 2 &&
                   old->call.questions[0].ids == r.blocks[2].ids && old->call.questions[1].image_ids == std::vector<int>{0, 0},
               "without D3_CAP_VIDEO the video pad ids are rewritten to the image pad id; blocks in the order asked");
        // no state images: no media; the default image id when the request has none
        flm::d3::EngineStateRequest t;
        t.state_ids = {1, 2};
        t.blocks = {flm::d3::EngineBlock{"w", {3}, {0}, {}}};
        auto plain = make_d3_fused_call(t, {0}, 15, 248056);
        expect(plain->call.media == nullptr && plain->call.video_token_id == -1 && plain->call.question_images.image_token_id == 248056, "text only call");
    }
    // ---- the per-row path knows image runs only: a video request is rewritten to image pads
    {
        MockEngine mock2;
        auto adapter2 = make_d3_engine_adapter(&mock2, nullptr, 0, true);
        flm::d3::EngineStateRequest r;
        r.image_token_id = 99;
        r.video_token_id = 98;
        r.state_ids = {1, 98, 98, 2};
        r.state_image_ids = {0, 1};
        flm::d3::EngineImage g;
        g.grid_h = g.grid_w = 2;
        g.pixels.assign(4 * 1536, 1);
        r.images = {g, g};
        r.blocks = {flm::d3::EngineBlock{"q", {10, 11}, {0}, {}}};
        adapter2->encode_state_once(r);
        expect(mock2.calls.size() == 1 && mock2.calls[0].ids[0] == std::vector<int>{1, 99, 99, 2, 10, 11} && mock2.calls[0].grids.size() == 2 &&
                   mock2.calls[0].state_len == std::vector<int>{4},
               "per-row path: video pad ids become image pad ids, the groups are image entries");
    }
    expect(replace_token({1, 2, 1, 3}, 1, 9) == std::vector<int>{9, 2, 9, 3}, "replace_token");
    std::cout << "d3 adapter: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
