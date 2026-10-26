/// \file d3_engine_adapter.cpp
/// \brief See d3_engine_adapter.hpp.
#include "AutoDecisionModel/d3_engine_adapter.hpp"

#include <cstring>
#include <map>
#include <stdexcept>

#include "AutoDecisionModel/d3_engine_adapter_call.hpp"   // + models/d3/flm/aie2p/d3_npu.hpp: d3_image_payload, d3_state_request

namespace {

using flm::d3::EngineBlock;
using flm::d3::EngineImage;
using flm::d3::EngineStateRequest;

void append_image(d3_image_payload& payload, const EngineImage& im) {
    payload.images.push_back({im.grid_h, im.grid_w});
    const size_t at = payload.pixels.size();
    payload.pixels.resize(at + im.pixels.size());
    static_assert(sizeof(bf16) == sizeof(uint16_t), "bf16 is 16 bits");
    std::memcpy(payload.pixels.data() + at, im.pixels.data(), im.pixels.size() * sizeof(uint16_t));
}

class D3EngineAdapter : public flm::d3::ReadoutEngine {
public:
    D3EngineAdapter(decision_model* engine, d3_npu* npu, unsigned caps, bool state_once, int image_token_id)
        : engine_(engine), npu_(npu), caps_(caps), state_once_(state_once), image_token_id_(image_token_id) {}

    int hidden_size() const override { return engine_->hidden_size(); }

    std::vector<std::vector<float>> encode_state_once(const EngineStateRequest& request) override {
        std::vector<std::vector<float>> out(request.blocks.size());
        std::vector<size_t> fused, per_row;
#if defined(FLM_HAS_D3_NPU) && defined(FLM_D3_NPU_REV2)
        const bool once = npu_ != nullptr && state_once_;
#else
        const bool once = false;
        (void)npu_; (void)state_once_;
#endif
        route_d3_blocks(request, caps_, once, fused, per_row);
        if (!fused.empty()) run_fused(request, fused, out);
        if (!per_row.empty()) run_per_row(request, per_row, out);
        return out;
    }

private:
    /// d3_npu::encode_state_once: the state once, the blocks as its questions.
    void run_fused(const EngineStateRequest& request, const std::vector<size_t>& blocks, std::vector<std::vector<float>>& out) {
#if defined(FLM_HAS_D3_NPU) && defined(FLM_D3_NPU_REV2)
        const std::unique_ptr<D3FusedCall> fused = make_d3_fused_call(request, blocks, caps_, image_token_id_);
        // moved, never copied: XRT buffer copies do not own their memory
        std::vector<buffer<bf16>> rows = npu_->encode_state_once(fused->call);
        if (rows.size() != blocks.size()) throw std::runtime_error("encode_state_once returned a different number of questions");
        const int hidden = engine_->hidden_size();
        for (size_t i = 0; i < blocks.size(); i++) {
            const size_t n = request.blocks[blocks[i]].readout_from_end.size();
            std::vector<float> h(n * hidden);
            const bf16* row = rows[i].data();
            for (size_t d = 0; d < h.size(); d++) h[d] = static_cast<float>(row[d]);
            out[blocks[i]] = std::move(h);
        }
#else
        (void)request; (void)blocks; (void)out;
        throw std::runtime_error("this FastFlowLM build has no state-once d3 engine");
#endif
    }

    /// M1: every block is a prompt [state | block] with state_len = |state|; one hidden row (the last token) comes back
    /// per prompt. Blocks that read other positions need the engine's multi-position output.
    void run_per_row(const EngineStateRequest& request, const std::vector<size_t>& blocks, std::vector<std::vector<float>>& out) {
        for (size_t b : blocks) {
            const EngineBlock& blk = request.blocks[b];
            if (blk.readout_from_end != std::vector<int>{0}) {
                throw std::runtime_error(
                    !blk.image_ids.empty()
                        ? "a Box question with images in its own text needs the engine's question-image support "
                          "(D3_CAP_QUESTION_IMAGES), which this d3 engine does not have yet"
                        : "this d3 engine run returns the last hidden row of a prompt only; a Box question reads four "
                          "(it needs the state-once engine, D3_CAP_STATE_ONCE; check D3_STATE_ONCE)");
            }
        }
        // prompts that carry the same images share one batch: the engine's payload lists the images of the whole batch
        std::map<std::vector<int>, std::vector<size_t>> by_images;
        for (size_t b : blocks) by_images[request.blocks[b].image_ids].push_back(b);
        const int hidden = engine_->hidden_size();
        for (const auto& group : by_images) {
            d3_image_payload payload;
            payload.image_token_id = request.image_token_id >= 0 ? request.image_token_id : image_token_id_;
            std::vector<int> order = request.state_image_ids;
            order.insert(order.end(), group.first.begin(), group.first.end());
            for (int idx : order) append_image(payload, request.images.at(idx));
            std::vector<std::vector<int>> ids;
            for (size_t b : group.second) {
                std::vector<int> row = request.state_ids;
                row.insert(row.end(), request.blocks[b].ids.begin(), request.blocks[b].ids.end());
                // the M1 engine knows image runs only: a video's groups are image runs (see make_d3_fused_call)
                if (request.video_token_id >= 0) row = replace_token(std::move(row), request.video_token_id, payload.image_token_id);
                ids.push_back(std::move(row));
                payload.state_len.push_back(static_cast<int>(request.state_ids.size()));
            }
            std::vector<buffer<bf16>> rows = engine_->encode_batch(ids, &payload);
            for (size_t i = 0; i < group.second.size(); i++) {
                std::vector<float> h(hidden);
                const bf16* row = rows[i].data();
                for (int d = 0; d < hidden; d++) h[d] = static_cast<float>(row[d]);
                out[group.second[i]] = std::move(h);
            }
        }
    }

    decision_model* engine_;
    d3_npu* npu_;
    unsigned caps_;
    bool state_once_;
    int image_token_id_;
};

}  // namespace

void route_d3_blocks(const EngineStateRequest& request, unsigned capabilities, bool state_once, std::vector<size_t>& fused,
                     std::vector<size_t>& per_row) {
    const bool once = state_once && (capabilities & D3_CAP_STATE_ONCE) != 0;
    const bool question_images = (capabilities & D3_CAP_QUESTION_IMAGES) != 0;
    const bool rows = (capabilities & D3_CAP_READOUT_ROWS) != 0;
    for (size_t b = 0; b < request.blocks.size(); b++) {
        const bool has_images = !request.blocks[b].image_ids.empty();
        const bool many_rows = request.blocks[b].readout_from_end != std::vector<int>{0};
        (once && (!has_images || question_images) && (!many_rows || rows) ? fused : per_row).push_back(b);
    }
}

std::vector<int> replace_token(std::vector<int> ids, int from, int to) {
    for (int& id : ids) {
        if (id == from) id = to;
    }
    return ids;
}

std::unique_ptr<D3FusedCall> make_d3_fused_call(const EngineStateRequest& request, const std::vector<size_t>& blocks,
                                                unsigned capabilities, int default_image_token_id) {
    auto fused = std::make_unique<D3FusedCall>();
    d3_image_payload& media = fused->media;
    d3_state_request& call = fused->call;
    media.image_token_id = request.image_token_id >= 0 ? request.image_token_id : default_image_token_id;
    for (int idx : request.state_image_ids) append_image(media, request.images.at(idx));
    const bool video_field = request.video_token_id >= 0 && (capabilities & D3_CAP_VIDEO) != 0;
    const bool rewrite = request.video_token_id >= 0 && !video_field;
    call.state_ids = request.state_ids;
    if (rewrite) call.state_ids = replace_token(std::move(call.state_ids), request.video_token_id, media.image_token_id);
    if (video_field) call.video_token_id = request.video_token_id;
    call.media = request.state_image_ids.empty() ? nullptr : &media;
    call.question_images.image_token_id = media.image_token_id;
    std::map<int, int> in_table;   // request image index -> position in call.question_images
    for (size_t b : blocks) {
        const EngineBlock& blk = request.blocks[b];
        d3_question_block q;
        q.ids = blk.ids;
        if (rewrite) q.ids = replace_token(std::move(q.ids), request.video_token_id, media.image_token_id);
        q.readout_from_end = blk.readout_from_end;
        for (int idx : blk.image_ids) {
            auto it = in_table.find(idx);
            if (it == in_table.end()) {
                append_image(call.question_images, request.images.at(idx));
                it = in_table.emplace(idx, static_cast<int>(call.question_images.images.size()) - 1).first;
            }
            q.image_ids.push_back(it->second);
        }
        call.questions.push_back(std::move(q));
    }
    return fused;
}

std::unique_ptr<flm::d3::ReadoutEngine> make_d3_engine_adapter(decision_model* engine, d3_npu* npu, unsigned capabilities,
                                                               bool state_once, int image_token_id) {
    return std::make_unique<D3EngineAdapter>(engine, npu, capabilities, state_once, image_token_id);
}
