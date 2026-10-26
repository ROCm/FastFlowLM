/// \file d3_engine_adapter_call.hpp
/// \brief The translation of a contract engine request (decision/d3_engine.hpp) into d3_npu's state-once call
///        (d3_state_request, models/d3/flm/aie2p/d3_npu.hpp). Pure data, no engine library: the adapter test checks it.
#pragma once

#include <memory>
#include <vector>

#include "decision/d3_engine.hpp"
#include "models/d3/flm/aie2p/d3_npu.hpp"

/// One d3_npu::encode_state_once call. `call.media` points into this object: keep it where it was made (not copyable).
struct D3FusedCall {
    d3_image_payload media;      //!< the state's images / video groups, in prompt order
    d3_state_request call;
    D3FusedCall() = default;
    D3FusedCall(const D3FusedCall&) = delete;
    D3FusedCall& operator=(const D3FusedCall&) = delete;
};

/// `blocks` (indices into request.blocks, in the order of the engine's answer) as one call.
///  * images inside blocks go to `question_images` (one table entry per distinct request image) with
///    `d3_question_block::image_ids`: the caller must have checked D3_CAP_QUESTION_IMAGES;
///  * video pad runs: with D3_CAP_VIDEO in `capabilities` the request's `video_token_id` is passed on (the groups are
///    entries of `media`, after the images); without it the pad ids are rewritten to the image pad id (same
///    positions and vision rows: bit-identical, see the engine's log), so any engine runs a plain video.
std::unique_ptr<D3FusedCall> make_d3_fused_call(const flm::d3::EngineStateRequest& request, const std::vector<size_t>& blocks,
                                                unsigned capabilities, int default_image_token_id);

/// `ids` with every `from` replaced by `to` (video pad ids of a prompt for an engine that knows image runs only).
std::vector<int> replace_token(std::vector<int> ids, int from, int to);

/// Which blocks of a request one d3_npu::encode_state_once call can take (`fused`: capability bits, not the revision: a
/// block with images needs D3_CAP_QUESTION_IMAGES, a block that reads several rows D3_CAP_READOUT_ROWS, all of it
/// D3_CAP_STATE_ONCE and `state_once`) and which fall back to the M1 per-row prompt path (`per_row`, one hidden row each).
void route_d3_blocks(const flm::d3::EngineStateRequest& request, unsigned capabilities, bool state_once, std::vector<size_t>& fused,
                     std::vector<size_t>& per_row);
