/// \file d3_engine.hpp
/// \brief The engine-facing interface of the d3 contract: what one state-once call needs and returns.
/// \note  This is the ONLY header between the contract (src/common/decision, CPU only) and an engine. The real
///        engine (d3_npu, IRON: `encode_state_once`, see work/m2_api.md) is wrapped by one adapter class
///        (src/common/AutoDecisionModel/d3_engine_adapter.*); when the engine's header changes only that file does.
///        `ScriptedEngine` (d3_engine_stub.hpp) is the CPU stand-in the tests use.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flm::d3 {

/// One image after the processor's resize, ready for the engine's vision tower.
struct EngineImage {
    int grid_h = 0;                    //!< patch grid (16-pixel patches); the image takes grid_h * grid_w / 4 tokens
    int grid_w = 0;
    std::vector<uint16_t> pixels;      //!< bf16 bits, per patch [3][2][16][16] in 2 x 2 merge order (empty in CPU tests)
};

/// The tokens after the state block of one question: its row is `[state | block]`.
struct EngineBlock {
    std::string tag;                   //!< the question's key (diagnostics, scripted engines)
    std::vector<int> ids;              //!< block token ids, image pad runs expanded
    /// Rows to return: offsets from the block's LAST token (0 = the last token itself), in the order the contract reads
    /// them: choice / noul / score {0}; box {9, 6, 3, 0}; track moment {12, 9, 6, 3, 0}. All are inside the block.
    std::vector<int> readout_from_end;
    /// Images inside the block (a question's instructions / options): one entry per image pad run of `ids`, in order,
    /// as indices into EngineStateRequest::images.
    std::vector<int> image_ids;
};

/// One state and the question blocks that read it: the state is run once, every block attends to it.
/// Equal, row for row, to running each `[state | block]` alone with `state_len = state_ids.size()` (the contract's
/// reference is upstream's `_composed_row`: per-row numerics within noise of the state-once path).
struct EngineStateRequest {
    std::vector<int> state_ids;        //!< the state block (system turn, state images, `State:` + state), pad runs expanded
    std::vector<int> state_image_ids;  //!< one entry per image pad run of `state_ids`, in order: indices into `images`
    std::vector<EngineBlock> blocks;
    /// Every image the call needs, deduplicated by source (an asset used 20 times is one entry; the engine may run the
    /// vision tower once per entry and copy the rows to each use).
    std::vector<EngineImage> images;
    int image_token_id = -1;           //!< `<|image_pad|>`
    int video_token_id = -1;           //!< `<|video_pad|>` (M4)
};

/// The engine as the contract sees it. One call = one state; the contract makes one call per distinct state of a
/// request (the request's own, or a video window's).
class ReadoutEngine {
public:
    virtual ~ReadoutEngine() = default;
    /// Width of a returned hidden row.
    virtual int hidden_size() const = 0;
    /// Final-norm hidden rows: for block b, `blocks[b].readout_from_end.size()` rows of `hidden_size()` floats,
    /// row-major, in the listed order. Throws std::runtime_error when the engine cannot run the request.
    virtual std::vector<std::vector<float>> encode_state_once(const EngineStateRequest& request) = 0;
};

}  // namespace flm::d3
