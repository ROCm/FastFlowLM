/// \file d3_engine_adapter.hpp
/// \brief The one place where the d3 contract's engine interface (decision/d3_engine.hpp) meets the real engine
///        (d3_npu, IRON FLM_DLL/detail/d3_npu; API: work/m2_api.md, d3_npu.hpp revision 2).
/// \note  When the engine's header changes only this file and d3_engine_adapter.cpp do. What each path runs:
///        - state-once (`d3_npu::encode_state_once`, revision 2): the state once, every block continues from it,
///          readout rows at `readout_from_end` (choice / noul / score: {0}; Box {9,6,3,0}; track {12,9,6,3,0});
///        - images inside question blocks need D3_CAP_QUESTION_IMAGES (not in revision 2). Without it a block with
///          images that reads one row is run per prompt through `encode_batch` with `state_len` (the M1 path, which
///          serves any image layout), and a Box block with images is refused with a message naming the missing feature;
///        - D3_STATE_ONCE=0, or an engine without D3_CAP_STATE_ONCE: the M1 per-row path for every block (one hidden
///          row per prompt: a Box block is refused).
#pragma once

#include <memory>

#include "decision/d3_engine.hpp"
#include "decision_model.hpp"

class d3_npu;

/// \param engine the loaded d3 engine (not owned; must outlive the adapter)
/// \param npu    the same engine as a d3_npu (nullptr without the d3 engine library: only the M1 path)
/// \param capabilities `d3_npu_capabilities()` of the loaded library
/// \param state_once false (D3_STATE_ONCE=0) forces the per-row path
std::unique_ptr<flm::d3::ReadoutEngine> make_d3_engine_adapter(decision_model* engine, d3_npu* npu, unsigned capabilities,
                                                               bool state_once, int image_token_id = 248056);
