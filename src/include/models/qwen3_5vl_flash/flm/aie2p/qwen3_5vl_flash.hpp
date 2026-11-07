/// \file qwen3_5vl_flash.hpp
/// \brief qwen3_5vl_flash class
/// \author FastFlowLM Team
/// \date 2026-11-03
/// \note This is a header file for the qwen3_5vl_flash class
///
/// qwen3_5vl_flash is a second engine for the Qwen3.5-2B checkpoint, next to
/// qwen3_5vl_npu. It runs the kernels from FLM_Xclbin/Qwen3_5_2B (2B shapes
/// only): bfp16 prefill matmul and attention, the faster GateDeltaNet and conv,
/// lm_head inside layer.xclbin and the vision overlay. qwen3_5vl_npu stays the
/// engine for every Qwen3.5 size.
///
/// The image payload types are shared with qwen3_5vl_npu, so this header pulls
/// them in rather than redefining them.
#pragma once
#include "models/qwen3_5vl/flm/aie2p/qwen3_5vl_npu.hpp"
#include "utils/hybrid_state.hpp"   // hybrid_engine_state (the IRON copy of qwen3_5vl_npu.hpp pulls it in; the FLM copy of that header is older)


class qwen3_5vl_flash : public causal_lm{
public:
    /// \brief  initialize the qwen3_5vl_flash
    /// \param config the configuration
    /// \param npu_instance the npu instance
    qwen3_5vl_flash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_5vl_flash();

    /// \brief forward the qwen3_5vl_flash
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    // ---- saved states (see flm_state_api.h) --------------------------------
    // The engine holds no copy of its own context: read_state() copies the live one into a caller's
    // state, write_state() puts one back. checkpoint()/restore() keep one private copy for the legacy
    // single-checkpoint path.
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    // parameters for vision process in qwen3.5 vl

    unsigned int QWEN3_5_PATCH_SIZE;
    unsigned int QWEN3_5_IMAGE_MERGE_SIZE;
    unsigned int QWEN3_5_SPATIAL_MERGE_SIZE;
    unsigned int QWEN3_5_SHORTEST_EDGE;
    unsigned int QWEN3_5_LONGEST_EDGE;
    float QWEN3_5_VISION_RESCALE_FACTOR;
    float QWEN3_5_VISION_RESCALE_IMAGE_MEAN;
    float QWEN3_5_VISION_RESCALE_IMAGE_STD;
    unsigned int QWEN3_5_TEMPORAL_PATCH_SIZE;
    unsigned int QWEN3_5_MERGE_SIZE;


    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        const nlohmann::json& vc = config.sub("vision_config");
        QWEN3_5_PATCH_SIZE  = vc.value("QWEN3_5_PATCH_SIZE", -1);
        QWEN3_5_IMAGE_MERGE_SIZE = vc.value("QWEN3_5_IMAGE_MERGE_SIZE", -1);
        QWEN3_5_SPATIAL_MERGE_SIZE = vc.value("QWEN3_5_SPATIAL_MERGE_SIZE", -1);
        QWEN3_5_SHORTEST_EDGE = vc.value("QWEN3_5_SHORTEST_EDGE", -1);
        QWEN3_5_LONGEST_EDGE = vc.value("QWEN3_5_LONGEST_EDGE", -1);
        QWEN3_5_VISION_RESCALE_FACTOR = vc.value("QWEN3_5_VISION_RESCALE_FACTOR", -1.0f);
        QWEN3_5_VISION_RESCALE_IMAGE_MEAN = vc.value("QWEN3_5_VISION_RESCALE_IMAGE_MEAN", -1.0f);
        QWEN3_5_VISION_RESCALE_IMAGE_STD = vc.value("QWEN3_5_VISION_RESCALE_IMAGE_STD", -1.0f);
        QWEN3_5_TEMPORAL_PATCH_SIZE = vc.value("QWEN3_5_TEMPORAL_PATCH_SIZE", -1);

        QWEN3_5_MERGE_SIZE = QWEN3_5_IMAGE_MERGE_SIZE;

    }
private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a qwen3_5vl_flash
extern "C" const flm_state_api* qwen3_5vl_flash_state_api(void* engine);
