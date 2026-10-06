/// \file qwen3_6_moe_npu.hpp
/// \brief qwen3_6_moe_npu for the aie_next NPU
/// \note Same class, fields and order as the aie2p header, so FLM's frontend
///       can read the vision fields directly; only one of the two headers is
///       ever compiled into a build. The constructor differs: aie_next has no
///       xclbin manager, so it takes the frontend's device instead.
#pragma once
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"

// The device is opened once, by the frontend, and borrowed by every engine.
namespace xrt { class device; }

typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    int grid_h;
    int grid_w;

    bytes _data;

} qwen3_6_moe_image_t;



typedef struct {
    std::vector<qwen3_6_moe_image_t> images;
    std::vector<bf16> _data__processed;    
    unsigned int num_images;
}qwen3_6_moe_image_payload_t;



class qwen3_6_moe_npu : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    /// \param device the frontend's device, borrowed; must outlive the engine
    qwen3_6_moe_npu(LM_Config config, xrt::device *device, int MAX_L = 4096);
    ~qwen3_6_moe_npu();

    /// \brief forward the qwen3vl_npu
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
    // parameters for vision process in qwen3.5 vl

    unsigned int QWEN3_6_MOE_PATCH_SIZE;
    unsigned int QWEN3_6_MOE_IMAGE_MERGE_SIZE;
    unsigned int QWEN3_6_MOE_SPATIAL_MERGE_SIZE;
    unsigned int QWEN3_6_MOE_SHORTEST_EDGE;
    unsigned int QWEN3_6_MOE_LONGEST_EDGE;
    float QWEN3_6_MOE_VISION_RESCALE_FACTOR;
    float QWEN3_6_MOE_VISION_RESCALE_IMAGE_MEAN;
    float QWEN3_6_MOE_VISION_RESCALE_IMAGE_STD;
    unsigned int QWEN3_6_MOE_TEMPORAL_PATCH_SIZE;
    unsigned int QWEN3_6_MOE_MERGE_SIZE;


    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        const nlohmann::json& vc = config.sub("vision_config");
        QWEN3_6_MOE_PATCH_SIZE  = vc.value("QWEN3_6_MOE_PATCH_SIZE", -1);
        QWEN3_6_MOE_IMAGE_MERGE_SIZE = vc.value("QWEN3_6_MOE_IMAGE_MERGE_SIZE", -1);
        QWEN3_6_MOE_SPATIAL_MERGE_SIZE = vc.value("QWEN3_6_MOE_SPATIAL_MERGE_SIZE", -1);
        QWEN3_6_MOE_SHORTEST_EDGE = vc.value("QWEN3_6_MOE_SHORTEST_EDGE", -1);
        QWEN3_6_MOE_LONGEST_EDGE = vc.value("QWEN3_6_MOE_LONGEST_EDGE", -1);
        QWEN3_6_MOE_VISION_RESCALE_FACTOR = vc.value("QWEN3_6_MOE_VISION_RESCALE_FACTOR", -1.0f);
        QWEN3_6_MOE_VISION_RESCALE_IMAGE_MEAN = vc.value("QWEN3_6_MOE_VISION_RESCALE_IMAGE_MEAN", -1.0f);
        QWEN3_6_MOE_VISION_RESCALE_IMAGE_STD = vc.value("QWEN3_6_MOE_VISION_RESCALE_IMAGE_STD", -1.0f);
        QWEN3_6_MOE_TEMPORAL_PATCH_SIZE = vc.value("QWEN3_6_MOE_TEMPORAL_PATCH_SIZE", -1);

        QWEN3_6_MOE_MERGE_SIZE = QWEN3_6_MOE_IMAGE_MERGE_SIZE;

    }
private:
    struct Impl;
    Impl* _impl;
};

