/// \file qwen3_5vl_npu.hpp
/// \brief qwen3_5vl_npu for the aie_next NPU: the only header the application compiles
///        against (FLM_DLL/include/models/qwen3_5vl/aie_next/qwen3_5vl_npu.hpp).
/// \note  The aie2p engine of the same class exists: this class has the same
///        public fields in the same order as its header
///        (FastFlowLM src/include/models/qwen3_5vl/flm/aie2p/qwen3_5vl_npu.hpp),
///        and the same image payload types. FLM's frontend reads the vision
///        preprocessing fields directly, and only one of the two headers is
///        compiled into a build. Only the constructor differs: aie_next has no
///        xclbin manager, so it takes the frontend's device.
#pragma once
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#include "flm_state_api.h"

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

} qwen3_5vl_image_t;

/// prefill()'s payload: the images of the prompt in order, and their pixels
/// [grid_h * grid_w][3 * 2 * 16 * 16] bf16 each, back to back.
typedef struct {
    std::vector<qwen3_5vl_image_t> images;
    std::vector<bf16> _data__processed;
    unsigned int num_images;
} qwen3_5vl_image_payload_t;

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a qwen3_5vl_npu
extern "C" const flm_state_api* qwen3_5vl_state_api(void* engine);

class qwen3_5vl_npu : public causal_lm {
public:
    /// \param config the model's config.json (+ FLM's fields: model_name, exec_path)
    /// \param device the frontend's device, borrowed; must outlive the engine
    /// \param MAX_L the context FLM asks for; rounded up to a power of 2
    qwen3_5vl_npu(LM_Config config, xrt::device *device, int MAX_L = 4096);
    ~qwen3_5vl_npu();

    buffer<bf16> forward(int ids) override;
    /// \param payload a qwen3_5vl_image_payload_t * when the prompt has images, else nullptr
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int L) override;
    void load_weights(Q4NX& q4nx) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;
    void update_max_length(uint32_t MAX_L) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    // parameters for vision process in qwen3.5 vl (the frontend's image preprocessing reads them)
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
        // called by the constructor
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
    friend const flm_state_api* ::qwen3_5vl_state_api(void* engine);
    struct Impl;
    Impl* _impl;
};
