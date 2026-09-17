/// \file phi4_aie2p.hpp
/// \brief Phi-4 on aie2p: FastFlowLM's own NPU kernels
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note The class is still called phi4_npu, and must stay that way: it is
///       defined in the prebuilt engine library lib/<runtime>/libphi4_npu.so
///       (phi4_npu.lib on Windows), which this tree links by name and cannot
///       rebuild. Renaming the class would change the mangling of its
///       constructor and vtable and break the link. The path and file name say
///       aie2p; the symbol keeps the name of the library it declares.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "phi4_aie2p_sequence.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif


class phi4_npu : public causal_lm{
public:
    /// \brief  initialize the phi4_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    phi4_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~phi4_npu();

    /// \brief forward the phi4_npu
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

private:
    struct Impl;
    Impl* _impl;
};

