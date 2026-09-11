/// \file qwen3vl_flash.hpp
/// \brief qwen3vl_flash class
/// \author FastFlowLM Team
/// \date 2026-09-10
/// \version 0.9.28
/// \note This is a header file for the qwen3vl_flash class
///
/// qwen3vl_flash is a second engine for the same Qwen3-VL checkpoint as
/// qwen3vl_npu. It runs prefill on a single fused overlay -- columns 0-5 a
/// dequant+mm array, columns 6-7 one attention CU (mha_d128_q4_1cu) -- instead
/// of reconfiguring the array between mm.xclbin and attn.xclbin twice per
/// layer, and overlaps the vision encoder with the text prefill setup. It is
/// tuned for short contexts; qwen3vl_npu remains the general-purpose engine.
///
/// The image payload types and the QWEN3_* preprocessing constants are shared
/// with qwen3vl_npu, so this header pulls them in rather than redefining them:
/// the application builds one qwen3vl_image_payload_t and hands it to either
/// engine.
#pragma once
#include "models/qwen3vl/qwen3vl_npu.hpp"


class qwen3vl_flash : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_flash
    /// \param config the configuration
    /// \param npu_instance the npu instance
    qwen3vl_flash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3vl_flash();

    /// \brief forward the qwen3vl_flash
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
