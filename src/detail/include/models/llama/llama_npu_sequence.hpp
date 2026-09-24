/// \file llama_npu_sequence.hpp
/// \brief llama_npu_sequence class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the llama_npu_sequence class
#pragma once
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"

struct llama_desc;

/// \brief llama_npu_sequence class
/// \note This is a class for the llama_npu_sequence
class llama_npu_sequence{
public:
    llama_npu_sequence(){}

    /// \brief Constructor
    /// \param desc the model definition, owned by llama_npu::Impl; the weight
    ///        descriptors it holds are read directly when moving weights
    /// \param config the configuration
    /// \param MAX_L the max length
    llama_npu_sequence(llama_desc& desc, LM_Config config, uint32_t MAX_L);
    ~llama_npu_sequence();

    /// \brief Generate the decode rtp sequence, run once per token ahead of
    ///        the per-layer sequences (the RTPs it writes are the same for
    ///        every layer)
    /// \param seq the sequence
    /// \param L the length
    void gen_layer_rtp_seq(npu_sequence* seq, const uint32_t L);

    /// \brief Generate the layer sequence
    /// \param seq the sequence
    /// \param L the length
    void gen_layer_seq(npu_sequence* seq, const uint32_t L);

    /// \brief Generate the dequant sequence
    /// \param seq the sequence
    /// \param D_in the input dimension
    /// \param D_out the output dimension
    /// \param weight_offset the weight offset
    void gen_dequant_seq(npu_sequence* seq, const size_t D_in, const size_t D_out, const size_t weight_offset);

    /// \brief Generate the lm head sequence
    /// \param seq the sequence
    void gen_lm_head_seq(npu_sequence* seq);

    /// \brief Set the max length
    /// \param MAX_L the max length
    void set_max_length(const uint32_t MAX_L);

    /// \brief Generate the mha engine sequence
    /// \param seq the sequence
    /// \param L_begin the begin length
    /// \param L_end the end length
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);

    /// \brief Get the k03 offset
    size_t get_k03_offset() const;
    size_t get_k47_offset() const;
    size_t get_v03_offset() const;
    size_t get_v47_offset() const;

private:
    struct Impl;
    Impl* _impl;
};
