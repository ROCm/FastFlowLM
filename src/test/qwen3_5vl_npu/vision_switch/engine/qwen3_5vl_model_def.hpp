#ifndef __QWEN3_5VL_MODEL_DEF_HPP__
#define __QWEN3_5VL_MODEL_DEF_HPP__
#include <memory>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
struct qwen3_5vl_layer_weright_def{
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    weight_desc_t ffn_down;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_up;
    virtual ~qwen3_5vl_layer_weright_def() = default;
};

struct qwen3_5vl_linear_layer : public qwen3_5vl_layer_weright_def{
public:
    weight_desc_t linear_attn_qkv;
    weight_desc_t attn_gate;
    weight_desc_t ssm_a;
    weight_desc_t ssm_alpha;
    weight_desc_t ssm_beta;
    weight_desc_t ssm_conv1d;
    weight_desc_t ssm_dt;
    weight_desc_t ssm_norm;
    weight_desc_t ssm_out;
    weight_desc_t ssm_alpha_cpu;
    weight_desc_t ssm_beta_cpu;

    std::vector<weight_desc_t*> all() {
        return {
            &linear_attn_qkv, &attn_gate, &ssm_a, &ssm_alpha, &ssm_beta,
            &ssm_conv1d, &ssm_dt, &ssm_norm, &ssm_out, &input_layer_norm,
            &post_attntion_norm, &ffn_down,
            &ffn_gate, &ffn_up
        };
    }
};

struct qwen3_5vl_attn_layer : public qwen3_5vl_layer_weright_def{
public:
    weight_desc_t attn_k;
    weight_desc_t attn_k_norm;
    weight_desc_t input_layer_norm;
    weight_desc_t attn_q;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_v;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_output;
    weight_desc_t attn_gate;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_k, &attn_k_norm, &input_layer_norm, &attn_q, &attn_q_norm,
            &attn_v, &ffn_down,
            &ffn_gate, &ffn_up,
            &post_attntion_norm, &attn_output, &attn_gate,
        };
    }
};

/// @brief  Owns the weight-shape definition for the qwen3.5vl model: the
///         model dimensions derived from LM_Config, the per-layer weight
///         descriptors, and the aggregated buffer byte sizes.
struct qwen3_5vl_desc {
    static constexpr f32 inv_freq[32] = {
        1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01, 1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
        1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03, 2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 5.232990952e-04,
        3.162277862e-04, 1.910952997e-04, 1.154781930e-04, 6.978305464e-05, 4.216965317e-05, 2.548296652e-05, 1.539926416e-05, 9.305720596e-06,
        5.623413472e-06, 3.398208264e-06, 2.053524895e-06, 1.240937763e-06, 7.498942409e-07, 4.531583784e-07, 2.738419767e-07, 1.654817083e-07,
    };

    int full_attention_interval = 4;
    int attn_idx;
    int lin_idx;

    // dimensions derived from config
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;
    uint32_t DH = 0;
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t Linear_DQ = 0;
    uint32_t Linear_DK = 0;
    uint32_t Linear_DV = 0;
    uint32_t Linear_DQKV = 0;
    uint32_t linear_num_qk_heads = 0;
    uint32_t linear_num_v_heads = 0;
    uint32_t linear_num_v_heads_padded = 0;
    uint32_t linear_qk_head_dim = 0;
    uint32_t linear_v_head_dim = 0;
    uint32_t conv_kernel_size = 0;
    uint32_t RoPE_DIM = 0;
    uint32_t intermediate_size = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<qwen3_5vl_layer_weright_def>> weight_desc;

    // aggregated buffer byte sizes (one representative layer of each kind)
    size_t attn_total_proj_weights_byte_size = 0;
    size_t attn_rms_weights_byte_size = 0;
    size_t linear_total_proj_weights_byte_size = 0;
    size_t linear_rms_weights_byte_size = 0;
    size_t linear_ssm_weights_byte_size = 0;

    qwen3_5vl_desc() {}
    qwen3_5vl_desc(LM_Config config) { build(config); }

    inline bool is_attn_layer(int layer_idx) {
        return (layer_idx % full_attention_interval) == (full_attention_interval - 1);
    }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return MAX_L * (DK + DV); }
    inline size_t get_linear_cache_size() { return (conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16) + linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * sizeof(float); }
    inline size_t get_lm_head_w_size() { return (size_t)vocab_size_padded * D + (size_t)vocab_size_padded * D / 32 * 2; }
    inline size_t get_conv_w_size() { return conv_kernel_size * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16); }
    inline size_t get_s_offset() { return (conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV); }

    inline uint32_t get_upgate_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ffn_up.offset; }
    inline uint32_t get_down_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ffn_down.offset; }

    inline uint32_t get_attn_q_offset() {return ((qwen3_5vl_attn_layer*)(weight_desc[attn_idx].get()))->attn_q.offset; }
    inline uint32_t get_attn_k_offset() {return ((qwen3_5vl_attn_layer*)(weight_desc[attn_idx].get()))->attn_k.offset; }
    inline uint32_t get_attn_v_offset() {return ((qwen3_5vl_attn_layer*)(weight_desc[attn_idx].get()))->attn_v.offset; }
    inline uint32_t get_attn_g_offset() {return ((qwen3_5vl_attn_layer*)(weight_desc[attn_idx].get()))->attn_gate.offset; }
    inline uint32_t get_attn_o_offset() {return ((qwen3_5vl_attn_layer*)(weight_desc[attn_idx].get()))->attn_output.offset; }

    inline uint32_t get_linear_qkv_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->linear_attn_qkv.offset; }
    inline uint32_t get_linear_g_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->attn_gate.offset; }
    inline uint32_t get_linear_o_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_out.offset; }
    inline uint32_t get_linear_ssm_norm_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_norm.offset; }
    inline uint32_t get_linear_alpha_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_alpha.offset; }
    inline uint32_t get_linear_beta_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_beta.offset; }
    inline uint32_t get_linear_A_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_a.offset; }
    inline uint32_t get_linear_dt_offset() {return ((qwen3_5vl_linear_layer*)(weight_desc[lin_idx].get()))->ssm_dt.offset; }

    inline void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        DH = config.get("head_dim");
        DQ = config.get("head_dim") * config.get("num_attention_heads");
        DK = config.get("head_dim") * config.get("num_key_value_heads");
        DV = config.get("head_dim") * config.get("num_key_value_heads");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads");
        linear_qk_head_dim = config.get("linear_key_head_dim");
        linear_v_head_dim = config.get("linear_value_head_dim");
        linear_num_qk_heads = config.get("linear_num_key_heads");
        linear_num_v_heads = config.get("linear_num_value_heads");
        linear_num_v_heads_padded = (linear_num_v_heads + 32 - 1) / 32 * 32;
        conv_kernel_size = config.get("linear_conv_kernel_dim");
        Linear_DQ = linear_qk_head_dim * linear_num_qk_heads;
        Linear_DK = linear_qk_head_dim * linear_num_qk_heads;
        Linear_DV = linear_v_head_dim * linear_num_v_heads;
        Linear_DQKV = Linear_DQ + Linear_DK + Linear_DV;
        intermediate_size = config.get("intermediate_size");
        RoPE_DIM = config.get("head_dim") * 0.25;
        JSON_GET(full_attention_interval, config._json_config, "full_attention_interval", 4, int);
        vocab_size = config.get("vocab_size");
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;
        lin_idx = 0;
        attn_idx = full_attention_interval - 1;
        DEBUG_BLOCK(1,
        std::cout << "================ qwen3_5vl_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"        << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"          << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"            << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ"                       << " = " << DQ << std::endl;
        std::cout << std::left << std::setw(28) << "DK"                       << " = " << DK << std::endl;
        std::cout << std::left << std::setw(28) << "DV"                       << " = " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"                << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"             << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_qk_head_dim"       << " = " << linear_qk_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_v_head_dim"        << " = " << linear_v_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_qk_heads"      << " = " << linear_num_qk_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_v_heads"       << " = " << linear_num_v_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_v_heads_padded"<< " = " << linear_num_v_heads_padded << std::endl;
        std::cout << std::left << std::setw(28) << "conv_kernel_size"         << " = " << conv_kernel_size << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DQ"                << " = " << Linear_DQ << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DK"                << " = " << Linear_DK << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DV"                << " = " << Linear_DV << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"        << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "RoPE_DIM"                 << " = " << RoPE_DIM << std::endl;
        std::cout << std::left << std::setw(28) << "full_attention_interval"  << " = " << full_attention_interval << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"               << " = " << vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size_padded"        << " = " << vocab_size_padded << std::endl;
        std::cout << std::left << std::setw(28) << "lin_idx"                  << " = " << lin_idx << std::endl;
        std::cout << std::left << std::setw(28) << "attn_idx"                 << " = " << attn_idx << std::endl;
        std::cout << "==================================================" << std::endl;
        )
        weight_desc.resize(num_hidden_layers);

        weight_container attn_proj_weights_helper;
        weight_container attn_rms_weights_helper;

        weight_container linear_proj_weights_helper;
        weight_container linear_rms_weights_helper;
        weight_container linear_ssm_weights_helper;
        bool is_attn_helper_initialized = false;
        bool is_linear_helper_initialized = false;
        for (size_t i = 0; i < num_hidden_layers; i++){
            if (is_attn_layer(i)){
                auto layer = std::make_unique<qwen3_5vl_attn_layer>();
                layer->attn_k = weight_desc_t(flm_q4k, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
                layer->attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");
                layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
                layer->attn_q = weight_desc_t(flm_q4k, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
                layer->attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
                layer->attn_v = weight_desc_t(flm_q4k, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
                layer->ffn_down = weight_desc_t(flm_q4k, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
                layer->ffn_up = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
                layer->ffn_gate = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
                layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
                layer->attn_output = weight_desc_t(flm_q4k, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
                layer->attn_gate = weight_desc_t(flm_q4k, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight"); // special actully merged with q
                if (!is_attn_helper_initialized) {
                    attn_proj_weights_helper.add_weight(layer->ffn_up);
                    attn_proj_weights_helper.add_weight(layer->ffn_gate);
                    attn_proj_weights_helper.add_weight(layer->ffn_down);
                    attn_proj_weights_helper.add_weight(layer->attn_q);
                    attn_proj_weights_helper.add_weight(layer->attn_k);
                    attn_proj_weights_helper.add_weight(layer->attn_v);
                    attn_proj_weights_helper.add_weight(layer->attn_gate);
                    attn_proj_weights_helper.add_weight(layer->attn_output);

                    attn_rms_weights_helper.add_weight(layer->input_layer_norm);
                    attn_rms_weights_helper.add_weight(layer->post_attntion_norm);

                    is_attn_helper_initialized = true;
                }
                weight_desc[i] = std::move(layer);
            }
            else{
                auto layer = std::make_unique<qwen3_5vl_linear_layer>();
                layer->linear_attn_qkv = weight_desc_t(flm_q4k, {D, Linear_DQ + Linear_DK + Linear_DV}, "model.layers.%d.linear_attn.qkv_proj.weight");
                layer->attn_gate = weight_desc_t(flm_q4k, {D, Linear_DV}, "model.layers.%d.self_attn.gate_proj.weight");
                layer->ssm_a = weight_desc_t(flm_f32, {linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_a");
                layer->ssm_alpha = weight_desc_t(flm_q80, {D, linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_alpha_proj.weight");
                layer->ssm_beta = weight_desc_t(flm_q80, {D, linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_beta_proj.weight");
                layer->ssm_conv1d = weight_desc_t(flm_bf16, {Linear_DV + Linear_DQ + Linear_DK, conv_kernel_size}, "model.layers.%d.linear_attn.ssm_conv1d.weight");
                layer->ssm_dt = weight_desc_t(flm_f32, {linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_dt.bias");
                layer->ssm_norm = weight_desc_t(flm_bf16, {linear_v_head_dim}, "model.layers.%d.linear_attn.ssm_norm.weight");
                layer->ssm_out = weight_desc_t(flm_q4k, {Linear_DV, D}, "model.layers.%d.linear_attn.ssm_out_proj.weight");
                layer->ssm_alpha_cpu = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layers.%d.linear_attn.ssm_alpha_proj.bf16.weight");
                layer->ssm_beta_cpu = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layers.%d.linear_attn.ssm_beta_proj.bf16.weight");
                layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
                layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
                layer->ffn_down = weight_desc_t(flm_q4k, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
                layer->ffn_up = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
                layer->ffn_gate = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
                if (!is_linear_helper_initialized) {
                    linear_proj_weights_helper.add_weight(layer->ffn_up);
                    linear_proj_weights_helper.add_weight(layer->ffn_gate);
                    linear_proj_weights_helper.add_weight(layer->ffn_down);
                    linear_proj_weights_helper.add_weight(layer->linear_attn_qkv);
                    linear_proj_weights_helper.add_weight(layer->attn_gate);
                    linear_proj_weights_helper.add_weight(layer->ssm_out);

                    linear_rms_weights_helper.add_weight(layer->input_layer_norm);
                    linear_rms_weights_helper.add_weight(layer->post_attntion_norm);

                    linear_ssm_weights_helper.add_weight(layer->ssm_conv1d);
                    linear_ssm_weights_helper.add_weight(layer->ssm_norm);
                    linear_ssm_weights_helper.add_weight(layer->ssm_a);
                    linear_ssm_weights_helper.add_weight(layer->ssm_dt);
                    linear_ssm_weights_helper.add_weight(layer->ssm_alpha);
                    linear_ssm_weights_helper.add_weight(layer->ssm_beta);
                    is_linear_helper_initialized = true;
                }
                weight_desc[i] = std::move(layer);
            }
        }

        attn_total_proj_weights_byte_size = attn_proj_weights_helper.get_size();
        attn_rms_weights_byte_size = attn_rms_weights_helper.get_size();

        linear_total_proj_weights_byte_size = linear_proj_weights_helper.get_size();
        linear_rms_weights_byte_size = linear_rms_weights_helper.get_size();
        linear_ssm_weights_byte_size = linear_ssm_weights_helper.get_size();
    }

    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2)
    {
        assert(is_quantize(dtype));
        const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int rows = src.size() / a_block_size / blocks_per_row;

        // Step 1: vertical block reorder at a_block_size granularity
        u8 *dst_ptr = dst;
        std::vector<u8 *> src_ptr(vertical_blocks);
        for (int i = 0; i < vertical_blocks; i++)
            src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;

        for (int r = 0; r < rows; r += vertical_blocks)
        {
            for (int c = 0; c < blocks_per_row; c++)
            {
                for (int i = 0; i < vertical_blocks; i++)
                {
                    memcpy(dst_ptr, src_ptr[i], a_block_size);
                    dst_ptr += a_block_size;
                    src_ptr[i] += a_block_size;
                }
            }
            for (int i = 0; i < vertical_blocks; i++)
            {
                src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
                if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
                    src_ptr[i] = src.data(); // useless padding
            }
        }
    }

    /// @brief lm_head variant of reorder_cpy: lays a band's rows out as two
    ///        contiguous halves instead of interleaving them.
    /// @note  Each column of the lm_head xclbin is now fed by two independent shim
    ///        MM2S channels, one per pair of cores, because a single channel
    ///        saturates at roughly a quarter of what the column can consume. The two
    ///        halves have to sit back to back in DDR so gen_lm_head_seq can issue one
    ///        contiguous descriptor per stream. This is deliberately separate from
    ///        the generic reorder_cpy above: that one still feeds the body weights,
    ///        whose xclbin wants the old interleave.
    void reorder_cpy_lm_head(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 4)
    {
        assert(is_quantize(dtype));
        assert(vertical_blocks % 2 == 0 && "the two weight streams split the band evenly");

        const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int rows = src.size() / a_block_size / blocks_per_row;
        const int rows_per_stream = vertical_blocks / 2;

        u8 *dst_ptr = dst;
        for (int r = 0; r < rows; r += vertical_blocks)
        {
            for (int half = 0; half < 2; half++)
            {
                for (int c = 0; c < blocks_per_row; c++)
                {
                    for (int i = 0; i < rows_per_stream; i++)
                    {
                        const int row = r + half * rows_per_stream + i;
                        const u8 *src_ptr = (row < rows)
                            ? src.data() + (size_t)(row * blocks_per_row + c) * a_block_size
                            : src.data(); // useless padding
                        memcpy(dst_ptr, src_ptr, a_block_size);
                        dst_ptr += a_block_size;
                    }
                }
            }
        }
    }

    /// @brief lm_head for the layer xclbin's projection engine (mode 2): every Q8_0 32 x 256 chunk is cut into two
    ///        32 x 128 halves, each padded to one Q4_K chunk; laid out one weight stream after another.
    /// @note  See host/README.md, "lm_head inside layer.xclbin".
    static constexpr int lm_head_streams = 8;
    static constexpr int lm_head_rows_per_round = 512;
    inline int get_lm_head_in_layer_rows() const { return (vocab_size + lm_head_rows_per_round - 1) / lm_head_rows_per_round * lm_head_rows_per_round; }
    inline size_t get_lm_head_in_layer_w_size() const {
        const size_t padded_block = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
        const size_t half_blocks_per_row_block = 2 * (D / QXNX_COL_BLOCK_SIZE);
        return (size_t)get_lm_head_in_layer_rows() / QXNX_ROW_BLOCK_SIZE * half_blocks_per_row_block * padded_block;
    }

    void reorder_cpy_lm_head_in_layer(u8 *dst, const buffer<u8> &src) const
    {
        const size_t chunk = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q80);
        const size_t padded_block = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
        const size_t scale_bytes = QXNX_ROW_BLOCK_SIZE * (QXNX_COL_BLOCK_SIZE / 32) * sizeof(bf16);
        const size_t half_scale_bytes = scale_bytes / 2;
        const size_t half_q_bytes = (chunk - scale_bytes) / 2;
        const int chunks_per_row_block = D / QXNX_COL_BLOCK_SIZE;
        const int stored_row_blocks = src.size() / chunk / chunks_per_row_block;
        const int rounds = get_lm_head_in_layer_rows() / lm_head_rows_per_round;
        const int row_blocks_per_pair = 2;

        memset(dst, 0, get_lm_head_in_layer_w_size());
        u8 *dst_ptr = dst;
        for (int stream = 0; stream < lm_head_streams; stream++) {
            for (int r = 0; r < rounds; r++) {
                const int pair = r * lm_head_streams + stream;
                for (int c = 0; c < chunks_per_row_block; c++) {
                    for (int half = 0; half < 2; half++) {
                        for (int i = 0; i < row_blocks_per_pair; i++) {
                            const int row_block = pair * row_blocks_per_pair + i;
                            if (row_block < stored_row_blocks) {
                                const u8 *src_chunk = src.data() + ((size_t)row_block * chunks_per_row_block + c) * chunk;
                                memcpy(dst_ptr, src_chunk + half * half_scale_bytes, half_scale_bytes);
                                memcpy(dst_ptr + half_scale_bytes, src_chunk + scale_bytes + half * half_q_bytes, half_q_bytes);
                            }
                            dst_ptr += padded_block;
                        }
                    }
                }
            }
        }
    }

    void load_linear_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& ssm_weights, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu){
        constexpr int GLU_SLICE = 1024;
        constexpr int UP_GATE_OUTDIM_REORDEER = 512;
        qwen3_5vl_linear_layer& _weight_desc = static_cast<qwen3_5vl_linear_layer&>(*this->weight_desc[this->lin_idx]);
        //constexpr int GLU_OUTDIM = 512;
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc = _weight_desc.linear_attn_qkv;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            reorder_cpy(w_ptr, w, D, desc.dtype);
            desc.load();
        }
        // read gate weights
        {
            LOG_VERBOSE(1, "Loading gate weights");
            weight_desc_t& desc = _weight_desc.attn_gate;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            reorder_cpy(w_ptr, w, D, desc.dtype);
            desc.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.ssm_out;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            assert(w_ptr + w.size() <= proj_weights.end());
            reorder_cpy(w_ptr, w, Linear_DV, desc.dtype);
            desc.load();
        }
        // read up weights
        // read up/gate weights
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            u8* w_ptr = w_ptr_begin + desc_up.offset;

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDEER;
            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, _weight_desc.ffn_up.dtype);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, _weight_desc.ffn_gate.dtype);
                w_ptr += slice_size;
            }
            desc_up.load();
            desc_gate.load();
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading down weights");
            weight_desc_t& desc = _weight_desc.ffn_down;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w, intermediate_size, desc.dtype);
            desc.load();
        }

        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_rms_post;

            u8* rms_weights_begin = (u8*)rms_weights.data();
            q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
            q4nx.load_weights(w_rms_post, _weight_desc.post_attntion_norm.format_name(layer_idx));

            // w_rms_post.memset(1);
            memcpy(rms_weights_begin + _weight_desc.input_layer_norm.offset, w_rms_pre.data(), D * sizeof(bf16));
            memcpy(rms_weights_begin + _weight_desc.post_attntion_norm.offset, w_rms_post.data(), D * sizeof(bf16));
            _weight_desc.input_layer_norm.load();
            _weight_desc.post_attntion_norm.load();
        }

        u8* other_weights_begin = (u8*)ssm_weights.data();
        {
            // load conv_weights
            LOG_VERBOSE(1, "Loading ssm conv weights");
            weight_desc_t& desc = _weight_desc.ssm_conv1d;
            u8* w_ptr = other_weights_begin + desc.offset;
            buffer<bf16> conv_weights;
            q4nx.load_weights(conv_weights, desc.format_name(layer_idx));
            memcpy(w_ptr, conv_weights.data(), conv_weights.size() * sizeof(bf16));
            desc.load();
        }

        {
            // load ssm_norm_weights
            LOG_VERBOSE(1, "Loading ssm norm weights");
            weight_desc_t& desc = _weight_desc.ssm_norm;
            u8* w_ptr = other_weights_begin + desc.offset;
            buffer<bf16> norm_weights;
            q4nx.load_weights(norm_weights, desc.format_name(layer_idx));
            memcpy(w_ptr, norm_weights.data(), linear_v_head_dim * sizeof(bf16));
            desc.load();
        }
        {
            // load ssm a, dt
            LOG_VERBOSE(1, "Loading ssm a, dt weights");
            weight_desc_t& desc_a = _weight_desc.ssm_a;
            weight_desc_t& desc_dt = _weight_desc.ssm_dt;
            buffer<float> a;
            buffer<float> dt;

            q4nx.load_weights(a, desc_a.format_name(layer_idx));
            q4nx.load_weights(dt, desc_dt.format_name(layer_idx));
            memcpy(other_weights_begin + desc_a.offset, a.data(), linear_num_v_heads * sizeof(float));
            memcpy(other_weights_begin + desc_dt.offset, dt.data(), linear_num_v_heads * sizeof(float));

            desc_a.load();
            desc_dt.load();
        }
        {
            // load ssm alpha, beta
            LOG_VERBOSE(1, "Loading ssm alpha, beta weights");
            weight_desc_t& desc_alpha = _weight_desc.ssm_alpha;
            weight_desc_t& desc_beta = _weight_desc.ssm_beta;
            buffer<bf16> alpha;
            buffer<bf16> beta;

            q4nx.load_weights(alpha, desc_alpha.format_name(layer_idx));
            q4nx.load_weights(beta, desc_beta.format_name(layer_idx));

            memcpy(other_weights_begin + desc_alpha.offset, alpha.data(), desc_alpha.get_size());
            memcpy(other_weights_begin + desc_beta.offset, beta.data(), desc_beta.get_size());

            desc_alpha.load();
            desc_beta.load();
        }
        {
            // load cpu (bf16) ssm alpha, beta for host-side computation
            LOG_VERBOSE(1, "Loading cpu ssm alpha, beta weights");
            weight_desc_t& desc_alpha_cpu = _weight_desc.ssm_alpha_cpu;
            weight_desc_t& desc_beta_cpu = _weight_desc.ssm_beta_cpu;
            buffer<bf16> alpha;
            buffer<bf16> beta;

            q4nx.load_weights(alpha, desc_alpha_cpu.format_name(layer_idx));
            q4nx.load_weights(beta, desc_beta_cpu.format_name(layer_idx));

            memcpy(alpha_proj_cpu.data(), alpha.data(), desc_alpha_cpu.get_size());
            memcpy(beta_proj_cpu.data(), beta.data(), desc_beta_cpu.get_size());

            desc_alpha_cpu.load();
            desc_beta_cpu.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        ssm_weights.sync_to_device();
    }

    void load_attn_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights){
        constexpr int GLU_SLICE = 1024;
        constexpr int UP_GATE_OUTDIM_REORDEER = 512;
        qwen3_5vl_attn_layer& _weight_desc = static_cast<qwen3_5vl_attn_layer&>(*this->weight_desc[this->attn_idx]);
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc_q = _weight_desc.attn_q;
            weight_desc_t& desc_k = _weight_desc.attn_k;
            weight_desc_t& desc_v = _weight_desc.attn_v;
            weight_desc_t& desc_gate = _weight_desc.attn_gate;
            u8* w_ptr = w_ptr_begin + desc_q.offset;
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, desc_q.format_name(layer_idx));
            q4nx.load_weights(w_k, desc_k.format_name(layer_idx));
            q4nx.load_weights(w_v, desc_v.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            size_t w_q_gate_size = w_q.size();
            buffer<u8> w_q_q(w_q.data(),w_q_gate_size / 2);
            buffer<u8> w_gate(w_q.data() + w_q_gate_size / 2, w_q_gate_size / 2);
            reorder_cpy(w_ptr, w_q_q, D, desc_q.dtype);
            w_ptr += desc_q.get_size();
            reorder_cpy(w_ptr, w_k, D, desc_k.dtype);
            w_ptr += desc_k.get_size();
            reorder_cpy(w_ptr, w_v, D, desc_v.dtype);
            w_ptr += desc_v.get_size();
            reorder_cpy(w_ptr, w_gate, D, desc_gate.dtype);
            desc_q.load();
            desc_k.load();
            desc_v.load();
            desc_gate.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w_o, DQ, desc.dtype);
            desc.load();
        }
        // read up/gate weights
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            u8* w_ptr = w_ptr_begin + desc_up.offset;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDEER;

            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, desc_up.dtype);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, desc_gate.dtype);
                w_ptr += slice_size;
            }
            desc_up.load();
            desc_gate.load();
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading shared down weights");
            weight_desc_t& desc = _weight_desc.ffn_down;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_down;
            q4nx.load_weights(w_down, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w_down, intermediate_size, desc.dtype);
            desc.load();
        }
        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            weight_desc_t& desc_pre = _weight_desc.input_layer_norm;
            weight_desc_t& desc_post = _weight_desc.post_attntion_norm;
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_rms_post;

            u8* rms_weights_begin = (u8*)rms_weights.data();
            q4nx.load_weights(w_rms_pre, desc_pre.format_name(layer_idx));
            q4nx.load_weights(w_rms_post, desc_post.format_name(layer_idx));

            // w_rms_post.memset(1);
            memcpy(rms_weights_begin + desc_pre.offset, w_rms_pre.data(), D * sizeof(bf16));
            memcpy(rms_weights_begin + desc_post.offset, w_rms_post.data(), D * sizeof(bf16));
            desc_pre.load();
            desc_post.load();
        }
        {
            LOG_VERBOSE(1, "Loading attention norm weights");
            weight_desc_t& desc_q_norm = _weight_desc.attn_q_norm;
            weight_desc_t& desc_k_norm = _weight_desc.attn_k_norm;
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;
            q4nx.load_weights(q_attn_norm, desc_q_norm.format_name(layer_idx));
            q4nx.load_weights(k_attn_norm, desc_k_norm.format_name(layer_idx));
            memcpy(rope_weights.data() + RoPE_DIM, q_attn_norm.data(), DH * sizeof(bf16));
            memcpy(rope_weights.data() + RoPE_DIM + DH, k_attn_norm.data(), DH * sizeof(bf16));
            desc_q_norm.load();
            desc_k_norm.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        rope_weights.sync_to_device();
    }
};

#endif
