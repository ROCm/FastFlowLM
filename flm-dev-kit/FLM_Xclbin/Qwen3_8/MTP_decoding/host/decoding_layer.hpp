// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __DECODING_LAYER_HPP__
#define __DECODING_LAYER_HPP__

#include "models.h"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "tensor_2d.hpp"
class decoding_layer {
private:
    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;
    npu_app lm_head;

    int x_arg_id = 0;
    int proj_arg_id = 1;
    int rms_arg_id = 2;
    int rope_rms_arg_id = 3;
    int kv_cache_arg_id = 4;

    int x_gropu_id = 3;
    int proj_group_id = 4;
    int rms_group_id = 5;
    int rope_rms_group_id = 6;
    int kv_cache_group_id = 7;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    buffer<bf16> rope_rms_buffer;
    buffer<bf16> kv_cache;

    int current_context_length;
    int num_hidden_layers;

    void _gen_sequence(npu_sequence* seq, int L);

    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);

    void _send_fc_input(npu_sequence* seq);
    void _send_fc_norm_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, int slot);
    void _move_kv_cache(npu_sequence* seq);
    void _receive_layer_out(npu_sequence* seq);
    void _write_attn_L(npu_sequence* seq);


    // model specific config
    int DH;
    int D;
    int DQ;
    int DK;
    int DV;
    int RoPE_DIM;
    int num_kv_heads;
    int num_kv_per_round;
    /// The attention partitioning, mirroring iron/model_config.py. Two columns
    /// of two CUs, one kv head each; a column therefore owns two heads and
    /// kv_heads_per_column * DH of every cache row.
    int num_attn_columns = 2;
    int cus_per_column = 2;
    int kv_heads_per_column;
    bool has_qk_norm;
    int intermediate_size;
    int MAX_L;

    /// The KV cache BO, in bf16 elements: MAX_L * (DK + DV).
    ///
    /// **Token-major, TWO CONTIGUOUS PLANES**: all MAX_L rows of k first, then
    /// all MAX_L rows of v, starting at MAX_L * DK. Row `t`'s k is at `t * DK`
    /// and its v at `MAX_L * DK + t * DV`. The four accessors below are the
    /// single place the layout is expressed -- flip them and every descriptor,
    /// the seeder and the read-back follow.
    ///
    /// Note this is *not* the DLL port's layout: `q38npu::mtp_layout`
    /// interleaves the row, `[MAX_L][DK + DV]`, so a descriptor diffed between
    /// this scaffold and `mtp_sequence.cpp` will disagree on the cache. That is
    /// by design, and SECOND_DISPATCH_TIMEOUT.md records the diff it is
    /// expected to produce -- six words, all of them the cache's.
    ///
    /// **This layout used to carry a cap, and no longer does.** The append
    /// crossed from a row's k to its v with a stride of MAX_L * DK, because the
    /// RoPE tile emitted both on one channel and one BD had to write both. A
    /// shim BD's stride field is 20 bits and bf16 strides are halved on
    /// lowering, so that stopped encoding past MAX_L = 2048 -- silently:
    /// npu_instr_utils prints "stride_2 out of range" to stderr, masks the
    /// value into 20 bits, and every v row lands at a wrapped address, so only
    /// the *next* pass sees the damage.
    ///
    /// The fix was on the producer. k and v now carry different packet ids and
    /// land on different S2MM channels of the column's shim, so each plane gets
    /// its own BD whose only stride is one row -- DK or DV, constants of the
    /// model. Nothing in the append scales with MAX_L. See
    /// `iron/constants.py::kv_out_pkt_id`, `iron/RoPE.py`, and
    /// `_receive_kv_cache`.
    size_t kv_cache_size;
    /// The largest MAX_L the descriptors can express, and why.
    ///
    /// With the append's stride gone, the binding limit is the replay's: dim1
    /// counts whole L_CHUNKs of context and a BD size field is 10 bits, so one
    /// pass walks at most 1024 chunks. L_CHUNK = 16, hence 16384 rows -- 64 MB
    /// of cache for this head, which is a descriptor limit rather than a memory
    /// recommendation. MAX_L must also be a whole number of L_CHUNKs: the
    /// replay rounds the key count up to a chunk boundary and the cache has to
    /// own the rows that tail reads. The constructor checks both.
    static constexpr int MAX_L_LIMIT = 1024 * 16;
    /// Elements from one k row to the next, and from one v row to the next.
    /// The planes are walked separately, so each has its own.
    size_t k_row_stride() const { return (size_t)DK; }
    size_t v_row_stride() const { return (size_t)DV; }
    /// Element offset of row `t`'s k.
    size_t k_at(int t) const { return (size_t)t * k_row_stride(); }
    /// Element offset of row `t`'s v -- past the whole k plane.
    size_t v_at(int t) const {
        return (size_t)MAX_L * (size_t)DK + (size_t)t * v_row_stride();
    }
    /// Where the layer output lands in hidden_states_buffer, in elements. It
    /// sits behind the 2 * D input rather than over it: the cores are still
    /// reading x when the first row blocks come back.
    ///
    /// D wide, and it is the whole layer:
    /// h2 = h + W_down @ (silu(gate(yn)) * up(yn)). The shim channel and the
    /// descriptor are down_proj's, unchanged -- when down became the rms tile's
    /// third-phase x the tap stayed put and only the producer in front of it
    /// changed. down itself is no longer separately observable for that reason,
    /// the same trade the post-attention norm and the SwiGLU row each made when
    /// they became the next stage's x.
    size_t layer_out_offset;
    /// Where the two qk-norm gains start in rope_rms_buffer, i.e. how many
    /// elements the cos/sin half in front of them occupies. T = 1, so that is
    /// one ROPE_DIM-wide pair. The layout is the ABI between set_rope_position,
    /// iron/RoPE.py's buffer type and kernels/rope.cc's pointer arithmetic --
    /// three files, and nothing checks them against each other.
    size_t rope_w_offset;
    size_t w_fc_offset;
    size_t w_qkv_offset;
    size_t w_gate_offset;
    size_t w_o_offset;
    size_t w_up_offset;
    size_t w_down_offset;

    // Zeroed because _load_rtp_addresses uses "still 0" to mean "not yet seen"
    // when it cross-checks the eight cores against each other.
    uint32_t l_qk_address = 0;
    uint32_t l_kv_address = 0;
    uint32_t l_begin_mha_address;
    uint32_t l_end_mha_address;
    uint32_t kk_address;
    uint32_t rms_address;

    // npu configs

    /// The fc input stager (CT[3][3]) is fed from IT3. The rms tile (CT[0][2])
    /// takes input_layernorm from IT2; the RoPE tile (CT[1][2]) takes its
    /// cos/sin block from IT2 as well, on the other MM2S channel.
    ///
    /// k and v no longer land on IT3: they are split by kv head and appended on
    /// the two *attention* shims, because npu_dma_wait only orders transfers
    /// within one shim and the replay reads from those same two. They are split
    /// again by plane, k onto each shim's S2MM channel 0 and v onto channel 1,
    /// which is what lets each plane be appended by its own BD -- see
    /// kv_cache_size. Both splits are stamped as packet ids by the RoPE tile.
    /// All of these must match iron/MTP_layer_aie.py.
    static constexpr npu_tiles pass_tile    = IT3;
    /// The **gated** attention output, o * sigmoid(z). It lands on IT2, not
    /// IT3: MT[3] now forwards o east to the gate core at CT[2][2] rather than
    /// south to a shim, so the DDR tap moved one tile downstream with it. IT2's
    /// S2MM is otherwise free -- its two MM2S channels carry the rms and RoPE
    /// weight blocks, which is the other direction.
    /// (the tap was relocated, not deleted,
    /// so every channel that drained on each pass still does.)
    static constexpr npu_tiles o_tap_tile   = IT2;
    /// h2, the layer's scored output. IT2 again, on the other S2MM channel from
    /// the gate tap: the rms tile sits in column 2 and its y port already
    /// reaches this shim, so the tap is one hop straight south.
    ///
    /// The shim, the channel and the descriptor have outlived three producers
    /// now -- the post-attention norm, then down_proj, now the rms tile's third
    /// phase. Each time the stage in front became the next one's x and the tap
    /// stayed where it was. o_proj and down never surface: both go to the rms
    /// tile as residual operands.
    static constexpr npu_tiles layer_out_tap_tile = IT2;
    static constexpr npu_tiles attn_col_tiles[2] = {IT4, IT5};
    /// it1_pkt_id.to_pass in iron/constants.py.
    static constexpr int pass_pkt_id = 2;
    /// it1_pkt_id.to_rms in iron/constants.py.
    static constexpr int rms_w_pkt_id = 0;
    /// it1_pkt_id.to_rope in iron/constants.py.
    static constexpr int rope_w_pkt_id = 1;

    static constexpr npu_tiles xyr_tile = IT2;
    static constexpr npu_tiles logits_out_tile = IT3;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    /// Four CUs across two columns, each a qk core over its kv core. The order
    /// is CU order -- column 4's two, then column 5's -- matching the q fan-out
    /// in iron/attn_hub.py.
    static constexpr npu_tiles attn_qk_tiles[4] = {CT04, CT24, CT05, CT25};
    static constexpr npu_tiles attn_kv_tiles[4] = {CT14, CT34, CT15, CT35};
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    static constexpr npu_tiles rms_tile = CT02;
    static constexpr npu_tiles gate_tile = CT22;

    int L_CHUNK = 16;

public:
    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param xclbin_mgr the xclbin manager
    /// \param context_length the context length
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*, int);

    /// \brief load weights
    ///
    /// Loads the MTP draft head. Only the stages the array currently
    /// implements are read -- today that is the fusion projection
    /// (model.mtp.fc.weight), the draft layer's k|v|q, its output gate, o_proj,
    /// and the three norms those need. The MLP joins the same walk as its
    /// phases are wired.
    ///
    /// The walk order **is** the order _gen_sequence issues _move_weights in,
    /// which **is** the order proj_main() consumes rounds in. All three move
    /// together or the cores dequantise the wrong matrix.
    ///
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&);

    /// \brief operator()
    ///
    /// One pass of the draft head at the decode shape, T = 1. The input is the
    /// fusion's two halves, **raw**, concatenated:
    ///
    ///     input[0 : D)     = e, the token embedding
    ///     input[D : 2 * D) = h, the main model's last hidden state
    ///
    /// The passthrough tile applies pre_fc_norm_embedding and
    /// pre_fc_norm_hidden on the way past, so the result is
    ///
    ///     fc_out = W_fc @ cat[rms(e, w_emb), rms(h, w_hid)]
    ///
    /// D wide. The concat exists only on the host: the array sees two D-wide
    /// packets and the order in which they arrive *is* the concatenation.
    ///
    /// \param input the fusion input, 2 * D elements, un-normalised
    /// \return **the whole layer**, D elements:
    ///
    ///         h2   = h + W_down @ (silu(gate(yn)) * up(yn)),
    ///         yn   = rms(h, post_attention_layernorm),
    ///         h    = fc_out + W_o @ (o * sigmoid(z)),
    ///         o    = softmax(q K^T / sqrt(DH)) V
    ///
    ///         over the whole cache, where z is q_proj's second DQ rows. This
    ///         closes the decoder layer: all that remains of the head is
    ///         rms(h2, mtp.norm) and lm_head, neither of which is in this
    ///         xclbin.
    ///
    ///         **Read this cosine with the residual in mind.** h2 carries h
    ///         straight through an identity term, so its raw cosine is
    ///         flattered by a gain that has nothing to do with the MLP. Score
    ///         h2 - h, not h2, when the question is whether the MLP is right;
    ///         the raw number is the one that matters only when the question is
    ///         whether the layer is. host.cpp prints both.
    ///
    ///         No intermediate stage is separately observable any more -- the
    ///         post-attention norm, the SwiGLU row and down each became the next
    ///         stage's x, and no producer's single MM2S had room for a second
    ///         destination. This row covers all of them.
    buffer<bf16> operator()(buffer<bf16>&);

    /// \brief seed the KV cache with `n` rows of context
    ///
    /// Rows [0, n), head-major and DK / DV wide. `k` and `v` are the caller's
    /// own two planes -- `k[t * DK ..]`, `v[t * DV ..]` -- and the BO keeps two
    /// planes as well, so this is a straight copy per row. It still goes through
    /// `k_at`/`v_at` rather than assuming that, so a layout flip carries it. The
    /// pass then appends its own k|v at row n and attends over all n + 1.
    void seed_kv_cache(const std::vector<float>& k, const std::vector<float>& v, int n);

    /// \brief the cos/sin pair for one position
    ///
    /// T = 1, so this is a single pair, not a table: the head drafts one token
    /// per step and every M-RoPE channel carries the same position, which is
    /// what collapses interleaved M-RoPE to plain 1-D RoPE here.
    ///
    /// Both halves are bf16 -- transformers builds the table in fp32 and casts,
    /// and the reference consumed the cast one, so rounding here is matching
    /// the reference rather than losing to it.
    ///
    /// \param position the absolute position of the drafted token
    void set_rope_position(int position);

    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
    void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
    float bench(buffer<bf16>&);

    
    /// \brief set context length
    /// \param length the context length
    void set_context_length(int length);

    /// One row of the k / v cache, DK / DV wide, head-major. Reads back what
    /// the array wrote, so after operator() row `current_context_length` is
    /// the pass's own appended k|v -- which is how the rotation and the qk
    /// norm get scored before attention reduces them away.
    ///
    /// `idx` is a row index into a plane, not a byte cursor into an interleaved
    /// row: k comes from `k_at(idx)` and v from `v_at(idx)`, a whole k plane
    /// further in. Reading v back is therefore the direct check that the append
    /// addressed the second plane -- see kv_cache_size and FLM_MAX_L.
    buffer<bf16> get_k_cache(int idx);
    buffer<bf16> get_v_cache(int idx);
};

#endif