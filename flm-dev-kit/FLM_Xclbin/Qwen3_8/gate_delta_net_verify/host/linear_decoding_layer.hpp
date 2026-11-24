// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LINEAR_DECODING_LAYER_HPP__
#define __LINEAR_DECODING_LAYER_HPP__

#include "models.h"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "tensor_2d.hpp"
#include "quant_size.hpp"

// Query rows per forward pass, normally set by the top-level Makefile via
// -DL_TOKENS. Repeated here so the header stands alone in an IDE.
#ifndef L_TOKENS
#define L_TOKENS 8
#endif

/// \brief One gated delta-net layer, end to end: input_layernorm + qkv_proj +
///        causal conv1d + z (gate) proj + the a / b projections + the gated
///        delta rule scan + the gated RMSNorm + ssm_out_proj + the residual +
///        post_attention_layernorm + the MLP (up | gate, SwiGLU, down) + the
///        second residual.
///
/// Delta-net only. Qwen3.8-27B's every-fourth full-attention layer is a separate
/// xclbin with its own host class; see ../../gated_attention_verify. There is no
/// layer-type select here and no attention state to carry.
///
/// Per pass this sends L_TOKENS rows of hidden states, the two layernorm weight
/// vectors, the conv taps, ssm_norm and conv cache, the q4_k qkv / z / o /
/// up|gate / down weights and the q8_0 a/b weights.
///
/// Two results, one round trip, nothing else: h2 (IT3 S2MM ch1, returned by
/// operator()) and the causal conv window (IT5 S2MM ch0, get_conv_out()).
///
/// The recurrent state used to be a third, on IT4 S2MM ch0. It is not read back
/// at all now: the cores can only report S after *all* L_TOKENS steps, and
/// speculative decode needs S after whichever prefix verification accepts, so
/// the host replays it from the triplet arena instead (get_all_S_at()). S is
/// still sent *in* every pass -- the scan has to start somewhere -- it just does
/// not come back. advance_state() rolls both carried states forward.
/// The only other DDR traffic is one the array itself depends on:
/// IT6 S2MM ch0 + IT5 S2MM ch1 scatter the conv tile's k | v | q triplets and
/// the a/b tile's alpha / beta into the triplet arena, which goes back out to
/// the delta cores head-major -- that round trip *is* the token-major ->
/// head-major transpose. Everything else stays on the array: o and down are
/// un-permuted in MT[3] and handed one hop north into the rms tile, and the
/// post-attention norm goes straight to MT[2] as up|gate's x.
///
/// The MLP's hidden state is the one activation that has to fill while another
/// is still replaying, so it is the one `dedicated` phase in
/// iron/deltanet_layer_aie.py, with its own 272 KiB buffer and inbound channel
/// on MT[2].
class linear_decoding_layer {
private:
    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;

    /// Positional against layer_app()'s five buffer arguments in operator().
    int x_arg_id = 0;
    int proj_arg_id = 1;
    int rms_arg_id = 2;
    int aux_w_arg_id = 3;
    int kv_cache_arg_id = 4;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    /// | conv taps | ssm_norm |, the small per-layer weights that land in MT[5],
    /// adjacent so one transfer feeds both of that memtile's BDs. Named for RoPE
    /// once; it has held none of it since stage 7a. It held the a/b activation
    /// bias too until the bias moved into the a/b weight stream -- see
    /// w_ab_bias_offset.
    buffer<bf16> aux_w_buffer;
    buffer<bf16> kv_cache;

    int num_hidden_layers;

    /// \param stages -1 builds the whole pass; >= 0 stops after that many
    ///        numbered steps, which is how probe_prefix() bisects a wedged array.
    void _gen_linear_sequence(npu_sequence* seq, int stages = -1);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _receive_ab(npu_sequence* seq);
    void _receive_arena(npu_sequence* seq);
    /// h2 = h1 + down_proj(hid), off the rms tile's third phase. The layer's
    /// output, and the last thing the pass produces.
    void _receive_layer_out(npu_sequence* seq);
    void _send_arena_to_delta(npu_sequence* seq);
    void _send_linear_conv_weights(npu_sequence* seq);
    void _receive_linear_kv_cache(npu_sequence* seq);
    void _move_linear_kv_cache(npu_sequence* seq);
    void _move_s(npu_sequence* seq);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);


    // model specific config
    int D;
    int Linear_DQ;
    int Linear_DK;
    int Linear_DV;
    int qkv_out;      ///< Linear_DQ + Linear_DK + Linear_DV, the qkv projection width
    int z_out;        ///< Linear_DV, the z (gate) projection width
    int ab_out;       ///< 2 * linear_num_v_heads_padded, the alpha|beta f32 count per token

    /// The DDR triplet arena that reorders the conv output token-major ->
    /// head-major for the delta cores. One (head, token) block is
    ///
    ///   [ k(128) | alpha(2) | v(128) | beta(2) | q(128) pad(2) ]
    ///
    /// i.e. three triplet_stride-word sub-blocks. The conv tile scatters k, v
    /// and q into it; the a/b tile scatters alpha and beta into the two-word
    /// holes. Blocks are ordered [head][token], so reading heads 0 ..
    /// linear_num_v_heads back out is a single linear DMA.
    int triplet_stride;   ///< linear_v_head_dim + 2
    int arena_block;      ///< 3 * triplet_stride
    int arena_heads;      ///< linear_num_v_heads_padded -- a/b emits padded heads
    /// Where each DDR region starts in hidden_states_buffer, in bf16 words:
    /// | x | arena | h2 |.
    size_t arena_offset;
    size_t layer_out_offset;
    int linear_num_qk_heads;
    int linear_num_v_heads;
    int linear_num_v_heads_padded;
    int linear_qk_head_dim;
    int linear_v_head_dim;
    int conv_kernel_size;
    int intermediate_size;

    size_t linear_kv_cache_size;
    size_t conv_w_size;
    size_t linear_w_qkv_offset;
    size_t linear_w_gate_offset;
    size_t linear_w_o_offset;
    size_t linear_w_upgate_offset;
    size_t linear_w_down_offset;
    /// One AB_W_BLOCK_SIZE-word block carrying the a/b activation bias --
    /// A in the first linear_num_v_heads_padded f32 lanes, ssm_dt.bias in the
    /// next, the remainder zero. It leads the a/b weight region because the a/b
    /// tile takes it as an ordinary weight block: see _move_alpha_beta_weights().
    size_t w_ab_bias_offset;
    size_t w_ab_offset;
    /// AB_W_BLOCK_SIZE in bf16 words -- the a/b tile's w_buffer_0/1 size, which
    /// is what every BD on that channel transfers.
    size_t ab_w_block_words;
    /// The two carried states, both in kv_cache, laid out
    /// | conv in | S in | conv out |.
    ///
    /// S has an in region only: the array does not write it back. advance_state()
    /// closes the loop for a multi-pass run, replaying S over s_offset.
    size_t s_offset;
    size_t s_words;   ///< bf16 words of one full recurrent state
    /// The causal conv1d window going *in*: the last conv_kernel_size - 1
    /// *pre-conv* qkv rows, which is what the next pass convolves its first
    /// token against.
    size_t conv_words;
    size_t conv_out_offset;
    /// The window region coming *out*, which is larger. Speculative decode does
    /// not know at dispatch time how many of the L_TOKENS drafted tokens will
    /// survive verification, so one window is not enough: accept-j needs the
    /// last conv_kernel_size - 1 pre-conv rows *as of token j*. Those runs
    /// overlap, so their union is one contiguous
    /// L_TOKENS + conv_kernel_size - 2 row block and get_conv_out(j) just
    /// indexes into it.
    size_t conv_out_words;
    /// Heap staging for get_all_S_at(). The scan re-reads S_0 dk times per
    /// token, and reads that land on an XRT BO cost ~100x what the same reads
    /// cost off ordinary memory, so the inputs are copied out once per call.
    /// Members rather than locals so the allocation is paid once.
    std::vector<float> s_stage;
    std::vector<bf16> arena_stage;

    // npu configs

    /// The rms tile's own shim, in column 3: x and both norm weights in on
    /// MM2S, post_attention_layernorm back out on S2MM ch0 and h2 on ch1.
    static constexpr npu_tiles xyr_tile = IT3;
    /// The four projection columns' shims, in engine order.
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};

public:
    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param layer_app_mgr the app manager the layer's sequence is built on
    linear_decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*);

    /// \brief load weights
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&);

    /// \brief Run one pass over L_TOKENS rows.
    /// \param input L_TOKENS x D hidden states, row-major.
    /// \return h2 = h1 + mlp.down_proj(hid), the layer's output, L_TOKENS x D
    ///         bf16 in row-major token order -- what the next layer takes as its
    ///         input. The rms tile emits whole rows, so nothing has to be
    ///         un-permuted here.
    buffer<bf16> operator()(buffer<bf16>&);

    /// \brief Run one pass with every device-written region poisoned first, and
    ///        report which of them the pass actually reached.
    ///
    /// A stalled dispatch is invisible to a value check: the output BO still
    /// holds the previous pass, so h2 compares bit-identical and the bench
    /// scores a success. Poisoning first turns "did it run" into a question the
    /// host can answer, and turns *where* it stopped into one too -- the three
    /// regions are written by three different points of the pipeline, so the
    /// first one left poisoned is downstream of the stall.
    ///
    ///  * arena -- the conv and a/b tiles, early: their scatter is the first
    ///    thing in the pass that reaches DDR at all.
    ///  * conv out -- MT[5], once the whole window has passed through.
    ///  * h2 -- the rms tile's third phase, i.e. the very end of the layer.
    ///
    /// \param input the same L_TOKENS x D rows operator() takes.
    /// \param label printed with each line, to tell dispatches apart.
    /// \return true if the pass completed and touched all three regions.
    bool probe_pass(buffer<bf16>& input, const std::string& label);

    /// \brief Run `n` passes as a single runlist, and report where it stopped.
    ///
    /// The other shape the engine could use. A runlist is one submission of n
    /// commands against one hardware context, so if the array survives n passes
    /// this way but not as n separate commands, the difference is in how XRT
    /// hands the array over between commands and not in the design's own ledger.
    bool probe_runlist(buffer<bf16>& input, int n);

    /// \brief Rebuild the sequence truncated to `stages` steps and dispatch it.
    ///
    /// For finding which transfer a wedged array stops on. Destructive: the
    /// array is left mid-pass, so one call per process.
    bool probe_prefix(buffer<bf16>& input, int stages);


    /// \brief One head's recurrent state after just the first `accept_len`
    ///        of the L_TOKENS drafted tokens.
    /// \param head the v head, 0 .. linear_num_v_heads - 1.
    /// \param accept_len how many drafted tokens verification kept, 1 .. L_TOKENS.
    /// \return linear_qk_head_dim * linear_v_head_dim floats, [dk][dv].
    ///
    /// Reconstructed on the host. Nothing is read back from the array: the delta
    /// cores keep S resident for all L_TOKENS steps precisely so it crosses DDR
    /// once per pass instead of twice per token, and emitting the intermediate
    /// states would not fit MT[4] anyway -- eight 64 KB state buffers is the
    /// whole 512 KB tile.
    ///
    /// It does not have to. The triplet arena already holds the exact bf16 k, v,
    /// alpha and beta the cores consumed, for every head and token, and the S
    /// input region holds S_0, so
    ///
    ///     u_t = beta_t * (v_t - alpha_t * (S_{t-1} @ k_t))
    ///     S_t = alpha_t * S_{t-1} + k_t (x) u_t
    ///
    /// replays exactly what the core did. Note beta is folded into u_t rather
    /// than multiplying the outer product -- applying it in both places would
    /// square it.
    ///
    /// This replays from S_0 each call, so walking every head and every accept
    /// length is O(L^2) in state passes. Callers that need many of them should
    /// ask for the lengths they need in increasing order and cache.
    buffer<float> get_S_at(int head, int accept_len);

    /// \brief Every head's state at `accept_len`, in one OpenMP-parallel sweep.
    /// \param accept_len 1 .. L_TOKENS.
    /// \param out filled with linear_num_v_heads * dk * dv floats, head-major --
    ///        the same layout the s_offset region takes, so carrying it forward
    ///        is a straight memcpy.
    ///
    /// This is the unit of work worth overlapping with a dispatch: one call is
    /// ~0.09 ms against a ~28 ms pass. Prefer it to looping get_S_at(), which
    /// re-syncs the BO per head and leaves the parallelism on the floor.
    void get_all_S_at(int accept_len, buffer<float>& out);

    /// \brief Carry a reconstructed state forward as the next pass's S input.
    /// \param S linear_num_v_heads * dk * dv floats, as get_all_S_at() fills.
    ///
    /// advance_state() copies the device's accept-all state, which is only
    /// right when every drafted token survived. For a shorter prefix, call this
    /// with the replayed state instead -- otherwise the next pass continues
    /// from a state that ran past the accepted prefix.
    void set_S_in(const buffer<float>& S);

    /// \brief Just the kv_cache BO sync get_all_S_at() does, for attributing
    ///        its cost between the transfer and the arithmetic.
    void sync_state_in();

    /// \brief The causal conv1d window after the last operator() call, for a
    ///        given accept length.
    /// \param accept_len how many of the L_TOKENS drafted tokens verification
    ///        kept, 1 .. L_TOKENS. L_TOKENS is "all accepted".
    /// \return (conv_kernel_size - 1) * (DQ + DK + DV) bf16, [tap][channel],
    ///         oldest tap first -- the same layout the pass reads its cache in
    ///         as, and the same as transformers' conv_states.
    buffer<bf16> get_conv_out(int accept_len = L_TOKENS);

    /// \brief Roll both carried states forward to the accepted prefix, for the
    ///        next pass.
    ///
    /// Call it between operator() calls; without it a second pass re-runs
    /// against the first pass's *input* state.
    ///
    /// \param accept_len how many drafted tokens verification kept. Honoured by
    ///        both halves: it selects the conv window, and it is the prefix the
    ///        state is replayed to. Callers no longer have to rebuild S
    ///        themselves for accept_len < L_TOKENS -- this does it.
    void advance_state(int accept_len = L_TOKENS);

    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);

    /// \brief Steady-state latency: num_hidden_layers passes back to back on one
    ///        runlist, so XRT dispatch is amortised instead of dominating.
    float bench(buffer<bf16>&);

    /// \brief Start a pass without waiting for it, so the host can work while
    ///        the array runs.
    ///
    /// The engine's loop is a chain of dependent layers, so the NPU work cannot
    /// overlap *itself* -- layer i+1 needs layer i's h2. What can overlap is the
    /// host-side state reconstruction: get_S_at() for layer i reads only layer
    /// i's arena and S_0, neither of which layer i+1's dispatch consumes, so it
    /// can run in the dispatch's shadow.
    ///
    /// The catch, and the reason this is a pair of calls rather than a thread
    /// inside operator(): every pass writes the *same* hidden_states_buffer and
    /// kv_cache. One layer object cannot have two passes in flight, and the
    /// replay of pass n must finish before pass n+1 is started on the same
    /// object or it reads a half-overwritten arena. Overlap therefore needs two
    /// layer objects (the engine has one per layer anyway), and the discipline
    /// is: start(i+1), replay(i), wait(i+1).
    ///
    /// \param input L_TOKENS x D hidden states, copied before the run starts.
    /// \return the run handle to hand to finish_pass().
    xrt::run start_pass(buffer<bf16>& input);

    /// \brief Wait for a start_pass() run and collect h2.
    /// \return h2, exactly as operator() returns it.
    buffer<bf16> finish_pass(xrt::run& run);
};

#endif