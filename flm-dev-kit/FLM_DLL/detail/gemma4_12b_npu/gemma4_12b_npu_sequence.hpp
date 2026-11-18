// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4_12b/gemma4_12b_npu.hpp"

#define MIN_BF16_PAD 32 // minimum padding in number of bf16 elements to avoid NPU OOM when processing long sequences, which is determined empirically

#define LAYER_SPECIFIC_DIM(type) \
    int _DH = is_global_layer(type) ? DH : SWA_DH; \
    int _DQ = is_global_layer(type) ? DQ : SWA_DQ; \
    int _DK = is_global_layer(type) ? DK : SWA_DK; \
    int _DV = is_global_layer(type) ? DV : SWA_DV; \
    int _INTERMEDIATE_SIZE = INTERMEDIATE_SIZE;

typedef struct {
    uint32_t qkv_offset;
    uint32_t o_offset;
    uint32_t upgate_offset;
    uint32_t down_offset;
} gemma4_12b_buffer_offsets_t;

typedef struct {
    int D;
    int DH;
    int DQ;
    int DK;
    int DV;
    int SWA_DH;
    int SWA_DQ;
    int SWA_DK;
    int SWA_DV;
    int INTERMEDIATE_SIZE;
    int NUM_ATTENTION_HEADS;
    int NUM_KEY_VALUE_HEADS;
    int SWA_NUM_KEY_VALUE_HEADS;
    int SLIDING_WINDOW_SIZE;
    int VOCAB_SIZE_PADDED;
} gemma4_12b_seq_gen_parameters_t;

struct gemma4_12b_npu_sequence{
    typedef enum: int{
        NORMAL_DEQUANT = 0,
        UP_MATRIX      = 1,
        GATE_MATRIX    = 2
    } dequant_output_mode_t;

    /// \brief Activation the dequant_mm cores fuse into the write back of y, i.e. the
    ///        rtp_activate slot the kernel reads once the accumulator is drained.
    typedef enum: int{
        DEQUANT_MM_ACT_NONE = 0,
        DEQUANT_MM_ACT_SILU = 1,
        DEQUANT_MM_ACT_GELU = 2  //!< 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))), the
                                 //!< same approximation cpu_func::_gelu_glu uses
    } dequant_mm_act_t;

    typedef struct {
        // for decoding layer
        uint32_t l_qk_address;
        uint32_t l_kv_address;
        uint32_t swa_l_qk_address;
        uint32_t swa_l_kv_address;
        uint32_t proj_swa_address;
        uint32_t rms_swa_address;
        uint32_t lm_head_final_tune_address;
        // for others
        uint32_t place_holder_0;
    } rtp_address_book_t;

    /// \brief RTP slots of the prefill attention bitstreams.
    ///
    /// These come off attention_DH_{512,256}_prefill, not off layer.xclbin, so they are a
    /// separate book from rtp_addresses above -- the two overlays place their buffers
    /// independently and the numbers do not transfer.
    typedef struct {
        uint32_t l_begin_address;     //!< absolute position of query row 0
        uint32_t l_chunks_address;    //!< query chunks of 128 rows this run covers
        uint32_t window_size_address; //!< sliding window, or >= the context for plain causal
    } attn_prefill_rtp_book_t;

    //!< read off attention_DH_512_prefill (the global / full attention layers)
    static constexpr attn_prefill_rtp_book_t global_attn_prefill_rtp = {
        .l_begin_address     = 59520,
        .l_chunks_address    = 11520,
        .window_size_address = 59552
    };

    //!< read off attention_DH_256_prefill (the sliding window layers)
    static constexpr attn_prefill_rtp_book_t swa_attn_prefill_rtp = {
        .l_begin_address     = 61568,
        .l_chunks_address    = 11904,
        .window_size_address = 61600
    };

    /// \brief RTP slots of attention_global_image, the bidirectional variant of the DH_512
    ///        overlay used for the image / video block rows.
    ///
    /// That overlay drops the causal term outright -- every query row it is given attends kv
    /// rows [0, L) -- so the three slots do not mean what the causal build's mean, and this
    /// is a separate book even though the *addresses* coincide. They coincide because
    /// attn_global_image_aie.py declares the same three i32 buffers, in the same order, on
    /// the same tiles as attn_global_aie.py does; the meanings changed, the placement did
    /// not. Read them off the build rather than assuming that stays true.
    typedef struct {
        uint32_t kv_chunks_address;  //!< kv blocks of 128 rows the stream covers
        uint32_t q_chunks_address;   //!< query chunks of 128 rows this run covers
        uint32_t l_address;          //!< valid kv rows; every key from L on is masked off
    } attn_prefill_image_rtp_book_t;

    //!< read off attention_global_image (the vision block rows of the global layers)
    static constexpr attn_prefill_image_rtp_book_t global_image_attn_prefill_rtp = {
        .kv_chunks_address = 59520,
        .q_chunks_address  = 11520,
        .l_address         = 59552
    };

    static constexpr rtp_address_book_t rtp_addresses = {
        .l_qk_address = 57344,
        .l_kv_address = 14976,
        .swa_l_qk_address = 9216,
        .swa_l_kv_address = 40960,
        .proj_swa_address = 33280,
        .rms_swa_address = 11776,
        // Read off the 12B lm_head build (rtp_2_0 in input_with_addresses.mlir); verified on
        // hardware. 10240 was the E2B/E4B slot and on this head it is y_0_2_0, so poking it
        // corrupts an output buffer instead of setting the softcap. 
        .lm_head_final_tune_address = 49280,
        .place_holder_0 = 0x20050625
    };
    /// @brief constexprs
    static constexpr npu_tiles xr_tile  = IT3;
    static constexpr npu_tiles out_tile = IT3;
    // The kv cache is always written back through the global attention tile, but it is
    // read back by whichever engine owns the layer, see decoding_layer.hpp of the xclbin.
    static constexpr npu_tiles attn_tile     = IT2; //!< global attention
    static constexpr npu_tiles swa_attn_tile = IT4; //!< sliding window attention
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles rms_tile = CT03;
    static constexpr npu_tiles glu_tile = CT13;
    static constexpr npu_tiles rope_ct = CT23;
    static constexpr npu_tiles swa_rope_ct = CT33;
    static constexpr npu_tiles attn_qk_tile[] = {CT02, CT22};
    static constexpr npu_tiles attn_kv_tile[] = {CT12, CT32};
    static constexpr npu_tiles swa_attn_qk_tile[] = {CT04, CT24};
    static constexpr npu_tiles swa_attn_kv_tile[] = {CT14, CT34};
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };

    static constexpr int x_arg_id = 0;
    static constexpr int proj_arg_id = 1;
    static constexpr int rms_arg_id = 2;
    static constexpr int rope_rms_arg_id = 3;
    static constexpr int kv_cache_arg_id = 4;

    static constexpr int L_CHUNK = 16;

    std::vector<gemma4_12b_buffer_offsets_t> buffer_offsets;
    rtp_address_book_t e12b_rtp_addresses;

    uint32_t D;
    uint32_t DH;
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t SWA_DH;
    uint32_t SWA_DQ;
    uint32_t SWA_DK;
    uint32_t SWA_DV;

    uint32_t MAX_L;
    uint32_t HIDDEN_SIZE;
    uint32_t INTERMEDIATE_SIZE;
    uint32_t SLIDING_LENGTH;

    int num_attn_heads;
    int num_kv_heads;
    int swa_num_kv_heads;
    int num_kv_per_round;
    int swa_num_kv_per_round;

    uint32_t rms_addr;

    int VOCAB_SIZE;

    gemma4_12b_npu_sequence(){}
    gemma4_12b_npu_sequence(gemma4_12b_seq_gen_parameters_t params, uint32_t MAX_L);
    ~gemma4_12b_npu_sequence();
    void update_buffer_offsets(std::vector<gemma4_12b_buffer_offsets_t>& buffer_offsets) {
        this->buffer_offsets = buffer_offsets;
        DEBUG_BLOCK(2,
            header_print("info", "Updated buffer offsets:");
            for (size_t i = 0; i < buffer_offsets.size(); i++){
                std::string layer_info = "Type " + std::to_string(i) + ": qkv_offset=" + std::to_string(buffer_offsets[i].qkv_offset) + 
                                        ", o_offset=" + std::to_string(buffer_offsets[i].o_offset) + 
                                        ", upgate_offset=" + std::to_string(buffer_offsets[i].upgate_offset) + 
                                        ", down_offset=" + std::to_string(buffer_offsets[i].down_offset);
                std::cout << layer_info << std::endl;
            }
        )
    }

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq, gemma4_12b_layer_type_t layer_type);
    void _receive_kv_cache(npu_sequence* seq, const int L, gemma4_12b_layer_type_t layer_type);
    void _move_kv_cache(npu_sequence* seq, const size_t L, gemma4_12b_layer_type_t layer_type);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void gen_lm_head_seq(npu_sequence* seq, float final_scale);

    void gen_layer_seq(npu_sequence* seq, const uint32_t L, gemma4_12b_layer_type_t layer_type);
    /// \param L context length RTP to publish for the whole token, or < 0 to write only
    ///          the scale. Only layer 0 needs to carry it; see gen_layer_scale_seq().
    void gen_layer_scale_seq(npu_sequence* seq, const bf16 layer_scale, const int L = -1);
    void gen_dequant_mm_512(npu_sequence* seq_ptr, const uint32_t L, const uint32_t D_in, const uint32_t D_out, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t glu_slab = 0, const dequant_output_mode_t glu_part = NORMAL_DEQUANT);

    /// \brief Prefill flash attention for the global (full attention) layers, targeting the
    ///        attention_DH_512_prefill overlay rather than layer.xclbin.
    ///
    /// The run takes three buffers, in this argument order:
    ///   arg 0  o   -- [q_rows][DQ] bf16, the attention output rows this call fills
    ///   arg 1  q   -- [q_rows][DQ] bf16, RoPE'd and normed queries, head major within a row
    ///   arg 2  kv  -- the layer's kv cache, [kv_rows][DK] then [kv_rows][DV], exactly the
    ///                 layout gemma4_12b_desc::get_kv_cache_size() lays out for a global layer
    ///
    /// q and o are indexed from *row 0 of the buffer*, while the mask is built from absolute
    /// positions: row r of q is context position L_begin + r and attends to kv rows
    /// [0, L_begin + r]. The kv cache, by contrast, is indexed absolutely -- it must already
    /// hold every key and value up to L_begin + q_rows.
    ///
    /// \param L_begin  absolute context position of query row 0
    /// \param q_rows   query rows to process; a multiple of 128
    /// \param kv_rows  row capacity of the kv cache, i.e. the real MAX_L and *not* the
    ///                 padded length of this chunk. It doubles as the "no window" size, so
    ///                 understating it would silently mask off early keys.
    /// \param kv_elems total element count of the kv buffer; the V half starts at half of it
    void gen_attn_prefill_global(npu_sequence* seq, const uint32_t L_begin, const uint32_t q_rows, const uint32_t kv_rows, const size_t kv_elems);

    /// \brief Prefill flash attention for the sliding window layers, targeting the
    ///        attention_DH_256_prefill overlay.
    ///
    /// Same three run arguments as gen_attn_prefill_global(), with SWA dims:
    ///   arg 0  o   -- [q_rows][SWA_DQ] bf16
    ///   arg 1  q   -- [q_rows][SWA_DQ] bf16
    ///   arg 2  kv  -- [kv_rows][SWA_DK] then [kv_rows][SWA_DV] bf16
    ///
    /// The difference that matters is what "position" means. The overlay indexes kv rows
    /// by the same counter it masks with, so it cannot be pointed at the decoding ring,
    /// which is indexed by (position % sliding). The caller instead hands it the *unrolled*
    /// window -- kv row 0 being the oldest key any query row can still see -- and shifts
    /// L_begin down by that row's absolute position. Both mask terms (`k <= q` and
    /// `k > q - window`) are differences of positions, so a shift applied to the query and
    /// the key frames alike leaves them unchanged.
    ///
    /// \param L_begin  position of query row 0 *relative to kv row 0*, i.e.
    ///                 min(absolute L_begin, window - 1) once the window has filled
    /// \param q_rows   query rows to process; a multiple of 128
    /// \param kv_rows  row capacity of the unrolled kv buffer. The overlay streams keys up
    ///                 to L_begin + q_rows + 127 and masks the tail off, so the buffer has
    ///                 to be that tall even though only part of it holds real keys.
    /// \param window   sliding window, in positions. A multiple of 128, which is what makes
    ///                 the kernel's kv block count agree with the one baked in here.
    /// \param kv_elems total element count of the kv buffer; the V half starts at half of it
    void gen_attn_prefill_swa(npu_sequence* seq, const uint32_t L_begin, const uint32_t q_rows, const uint32_t kv_rows, const uint32_t window, const size_t kv_elems);

    /// \brief Prefill attention for the query rows of one image / video block of a global
    ///        layer, targeting the attention_global_image overlay.
    ///
    /// Same three run arguments and the same DH_512 floorplan as gen_attn_prefill_global():
    ///   arg 0  o   -- [q_rows][DQ] bf16
    ///   arg 1  q   -- [q_rows][DQ] bf16, the block's query rows starting at *row 0*
    ///   arg 2  kv  -- the layer's kv cache, [kv_rows][DK] then [kv_rows][DV]
    ///
    /// What changes is the mask. The overlay has no causal term at all: every query row it
    /// is given attends kv rows [0, kv_valid_rows), and the only masking left is the ragged
    /// tail of the last 8 key group. That is exactly the vision block rule of a global
    /// layer -- a query inside a block sees [0, block_end), *the same range for every row of
    /// the block*, because the causal bound opens up to the block end and the window bound
    /// is absent (see gemma4_12b_mm_attn_block::_stage_vision_blocks and
    /// models_simple/GEMMA4_12B_QAT_MASKS.md). A row independent range over an extent that
    /// starts at position 0 is all this overlay can express, and all the rule needs.
    ///
    /// Since kv extent row 0 *is* context position 0, the decoding cache goes in verbatim,
    /// exactly as it does for the causal pass. There is no staging buffer and no relative
    /// frame -- unlike the sliding kind, whose ring forces both.
    ///
    /// \param q_rows        query rows to process, rounded up to a multiple of 128. The
    ///                      rounding tail is computed with the block mask, which is wrong for
    ///                      it, so the caller must not write those rows back.
    /// \param kv_valid_rows keys the block may see, i.e. min(block end, L_end). Everything
    ///                      from here to the end of the streamed extent is masked off.
    /// \param kv_rows       row capacity of the kv cache, i.e. the real MAX_L. The stream is
    ///                      rounded up to whole 128 row blocks and must stay inside it.
    /// \param kv_elems      total element count of the kv buffer; the V half starts at half
    ///                      of it. Never L_padded * DK -- that is the standalone testbench's
    ///                      geometry, not the DLL's.
    void gen_attn_prefill_global_image(npu_sequence* seq, const uint32_t q_rows, const uint32_t kv_valid_rows, const uint32_t kv_rows, const size_t kv_elems);

    void set_max_length(const uint32_t MAX_L);
};

// ===========================================================================
//  The multi modal projector overlay -- audio_image_mm
//
//  These are free functions rather than members of gemma4_12b_npu_sequence
//  because a plain bf16 x bf16 matmul needs nothing from the model
//  description: M, K and N are the whole of it. That is also what lets
//  gemma4_12b_mm_projector stay a leaf, depending on dimensions rather than
//  on the model object that owns it.
// ===========================================================================

/// \brief The audio_image_mm floorplan, shared by the two generators below and by the
///        projector that sizes buffers for them.
///
/// The bitstream is *two independent engines* in one overlay, split down the middle:
///
///   cols 0..3, IT0..IT3   the plain half, `mm.cc`,      K tile hard wired to 640
///   cols 4..7, IT4..IT7   the bias half,  `mm_bias.cc`, K tile 768, bias optional
///
/// Shim column c of a half owns row band c of x and y *and* output column band c of w,
/// because MT[c] is both the w source for core column c and the x source for core row c
/// (see iron/mm_aie.py). A "round" therefore covers cols * m = 256 rows and cols * n = 256
/// output columns at once, which is where the two alignment constants come from.
struct mm_projector_geometry {
    static constexpr uint32_t cols    = 4;   //!< shim columns per half
    static constexpr uint32_t rows    = 4;   //!< core rows per column
    static constexpr uint32_t m       = 64;  //!< y rows one core produces per block, NPU_M_TILE
    static constexpr uint32_t n       = 64;  //!< y columns one core produces per block, NPU_N_TILE
    static constexpr uint32_t k_plain = 640; //!< K tile of the plain half, NPU_K_TILE
    static constexpr uint32_t k_bias  = 768; //!< K tile of the bias half, NPU_K_TILE_BIAS

    static constexpr uint32_t M_ALIGN = m * cols;  //!< 256; pad the row count up to this
    static constexpr uint32_t N_ALIGN = n * cols;  //!< 256
};

/// \brief RTP slots of the multi modal projector overlay.
///
/// A third overlay means a third address book: these come off mm_projector.xclbin and share
/// nothing with layer.xclbin's rtp_addresses or with attn_prefill_rtp_book_t. Only the bias
/// half has RTPs at all -- the plain half's loop bounds are compile time constants in
/// mm.cc, which is also why its K is not a parameter.
struct mm_projector_rtp_book_t {
    uint32_t rtp_base_address;  //!< four consecutive u32: N blocks, add_bias, M blocks, K tiles
    uint32_t sync_lock_address; //!< the core blocks on this until the four above are written
};

static constexpr mm_projector_rtp_book_t mm_projector_rtp = {
    .rtp_base_address  = 57600,
    //!< core tile lock base 0x1F000, 16 bytes per lock, l_prod_lock is id 6
    .sync_lock_address = 0x1F000 + 16 * 6
};

/// \brief y[M, N] = x[M, K] * w[N, K]^T on the *plain* half of the projector overlay.
///
/// Run arguments, in order: arg 0 `y` [M][N], arg 1 `x` [M][K], arg 2 `w` [N][K]. All three
/// are plain row major bf16 -- w in the checkpoint's own [out_features, in_features]
/// convention, so nothing has to be repacked on the way in.
///
/// \param M rows, a multiple of mm_projector_geometry::M_ALIGN
/// \param K must be exactly mm_projector_geometry::k_plain; mm.cc's K loop is static
/// \param N output columns, a multiple of mm_projector_geometry::N_ALIGN
void gen_mm_projector_plain(npu_sequence* seq, const uint32_t M, const uint32_t K,
                            const uint32_t N);

/// \brief Same product on the *bias* half, optionally seeding the accumulator with a bias.
///
/// Run arguments are the same three buffers, but `w` carries the bias in front of the
/// weights when `add_bias` is set: `[N] bias | [N][K] weight`, and the generated descriptors
/// read the weight region from element offset N. That packing is not a convention this
/// function is free to change -- the bias has nowhere else to ride, because a run takes
/// exactly three buffers.
///
/// \param K a multiple of mm_projector_geometry::k_bias
/// \param add_bias true for image patch_embd, false for the two input projections
void gen_mm_projector_bias(npu_sequence* seq, const uint32_t M, const uint32_t K,
                           const uint32_t N, const bool add_bias);

