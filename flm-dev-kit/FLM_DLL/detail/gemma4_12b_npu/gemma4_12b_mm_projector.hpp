// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file gemma4_12b_mm_projector.hpp
/// \brief The gemma4-12b multi modal projector: raw pixels / raw PCM -> language embeddings
/// \author FastFlowLM Team
///
/// gemma4-12b is *encoder free*. There is no SigLIP tower and no conformer: a 48x48 RGB
/// tile and a 640 sample PCM frame each become one HIDDEN_SIZE vector through a couple of
/// norms and at most two matmuls. That whole front end is this file.
///
///   vision   LN(patch_norm.1)            6912 wide, torch eps 1e-5, gamma+beta
///          -> dense patch_embd           [3840, 6912] + bias
///          -> LN(patch_norm.2)           3840 wide
///          -> + posemb[axis0][x] + posemb[axis1][y]     (factorized, not a 2d table)
///          -> LN(patch_norm.3)           3840 wide
///          -> RMSNorm, no scale          eps 1e-6, no gamma, no mean subtraction
///          -> dense mm.input_projection  [3840, 3840], no bias
///
///   audio    RMSNorm, no scale           640 wide, eps 1e-6
///          -> dense mm.a.input_projection [3840, 640], no bias
///
/// \note The rows this projector produces are handed to the language model *as is*. They
///       are NOT multiplied by the sqrt(hidden_size) embedding scale -- that scale belongs
///       to the token embedding table only. Scaling media rows by it is the classic
///       silent failure of this path: the model still speaks fluently and simply ignores
///       the image.
///
/// \note The projector deliberately knows nothing about payload structs, token ids or the
///       npu model object. It takes dimensions (gemma4_12b_mm_dims, built from the model
///       config) and rows of already valid features. Compaction of padded rows and the
///       scatter into the prompt embeddings are the caller's job.
#ifndef __GEMMA4_12B_MM_PROJECTOR_HPP__
#define __GEMMA4_12B_MM_PROJECTOR_HPP__

#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "avx512_util.hpp"
#include "utils/utils.hpp"

#include <cmath>
#include <cstring>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

// ===========================================================================
//  Backend selection
//
//  The projector's matmuls -- patch_embd (6912 -> 3840) and the two input
//  projections -- can run either on the host or on the array. Pick at compile
//  time, from the Makefile:
//
//    make MM_PROJ=npu   # no define             (default)
//    make MM_PROJ=cpu   # -DFLM_MM_PROJ_ON_CPU
//
//  The switch is an opt *out*, so the array path is what a plain `make` builds
//  and the host path is the fallback. Keep this polarity the same as the
//  Makefile's: they only agree by convention, and if they drift the build goes
//  silently to the host backend rather than failing.
//
//  Same shape as the prefill's PREFILL_MM switch, and for the same reason: on
//  the host backend no xclbin is registered and every buffer here is plain host
//  memory, so the path runs on a machine with no device at all.
//
//  Only the three dense() calls move. The LayerNorms, the RMS norm and the
//  factorized position embedding stay on the host in both configurations --
//  they are a rounding error next to patch_embd's 6.8 GMAC, and keeping them
//  in one place keeps the two backends comparable row for row.
// ===========================================================================
#ifdef FLM_MM_PROJ_ON_CPU
#define GEMMA4_12B_MM_PROJ_ON_NPU 0
#else
#define GEMMA4_12B_MM_PROJ_ON_NPU 1
#endif

/// \brief Host kernels of the projector. The signatures are what an NPU backend has to
///        match: same buffers, same row major layouts, same call order.
namespace gemma4_12b_mm_cpu {

/// \brief torch nn.LayerNorm over the last axis, in f32.
/// \param gamma  scale, one per column
/// \param beta   shift, one per column
/// \param eps    torch's default is 1e-5 -- NOT the model's 1e-6 rms eps
/// \note  The variance is the *biased* one (divided by n, not n-1), like torch.
inline void layer_norm(bf16* out, const bf16* in, const bf16* gamma, const bf16* beta,
                       size_t rows, int n, float eps) {
    const float inv_n = 1.0f / (float)n;
    #pragma omp parallel for schedule(static)
    for (long long m = 0; m < (long long)rows; m++) {
        const bf16* src = in  + (size_t)m * n;
        bf16*       dst = out + (size_t)m * n;

        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += (float)src[i];
        mean *= inv_n;

        float var = 0.0f;
        for (int i = 0; i < n; i++) {
            const float d = (float)src[i] - mean;
            var += d * d;
        }
        var *= inv_n;

        const float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < n; i++) {
            dst[i] = (bf16)((((float)src[i] - mean) * inv) * (float)gamma[i] + (float)beta[i]);
        }
    }
}

/// \brief RMS norm with `with_scale = false`: no gamma, no mean subtraction, in f32.
/// \note  This is the gemma flavour, eps 1e-6, and it is a different function from
///        layer_norm() above in three ways at once. Mixing them up is the easiest way to
///        get a projector that "almost" works.
inline void rms_norm_no_scale(bf16* out, const bf16* in, size_t rows, int n, float eps) {
    const float inv_n = 1.0f / (float)n;
    #pragma omp parallel for schedule(static)
    for (long long m = 0; m < (long long)rows; m++) {
        const bf16* src = in  + (size_t)m * n;
        bf16*       dst = out + (size_t)m * n;

        float ms = 0.0f;
        for (int i = 0; i < n; i++) {
            const float v = (float)src[i];
            ms += v * v;
        }
        ms *= inv_n;

        const float inv = 1.0f / std::sqrt(ms + eps);
        for (int i = 0; i < n; i++) dst[i] = (bf16)((float)src[i] * inv);
    }
}

/// \brief y[m][n] = sum_k x[m][k] * w[n][k] (+ bias[n]), f32 accumulation.
/// \param w     [N, K] row major, i.e. the checkpoint's [out_features, in_features]
///              convention used verbatim -- no load time transpose, each output column is
///              one contiguous dot product.
/// \param bias  may be null (both input projections have no bias)
/// \note This is the one function the NPU backend replaces, with a run on the
///       audio_image_mm overlay. The array wants exactly this layout for all three
///       buffers, which is why the weights are never repacked on either backend.
inline void dense(bf16* out, const bf16* x, const bf16* w, const bf16* bias,
                  size_t rows, int K, int N) {
    const int K_vec = (K / 16) * 16;
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long m = 0; m < (long long)rows; m++) {
        for (long long n = 0; n < (long long)N; n++) {
            const bf16* xr = x + (size_t)m * K;
            const bf16* wr = w + (size_t)n * K;

            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k < K_vec; k += 16) {
                acc = _mm512_fmadd_ps(load_bfloat16_to_m512(xr + k),
                                      load_bfloat16_to_m512(wr + k), acc);
            }
            float s = _mm512_reduce_add_ps(acc);
            for (; k < K; k++) s += (float)xr[k] * (float)wr[k];
            if (bias != nullptr) s += (float)bias[n];

            out[(size_t)m * N + n] = (bf16)s;
        }
    }
}

/// \brief in place `row += posemb[0][x] + posemb[1][y]`.
///
/// The position embedding is *factorized*: one table per axis, looked up with the patch's
/// x and y separately and added, rather than one table indexed by a flattened 2d
/// position. The checkpoint stores both tables in a single [2, posemb_size, dim] tensor,
/// axis major, so axis a of coordinate c starts at (a * posemb_size + c) * dim.
///
/// \param pos_ids [rows][2] as (x, y); a row with either coordinate < 0 is padding and is
///                left untouched (the caller is expected to have dropped it already).
inline void add_factorized_pos_embedding(bf16* rows_inout, const int* pos_ids,
                                         const bf16* table, size_t rows,
                                         int dim, int posemb_size) {
    #pragma omp parallel for schedule(static)
    for (long long m = 0; m < (long long)rows; m++) {
        const int px = pos_ids[(size_t)m * 2 + 0];
        const int py = pos_ids[(size_t)m * 2 + 1];
        if (px < 0 || py < 0 || px >= posemb_size || py >= posemb_size) continue;

        bf16*       dst = rows_inout + (size_t)m * dim;
        const bf16* ex  = table + ((size_t)0 * posemb_size + px) * dim;
        const bf16* ey  = table + ((size_t)1 * posemb_size + py) * dim;
        for (int i = 0; i < dim; i++) {
            dst[i] = (bf16)((float)dst[i] + (float)ex[i] + (float)ey[i]);
        }
    }
}

} // namespace gemma4_12b_mm_cpu

/// \brief Allocate `n` bf16 for the compiled in backend: a device BO when the projector
///        runs on the array, plain host memory otherwise.
inline buffer<bf16> gemma4_12b_mm_proj_alloc(npu_app_manager* mgr, size_t n) {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    return mgr->create_bo_buffer<bf16>(n);
#else
    (void)mgr;
    return buffer<bf16>(n);
#endif
}

/// \brief Push host writes to the device; a no-op when the buffers are host memory.
/// \note buffer<T>::sync_to_device() asserts on a non BO buffer, so this cannot just be
///       called unconditionally.
inline void gemma4_12b_mm_proj_push(buffer<bf16>& b) {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    b.sync_to_device();
#else
    (void)b;
#endif
}

/// \brief Pull device writes back to the host; a no-op when the buffers are host memory.
inline void gemma4_12b_mm_proj_pull(buffer<bf16>& b) {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    b.sync_from_device();
#else
    (void)b;
#endif
}

/// \brief Everything the projector needs to know about the model, read from config.json /
///        processor_config.json once.
///
/// This is what replaces a back pointer to gemma4_12b_npu. The projector is a leaf: it
/// depends on dimensions, not on the model object that happens to own it, which keeps it
/// testable on its own and keeps the vision/audio preprocessing parameters where they
/// belong (on the model, next to the preprocessor that consumes them).
struct gemma4_12b_mm_dims {
    // --- language side ---
    int hidden_size     = 3840;   //!< width of one projected row, == the model's D

    // --- vision ---
    int patch_size      = 16;     //!< pixels per raw patch edge
    int pooling_kernel  = 3;      //!< raw patches pooled per model patch edge
    int model_patch     = 48;     //!< patch_size * pooling_kernel
    int patch_dim       = 6912;   //!< model_patch^2 * 3, the width of one raw tile
    int mm_embed_dim    = 3840;   //!< width inside the vision projector
    int mm_posemb_size  = 1120;   //!< rows per axis of the factorized position table
    int img_max_soft_tok = 280;   //!< rows one image can contribute
    int vid_max_soft_tok = 70;    //!< rows one video frame can contribute

    // --- audio ---
    int audio_embed_dim   = 640;  //!< raw PCM samples in one frame == one row
    int audio_max_soft_tok = 750; //!< 30 s at 40 ms per token

    // --- norms ---
    float rms_norm_eps   = 1e-6f; //!< the model's own rms eps
    float layer_norm_eps = 1e-5f; //!< torch nn.LayerNorm default, not read from config

    /// \brief Fill from the model config; every field keeps its released default when the
    ///        checkpoint does not carry the key.
    static gemma4_12b_mm_dims from_config(const LM_Config& config);

    /// \brief Human readable one liner for the debug log.
    std::string _str() const;
};

/// \brief One projection that has been issued and not yet collected.
///
/// On an NPU backend it would hold the xrt::run so the caller can overlap host work; on
/// the host backend the matmul has already run by the time the handle exists and wait()
/// has nothing to do. Call sites are identical on both.
struct gemma4_12b_mm_proj_run {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    flm_rt::run      run;
    buffer<bf16>* dst = nullptr;
    void wait() {
        if (dst == nullptr) return;
        flm::npu::wait_for_completion(run, "gemma4_12b_mm_proj_run");
        dst->sync_from_device();
        dst = nullptr;
    }
#else
    void wait() {}
#endif
};

/// \brief The gemma4-12b image + audio projector.
///
/// One instance owns both towers: they share nothing but the output width, but they do
/// come out of one checkpoint file and have the same lifetime, and keeping them together
/// is what lets a prompt with an image and an audio clip touch a single object.
///
/// Construction is cheap; load_weights() is what reads the ~105 MB checkpoint, so the
/// owner is expected to build this lazily, on the first prompt that actually has media.
class gemma4_12b_mm_projector {
public:
    /// \param npu     used only when the projector is compiled for the array; ignored on
    ///                the host backend, so a text only or device-less session pays nothing
    /// \param config  the model config, for gemma4_12b_mm_dims and for the weights path
    gemma4_12b_mm_projector(npu_xclbin_manager* npu, LM_Config& config);
    ~gemma4_12b_mm_projector();

    /// \brief Read the projector weights out of an already opened checkpoint.
    /// \note  Audio-only or vision-only checkpoints are accepted: the missing tower simply
    ///        reports has_vision() / has_audio() false and refuses to run.
    void load_weights(SafeTensors& st);

    /// \brief Same, opening the default checkpoint: config's "mm_projector_weight" /
    ///        "vision_model_weight" if present, else <model_path>/image_audio_weights.q4nx.
    void load_weights();

    /// \brief Project `rows` image patches into `rows` language embeddings.
    /// \param patches [rows][patch_dim] raw tiles, already rescaled by the preprocessor
    /// \param pos_ids [rows][2] as (x, y), the patch's coordinates in the pooled grid
    /// \param out     [rows][hidden_size], written in full
    /// \note Every row must be a *valid* patch. The preprocessor pads its payload out to
    ///       img_max_soft_tok rows with zeros and (-1, -1) positions; those rows are not
    ///       zero after the first LayerNorm+dense, they are finite garbage, so the caller
    ///       has to drop them before getting here rather than letting them through.
    void project_vision(const bf16* patches, const int* pos_ids, size_t rows, bf16* out);

    /// \brief Reorder the patch input axis of ln1 and of the dense weight from the
    ///        checkpoint's CHW order into the HWC order the preprocessor emits.
    ///
    /// The packed tensors come from llama.cpp's mmproj, where the patch embedding is a
    /// conv weight [out, C, H, W] flattened in place, so its K axis (and the ln1 gamma /
    /// beta that share it) is indexed c * 48 * 48 + h * 48 + w. The image preprocessor
    /// follows the reference processor and hands out patches in HWC, (h * 48 + w) * 3 + c.
    /// Permuting the weights once here keeps the hot path a plain dense and leaves the
    /// preprocessor matching the reference.
    void _patch_input_chw_to_hwc();

    /// \brief Project `rows` raw PCM frames into `rows` language embeddings.
    /// \param frames [rows][audio_embed_dim] raw samples (not a mel spectrogram)
    /// \param out    [rows][hidden_size]
    void project_audio(const bf16* frames, size_t rows, bf16* out);

    const gemma4_12b_mm_dims& dims() const { return this->_dims; }
    bool has_vision() const { return this->_has_vision; }
    bool has_audio()  const { return this->_has_audio; }
    bool is_loaded()  const { return this->_loaded; }

private:
    gemma4_12b_mm_dims _dims;
    LM_Config          _config;

    //!< null on the host backend; the projector's xclbin on the array backend
    npu_app_manager* _mm_app_manager = nullptr;

    bool _loaded     = false;
    bool _has_vision = false;
    bool _has_audio  = false;

    // ---- weights, all bf16, all [out, in] row major as stored ----
    buffer<bf16> _patch_ln1_gamma;   //!< v.patch_norm.1.weight   [6912]
    buffer<bf16> _patch_ln1_beta;    //!< v.patch_norm.1.bias     [6912]

    /// \brief v.patch_embd.bias [3840] followed by v.patch_embd.weight [3840, 6912], in one
    ///        allocation. Use _patch_dense_bias() / _patch_dense_weight(), never data().
    ///
    /// The two tensors are fused because a run on the array takes exactly three buffers --
    /// y, x, w -- so the bias has nowhere else to ride, and gen_mm_projector_bias() reads it
    /// from the front of the weight buffer. Fusing on the host backend too is deliberate: a
    /// layout that differs between backends is a layout the cross reference cannot check.
    buffer<bf16> _patch_dense;
    buffer<bf16> _patch_ln2_gamma;   //!< v.patch_norm.2.weight   [3840]
    buffer<bf16> _patch_ln2_beta;    //!< v.patch_norm.2.bias     [3840]
    buffer<bf16> _pos_table;         //!< v.position_embd.weight  [2, 1120, 3840]
    buffer<bf16> _pos_ln_gamma;      //!< v.patch_norm.3.weight   [3840]
    buffer<bf16> _pos_ln_beta;       //!< v.patch_norm.3.bias     [3840]
    buffer<bf16> _vision_proj_w;     //!< mm.input_projection.weight    [3840, 3840]
    buffer<bf16> _audio_proj_w;      //!< mm.a.input_projection.weight  [3840, 640]

    /// \brief The bias region of _patch_dense: [hidden_size], indexed by output channel.
    /// \note  Not touched by _patch_input_chw_to_hwc() -- that permutation is on the K
    ///        (input) axis, and the bias has no K axis. Permuting it would shift every
    ///        weight row by one output channel and still produce finite, plausible rows.
    bf16* _patch_dense_bias() { return this->_patch_dense.data(); }

    /// \brief The weight region of _patch_dense: [mm_embed_dim][patch_dim] row major.
    bf16* _patch_dense_weight() {
        return this->_patch_dense.data() + (size_t)this->_dims.mm_embed_dim;
    }

    // ---- activation scratch, grown on demand ----
    //
    // Row counts are padded up to mm_projector_geometry::M_ALIGN, because that is the row
    // block a round of the overlay covers. Matmul rows are independent, so the pad rows
    // only ever produce pad output -- but they are zeroed at allocation anyway, to keep
    // whatever the allocator handed back from reaching a bf16 multiply.
    buffer<bf16> _v_wide;      //!< [M][patch_dim]      normed raw tiles
    buffer<bf16> _v_narrow;    //!< [M][mm_embed_dim]   everything after patch_embd
    buffer<bf16> _v_out;       //!< [M][hidden_size]    vision proj output, array backend only
    size_t       _v_rows = 0;  //!< padded rows the vision scratch covers
    buffer<bf16> _a_frames;    //!< [M][audio_embed_dim] normed PCM
    buffer<bf16> _a_out;       //!< [M][hidden_size]     audio proj output, array backend only
    size_t       _a_rows = 0;  //!< padded rows the audio scratch covers

    void _reserve_vision(size_t rows);
    void _reserve_audio(size_t rows);

    /// \brief Round a row count up to the overlay's row block.
    static size_t _padded_rows(size_t rows);

#if GEMMA4_12B_MM_PROJ_ON_NPU
    // One app per matmul, because an app owns one control sequence and the three matmuls
    // have three different (K, N, half, bias) shapes. Built once; only the sequences are
    // regenerated, and only when the padded row count moves.
    npu_app _patch_dense_app;    //!< bias half, add_bias = 1
    npu_app _vision_proj_app;    //!< bias half, add_bias = 0
    npu_app _audio_proj_app;     //!< plain half
    size_t  _patch_seq_rows  = 0;
    size_t  _vision_seq_rows = 0;
    size_t  _audio_seq_rows  = 0;

    /// \brief Run one projector matmul on the array, and leave `y` readable on the host.
    ///
    /// Regenerates `app`'s control sequence when `M` has moved since the last call, which
    /// is once per distinct padded row count rather than once per image.
    ///
    /// \param seq_rows  the guard for this app's sequence, updated in place
    /// \param bias_half true for the mm_bias columns (4..7), false for the plain ones
    /// \param add_bias  only meaningful on the bias half; `w` must then carry the bias in
    ///                  front of the weights
    void _dense_npu(npu_app& app, size_t& seq_rows, buffer<bf16>& y, buffer<bf16>& x,
                    buffer<bf16>& w, size_t M, int K, int N,
                    bool bias_half, bool add_bias);
#endif

    /// \brief Read one tensor into `dst`, checking its element count.
    /// \return false when the tensor is absent (the caller decides whether that is fatal)
    bool _load_tensor(SafeTensors& st, const char* name, buffer<bf16>& dst, size_t expect);

    /// \brief Read `bias` then `weight` into one allocation, bias first.
    /// \see   _patch_dense for why the two are fused.
    bool _load_dense_with_bias(SafeTensors& st, const char* bias_name, const char* w_name,
                               buffer<bf16>& dst, size_t n_bias, size_t n_weight);

    /// \brief Where load_weights() looks for the checkpoint.
    std::string _weights_path() const;
};

#endif // __GEMMA4_12B_MM_PROJECTOR_HPP__
