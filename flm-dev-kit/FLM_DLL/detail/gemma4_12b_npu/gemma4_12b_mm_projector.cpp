// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file gemma4_12b_mm_projector.cpp
/// \brief Implementation of the gemma4-12b multi modal projector
/// \author FastFlowLM Team
/// \see gemma4_12b_mm_projector.hpp for the shape of both towers and for why the
///      projector takes dimensions rather than a pointer back to the model.

#include "gemma4_12b_mm_projector.hpp"
#include "gemma4_12b_npu_sequence.hpp"  //!< gen_mm_projector_*, mm_projector_geometry

#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
//  Tensor names in image_audio_weights.q4nx
//
//  These are the *packaged* names, which are shorter than the upstream torch ones
//  (model.vision_embedder.* / model.embed_vision.* / model.embed_audio.*). The converter
//  renames them, so this is the list to trust.
// ---------------------------------------------------------------------------
namespace {
    constexpr const char* TN_PATCH_LN1_W  = "v.patch_norm.1.weight";
    constexpr const char* TN_PATCH_LN1_B  = "v.patch_norm.1.bias";
    constexpr const char* TN_PATCH_W      = "v.patch_embd.weight";
    constexpr const char* TN_PATCH_B      = "v.patch_embd.bias";
    constexpr const char* TN_PATCH_LN2_W  = "v.patch_norm.2.weight";
    constexpr const char* TN_PATCH_LN2_B  = "v.patch_norm.2.bias";
    constexpr const char* TN_POS_TABLE    = "v.position_embd.weight";
    constexpr const char* TN_POS_LN_W     = "v.patch_norm.3.weight";
    constexpr const char* TN_POS_LN_B     = "v.patch_norm.3.bias";
    constexpr const char* TN_VISION_PROJ  = "mm.input_projection.weight";
    constexpr const char* TN_AUDIO_PROJ   = "mm.a.input_projection.weight";

    constexpr const char* MM_WEIGHTS_FILE = "image_audio_weights.q4nx";
}

// ===========================================================================
//  gemma4_12b_mm_dims
// ===========================================================================
gemma4_12b_mm_dims gemma4_12b_mm_dims::from_config(const LM_Config& config) {
    gemma4_12b_mm_dims d;

    const nlohmann::json& vc = config.sub("vision_config");
    const nlohmann::json& ac = config.sub("audio_config");
    const nlohmann::json& pc = cfg_sub(config._json_config, "processor_config");
    const nlohmann::json& ip = cfg_sub(pc, "image_processor");
    const nlohmann::json& vp = cfg_sub(pc, "video_processor");
    const nlohmann::json& fe = cfg_sub(pc, "feature_extractor");

    d.hidden_size = (int)config.get<u32>("hidden_size", 3840);

    // The projector's own width is output_proj_dims when the checkpoint states it; it is
    // the language hidden size on every shipped gemma4-12b, but the two are separate
    // numbers in the config and a future checkpoint is free to disagree.
    d.mm_embed_dim   = (int)cfg_get<u32>(vc, "mm_embed_dim", (u32)d.hidden_size);
    d.mm_posemb_size = (int)cfg_get<u32>(vc, "mm_posemb_size", 1120u);
    d.patch_size     = (int)cfg_get<u32>(ip, "patch_size", cfg_get<u32>(vc, "patch_size", 16u));
    d.pooling_kernel = (int)cfg_get<u32>(ip, "pooling_kernel_size",
                                         cfg_get<u32>(vc, "pooling_kernel_size", 3u));
    // A "model patch" is what the preprocessor pools into one soft token: a
    // pooling_kernel x pooling_kernel square of patch_size x patch_size pixels, in RGB.
    d.model_patch    = d.patch_size * d.pooling_kernel;
    d.patch_dim      = d.model_patch * d.model_patch * 3;

    d.img_max_soft_tok = (int)cfg_get<u32>(ip, "max_soft_tokens",
                                           cfg_get<u32>(pc, "image_seq_length", 280u));
    d.vid_max_soft_tok = (int)cfg_get<u32>(vp, "max_soft_tokens", 70u);

    d.audio_embed_dim = (int)cfg_get<u32>(fe, "feature_size",
                                          cfg_get<u32>(ac, "audio_embed_dim", 640u));
    // audio_seq_length sits at the top of processor_config.json, next to image_seq_length,
    // rather than inside the feature extractor block.
    d.audio_max_soft_tok = (int)cfg_get<u32>(pc, "audio_seq_length",
                                             cfg_get<u32>(ac, "audio_seq_length", 750u));

    d.rms_norm_eps = cfg_get<f32>(vc, "rms_norm_eps", config.get<f32>("rms_norm_eps", 1e-6f));
    // Deliberately not read from the config: this is torch's nn.LayerNorm default, and the
    // reference implementation constructs those layers without passing eps.
    d.layer_norm_eps = 1e-5f;

    return d;
}

std::string gemma4_12b_mm_dims::_str() const {
    return "hidden " + std::to_string(this->hidden_size)
         + ", patch " + std::to_string(this->model_patch) + "x" + std::to_string(this->model_patch)
         + " (" + std::to_string(this->patch_dim) + " wide)"
         + ", mm_embed " + std::to_string(this->mm_embed_dim)
         + ", posemb " + std::to_string(this->mm_posemb_size)
         + ", img<=" + std::to_string(this->img_max_soft_tok)
         + ", audio " + std::to_string(this->audio_embed_dim)
         + "<=" + std::to_string(this->audio_max_soft_tok) + " frames";
}

// ===========================================================================
//  construction
// ===========================================================================
gemma4_12b_mm_projector::gemma4_12b_mm_projector(npu_xclbin_manager* npu, LM_Config& config)
    : _dims(gemma4_12b_mm_dims::from_config(config)), _config(config) {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    this->_mm_app_manager = npu->register_xclbin(
        utils::path_join(config.exec_path, "xclbins", config.model_name, "audio_image_mm.xclbin"));

    // Three apps for three matmul shapes. Creating them costs a control sequence object
    // each and nothing on the array; the sequences themselves are generated on the first
    // projection, once the row count is known.
    this->_patch_dense_app  = this->_mm_app_manager->create_app();
    this->_vision_proj_app  = this->_mm_app_manager->create_app();
    this->_audio_proj_app   = this->_mm_app_manager->create_app();

    // The plain half is not a K-parametric engine (see gen_mm_projector_plain), so a
    // checkpoint with a different audio frame width cannot run on it. Catch that here,
    // at bring up, rather than as a silently wrong first projection.
    if (this->_dims.audio_embed_dim != (int)mm_projector_geometry::k_plain) {
        throw std::runtime_error("mm projector: audio_embed_dim is "
                                 + std::to_string(this->_dims.audio_embed_dim)
                                 + ", but the overlay's plain half is built for "
                                 + std::to_string(mm_projector_geometry::k_plain));
    }
    if (this->_dims.patch_dim % (int)mm_projector_geometry::k_bias != 0
        || this->_dims.mm_embed_dim % (int)mm_projector_geometry::k_bias != 0) {
        throw std::runtime_error("mm projector: patch_dim / mm_embed_dim must be a whole"
                                 " number of "
                                 + std::to_string(mm_projector_geometry::k_bias)
                                 + " wide k tiles");
    }
#else
    //!< nothing runs on the array, so a text only session -- or a build with no device --
    //!< never touches an xclbin here
    (void)npu;
#endif

    DEBUG_BLOCK(1,
    header_print("info", "mm projector: " + this->_dims._str()
                 + ", matmuls on " + (GEMMA4_12B_MM_PROJ_ON_NPU ? "npu" : "cpu"));
    )
}

gemma4_12b_mm_projector::~gemma4_12b_mm_projector() = default;

// ===========================================================================
//  weight loading
// ===========================================================================
std::string gemma4_12b_mm_projector::_weights_path() const {
    // A checkpoint may name the file explicitly; the shipped ones just place it next to
    // config.json under its canonical name.
    const std::string named = this->_config.get<std::string>("mm_projector_weight", "");
    if (!named.empty()) return named;
    const std::string legacy = this->_config.get<std::string>("vision_model_weight", "");
    if (!legacy.empty()) return legacy;
    return utils::path_join(this->_config.model_path, MM_WEIGHTS_FILE);
}

bool gemma4_12b_mm_projector::_load_tensor(SafeTensors& st, const char* name,
                                           buffer<bf16>& dst, size_t expect) {
    if (!st.has_tensor(name)) return false;

    buffer<bf16> tmp;
    st.load_weights(tmp, name);
    if (tmp.size() != expect) {
        throw std::runtime_error(std::string("mm projector: ") + name + " has "
                                 + std::to_string(tmp.size()) + " elements, expected "
                                 + std::to_string(expect));
    }

    // Copy rather than keep `tmp`: on the array backend the destination has to be a device
    // BO, and going through the same two steps on both backends keeps this one code path.
    dst = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, expect);
    memcpy(dst.data(), tmp.data(), expect * sizeof(bf16));
    gemma4_12b_mm_proj_push(dst);
    return true;
}

bool gemma4_12b_mm_projector::_load_dense_with_bias(SafeTensors& st, const char* bias_name,
                                                    const char* w_name, buffer<bf16>& dst,
                                                    size_t n_bias, size_t n_weight) {
    if (!st.has_tensor(bias_name) || !st.has_tensor(w_name)) return false;

    buffer<bf16> bias_tmp;
    buffer<bf16> w_tmp;
    st.load_weights(bias_tmp, bias_name);
    st.load_weights(w_tmp, w_name);
    if (bias_tmp.size() != n_bias) {
        throw std::runtime_error(std::string("mm projector: ") + bias_name + " has "
                                 + std::to_string(bias_tmp.size()) + " elements, expected "
                                 + std::to_string(n_bias));
    }
    if (w_tmp.size() != n_weight) {
        throw std::runtime_error(std::string("mm projector: ") + w_name + " has "
                                 + std::to_string(w_tmp.size()) + " elements, expected "
                                 + std::to_string(n_weight));
    }

    //!< bias first, then the weight rows: the offset gen_mm_projector_bias() encodes
    dst = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, n_bias + n_weight);
    memcpy(dst.data(), bias_tmp.data(), n_bias * sizeof(bf16));
    memcpy(dst.data() + n_bias, w_tmp.data(), n_weight * sizeof(bf16));
    gemma4_12b_mm_proj_push(dst);
    return true;
}

// ---------------------------------------------------------------------------
//  patch input layout
// ---------------------------------------------------------------------------
void gemma4_12b_mm_projector::_patch_input_chw_to_hwc() {
    const int    S = this->_dims.model_patch;             // 48
    const size_t K = (size_t)this->_dims.patch_dim;      // 48 * 48 * 3
    const size_t D = (size_t)this->_dims.mm_embed_dim;
    if (K != (size_t)S * S * 3) return;

    //!< chw index of the element that hwc index i names, i.e. perm[i] tells the new
    //!< layout where to read the old one from
    std::vector<int> perm(K);
    for (int h = 0; h < S; ++h)
        for (int w = 0; w < S; ++w)
            for (int c = 0; c < 3; ++c)
                perm[((size_t)h * S + w) * 3 + c] = c * S * S + h * S + w;

    std::vector<bf16> row(K);
    auto permute_row = [&](bf16* dst){
        memcpy(row.data(), dst, K * sizeof(bf16));
        for (size_t i = 0; i < K; ++i) dst[i] = row[perm[i]];
    };

    permute_row(this->_patch_ln1_gamma.data());
    permute_row(this->_patch_ln1_beta.data());
    //!< the weight region only: the bias in front of it is indexed by output channel and
    //!< has no K axis to permute, see _patch_dense_bias()
    bf16* w = this->_patch_dense_weight();
    for (size_t n = 0; n < D; ++n) permute_row(w + n * K);

    gemma4_12b_mm_proj_push(this->_patch_ln1_gamma);
    gemma4_12b_mm_proj_push(this->_patch_ln1_beta);
    gemma4_12b_mm_proj_push(this->_patch_dense);
}

void gemma4_12b_mm_projector::load_weights() {
    const std::string path = this->_weights_path();
    DEBUG_BLOCK(1,
    header_print("info", "mm projector: loading " + path);
    )
    SafeTensors st(path);
    this->load_weights(st);
}

void gemma4_12b_mm_projector::load_weights(SafeTensors& st) {
    if (this->_loaded) return;

    const size_t D    = (size_t)this->_dims.mm_embed_dim;
    const size_t K    = (size_t)this->_dims.patch_dim;
    const size_t P    = (size_t)this->_dims.mm_posemb_size;
    const size_t A    = (size_t)this->_dims.audio_embed_dim;
    const size_t Dout = (size_t)this->_dims.hidden_size;

    // ---- vision ----
    // Every tensor of the tower has to be there or the tower is not there: a partially
    // present vision tower would silently project through an uninitialised buffer.
    bool v = true;
    v &= this->_load_tensor(st, TN_PATCH_LN1_W, this->_patch_ln1_gamma, K);
    v &= this->_load_tensor(st, TN_PATCH_LN1_B, this->_patch_ln1_beta,  K);
    //!< the one pair that does not get its own buffer, see _patch_dense
    v &= this->_load_dense_with_bias(st, TN_PATCH_B, TN_PATCH_W, this->_patch_dense, D, D * K);
    v &= this->_load_tensor(st, TN_PATCH_LN2_W, this->_patch_ln2_gamma, D);
    v &= this->_load_tensor(st, TN_PATCH_LN2_B, this->_patch_ln2_beta,  D);
    v &= this->_load_tensor(st, TN_POS_TABLE,   this->_pos_table,       2 * P * D);
    v &= this->_load_tensor(st, TN_POS_LN_W,    this->_pos_ln_gamma,    D);
    v &= this->_load_tensor(st, TN_POS_LN_B,    this->_pos_ln_beta,     D);
    v &= this->_load_tensor(st, TN_VISION_PROJ, this->_vision_proj_w,   Dout * D);
    if (v) this->_patch_input_chw_to_hwc();
    this->_has_vision = v;

    // ---- audio ----
    // The whole audio tower is this one matmul: there is no conformer, and the RMS norm in
    // front of it carries no scale, so it has no tensor either.
    this->_has_audio = this->_load_tensor(st, TN_AUDIO_PROJ, this->_audio_proj_w, Dout * A);

    this->_loaded = true;

    DEBUG_BLOCK(1,
    header_print("info", std::string("mm projector ready: vision ")
                 + (this->_has_vision ? "yes" : "no") + ", audio "
                 + (this->_has_audio ? "yes" : "no"));
    )
    if (!this->_has_vision && !this->_has_audio) {
        header_print_r("warning", "mm projector: checkpoint has neither tower, media will be ignored");
    }
}

// ===========================================================================
//  scratch
// ===========================================================================
size_t gemma4_12b_mm_projector::_padded_rows(size_t rows) {
#if GEMMA4_12B_MM_PROJ_ON_NPU
    // A round of the overlay covers cols * m rows at once and the descriptors are built
    // from whole rounds, so the buffers have to be that tall even for a 3 frame clip.
    constexpr size_t align = mm_projector_geometry::M_ALIGN;
    return ((rows + align - 1) / align) * align;
#else
    return rows;
#endif
}

void gemma4_12b_mm_projector::_reserve_vision(size_t rows) {
    const size_t M = _padded_rows(rows);
    if (M <= this->_v_rows) return;

    const size_t K    = (size_t)this->_dims.patch_dim;
    const size_t D    = (size_t)this->_dims.mm_embed_dim;
    const size_t Dout = (size_t)this->_dims.hidden_size;

    this->_v_wide   = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, M * K);
    this->_v_narrow = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, M * D);
    this->_v_wide.memset((bf16)0.0f);
    this->_v_narrow.memset((bf16)0.0f);
    gemma4_12b_mm_proj_push(this->_v_wide);
    gemma4_12b_mm_proj_push(this->_v_narrow);
#if GEMMA4_12B_MM_PROJ_ON_NPU
    //!< the array cannot write into the caller's std::vector, so the last projection lands
    //!< here and is copied out; on the host backend it writes `out` directly
    this->_v_out = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, M * Dout);
    this->_v_out.memset((bf16)0.0f);
    gemma4_12b_mm_proj_push(this->_v_out);
#else
    (void)Dout;
#endif
    this->_v_rows = M;
}

void gemma4_12b_mm_projector::_reserve_audio(size_t rows) {
    const size_t M = _padded_rows(rows);
    if (M <= this->_a_rows) return;

    const size_t A    = (size_t)this->_dims.audio_embed_dim;
    const size_t Dout = (size_t)this->_dims.hidden_size;

    this->_a_frames = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, M * A);
    this->_a_frames.memset((bf16)0.0f);
    gemma4_12b_mm_proj_push(this->_a_frames);
#if GEMMA4_12B_MM_PROJ_ON_NPU
    this->_a_out = gemma4_12b_mm_proj_alloc(this->_mm_app_manager, M * Dout);
    this->_a_out.memset((bf16)0.0f);
    gemma4_12b_mm_proj_push(this->_a_out);
#else
    (void)Dout;
#endif
    this->_a_rows = M;
}

// ===========================================================================
//  the array backend of gemma4_12b_mm_cpu::dense()
// ===========================================================================
#if GEMMA4_12B_MM_PROJ_ON_NPU
void gemma4_12b_mm_projector::_dense_npu(npu_app& app, size_t& seq_rows,
                                         buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w,
                                         size_t M, int K, int N,
                                         bool bias_half, bool add_bias) {
    // M, K and N are all baked into the descriptors, and only M ever moves: K and N are
    // fixed by the checkpoint. So this regenerates once per distinct padded row count --
    // typically once for every image in a session, since they all pad to the same M.
    if (M != seq_rows) {
        if (bias_half) {
            gen_mm_projector_bias(app.seq(), (uint32_t)M, (uint32_t)K, (uint32_t)N, add_bias);
        } else {
            gen_mm_projector_plain(app.seq(), (uint32_t)M, (uint32_t)K, (uint32_t)N);
        }
        seq_rows = M;
    }

    gemma4_12b_mm_proj_run h;
    h.run = app.create_run(y, x, w);
    h.dst = &y;
    h.run.start();
    h.wait();  //!< run.wait() + y.sync_from_device()
}
#endif

// ===========================================================================
//  forward
// ===========================================================================
void gemma4_12b_mm_projector::project_vision(const bf16* patches, const int* pos_ids,
                                             size_t rows, bf16* out) {
    if (rows == 0) return;
    if (!this->_has_vision) {
        throw std::runtime_error("mm projector: project_vision() without a vision tower");
    }

    const int K    = this->_dims.patch_dim;
    const int D    = this->_dims.mm_embed_dim;
    const int Dout = this->_dims.hidden_size;

    this->_reserve_vision(rows);
    bf16* wide   = this->_v_wide.data();
    bf16* narrow = this->_v_narrow.data();

    // 1. LayerNorm over the raw tile, still 6912 wide. Note the torch eps, 1e-5.
    gemma4_12b_mm_cpu::layer_norm(wide, patches,
                                  this->_patch_ln1_gamma.data(), this->_patch_ln1_beta.data(),
                                  rows, K, this->_dims.layer_norm_eps);
    gemma4_12b_mm_proj_push(this->_v_wide);

    // 2. patch_embd: 6912 -> 3840, with a bias. This single matmul is ~64% of the cost of
    //    an image (6.8 GMAC of 10.6 for a full 280 tile one).
#if GEMMA4_12B_MM_PROJ_ON_NPU
    this->_dense_npu(this->_patch_dense_app, this->_patch_seq_rows,
                     this->_v_narrow, this->_v_wide, this->_patch_dense,
                     this->_v_rows, K, D, /*bias_half=*/true, /*add_bias=*/true);
#else
    gemma4_12b_mm_cpu::dense(narrow, wide, this->_patch_dense_weight(),
                             this->_patch_dense_bias(), rows, K, D);
#endif

    // 3. LayerNorm again, now 3840 wide.
    gemma4_12b_mm_cpu::layer_norm(narrow, narrow,
                                  this->_patch_ln2_gamma.data(), this->_patch_ln2_beta.data(),
                                  rows, D, this->_dims.layer_norm_eps);

    // 4. Position, added from two 1-d tables rather than looked up in a 2-d one.
    gemma4_12b_mm_cpu::add_factorized_pos_embedding(narrow, pos_ids, this->_pos_table.data(),
                                                    rows, D, this->_dims.mm_posemb_size);

    // 5. LayerNorm over the positioned row.
    gemma4_12b_mm_cpu::layer_norm(narrow, narrow,
                                  this->_pos_ln_gamma.data(), this->_pos_ln_beta.data(),
                                  rows, D, this->_dims.layer_norm_eps);

    // 6. RMS norm with no scale -- a different normalisation from the three above: no
    //    gamma, no beta, no mean subtraction, and the model's 1e-6 eps.
    gemma4_12b_mm_cpu::rms_norm_no_scale(narrow, narrow, rows, D, this->_dims.rms_norm_eps);
    gemma4_12b_mm_proj_push(this->_v_narrow);

    // 7. Into the language space. No bias -- but still the bias half of the overlay, which
    //    is the only one whose K is programmable, and 3840 is 5 tiles of 768.
#if GEMMA4_12B_MM_PROJ_ON_NPU
    this->_dense_npu(this->_vision_proj_app, this->_vision_seq_rows,
                     this->_v_out, this->_v_narrow, this->_vision_proj_w,
                     this->_v_rows, D, Dout, /*bias_half=*/true, /*add_bias=*/false);
    std::memcpy(out, this->_v_out.data(), rows * (size_t)Dout * sizeof(bf16));
#else
    gemma4_12b_mm_cpu::dense(out, narrow, this->_vision_proj_w.data(), nullptr, rows, D, Dout);
#endif
}

void gemma4_12b_mm_projector::project_audio(const bf16* frames, size_t rows, bf16* out) {
    if (rows == 0) return;
    if (!this->_has_audio) {
        throw std::runtime_error("mm projector: project_audio() without an audio tower");
    }

    const int A    = this->_dims.audio_embed_dim;
    const int Dout = this->_dims.hidden_size;

    this->_reserve_audio(rows);
    bf16* normed = this->_a_frames.data();

    // The entire audio tower: normalise the raw 40 ms window, then one matmul. The input
    // is PCM, not a mel spectrogram, however the payload field happens to be named.
    gemma4_12b_mm_cpu::rms_norm_no_scale(normed, frames, rows, A, this->_dims.rms_norm_eps);
    gemma4_12b_mm_proj_push(this->_a_frames);

    // The plain half of the overlay, whose K tile is compiled in at 640 -- exactly one
    // audio frame, which is why the audio projection needs no K loop at all.
#if GEMMA4_12B_MM_PROJ_ON_NPU
    this->_dense_npu(this->_audio_proj_app, this->_audio_seq_rows,
                     this->_a_out, this->_a_frames, this->_audio_proj_w,
                     this->_a_rows, A, Dout, /*bias_half=*/false, /*add_bias=*/false);
    std::memcpy(out, this->_a_out.data(), rows * (size_t)Dout * sizeof(bf16));
#else
    gemma4_12b_mm_cpu::dense(out, normed, this->_audio_proj_w.data(), nullptr, rows, A, Dout);
#endif
}
