/// \file d3_npu.hpp
/// \brief d3_npu: the Decision 3.0 (vllm-sr/d3-*) encoder on the NPU
/// \note  The Qwen3.5 text tower with bidirectional full attention, run over packed batches of
///        independent prompts; returns final-norm hidden states for the 255-way code readout
///        that the application applies (FastFlowLM AutoDecisionModel).
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "decision_model.hpp"

/// One image of a request: its patch grid after the processor's resize (16-pixel patches; the
/// image takes grid_h * grid_w / 4 input tokens after the 2 x 2 merge).
struct d3_image {
    int grid_h = 0;
    int grid_w = 0;
};

/// encode_batch()'s `payload`: the request's images (shared by every prompt) and, for the
/// `prefix_bidir` attention mode, each prompt's state length. Text-only calls may pass a payload
/// with no images (the vision tower is then not touched).
/// Each prompt holds, in image order, one run of grid_h * grid_w / 4 `image_token_id` tokens per
/// image (the expanded <|image_pad|> placeholders).
///
/// ABI: fields are append-only. The payload is read by the engine through a `void*`, so FastFlowLM
/// and libd3_npu.so must be built from the same header revision (a caller built before `state_len`
/// existed would hand over a shorter object).
struct d3_image_payload {
    std::vector<d3_image> images;
    /// every image's patches in order, as the Qwen2-VL processor flattens them: per patch
    /// [3 channels][2 frames][16][16] bf16, rescaled and normalized; patches in 2 x 2 merge order
    std::vector<bf16> pixels;
    int image_token_id = 248056;

    /// `prefix_bidir` attention (decision_config.json `attention_mode`): per prompt (same order and
    /// count as `ids`), the number S of leading tokens that form the state block. In the
    /// full-attention layers the first S tokens attend bidirectionally within themselves only; the
    /// remaining tokens attend to all S + their own tokens (bidirectionally). The gated delta-net
    /// layers stay causal over the whole prompt. S <= 0 or S >= the prompt's length means no split
    /// for that prompt (every token sees every token, the `noncausal_full_attention` numerics,
    /// bit-identical to a call without state_len). Empty (the default): no prompt is split, which is
    /// what lite / nano / mini use. A non-empty vector must hold one entry per prompt.
    std::vector<int> state_len;
};

/// Revision of the engine's public surface, bumped on every append to this header.
///   1: d3_image_payload (+ state_len, prefix_bidir); no function below existed
///   2: state-once: d3_question_block, d3_state_request, d3_npu::encode_state_once, the two functions below
/// FastFlowLM compares D3_NPU_ABI_VERSION (its mirror of this header) with d3_npu_abi_version() (the library
/// it loaded) and refuses a mismatch; a revision-1 library does not export the functions at all (resolve
/// them weakly or with dlsym, a missing symbol means revision 1: per-row prefix_bidir only).
///
/// Minor revision inside ABI 2 (append-only behaviour, no layout change: every struct of revision 2 is unchanged):
///   0: revision 2 as frozen (question_images / image_ids / video_token_id rejected)
///   1: M3 / M4: images inside question blocks (D3_CAP_QUESTION_IMAGES), video pad runs (D3_CAP_VIDEO), the vision
///      tower called in chunks of at most 65,536 patches; d3_npu_abi_minor() exists. A library without the symbol is minor 0.
constexpr int D3_NPU_ABI_VERSION = 2;
constexpr int D3_NPU_ABI_MINOR = 1;
/// Capabilities of the loaded library (bit set: usable)
enum d3_npu_capability : unsigned {
    D3_CAP_STATE_ONCE     = 1u << 0,   ///< encode_state_once()
    D3_CAP_READOUT_ROWS   = 1u << 1,   ///< d3_question_block::readout_from_end with several rows (M3)
    D3_CAP_QUESTION_IMAGES = 1u << 2,  ///< d3_state_request::question_images / image_ids (M3, minor 1)
    D3_CAP_VIDEO          = 1u << 3,   ///< d3_state_request::video_token_id (M4, minor 1)
};
extern "C" int d3_npu_abi_version();       ///< D3_NPU_ABI_VERSION of the library
extern "C" unsigned d3_npu_capabilities(); ///< d3_npu_capability bits
extern "C" int d3_npu_abi_minor();         ///< D3_NPU_ABI_MINOR of the library (absent in minor 0 libraries)

/// One question of a shared state (encode_state_once).
struct d3_question_block {
    /// the tokens after the state block, image / video pad runs expanded (the question rows: the state's
    /// last token is NOT repeated). Never empty.
    std::vector<int> ids;
    /// Rows to return, as offsets from the block's last token (0 = the last token, 3 = three before it),
    /// each row the final-norm hidden state, returned in the listed order. Empty = {0}: the last token only
    /// (what a choice / noul / score question reads). M3: box edges read several rows of one block (4
    /// colon tokens at offsets 9, 6, 3, 0; track: 12, 9, 6, 3, 0). Every offset must be in [0, ids.size()).
    std::vector<int> readout_from_end;
    /// M3 (D3_CAP_QUESTION_IMAGES): indices into d3_state_request::question_images of the images whose pad runs
    /// appear in `ids`, in pad-run order (the same table entry may be listed by many blocks, and twice in one
    /// block). The pad ids are question_images.image_token_id and, when video_token_id >= 0, that id. The runs'
    /// M-RoPE positions start at the block's first position (the state's next position); every run advances it by
    /// max(grid_h, grid_w) / 2, as in the state. A pad token in a block without image_ids is an error.
    std::vector<int> image_ids;
};

/// A state block and the questions that share it: one state pass, every question continues from its cache.
/// Per question the result equals running the prompt `state_ids + ids` through encode_batch() with
/// payload->state_len = {state_ids.size()} (the prefix_bidir per-row path): the question rows attend to the
/// whole state and to the block (bidirectionally), the state rows to the state only.
struct d3_state_request {
    /// the state block, images / videos expanded: ends with the token before the question block (upstream
    /// requires a text token last; an image pad run last works but is not what the contract produces)
    std::vector<int> state_ids;
    /// images of the state (d3_image_payload semantics; their pad runs lie inside state_ids; state_len unused).
    /// nullptr = none. Not copied: must outlive the call.
    const d3_image_payload* media = nullptr;
    std::vector<d3_question_block> questions;

    // ---- M3 / M4 (library minor 1; a minor-0 library throws std::invalid_argument for a non-default value)
    /// M3: images that occur inside question blocks (grid, pixels, image_token_id as a payload), referenced by
    /// d3_question_block::image_ids. The caller may deduplicate; the engine also merges entries with equal grid and
    /// pixels (one vision tower run, the rows are shared by every use). Large tables run as several tower calls.
    d3_image_payload question_images;
    /// M4: id of <|video_pad|>; its runs are visual runs just like image_token_id runs, in the state (a video's
    /// temporal groups are entries of `media->images`, one {grid_h, grid_w} per group, in prompt order, pixels
    /// [3][2][16][16] per patch) and in question blocks. -1: none.
    int video_token_id = -1;
};

class d3_npu : public decision_model{
public:
    /// \param config the package's config.json
    /// \param npu_instance the xclbin manager (mm.xclbin is registered from it)
    /// \param max_length longest prompt accepted, in tokens (default: 32,768, FastFlowLM's limit)
    d3_npu(LM_Config config, npu_xclbin_manager* npu_instance, int max_length = 32768);
    ~d3_npu();
    d3_npu(const d3_npu&) = delete;
    d3_npu& operator=(const d3_npu&) = delete;

    void load_weights(Q4NX& q4nx) override;
    /// \param payload nullptr (text) or a d3_image_payload*: the vision tower runs once per call and
    ///        its output fills every prompt's image tokens; positions are Qwen3.5's 3D M-RoPE ones.
    ///        The vision bitstreams are registered on the first call with images.
    std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, void* payload = nullptr) override;
    buffer<bf16> encode(const std::vector<int>& ids, void* payload = nullptr) override;
    int max_length() override;
    int hidden_size() override;

    /// \brief rows packed into one pass (default 2048, four array GEMM row tiles); 0 = one prompt per pass
    void set_pack_rows(int rows);
    /// \brief rig only: run the full-attention layers causally (the stock-engine comparison baseline)
    void set_causal(bool causal);

    /// \brief state-once (NON-virtual; revision 2, see d3_npu_abi_version()): encode the state block once and
    ///        every question of `req` as a block continuing from it.
    ///
    /// Returns one buffer per question, in order: max(1, readout_from_end.size()) rows of final-norm bf16
    /// [hidden_size()], row-major. Numerics: with host attention on both sides (D3_ATTN_ARRAY_MIN=0 or a
    /// state under the array threshold) bit-identical to encode_batch(state + question, state_len = S) for
    /// every question, whatever other questions share the call, their order or the pass packing; with array
    /// attention the state rows may differ from the host path by the array's known noise (hidden cosine
    /// >= 0.998), the question rows are always computed on the host.
    /// RoPE positions of a question block continue at the state's next position on all three axes (text:
    /// state_ids.size(); images in the state: the M-RoPE planner's position after the last state token).
    /// Pass 0 holds the state and as many questions as fit into the padding rows of its last 512-row tile
    /// (and up to pack_rows); later passes hold questions only. The per-layer caches (attention K/V of the
    /// state, GDN conv tail and recurrent state) live on the host heap for the duration of the call only
    /// (d3-turbo: 64 KiB per state token plus 147 MiB, i.e. 268 MB at 4k tokens; in general
    /// 2 * attention layers * kv heads * head_dim * 2 bytes per token plus 3 MiB per GDN layer of the 48 value heads).
    /// M3 / M4 (library minor 1, d3_npu_capabilities()): question blocks may hold images (question_images / image_ids:
    /// one vision tower run per distinct image, positions started at the block's first position), and <|video_pad|>
    /// runs are visual runs (video_token_id; a video's temporal groups are entries of media->images). A video clip's
    /// windows are one call each (the state of a call is the window). The vision tower is called in chunks of at most
    /// 65,536 patches.
    /// \throws std::invalid_argument bad request (empty state or block, readout offset out of range, an
    ///         image run mismatch, a field this library does not support), std::length_error state + block
    ///         over max_length(), std::runtime_error load_weights() missing / vision tower
    /// Not thread-safe against other calls on the same engine (as encode_batch).
    std::vector<buffer<bf16>> encode_state_once(const d3_state_request& req);

private:
    struct Impl;
    Impl* _impl;
};
