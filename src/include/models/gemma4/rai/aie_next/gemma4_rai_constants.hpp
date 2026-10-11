#pragma once

#include <cstddef>
#include <cstdint>

namespace flm::gemma4 {

/// The PDI pair Gemma 4's ELFs were built for. Source: corelib python/
/// conftest.py, GEMMA4_PDI. Required by ryzenai_corelib_create_stream since
/// 0.5.0 and given no default by corelib: the silicon ships the same
/// operator under several PDI tags and a model's ELFs live under exactly
/// one. A stream opened on the wrong pair silently loses every shape only
/// that pair ships, surfacing as a missing artifact rather than a wrong
/// answer -- see phi4_rai_constants.hpp's kPrefillPdi/kTokenPdi for the same
/// reasoning applied to Phi-4's own (different) pair.
inline constexpr int kPrefillPdi = 8;
inline constexpr int kTokenPdi = 17;
/// Decode PDI for a context longer than one attention window. Prefill stays
/// on kPrefillPdi; the paged kernels ship on this overlay and nowhere else.
inline constexpr int kPagedTokenPdi = 33;

/// Shared by both shipped rows (E2B and E4B) -- read from the GGUF's own
/// `tokenizer.ggml.tokens` array length in a real load, but the two models
/// this port supports both ship this exact vocabulary.
inline constexpr std::int64_t kVocabularySize = 262144;

/// The group size DynamicDispatch's ELF set at PDI pair (8, 17) ships at.
/// A fact about which kernels exist, not about either model -- see
/// Gemma4Row's doc comment in gemma4_rai_gguf.hpp for how this and the two
/// group fields below it were established.
inline constexpr std::uint32_t kGroupSize = 32;

/// Longest context flat_mha ships, and the page a longer context is cut into.
inline constexpr std::int64_t kPageSize = 4096;
inline constexpr std::int64_t kMaxSequenceLength = 4096;
inline constexpr std::int64_t kMaxDecodeWindow = 4095;
/// Longest paged context: 16384, the validated ceiling. corelib upstream
/// validated paged Gemma 4 to 16k (aie4-models.md, "Past 4096 tokens"), well
/// short of the model's own 131072. A GGUF stating a shorter
/// `gemma4.context_length` caps it further. Whole pages.
inline constexpr std::int64_t kMaxPagedContext = 16384;
/// Retries for a non-finite paged prefill chunk past page 0. The chunk
/// rewrites only its own page, so a repeat is the same work.
inline constexpr int kPagedChunkAttempts = 3;

/// \return true when `length` is 1..4096, or a multiple of kPageSize up to
///         kMaxPagedContext
inline bool ContextFits(std::uint32_t length) {
    if (length < 1 || length > static_cast<std::uint32_t>(kMaxPagedContext)) return false;
    if (length <= static_cast<std::uint32_t>(kMaxSequenceLength)) return true;
    return length % static_cast<std::uint32_t>(kPageSize) == 0;
}

/// Applied to the final logits on every Gemma 4 size: tanh(x / cap) * cap.
inline constexpr float kLogitSoftcap = 30.0f;

/// rms epsilon, read from `gemma4.attention.layer_norm_rms_epsilon` on both
/// E2B and E4B (9.999999974752427e-07, i.e. 1e-6). NOT Phi-4's 1e-5 --
/// see phi4_rai_constants.hpp's kRmsEpsilon. Kept here only as the value a
/// fixture or a sanity check may compare against; the real loader reads the
/// metadata key rather than trusting this constant.
inline constexpr float kRmsEpsilon = 1.0e-6f;

/// See phi4_rai_constants.hpp: 0 lets corelib choose the per-create packer
/// threads (8 at 0.25.0), on top of the concurrent creates below.
inline constexpr std::uint32_t kRequantizeThreads = 0;
inline constexpr std::size_t kWeightCreateConcurrency = 8;

/// \brief bumped whenever the packed bytes change for an unchanged GGUF,
///        corelib version and descriptor; part of the weight-cache layout
/// \note 1: ple's gate and proj are packed from the transpose of the GGUF
///       mapping.
inline constexpr std::uint32_t kPackedLayoutRevision = 1;

}  // namespace flm::gemma4
