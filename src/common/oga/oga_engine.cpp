// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_engine.cpp
/// \brief Exact-token execution of an OGA model, with the images of its prompts
#include "oga/oga_engine.hpp"

#include "utils/debug_utils.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define FLM_OGA_HAS_SSE2 1
#else
#define FLM_OGA_HAS_SSE2 0
#endif

namespace flm::oga {
namespace {
static_assert(sizeof(int) == sizeof(std::int32_t), "token ids are passed to OGA as int32_t");
static_assert(sizeof(bf16) == sizeof(std::uint16_t), "bf16 must be 16 bits");
static_assert(std::is_trivially_copyable_v<bf16>, "bf16 must be trivially copyable");

/// \brief minimum unsplit first-run length that proves a model does not hit the Laguna 2.1
///        all-zero-logits bug (which triggers on runs of 8 or more tokens on hip-ep 0.5.3)
constexpr std::size_t kFirstRunVerifyMinTokens = 8;

#if FLM_OGA_HAS_SSE2
/// \brief convert 4 IEEE-754 float32 bit patterns in `v` to 4 bfloat16 bit patterns in
///        the low 16 bits of each 32-bit lane, matching biovault::bfloat16_t(float):
///        subnormals and +-0 flush to signed zero; normals and +-Inf round to nearest
///        even (+-Inf + 0x7fff >> 16 preserves +-Inf); NaNs truncate with quiet bit 0x0040.
inline __m128i Convert4Fp32ToBf16Lanes(__m128i v) noexcept {
    const __m128i abs_mask = _mm_set1_epi32(0x7fffffff);
    const __m128i min_norm = _mm_set1_epi32(0x00800000);
    const __m128i inf_bits = _mm_set1_epi32(0x7f800000);
    const __m128i one = _mm_set1_epi32(1);
    const __m128i bias_base = _mm_set1_epi32(0x00007fff);
    const __m128i qnan_bit = _mm_set1_epi32(0x00000040);
    const __m128i sign16 = _mm_set1_epi32(0x00008000);

    const __m128i abs_v = _mm_and_si128(v, abs_mask);
    const __m128i is_sub_or_zero = _mm_cmplt_epi32(abs_v, min_norm);
    const __m128i is_nan = _mm_cmpgt_epi32(abs_v, inf_bits);

    const __m128i hi16 = _mm_srli_epi32(v, 16);
    const __m128i lsb = _mm_and_si128(hi16, one);
    const __m128i rne = _mm_srli_epi32(_mm_add_epi32(v, _mm_add_epi32(bias_base, lsb)), 16);
    const __m128i nan_out = _mm_or_si128(hi16, qnan_bit);
    const __m128i zero_out = _mm_and_si128(hi16, sign16);

    __m128i res = _mm_or_si128(_mm_and_si128(is_nan, nan_out), _mm_andnot_si128(is_nan, rne));
    res = _mm_or_si128(_mm_and_si128(is_sub_or_zero, zero_out), _mm_andnot_si128(is_sub_or_zero, res));
    return res;
}
#endif

void Require(const void* function, const char* symbol) {
    if (!function)
        throw std::runtime_error(std::string("the hip-ep onnxruntime-genai is missing ") + symbol);
}

std::vector<int> ReadInputIds(const OgaApi& api, OgaTensor* tensor) {
    const OgaFunctions& f = api.functions();
    OgaElementType type = OgaElementType_undefined;
    api.Check(f.tensor_get_type(tensor, &type), "OgaTensorGetType(input_ids)");
    std::size_t rank = 0;
    api.Check(f.tensor_get_shape_rank(tensor, &rank), "OgaTensorGetShapeRank(input_ids)");
    std::vector<std::int64_t> shape(rank);
    if (rank) api.Check(f.tensor_get_shape(tensor, shape.data(), rank), "OgaTensorGetShape(input_ids)");
    if (rank == 2 && shape[0] == 1) shape.erase(shape.begin());
    if (shape.size() != 1 || shape[0] <= 0)
        throw std::runtime_error("processor input_ids are not a single token row");
    const auto count = static_cast<std::size_t>(shape[0]);
    void* data = nullptr;
    api.Check(f.tensor_get_data(tensor, &data), "OgaTensorGetData(input_ids)");
    if (!data) throw std::runtime_error("processor input_ids have no data");
    std::vector<int> ids(count);
    if (type == OgaElementType_int32) {
        const auto* from = static_cast<const std::int32_t*>(data);
        std::copy(from, from + count, ids.begin());
    } else if (type == OgaElementType_int64) {
        const auto* from = static_cast<const std::int64_t*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            if (from[i] < 0 || from[i] > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("processor input id does not fit in int");
            ids[i] = static_cast<int>(from[i]);
        }
    } else {
        throw std::runtime_error("processor input_ids are not int32 or int64");
    }
    return ids;
}

float HalfToFloat(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x3ffu;
    std::uint32_t bits = sign;
    if (exponent == 0x1f) {
        bits |= 0x7f800000u | (mantissa << 13);
    } else if (exponent != 0) {
        bits |= ((exponent + 112) << 23) | (mantissa << 13);
    } else if (mantissa != 0) {
        exponent = 113;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        bits |= (exponent << 23) | ((mantissa & 0x3ffu) << 13);
    }
    return std::bit_cast<float>(bits);
}

std::string ShapeText(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

std::uint64_t BytesHash(const std::string& bytes) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const unsigned char c : bytes) h = (h ^ c) * 0x100000001b3ull;
    return h ^ bytes.size();
}

std::uint64_t ImageHash(const OgaImage& image) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&h](std::uint64_t word) {
        h = (h ^ word) * 0x100000001b3ull;
        h ^= h >> 29;
    };
    mix(static_cast<std::uint64_t>(image.grid_t));
    mix(static_cast<std::uint64_t>(image.grid_h));
    mix(static_cast<std::uint64_t>(image.grid_w));
    mix(image.pixels.size());
    const std::uint16_t* values = image.pixels.data();
    std::size_t i = 0;
    for (; i + 4 <= image.pixels.size(); i += 4) {
        std::uint64_t word;
        std::memcpy(&word, values + i, sizeof word);
        mix(word);
    }
    for (; i < image.pixels.size(); ++i) mix(values[i]);
    return h;
}

}  // namespace

std::size_t SlotsWithin(const OgaSlotLimits& limits, std::uint32_t max_length) {
    const std::size_t wanted = std::max<std::size_t>(limits.count, 1);
    if (!limits.bytes_per_position || !max_length) return wanted;
    const std::uint64_t extra = limits.budget_bytes / (limits.bytes_per_position * max_length);
    return static_cast<std::size_t>(std::min<std::uint64_t>(wanted, 1 + extra));
}

void ConvertFp32ToBf16(const float* from, bf16* to, std::size_t count) noexcept {
    std::size_t i = 0;
#if FLM_OGA_HAS_SSE2
    const __m128i pack_bias32 = _mm_set1_epi32(0x00008000);
    const __m128i pack_bias16 = _mm_set1_epi16(static_cast<short>(-32768));
    for (; i + 8 <= count; i += 8) {
        std::uint32_t raw[8];
        std::memcpy(raw, from + i, sizeof(raw));
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + 4));
        const __m128i r0 = _mm_sub_epi32(Convert4Fp32ToBf16Lanes(v0), pack_bias32);
        const __m128i r1 = _mm_sub_epi32(Convert4Fp32ToBf16Lanes(v1), pack_bias32);
        const __m128i packed = _mm_xor_si128(_mm_packs_epi32(r0, r1), pack_bias16);
        std::uint16_t out[8];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out), packed);
        std::memcpy(to + i, out, sizeof(out));
    }
#endif
    for (; i < count; ++i) to[i] = bf16(from[i]);
}

OgaEngine::OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
                     std::string display_name, std::uint32_t vocab_size,
                     std::uint32_t context_limit, std::uint32_t max_length,
                     std::optional<OgaVisionConfig> vision, bool empty_image_batch,
                     OgaSlotLimits slots)
    : api_(std::move(api)),
      display_name_(std::move(display_name)),
      vocab_size_(vocab_size),
      context_limit_(context_limit),
      max_length_(max_length),
      vision_(std::move(vision)),
      empty_image_batch_(empty_image_batch),
      limits_(slots),
      model_(std::move(model)) {
    if (!api_ || !model_) throw std::invalid_argument("OGA engine needs an API and a model");
    if (!vocab_size_) throw std::invalid_argument(display_name_ + " vocabulary size is 0");
    if (!max_length_ || max_length_ > context_limit_)
        throw std::invalid_argument(display_name_ + " maximum length must be in 1.." +
                                    std::to_string(context_limit_));
    if (vision_) {
        const OgaVisionConfig& v = *vision_;
        const auto in_vocabulary = [&](int id) {
            return id >= 0 && static_cast<std::uint32_t>(id) < vocab_size_;
        };
        if (!in_vocabulary(v.image_token_id) || !in_vocabulary(v.vision_start_token_id) ||
            v.image_token_id == v.vision_start_token_id || v.spatial_merge_size <= 0 ||
            v.patch_values <= 0)
            throw std::invalid_argument(display_name_ + " vision configuration is invalid");
    }
    const OgaFunctions& f = api_->functions();
    rewinds_ = AtLeast(api_->package().oga_version, kRewindOgaVersion) &&
               f.generator_snapshot_state && f.generator_rewind_to;
}

OgaEngine::~OgaEngine() = default;

buffer<bf16> OgaEngine::forward(int id) {
    const bool skip_logits = std::exchange(skip_next_logits_, false);
    return Run({id}, nullptr, skip_logits, /*cache_logits=*/false);
}

buffer<bf16> OgaEngine::prefill(std::vector<int>& ids, void* payload) {
    const bool skip_logits = std::exchange(skip_next_logits_, false);
    if (staged_prompt_) {
        // Taken first, so a refused prefill does not leave the stage to the next one.
        std::unique_ptr<StagedPrompt> staged = std::move(staged_prompt_);
        if (payload)
            throw std::invalid_argument(display_name_ + " staged image prompt takes no vision payload");
        if (ids != staged->ids)
            throw std::runtime_error(display_name_ +
                                     " image prompt must be prefilled as the processor's whole sequence");
        return PrefillStaged(*staged, skip_logits);
    }
    if (payload && !vision_)
        throw std::invalid_argument(display_name_ + " on OGA does not run a vision payload");
    if (ids.empty()) throw std::invalid_argument(display_name_ + " request contains no token IDs");
    return Run(ids, static_cast<const OgaVisionInput*>(payload), skip_logits, /*cache_logits=*/true);
}

void OgaEngine::set_context_length(int length) {
    skip_next_logits_ = false;
    if (length < 0 || static_cast<std::uint32_t>(length) > max_length_)
        throw std::out_of_range(display_name_ + " context length is out of range");
    const auto kept = static_cast<std::size_t>(length);
    if (kept > logical_.size())
        throw std::runtime_error(display_name_ + " cannot rewind past the tokens it has run");
    while (!images_.empty() && images_.back().start + images_.back().length > kept) {
        if (images_.back().start < kept)
            throw std::invalid_argument(display_name_ + " cannot keep part of an image");
        images_.pop_back();
    }
    logical_.resize(kept);
    expected_.clear();
}

void OgaEngine::load_weights(Q4NX&) {
    throw std::runtime_error(display_name_ + " on OGA loads its weights from the ONNX package");
}

void OgaEngine::update_max_length(std::uint32_t max_length) {
    skip_next_logits_ = false;
    if (!max_length || max_length > context_limit_ || max_length < logical_.size())
        throw std::out_of_range(display_name_ + " maximum length is invalid");
    if (saved_ && max_length < saved_->tokens.size()) saved_.reset();
    max_length_ = max_length;
    // A larger one fits, but next to others it would overrun the budget they were counted for.
    const bool several = slot_limit() > 1;
    for (std::size_t i = slots_.size(); i-- > 0;)
        if (max_length > slots_[i].capacity || (several && max_length < slots_[i].capacity)) Drop(i);
    expected_.clear();
    Trim();
}

void OgaEngine::clear_context() {
    skip_next_logits_ = false;
    logical_.clear();
    images_.clear();
    expected_.clear();
    saved_.reset();
    // staged_prompt_ stays. The caller clears a divergent history and then
    // prefills the image sequence that was just staged.
}

void OgaEngine::expect_prefill(const std::vector<int>& ids) {
    expected_ = logical_;
    expected_.insert(expected_.end(), ids.begin(), ids.end());
}

void OgaEngine::skip_next_logits() noexcept { skip_next_logits_ = true; }

buffer<bf16> OgaEngine::get_k_cache(int, int) {
    throw std::runtime_error(display_name_ + " on OGA does not expose the KV cache");
}

buffer<bf16> OgaEngine::get_v_cache(int, int) {
    throw std::runtime_error(display_name_ + " on OGA does not expose the KV cache");
}

int OgaEngine::get_current_context_length() { return static_cast<int>(logical_.size()); }

int OgaEngine::checkpoint() {
    skip_next_logits_ = false;
    saved_ = Saved{logical_, images_};
    expected_.clear();
    // A slot rewound to this very checkpoint and given nothing since keeps the snapshot and logits it has.
    if (rewinds_ && active_ && !logical_.empty() && Active().ran == logical_ && !Active().rewound) {
        Live& run = Active();
        try {
            if (run.last_logits.empty()) (void)EnsureLogits(run, /*cache_logits=*/true);
            std::vector<bf16> kept = run.last_logits;
            api_->Check(api_->functions().generator_snapshot_state(run.generator.get()),
                        "OgaGenerator_SnapshotState");
            run.snapshot = logical_.size();
            run.snapshot_logits = std::move(kept);
        } catch (const std::exception& e) {
            if (active_) {
                Live& current = Active();
                current.snapshot.reset();
                current.snapshot_logits.clear();
                if (!current.generator) Drop(*active_);
            }
            header_print("WARNING", display_name_ << ": " << e.what()
                                    << "; a prompt that leaves this checkpoint replays its history");
        }
    }
    return static_cast<int>(logical_.size());
}

int OgaEngine::restore() {
    skip_next_logits_ = false;
    if (!saved_) return -1;
    logical_ = saved_->tokens;
    images_ = saved_->images;
    expected_.clear();
    return static_cast<int>(logical_.size());
}

buffer<bf16> OgaEngine::Run(const std::vector<int>& fresh, const OgaVisionInput* vision,
                            bool skip_logits, bool cache_logits) {
    CheckIds(fresh);
    if (logical_.size() + fresh.size() > max_length_)
        throw std::out_of_range(display_name_ + " request exceeds configured context capacity");
    std::vector<ImageSpan> spans = ImageSpans(fresh, logical_.size(), vision);
    const std::size_t before = logical_.size();
    const std::size_t images_before = images_.size();
    logical_.insert(logical_.end(), fresh.begin(), fresh.end());
    images_.insert(images_.end(), std::make_move_iterator(spans.begin()),
                   std::make_move_iterator(spans.end()));
    try {
        const char* why = nullptr;
        const auto [reuse, reused] = Append(&why, skip_logits);
        if (reused < before)
            replayed_ += before - reused;
        else
            reused_ += reused - before;
        const std::size_t total = logical_.size();
        Live& run = Active();
        if (reuse == Reuse::kRebuilt) {
            const std::string with =
                images_.empty() ? "" : " with " + std::to_string(images_.size()) + " image(s)";
            header_print("FLM", "OGA generator: replaying all " << total << " tokens" << with
                                    << SlotText(why));
        } else if (reuse == Reuse::kRewound) {
            const std::size_t at = *run.snapshot;
            header_print("FLM", "OGA generator: rewound to " << at << " tokens"
                                    << ", appending " << (total > at ? total - at : 0) << SlotText());
        } else if (total - reused > 1)
            header_print("FLM", "OGA generator: reusing " << reused << " tokens"
                                    << ", appending " << total - reused << SlotText());
        if (skip_logits) return {};
        // Ahead of a chunked prompt: no caller reads the logits of a chunk before the last.
        if (run.ran.size() > total) {
            buffer<bf16> zeros(vocab_size_);
            std::fill(zeros.data(), zeros.data() + vocab_size_, bf16(0.0f));
            return zeros;
        }
        if (run.rewound) {
            buffer<bf16> out(vocab_size_);
            std::copy(run.snapshot_logits.begin(), run.snapshot_logits.end(), out.data());
            return out;
        }
        return EnsureLogits(run, cache_logits);
    } catch (...) {
        logical_.resize(before);
        images_.resize(images_before);
        if (active_) Drop(*active_);
        throw;
    }
}

std::vector<OgaEngine::ImageSpan> OgaEngine::ImageSpans(const std::vector<int>& fresh,
                                                        std::size_t offset,
                                                        const OgaVisionInput* vision) const {
    std::vector<ImageSpan> spans;
    if (!vision_) return spans;
    const OgaVisionConfig& config = *vision_;
    const std::size_t images = vision ? vision->images.size() : 0;
    const std::int64_t merged = config.spatial_merge_size * config.spatial_merge_size;
    for (std::size_t i = 0; i < fresh.size();) {
        if (fresh[i] != config.image_token_id) {
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < fresh.size() && fresh[end] == config.image_token_id) ++end;
        const std::string which = display_name_ + " image " + std::to_string(spans.size() + 1);
        if (spans.size() == images)
            throw std::invalid_argument(display_name_ + " prompt has more runs of image tokens than its " +
                                        std::to_string(images) + " image(s)");
        const int before = i ? fresh[i - 1] : (offset ? logical_[offset - 1] : -1);
        if (before != config.vision_start_token_id)
            throw std::invalid_argument(which + " does not follow a vision start token");
        const OgaImage& image = vision->images[spans.size()];
        if (image.grid_t <= 0 || image.grid_h <= 0 || image.grid_w <= 0 ||
            image.grid_h % config.spatial_merge_size || image.grid_w % config.spatial_merge_size)
            throw std::invalid_argument(which + " has an invalid grid");
        const std::int64_t grid = image.grid_t * image.grid_h * image.grid_w;
        if (image.pixels.size() != static_cast<std::size_t>(grid * config.patch_values))
            throw std::invalid_argument(which + " has " + std::to_string(image.pixels.size()) +
                                        " pixel values, not " + std::to_string(grid * config.patch_values));
        if (end - i != static_cast<std::size_t>(grid / merged))
            throw std::invalid_argument(which + " has " + std::to_string(grid / merged) +
                                        " image tokens, but its run in the prompt has " +
                                        std::to_string(end - i));
        spans.push_back({offset + i, end - i, std::make_shared<const OgaImage>(image), ImageHash(image)});
        i = end;
    }
    if (spans.size() != images)
        throw std::invalid_argument(display_name_ + " prompt has " + std::to_string(spans.size()) +
                                    " runs of image tokens for " + std::to_string(images) + " image(s)");
    return spans;
}

bool OgaEngine::ImageAfter(std::size_t position) const {
    return !images_.empty() && images_.back().start + images_.back().length > position;
}

std::pair<OgaEngine::Reuse, std::size_t> OgaEngine::Append(const char** why, bool skip_logits) {
    active_.reset();  // so that a failure before Resume drops no other conversation's slot
    const Reuse reuse = Resume(why);
    Live& run = Active();
    const std::size_t from = run.ran.size();
    // Equal lengths: the slot already ran the last token, so its logits are current, or saved
    // with the snapshot it was rewound to. Longer: it is ahead of a chunked prompt.
    if (from >= logical_.size()) return {reuse, logical_.size()};
    run.last_logits.clear();
    if (from == 0 && (!images_.empty() || empty_image_batch_)) {
        std::vector<const OgaImage*> images;
        for (const ImageSpan& span : images_) {
            images.push_back(span.image.get());
            run.images.push_back(span.hash);
        }
        SetInputs(run.generator.get(), logical_, images);
        run.needs_first_run_check = false;
        run.first_run_len = 0;
    } else if (from == 0) {
        AppendFirst(run, logical_, skip_logits);
    } else {
        AppendTokens(run.generator.get(), logical_.data() + from, logical_.size() - from);
    }
    run.rewound = false;
    const auto rest = logical_.begin() + static_cast<std::ptrdiff_t>(from);
    run.ran.insert(run.ran.end(), rest, logical_.end());
    return {reuse, from};
}

const std::vector<int>& OgaEngine::Target() {
    if (expected_.size() > logical_.size() &&
        std::equal(logical_.begin(), logical_.end(), expected_.begin()))
        return expected_;
    expected_.clear();
    return logical_;
}

OgaEngine::Reach OgaEngine::ReachOf(const Live& slot, const std::vector<int>& target) const {
    if (target.size() > std::min(slot.capacity, max_length_)) return {};
    // Once another slot proved the package needs a split first run, never reuse an unsplit slot.
    if (split_first_run_ && slot.needs_first_run_check) return {};
    // The tokens of an image tell only its size.
    const bool same_images =
        std::equal(slot.images.begin(), slot.images.end(), images_.begin(), images_.end(),
                   [](std::uint64_t hash, const ImageSpan& span) { return hash == span.hash; });
    if (!same_images) return {};
    // OGA takes images only in a generator's first call.
    const auto reaches = [&](std::size_t length) {
        return length <= target.size() &&
               std::equal(slot.ran.begin(), slot.ran.begin() + static_cast<std::ptrdiff_t>(length),
                          target.begin()) &&
               !ImageAfter(length);
    };
    Reach reach;
    if (reaches(slot.ran.size())) reach = {slot.ran.size(), false};
    // The snapshot's logits are saved with it, so it may end where the target ends.
    if (rewinds_ && slot.snapshot && *slot.snapshot <= slot.ran.size() &&
        *slot.snapshot > reach.length && reaches(*slot.snapshot))
        reach = {*slot.snapshot, true};
    return reach;
}

OgaEngine::Reuse OgaEngine::Resume(const char** why) {
    const std::vector<int>& target = Target();
    std::optional<std::size_t> best;
    Reach reach;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const Reach r = ReachOf(slots_[i], target);
        if (r.length == 0) continue;
        if (!best || r.length > reach.length ||
            (r.length == reach.length && slots_[i].used > slots_[*best].used)) {
            best = i;
            reach = r;
        }
    }
    if (!best) return Rebuild(std::nullopt, why);
    active_ = best;
    Live& run = Active();
    run.used = ++clock_;
    if (!reach.rewind) return Reuse::kSameTokens;
    try {
        api_->Check(api_->functions().generator_rewind_to(run.generator.get(), reach.length),
                    "OgaGenerator_RewindTo");
    } catch (const OgaError& e) {
        // OGA refuses every rewind of a model it cannot rewind, such as gemma4 in OGA 0.19
        rewinds_ = false;
        header_print("WARNING", display_name_ << ": " << e.what()
                                << "; replaying instead, and after every later checkpoint");
        if (why) *why = "its rewind was refused";
        return Rebuild(best, why);
    }
    run.ran.resize(reach.length);
    run.rewound = true;
    run.last_logits.clear();
    run.needs_first_run_check = false;
    run.first_run_len = 0;
    return Reuse::kRewound;
}

OgaEngine::Reuse OgaEngine::Rebuild(std::optional<std::size_t> slot, const char** why) {
    if (!slot) {
        if (slots_.size() < slot_limit()) {
            slots_.emplace_back();
            slot = slots_.size() - 1;
            if (why) *why = "new";
        } else {
            slot = LeastRecentlyUsed();
            if (why) *why = "replacing the least recently used";
        }
    }
    active_ = slot;
    Live& run = Active();
    // The old generator goes before the new one takes its memory.
    run = Live{};
    run.used = ++clock_;
    run.generator = NewGenerator(max_length_);
    run.capacity = max_length_;
    return Reuse::kRebuilt;
}

std::size_t OgaEngine::LeastRecentlyUsed() const {
    std::size_t oldest = 0;
    for (std::size_t i = 1; i < slots_.size(); ++i)
        if (slots_[i].used < slots_[oldest].used) oldest = i;
    return oldest;
}

void OgaEngine::Drop(std::size_t slot) {
    slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(slot));
    if (!active_) return;
    if (*active_ == slot)
        active_.reset();
    else if (*active_ > slot)
        --*active_;
}

void OgaEngine::Trim() {
    while (slots_.size() > slot_limit()) Drop(LeastRecentlyUsed());
}

std::string OgaEngine::SlotText(const char* why) const {
    const std::size_t limit = slot_limit();
    if (limit <= 1 || !active_) return "";
    std::string text = " (slot " + std::to_string(*active_ + 1) + " of " + std::to_string(limit);
    if (why) text += std::string(", ") + why;
    return text + ")";
}

void OgaEngine::SetInputs(OgaGenerator* generator, const std::vector<int>& ids,
                          const std::vector<const OgaImage*>& images) const {
    const OgaFunctions& f = api_->functions();
    OgaNamedTensors* raw_named = nullptr;
    api_->Check(f.create_named_tensors(&raw_named), "OgaCreateNamedTensors");
    OgaApi::Owned<OgaNamedTensors> named(raw_named, {f.destroy_named_tensors});
    // The named tensors hold their own reference, and so does the generator once it has them.
    const auto set = [&](const char* name, const OgaApi::Owned<OgaTensor>& tensor) {
        api_->Check(f.named_tensors_set(named.get(), name, tensor.get()),
                    std::string("OgaNamedTensorsSet(") + name + ")");
    };
    set("input_ids", NewTensor({1, static_cast<std::int64_t>(ids.size())}, OgaElementType_int32,
                               ids.data(), ids.size() * sizeof(int)));
    if (images.empty()) {
        // OGA counts the images by pixel_values' leading dimension; without image tokens the
        // vision model does not run
        set("pixel_values", NewTensor({1, 0, 0}, OgaElementType_float16, nullptr, 0));
        api_->Check(f.generator_set_inputs(generator, named.get()), "OgaGenerator_SetInputs");
        return;
    }
    const std::int64_t merged = vision_->spatial_merge_size * vision_->spatial_merge_size;
    std::int64_t patches = 0;
    std::vector<std::int64_t> grids;
    std::vector<std::int64_t> tokens;
    for (const OgaImage* image : images) {
        const std::int64_t grid = image->grid_t * image->grid_h * image->grid_w;
        patches += grid;
        grids.insert(grids.end(), {image->grid_t, image->grid_h, image->grid_w});
        tokens.push_back(grid / merged);
    }
    const auto count = static_cast<std::int64_t>(images.size());
    const OgaApi::Owned<OgaTensor> pixels =
        NewTensor({patches, vision_->patch_values}, OgaElementType_float16, nullptr, 0);
    void* data = nullptr;
    api_->Check(f.tensor_get_data(pixels.get(), &data), "OgaTensorGetData(pixel_values)");
    auto* to = static_cast<std::uint16_t*>(data);
    for (const OgaImage* image : images) to = std::copy(image->pixels.begin(), image->pixels.end(), to);
    set("pixel_values", pixels);
    set("image_grid_thw", NewTensor({count, 3}, OgaElementType_int64, grids.data(),
                                    grids.size() * sizeof(std::int64_t)));
    set("num_image_tokens", NewTensor({count}, OgaElementType_int64, tokens.data(),
                                      tokens.size() * sizeof(std::int64_t)));
    api_->Check(f.generator_set_inputs(generator, named.get()), "OgaGenerator_SetInputs");
}

OgaApi::Owned<OgaTensor> OgaEngine::NewTensor(const std::vector<std::int64_t>& shape,
                                              OgaElementType type, const void* data,
                                              std::size_t bytes) const {
    const OgaFunctions& f = api_->functions();
    OgaTensor* raw_tensor = nullptr;
    api_->Check(f.create_tensor_from_buffer(nullptr, shape.data(), shape.size(), type, &raw_tensor),
                "OgaCreateTensorFromBuffer");
    OgaApi::Owned<OgaTensor> tensor(raw_tensor, {f.destroy_tensor});
    if (data && bytes) {
        void* to = nullptr;
        api_->Check(f.tensor_get_data(tensor.get(), &to), "OgaTensorGetData");
        std::memcpy(to, data, bytes);
    }
    return tensor;
}

OgaApi::Owned<OgaGenerator> OgaEngine::NewGenerator(std::uint32_t max_length) const {
    const auto& f = api_->functions();
    OgaGeneratorParams* raw_params = nullptr;
    api_->Check(f.create_generator_params(model_.get(), &raw_params), "OgaCreateGeneratorParams");
    OgaApi::Owned<OgaGeneratorParams> params(raw_params, {f.destroy_generator_params});
    api_->Check(f.generator_params_set_search_number(params.get(), "max_length", max_length),
                "OgaGeneratorParamsSetSearchNumber(max_length)");
    OgaGenerator* raw_generator = nullptr;
    api_->Check(f.create_generator(model_.get(), params.get(), &raw_generator),
                "OgaCreateGenerator");
    return OgaApi::Owned<OgaGenerator>(raw_generator, {f.destroy_generator});
}

void OgaEngine::AppendTokens(OgaGenerator* generator, const int* ids, std::size_t count) const {
    api_->Check(api_->functions().generator_append_tokens(generator, ids, count),
                "OgaGenerator_AppendTokens");
}

void OgaEngine::AppendFirst(Live& run, const std::vector<int>& ids, bool skip_logits) {
    const std::size_t count = ids.size();
    if (!split_first_run_ || count < 2) {
        AppendTokens(run.generator.get(), ids.data(), count);
        if (split_first_run_ || count < 2 || first_run_verified_) return;
        if (!skip_logits || count < kFirstRunVerifyMinTokens) {
            run.needs_first_run_check = true;
            run.first_run_len = count;
            return;
        }
        // One-time probe when the first >=8-token run is on a skipped intermediate chunk:
        // a shorter final chunk may not expose the Laguna 2.1 all-zero-logits failure.
        const buffer<bf16> logits = Logits(run.generator.get());
        if (std::any_of(logits.data(), logits.data() + logits.size(),
                        [](const bf16& value) { return static_cast<float>(value) != 0.0f; })) {
            first_run_verified_ = true;
            run.last_logits.assign(logits.data(), logits.data() + logits.size());
            return;
        }
        split_first_run_ = true;
        header_print("WARNING", display_name_ << ": a new generator's first run gave all-zero logits;"
                                " running it again, and from now on each new generator runs the"
                                " first prompt token on its own");
        // The old generator goes before the new one takes its memory.
        run.generator.reset();
        auto fresh = NewGenerator(run.capacity);
        AppendTokens(fresh.get(), ids.data(), 1);
        AppendTokens(fresh.get(), ids.data() + 1, count - 1);
        run.generator = std::move(fresh);
        return;
    }
    AppendTokens(run.generator.get(), ids.data(), 1);
    AppendTokens(run.generator.get(), ids.data() + 1, count - 1);
}

buffer<bf16> OgaEngine::EnsureLogits(Live& run, bool cache_logits) {
    if (!run.last_logits.empty() && !(run.needs_first_run_check && split_first_run_)) {
        buffer<bf16> out(vocab_size_);
        std::memcpy(out.data(), run.last_logits.data(), vocab_size_ * sizeof(bf16));
        return out;
    }
    if (run.needs_first_run_check && split_first_run_ && run.ran.size() >= 2) {
        run.needs_first_run_check = false;
        run.first_run_len = 0;
        run.last_logits.clear();
        run.generator.reset();
        auto fresh = NewGenerator(run.capacity);
        AppendTokens(fresh.get(), run.ran.data(), 1);
        AppendTokens(fresh.get(), run.ran.data() + 1, run.ran.size() - 1);
        run.generator = std::move(fresh);
        buffer<bf16> out = Logits(run.generator.get());
        if (cache_logits) run.last_logits.assign(out.data(), out.data() + out.size());
        return out;
    }
    buffer<bf16> out = Logits(run.generator.get());
    if (run.needs_first_run_check) {
        const std::size_t first_len = run.first_run_len;
        run.needs_first_run_check = false;
        run.first_run_len = 0;
        const bool any_nonzero =
            std::any_of(out.data(), out.data() + out.size(),
                        [](const bf16& value) { return static_cast<float>(value) != 0.0f; });
        if (any_nonzero) {
            if (first_len >= kFirstRunVerifyMinTokens) first_run_verified_ = true;
        } else if (run.ran.size() >= 2) {
            split_first_run_ = true;
            header_print("WARNING", display_name_ << ": a new generator's first run gave all-zero logits;"
                                    " running it again, and from now on each new generator runs the"
                                    " first prompt token on its own");
            // The old generator goes before the new one takes its memory.
            run.generator.reset();
            auto fresh = NewGenerator(run.capacity);
            AppendTokens(fresh.get(), run.ran.data(), 1);
            AppendTokens(fresh.get(), run.ran.data() + 1, run.ran.size() - 1);
            run.generator = std::move(fresh);
            out = Logits(run.generator.get());
        }
    }
    if (cache_logits) run.last_logits.assign(out.data(), out.data() + out.size());
    return out;
}

buffer<bf16> OgaEngine::Logits(OgaGenerator* generator) const {
    const auto& f = api_->functions();
    OgaTensor* raw_tensor = nullptr;
    api_->Check(f.generator_get_logits(generator, &raw_tensor),
                "OgaGenerator_GetLogits");
    OgaApi::Owned<OgaTensor> tensor(raw_tensor, {f.destroy_tensor});
    OgaElementType type = OgaElementType_undefined;
    api_->Check(f.tensor_get_type(tensor.get(), &type), "OgaTensorGetType");
    std::size_t rank = 0;
    api_->Check(f.tensor_get_shape_rank(tensor.get(), &rank), "OgaTensorGetShapeRank");
    std::vector<std::int64_t> shape(rank);
    api_->Check(f.tensor_get_shape(tensor.get(), shape.data(), rank), "OgaTensorGetShape");
    std::size_t count = 1;
    for (const std::int64_t dim : shape) count *= dim > 0 ? static_cast<std::size_t>(dim) : 0;
    if (rank < 2 || shape.front() != 1 || shape.back() != vocab_size_ || count == 0)
        throw std::runtime_error(display_name_ + " logits have shape " + ShapeText(shape) +
                                 ", not [1, ..., " + std::to_string(vocab_size_) + "]");
    void* data = nullptr;
    api_->Check(f.tensor_get_data(tensor.get(), &data), "OgaTensorGetData");

    const std::size_t last_row = count - vocab_size_;
    buffer<bf16> out(vocab_size_);
    bf16* to = out.data();
    switch (type) {
    case OgaElementType_float32: {
        const float* from = static_cast<const float*>(data) + last_row;
        ConvertFp32ToBf16(from, to, vocab_size_);
        break;
    }
    case OgaElementType_float16: {
        const std::uint16_t* from = static_cast<const std::uint16_t*>(data) + last_row;
        for (std::uint32_t i = 0; i < vocab_size_; ++i) to[i] = bf16(HalfToFloat(from[i]));
        break;
    }
    case OgaElementType_bfloat16: {
        const std::uint16_t* from = static_cast<const std::uint16_t*>(data) + last_row;
        for (std::uint32_t i = 0; i < vocab_size_; ++i) to[i] = bf16(from[i], true);
        break;
    }
    default:
        throw std::runtime_error(display_name_ + " logits have OGA element type " +
                                 std::to_string(static_cast<int>(type)) +
                                 ", not float32, float16 or bfloat16");
    }
    return out;
}

buffer<bf16> OgaEngine::PrefillStaged(const StagedPrompt& staged, bool skip_logits) {
    // The processor inputs hold the whole sequence and its images, which OGA takes only in a
    // generator's first call. A slot given the same images that ran a prefix of the sequence
    // takes the rest as text; otherwise a new generator, in a free slot or the least recently used one.
    active_.reset();  // so that a failure before Rebuild drops no other conversation's slot
    try {
        const char* why = nullptr;
        std::optional<std::size_t> found;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            const Live& slot = slots_[i];
            if (!slot.staged_images.empty() && slot.staged_images == staged.images &&
                slot.ran.size() < staged.ids.size() &&
                staged.ids.size() <= std::min(slot.capacity, max_length_) &&
                std::equal(slot.ran.begin(), slot.ran.end(), staged.ids.begin()) &&
                (!found || slot.ran.size() > slots_[*found].ran.size()))
                found = i;
        }
        std::size_t reused = 0;
        if (found) {
            active_ = found;
            Live& run = Active();
            run.used = ++clock_;
            reused = run.ran.size();
            run.last_logits.clear();
            run.needs_first_run_check = false;
            run.first_run_len = 0;
            AppendTokens(run.generator.get(), staged.ids.data() + reused, staged.ids.size() - reused);
            run.rewound = false;
        } else {
            Rebuild(std::nullopt, &why);
            api_->Check(api_->functions().generator_set_inputs(Active().generator.get(), staged.inputs.get()),
                        "OgaGenerator_SetInputs");
        }
        Live& run = Active();
        buffer<bf16> logits;
        if (!skip_logits) {
            logits = Logits(run.generator.get());
            run.last_logits.assign(logits.data(), logits.data() + logits.size());
        }
        // Only now: a failed prefill leaves `logical` without image tokens that a later
        // rebuild would replay without their images.
        run.ran = staged.ids;
        run.staged_images = staged.images;
        logical_ = staged.ids;
        images_.clear();
        expected_.clear();
        saved_.reset();
        reused_ += reused;
        if (found)
            header_print("FLM", "OGA generator: image prompt of " << logical_.size() << " tokens, reusing "
                                    << reused << " with its images, appending "
                                    << logical_.size() - reused << SlotText());
        else
            header_print("FLM", "OGA generator: image prompt of " << logical_.size() << " tokens"
                                    << SlotText(why));
        return logits;
    } catch (...) {
        if (active_) Drop(*active_);
        throw;
    }
}

void OgaEngine::DropStagedPrompt() noexcept { staged_prompt_.reset(); }

std::vector<int> OgaEngine::StagePromptImages(const std::string& prompt,
                                              const std::vector<std::string>& encoded_images) {
    staged_prompt_.reset();
    if (encoded_images.empty())
        throw std::invalid_argument(display_name_ + " image prompt has no images");
    if (prompt.empty())
        throw std::invalid_argument(display_name_ + " image prompt is empty");
    const OgaFunctions& f = api_->functions();
    Require(reinterpret_cast<const void*>(f.load_images_from_buffers), "OgaLoadImagesFromBuffers");
    Require(reinterpret_cast<const void*>(f.destroy_images), "OgaDestroyImages");
    Require(reinterpret_cast<const void*>(f.create_multimodal_processor), "OgaCreateMultiModalProcessor");
    Require(reinterpret_cast<const void*>(f.destroy_multimodal_processor), "OgaDestroyMultiModalProcessor");
    Require(reinterpret_cast<const void*>(f.processor_process_images), "OgaProcessorProcessImages");
    Require(reinterpret_cast<const void*>(f.named_tensors_get), "OgaNamedTensorsGet");

    std::vector<const void*> buffers;
    std::vector<std::size_t> sizes;
    buffers.reserve(encoded_images.size());
    sizes.reserve(encoded_images.size());
    for (const std::string& image : encoded_images) {
        if (image.size() < 8)
            throw std::invalid_argument(display_name_ + " image is not an encoded file");
        buffers.push_back(image.data());
        sizes.push_back(image.size());
    }
    OgaImages* raw_images = nullptr;
    api_->Check(f.load_images_from_buffers(buffers.data(), sizes.data(), buffers.size(), &raw_images),
                "OgaLoadImagesFromBuffers");
    OgaApi::Owned<OgaImages> images(raw_images, {f.destroy_images});
    OgaMultiModalProcessor* raw_processor = nullptr;
    api_->Check(f.create_multimodal_processor(model_.get(), &raw_processor),
                "OgaCreateMultiModalProcessor");
    OgaApi::Owned<OgaMultiModalProcessor> processor(raw_processor, {f.destroy_multimodal_processor});
    OgaNamedTensors* raw_inputs = nullptr;
    api_->Check(f.processor_process_images(processor.get(), prompt.c_str(), images.get(), &raw_inputs),
                "OgaProcessorProcessImages");
    OgaApi::Owned<OgaNamedTensors> inputs(raw_inputs, {f.destroy_named_tensors});
    OgaTensor* raw_ids = nullptr;
    api_->Check(f.named_tensors_get(inputs.get(), "input_ids", &raw_ids),
                "OgaNamedTensorsGet(input_ids)");
    std::vector<int> ids = ReadInputIds(*api_, raw_ids);
    CheckIds(ids);
    if (ids.size() > max_length_)
        throw std::out_of_range(display_name_ + " image prompt has " + std::to_string(ids.size()) +
                                " tokens, past the context length of " + std::to_string(max_length_));
    std::vector<std::uint64_t> hashes;
    for (const std::string& image : encoded_images) hashes.push_back(BytesHash(image));
    staged_prompt_ = std::make_unique<StagedPrompt>(StagedPrompt{std::move(inputs), ids, std::move(hashes)});
    header_print("FLM", "OGA image processor: " << encoded_images.size() << " image(s), "
                                                << ids.size() << " tokens");
    return ids;
}

void OgaEngine::CheckIds(const std::vector<int>& ids) const {
    for (const int id : ids)
        if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_size_)
            throw std::out_of_range(display_name_ + " token " + std::to_string(id) +
                                    " is outside the vocabulary of " +
                                    std::to_string(vocab_size_));
}

}  // namespace flm::oga
