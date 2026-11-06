// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_engine.cpp
/// \brief Exact-token execution of an OGA model, with the images of its prompts
#include "oga/oga_engine.hpp"

#include "utils/debug_utils.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace flm::oga {
namespace {
static_assert(sizeof(int) == sizeof(std::int32_t), "token ids are passed to OGA as int32_t");

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

OgaEngine::OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
                     std::string display_name, std::uint32_t vocab_size,
                     std::uint32_t context_limit, std::uint32_t max_length,
                     std::optional<OgaVisionConfig> vision, bool empty_image_batch)
    : api_(std::move(api)),
      display_name_(std::move(display_name)),
      vocab_size_(vocab_size),
      context_limit_(context_limit),
      max_length_(max_length),
      vision_(std::move(vision)),
      empty_image_batch_(empty_image_batch),
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
            v.patch_values <= 0 || v.max_patches < v.spatial_merge_size * v.spatial_merge_size)
            throw std::invalid_argument(display_name_ + " vision configuration is invalid");
    }
    const OgaFunctions& f = api_->functions();
    rewinds_ = AtLeast(api_->package().oga_version, kRewindOgaVersion) &&
               f.generator_snapshot_state && f.generator_rewind_to;
}

OgaEngine::~OgaEngine() = default;

buffer<bf16> OgaEngine::forward(int id) { return Run({id}); }

buffer<bf16> OgaEngine::prefill(std::vector<int>& ids, void* payload) {
    if (payload && !vision_)
        throw std::invalid_argument(display_name_ + " on OGA does not run a vision payload");
    if (ids.empty()) throw std::invalid_argument(display_name_ + " request contains no token IDs");
    return Run(ids, static_cast<const OgaVisionInput*>(payload));
}

void OgaEngine::set_context_length(int length) {
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
}

void OgaEngine::load_weights(Q4NX&) {
    throw std::runtime_error(display_name_ + " on OGA loads its weights from the ONNX package");
}

void OgaEngine::update_max_length(std::uint32_t max_length) {
    if (!max_length || max_length > context_limit_ || max_length < logical_.size())
        throw std::out_of_range(display_name_ + " maximum length is invalid");
    if (live_ && max_length > live_->capacity) live_.reset();
    if (saved_ && max_length < saved_->tokens.size()) saved_.reset();
    max_length_ = max_length;
}

void OgaEngine::clear_context() {
    logical_.clear();
    images_.clear();
    saved_.reset();
}

buffer<bf16> OgaEngine::get_k_cache(int, int) {
    throw std::runtime_error(display_name_ + " on OGA does not expose the KV cache");
}

buffer<bf16> OgaEngine::get_v_cache(int, int) {
    throw std::runtime_error(display_name_ + " on OGA does not expose the KV cache");
}

int OgaEngine::get_current_context_length() { return static_cast<int>(logical_.size()); }

int OgaEngine::checkpoint() {
    saved_ = Saved{logical_, images_};
    if (rewinds_ && live_ && !logical_.empty() && live_->ran == logical_) {
        try {
            api_->Check(api_->functions().generator_snapshot_state(live_->generator.get()),
                        "OgaGenerator_SnapshotState");
            live_->snapshot = logical_.size();
        } catch (const OgaError& e) {
            live_->snapshot.reset();
            header_print("WARNING", display_name_ << ": " << e.what()
                                    << "; a prompt that leaves this checkpoint replays its history");
        }
    }
    return static_cast<int>(logical_.size());
}

int OgaEngine::restore() {
    if (!saved_) return -1;
    logical_ = saved_->tokens;
    images_ = saved_->images;
    return static_cast<int>(logical_.size());
}

buffer<bf16> OgaEngine::Run(const std::vector<int>& fresh, const OgaVisionInput* vision) {
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
        const auto [reuse, reused] = Append();
        if (reused < before) replayed_ += before - reused;
        const std::size_t total = logical_.size();
        if (reuse == Reuse::kRebuilt) {
            const std::string with =
                images_.empty() ? "" : " with " + std::to_string(images_.size()) + " image(s)";
            header_print("FLM", "OGA generator: replaying all " << total << " tokens" << with);
        } else if (reuse == Reuse::kRewound)
            header_print("FLM", "OGA generator: rewound to " << reused << " tokens"
                                    << ", appending " << total - reused);
        else if (total - reused > 1)
            header_print("FLM", "OGA generator: reusing " << reused << " tokens"
                                    << ", appending " << total - reused);
        return Logits(live_->generator.get());
    } catch (...) {
        logical_.resize(before);
        images_.resize(images_before);
        live_.reset();
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
    std::int64_t patches = 0;
    for (const ImageSpan& span : images_)
        patches += span.image->grid_t * span.image->grid_h * span.image->grid_w;
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
        patches += grid;
        if (patches > config.max_patches)
            throw std::invalid_argument(display_name_ + " prompt images have " + std::to_string(patches) +
                                        " patches, more than the " + std::to_string(config.max_patches) +
                                        " the vision model runs at once");
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

std::pair<OgaEngine::Reuse, std::size_t> OgaEngine::Append() {
    if (!images_.empty() && !vision_warm_) WarmUpVision();
    const Reuse reuse = Resume();
    Live& run = *live_;
    const std::size_t from = run.ran.size();
    // Equal lengths: the generator already ran the last token, so its logits are current.
    if (from == logical_.size()) return {reuse, from};
    if (from == 0 && (!images_.empty() || empty_image_batch_)) {
        std::vector<const OgaImage*> images;
        for (const ImageSpan& span : images_) {
            images.push_back(span.image.get());
            run.images.push_back(span.hash);
        }
        SetInputs(run.generator.get(), logical_, images);
    } else {
        AppendTokens(run.generator.get(), logical_.data() + from, logical_.size() - from);
    }
    const auto rest = logical_.begin() + static_cast<std::ptrdiff_t>(from);
    run.ran.insert(run.ran.end(), rest, logical_.end());
    return {reuse, from};
}

OgaEngine::Reuse OgaEngine::Resume() {
    if (!live_ || logical_.size() > std::min(live_->capacity, max_length_)) return Rebuild();
    Live& run = *live_;
    // The tokens of an image tell only its size.
    const bool same_images =
        std::equal(run.images.begin(), run.images.end(), images_.begin(), images_.end(),
                   [](std::uint64_t hash, const ImageSpan& span) { return hash == span.hash; });
    // OGA takes images only in a generator's first call.
    if (run.ran.size() <= logical_.size() &&
        std::equal(run.ran.begin(), run.ran.end(), logical_.begin()) && !ImageAfter(run.ran.size()) &&
        same_images)
        return Reuse::kSameTokens;
    // A rewind has to leave at least one token to run, or the logits are stale.
    if (run.snapshot && *run.snapshot <= run.ran.size() && *run.snapshot < logical_.size() &&
        std::equal(run.ran.begin(), run.ran.begin() + static_cast<std::ptrdiff_t>(*run.snapshot),
                   logical_.begin()) &&
        !ImageAfter(*run.snapshot) && same_images) {
        try {
            api_->Check(api_->functions().generator_rewind_to(run.generator.get(), *run.snapshot),
                        "OgaGenerator_RewindTo");
        } catch (const OgaError& e) {
            header_print("WARNING", display_name_ << ": " << e.what() << "; replaying instead");
            return Rebuild();
        }
        run.ran.resize(*run.snapshot);
        return Reuse::kRewound;
    }
    return Rebuild();
}

OgaEngine::Reuse OgaEngine::Rebuild() {
    live_.reset();
    live_.emplace(Live{NewGenerator(max_length_), max_length_, {}, std::nullopt});
    return Reuse::kRebuilt;
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

void OgaEngine::WarmUpVision() {
    const OgaVisionConfig& config = *vision_;
    auto side = static_cast<std::int64_t>(std::sqrt(static_cast<double>(config.max_patches)));
    while (side * side > config.max_patches) --side;
    side -= side % config.spatial_merge_size;
    OgaImage image;
    image.grid_h = side;
    image.grid_w = side;
    image.pixels.assign(static_cast<std::size_t>(side * side * config.patch_values), 0);
    std::vector<int> ids(static_cast<std::size_t>(
                             1 + side * side / (config.spatial_merge_size * config.spatial_merge_size)),
                         config.image_token_id);
    ids.front() = config.vision_start_token_id;
    const auto start = std::chrono::steady_clock::now();
    const OgaApi::Owned<OgaGenerator> generator = NewGenerator(static_cast<std::uint32_t>(ids.size()));
    SetInputs(generator.get(), ids, {&image});
    Logits(generator.get());
    vision_warm_ = true;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    header_print("FLM", "OGA vision: warmed up with a " << side << "x" << side << "-patch image in "
                                                        << ms << " ms");
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
        for (std::uint32_t i = 0; i < vocab_size_; ++i) to[i] = bf16(from[i]);
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

void OgaEngine::CheckIds(const std::vector<int>& ids) const {
    for (const int id : ids)
        if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_size_)
            throw std::out_of_range(display_name_ + " token " + std::to_string(id) +
                                    " is outside the vocabulary of " +
                                    std::to_string(vocab_size_));
}

}  // namespace flm::oga
