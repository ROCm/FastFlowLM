/// \file oga_engine.cpp
/// \brief Exact-token execution of an OGA text model
#include "oga/oga_engine.hpp"

#include "utils/debug_utils.hpp"

#include <algorithm>
#include <bit>
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

}  // namespace

OgaEngine::OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
                     std::string display_name, std::uint32_t vocab_size,
                     std::uint32_t context_limit, std::uint32_t max_length)
    : api_(std::move(api)),
      display_name_(std::move(display_name)),
      vocab_size_(vocab_size),
      context_limit_(context_limit),
      max_length_(max_length),
      model_(std::move(model)) {
    if (!api_ || !model_) throw std::invalid_argument("OGA engine needs an API and a model");
    if (!vocab_size_) throw std::invalid_argument(display_name_ + " vocabulary size is 0");
    if (!max_length_ || max_length_ > context_limit_)
        throw std::invalid_argument(display_name_ + " maximum length must be in 1.." +
                                    std::to_string(context_limit_));
}

OgaEngine::~OgaEngine() = default;

buffer<bf16> OgaEngine::forward(int id) { return Run({id}); }

buffer<bf16> OgaEngine::prefill(std::vector<int>& ids, void* payload) {
    if (payload) throw std::invalid_argument(display_name_ + " on OGA does not run a vision payload");
    if (ids.empty()) throw std::invalid_argument(display_name_ + " request contains no token IDs");
    return Run(ids);
}

void OgaEngine::set_context_length(int length) {
    if (length < 0 || static_cast<std::uint32_t>(length) > max_length_)
        throw std::out_of_range(display_name_ + " context length is out of range");
    if (static_cast<std::size_t>(length) > logical_.size())
        throw std::runtime_error(display_name_ + " cannot rewind past the tokens it has run");
    logical_.resize(static_cast<std::size_t>(length));
}

void OgaEngine::load_weights(Q4NX&) {
    throw std::runtime_error(display_name_ + " on OGA loads its weights from the ONNX package");
}

void OgaEngine::update_max_length(std::uint32_t max_length) {
    if (!max_length || max_length > context_limit_ || max_length < logical_.size())
        throw std::out_of_range(display_name_ + " maximum length is invalid");
    if (live_ && max_length > live_->capacity) live_.reset();
    if (saved_ && max_length < saved_->size()) saved_.reset();
    max_length_ = max_length;
}

void OgaEngine::clear_context() {
    logical_.clear();
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
    saved_ = logical_;
    return static_cast<int>(logical_.size());
}

int OgaEngine::restore() {
    if (!saved_) return -1;
    logical_ = *saved_;
    return static_cast<int>(logical_.size());
}

buffer<bf16> OgaEngine::Run(const std::vector<int>& fresh) {
    CheckIds(fresh);
    if (logical_.size() + fresh.size() > max_length_)
        throw std::out_of_range(display_name_ + " request exceeds configured context capacity");
    const std::size_t before = logical_.size();
    logical_.insert(logical_.end(), fresh.begin(), fresh.end());
    try {
        const auto [reuse, reused] = Append();
        const std::size_t total = logical_.size();
        if (reuse == Reuse::kRebuilt)
            header_print("FLM", "OGA generator: replaying all " << total << " tokens");
        else if (total - reused > 1)
            header_print("FLM", "OGA generator: reusing " << reused << " tokens"
                                    << ", appending " << total - reused);
        return Logits(live_->generator.get());
    } catch (...) {
        logical_.resize(before);
        live_.reset();
        throw;
    }
}

std::pair<OgaEngine::Reuse, std::size_t> OgaEngine::Append() {
    const Reuse reuse = Resume();
    Live& run = *live_;
    const std::size_t from = run.ran.size();
    // Equal lengths: the generator already ran the last token, so its logits are current.
    if (from == logical_.size()) return {reuse, from};
    AppendTokens(run.generator.get(), logical_.data() + from, logical_.size() - from);
    const auto rest = logical_.begin() + static_cast<std::ptrdiff_t>(from);
    run.ran.insert(run.ran.end(), rest, logical_.end());
    return {reuse, from};
}

OgaEngine::Reuse OgaEngine::Resume() {
    if (live_ && logical_.size() <= std::min(live_->capacity, max_length_) &&
        live_->ran.size() <= logical_.size() &&
        std::equal(live_->ran.begin(), live_->ran.end(), logical_.begin()))
        return Reuse::kSameTokens;
    return Rebuild();
}

OgaEngine::Reuse OgaEngine::Rebuild() {
    live_.reset();
    live_.emplace(Live{NewGenerator(max_length_), max_length_, {}});
    return Reuse::kRebuilt;
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
