#include "models/qwen3/rai/aie_next/qwen3_rai.hpp"

#include "models/qwen3/rai/aie_next/qwen3_rai_shape_plan.hpp"
#include "rai/corelib_object.hpp"
#include "rai/host_ops.hpp"
#include "rai/weight_cache.hpp"
#include "rai/weight_source.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace flm::qwen3 {
namespace {
using namespace flm::corelib;

std::string Name(std::size_t i, const char* suffix) { return "blk." + std::to_string(i) + suffix; }

/// \brief every packing parameter the cached bytes depend on
std::string CacheLayout(const Qwen3Config& c, bool tied) {
    std::ostringstream out;
    out << "qwen3-" << c.size << " L" << c.layers << " h" << c.hidden << " q" << c.q_dim()
        << " kv" << c.kv_dim() << " i" << c.intermediate << " v" << c.vocab << " g" << c.group
        << " hg" << c.head_group << (tied ? " tied" : " untied");
    return out.str();
}

/// \brief load-time phase accounting, printed when FLM_RAI_PROFILE_LOAD is set
struct LoadPhases {
    std::chrono::steady_clock::time_point mark{std::chrono::steady_clock::now()};
    double host_prep{}, shape_plan{}, weight_create{}, device_tensors{}, cache_write{};
    std::uint64_t cache_reclaimed{};
    bool from_cache{};

    double Lap() {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - mark).count();
        mark = now;
        return seconds;
    }

    void Report() const {
        const char* enabled = std::getenv("FLM_RAI_PROFILE_LOAD");
        if (!enabled || !*enabled || *enabled == '0') return;
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[FLM]  rai load: "
            << host_prep + shape_plan + weight_create + device_tensors + cache_write
            << " s total  (host prep " << host_prep << ", shape plan " << shape_plan
            << ", weight " << (from_cache ? "cache" : "requantize") << " " << weight_create
            << ", device tensors " << device_tensors << ")";
        if (cache_write > 0.0) out << "  [cache written in " << cache_write << " s]";
        if (cache_reclaimed > 0)
            out << "  [reclaimed " << cache_reclaimed / (1024 * 1024) << " MB of stale cache]";
        std::cout << out.str() << std::endl;
    }
};
}  // namespace

struct qwen3_rai::Impl {
    std::shared_ptr<Qwen3GgufPackage> package;
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    const Qwen3Config* c{};
    LoadPhases phases;
    Qwen3ShapePlan plan;
    std::uint32_t max_length;
    int position{};
    std::optional<int> saved;
    /// \brief the tokens whose K/V rows fill [0, history.size())
    std::vector<int> history;
    bool poisoned{};
    UniqueStream stream;
    std::vector<UniqueMatMulWeights> q_weights, k_weights, v_weights, o_weights;
    std::vector<UniqueSsMlpWeights> mlp_weights;
    std::vector<UniqueRmsNormWeights> q_norm_weights, k_norm_weights;
    UniqueMatMulWeights lm_weights;
    // flat_mha reads the rotary tables on the host and corelib copies nothing,
    // so this storage must outlive the views below -- hence declared first.
    std::vector<float> rope_cosine, rope_sine;
    UniqueHostView cosine, sine;
    UniqueTensor hidden, residual, skip, q, k, q_normed, k_normed, attention, lm_input, logits;
    std::vector<UniqueTensor> k_cache, v_cache;

    Impl(std::shared_ptr<Qwen3GgufPackage> pkg, std::shared_ptr<CorelibRuntime> rt,
         std::uint32_t maximum)
        : package(std::move(pkg)), runtime(std::move(rt)),
          api(runtime ? runtime->api() : nullptr), max_length(maximum) {
        if (!package) throw std::invalid_argument("Qwen3 GGUF package is null");
        if (!runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > kMaxSequenceLength)
            throw std::invalid_argument("Qwen3 maximum length must be in 1..4096");
        c = &package->Config();
        const auto layers = static_cast<std::size_t>(c->layers);

        std::vector<std::vector<std::uint16_t>> attn_norm(layers), ffn_norm(layers), q_norm(layers),
            k_norm(layers);
        for (std::size_t i = 0; i < layers; ++i) {
            const auto& t = package->Layer(i);
            attn_norm[i] = rai::F32ToBf16(t.attn_norm.values);
            ffn_norm[i] = rai::F32ToBf16(t.ffn_norm.values);
            q_norm[i] = rai::F32ToBf16(t.q_norm.values);
            k_norm[i] = rai::F32ToBf16(t.k_norm.values);
        }
        const auto final_norm = rai::F32ToBf16(package->OutputNorm().values);
        const std::array<float, 1> epsilon_f32{c->epsilon};
        const auto epsilon = rai::F32ToBf16(epsilon_f32);
        BuildRopeTables();
        phases.host_prep = phases.Lap();

        auto lease = runtime->AcquireExecution();
        stream = corelib::OpenModelStream(api, kPrefillPdi, kTokenPdi, "qwen3");
        plan = Qwen3ShapePlan::Build(api, stream.get(), *c);
        phases.shape_plan = phases.Lap();

        q_weights.resize(layers); k_weights.resize(layers); v_weights.resize(layers);
        o_weights.resize(layers); mlp_weights.resize(layers);
        q_norm_weights.resize(layers); k_norm_weights.resize(layers);

        // QK-Norm scales go through the reference packer: they are 128 wide,
        // cost nothing to pack, and so are not worth a place in the cache.
        const ryzenai_corelib_rmsnorm_bf16_weights_desc norm_desc{kQkNormWidth, c->epsilon};
        const auto norm = [&](const std::vector<std::uint16_t>& scale, const std::string& label) {
            const ryzenai_corelib_weights_source sources[]{
                corelib::Bf16(ryzenai_corelib_weights_role_scale, scale.data(), scale.size())};
            void* p = nullptr;
            api->Check(api->functions().rmsnorm_weights_create(&norm_desc, sources, std::size(sources),
                                                               nullptr, &p),
                       "ryzenai_corelib_rmsnorm_bf16_weights_create " + label);
            return UniqueRmsNormWeights(api, p);
        };
        for (std::size_t i = 0; i < layers; ++i) {
            q_norm_weights[i] = norm(q_norm[i], Name(i, ".attn_q_norm"));
            k_norm_weights[i] = norm(k_norm[i], Name(i, ".attn_k_norm"));
        }

        const auto matmul = [&](const rai::GgufTensor& tensor, std::int64_t k_in, std::int64_t n_out,
                                std::uint32_t group, const std::string& label) {
            const ryzenai_corelib_matmul_bf16_weights_desc desc{k_in, n_out, group, false};
            const ryzenai_corelib_weights_source sources[]{corelib::GgufQ8(
                ryzenai_corelib_weights_role_qweight, tensor.bytes.data(), tensor.bytes.size(),
                n_out, k_in)};
            const auto options = corelib::PackOptions(kRequantizeThreads);
            void* p = nullptr;
            api->Check(api->functions().matmul_weights_create(
                           &desc, sources, std::size(sources), &options, &p),
                       "ryzenai_corelib_matmul_bf16_weights_create " + label);
            return UniqueMatMulWeights(api, p);
        };
        std::vector<std::function<void()>> creates;
        creates.reserve(layers * 5 + 1);
        for (std::size_t i = 0; i < layers; ++i) {
            creates.push_back([&, i] { q_weights[i] = matmul(package->Layer(i).q, c->hidden, c->q_dim(), c->group, Name(i, ".attn_q")); });
            creates.push_back([&, i] { k_weights[i] = matmul(package->Layer(i).k, c->hidden, c->kv_dim(), c->group, Name(i, ".attn_k")); });
            creates.push_back([&, i] { v_weights[i] = matmul(package->Layer(i).v, c->hidden, c->kv_dim(), c->group, Name(i, ".attn_v")); });
            creates.push_back([&, i] { o_weights[i] = matmul(package->Layer(i).o, c->q_dim(), c->hidden, c->group, Name(i, ".attn_output")); });
            creates.push_back([&, i] {
                const auto& t = package->Layer(i);
                // The fused MLP applies the NEXT block's input norm to its own
                // output, so norm1 is layer i+1's attn_norm, or output_norm
                // after the last layer -- which is why nothing applies a final
                // norm before lm_head.
                const auto& next = i + 1 < layers ? attn_norm[i + 1] : final_norm;
                const ryzenai_corelib_ssmlp_bf16_weights_desc desc{c->hidden, c->intermediate, c->group};
                const ryzenai_corelib_weights_source sources[]{
                    corelib::Bf16(ryzenai_corelib_weights_role_epsilon, epsilon.data(), epsilon.size()),
                    corelib::Bf16(ryzenai_corelib_weights_role_norm0, ffn_norm[i].data(), ffn_norm[i].size()),
                    corelib::Bf16(ryzenai_corelib_weights_role_norm1, next.data(), next.size()),
                    corelib::GgufQ8(ryzenai_corelib_weights_role_gate_qweight, t.gate.bytes.data(), t.gate.bytes.size(), c->intermediate, c->hidden),
                    corelib::GgufQ8(ryzenai_corelib_weights_role_up_qweight, t.up.bytes.data(), t.up.bytes.size(), c->intermediate, c->hidden),
                    corelib::GgufQ8(ryzenai_corelib_weights_role_down_qweight, t.down.bytes.data(), t.down.bytes.size(), c->hidden, c->intermediate)};
                const auto options = corelib::PackOptions(kRequantizeThreads);
                void* p = nullptr;
                api->Check(api->functions().ssmlp_weights_create(
                               &desc, sources, std::size(sources), &options, &p),
                           "ryzenai_corelib_ssmlp_bf16_weights_create layer " +
                               std::to_string(i));
                mlp_weights[i] = UniqueSsMlpWeights(api, p);
            });
        }
        creates.push_back([&] {
            lm_weights = matmul(package->LmHead(), c->hidden, c->vocab, c->head_group,
                                std::string(package->LmHead().name));
        });

        const auto cache = rai::WeightCache::ForGguf(package->Path());
        const auto version = api->runtime_version();
        const auto key = rai::MakeWeightCacheKey(package->Path(), version.major, version.minor,
                                                 version.patch,
                                                 CacheLayout(*c, package->TiedLmHead()),
                                                 creates.size());
        bool from_cache = false;
        if (cache) {
            if (const auto spans = cache->ReadIndex(key))
                from_cache = LoadWeightsFromCache(cache->DataPath(), *spans);
        }
        if (!from_cache) {
            std::cout << "[FLM]  Packing " << creates.size() << " Qwen3-" << c->size
                      << " weights from Q8_0" << (cache ? "; cached for the next load" : "")
                      << std::endl;
            RunCreates(creates);
        }
        phases.weight_create = phases.Lap();
        phases.from_cache = from_cache;
        if (cache && !from_cache) {
            // Whatever is there did not match, or it would have been used.
            phases.cache_reclaimed = cache->Remove();
            WriteWeightCache(*cache, key);
            phases.cache_write = phases.Lap();
        }

        const auto tensor = [&](std::initializer_list<std::int64_t> dims, const char* label) {
            std::vector<std::int64_t> shape(dims);
            void* p = nullptr;
            api->Check(api->functions().create_device_tensor(ryzenai_corelib_data_type_bf16,
                                                             shape.data(), shape.size(), &p),
                       std::string("ryzenai_corelib_create_device_tensor ") + label);
            return UniqueTensor(api, p);
        };
        constexpr auto rows = kMaxSequenceLength;
        hidden = tensor({rows, c->hidden}, "hidden");
        residual = tensor({rows, c->hidden}, "residual");
        skip = tensor({rows, c->hidden}, "skip");
        q = tensor({rows, c->q_dim()}, "query");
        k = tensor({rows, c->kv_dim()}, "key");
        // QK-Norm's destinations. Not in place: at k=128 an in-place RMSNorm
        // corrupts a short run of rows for M >= 1024, on exact kernels too.
        q_normed = tensor({rows, c->q_dim()}, "normed query");
        k_normed = tensor({rows, c->kv_dim()}, "normed key");
        attention = tensor({rows, c->q_dim()}, "attention");
        // lm_head runs one row; a one-row tensor is what selects the token PDI.
        lm_input = tensor({1, c->hidden}, "lm input");
        logits = tensor({1, c->vocab}, "logits");
        k_cache.resize(layers);
        v_cache.resize(layers);
        for (std::size_t i = 0; i < layers; ++i) {
            k_cache[i] = tensor({c->kv_heads, rows, c->head_dim}, "K cache");
            v_cache[i] = tensor({c->kv_heads, rows, c->head_dim}, "V cache");
        }
        const auto host_view = [&](const std::vector<float>& values, const char* label) {
            const std::array<std::int64_t, 2> shape{kMaxSequenceLength, c->head_dim / 2};
            void* p = nullptr;
            api->Check(api->functions().create_host_view(ryzenai_corelib_data_type_fp32, shape.data(),
                                                         shape.size(), values.data(), &p),
                       std::string("ryzenai_corelib_create_host_view ") + label);
            return UniqueHostView(api, p);
        };
        cosine = host_view(rope_cosine, "cosine");
        sine = host_view(rope_sine, "sine");
        phases.device_tensors = phases.Lap();
        phases.Report();
    }

    /// \brief cos/sin over the whole head, from the file's own rope theta
    /// \note Qwen3 rotates all of head_dim and ships no rope factors, so theta
    ///       is the whole derivation. Double precision throughout; float32
    ///       drifts measurably at the far end of the table.
    void BuildRopeTables() {
        const auto half = static_cast<std::size_t>(c->head_dim / 2);
        rope_cosine.resize(static_cast<std::size_t>(kMaxSequenceLength) * half);
        rope_sine.resize(rope_cosine.size());
        for (std::size_t i = 0; i < half; ++i) {
            const double inv_freq =
                1.0 / std::pow(c->rope_theta, 2.0 * static_cast<double>(i) / static_cast<double>(c->head_dim));
            for (std::size_t p = 0; p < static_cast<std::size_t>(kMaxSequenceLength); ++p) {
                const double angle = static_cast<double>(p) * inv_freq;
                rope_cosine[p * half + i] = static_cast<float>(std::cos(angle));
                rope_sine[p * half + i] = static_cast<float>(std::sin(angle));
            }
        }
    }

    void RunCreates(const std::vector<std::function<void()>>& creates) {
        std::atomic<std::size_t> next{0};
        std::mutex failure_mutex;
        std::exception_ptr first_failure;
        const auto work = [&] {
            for (std::size_t i = next++; i < creates.size(); i = next++) {
                try {
                    creates[i]();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!first_failure) first_failure = std::current_exception();
                    next = creates.size();
                    return;
                }
            }
        };
        const auto workers = std::min(kWeightCreateConcurrency, creates.size());
        std::vector<std::thread> pool;
        for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(work);
        work();
        for (auto& worker : pool) worker.join();
        if (first_failure) std::rethrow_exception(first_failure);
    }

    /// \note Slots are five per layer (Q, K, V, O, MLP) then lm_head. Both cache
    ///       directions walk this order, so an index entry always refers to the
    ///       weight it was written from.
    std::size_t HeadSlot() const { return static_cast<std::size_t>(c->layers) * 5; }
    bool SlotIsMatmul(std::size_t slot) const { return slot == HeadSlot() || slot % 5 != 4; }

    ryzenai_corelib_matmul_bf16_weights_desc MatmulDescAt(std::size_t slot) const {
        if (slot == HeadSlot()) return {c->hidden, c->vocab, c->head_group, false};
        switch (slot % 5) {
            case 0: return {c->hidden, c->q_dim(), c->group, false};
            case 1:
            case 2: return {c->hidden, c->kv_dim(), c->group, false};
            default: return {c->q_dim(), c->hidden, c->group, false};
        }
    }

    void* WeightAt(std::size_t slot) const {
        if (slot == HeadSlot()) return lm_weights.get();
        const auto layer = slot / 5;
        switch (slot % 5) {
            case 0: return q_weights[layer].get();
            case 1: return k_weights[layer].get();
            case 2: return v_weights[layer].get();
            case 3: return o_weights[layer].get();
            default: return mlp_weights[layer].get();
        }
    }

    void AssignWeightAt(std::size_t slot, void* handle) {
        if (slot == HeadSlot()) { lm_weights = UniqueMatMulWeights(api, handle); return; }
        const auto layer = slot / 5;
        switch (slot % 5) {
            case 0: q_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 1: k_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 2: v_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 3: o_weights[layer] = UniqueMatMulWeights(api, handle); break;
            default: mlp_weights[layer] = UniqueSsMlpWeights(api, handle); break;
        }
    }

    void ReleaseCachedWeights() {
        for (std::size_t slot = 0; slot <= HeadSlot(); ++slot) AssignWeightAt(slot, nullptr);
    }

    /// \return true when every slot bound from the file; false leaves none bound
    /// \note corelib rejects a slice that is not exactly what its descriptor
    ///       packs to, so a stale file is caught here, and a half-cached model
    ///       is never kept: the caller simply packs.
    bool LoadWeightsFromCache(const std::filesystem::path& data_path,
                              const std::vector<rai::CachedWeightSpan>& spans) {
        const auto path = data_path.string();
        if (spans.size() != HeadSlot() + 1) return false;
        for (std::size_t slot = 0; slot < spans.size(); ++slot) {
            void* handle = nullptr;
            ryzenai_corelib_status status;
            const auto packed = corelib::PackedFile(path.c_str(), spans[slot].offset, spans[slot].size);
            if (SlotIsMatmul(slot)) {
                const auto desc = MatmulDescAt(slot);
                status = api->functions().matmul_weights_create(&desc, &packed, 1, nullptr, &handle);
            } else {
                const ryzenai_corelib_ssmlp_bf16_weights_desc desc{c->hidden, c->intermediate, c->group};
                status = api->functions().ssmlp_weights_create(&desc, &packed, 1, nullptr, &handle);
            }
            if (status != ryzenai_corelib_status_success || handle == nullptr) {
                ReleaseCachedWeights();
                return false;
            }
            AssignWeightAt(slot, handle);
        }
        return true;
    }

    /// \note Best effort: a cache that cannot be written only means the next
    ///       launch packs again. The index goes last, so a data file without a
    ///       matching index is never used.
    void WriteWeightCache(const rai::WeightCache& cache, const rai::WeightCacheKey& key) {
        try {
            std::error_code error;
            std::filesystem::create_directories(cache.DataPath().parent_path(), error);
            const auto data_path = cache.DataPath();
            const auto temporary = data_path.string() + ".tmp";
            std::vector<rai::CachedWeightSpan> spans;
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                if (!output) return;
                std::vector<char> buffer;
                std::uint64_t offset = 0;
                for (std::size_t slot = 0; slot <= HeadSlot(); ++slot) {
                    std::size_t size = 0;
                    if (api->functions().weights_copy_data(WeightAt(slot), nullptr, 0, &size) !=
                            ryzenai_corelib_status_success || size == 0)
                        return;
                    buffer.resize(size);
                    if (api->functions().weights_copy_data(WeightAt(slot), buffer.data(),
                                                           buffer.size(), &size) !=
                        ryzenai_corelib_status_success)
                        return;
                    output.write(buffer.data(), static_cast<std::streamsize>(size));
                    if (!output) return;
                    spans.push_back({offset, static_cast<std::uint64_t>(size)});
                    offset += size;
                }
            }
            std::filesystem::rename(temporary, data_path, error);
            if (error) { std::filesystem::remove(temporary, error); return; }
            if (!cache.WriteIndex(key, spans)) std::filesystem::remove(data_path, error);
        } catch (...) {
            // Caching is an optimisation; never let it fail a load.
        }
    }

    void usable() const {
        if (poisoned) throw std::runtime_error("Qwen3 corelib engine is poisoned");
    }

    /// \brief one pass of `ids` at the current position
    buffer<bf16> Run(std::span<const int> ids) {
        usable();
        const auto live = ids.size();
        if (live == 0) throw std::invalid_argument("Qwen3 request contains no token IDs");
        if (live > 1 && position != 0)
            throw std::logic_error("Qwen3 multi-row pass must start at position zero");
        if (position + live > max_length || position + live > kMaxSequenceLength)
            throw std::out_of_range("Qwen3 request exceeds configured context capacity");
        if (live == 1 && position + 1 > kMaxDecodeWindow)
            throw std::out_of_range("Qwen3 decode window stops at position 4095");

        const auto rows = plan.RowsFor(live);
        const auto& layer0 = package->Layer(0);
        const auto decoded = rai::DecodeQ8Rows(package->Embedding(), ids);
        std::vector<float> normalized(decoded.size());
        rai::RmsNorm(decoded, layer0.attn_norm.values, static_cast<std::int64_t>(live), c->hidden,
                     c->epsilon, normalized);
        std::vector<float> input(static_cast<std::size_t>(rows * c->hidden), 0.0f);
        std::vector<float> residual_input(input.size(), 0.0f);
        std::copy(normalized.begin(), normalized.end(), input.begin());
        std::copy(decoded.begin(), decoded.end(), residual_input.begin());

        auto lease = runtime->AcquireExecution();
        bool submitted = false;
        try {
            const auto& f = api->functions();
            api->Check(f.tensor_write(hidden.get(), ryzenai_corelib_data_type_fp32, input.data(),
                                      input.size(), 0),
                       "ryzenai_corelib_tensor_write hidden");
            api->Check(f.tensor_write(residual.get(), ryzenai_corelib_data_type_fp32,
                                      residual_input.data(), residual_input.size(), 0),
                       "ryzenai_corelib_tensor_write residual embedding");

            // Each op reads M off its operand's shape, so every buffer is seen
            // through a window at this pass's row count. Built once per pass.
            const auto view = [&](void* parent, std::int64_t r, std::int64_t cols, const char* label) {
                const std::array<std::int64_t, 2> shape{r, cols};
                void* p = nullptr;
                api->Check(f.create_tensor_window(parent, shape.data(), shape.size(), 0, nullptr, 0, &p),
                           std::string("ryzenai_corelib_create_tensor_window ") + label);
                return UniqueTensorWindow(api, p);
            };
            const auto hidden_w = view(hidden.get(), rows, c->hidden, "hidden");
            const auto q_w = view(q.get(), rows, c->q_dim(), "query");
            const auto k_w = view(k.get(), rows, c->kv_dim(), "key");
            const auto q_attn = view(q_normed.get(), rows, c->q_dim(), "normed query");
            const auto k_attn = view(k_normed.get(), rows, c->kv_dim(), "normed key");
            const auto attention_w = view(attention.get(), rows, c->q_dim(), "attention");
            // The per-head view QK-Norm runs over: [rows, heads * 128] read as
            // [rows * heads, 128] is the same row-major memory.
            const auto q_heads_in = view(q.get(), rows * c->q_heads, c->head_dim, "query heads");
            const auto k_heads_in = view(k.get(), rows * c->kv_heads, c->head_dim, "key heads");
            const auto q_heads_out = view(q_normed.get(), rows * c->q_heads, c->head_dim, "normed query heads");
            const auto k_heads_out = view(k_normed.get(), rows * c->kv_heads, c->head_dim, "normed key heads");
            const auto residual_w = view(residual.get(), rows, c->hidden, "residual");
            const auto skip_w = view(skip.get(), rows, c->hidden, "skip");
            void* res = residual_w.get();
            void* sk = skip_w.get();

            for (std::size_t i = 0; i < static_cast<std::size_t>(c->layers); ++i) {
                const auto layer = std::to_string(i);
                const auto query_status = f.matmul(stream.get(), hidden_w.get(), q_weights[i].get(), q_w.get());
                submitted = submitted || query_status == ryzenai_corelib_status_success ||
                            query_status == ryzenai_corelib_status_failure;
                api->Check(query_status, "ryzenai_corelib_matmul_bf16 query layer " + layer);
                api->Check(f.matmul(stream.get(), hidden_w.get(), k_weights[i].get(), k_w.get()),
                           "ryzenai_corelib_matmul_bf16 key layer " + layer);
                // v_proj writes the V cache directly: a window positioned at
                // this pass's first row, whose height bounds what lands there.
                const std::array<std::int64_t, 3> v_shape{c->kv_heads, kMaxSequenceLength - position,
                                                          c->head_dim};
                const std::array<std::int64_t, 3> v_strides{kMaxSequenceLength * c->head_dim,
                                                            c->head_dim, 1};
                void* p = nullptr;
                api->Check(f.create_tensor_window(v_cache[i].get(), v_shape.data(), v_shape.size(),
                                                  static_cast<std::size_t>(position * c->head_dim),
                                                  v_strides.data(), v_strides.size(), &p),
                           "ryzenai_corelib_create_tensor_window V");
                const UniqueTensorWindow v_window(api, p);
                api->Check(f.matmul(stream.get(), hidden_w.get(), v_weights[i].get(), v_window.get()),
                           "ryzenai_corelib_matmul_bf16 value layer " + layer);
                // QK-Norm before attention: flat_mha applies the rotary itself.
                api->Check(f.rmsnorm(stream.get(), q_heads_in.get(), q_norm_weights[i].get(), q_heads_out.get()),
                           "ryzenai_corelib_rmsnorm_bf16 q-norm layer " + layer);
                api->Check(f.rmsnorm(stream.get(), k_heads_in.get(), k_norm_weights[i].get(), k_heads_out.get()),
                           "ryzenai_corelib_rmsnorm_bf16 k-norm layer " + layer);
                api->Check(f.flat_mha(stream.get(), &plan.attention_desc(), q_attn.get(), k_attn.get(),
                                      position, cosine.get(), sine.get(), k_cache[i].get(),
                                      v_cache[i].get(), attention_w.get()),
                           "ryzenai_corelib_flat_mha_bf16 layer " + layer);
                api->Check(f.matmul(stream.get(), attention_w.get(), o_weights[i].get(), hidden_w.get()),
                           "ryzenai_corelib_matmul_bf16 output layer " + layer);
                api->Check(f.ssmlp(stream.get(), hidden_w.get(), res, mlp_weights[i].get(), sk, hidden_w.get()),
                           "ryzenai_corelib_ssmlp_bf16 layer " + layer);
                std::swap(res, sk);
            }
            api->Check(f.stream_synchronize(stream.get()), "ryzenai_corelib_stream_synchronize hidden");
            std::vector<std::uint16_t> row(static_cast<std::size_t>(c->hidden));
            api->Check(f.tensor_read(hidden.get(), ryzenai_corelib_data_type_bf16, row.data(), row.size(),
                                     (live - 1) * static_cast<std::size_t>(c->hidden)),
                       "ryzenai_corelib_tensor_read final hidden row");
            api->Check(f.tensor_write(lm_input.get(), ryzenai_corelib_data_type_bf16, row.data(), row.size(), 0),
                       "ryzenai_corelib_tensor_write LM head input");
            api->Check(f.matmul(stream.get(), lm_input.get(), lm_weights.get(), logits.get()),
                       "ryzenai_corelib_matmul_bf16 LM head");
            api->Check(f.stream_synchronize(stream.get()), "ryzenai_corelib_stream_synchronize logits");
            buffer<bf16> out(static_cast<std::size_t>(c->vocab));
            api->Check(f.tensor_read(logits.get(), ryzenai_corelib_data_type_bf16, out.data(), out.size(), 0),
                       "ryzenai_corelib_tensor_read logits");
            position += static_cast<int>(live);
            return out;
        } catch (...) {
            if (submitted) {
                (void)api->functions().stream_synchronize(stream.get());
                poisoned = true;
                position = 0;
                saved.reset();
                history.clear();
            }
            throw;
        }
    }

    buffer<bf16> Prefill(const std::vector<int>& ids) {
        usable();
        if (ids.empty()) throw std::invalid_argument("Qwen3 request contains no token IDs");
        std::vector<int> tokens;
        if (position != 0) {
            if (history.size() != static_cast<std::size_t>(position))
                throw std::runtime_error("Qwen3 cannot continue from position " + std::to_string(position) +
                                         " without the tokens behind it");
            tokens.reserve(history.size() + ids.size());
            tokens = history;
        }
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        const int resume = position;
        position = 0;
        try {
            auto out = Run(tokens);
            history = std::move(tokens);
            return out;
        } catch (...) {
            if (!poisoned) position = resume;
            throw;
        }
    }

    buffer<bf16> Forward(int id) {
        auto out = Run(std::span<const int>(&id, 1));
        history.push_back(id);
        return out;
    }

    void Rewind(int to) {
        position = to;
        if (history.size() > static_cast<std::size_t>(to)) history.resize(static_cast<std::size_t>(to));
    }

    buffer<bf16> ReadCache(bool is_k, int layer, int index) {
        usable();
        if (layer < 0 || layer >= c->layers || index < 0 || index >= kMaxSequenceLength)
            throw std::out_of_range("Qwen3 cache index is out of range");
        auto lease = runtime->AcquireExecution();
        api->Check(api->functions().stream_synchronize(stream.get()),
                   "ryzenai_corelib_stream_synchronize cache read");
        const auto heads = static_cast<std::size_t>(c->kv_heads);
        const auto width = static_cast<std::size_t>(c->head_dim);
        buffer<bf16> out(heads * width);
        void* cache = is_k ? k_cache[layer].get() : v_cache[layer].get();
        for (std::size_t head = 0; head < heads; ++head) {
            const auto offset = (head * kMaxSequenceLength + static_cast<std::size_t>(index)) * width;
            api->Check(api->functions().tensor_read(cache, ryzenai_corelib_data_type_bf16,
                                                    out.data() + head * width, width, offset),
                       "ryzenai_corelib_tensor_read cache head " + std::to_string(head));
        }
        return out;
    }
};

qwen3_rai::qwen3_rai(LM_Config, std::shared_ptr<Qwen3GgufPackage> package,
                     std::shared_ptr<CorelibRuntime> runtime, std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(std::move(package), std::move(runtime), max_length)) {}
qwen3_rai::~qwen3_rai() = default;

buffer<bf16> qwen3_rai::forward(int id) { return impl_->Forward(id); }
buffer<bf16> qwen3_rai::prefill(std::vector<int>& ids, void*) { return impl_->Prefill(ids); }

void qwen3_rai::set_context_length(int n) {
    impl_->usable();
    if (n < 0 || static_cast<std::uint32_t>(n) > impl_->max_length)
        throw std::out_of_range("Qwen3 context length is out of range");
    impl_->Rewind(n);
}

// An ABI shim, not a capability. load_weights is pure virtual in causal_lm.hpp,
// which is frozen because the engine libraries in src/lib/<runtime> are prebuilt
// against it. Nothing calls this: this engine's weights come from the GGUF
// package it was constructed with. See AutoModel/model_backend.hpp.
void qwen3_rai::load_weights(Q4NX&) {
    impl_->usable();
    throw std::runtime_error("Qwen3 rai weights are loaded only from GGUF");
}

void qwen3_rai::update_max_length(std::uint32_t n) {
    impl_->usable();
    if (!n || n > kMaxSequenceLength || n < static_cast<std::uint32_t>(impl_->position))
        throw std::out_of_range("Qwen3 maximum length is invalid");
    impl_->max_length = n;
}

void qwen3_rai::clear_context() {
    impl_->usable();
    impl_->position = 0;
    impl_->saved.reset();
    impl_->history.clear();
}

buffer<bf16> qwen3_rai::get_k_cache(int l, int i) { return impl_->ReadCache(true, l, i); }
buffer<bf16> qwen3_rai::get_v_cache(int l, int i) { return impl_->ReadCache(false, l, i); }

int qwen3_rai::get_current_context_length() {
    impl_->usable();
    return impl_->position;
}

int qwen3_rai::checkpoint() {
    impl_->usable();
    impl_->saved = impl_->position;
    return impl_->position;
}

int qwen3_rai::restore() {
    impl_->usable();
    if (!impl_->saved) return -1;
    impl_->Rewind(*impl_->saved);
    return impl_->position;
}

bool qwen3_rai::poisoned() const noexcept { return impl_ && impl_->poisoned; }

}  // namespace flm::qwen3
