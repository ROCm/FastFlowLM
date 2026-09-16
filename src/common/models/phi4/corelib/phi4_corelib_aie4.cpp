#include "models/phi4/corelib/phi4_corelib_aie4.hpp"
#include "corelib/corelib_object.hpp"
#include "models/phi4/corelib/phi4_corelib_constants.hpp"
#include "models/phi4/corelib/phi4_corelib_host.hpp"
#include "models/phi4/corelib/phi4_corelib_shape_plan.hpp"
#include "models/phi4/corelib/phi4_corelib_weight_cache.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::phi4 {
namespace {
using namespace flm::corelib;
std::string Name(std::size_t i, const char* suffix) {
    return "blk." + std::to_string(i) + suffix;
}

/// Load-time phase accounting. Model load on this backend is dominated by
/// requantizing every weight from Q8_0, and without a breakdown there is no way
/// to tell that from disk I/O or from shape planning. Set FLM_AIE4_PROFILE_LOAD
/// to print it; the timer itself always runs, it costs five clock reads.
struct LoadPhases {
    std::chrono::steady_clock::time_point mark{std::chrono::steady_clock::now()};
    double shape_plan{}, tensor_resolve{}, host_prep{}, weight_create{}, device_tensors{};
    double cache_write{};
    std::uint64_t cache_reclaimed{};
    bool from_cache{};

    double Lap() {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - mark).count();
        mark = now;
        return seconds;
    }

    void Report() const {
        const char* enabled = std::getenv("FLM_AIE4_PROFILE_LOAD");
        if (!enabled || !*enabled || *enabled == '0') return;
        const double total = shape_plan + tensor_resolve + host_prep +
                             weight_create + device_tensors + cache_write;
        std::ostringstream out;
        out << std::fixed << std::setprecision(2)
            << "[FLM]  AIE4 load: " << total << " s total"
            << "  (shape plan " << shape_plan
            << ", GGUF resolve " << tensor_resolve
            << ", host prep " << host_prep
            << ", weight requantize " << weight_create
            << ", device tensors " << device_tensors << ")"
            << (from_cache ? "  [weights from cache]"
                           : (cache_write > 0.0 ? "  [cache written in " +
                                 [&]{ std::ostringstream w; w << std::fixed
                                      << std::setprecision(2) << cache_write; return w.str(); }() + " s]"
                                               : std::string()))
            << (cache_reclaimed > 0
                    ? "  [reclaimed " + std::to_string(cache_reclaimed / (1024 * 1024)) +
                          " MB of stale cache]"
                    : std::string());
        std::cout << out.str() << std::endl;
    }
};
}

struct phi4_corelib_aie4::Impl {
    std::shared_ptr<Phi4GgufPackage> package;
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    // Declared before `plan` so it starts before the initializer list builds it.
    LoadPhases phases;
    Phi4ShapePlan plan;
    std::uint32_t max_length;
    int position{};
    std::optional<int> saved;
    bool poisoned{};
    UniqueStream stream;
    std::array<UniqueMatMulWeights, kLayerCount> q_weights, k_weights, v_weights, o_weights;
    std::array<UniqueSsMlpWeights, kLayerCount> mlp_weights;
    UniqueMatMulWeights lm_weights;
    UniqueTensor hidden, residual, skip, q, k, attention, lm_input, logits, cosine, sine;
    std::array<UniqueTensor, kLayerCount> k_cache, v_cache;
    TensorView embedding;
    FloatTensorView first_norm_scale;

    Impl(LM_Config, std::shared_ptr<Phi4GgufPackage> pkg,
         std::shared_ptr<CorelibRuntime> rt, std::uint32_t maximum)
        : package(std::move(pkg)), runtime(std::move(rt)),
          api(runtime ? runtime->api() : nullptr), plan(Phi4ShapePlan::Build(api)),
          max_length(maximum) {
        if (!package) throw std::invalid_argument("Phi-4 GGUF package is null");
        if (!runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > kMaxSequenceLength)
            throw std::invalid_argument("Phi-4 maximum length must be in 1..4096");
        phases.shape_plan = phases.Lap();

        // Validate and capture every mapped span before the first device create.
        embedding = package->RequireQ8("token_embd.weight", std::array<std::int64_t,2>{kVocabularySize,kHiddenSize});
        auto final_norm = package->RequireF32("output_norm.weight", std::array<std::int64_t,1>{kHiddenSize});
        std::array<FloatTensorView,kLayerCount> an, fn;
        std::array<ProjectionViews,kLayerCount> qkv, gu;
        std::array<TensorView,kLayerCount> ow, dw;
        for (std::size_t i=0;i<kLayerCount;++i) {
            an[i]=package->RequireF32(Name(i,".attn_norm.weight"),std::array<std::int64_t,1>{kHiddenSize});
            fn[i]=package->RequireF32(Name(i,".ffn_norm.weight"),std::array<std::int64_t,1>{kHiddenSize});
            qkv[i]=package->AttentionQkv(i); gu[i]=package->GateUp(i);
            ow[i]=package->RequireQ8(Name(i,".attn_output.weight"),std::array<std::int64_t,2>{kHiddenSize,kHiddenSize});
            dw[i]=package->RequireQ8(Name(i,".ffn_down.weight"),std::array<std::int64_t,2>{kHiddenSize,kIntermediateSize});
        }
        phases.tensor_resolve = phases.Lap();
        std::optional<FloatTensorView> factors;
        try { factors=package->RequireF32("rope_factors_short.weight",std::array<std::int64_t,1>{48}); }
        catch (const std::runtime_error&) {}
        auto rope=BuildShortRopeTables(package->Metadata(),factors);
        auto final_bf=ConvertF32ToBf16(final_norm.values);
        std::array<std::vector<std::uint16_t>,kLayerCount> an_bf,fn_bf;
        for(std::size_t i=0;i<kLayerCount;++i){an_bf[i]=ConvertF32ToBf16(an[i].values);fn_bf[i]=ConvertF32ToBf16(fn[i].values);}
        const std::array<float,1> epsf{kRmsEpsilon}; auto eps=ConvertF32ToBf16(epsf);

        first_norm_scale = an[0];
        phases.host_prep = phases.Lap();
        auto lease=runtime->AcquireExecution(); void* raw=nullptr;
        api->Check(api->functions().create_stream(&raw),"ryzenai_corelib_create_stream"); stream=UniqueStream(api,raw);
        auto mm=[&](const TensorView& tv,std::int64_t kk,std::int64_t nn,const std::string& label){
            ryzenai_corelib_matmul_bf16_weights_desc d{kk,nn,kRequantizedGroupSize,false};
            ryzenai_corelib_matmul_bf16_gguf_components c{tv.bytes.data(),ryzenai_corelib_gguf_quant_type_q8_0}; void* p=nullptr;
            api->Check(api->functions().matmul_weights_create_gguf_requantized(&d,&c,kRequantizeThreads,&p),"ryzenai_corelib_matmul_bf16_weights_create_gguf_requantized "+label);
            return UniqueMatMulWeights(api,p);
        };
        // Each create reads its own mapped range and writes its own array slot,
        // so they are independent. Collect them first, then run the list across
        // a small pool -- packing is the whole of model load and it is the one
        // phase with nothing to serialize.
        std::vector<std::function<void()>> creates;
        creates.reserve(kLayerCount * 5 + 1);
        for(std::size_t i=0;i<kLayerCount;++i){
            creates.push_back([&,i]{ q_weights[i]=mm(qkv[i].values[0],kHiddenSize,kQueryDimension,Name(i,".q")); });
            creates.push_back([&,i]{ k_weights[i]=mm(qkv[i].values[1],kHiddenSize,kKvDimension,Name(i,".k")); });
            creates.push_back([&,i]{ v_weights[i]=mm(qkv[i].values[2],kHiddenSize,kKvDimension,Name(i,".v")); });
            creates.push_back([&,i]{ o_weights[i]=mm(ow[i],kHiddenSize,kHiddenSize,Name(i,".attn_output.weight")); });
            creates.push_back([&,i]{
                const auto& next=i+1<kLayerCount?an_bf[i+1]:final_bf;
                ryzenai_corelib_ssmlp_bf16_weights_desc d{kHiddenSize,kIntermediateSize,kRequantizedGroupSize};
                ryzenai_corelib_ssmlp_bf16_gguf_components c{eps.data(),fn_bf[i].data(),next.data(),gu[i].values[0].bytes.data(),gu[i].values[1].bytes.data(),dw[i].bytes.data(),ryzenai_corelib_gguf_quant_type_q8_0};
                void* p=nullptr;
                api->Check(api->functions().ssmlp_weights_create_gguf_requantized(&d,&c,kRequantizeThreads,&p),"ryzenai_corelib_ssmlp_bf16_weights_create_gguf_requantized layer "+std::to_string(i));
                mlp_weights[i]=UniqueSsMlpWeights(api,p);
            });
        }
        creates.push_back([&]{ lm_weights=mm(embedding,kHiddenSize,kVocabularySize,"token_embd.weight"); });

        std::atomic<std::size_t> next_create{0};
        std::mutex failure_mutex;
        std::exception_ptr first_failure;
        const auto run_creates=[&]{
            for(std::size_t k=next_create++;k<creates.size();k=next_create++){
                try { creates[k](); }
                catch(...){
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if(!first_failure) first_failure=std::current_exception();
                    // Stop the others: the first diagnostic is the useful one,
                    // and a package that fails one create fails the load.
                    next_create=creates.size();
                    return;
                }
            }
        };
        // A cache hit replaces every create above with a mapped slice, which is
        // the whole point: the refit is paid once per GGUF rather than once per
        // launch. A miss, a stale key or any failure just packs as normal.
        const auto cache_directory = WeightCacheDirectory(package->Path().parent_path());
        const auto cache_key = cache_directory
            ? std::optional<WeightCacheKey>(MakeWeightCacheKey(
                  package->Path(), api->runtime_version().major,
                  api->runtime_version().minor, api->runtime_version().patch,
                  kRequantizedGroupSize, creates.size()))
            : std::nullopt;
        bool loaded_from_cache = false;
        if (cache_directory && cache_key) {
            if (const auto index = ReadWeightCacheIndex(*cache_directory, *cache_key)) {
                loaded_from_cache = LoadWeightsFromCache(
                    WeightCacheDataPath(*cache_directory), index->spans);
            }
        }

        if (!loaded_from_cache) {
            const std::size_t workers=std::min(kWeightCreateConcurrency,creates.size());
            std::vector<std::thread> pool;
            pool.reserve(workers>0?workers-1:0);
            for(std::size_t t=1;t<workers;++t) pool.emplace_back(run_creates);
            run_creates();
            for(auto& worker:pool) worker.join();
            if(first_failure) std::rethrow_exception(first_failure);
        }
        phases.weight_create = phases.Lap();
        phases.from_cache = loaded_from_cache;
        if (cache_directory && cache_key && !loaded_from_cache) {
            // Whatever is there did not match, or it would have been used.
            // Reclaim it now rather than leaving two gigabytes of dead cache
            // around until some later write happens to succeed.
            phases.cache_reclaimed = RemoveWeightCache(*cache_directory);
            WriteWeightCache(*cache_directory, *cache_key);
            phases.cache_write = phases.Lap();
        }
        const auto& e=plan.maximum_extents();
        const auto rows=std::max({e.query_rows,e.kv_rows,e.output_rows,
                                  e.ssmlp_rows});
        const auto query_rows=std::max(e.query_rows,e.flat_mha_rows);
        const auto key_rows=std::max(e.kv_rows,e.flat_mha_rows);
        const auto attention_rows=std::max(e.flat_mha_rows,e.output_rows);
        auto tensor=[&](ryzenai_corelib_data_type type,std::initializer_list<std::int64_t> dims,const char* label){
            std::vector<std::int64_t> shape(dims);void* p=nullptr;
            api->Check(api->functions().create_device_tensor(type,shape.data(),shape.size(),&p),std::string("ryzenai_corelib_create_device_tensor ")+label);
            return UniqueTensor(api,p);
        };
        hidden=tensor(ryzenai_corelib_data_type_bf16,{rows,kHiddenSize},"hidden");
        residual=tensor(ryzenai_corelib_data_type_bf16,{rows,kHiddenSize},"residual");
        skip=tensor(ryzenai_corelib_data_type_bf16,{rows,kHiddenSize},"skip");
        q=tensor(ryzenai_corelib_data_type_bf16,{query_rows,kQueryDimension},"query");
        k=tensor(ryzenai_corelib_data_type_bf16,{key_rows,kKvDimension},"key");
        attention=tensor(ryzenai_corelib_data_type_bf16,{attention_rows,kQueryDimension},"attention");
        lm_input=tensor(ryzenai_corelib_data_type_bf16,{1,kHiddenSize},"lm input");
        logits=tensor(ryzenai_corelib_data_type_bf16,{1,kVocabularySize},"logits");
        cosine=tensor(ryzenai_corelib_data_type_fp32,{kMaxSequenceLength,48},"cosine");
        sine=tensor(ryzenai_corelib_data_type_fp32,{kMaxSequenceLength,48},"sine");
        for(std::size_t i=0;i<kLayerCount;++i){k_cache[i]=tensor(ryzenai_corelib_data_type_bf16,{8,4096,128},"K cache");v_cache[i]=tensor(ryzenai_corelib_data_type_bf16,{8,4096,128},"V cache");}
        api->Check(api->functions().tensor_write(cosine.get(),ryzenai_corelib_data_type_fp32,rope.cosine.data(),rope.cosine.size(),0),"ryzenai_corelib_tensor_write cosine");
        api->Check(api->functions().tensor_write(sine.get(),ryzenai_corelib_data_type_fp32,rope.sine.data(),rope.sine.size(),0),"ryzenai_corelib_tensor_write sine");
        phases.device_tensors = phases.Lap();
        phases.Report();
    }

    /// \brief the weight slot order the cache is written and read in
    /// \param slot 0..160: five per layer, then the LM head
    /// \note Both directions walk this same order, so an index entry always
    ///       refers to the weight it was written from. The creates complete out
    ///       of order, but they are *assigned* by this index, not by completion.
    static bool SlotIsMatmul(std::size_t slot) { return slot % 5 != 4 || slot == kLayerCount * 5; }

    ryzenai_corelib_matmul_bf16_weights_desc MatmulDescAt(std::size_t slot) const {
        if (slot == kLayerCount * 5)
            return {kHiddenSize, kVocabularySize, kRequantizedGroupSize, false};
        switch (slot % 5) {
            case 0: return {kHiddenSize, kQueryDimension, kRequantizedGroupSize, false};
            case 1:
            case 2: return {kHiddenSize, kKvDimension, kRequantizedGroupSize, false};
            default: return {kHiddenSize, kHiddenSize, kRequantizedGroupSize, false};
        }
    }

    void* WeightHandleAt(std::size_t slot) const {
        if (slot == kLayerCount * 5) return lm_weights.get();
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
        if (slot == kLayerCount * 5) { lm_weights = UniqueMatMulWeights(api, handle); return; }
        const auto layer = slot / 5;
        switch (slot % 5) {
            case 0: q_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 1: k_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 2: v_weights[layer] = UniqueMatMulWeights(api, handle); break;
            case 3: o_weights[layer] = UniqueMatMulWeights(api, handle); break;
            default: mlp_weights[layer] = UniqueSsMlpWeights(api, handle); break;
        }
    }

    /// \brief bind every weight from the cache file instead of packing
    /// \return true when all of them loaded; false leaves nothing bound
    /// \note Any failure abandons the whole attempt rather than packing the
    ///       remainder: a half-cached model is not a state worth having, and
    ///       the caller simply packs. corelib rejects a slice that is not
    ///       exactly what the descriptor packs to, so a stale file is caught
    ///       here rather than becoming confident nonsense.
    bool LoadWeightsFromCache(const std::filesystem::path& data_path,
                              const std::vector<CachedWeightSpan>& spans) {
        const auto path = data_path.string();
        const std::size_t total = kLayerCount * 5 + 1;
        if (spans.size() != total) return false;
        for (std::size_t slot = 0; slot < total; ++slot) {
            void* handle = nullptr;
            ryzenai_corelib_status status;
            if (SlotIsMatmul(slot)) {
                const auto desc = MatmulDescAt(slot);
                status = api->functions().matmul_weights_create_from_file(
                    &desc, path.c_str(), spans[slot].offset, spans[slot].size, &handle);
            } else {
                const ryzenai_corelib_ssmlp_bf16_weights_desc desc{
                    kHiddenSize, kIntermediateSize, kRequantizedGroupSize};
                status = api->functions().ssmlp_weights_create_from_file(
                    &desc, path.c_str(), spans[slot].offset, spans[slot].size, &handle);
            }
            if (status != ryzenai_corelib_status_success || handle == nullptr) {
                ReleaseAllWeights();
                return false;
            }
            AssignWeightAt(slot, handle);
        }
        return true;
    }

    void ReleaseAllWeights() {
        for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
            q_weights[layer] = {}; k_weights[layer] = {};
            v_weights[layer] = {}; o_weights[layer] = {};
            mlp_weights[layer] = {};
        }
        lm_weights = {};
    }

    /// \brief copy the packed bytes out and write them beside the model
    /// \note Best effort: a cache that cannot be written is not a load failure,
    ///       it only means the next launch packs again. The index is written
    ///       last, so a data file without a matching index is never used.
    void WriteWeightCache(const std::filesystem::path& directory,
                          const WeightCacheKey& key) {
        try {
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            const auto data_path = WeightCacheDataPath(directory);
            const auto temporary = data_path.string() + ".tmp";
            std::vector<CachedWeightSpan> spans;
            spans.reserve(kLayerCount * 5 + 1);
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                if (!output) return;
                std::vector<char> buffer;
                std::uint64_t offset = 0;
                for (std::size_t slot = 0; slot < kLayerCount * 5 + 1; ++slot) {
                    std::size_t size = 0;
                    if (api->functions().weights_copy_data(
                            WeightHandleAt(slot), nullptr, 0, &size) !=
                        ryzenai_corelib_status_success || size == 0) return;
                    buffer.resize(size);
                    if (api->functions().weights_copy_data(
                            WeightHandleAt(slot), buffer.data(), buffer.size(), &size) !=
                        ryzenai_corelib_status_success) return;
                    output.write(buffer.data(), static_cast<std::streamsize>(size));
                    if (!output) return;
                    spans.push_back({offset, static_cast<std::uint64_t>(size)});
                    offset += size;
                }
            }
            std::filesystem::rename(temporary, data_path, error);
            if (error) { std::filesystem::remove(temporary, error); return; }
            if (!WriteWeightCacheIndex(directory, key, spans)) {
                // No index means the data file would never be used; drop it
                // rather than leave the space occupied.
                std::filesystem::remove(data_path, error);
            }
        } catch (...) {
            // Caching is an optimisation; never let it fail a load.
        }
    }

    void usable() const {if(poisoned)throw std::runtime_error("Phi-4 corelib engine is poisoned");}
    buffer<bf16> run(std::span<const int> ids,bool prefill){
        usable(); if(ids.empty())throw std::invalid_argument("Phi-4 request contains no token IDs");
        if(prefill&&position)throw std::runtime_error("Phi-4 prefill must start at logical position zero");
        if(ids.size()>max_length||position+ids.size()>max_length||position+ids.size()>kMaxSequenceLength)throw std::out_of_range("Phi-4 request exceeds configured context capacity");
        if(!prefill&&position+ids.size()>kMaxDecodeWindow)throw std::out_of_range("Phi-4 decode window stops at position 4095");
        auto decoded=DecodeEmbeddingRowsQ8(embedding,ids);const auto&e=plan.ForRows(ids.size());
        auto rows=std::max({e.query_rows,e.kv_rows,e.output_rows,e.ssmlp_rows});
        std::vector<float> normalized(decoded.size());
        HostRmsNorm(decoded,first_norm_scale.values,ids.size(),kHiddenSize,
                    kRmsEpsilon,normalized);
        std::vector<float> input(static_cast<std::size_t>(rows*kHiddenSize),0);
        std::vector<float> residual_input(static_cast<std::size_t>(rows*kHiddenSize),0);
        std::copy(normalized.begin(),normalized.end(),input.begin());
        std::copy(decoded.begin(),decoded.end(),residual_input.begin());

        auto lease=runtime->AcquireExecution();bool submitted=false;
        try{
            api->Check(api->functions().tensor_write(hidden.get(),ryzenai_corelib_data_type_fp32,input.data(),input.size(),0),"ryzenai_corelib_tensor_write hidden");
            api->Check(api->functions().tensor_write(residual.get(),ryzenai_corelib_data_type_fp32,residual_input.data(),residual_input.size(),0),"ryzenai_corelib_tensor_write residual embedding");
            void* res=residual.get();void* sk=skip.get();
            for(std::size_t i=0;i<kLayerCount;++i){
                const auto query_status=api->functions().matmul(
                    stream.get(),hidden.get(),ids.size(),q_weights[i].get(),q.get());
                submitted=submitted || query_status==ryzenai_corelib_status_success ||
                          query_status==ryzenai_corelib_status_failure;
                api->Check(query_status,"ryzenai_corelib_matmul_bf16 query layer "+std::to_string(i));
                api->Check(api->functions().matmul(stream.get(),hidden.get(),ids.size(),k_weights[i].get(),k.get()),"ryzenai_corelib_matmul_bf16 key layer "+std::to_string(i));
                std::array<std::int64_t,3> shape{8,kMaxSequenceLength-position,128};void* p=nullptr;
                api->Check(api->functions().create_tensor_window(v_cache[i].get(),shape.data(),shape.size(),static_cast<std::size_t>(position)*128,&p),"ryzenai_corelib_create_tensor_window V");UniqueTensorWindow win(api,p);
                api->Check(api->functions().matmul(stream.get(),hidden.get(),ids.size(),v_weights[i].get(),win.get()),"ryzenai_corelib_matmul_bf16 value layer "+std::to_string(i));
                api->Check(api->functions().flat_mha(stream.get(),&plan.attention_desc(),q.get(),k.get(),ids.size(),position,cosine.get(),sine.get(),k_cache[i].get(),v_cache[i].get(),attention.get()),"ryzenai_corelib_flat_mha_bf16 layer "+std::to_string(i));
                api->Check(api->functions().matmul(stream.get(),attention.get(),ids.size(),o_weights[i].get(),hidden.get()),"ryzenai_corelib_matmul_bf16 output layer "+std::to_string(i));
                api->Check(api->functions().ssmlp(stream.get(),hidden.get(),res,ids.size(),mlp_weights[i].get(),sk,hidden.get()),"ryzenai_corelib_ssmlp_bf16 layer "+std::to_string(i));std::swap(res,sk);
            }
            api->Check(api->functions().stream_synchronize(stream.get()),"ryzenai_corelib_stream_synchronize hidden");
            std::vector<std::uint16_t> row(kHiddenSize);api->Check(api->functions().tensor_read(hidden.get(),ryzenai_corelib_data_type_bf16,row.data(),row.size(),(ids.size()-1)*kHiddenSize),"ryzenai_corelib_tensor_read final hidden row");
            api->Check(api->functions().tensor_write(lm_input.get(),ryzenai_corelib_data_type_bf16,row.data(),row.size(),0),"ryzenai_corelib_tensor_write LM head input");
            api->Check(api->functions().matmul(stream.get(),lm_input.get(),1,lm_weights.get(),logits.get()),"ryzenai_corelib_matmul_bf16 LM head");
            api->Check(api->functions().stream_synchronize(stream.get()),"ryzenai_corelib_stream_synchronize logits");
            buffer<bf16> out(kVocabularySize);api->Check(api->functions().tensor_read(logits.get(),ryzenai_corelib_data_type_bf16,out.data(),out.size(),0),"ryzenai_corelib_tensor_read logits");position+=static_cast<int>(ids.size());return out;
        }catch(...){if(submitted){(void)api->functions().stream_synchronize(stream.get());poisoned=true;position=0;saved.reset();}throw;}
    }
    buffer<bf16> read_cache(bool is_k,int layer,int index){
        usable();
        if(layer<0||layer>=kLayerCount||index<0||index>=kMaxSequenceLength)
            throw std::out_of_range("Phi-4 cache index is out of range");
        auto lease=runtime->AcquireExecution();
        api->Check(api->functions().stream_synchronize(stream.get()),
                   "ryzenai_corelib_stream_synchronize cache read");
        buffer<bf16> out(kKvHeadCount*kHeadSize);
        void* cache=is_k?k_cache[layer].get():v_cache[layer].get();
        for(std::size_t head=0;head<kKvHeadCount;++head){
            const auto offset=(head*kMaxSequenceLength+
                               static_cast<std::size_t>(index))*kHeadSize;
            api->Check(api->functions().tensor_read(
                           cache,ryzenai_corelib_data_type_bf16,
                           out.data()+head*kHeadSize,kHeadSize,offset),
                       "ryzenai_corelib_tensor_read cache head "+
                           std::to_string(head));
        }
        return out;
    }
};

phi4_corelib_aie4::phi4_corelib_aie4(LM_Config c,std::shared_ptr<Phi4GgufPackage> p,std::shared_ptr<CorelibRuntime> r,std::uint32_t m):impl_(std::make_unique<Impl>(std::move(c),std::move(p),std::move(r),m)){}
phi4_corelib_aie4::~phi4_corelib_aie4()=default;
buffer<bf16> phi4_corelib_aie4::forward(int id){return impl_->run(std::span<const int>(&id,1),false);}
buffer<bf16> phi4_corelib_aie4::prefill(std::vector<int>&ids,void*){return impl_->run(ids,true);}
void phi4_corelib_aie4::set_context_length(int n){impl_->usable();if(n<0||static_cast<std::uint32_t>(n)>impl_->max_length)throw std::out_of_range("Phi-4 context length is out of range");impl_->position=n;}
// An ABI shim, not a capability. load_weights is pure virtual in causal_lm.hpp,
// which is frozen because the engine libraries in src/lib/<runtime> are prebuilt
// against it. Nothing calls this: FlmNpuBackend loads weights through the
// concrete engine type, and this engine's weights come from the GGUF package it
// was constructed with. See AutoModel/model_backend.hpp.
void phi4_corelib_aie4::load_weights(Q4NX&){impl_->usable();throw std::runtime_error("Phi-4 AIE4 weights are loaded only from GGUF");}
void phi4_corelib_aie4::update_max_length(std::uint32_t n){impl_->usable();if(!n||n>kMaxSequenceLength||n<static_cast<std::uint32_t>(impl_->position))throw std::out_of_range("Phi-4 maximum length is invalid");impl_->max_length=n;}
void phi4_corelib_aie4::clear_context(){impl_->usable();impl_->position=0;impl_->saved.reset();}
buffer<bf16> phi4_corelib_aie4::get_k_cache(int l,int i){return impl_->read_cache(true,l,i);}
buffer<bf16> phi4_corelib_aie4::get_v_cache(int l,int i){return impl_->read_cache(false,l,i);}
int phi4_corelib_aie4::get_current_context_length(){impl_->usable();return impl_->position;}
int phi4_corelib_aie4::checkpoint(){impl_->usable();impl_->saved=impl_->position;return impl_->position;}
int phi4_corelib_aie4::restore(){impl_->usable();if(!impl_->saved)return -1;return impl_->position=*impl_->saved;}
bool phi4_corelib_aie4::poisoned()const noexcept{return impl_&&impl_->poisoned;}
} // namespace flm::phi4
