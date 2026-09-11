#include "models/phi4/phi4_corelib_aie4.hpp"
#include "corelib/corelib_object.hpp"
#include "models/phi4/phi4_corelib_constants.hpp"
#include "models/phi4/phi4_corelib_host.hpp"
#include "models/phi4/phi4_corelib_shape_plan.hpp"
#include <algorithm>
#include <array>
#include <optional>
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
}

struct phi4_corelib_aie4::Impl {
    std::shared_ptr<Phi4GgufPackage> package;
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    Phi4ShapePlan plan;
    std::uint32_t max_length;
    int position{};
    std::optional<int> saved;
    bool poisoned{};
    UniqueStream stream;
    UniqueRmsNormWeights first_norm;
    std::array<UniqueMatMulWeights, kLayerCount> q_weights, k_weights, v_weights, o_weights;
    std::array<UniqueSsMlpWeights, kLayerCount> mlp_weights;
    UniqueMatMulWeights lm_weights;
    UniqueTensor hidden, residual, skip, q, k, attention, lm_input, logits, cosine, sine;
    std::array<UniqueTensor, kLayerCount> k_cache, v_cache;
    TensorView embedding;

    Impl(LM_Config, std::shared_ptr<Phi4GgufPackage> pkg,
         std::shared_ptr<CorelibRuntime> rt, std::uint32_t maximum)
        : package(std::move(pkg)), runtime(std::move(rt)),
          api(runtime ? runtime->api() : nullptr), plan(Phi4ShapePlan::Build(api)),
          max_length(maximum) {
        if (!package) throw std::invalid_argument("Phi-4 GGUF package is null");
        if (!runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > kMaxSequenceLength)
            throw std::invalid_argument("Phi-4 maximum length must be in 1..4096");

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
        std::optional<FloatTensorView> factors;
        try { factors=package->RequireF32("rope_factors_short.weight",std::array<std::int64_t,1>{48}); }
        catch (const std::runtime_error&) {}
        auto rope=BuildShortRopeTables(package->Metadata(),factors);
        auto final_bf=ConvertF32ToBf16(final_norm.values);
        std::array<std::vector<std::uint16_t>,kLayerCount> an_bf,fn_bf;
        for(std::size_t i=0;i<kLayerCount;++i){an_bf[i]=ConvertF32ToBf16(an[i].values);fn_bf[i]=ConvertF32ToBf16(fn[i].values);}
        const std::array<float,1> epsf{kRmsEpsilon}; auto eps=ConvertF32ToBf16(epsf);

        auto lease=runtime->AcquireExecution(); void* raw=nullptr;
        api->Check(api->functions().create_stream(&raw),"ryzenai_corelib_create_stream"); stream=UniqueStream(api,raw);
        ryzenai_corelib_rmsnorm_bf16_weights_desc rd{kHiddenSize,kRmsEpsilon}; raw=nullptr;
        api->Check(api->functions().rmsnorm_weights_create_scale(&rd,an_bf[0].data(),&raw),"ryzenai_corelib_rmsnorm_bf16_weights_create_scale blk.0.attn_norm.weight");
        first_norm=UniqueRmsNormWeights(api,raw);
        auto mm=[&](const TensorView& tv,std::int64_t kk,std::int64_t nn,const std::string& label){
            ryzenai_corelib_matmul_bf16_weights_desc d{kk,nn,kRequantizedGroupSize,false};
            ryzenai_corelib_matmul_bf16_gguf_components c{tv.bytes.data(),ryzenai_corelib_gguf_quant_type_q8_0}; void* p=nullptr;
            api->Check(api->functions().matmul_weights_create_gguf_requantized(&d,&c,0,&p),"ryzenai_corelib_matmul_bf16_weights_create_gguf_requantized "+label);
            return UniqueMatMulWeights(api,p);
        };
        for(std::size_t i=0;i<kLayerCount;++i){
            q_weights[i]=mm(qkv[i].values[0],kHiddenSize,kQueryDimension,Name(i,".q"));
            k_weights[i]=mm(qkv[i].values[1],kHiddenSize,kKvDimension,Name(i,".k"));
            v_weights[i]=mm(qkv[i].values[2],kHiddenSize,kKvDimension,Name(i,".v"));
            o_weights[i]=mm(ow[i],kHiddenSize,kHiddenSize,Name(i,".attn_output.weight"));
            const auto& next=i+1<kLayerCount?an_bf[i+1]:final_bf;
            ryzenai_corelib_ssmlp_bf16_weights_desc d{kHiddenSize,kIntermediateSize,kRequantizedGroupSize};
            ryzenai_corelib_ssmlp_bf16_gguf_components c{eps.data(),fn_bf[i].data(),next.data(),gu[i].values[0].bytes.data(),gu[i].values[1].bytes.data(),dw[i].bytes.data(),ryzenai_corelib_gguf_quant_type_q8_0}; raw=nullptr;
            api->Check(api->functions().ssmlp_weights_create_gguf_requantized(&d,&c,0,&raw),"ryzenai_corelib_ssmlp_bf16_weights_create_gguf_requantized layer "+std::to_string(i));
            mlp_weights[i]=UniqueSsMlpWeights(api,raw);
        }
        lm_weights=mm(embedding,kHiddenSize,kVocabularySize,"token_embd.weight");
        const auto& e=plan.maximum_extents();
        const auto rows=std::max({e.query_rows,e.kv_rows,e.output_rows,
                                  e.ssmlp_rows,e.rmsnorm_rows});
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
    }

    void usable() const {if(poisoned)throw std::runtime_error("Phi-4 corelib engine is poisoned");}
    buffer<bf16> run(std::span<const int> ids,bool prefill){
        usable(); if(ids.empty())throw std::invalid_argument("Phi-4 request contains no token IDs");
        if(prefill&&position)throw std::runtime_error("Phi-4 prefill must start at logical position zero");
        if(ids.size()>max_length||position+ids.size()>max_length||position+ids.size()>kMaxSequenceLength)throw std::out_of_range("Phi-4 request exceeds configured context capacity");
        if(!prefill&&position+ids.size()>kMaxDecodeWindow)throw std::out_of_range("Phi-4 decode window stops at position 4095");
        auto decoded=DecodeEmbeddingRowsQ8(embedding,ids);const auto&e=plan.ForRows(ids.size());
        auto rows=std::max({e.query_rows,e.kv_rows,e.output_rows,
                            e.ssmlp_rows,e.rmsnorm_rows});
        std::vector<float> input(static_cast<std::size_t>(rows*kHiddenSize),0);std::copy(decoded.begin(),decoded.end(),input.begin());
        std::vector<std::uint16_t> zeros(static_cast<std::size_t>(rows*kHiddenSize),0);
        auto lease=runtime->AcquireExecution();bool submitted=false;
        try{
            api->Check(api->functions().tensor_write(hidden.get(),ryzenai_corelib_data_type_fp32,input.data(),input.size(),0),"ryzenai_corelib_tensor_write hidden");
            api->Check(api->functions().tensor_write(residual.get(),ryzenai_corelib_data_type_bf16,zeros.data(),zeros.size(),0),"ryzenai_corelib_tensor_write residual padding");
            const auto rms_status=api->functions().rmsnorm(
                stream.get(),hidden.get(),ids.size(),first_norm.get(),hidden.get());
            submitted=rms_status==ryzenai_corelib_status_success ||
                      rms_status==ryzenai_corelib_status_failure;
            api->Check(rms_status,"ryzenai_corelib_rmsnorm_bf16 initial");
            void* res=residual.get();void* sk=skip.get();
            for(std::size_t i=0;i<kLayerCount;++i){
                api->Check(api->functions().matmul(stream.get(),hidden.get(),ids.size(),q_weights[i].get(),q.get()),"ryzenai_corelib_matmul_bf16 query layer "+std::to_string(i));
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
