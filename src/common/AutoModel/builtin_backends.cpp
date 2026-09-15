/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \note One place knows both the family names from model_list.json and the
///       engine types behind them. Everything else goes through the registry.
#include "AutoModel/automodel.hpp"
#include "AutoModel/flm_npu_backend.hpp"
#include "AutoModel/model_backend.hpp"

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include "models/phi4/corelib/phi4_corelib_backend.hpp"
#endif

namespace flm::backend {
namespace {

/// \brief register the FastFlowLM NPU backend for one family
/// \tparam Engine the concrete engine type
/// \param registry the registry to populate
/// \param family the family name, as in details.family
template <class Engine>
void RegisterFlmNpu(BackendRegistry& registry, const char* family) {
    registry.register_backend(family, kDefaultBackendId,
                              flm_npu_factory<Engine>());
}

}  // namespace

void register_builtin_backends(BackendRegistry& registry) {
    RegisterFlmNpu<llama_npu>(registry, "llama3");
    RegisterFlmNpu<llama_npu>(registry, "deepseek-r1");
    RegisterFlmNpu<qwen3_npu>(registry, "deepseek-r1-0528");
    RegisterFlmNpu<qwen2_npu>(registry, "qwen2");
    RegisterFlmNpu<qwen2vl_npu>(registry, "qwen2vl");
    RegisterFlmNpu<qwen3_npu>(registry, "qwen3");
    RegisterFlmNpu<qwen3_npu>(registry, "qwen3-it");
    RegisterFlmNpu<qwen3_npu>(registry, "qwen3-tk");
    RegisterFlmNpu<qwen3vl_npu>(registry, "qwen3vl");
    RegisterFlmNpu<qwen3_5vl_npu>(registry, "qwen3.5");
    RegisterFlmNpu<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterFlmNpu<gemma_npu>(registry, "gemma3");
    RegisterFlmNpu<gemma_text_npu>(registry, "gemma3-text");
    RegisterFlmNpu<gemma4e_npu>(registry, "gemma4e");
    RegisterFlmNpu<gpt_oss_npu>(registry, "gpt-oss");
    RegisterFlmNpu<lfm2_npu>(registry, "lfm2");
    RegisterFlmNpu<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterFlmNpu<nanbeige_npu>(registry, "nanbeige");
    RegisterFlmNpu<phi4_npu>(registry, "phi4");

#if defined(FLM_ENABLE_CORELIB_AIE4)
    registry.register_backend("phi4", flm::phi4::kCorelibAie4BackendId,
                              flm::phi4::corelib_aie4_factory(),
                              flm::phi4::corelib_aie4_traits());
#endif
}

}  // namespace flm::backend
