/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \note One place knows both the family names from model_list.json and the
///       engine types behind them. Everything else goes through the registry.
/// \note Every family registers the flm backend. Only families with a corelib
///       engine also register a rai backend, and only when this build has one.
#include "AutoModel/automodel.hpp"
#include "AutoModel/flm_backend.hpp"
#include "AutoModel/model_backend.hpp"

#if defined(FLM_ENABLE_RAI)
#include "models/phi4/rai/phi4_rai_backend.hpp"
#endif

namespace flm::backend {
namespace {

/// \brief register the flm backend for one family
/// \tparam Engine the concrete engine type
/// \param registry the registry to populate
/// \param family the family name, as in details.family
template <class Engine>
void RegisterFlm(BackendRegistry& registry, const char* family) {
    registry.register_backend(family, kFlmBackendId,
                              flm_factory<Engine>());
}

}  // namespace

void register_builtin_backends(BackendRegistry& registry) {
    RegisterFlm<llama_npu>(registry, "llama3");
    RegisterFlm<llama_npu>(registry, "deepseek-r1");
    RegisterFlm<qwen3_npu>(registry, "deepseek-r1-0528");
    RegisterFlm<qwen2_npu>(registry, "qwen2");
    RegisterFlm<qwen2vl_npu>(registry, "qwen2vl");
    RegisterFlm<qwen3_npu>(registry, "qwen3");
    RegisterFlm<qwen3_npu>(registry, "qwen3-it");
    RegisterFlm<qwen3_npu>(registry, "qwen3-tk");
    RegisterFlm<qwen3vl_npu>(registry, "qwen3vl");
    RegisterFlm<qwen3vl_flash>(registry, "qwen3vl-flash");
    RegisterFlm<qwen3_5vl_npu>(registry, "qwen3.5");
    RegisterFlm<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterFlm<gemma_npu>(registry, "gemma3");
    RegisterFlm<gemma_text_npu>(registry, "gemma3-text");
    RegisterFlm<gemma4e_npu>(registry, "gemma4e");
    RegisterFlm<gemma4e_flash>(registry, "gemma4e-flash");
    RegisterFlm<gemma4_12b_npu>(registry, "gemma4-12b");
    RegisterFlm<hunyuan_npu>(registry, "hunyuan");
    RegisterFlm<gpt_oss_npu>(registry, "gpt-oss");
    RegisterFlm<lfm2_npu>(registry, "lfm2");
    RegisterFlm<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterFlm<nanbeige_npu>(registry, "nanbeige");
    RegisterFlm<phi4_npu>(registry, "phi4");

#if defined(FLM_ENABLE_RAI)
    registry.register_backend("phi4", flm::backend::kRaiBackendId,
                              flm::phi4::rai_factory(),
                              flm::phi4::rai_traits());
#endif
}

}  // namespace flm::backend
