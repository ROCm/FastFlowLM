/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \note One place knows both the family names from model_list.json and the
///       engine types behind them. Everything else goes through the registry.
/// \note Every family registers the flm backend. Only families with a corelib
///       engine also register a core backend, and only when this build has one.
#include "AutoModel/automodel.hpp"
#include "AutoModel/flm_backend.hpp"
#include "AutoModel/model_backend.hpp"

#if defined(FLM_ENABLE_CORE)
#include "models/gemma4/core/dev04/gemma4_core_backend.hpp"
#include "models/llama/core/dev04/llama_core_backend.hpp"
#include "models/phi4/core/dev04/phi4_core_backend.hpp"
#include "models/qwen3/core/dev04/qwen3_core_backend.hpp"
#include "models/qwen35/core/dev04/qwen35_core_backend.hpp"
#include "models/qwen36/core/dev04/qwen36_core_backend.hpp"
#include "models/qwen38/core/dev04/qwen38_core_backend.hpp"
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
    RegisterFlm<minicpm_v_4_7_npu>(registry, "minicpm-v-4.7");
    RegisterFlm<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterFlm<qwen3_8mtp_npu>(registry, "qwen3.8-mtp");
    RegisterFlm<gemma_npu>(registry, "gemma3");
    RegisterFlm<gemma_text_npu>(registry, "gemma3-text");
    RegisterFlm<gemma4e_npu>(registry, "gemma4e");
    RegisterFlm<gemma4e_flash>(registry, "gemma4e-flash");
    RegisterFlm<gemma4e_dflash>(registry, "gemma4e-dflash");
    RegisterFlm<gemma4_12b_npu>(registry, "gemma4-12b");
    RegisterFlm<hunyuan_npu>(registry, "hunyuan");
    RegisterFlm<gpt_oss_npu>(registry, "gpt-oss");
    RegisterFlm<lfm2_npu>(registry, "lfm2");
    RegisterFlm<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterFlm<lfm3_npu>(registry, "lfm3");
    RegisterFlm<nanbeige_npu>(registry, "nanbeige");
    RegisterFlm<phi4_npu>(registry, "phi4");

#if defined(FLM_ENABLE_CORE)
    registry.register_backend("gemma4e", flm::backend::kCoreBackendId,
                              flm::gemma4::core_factory(),
                              flm::gemma4::core_traits());
    registry.register_backend("phi4", flm::backend::kCoreBackendId,
                              flm::phi4::core_factory(),
                              flm::phi4::core_traits());
    registry.register_backend("qwen3", flm::backend::kCoreBackendId,
                              flm::qwen3::core_factory(),
                              flm::qwen3::core_traits());
    registry.register_backend("qwen3.5", flm::backend::kCoreBackendId,
                              flm::qwen35::core_factory(),
                              flm::qwen35::core_traits());
    registry.register_backend("qwen3.6-moe", flm::backend::kCoreBackendId,
                              flm::qwen36::core_factory(),
                              flm::qwen36::core_traits());
    registry.register_backend("qwen3.8-mtp", flm::backend::kCoreBackendId,
                              flm::qwen38::core_factory(),
                              flm::qwen38::core_traits());
    registry.register_backend("llama3", flm::backend::kCoreBackendId,
                              flm::llama::core_factory(),
                              flm::llama::core_traits());
#endif
}

}  // namespace flm::backend
