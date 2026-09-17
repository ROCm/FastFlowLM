/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \note One place knows both the family names from model_list.json and the
///       engine types behind them. Everything else goes through the registry.
/// \note Every family runs on aie2p. Only families with a corelib engine also
///       register an aie4 backend, and only when this build has one.
#include "AutoModel/automodel.hpp"
#include "AutoModel/aie2p_backend.hpp"
#include "AutoModel/model_backend.hpp"

#if defined(FLM_ENABLE_AIE4)
#include "models/phi4/aie4/phi4_aie4_backend.hpp"
#endif

namespace flm::backend {
namespace {

/// \brief register the aie2p backend for one family
/// \tparam Engine the concrete engine type
/// \param registry the registry to populate
/// \param family the family name, as in details.family
template <class Engine>
void RegisterAie2p(BackendRegistry& registry, const char* family) {
    registry.register_backend(family, kAie2pBackendId,
                              aie2p_factory<Engine>());
}

}  // namespace

void register_builtin_backends(BackendRegistry& registry) {
    RegisterAie2p<llama_npu>(registry, "llama3");
    RegisterAie2p<llama_npu>(registry, "deepseek-r1");
    RegisterAie2p<qwen3_npu>(registry, "deepseek-r1-0528");
    RegisterAie2p<qwen2_npu>(registry, "qwen2");
    RegisterAie2p<qwen2vl_npu>(registry, "qwen2vl");
    RegisterAie2p<qwen3_npu>(registry, "qwen3");
    RegisterAie2p<qwen3_npu>(registry, "qwen3-it");
    RegisterAie2p<qwen3_npu>(registry, "qwen3-tk");
    RegisterAie2p<qwen3vl_npu>(registry, "qwen3vl");
    RegisterAie2p<qwen3_5vl_npu>(registry, "qwen3.5");
    RegisterAie2p<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterAie2p<gemma_npu>(registry, "gemma3");
    RegisterAie2p<gemma_text_npu>(registry, "gemma3-text");
    RegisterAie2p<gemma4e_npu>(registry, "gemma4e");
    RegisterAie2p<gpt_oss_npu>(registry, "gpt-oss");
    RegisterAie2p<lfm2_npu>(registry, "lfm2");
    RegisterAie2p<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterAie2p<nanbeige_npu>(registry, "nanbeige");
    RegisterAie2p<phi4_npu>(registry, "phi4");

#if defined(FLM_ENABLE_AIE4)
    registry.register_backend("phi4", flm::backend::kAie4BackendId,
                              flm::phi4::aie4_factory(),
                              flm::phi4::aie4_traits());
#endif
}

}  // namespace flm::backend
