/// \file builtin_frontends.cpp
/// \brief Causal-generation frontends shipped with FastFlowLM
#include "AutoModel/model_frontend.hpp"

#include "AutoModel/modeling_gemma3.hpp"
#include "AutoModel/modeling_gemma3_text.hpp"
#include "AutoModel/modeling_gemma4_12b.hpp"
#include "AutoModel/modeling_gemma4e.hpp"
#include "AutoModel/modeling_gpt_oss.hpp"
#include "AutoModel/modeling_hunyuan.hpp"
#include "AutoModel/modeling_lfm2.hpp"
#include "AutoModel/modeling_lfm3.hpp"
#include "AutoModel/modeling_llama3.hpp"
#include "AutoModel/modeling_minicpm_v.hpp"
#include "AutoModel/modeling_nanbeige.hpp"
#include "AutoModel/modeling_phi4.hpp"
#include "AutoModel/modeling_qwen2.hpp"
#include "AutoModel/modeling_qwen2vl.hpp"
#include "AutoModel/modeling_qwen3.hpp"
#include "AutoModel/modeling_qwen3_5_omni.hpp"
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "AutoModel/modeling_qwen3_6_moe.hpp"
#include "AutoModel/modeling_qwen3_8mtp.hpp"
#include "AutoModel/modeling_qwen3vl.hpp"

#include "device_runtime.hpp"

#include <memory>

namespace flm::frontend {
namespace {

void DeleteHostFrontend(void*, void* frontend) {
    delete static_cast<AutoModel*>(frontend);
}

template <class Frontend>
void Register(FrontendRegistry& registry, const char* family) {
    registry.register_frontend(
        family,
        [](void* device) {
            return own_frontend(std::make_unique<Frontend>(
                static_cast<flm_rt::device*>(device)));
        });
}

}  // namespace

FrontendPtr own_frontend(std::unique_ptr<AutoModel> frontend) {
    return FrontendPtr(
        frontend.release(),
        FrontendDeleter{nullptr, DeleteHostFrontend, nullptr});
}

void register_builtin_frontends(FrontendRegistry& registry) {
    Register<Llama3>(registry, "llama3");
    Register<DeepSeek_r1_8b>(registry, "deepseek-r1");
    Register<DeepSeek_r1_0528_8b>(registry, "deepseek-r1-0528");
    Register<Qwen2>(registry, "qwen2");
    Register<Qwen2VL>(registry, "qwen2vl");
    Register<Qwen3>(registry, "qwen3");
    Register<Qwen3_IT>(registry, "qwen3-it");
    Register<Qwen3_TK>(registry, "qwen3-tk");
    Register<Qwen3VL>(registry, "qwen3vl");
    Register<Qwen3VL_Flash>(registry, "qwen3vl-flash");
    Register<Qwen3_5VL>(registry, "qwen3.5");
    Register<MiniCPM_V>(registry, "minicpm-v-4.7");
    Register<Qwen3_5_Omni>(registry, "qwen3.5-omni");
    Register<Qwen3_6_MOE>(registry, "qwen3.6-moe");
    Register<Qwen3_8MTP>(registry, "qwen3.8-mtp");
    Register<Gemma3>(registry, "gemma3");
    Register<Gemma3_Text_Only>(registry, "gemma3-text");
    Register<Gemma4e>(registry, "gemma4e");
    Register<Gemma4e_Flash>(registry, "gemma4e-flash");
    Register<Gemma4e_DFlash>(registry, "gemma4e-dflash");
    Register<Gemma4_12B>(registry, "gemma4-12b");
    Register<GPT_OSS>(registry, "gpt-oss");
    Register<LFM2>(registry, "lfm2");
    Register<LFM2_5_TK>(registry, "lfm2.5-tk");
    Register<LFM3>(registry, "lfm3");
    Register<Nanbeige>(registry, "nanbeige");
    Register<Hunyuan>(registry, "hunyuan");
    Register<Phi4>(registry, "phi4");
}

}  // namespace flm::frontend
