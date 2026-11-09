#define FLM_MODEL_EXTENSION_IMPLEMENTATION
#include "AutoModel/model_extension.h"

static int Register(
    const flm_model_extension_registrar_v1* registrar,
    const flm_error_sink_v1* error) {
    (void)registrar;
    (void)error;
    return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
}

static const flm_model_extension_v1 kExtension = {
    sizeof(flm_model_extension_v1),
    FLM_MODEL_EXTENSION_ABI_V1,
#if defined(FLM_USE_HRX)
    FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX,
#else
    FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT,
#endif
    "installed-package-contract-test",
    "1.0.0",
    Register,
};

FLM_MODEL_EXTENSION_EXPORT
const flm_model_extension_v1* flm_model_extension_query_v1(void) {
    return &kExtension;
}
