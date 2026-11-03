/// \file backends.cpp
/// \brief registers the one engine this harness drives
/// \note builtin_backends.cpp names every engine and so links every prebuilt
///       library; test_hunyuan and bench_prefill both link this instead.
#include "AutoModel/modeling_hunyuan.hpp"
#include "AutoModel/flm_backend.hpp"

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("hunyuan", kFlmBackendId, flm_factory<hunyuan_npu>());
}
}  // namespace flm::backend
