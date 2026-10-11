#pragma once

#include "rai/corelib_api.hpp"

#include <memory>
#include <string>
#include <utility>

namespace flm::corelib {

struct StreamTag {};
struct TensorTag {};
struct TensorWindowTag {};
struct HostViewTag {};
struct MatMulWeightsTag {};
struct SsMlpWeightsTag {};
struct RmsNormWeightsTag {};
struct PleWeightsTag {};
struct DwConvWeightsTag {};
struct ModelTag {};
struct ModelWeightsTag {};
struct ModelSnapshotTag {};

template <typename Tag>
class UniqueObject final {
public:
    UniqueObject() noexcept = default;

    UniqueObject(std::shared_ptr<const CorelibApi> api, void* object) noexcept
        : api_(std::move(api)), object_(object) {
        if (object_) api_->RegisterObject();
    }

    ~UniqueObject() { reset(); }

    UniqueObject(const UniqueObject&) = delete;
    UniqueObject& operator=(const UniqueObject&) = delete;

    UniqueObject(UniqueObject&& other) noexcept
        : api_(std::move(other.api_)), object_(std::exchange(other.object_, nullptr)) {}

    UniqueObject& operator=(UniqueObject&& other) noexcept {
        if (this != &other) {
            reset();
            api_ = std::move(other.api_);
            object_ = std::exchange(other.object_, nullptr);
        }
        return *this;
    }

    void reset() noexcept {
        if (object_) api_->Release(std::exchange(object_, nullptr));
        api_.reset();
    }

    void* get() const noexcept { return object_; }
    explicit operator bool() const noexcept { return object_ != nullptr; }

private:
    std::shared_ptr<const CorelibApi> api_;
    void* object_{};
};

using UniqueStream = UniqueObject<StreamTag>;
using UniqueTensor = UniqueObject<TensorTag>;
using UniqueTensorWindow = UniqueObject<TensorWindowTag>;
/// \note A host view borrows the caller's bytes and never copies them, so
///       whatever owns those bytes must outlive the view.
using UniqueHostView = UniqueObject<HostViewTag>;
using UniqueMatMulWeights = UniqueObject<MatMulWeightsTag>;
using UniqueSsMlpWeights = UniqueObject<SsMlpWeightsTag>;
using UniqueRmsNormWeights = UniqueObject<RmsNormWeightsTag>;
using UniquePleWeights = UniqueObject<PleWeightsTag>;
using UniqueDwConvWeights = UniqueObject<DwConvWeightsTag>;
using UniqueModel = UniqueObject<ModelTag>;
using UniqueModelWeights = UniqueObject<ModelWeightsTag>;
using UniqueModelSnapshot = UniqueObject<ModelSnapshotTag>;

/// \brief a stream on the PDI pair that resolves the kernels `bundle` ships
/// \note A new stream resolves no kernel at all: every enumeration comes back
///       empty and every dispatch runs on the host until a bundle is enabled,
///       so this has to precede the shape plan and the weights.
inline UniqueStream OpenModelStream(const std::shared_ptr<const CorelibApi>& api,
                                    int prefill_pdi, int token_pdi, const char* bundle) {
    void* raw = nullptr;
    api->Check(api->functions().create_stream(prefill_pdi, token_pdi, &raw),
               "ryzenai_corelib_create_stream");
    UniqueStream stream(api, raw);
    api->Check(api->functions().stream_enable_kernel_bundle(stream.get(), bundle),
               std::string("ryzenai_corelib_stream_enable_kernel_bundle ") + bundle);
    return stream;
}

}  // namespace flm::corelib
