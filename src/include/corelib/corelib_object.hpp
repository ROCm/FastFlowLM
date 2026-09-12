#pragma once

#include "corelib/corelib_api.hpp"

#include <memory>
#include <utility>

namespace flm::corelib {

struct StreamTag {};
struct TensorTag {};
struct TensorWindowTag {};
struct MatMulWeightsTag {};
struct SsMlpWeightsTag {};

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
using UniqueMatMulWeights = UniqueObject<MatMulWeightsTag>;
using UniqueSsMlpWeights = UniqueObject<SsMlpWeightsTag>;

}  // namespace flm::corelib
