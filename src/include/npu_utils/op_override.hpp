/// \file op_override.hpp
/// \brief Interface for replacing an individual model operation with a user
///        supplied implementation.
/// \note  An override owns everything it needs to run: its own xclbin, its own
///        instruction streams and its own weights, all obtained through the
///        public npu_utils surface. The framework only tells it which operation
///        is being dispatched and hands it the buffers the default would have
///        received.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "buffer.hpp"
// The runtime alias flm_rt lives under a different name in the two trees this
// header is built in.
#if __has_include("device_runtime.hpp")
#include "device_runtime.hpp"
#else
#include "flm_runtime.hpp"
#endif

namespace flm {

/// \brief Outcome of an overridden dispatch.
///
/// Three states, distinguished because an override is not obliged to be exactly
/// one NPU dispatch:
///   - holds a run: the usual case, started and waited on by the caller;
///   - empty: the override already finished the work (on the host, or across
///     several dispatches it waited on itself);
///   - declined: the override does not serve this call, so the caller must run
///     the default implementation instead.
class op_result {
public:
    op_result() : declined_(false) {}
    explicit op_result(flm_rt::run run) : run_(std::move(run)), declined_(false) {}

    static op_result decline() {
        op_result r;
        r.declined_ = true;
        return r;
    }

    bool declined() const { return this->declined_; }

    void start() {
        if (this->run_.has_value()) this->run_->start();
    }

    ert_cmd_state wait() {
        if (!this->run_.has_value()) return ERT_CMD_STATE_COMPLETED;
        return this->run_->wait();
    }

private:
    std::optional<flm_rt::run> run_;
    bool declined_;
};

/// \brief Geometry of the sequence chunk a prefill dispatch is part of.
/// \note Every projection's M is `padded`. The engine refreshes this once per
///       chunk, before the first operation of that chunk runs.
struct op_extent {
    uint32_t padded = 0;      ///< rows dispatched, padded up to the engine's row granularity
    uint32_t effective = 0;   ///< rows of `padded` that hold real tokens
    uint32_t offset = 0;      ///< row at which this chunk's tokens begin
};

/// \brief One dispatch of a named model operation.
///
/// \note `args` are the buffers the default implementation receives, in the
///       default's own order. That order is part of each operation's documented
///       contract; it is not normalised across operations. The framework does
///       not interpret them, and an override is free to ignore them and use
///       weights it brought itself.
struct op_call {
    std::string_view role;        ///< operation name without the layer prefix, e.g. "mlp.up_proj"
    int layer;                    ///< layer index, or -1 for a model-level operation
    op_extent extent;             ///< geometry of the sequence chunk being processed
    bool blocking;                ///< the caller waits on this dispatch and overlaps nothing with it
    std::span<bytes* const> args;
};

/// \brief User supplied replacement for one or more model operations.
class op_override {
public:
    virtual ~op_override() = default;

    /// \brief Handle one dispatch.
    /// \return an op_result; return op_result::decline() to fall back to the
    ///         default implementation for this particular call.
    /// \note When call.blocking is set the caller does nothing until the work
    ///       finishes, so an override may run it synchronously and return an
    ///       empty op_result. That avoids materialising a run object, which is
    ///       the more expensive of the two dispatch paths.
    virtual op_result create_run(const op_call& call) = 0;
};

template <typename App>
class app_layer_ref;

/// \brief Per-layer override table, mixed into the backend's npu_app.
///
/// \note CRTP rather than virtual dispatch: npu_app's call operators are
///       variadic templates over buffer types, which cannot be virtual, and the
///       unoverridden path must stay a single predictable branch.
template <typename App>
class overridable_app {
public:
    /// \brief Bind this app to a layer for the duration of one dispatch.
    app_layer_ref<App> at(int layer) { return app_layer_ref<App>(static_cast<App*>(this), layer); }

    const std::string& op_role() const { return this->op_role_; }

    /// \note Called by op_registry; not part of the override-author surface.
    void _declare_op(std::string role, int layer_count, const op_extent* extent) {
        this->op_role_ = std::move(role);
        this->op_overrides_.resize(static_cast<size_t>(layer_count) + 1, nullptr);
        this->op_extent_ = extent;
    }

    /// \note Called by op_registry; not part of the override-author surface.
    void _set_op_override(int layer, op_override* hook) {
        op_override*& slot = this->op_overrides_.at(static_cast<size_t>(layer + 1));
        this->op_override_count_ += (hook != nullptr) - (slot != nullptr);
        slot = hook;
    }

    const op_extent* _op_extent() const { return this->op_extent_; }

    op_override* _op_override(int layer) const {
        if (this->op_override_count_ == 0) return nullptr;
        const size_t i = static_cast<size_t>(layer + 1);
        if (i >= this->op_overrides_.size()) return nullptr;
        return this->op_overrides_[i];
    }

protected:
    std::string op_role_;
    std::vector<op_override*> op_overrides_;  ///< indexed by layer + 1, so -1 addresses model-level ops
    int op_override_count_ = 0;
    const op_extent* op_extent_ = nullptr;    ///< owned by the registry, refreshed once per chunk
};

/// \brief An npu_app bound to a layer index, as returned by npu_app::at().
template <typename App>
class app_layer_ref {
public:
    app_layer_ref(App* app, int layer) : app_(app), layer_(layer) {}

    template <typename... BoArgs>
    ert_cmd_state operator()(BoArgs&&... args) {
        op_result result = this->_dispatch(true, std::forward<BoArgs>(args)...);
        if (result.declined()) return (*this->app_)(std::forward<BoArgs>(args)...);
        result.start();
        return result.wait();
    }

    template <typename... BoArgs>
    op_result create_run(BoArgs&&... args) {
        op_result result = this->_dispatch(false, std::forward<BoArgs>(args)...);
        if (result.declined()) return op_result(this->app_->create_run(std::forward<BoArgs>(args)...));
        return result;
    }

private:
    template <typename... BoArgs>
    op_result _dispatch(bool blocking, BoArgs&&... args) {
        op_override* hook = this->app_->_op_override(this->layer_);
        if (hook == nullptr) return op_result::decline();
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { static_cast<bytes*>(&args)... };
        op_call call{ this->app_->op_role(), this->layer_, *this->app_->_op_extent(),
                      blocking, std::span<bytes* const>(bo_args) };
        return hook->create_run(call);
    }

    App* app_;
    int layer_;
};

}  // namespace flm
