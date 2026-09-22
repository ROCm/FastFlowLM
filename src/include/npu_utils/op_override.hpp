// Interface for replacing one model operation with a user-supplied
// implementation. An override owns everything it needs to run -- xclbin,
// instruction streams, weights -- through the public npu_utils surface; the
// framework only tells it which operation is being dispatched and hands it
// the buffers the default implementation would have received.
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

// An override is not obliged to be exactly one NPU dispatch, so its result
// is one of three states: holds a run (the caller starts and waits on it),
// empty (the override already finished the work itself), or declined (the
// caller must run the default implementation instead).
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

    // For batching into an existing schedule (e.g. a runlist); an override
    // author does not need this.
    bool has_run() const { return this->run_.has_value(); }

    flm_rt::run& run() { return *this->run_; }

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

// Geometry of the sequence chunk a prefill dispatch is part of. Every
// projection's M is `padded`; the engine refreshes this once per chunk,
// before the first operation of that chunk runs.
struct op_extent {
    uint32_t padded = 0;      ///< rows dispatched, padded up to the engine's row granularity
    uint32_t effective = 0;   ///< rows of `padded` that hold real tokens
    uint32_t offset = 0;      ///< row at which this chunk's tokens begin
};

// One dispatch of a named model operation. `args` are the buffers the default
// implementation receives, in the default's own order -- part of each
// operation's documented contract, not normalised across operations. The
// framework does not interpret them; an override may ignore them and use
// weights it brought itself.
struct op_call {
    std::string_view name;        ///< the operation's name, as the engine declared it
    int index;                    ///< which of the engine's dispatch sites for that name
    op_extent extent;             ///< geometry of the sequence chunk being processed
    bool blocking;                ///< the caller waits on this dispatch and overlaps nothing with it
    std::span<bytes* const> args;
};

// User-supplied replacement for one or more model operations.
class op_override {
public:
    virtual ~op_override() = default;

    // Return op_result::decline() to fall back to the default implementation
    // for this call. When call.blocking is set, the caller does nothing until
    // the work finishes, so an override may run it synchronously and return
    // an empty op_result instead of materialising a run object.
    virtual op_result create_run(const op_call& call) = 0;
};

template <typename App>
class app_index_ref;

// Per-layer override table, mixed into the backend's npu_app. CRTP rather
// than virtual dispatch: npu_app's call operators are variadic templates
// over buffer types, which cannot be virtual, and the unoverridden path
// must stay a single predictable branch.
template <typename App>
class overridable_app {
public:
    // What an index means is the engine's business -- a layer, a block, a
    // position in a flattened nest. All this needs is that it be dense.
    app_index_ref<App> at(int index) { return app_index_ref<App>(static_cast<App*>(this), index); }

    const std::string& op_name() const { return this->op_name_; }

    // Called by op_registry; not part of the override-author surface.
    void _declare_op(std::string name, int index_count, const op_extent* extent) {
        this->op_name_ = std::move(name);
        this->op_overrides_.resize(static_cast<size_t>(index_count) + 1, nullptr);
        this->op_extent_ = extent;
    }

    // Called by op_registry; not part of the override-author surface.
    void _set_op_override(int index, op_override* hook) {
        op_override*& slot = this->op_overrides_.at(static_cast<size_t>(index + 1));
        this->op_override_count_ += (hook != nullptr) - (slot != nullptr);
        slot = hook;
    }

    const op_extent* _op_extent() const { return this->op_extent_; }

    op_override* _op_override(int index) const {
        if (this->op_override_count_ == 0) return nullptr;
        const size_t i = static_cast<size_t>(index + 1);
        if (i >= this->op_overrides_.size()) return nullptr;
        return this->op_overrides_[i];
    }

protected:
    std::string op_name_;
    std::vector<op_override*> op_overrides_;  ///< indexed by index + 1, so -1 addresses a lone site
    int op_override_count_ = 0;
    const op_extent* op_extent_ = nullptr;    ///< owned by the registry, refreshed once per chunk
};

// An npu_app bound to one dispatch site, as returned by npu_app::at().
template <typename App>
class app_index_ref {
public:
    app_index_ref(App* app, int index) : app_(app), index_(index) {}

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
        op_override* hook = this->app_->_op_override(this->index_);
        if (hook == nullptr) return op_result::decline();
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { static_cast<bytes*>(&args)... };
        op_call call{ this->app_->op_name(), this->index_, *this->app_->_op_extent(),
                      blocking, std::span<bytes* const>(bo_args) };
        return hook->create_run(call);
    }

    App* app_;
    int index_;
};

}  // namespace flm
