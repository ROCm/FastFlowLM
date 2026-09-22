// Interface for replacing one model operation with a user-supplied
// implementation. An override owns everything it needs to run -- xclbin,
// instruction streams, weights -- through the public npu_utils surface; the
// framework only tells it which operation is being dispatched and hands it
// the arguments the call site built for it.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "buffer.hpp"
// The runtime alias flm_rt lives under a different name in the two trees this
// header is built in.
#if __has_include("device_runtime.hpp")
#include "device_runtime.hpp"
#else
#include "flm_runtime.hpp"
#endif

namespace flm {

// Outcome of a hook's create_run(): defer to the default implementation,
// skip it (the hook already did the work itself, or it genuinely does not
// apply -- e.g. a dequant step whose weights are already in the right
// form), or hand back a run for the caller to start and wait on, or add to
// a runlist.
class op_result {
public:
    static op_result defer() { return op_result(state::defer); }
    static op_result skip() { return op_result(state::skip); }
    explicit op_result(flm_rt::run run) : state_(state::run), run_(std::move(run)) {}

    bool should_defer() const { return this->state_ == state::defer; }
    bool has_run() const { return this->state_ == state::run; }
    flm_rt::run& run() { return *this->run_; }

    void start() {
        if (this->run_.has_value()) this->run_->start();
    }

    ert_cmd_state wait() {
        if (!this->run_.has_value()) return ERT_CMD_STATE_COMPLETED;
        return this->run_->wait();
    }

private:
    enum class state { defer, skip, run };
    explicit op_result(state s) : state_(s) {}

    state state_;
    std::optional<flm_rt::run> run_;
};

// One argument to a hook: a buffer, or a plain integer (a layer index,
// a length, anything a specific operation's contract calls for).
using op_arg = std::variant<bytes*, int64_t>;

// One dispatch of a named model operation, passed to a bound hook's
// create_run(). `args` is exactly the operation's argument list -- how
// many, in what order, and which are buffers versus integers is specific
// to `name` and documented alongside it, the same way a plain function's
// signature is documented. The framework does not interpret them.
struct op_call {
    std::string_view name;
    bool blocking;    ///< the caller waits on this dispatch and overlaps nothing with it
    std::span<const op_arg> args;
};

// User-supplied replacement for one or more model operations.
class op_override {
public:
    virtual ~op_override() = default;
    virtual op_result create_run(const op_call& call) = 0;
};

// Runs `call` through `hook` if bound, else through `default_call` (a
// niladic callable the call site writes, e.g. `[&]{ return app(x, w); }`,
// covering whatever arguments the default implementation actually takes --
// not necessarily the same ones `call.args` carries for the hook).
// `hook` is a pointer op_registry::resolve() already produced; resolving
// by name happens once, not on this path.
template <typename DefaultCall>
ert_cmd_state dispatch(op_override* hook, const op_call& call, DefaultCall&& default_call) {
    if (hook != nullptr) {
        op_result result = hook->create_run(call);
        if (!result.should_defer()) {
            result.start();
            return result.wait();
        }
    }
    return default_call();
}

// Non-blocking counterpart: returns a run from `hook` if it accepted the
// call, else one freshly created from `default_call`. A hook bound to a
// runlist-eligible site must return a run here, never op_result::skip().
template <typename DefaultCreateRun>
op_result dispatch_async(op_override* hook, const op_call& call, DefaultCreateRun&& default_create_run) {
    if (hook != nullptr) {
        op_result result = hook->create_run(call);
        if (!result.should_defer()) return result;
    }
    return op_result(default_create_run());
}

// Whether T is one of the buffer types a hook's args and a default
// implementation both take, as opposed to a plain integer (a layer index,
// a length) meant for the hook alone.
template <typename T>
inline constexpr bool is_op_buffer_v = std::is_base_of_v<bytes, std::remove_cv_t<std::remove_reference_t<T>>>;

// `name` resolved to `hook` (if any) and bound to `app`, the default it
// falls back to. Built once, from op_registry::resolve(); called at each
// dispatch site the same way `app` itself would be, with any trailing
// plain integers (a layer index, a chunk size) a hook needs appended --
// `app` never sees those, only the buffer arguments do.
template <typename App>
class bound_op {
public:
    bound_op() = default;
    bound_op(std::string_view name, op_override* hook, App& app)
        : name_(name), hook_(hook), app_(&app) {}

    template <typename... Args>
    ert_cmd_state operator()(Args&&... args) {
        const std::array<op_arg, sizeof...(Args)> a{ _to_arg(args)... };
        if (this->hook_ != nullptr) {
            op_result result = this->hook_->create_run({ this->name_, true, a });
            if (!result.should_defer()) {
                result.start();
                return result.wait();
            }
        }
        return this->_default(args...);
    }

    template <typename... Args>
    op_result create_run(Args&&... args) {
        const std::array<op_arg, sizeof...(Args)> a{ _to_arg(args)... };
        if (this->hook_ != nullptr) {
            op_result result = this->hook_->create_run({ this->name_, false, a });
            if (!result.should_defer()) return result;
        }
        return op_result(this->_default_run(args...));
    }

private:
    template <typename T>
    static op_arg _to_arg(T& v) {
        if constexpr (is_op_buffer_v<T>) return op_arg(static_cast<bytes*>(&v));
        else return op_arg(static_cast<int64_t>(v));
    }

    // Only the buffer arguments reach `app_`; a trailing layer index or
    // chunk size is for the hook alone.
    template <typename T>
    static auto _buffer_ref(T& v) {
        if constexpr (is_op_buffer_v<T>) return std::tuple<T&>(v);
        else return std::tuple<>();
    }

    template <typename... Args>
    ert_cmd_state _default(Args&... args) {
        return std::apply([this](auto&... bufs) { return (*this->app_)(bufs...); },
                          std::tuple_cat(_buffer_ref(args)...));
    }

    template <typename... Args>
    auto _default_run(Args&... args) {
        return std::apply([this](auto&... bufs) { return this->app_->create_run(bufs...); },
                          std::tuple_cat(_buffer_ref(args)...));
    }

    std::string_view name_;
    op_override* hook_ = nullptr;
    App* app_ = nullptr;
};

}  // namespace flm
