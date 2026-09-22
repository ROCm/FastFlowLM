// Interface for replacing one model operation with a user-supplied
// implementation. An override owns everything it needs to run -- xclbin,
// instruction streams, weights -- through the public npu_utils surface; the
// framework only tells it which operation is being dispatched and hands it
// the arguments the call site built for it.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
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

}  // namespace flm
