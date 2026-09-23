// A plugin overrides one operation by binding a function to its name. The
// engine looks up that name once, at construction, and gets back a callable
// with the exact signature it would have called directly -- there is no
// override at all past that point, just a function pointer.
//
// A bound function returns hook_result<R>: done(value) to hand back a real
// result, skip() to say the operation happened and there is nothing further
// for the caller to do, or defer() to fall through to the engine's own
// implementation. skip() only matters to a caller that would otherwise do
// something with the value (e.g. start()/wait() an xrt::run); a caller that
// already discards its result can ignore the distinction.
#pragma once

#include <any>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace flm {

enum class hook_disposition { done, skip, defer };

template <typename T>
class hook_result {
public:
    hook_result(T value) : disposition_(hook_disposition::done), value_(std::move(value)) {}
    static hook_result skip()  { return hook_result(hook_disposition::skip); }
    static hook_result defer() { return hook_result(hook_disposition::defer); }

    bool is_defer() const { return disposition_ == hook_disposition::defer; }
    bool is_skip() const { return disposition_ == hook_disposition::skip; }
    T&& value() && { return std::move(*value_); }

    // skip() becomes nullopt; a caller that only cares whether there is a
    // run to start/wait on can use this instead of checking is_skip() itself.
    std::optional<T> to_optional() && {
        if (this->is_skip()) return std::nullopt;
        return std::move(*this).value();
    }

private:
    explicit hook_result(hook_disposition d) : disposition_(d) {}
    hook_disposition disposition_;
    std::optional<T> value_;
};

namespace detail {

template <size_t Keep, typename F, typename Tuple, size_t... I>
decltype(auto) call_prefix(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

}  // namespace detail

class hook_registry {
public:
    // Called by a plugin. Sig is always hook_result<R>(Args...); fn's
    // signature must match what the engine resolves under this name.
    template <typename Sig, typename Fn>
    void override_op(std::string_view name, Fn fn) {
        this->overrides_[std::string(name)] = std::function<Sig>(std::move(fn));
    }

    // Called by the engine, once per operation, after every plugin has
    // registered. Returns a callable with the exact signature the call site
    // uses: whichever plugin bound `name` runs first, falling through to
    // default_fn when it defers (or when nothing was bound).
    template <typename Sig>
    std::function<Sig> resolve(std::string_view name, std::function<Sig> default_fn) {
        auto it = this->overrides_.find(std::string(name));
        if (it == this->overrides_.end()) return default_fn;
        this->resolved_.insert(std::string(name));
        std::function<Sig> override_fn;
        try {
            override_fn = std::any_cast<std::function<Sig>>(it->second);
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("hook_registry: '" + std::string(name)
                                     + "' was bound with a signature the engine does not expect");
        }
        return [override_fn, default_fn](auto&&... args) {
            auto result = override_fn(args...);
            return result.is_defer() ? default_fn(args...) : result;
        };
    }

    // Same as resolve(), but for the common case where the default is App's
    // own operator(), called with Sig's last N arguments dropped -- the
    // scalar metadata (layer index, M or context length) a default
    // implementation never needs. See resolve_drop_trailing_async() for a
    // default that builds a run to start/wait on instead.
    template <typename Sig, size_t N, typename App>
    std::function<Sig> resolve_drop_trailing(std::string_view name, App& app) {
        return this->resolve<Sig>(name, std::function<Sig>([&app](auto&&... args) {
            constexpr size_t keep = sizeof...(args) - N;
            return detail::call_prefix<keep>(
                [&app](auto&&... kept) { return app(std::forward<decltype(kept)>(kept)...); },
                std::forward_as_tuple(std::forward<decltype(args)>(args)...),
                std::make_index_sequence<keep>{});
        }));
    }

    // Same as resolve_drop_trailing(), but the default is App's create_run().
    template <typename Sig, size_t N, typename App>
    std::function<Sig> resolve_drop_trailing_async(std::string_view name, App& app) {
        return this->resolve<Sig>(name, std::function<Sig>([&app](auto&&... args) {
            constexpr size_t keep = sizeof...(args) - N;
            return detail::call_prefix<keep>(
                [&app](auto&&... kept) { return app.create_run(std::forward<decltype(kept)>(kept)...); },
                std::forward_as_tuple(std::forward<decltype(args)>(args)...),
                std::make_index_sequence<keep>{});
        }));
    }

    // Throws if a plugin bound a name the engine never resolved -- almost
    // always a typo, or an override left behind by an engine refactor.
    void check_all_resolved() const {
        std::ostringstream unresolved;
        bool first = true;
        for (const auto& [name, unused] : this->overrides_) {
            if (this->resolved_.count(name)) continue;
            if (!first) unresolved << ", ";
            first = false;
            unresolved << name;
        }
        if (!first) throw std::runtime_error("hook_registry: bound but never resolved: " + unresolved.str());
    }

private:
    std::map<std::string, std::any> overrides_;
    std::set<std::string> resolved_;
};

}  // namespace flm
