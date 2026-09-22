// The set of model operations that can be overridden, and the binding of an
// override to them. A model engine declares its operations once, at
// construction, so the key space is fixed: override_op() on an unknown key
// throws rather than silently never firing.
//
// Nothing here names an operation or spells a key -- that's the engine's:
// it may have layers, blocks, a nest of them or none. All this needs is that
// keys be unique and that each operation's dispatch sites be numbered
// densely from zero.
#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "op_override.hpp"

namespace flm {

template <typename App>
class op_registry_t {
public:
    // site_count: the largest dispatch-site index any operation will use.
    explicit op_registry_t(int site_count) : site_count_(site_count) {}

    // Declare that `app` implements operation `name` at site `index`,
    // reachable by a plugin as `key`. Several sites may share one app, and
    // several apps may serve one name at different sites; both are resolved
    // here, so neither the apps nor an override have to know about it.
    void declare(std::string key, std::string_view name, int index, App& app) {
        app._declare_op(std::string(name), this->site_count_, &this->extent_);
        this->slots_.emplace(std::move(key), slot{ &app, index });
    }

    // Publish the geometry of the chunk about to be processed, once per
    // chunk; every op_call of that chunk carries it.
    void set_extent(const op_extent& extent) { this->extent_ = extent; }

    std::vector<std::string> list_ops() const {
        std::vector<std::string> keys;
        keys.reserve(this->slots_.size());
        for (const auto& [key, unused] : this->slots_) keys.push_back(key);
        return keys;
    }

    // Routes `hook` to every operation matching `key`, a declared key or a
    // pattern whose '*' matches one whole dot-separated segment (e.g.
    // "layers.*.mlp.up_proj"). Returns the number of operations bound;
    // throws std::runtime_error if `key` matches nothing.
    size_t override_op(std::string_view key, std::shared_ptr<op_override> hook) {
        size_t bound = 0;
        for (auto& [declared, s] : this->slots_) {
            if (!_matches(key, declared)) continue;
            s.app->_set_op_override(s.index, hook.get());
            bound++;
        }
        if (bound == 0) {
            std::ostringstream msg;
            msg << "no model operation matches '" << key << "'. Declared operations:";
            for (const auto& [declared, unused] : this->slots_) msg << "\n  " << declared;
            throw std::runtime_error(msg.str());
        }
        this->hooks_.push_back(std::move(hook));
        return bound;
    }

private:
    struct slot {
        App* app;
        int index;
    };

    static bool _matches(std::string_view pattern, std::string_view key) {
        size_t p = 0, k = 0;
        while (p < pattern.size() && k < key.size()) {
            const size_t p_end = std::min(pattern.find('.', p), pattern.size());
            const size_t k_end = std::min(key.find('.', k), key.size());
            const std::string_view p_seg = pattern.substr(p, p_end - p);
            const std::string_view k_seg = key.substr(k, k_end - k);
            if (p_seg != "*" && p_seg != k_seg) return false;
            p = p_end + 1;
            k = k_end + 1;
        }
        return p >= pattern.size() && k >= key.size();
    }

    int site_count_;
    op_extent extent_;
    std::map<std::string, slot> slots_;
    std::vector<std::shared_ptr<op_override>> hooks_;
};

using op_registry = op_registry_t<npu_app>;

}  // namespace flm
