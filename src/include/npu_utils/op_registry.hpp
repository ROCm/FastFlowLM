// The set of model operations that can be overridden, and the binding of an
// override to them. A model engine declares its operations once, at
// construction, so the set of names is fixed: override_op() on an unknown
// name throws rather than silently never firing.
//
// A name identifies one operation across the whole model, not one dispatch
// site: "self_attn.q_proj" covers every layer's q_proj. A hook that needs
// to tell layers apart reads that from op_call::args, same as any other
// argument -- the registry itself has no notion of layers, indices, or
// anything else model-specific.
#pragma once

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "op_override.hpp"

namespace flm {

class op_registry {
public:
    // Declares that this engine may dispatch a hook under `name`. Call for
    // every name an override might later bind to, before any override_op().
    void declare(std::string name) { this->slots_.emplace(std::move(name), nullptr); }

    // Binds hook to `name`. Throws std::runtime_error if `name` was never
    // declared, or already has a hook bound.
    void override_op(std::string_view name, std::shared_ptr<op_override> hook) {
        auto it = this->slots_.find(std::string(name));
        if (it == this->slots_.end()) {
            std::ostringstream msg;
            msg << "no model operation named '" << name << "'. Declared operations:";
            for (const auto& [declared, unused] : this->slots_) msg << "\n  " << declared;
            throw std::runtime_error(msg.str());
        }
        if (it->second != nullptr) {
            throw std::runtime_error("'" + std::string(name) + "' already has an override bound");
        }
        it->second = hook.get();
        this->hooks_.push_back(std::move(hook));
    }

    // The hook bound to `name`, or nullptr. Call once per operation, after
    // every plugin has registered (see flm_plugin.hpp / causal_lm's
    // resolve_overrides()), and keep the result rather than calling this
    // again on a hot path.
    op_override* resolve(std::string_view name) const {
        auto it = this->slots_.find(std::string(name));
        return it != this->slots_.end() ? it->second : nullptr;
    }

    std::vector<std::string> list_ops() const {
        std::vector<std::string> names;
        names.reserve(this->slots_.size());
        for (const auto& [name, unused] : this->slots_) names.push_back(name);
        return names;
    }

private:
    std::map<std::string, op_override*> slots_;
    std::vector<std::shared_ptr<op_override>> hooks_;
};

}  // namespace flm
