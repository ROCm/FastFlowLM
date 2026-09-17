/// \file op_registry.hpp
/// \brief The set of model operations that can be overridden, and the binding
///        of an override to them.
/// \note  A model engine declares its operations once, at construction. The key
///        space is therefore enumerable and fixed: override() on an unknown key
///        throws rather than silently never firing.
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

/// \brief Compose the key of a per-layer operation, e.g. "layers.12.mlp.up_proj".
inline std::string op_key(int layer, std::string_view role) {
    if (layer < 0) return std::string(role);
    return "layers." + std::to_string(layer) + "." + std::string(role);
}

/// \brief Roles of the operations a model may declare.
/// \note  A role names the operation, not the tensor it reads: dequant.qkv
///        produces one buffer from three tensors, and self_attn.core reads no
///        weight at all.
namespace role {
inline constexpr std::string_view q_proj    = "self_attn.q_proj";
inline constexpr std::string_view k_proj    = "self_attn.k_proj";
inline constexpr std::string_view v_proj    = "self_attn.v_proj";
inline constexpr std::string_view o_proj    = "self_attn.o_proj";
inline constexpr std::string_view attn_core = "self_attn.core";

inline constexpr std::string_view gate_proj = "mlp.gate_proj";
inline constexpr std::string_view up_proj   = "mlp.up_proj";
inline constexpr std::string_view down_proj = "mlp.down_proj";

inline constexpr std::string_view dequant_qkv  = "dequant.qkv";
inline constexpr std::string_view dequant_o    = "dequant.o";
inline constexpr std::string_view dequant_gate = "dequant.gate";
inline constexpr std::string_view dequant_up   = "dequant.up";
inline constexpr std::string_view dequant_down = "dequant.down";
}  // namespace role

template <typename App>
class op_registry_t {
public:
    explicit op_registry_t(int layer_count) : layer_count_(layer_count) {}

    /// \brief Declare that `app` implements `role` for `layer`.
    /// \note  Engine-side. Several layers may share one app, and several apps
    ///        may serve one role in different layers; both are resolved here so
    ///        that neither the apps nor an override have to know about it.
    void declare(std::string_view role, int layer, App& app) {
        app._declare_op(std::string(role), this->layer_count_, &this->extent_);
        this->slots_.emplace(op_key(layer, role), slot{ &app, layer });
    }

    /// \brief Publish the geometry of the chunk about to be processed.
    /// \note Engine-side, once per chunk. Every op_call of that chunk carries it.
    void set_extent(const op_extent& extent) { this->extent_ = extent; }

    /// \brief Every declared key, in lexicographic order.
    std::vector<std::string> list_ops() const {
        std::vector<std::string> keys;
        keys.reserve(this->slots_.size());
        for (const auto& [key, unused] : this->slots_) keys.push_back(key);
        return keys;
    }

    /// \brief Route `hook` to every operation matching `key`.
    /// \param key a declared key, or a pattern whose '*' matches one whole
    ///        dot-separated segment, e.g. "layers.*.mlp.up_proj".
    /// \return the number of operations bound.
    /// \throws std::runtime_error if `key` matches nothing.
    size_t override_op(std::string_view key, std::shared_ptr<op_override> hook) {
        size_t bound = 0;
        for (auto& [declared, s] : this->slots_) {
            if (!_matches(key, declared)) continue;
            s.app->_set_op_override(s.layer, hook.get());
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
        int layer;
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

    int layer_count_;
    op_extent extent_;
    std::map<std::string, slot> slots_;
    std::vector<std::shared_ptr<op_override>> hooks_;
};

using op_registry = op_registry_t<npu_app>;

}  // namespace flm
