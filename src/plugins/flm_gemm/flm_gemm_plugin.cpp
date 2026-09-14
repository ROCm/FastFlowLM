/// \file flm_gemm_plugin.cpp
/// \brief Runs Gemma4 E2B's prefill projections on the IRON FLMGEMM operator.
///
/// A worked example of the override API. Everything here uses public headers
/// only: the operator's xclbin and instruction streams are compiled out of tree
/// by IRON, its weights are packed out of tree into safetensors sidecars, and
/// the plugin brings both with it. Nothing about the engine's own weight layout
/// is assumed, so the weight argument each projection is handed goes unused.
///
/// Artifacts expected next to the model's xclbins:
///   FLM_GEMM_<config>.xclbin
///   FLM_GEMM_M<M>_K<K>_N<N>_<config>[_epigelu].bin
/// and next to the model's weights:
///   model.dq_bfp        packed mlp weights
///   model.dq_bfp_attn   packed attention weights
///
/// Environment:
///   FLM_GEMM_CONFIG   operator configuration tag, default tn64_ma32_emf_floor_npu2
///   FLM_GEMM_OFF      set to leave every projection on the engine's own GEMM

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "flm_plugin.hpp"
#include "nlohmann/json.hpp"
#include "tensor_utils/safe_tensors.hpp"

namespace {

/// \brief Roles this plugin serves, in the order the per-layer table stores them.
constexpr std::array<std::string_view, 7> roles = {
    flm::role::q_proj, flm::role::k_proj, flm::role::v_proj, flm::role::o_proj,
    flm::role::gate_proj, flm::role::up_proj, flm::role::down_proj,
};

/// \brief Whether a role's weight matrix has the hidden size on its K side.
/// \note With one of the two dimensions known, the other follows from the packed
///       byte count, so the plugin never has to work out a layer's type.
constexpr bool k_is_hidden(size_t role_index) { return role_index != 3 && role_index != 6; }

constexpr bool wants_gelu(size_t role_index) { return role_index == 4; }

/// \brief bfp16ebs8 stores 8 values as a shared exponent plus 8 mantissa bytes.
constexpr size_t values_from_packed_bytes(size_t bytes) { return bytes / 9 * 8; }

/// \brief One instruction stream, identified by the problem it was compiled for.
struct shape_key {
    uint32_t m, k, n;
    bool gelu;
    bool operator<(const shape_key& o) const {
        return std::tie(m, k, n, gelu) < std::tie(o.m, o.k, o.n, o.gelu);
    }
};

class flm_gemm_override : public flm::op_override {
public:
    flm_gemm_override(const flm::plugin_context& ctx) : npu_(*ctx.npu) {
        this->artifact_dir_ = ctx.xclbin_path;
        const char* tag = std::getenv("FLM_GEMM_CONFIG");
        this->config_ = (tag != nullptr) ? tag : "tn64_ma32_emf_floor_npu2";
        this->_scan_instruction_streams();
        if (this->streams_.empty()) return;
        if (!this->_load_weights(ctx)) return;
        this->_create_apps();
    }

    /// \brief Bind every projection the plugin can serve, and the dequant steps
    ///        that feed them.
    /// \note A layer is taken whole or not at all: a layer whose projections are
    ///       split between two xclbins pays a context switch at every crossing,
    ///       which costs more than the operator saves. The same hook takes the
    ///       layer's dequant steps, so that a call it declines still finds its
    ///       dequantized weights where the engine's own GEMM expects them.
    size_t bind(flm::op_registry& ops, std::shared_ptr<flm::op_override> self) const {
        size_t bound = 0;
        for (size_t layer = 0; layer < this->layers_.size(); layer++) {
            if (this->layers_[layer].served_m.empty()) continue;
            for (size_t r = 0; r < roles.size(); r++) {
                if (!this->layers_[layer].slots[r].has_value()) continue;
                bound += ops.override_op(flm::op_key((int)layer, roles[r]), self);
            }
            for (std::string_view dq : { flm::role::dequant_qkv, flm::role::dequant_o,
                                         flm::role::dequant_gate, flm::role::dequant_up,
                                         flm::role::dequant_down }) {
                ops.override_op(flm::op_key((int)layer, dq), self);
            }
        }
        return bound;
    }

    flm::op_result create_run(const flm::op_call& call) override {
        layer_entry& layer = this->layers_[call.layer];
        const bool serves = layer.served_m.count(call.extent.padded) != 0;
        if (call.role.rfind("dequant.", 0) == 0) {
            // Nothing downstream reads what this would have produced.
            return serves ? flm::op_result() : flm::op_result::decline();
        }
        if (!serves) return flm::op_result::decline();
        const size_t r = this->_role_index(call.role);
        if (r == roles.size() || !layer.slots[r].has_value()) return flm::op_result::decline();
        slot& s = *layer.slots[r];
        npu_app& app = this->apps_.at(shape_key{ call.extent.padded, s.k, s.n, wants_gelu(r) });
        // IRON's argument order is A, B, C; the engine's is C, A, B.
        if (call.blocking) {
            app(*call.args[1], s.weights, *call.args[0]);
            return flm::op_result();
        }
        return flm::op_result(app.create_run(*call.args[1], s.weights, *call.args[0]));
    }

    bool ready() const { return !this->apps_.empty(); }

private:
    struct slot {
        buffer<u8> weights;
        uint32_t k, n;
    };
    struct layer_entry {
        std::array<std::optional<slot>, roles.size()> slots;
        std::set<uint32_t> served_m;  ///< M values every present role of this layer can run
    };

    size_t _role_index(std::string_view role) const {
        for (size_t i = 0; i < roles.size(); i++) {
            if (roles[i] == role) return i;
        }
        return roles.size();
    }

    std::string _stream_path(const shape_key& s) const {
        std::string name = "FLM_GEMM_M" + std::to_string(s.m) + "_K" + std::to_string(s.k)
                           + "_N" + std::to_string(s.n) + "_" + this->config_;
        if (s.gelu) name += "_epigelu";
        return (std::filesystem::path(this->artifact_dir_) / (name + ".bin")).string();
    }

    void _scan_instruction_streams() {
        this->xclbin_ = (std::filesystem::path(this->artifact_dir_)
                         / ("FLM_GEMM_" + this->config_ + ".xclbin")).string();
        if (!std::filesystem::exists(this->xclbin_)) return;
        const std::string prefix = "FLM_GEMM_M";
        const std::string suffix = "_" + this->config_;
        for (const auto& entry : std::filesystem::directory_iterator(this->artifact_dir_)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0 || entry.path().extension() != ".bin") continue;
            const std::string stem = entry.path().stem().string();
            const bool gelu = stem.size() > 8 && stem.compare(stem.size() - 8, 8, "_epigelu") == 0;
            const std::string body = stem.substr(0, stem.size() - (gelu ? 8 : 0));
            if (body.size() < suffix.size() || body.compare(body.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
            unsigned m = 0, k = 0, n = 0;
            if (std::sscanf(body.c_str(), "FLM_GEMM_M%u_K%u_N%u", &m, &k, &n) != 3) continue;
            this->streams_.insert(shape_key{ m, k, n, gelu });
        }
    }

    bool _load_weights(const flm::plugin_context& ctx) {
        const std::filesystem::path model(ctx.model_path);
        const std::string mlp_path = (model / "model.dq_bfp").string();
        const std::string attn_path = (model / "model.dq_bfp_attn").string();
        if (!std::filesystem::exists(mlp_path) || !std::filesystem::exists(attn_path)) return false;

        std::ifstream config_file((model / "config.json").string());
        if (!config_file.is_open()) return false;
        const uint32_t hidden = nlohmann::json::parse(config_file).value("hidden_size", 0u);
        if (hidden == 0) return false;

        SafeTensors mlp(mlp_path);
        SafeTensors attn(attn_path);
        this->app_manager_ = this->npu_.register_xclbin(this->xclbin_);

        size_t resident = 0;
        for (int layer = 0;; layer++) {
            layer_entry entry;
            bool any = false;
            std::optional<std::set<uint32_t>> served;
            for (size_t r = 0; r < roles.size(); r++) {
                SafeTensors& src = (r < 4) ? attn : mlp;
                const std::string name = "model.layers." + std::to_string(layer) + "."
                                         + std::string(roles[r]) + ".weight.dq_bfp";
                if (!src.has_tensor(name)) continue;
                const size_t bytes = src.get_tensor_metadata(name).byte_size;
                const uint32_t other = (uint32_t)(values_from_packed_bytes(bytes) / hidden);
                slot s{ this->app_manager_->create_bo_buffer<u8>(bytes),
                        k_is_hidden(r) ? hidden : other,
                        k_is_hidden(r) ? other : hidden };
                src.load_weights(s.weights, name);
                s.weights.sync_to_device();
                resident += bytes;

                std::set<uint32_t> ms;
                for (const auto& stream : this->streams_) {
                    if (stream.k == s.k && stream.n == s.n && stream.gelu == wants_gelu(r)) ms.insert(stream.m);
                }
                if (served.has_value()) {
                    std::set<uint32_t> both;
                    for (uint32_t m : ms) {
                        if (served->count(m)) both.insert(m);
                    }
                    served = both;
                } else {
                    served = ms;
                }
                entry.slots[r] = std::move(s);
                any = true;
            }
            if (!any) break;
            entry.served_m = served.value_or(std::set<uint32_t>{});
            this->layers_.push_back(std::move(entry));
        }
        if (this->layers_.empty()) return false;
        header_print("info", "FLMGEMM packed weights resident: "
                     + std::to_string(resident / (1024 * 1024)) + " MiB");
        return true;
    }

    void _create_apps() {
        for (const layer_entry& layer : this->layers_) {
            for (uint32_t m : layer.served_m) {
                for (size_t r = 0; r < roles.size(); r++) {
                    if (!layer.slots[r].has_value()) continue;
                    const shape_key key{ m, layer.slots[r]->k, layer.slots[r]->n, wants_gelu(r) };
                    if (this->apps_.count(key)) continue;
                    npu_app app = this->app_manager_->create_app();
                    app.load_insts(this->_stream_path(key));
                    this->apps_.emplace(key, std::move(app));
                }
            }
        }
    }

    npu_xclbin_manager& npu_;
    npu_app_manager* app_manager_ = nullptr;
    std::string artifact_dir_;
    std::string config_;
    std::string xclbin_;
    std::set<shape_key> streams_;
    std::vector<layer_entry> layers_;
    std::map<shape_key, npu_app> apps_;
};

void register_overrides(const flm::plugin_context& ctx) {
    if (std::getenv("FLM_GEMM_OFF") != nullptr) return;
    auto hook = std::make_shared<flm_gemm_override>(ctx);
    if (!hook->ready()) {
        header_print("warning", "FLMGEMM plugin idle: artifacts or packed weights missing");
        return;
    }
    const size_t bound = hook->bind(*ctx.ops, hook);
    if (bound == 0) {
        header_print("warning", "FLMGEMM plugin idle: no layer fully covered");
        return;
    }
    header_print_g("info", "FLMGEMM serving " + std::to_string(bound) + " projections");
}

}  // namespace

FLM_PLUGIN(register_overrides)
