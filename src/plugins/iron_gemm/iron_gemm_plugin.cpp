/// \file iron_gemm_plugin.cpp
/// \brief Runs Gemma4 E2B's prefill projections on IRON's GEMM and DequantBFP.
///
/// A worked example of the override API. Everything here uses public headers
/// only: both operators are compiled out of tree by IRON, and the plugin brings
/// their xclbins and instruction streams with it.
///
/// It keeps the engine's own structure -- weights are dequantized per layer per
/// prefill chunk into staging buffers, exactly where dequant.xclbin would have
/// run -- and replaces both halves of that pipeline. The weights stay 4-bit in
/// DRAM and no extra weight file is needed.
///
/// Artifacts expected next to the model's xclbins:
///   FLM_GEMM_<config>.xclbin
///   FLM_GEMM_<config>_M<M>_K<K>_N<N>[_epi<act>].bin
///   FLM_DequantBFP_<config>.xclbin
///   FLM_DequantBFP_K<K>_N<N>[_run<R>p<P>]_<config>.bin
/// The FLM_ prefixes are IRON's, from each operator's own artifact naming.
///
/// The dequant streams must be built with qw_layout=engine: they read the block
/// order the engine's loader writes to DRAM, so the operator consumes the
/// layer's weight buffer in place.
///
/// Environment:
///   IRON_GEMM_CONFIG    pick one GEMM xclbin by stem, when several are present
///   IRON_GEMM_MODE      dequant (default), bf16 or bfp16; where the weights come from
///   IRON_GEMM_OFF       set to leave every projection on the engine's own operators
///   IRON_DEQUANT_VERIFY compare each dequantized buffer against model.dq_bfp

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
#include "lm_config.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"
#include "modules/gemm.hpp"
#include "nlohmann/json.hpp"
#include "tensor_utils/safe_tensors.hpp"

namespace {

// The operations this plugin serves, in the order its per-layer table
// stores them. These are the sidecar's own weight-tensor names.
constexpr std::array<std::string_view, 7> names = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
};

constexpr size_t R_Q = 0, R_K = 1, R_V = 2, R_O = 3, R_GATE = 4, R_UP = 5, R_DOWN = 6;

// Whether a role's weight matrix has the hidden size on its K side.
constexpr bool k_is_hidden(size_t r) { return r != R_O && r != R_DOWN; }

constexpr bool wants_gelu(size_t r) { return r == R_GATE; }

// Out-features of one up-or-gate run in the engine's layer buffer.
constexpr uint32_t UPGATE_RUN = 512;

// Out-features per column block, from the GEMM's B tiling.
constexpr uint32_t N_TILE = 64;

// bfp16ebs8 packs 8 values into 9 bytes; q4nx holds 5 bits per weight.
constexpr size_t packed_bytes(uint32_t k, uint32_t n) { return (size_t)k * n / 8 * 9; }
constexpr size_t q4_bytes(uint32_t k, uint32_t n) { return (size_t)k * n * 5 / 8; }

// LM_Config without the executable-path lookup. from_pretrained() resolves
// exec_path through utils::find_xclbin_path, which lives in the flm binary
// rather than a shared library and so is not linkable from here. The plugin
// is handed the xclbin directory already, so it fills the paths itself.
struct plugin_config : LM_Config {
    void load(const std::string& model, const std::string& xclbin_dir) {
        this->model_path = model;
        this->model_name = std::filesystem::path(model).filename().string();
        // <exec>/xclbins/<model_name>
        this->exec_path = std::filesystem::path(xclbin_dir).parent_path().parent_path().string();
        this->_load_json();
        this->_normalize_multi_modal();
        this->flm_version = this->template get<std::string>("flm_version", "0.0.0");
    }
};

enum class weight_mode {
    dequant,  ///< 4-bit in DRAM, dequantized per layer per chunk on the device
    bfp16,    ///< packed bfp16ebs8 read once from a sidecar and held resident
    bf16,     ///< bf16 read once from a sidecar and held resident, for the shipped mm
};

inline weight_mode read_mode() {
    const char* m = std::getenv("IRON_GEMM_MODE");
    if (m == nullptr) return weight_mode::dequant;
    const std::string name(m);
    if (name == "bfp16") return weight_mode::bfp16;
    if (name == "bf16") return weight_mode::bf16;
    return weight_mode::dequant;
}

// One GEMM instruction stream, identified by the problem it was built for.
struct shape_key {
    uint32_t m, k, n;
    bool gelu;
    bool operator<(const shape_key& o) const {
        return std::tie(m, k, n, gelu) < std::tie(o.m, o.k, o.n, o.gelu);
    }
};

// IRON's DequantBFP, reading the engine's own per-layer weight buffer. One
// xclbin covers every shape -- K and N are runtime parameters -- so each
// shape costs only an instruction stream, which matters against a budget of
// 16 hardware contexts.
class device_dequant {
public:
    void setup(npu_xclbin_manager& npu, const std::string& artifact_dir) {
        std::string xclbin;
        std::vector<std::filesystem::path> streams;
        for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
            const std::string stem = entry.path().stem().string();
            if (stem.rfind("FLM_DequantBFP_", 0) != 0) continue;
            if (entry.path().extension() == ".xclbin" && stem.find("_K") == std::string::npos) {
                xclbin = entry.path().string();
            } else if (entry.path().extension() == ".bin") {
                streams.push_back(entry.path());
            }
        }
        if (xclbin.empty() || streams.empty()) return;

        this->mgr_ = npu.register_xclbin(xclbin);
        for (const auto& path : streams) {
            unsigned k = 0, n = 0, run = 0, period = 0;
            const std::string stem = path.stem().string();
            if (std::sscanf(stem.c_str(), "FLM_DequantBFP_K%u_N%u", &k, &n) != 2) continue;
            const size_t at = stem.find("_run");
            if (at != std::string::npos) std::sscanf(stem.c_str() + at, "_run%up%u", &run, &period);
            npu_app app = this->mgr_->create_app();
            app.load_insts(path.string());
            this->apps_.emplace(std::make_pair(k, n), shape{ std::move(app), run, period });
        }
    }

    bool has(uint32_t k, uint32_t n) const {
        return this->apps_.count(std::make_pair(k, n)) != 0;
    }

    size_t shapes() const { return this->apps_.size(); }

    // qw must be a view starting at the projection, not at the layer: the
    // operator reads from offset 0.
    void run(uint32_t k, uint32_t n, bytes& qw, bytes& packed) {
        this->apps_.at(std::make_pair(k, n)).app(qw, packed);
    }

    // Bytes the operator reads, spanning the gaps of an interleaved matrix.
    size_t reads(uint32_t k, uint32_t n) const {
        const shape& sh = this->apps_.at(std::make_pair(k, n));
        const size_t cb = q4_bytes(k, N_TILE);
        const size_t blocks = n / N_TILE;
        if (sh.run == 0) return blocks * cb;
        const size_t run = sh.run / N_TILE, period = sh.period / N_TILE, last = blocks - 1;
        return ((last / run) * period + last % run + 1) * cb;
    }

    npu_app_manager* manager() const { return this->mgr_; }

private:
    struct shape {
        npu_app app;
        uint32_t run, period;  ///< out-features per run and between runs; 0 when contiguous
    };

    std::map<std::pair<uint32_t, uint32_t>, shape> apps_;
    npu_app_manager* mgr_ = nullptr;
};

class iron_gemm_state {
public:
    // The four dequant steps a layer of one kind runs; qkv covers three
    // projections at once (see run_dequant()).
    enum dequant_matrix { D_QKV = 0, D_O, D_GATE, D_UP, D_DOWN };

    iron_gemm_state(const flm::plugin_context& ctx) : npu_(*ctx.npu) {
        this->artifact_dir_ = ctx.xclbin_path;
        const char* tag = std::getenv("IRON_GEMM_CONFIG");
        this->config_ = (tag != nullptr) ? tag : "tn64_ma32_emf_floor_npu2";
        this->mode_ = read_mode();
        if (this->mode_ == weight_mode::bf16) {
            if (!this->_setup_mm(ctx)) return;
        } else {
            this->_scan_instruction_streams();
            if (this->stream_files_.empty()) return;
            // A resident mode never dispatches the dequant, so it does not
            // register its xclbin either, leaving that hardware context free.
            if (this->mode_ == weight_mode::dequant) {
                this->dequant_.setup(this->npu_, this->artifact_dir_);
                if (this->dequant_.shapes() == 0) return;
            }
            this->app_manager_ = this->npu_.register_xclbin(this->xclbin_);
        }
        if (!this->_plan_layers(ctx)) return;
        if (this->mode_ == weight_mode::bf16) this->_load_resident();
        else if (this->mode_ == weight_mode::dequant) { this->_create_apps(); this->_allocate_staging(); }
        else { this->_create_apps(); this->_load_resident(); }
    }

    // Called by the projection hooks bound with a blocking signature
    // (everything but k/v, which pipeline against CPU rope/rms work).
    flm::hook_result<ert_cmd_state> run_proj_sync(size_t r, bytes& out, bytes& in, bytes& weights,
                                                  int64_t layer_idx, int64_t padded_arg) {
        const uint32_t padded = (uint32_t)padded_arg;
        layer_entry& layer = this->layers_[(size_t)layer_idx];
        const bool serves = (this->mode_ == weight_mode::bf16) ? layer.any_m : layer.served_m.count(padded) != 0;
        if (!serves || !layer.slots[r].has_value()) return flm::hook_result<ert_cmd_state>::defer();
        const slot& s = *layer.slots[r];
        bytes& b = this->_weights(layer, r);
        if (this->mode_ == weight_mode::bf16) return this->_mm_app(padded, s, r)(out, in, b);
        npu_app& app = this->apps_.at(shape_key{ padded, s.k, s.n, wants_gelu(r) });
        // IRON's argument order is A, B, C; the engine's is C, A, B.
        return app(in, b, out);
    }

    // Called by k/v's hooks, which pipeline this run against CPU rope/rms
    // work between start() and wait().
    flm::hook_result<xrt::run> run_proj_async(size_t r, bytes& out, bytes& in, bytes& weights,
                                              int64_t layer_idx, int64_t padded_arg) {
        const uint32_t padded = (uint32_t)padded_arg;
        layer_entry& layer = this->layers_[(size_t)layer_idx];
        const bool serves = (this->mode_ == weight_mode::bf16) ? layer.any_m : layer.served_m.count(padded) != 0;
        if (!serves || !layer.slots[r].has_value()) return flm::hook_result<xrt::run>::defer();
        const slot& s = *layer.slots[r];
        bytes& b = this->_weights(layer, r);
        if (this->mode_ == weight_mode::bf16) return this->_mm_app(padded, s, r).create_run(out, in, b);
        npu_app& app = this->apps_.at(shape_key{ padded, s.k, s.n, wants_gelu(r) });
        return app.create_run(in, b, out);
    }

    // Called by every dequant hook, told which matrix by which name it was
    // bound under (fixed at registration, see register_overrides() below).
    flm::hook_result<ert_cmd_state> run_dequant(dequant_matrix m, bytes& /*dequantized*/, bytes& quantized,
                                                int64_t layer_idx, int64_t /*padded*/) {
        if (this->mode_ != weight_mode::dequant) return flm::hook_result<ert_cmd_state>::defer();  // the weights are already there
        layer_entry& layer = this->layers_[(size_t)layer_idx];
        if (!this->_active(layer)) return flm::hook_result<ert_cmd_state>::defer();

        if (m == D_QKV && layer.qkv_n != 0) {
            // q, k and v are adjacent out-features, so one dispatch at their
            // combined width writes all three in the order the GEMMs read them.
            const slot& q = *layer.slots[R_Q];
            this->_run_one(quantized, layer, R_Q, q.k, layer.qkv_n, q.offset);
            if (this->verify_) for (size_t r : { R_Q, R_K, R_V }) this->_verify((int)layer_idx, r, layer);
            return ERT_CMD_STATE_COMPLETED;
        }
        std::array<size_t, 3> covered{};
        size_t count = 0;
        switch (m) {
            case D_QKV:  covered = { R_Q, R_K, R_V }; count = 3; break;
            case D_O:    covered[0] = R_O;    count = 1; break;
            case D_GATE: covered[0] = R_GATE; count = 1; break;
            case D_UP:   covered[0] = R_UP;   count = 1; break;
            case D_DOWN: covered[0] = R_DOWN; count = 1; break;
        }
        for (size_t i = 0; i < count; i++) {
            const size_t r = covered[i];
            if (!layer.slots[r].has_value()) continue;
            if (layer.skip && (r == R_K || r == R_V)) continue;
            const slot& s = *layer.slots[r];
            this->_run_one(quantized, layer, r, s.k, s.n, s.offset);
            if (this->verify_) this->_verify((int)layer_idx, r, layer);
        }
        return ERT_CMD_STATE_COMPLETED;
    }

    bool ready() const {
        return (this->mode_ == weight_mode::bf16) ? this->mm_mgr_ != nullptr
                                                  : !this->apps_.empty();
    }

private:
    struct slot {
        uint32_t k, n;
        size_t offset;  ///< where this projection starts in the layer's weight buffer
    };
    struct layer_entry {
        std::array<std::optional<slot>, names.size()> slots;
        std::array<std::optional<flm_rt::bo>, names.size()> views;  ///< cut on first dispatch
        // What each projection's GEMM reads, where that is not the whole
        // staging buffer. q, k and v share one, at their own offsets.
        std::array<std::optional<flm_rt::bo>, names.size()> b_bo;
        std::array<std::optional<buffer<u8>>, names.size()> b_view;
        std::array<buffer<u8>, names.size()> resident;  ///< filled once, in a resident mode
        std::set<uint32_t> served_m;  ///< M values every role of this layer can run
        bool any_m = false;           ///< the shipped kernel takes M at run time
        bool skip = false;            ///< the engine's buffer holds no k or v here
        uint32_t qkv_n = 0;           ///< combined width of one q, k, v dequant; 0 when q runs alone
    };

    // One app on the shipped mm overlay, with the M its sequence was built for.
    struct mm_app {
        npu_app app;
        uint32_t m = 0;
    };

    // Adopt the shipped mm overlay so the port's kernel can be taken out of
    // the comparison while the pre-dequantized weights stay in.
    // register_xclbin returns the engine's existing manager for the same
    // file, so this costs no hardware context.
    bool _setup_mm(const flm::plugin_context& ctx) {
        const std::string mm = (std::filesystem::path(this->artifact_dir_) / "mm.xclbin").string();
        if (!std::filesystem::exists(mm)) return false;
        this->mm_config_.load(ctx.model_path, this->artifact_dir_);
        this->mm_gemm_ = std::make_unique<Gemm>(this->mm_config_);
        this->mm_mgr_ = this->npu_.register_xclbin(mm);
        return this->mm_mgr_ != nullptr;
    }

    // The weight offset is always zero: the plugin holds a buffer per
    // projection, and the dequant layout's leading term is (n / 128) * 128 * K,
    // so one projection's slice is a prefix rather than something that has to
    // be indexed out of a combined buffer.
    npu_app& _mm_app(uint32_t padded, const slot& s, size_t r) {
        const auto key = std::make_tuple(s.k, s.n, wants_gelu(r));
        mm_app& ma = this->mm_apps_[key];
        if (ma.m != padded) {
            if (ma.m == 0) ma.app = this->mm_mgr_->create_app();
            this->mm_gemm_->generate_seq(ma.app.seq(), padded, s.k, s.n, 0, false,
                                         wants_gelu(r) ? Gemm::GeLU : Gemm::NO_Activation, 0);
            ma.m = padded;
        }
        return ma.app;
    }

    bool _active(const layer_entry& l) const { return !l.served_m.empty() || l.any_m; }

    // Whichever manager can allocate; any of them reaches the same device.
    npu_app_manager* _alloc_mgr() const {
        return this->app_manager_ != nullptr ? this->app_manager_ : this->mm_mgr_;
    }

    bytes& _weights(layer_entry& l, size_t r) {
        if (this->mode_ != weight_mode::dequant) return l.resident[r];
        if (l.b_view[r].has_value()) return *l.b_view[r];
        return this->staging_[r];
    }

    // Reads every projection's weights from the sidecar, once: the prefill
    // loop touches each weight exactly once per request in a fixed order, so
    // there is no reuse for a smaller cache to exploit.
    void _load_resident() {
        const char* suffix = (this->mode_ == weight_mode::bfp16) ? ".dq_bfp" : ".dq_bf16";
        size_t total = 0;
        for (size_t layer = 0; layer < this->layers_.size(); layer++) {
            layer_entry& l = this->layers_[layer];
            if (this->mode_ != weight_mode::bf16 && l.served_m.empty()) continue;
            bool complete = true;
            for (size_t r = 0; r < names.size(); r++) {
                if (!l.slots[r].has_value()) continue;
                if (l.skip && (r == R_K || r == R_V)) continue;
                SafeTensors* src = this->_sidecar(r);
                const std::string name = "model.layers." + std::to_string(layer) + "."
                                         + std::string(names[r]) + ".weight" + suffix;
                if (src == nullptr || !src->has_tensor(name)) { complete = false; continue; }
                const size_t bytes = src->get_tensor_metadata(name).byte_size;
                l.resident[r] = this->_alloc_mgr()->create_bo_buffer<u8>(bytes);
                src->load_weights(l.resident[r], name);
                l.resident[r].sync_to_device();
                total += bytes;
            }
            l.any_m = complete && this->mode_ == weight_mode::bf16;
        }
        header_print("info", "FLMGEMM resident weights: "
                     + std::to_string(total / (1024 * 1024)) + " MiB");
    }

    SafeTensors* _sidecar(size_t r) const {
        if (this->mode_ == weight_mode::bf16) return this->sc_mlp_.get();
        return (r <= R_O) ? this->sc_attn_.get() : this->sc_mlp_.get();
    }

    void _run_one(bytes& quantized, layer_entry& layer, size_t r,
                  uint32_t k, uint32_t n, size_t offset) {
        if (!layer.views[r].has_value()) {
            layer.views[r].emplace(quantized.bo(), this->dequant_.reads(k, n), offset);
        }
        buffer<u8> qw(*layer.views[r]);
        this->dequant_.run(k, n, qw, this->staging_[r]);
    }

    // Finds the GEMM xclbin and every instruction stream built against it.
    // The xclbin's stem is the configuration tag each stream is prefixed
    // with, so it is read off disk rather than spelled out here: the tag
    // names the tuning the operator was built at and gains a field whenever
    // that gains a knob. Only M, K and N are parsed out, and the file each
    // shape came from is kept rather than rebuilt.
    void _scan_instruction_streams() {
        const char* want = std::getenv("IRON_GEMM_CONFIG");
        for (const auto& entry : std::filesystem::directory_iterator(this->artifact_dir_)) {
            const std::string stem = entry.path().stem().string();
            if (entry.path().extension() != ".xclbin") continue;
            if (stem.rfind("FLM_GEMM_", 0) != 0) continue;
            // A stream's stem carries _M<digits>; a configuration's never does.
            if (stem.find("_M") != std::string::npos) continue;
            if (want != nullptr && stem != want) continue;
            this->xclbin_ = entry.path().string();
            this->config_ = stem;
        }
        if (this->xclbin_.empty()) return;

        for (const auto& entry : std::filesystem::directory_iterator(this->artifact_dir_)) {
            if (entry.path().extension() != ".bin") continue;
            const std::string stem = entry.path().stem().string();
            if (stem.rfind(this->config_, 0) != 0) continue;
            unsigned m = 0, k = 0, n = 0;
            if (std::sscanf(stem.c_str() + this->config_.size(), "_M%u_K%u_N%u", &m, &k, &n) != 3) continue;
            const shape_key key{ m, k, n, stem.find("_epigelu") != std::string::npos };
            this->stream_files_[key] = entry.path().string();
        }
    }

    // Refuses to serve a layer whose artifacts are incomplete. Without this a
    // missing shape silently shrinks coverage: the layer falls back to the
    // engine, the model stays correct, and the only symptom is that prefill
    // is slower than it should be. M is not checked here -- a chunk arriving
    // at a length nothing was built for is a run-time fallback, not a broken
    // build.
    void _require_artifacts(const layer_entry& l, size_t layer) const {
        const std::string where = "layers." + std::to_string(layer) + ".";
        auto fail = [&](const std::string& what, std::string_view role, uint32_t k, uint32_t n) {
            throw std::runtime_error("FLMGEMM: no " + what + " for " + where + std::string(role)
                                     + " (K=" + std::to_string(k) + " N=" + std::to_string(n)
                                     + "); rebuild the artifacts for this model's shapes");
        };
        for (size_t r = 0; r < names.size(); r++) {
            if (!l.slots[r].has_value()) continue;
            if (l.skip && (r == R_K || r == R_V)) continue;
            const slot& s = *l.slots[r];
            bool any = false;
            for (const auto& [stream, unused] : this->stream_files_) {
                any = any || (stream.k == s.k && stream.n == s.n && stream.gelu == wants_gelu(r));
            }
            if (!any) fail("GEMM instruction stream", names[r], s.k, s.n);
            if (this->mode_ != weight_mode::dequant) continue;
            const bool fused = l.qkv_n != 0 && (r == R_Q || r == R_K || r == R_V);
            if (fused) continue;
            if (!this->dequant_.has(s.k, s.n)) fail("dequant instruction stream", names[r], s.k, s.n);
        }
        if (this->mode_ == weight_mode::dequant && l.qkv_n != 0
            && !this->dequant_.has(l.slots[R_Q]->k, l.qkv_n)) {
            fail("dequant instruction stream", "self_attn.qkv", l.slots[R_Q]->k, l.qkv_n);
        }
    }

    // Works out each layer's shapes and where its projections sit. Shapes
    // come from the model's own weight metadata, so the plugin never has to
    // decide whether a layer is sliding-window or global. Only the
    // double-wide MLP has to be recognised, because that is what tells it
    // the layer's buffer holds no k or v.
    bool _plan_layers(const flm::plugin_context& ctx) {
        const std::filesystem::path model(ctx.model_path);
        const std::string q4_path = (model / "model.q4nx").string();
        if (!std::filesystem::exists(q4_path)) return false;

        std::ifstream config_file((model / "config.json").string());
        if (!config_file.is_open()) return false;
        const nlohmann::json cfg = nlohmann::json::parse(config_file);
        const uint32_t hidden = cfg.value("hidden_size", 0u);
        if (hidden == 0) return false;
        // The engine's loader omits k and v on the layers that share a kv cache,
        // and those are the last ones. Stated rather than inferred: E4B's MLP is
        // the same width on every layer, so there is nothing to infer it from.
        const int non_skip = (int)cfg.value("num_hidden_layers", 0u)
                             - (int)cfg.value("num_kv_shared_layers", 0u);

        SafeTensors q4(q4_path);
        for (int layer = 0;; layer++) {
            layer_entry entry;
            bool any = false;
            for (size_t r = 0; r < names.size(); r++) {
                const std::string name = "model.layers." + std::to_string(layer) + "."
                                         + std::string(names[r]) + ".weight";
                if (!q4.has_tensor(name)) continue;
                const size_t bytes = q4.get_tensor_metadata(name).byte_size;
                const uint32_t other = (uint32_t)(bytes * 8 / 5 / hidden);
                entry.slots[r] = slot{ k_is_hidden(r) ? hidden : other,
                                       k_is_hidden(r) ? other : hidden, 0 };
                any = true;
            }
            if (!any) break;
            this->layers_.push_back(std::move(entry));
        }
        if (this->layers_.empty()) return false;

        for (size_t i = 0; i < this->layers_.size(); i++) {
            layer_entry& l = this->layers_[i];
            l.skip = (int)i >= non_skip;
            // Where the buffer holds k and v, one dispatch covers all three.
            l.qkv_n = (l.skip || this->mode_ != weight_mode::dequant)
                          ? 0
                          : l.slots[R_Q]->n + l.slots[R_K]->n + l.slots[R_V]->n;
            if (l.qkv_n != 0 && !this->dequant_.has(l.slots[R_Q]->k, l.qkv_n)) l.qkv_n = 0;
            // Only once qkv_n has settled: whether q, k and v are dequantized
            // together decides which shapes the layer needs.
            if (this->mode_ != weight_mode::bf16) {
                this->_require_artifacts(l, (size_t)(&l - this->layers_.data()));
            }
            this->_place_projections(l);
            l.served_m = this->_served_m(l);
        }
        this->verify_ = std::getenv("IRON_DEQUANT_VERIFY") != nullptr;
        if (this->verify_ || this->mode_ != weight_mode::dequant) this->_open_sidecars(model);
        return true;
    }

    // Byte offset of each projection inside the engine's per-layer buffer.
    // Mirrors the order the engine's loader writes: q, then k and v where the
    // layer has them, o, then up and gate interleaved UPGATE_RUN out-features
    // at a time, then down.
    void _place_projections(layer_entry& l) const {
        const uint32_t d = l.slots[R_Q]->k;
        size_t at = 0;
        l.slots[R_Q]->offset = at; at += q4_bytes(d, l.slots[R_Q]->n);
        if (!l.skip) {
            l.slots[R_K]->offset = at; at += q4_bytes(d, l.slots[R_K]->n);
            l.slots[R_V]->offset = at; at += q4_bytes(d, l.slots[R_V]->n);
        }
        l.slots[R_O]->offset = at; at += q4_bytes(l.slots[R_O]->k, d);
        l.slots[R_UP]->offset = at;
        l.slots[R_GATE]->offset = at + q4_bytes(d, UPGATE_RUN);
        at += 2 * q4_bytes(d, l.slots[R_UP]->n);
        l.slots[R_DOWN]->offset = at;
    }

    // M values at which every one of a layer's projections can run.
    std::set<uint32_t> _served_m(const layer_entry& l) const {
        std::optional<std::set<uint32_t>> served;
        for (size_t r = 0; r < names.size(); r++) {
            if (!l.slots[r].has_value()) continue;
            if (l.skip && (r == R_K || r == R_V)) continue;
            const slot& s = *l.slots[r];
            const bool fused = l.qkv_n != 0 && (r == R_Q || r == R_K || r == R_V);
            if (this->mode_ == weight_mode::dequant && !fused
                && !this->dequant_.has(s.k, s.n)) return {};
            std::set<uint32_t> ms;
            for (const auto& [stream, unused] : this->stream_files_) {
                if (stream.k == s.k && stream.n == s.n && stream.gelu == wants_gelu(r)) ms.insert(stream.m);
            }
            if (!served.has_value()) {
                served = ms;
            } else {
                std::set<uint32_t> both;
                for (uint32_t m : ms) {
                    if (served->count(m)) both.insert(m);
                }
                served = both;
            }
        }
        return served.value_or(std::set<uint32_t>{});
    }

    void _create_apps() {
        for (const layer_entry& layer : this->layers_) {
            if (!this->_active(layer)) continue;
            for (uint32_t m : layer.served_m) {
                for (size_t r = 0; r < names.size(); r++) {
                    if (!layer.slots[r].has_value()) continue;
                    if (layer.skip && (r == R_K || r == R_V)) continue;
                    const shape_key key{ m, layer.slots[r]->k, layer.slots[r]->n, wants_gelu(r) };
                    if (this->apps_.count(key)) continue;
                    npu_app app = this->app_manager_->create_app();
                    app.load_insts(this->stream_files_.at(key));
                    this->apps_.emplace(key, std::move(app));
                }
            }
        }
    }

    // One packed buffer per role, reused by every layer, sized for the widest.
    void _allocate_staging() {
        size_t total = 0;
        for (size_t r = 0; r < names.size(); r++) {
            size_t widest = 0;
            for (const layer_entry& l : this->layers_) {
                if (!this->_active(l) || !l.slots[r].has_value()) continue;
                if (l.skip && (r == R_K || r == R_V)) continue;
                // A fused layer writes q, k and v into q's buffer.
                if (l.qkv_n != 0 && (r == R_K || r == R_V)) continue;
                const uint32_t n = (l.qkv_n != 0 && r == R_Q) ? l.qkv_n : l.slots[r]->n;
                widest = std::max(widest, packed_bytes(l.slots[r]->k, n));
            }
            if (widest == 0) continue;
            this->staging_[r] = this->dequant_.manager()->create_bo_buffer<u8>(widest);
            total += widest;
        }
        this->_cut_b_views();
        header_print("info", "FLMGEMM staging: " + std::to_string(total / (1024 * 1024))
                     + " MiB over " + std::to_string(this->dequant_.shapes()) + " dequant shapes");
    }

    // Points k and v at their share of the buffer q was dequantized into.
    // The packed order is column-block-major over N, so the three land one
    // after another and a sub-buffer is all the GEMM needs -- the same thing
    // the shipped mm expresses as a weight_offset. Every boundary is a whole
    // number of column blocks, which at these K is page aligned.
    void _cut_b_views() {
        for (layer_entry& l : this->layers_) {
            if (l.qkv_n == 0 || !this->_active(l)) continue;
            const uint32_t d = l.slots[R_Q]->k;
            size_t at = packed_bytes(d, l.slots[R_Q]->n);
            for (size_t r : { R_K, R_V }) {
                const size_t bytes = packed_bytes(d, l.slots[r]->n);
                l.b_bo[r].emplace(this->staging_[R_Q].bo(), bytes, at);
                l.b_view[r].emplace(*l.b_bo[r]);
                at += bytes;
            }
        }
    }

    void _open_sidecars(const std::filesystem::path& model) {
        if (this->mode_ == weight_mode::bf16) {
            const std::string all = (model / "model.dq_bf16").string();
            if (std::filesystem::exists(all)) this->sc_mlp_ = std::make_unique<SafeTensors>(all);
            return;
        }
        const std::string mlp = (model / "model.dq_bfp").string();
        const std::string attn = (model / "model.dq_bfp_attn").string();
        if (std::filesystem::exists(mlp)) this->sc_mlp_ = std::make_unique<SafeTensors>(mlp);
        if (std::filesystem::exists(attn)) this->sc_attn_ = std::make_unique<SafeTensors>(attn);
    }

    // For bringing a new shape up: a wrong offset or stride yields a buffer
    // of the right size holding real weight values in the wrong order, and
    // the model still generates fluent text.
    void _verify(int layer, size_t r, layer_entry& l) {
        const slot& s = *l.slots[r];
        bytes& produced = l.b_view[r].has_value() ? (bytes&)*l.b_view[r] : (bytes&)this->staging_[r];
        SafeTensors* src = this->_sidecar(r);
        if (src == nullptr) return;
        const std::string name = "model.layers." + std::to_string(layer) + "."
                                 + std::string(names[r]) + ".weight.dq_bfp";
        if (!src->has_tensor(name)) return;
        const size_t bytes = src->get_tensor_metadata(name).byte_size;
        std::vector<u8> expected(bytes);
        buffer<u8> view(expected.data(), bytes);
        src->load_weights(view, name);
        produced.sync_from_device();
        size_t bad = 0;
        for (size_t i = 0; i < bytes; i++) {
            if (produced.data()[i] != expected[i]) bad++;
        }
        const std::string what = "layer " + std::to_string(layer) + " " + std::string(names[r])
                                 + " K=" + std::to_string(s.k) + " N=" + std::to_string(s.n);
        if (bad) {
            header_print_r("ERROR", what + ": " + std::to_string(bad) + " of "
                           + std::to_string(bytes) + " bytes differ from the sidecar");
        } else {
            header_print("info", what + ": matches the sidecar");
        }
    }

    npu_xclbin_manager& npu_;
    npu_app_manager* app_manager_ = nullptr;
    std::string artifact_dir_;
    std::string config_;
    std::string xclbin_;
    std::map<shape_key, std::string> stream_files_;
    device_dequant dequant_;
    std::vector<layer_entry> layers_;
    std::map<shape_key, npu_app> apps_;
    std::array<buffer<u8>, names.size()> staging_;
    weight_mode mode_ = weight_mode::dequant;
    plugin_config mm_config_;
    std::unique_ptr<Gemm> mm_gemm_;
    npu_app_manager* mm_mgr_ = nullptr;
    std::map<std::tuple<uint32_t, uint32_t, bool>, mm_app> mm_apps_;
    bool verify_ = false;
    std::unique_ptr<SafeTensors> sc_mlp_, sc_attn_;
};

void register_overrides(const flm::plugin_context& ctx) {
    if (std::getenv("IRON_GEMM_OFF") != nullptr) return;
    auto state = std::make_shared<iron_gemm_state>(ctx);
    if (!state->ready()) {
        header_print("warning", "FLMGEMM plugin idle: artifacts missing");
        return;
    }

    flm::hook_registry& hooks = ctx.npu->hooks;
    const auto bind_sync = [&](std::string_view name, size_t role) {
        hooks.override_op<gemma4e_ops::proj_sig_t>(name,
            [state, role](bytes& out, bytes& in, bytes& w, int64_t layer, int64_t padded) {
                return state->run_proj_sync(role, out, in, w, layer, padded);
            });
    };
    const auto bind_async = [&](std::string_view name, size_t role) {
        hooks.override_op<gemma4e_ops::proj_async_sig_t>(name,
            [state, role](bytes& out, bytes& in, bytes& w, int64_t layer, int64_t padded) {
                return state->run_proj_async(role, out, in, w, layer, padded);
            });
    };
    bind_sync(gemma4e_ops::op::q_swa_proj, R_Q);
    bind_sync(gemma4e_ops::op::q_global_proj, R_Q);
    bind_async(gemma4e_ops::op::k_swa_proj, R_K);
    bind_async(gemma4e_ops::op::k_global_proj, R_K);
    bind_async(gemma4e_ops::op::v_swa_proj, R_V);
    bind_async(gemma4e_ops::op::v_global_proj, R_V);
    bind_sync(gemma4e_ops::op::o_swa_proj, R_O);
    bind_sync(gemma4e_ops::op::o_global_proj, R_O);
    bind_sync(gemma4e_ops::op::gate_proj, R_GATE);
    bind_sync(gemma4e_ops::op::gate_skip_proj, R_GATE);
    bind_sync(gemma4e_ops::op::up_proj, R_UP);
    bind_sync(gemma4e_ops::op::up_skip_proj, R_UP);
    bind_sync(gemma4e_ops::op::down_proj, R_DOWN);
    bind_sync(gemma4e_ops::op::down_skip_proj, R_DOWN);

    const auto bind_dequant = [&](const std::string_view* variants, iron_gemm_state::dequant_matrix m) {
        for (int t = 0; t < 4; t++) {
            hooks.override_op<gemma4e_ops::dequant_sig_t>(variants[t],
                [state, m](bytes& out, bytes& in, int64_t layer, int64_t padded) {
                    return state->run_dequant(m, out, in, layer, padded);
                });
        }
    };
    bind_dequant(gemma4e_ops::op::dequant_qkv, iron_gemm_state::D_QKV);
    bind_dequant(gemma4e_ops::op::dequant_o, iron_gemm_state::D_O);
    bind_dequant(gemma4e_ops::op::dequant_gate, iron_gemm_state::D_GATE);
    bind_dequant(gemma4e_ops::op::dequant_up, iron_gemm_state::D_UP);
    bind_dequant(gemma4e_ops::op::dequant_down, iron_gemm_state::D_DOWN);

    header_print_g("info", "FLMGEMM active");
}

}  // namespace

FLM_PLUGIN(register_overrides)
