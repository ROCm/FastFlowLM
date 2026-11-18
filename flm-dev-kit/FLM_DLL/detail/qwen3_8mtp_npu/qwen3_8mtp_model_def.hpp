// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_8mtp_model_def.hpp
/// \brief Shape definition for Qwen3.8-27B: the dimensions derived from
///        LM_Config, the layer taxonomy, and the MTP head's geometry.
///
/// No weight_desc_t / weight_container machinery: nothing here packs weights
/// into a device buffer, since the CPU backends read the mmap'd model.q4nx
/// block by block.
#ifndef __QWEN3_8MTP_MODEL_DEF_HPP__
#define __QWEN3_8MTP_MODEL_DEF_HPP__

#include "common/cpu_ref_bridge.hpp"
#include "common/rope.hpp"   // cpu_ref::ROPE_THETA, the rope_theta default

#include "lm_config.hpp"
#include "utils/utils.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/// \brief Dimensions + layer taxonomy for the Qwen3.8-27B hybrid stack.
struct qwen3_8mtp_desc {
    // ---- the stack --------------------------------------------------------
    uint32_t num_hidden_layers = 0;
    uint32_t D                 = 0;   ///< hidden_size
    uint32_t intermediate_size = 0;
    uint32_t vocab_size        = 0;
    float    rms_eps           = 0.0f;
    int      full_attention_interval = 4;

    /// The explicit 64-entry array out of config.json. Authoritative; the
    /// (i+1)%4==0 shorthand is only cross-checked against it.
    std::vector<std::string> layer_types;

    // ---- full attention ---------------------------------------------------
    uint32_t num_heads    = 0;
    uint32_t num_kv_heads = 0;
    uint32_t DH           = 0;   ///< head_dim, 256
    uint32_t DQ           = 0;   ///< num_heads * DH, 6144
    uint32_t DKV          = 0;   ///< num_kv_heads * DH, 1024
    uint32_t GQA_GROUP    = 0;   ///< num_heads / num_kv_heads, 6
    uint32_t RoPE_DIM     = 0;   ///< partial_rotary_factor * DH, 64
    float    rope_theta   = 0.0f;
    float    attn_scale   = 0.0f;

    // ---- gated delta net --------------------------------------------------
    /// The delta-net block has a geometry struct of its own in the CPU
    /// reference; it is built here so there is exactly one place that reads
    /// config.json, but the struct itself is cpu_ref's (README section 3.2:
    /// do not re-derive it).
    cpu_ref::delta_net_geometry delta;

    // ---- MTP draft head ---------------------------------------------------
    uint32_t mtp_num_hidden_layers = 0;   ///< 1
    bool     mtp_dedicated_embeddings = false;
    /// fc takes cat(norm_embed, norm_hidden), so its input is two hidden
    /// vectors wide. Getting this from D rather than from the tensor shape
    /// is deliberate: a mismatch should fail at load, not read out of bounds.
    uint32_t mtp_fc_in = 0;

    /// Where the quantised checkpoint lives. The CPU reference mmaps it
    /// directly rather than going through Q4NX::load_weights, which would
    /// materialise 17 GB of packed blocks in host memory.
    std::string q4nx_path;

    qwen3_8mtp_desc() {}
    explicit qwen3_8mtp_desc(LM_Config config) { build(config); }

    /// Layer `i` is full attention. Both sources must agree; build() throws
    /// if they do not, so by the time anyone calls this the answer is safe.
    inline bool is_attn_layer(int layer_idx) const {
        return (layer_idx % full_attention_interval) == (full_attention_interval - 1);
    }

    /// fp32 element counts, for sizing host buffers.
    inline size_t kv_cache_elems(uint32_t MAX_L) const {
        return (size_t)MAX_L * DKV * 2;                       // k and v
    }
    inline size_t delta_state_elems() const {
        return (size_t)delta.v_heads * delta.k_head_dim * delta.v_head_dim;
    }
    inline size_t conv_state_elems() const {
        return (size_t)delta.conv_k * delta.conv_dim();
    }

    inline uint32_t num_attn_layers() const {
        return num_hidden_layers / full_attention_interval;
    }
    inline uint32_t num_delta_layers() const {
        return num_hidden_layers - num_attn_layers();
    }

    void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D                 = config.get("hidden_size");
        intermediate_size = config.get("intermediate_size");
        vocab_size        = config.get("vocab_size");
        rms_eps           = config.get<f32>("rms_norm_eps", cpu_ref::q38::RMS_EPS);
        JSON_GET(full_attention_interval, config._json_config,
                 "full_attention_interval", 4, int);

        num_heads    = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads");
        DH           = config.get("head_dim");
        DQ           = num_heads * DH;
        DKV          = num_kv_heads * DH;
        GQA_GROUP    = num_kv_heads ? num_heads / num_kv_heads : 0;

        // partial_rotary_factor sits at the top level in this config and again
        // under rope_parameters; they agree, and the top-level one is the key
        // every other engine reads.
        const float prf = config.get<f32>("partial_rotary_factor", 0.25f);
        RoPE_DIM   = static_cast<uint32_t>(DH * prf);
        rope_theta = config.get<f32>("rope_theta", cpu_ref::ROPE_THETA);
        attn_scale = 1.0f / std::sqrt(static_cast<float>(DH));

        delta = cpu_ref::geometry_from(config);
        delta.validate();

        mtp_num_hidden_layers     = config.get("mtp_num_hidden_layers", u32(0));
        mtp_dedicated_embeddings  = config.get<bool>("mtp_use_dedicated_embeddings", false);
        mtp_fc_in                 = 2 * D;

        q4nx_path = cpu_ref::q4nx_path_of(config.model_path);

        _load_layer_types(config);
        _check_rope_parameters(config);
        _check_against_cpu_ref();

        DEBUG_BLOCK(1, std::cout << this->str(););
    }

    std::string str() const {
        std::ostringstream o;
        auto row = [&](const char* k, auto v) {
            o << std::left << std::setw(28) << k << " = " << v << "\n";
        };
        o << "================ qwen3_8mtp_desc ================\n";
        row("num_hidden_layers",       num_hidden_layers);
        row("  full attention",        num_attn_layers());
        row("  linear attention",      num_delta_layers());
        row("D (hidden_size)",         D);
        row("intermediate_size",       intermediate_size);
        row("vocab_size",              vocab_size);
        row("num_heads",               num_heads);
        row("num_kv_heads",            num_kv_heads);
        row("DH (head_dim)",           DH);
        row("DQ",                      DQ);
        row("DKV",                     DKV);
        row("GQA_GROUP",               GQA_GROUP);
        row("RoPE_DIM",                RoPE_DIM);
        row("rope_theta",              rope_theta);
        row("full_attention_interval", full_attention_interval);
        o << delta.str() << "\n";
        row("mtp_num_hidden_layers",   mtp_num_hidden_layers);
        row("mtp_dedicated_embed",     mtp_dedicated_embeddings ? "true" : "false");
        row("mtp_fc_in",               mtp_fc_in);
        row("q4nx_path",               q4nx_path);
        o << "==================================================\n";
        return o.str();
    }

private:
    /// config.json ships the taxonomy explicitly. Read it; synthesise from the
    /// interval only if the key is absent (it is present in this checkpoint).
    void _load_layer_types(LM_Config& config) {
        layer_types.clear();
        const nlohmann::json& jc = config._json_config;
        if (jc.contains("layer_types") && jc["layer_types"].is_array()) {
            for (const auto& e : jc["layer_types"])
                layer_types.push_back(e.get<std::string>());
        }
        else {
            for (uint32_t i = 0; i < num_hidden_layers; i++)
                layer_types.push_back(is_attn_layer((int)i) ? "full_attention"
                                                            : "linear_attention");
        }
        if (layer_types.size() != num_hidden_layers)
            throw std::runtime_error(
                "layer_types has " + std::to_string(layer_types.size()) +
                " entries but num_hidden_layers is " +
                std::to_string(num_hidden_layers));

        // The interval shorthand is what the layer loop, the cache sizing and
        // every future NPU sequence will actually branch on. If the shipped
        // array ever disagrees, every one of those is silently wrong, so fail
        // here rather than produce fluent garbage.
        for (uint32_t i = 0; i < num_hidden_layers; i++) {
            const bool full = layer_types[i] == "full_attention";
            if (!full && layer_types[i] != "linear_attention")
                throw std::runtime_error("unknown layer_type '" + layer_types[i] +
                                         "' at index " + std::to_string(i));
            if (full != is_attn_layer((int)i))
                throw std::runtime_error(
                    "layer_types[" + std::to_string(i) + "] = " + layer_types[i] +
                    " disagrees with (i+1)%" + std::to_string(full_attention_interval) +
                    "==0 -- the interval assumption is wrong for this checkpoint");
        }
    }

    /// config.json describes the m-rope slot->channel assignment twice, and the
    /// two descriptions are mutually exclusive: `mrope_interleaved: true` means
    /// frequency slot j takes channel j%3, while `mrope_section: [11,11,10]`
    /// would mean a contiguous split of the slots into t|h|w runs.
    /// common/rope.hpp implements the interleave and reads NEITHER key, so a
    /// checkpoint shipping the other convention would load clean and produce
    /// positions that are wrong from the first image onward -- which is not a
    /// crash, just drift. Cross-check both here instead.
    void _check_rope_parameters(LM_Config& config) const {
        const nlohmann::json& rp = config.sub("rope_parameters");
        if (rp.is_null() || rp.empty()) return;   // nothing to contradict

        if (rp.contains("mrope_interleaved") &&
            rp["mrope_interleaved"].is_boolean() &&
            !rp["mrope_interleaved"].get<bool>())
            throw std::runtime_error(
                "config rope_parameters.mrope_interleaved is false, but "
                "common/rope.hpp assigns frequency slot j to channel j%3 "
                "unconditionally -- a contiguous mrope_section split would need "
                "a different table, so this build cannot run this checkpoint");

        // Any rope_type other than `default` means a scaling law (yarn, linear,
        // dynamic) that mrope_table does not implement. Ignoring it silently is
        // a much larger drift than a wrong theta.
        if (rp.contains("rope_type") && rp["rope_type"].is_string()) {
            const std::string rt = rp["rope_type"].get<std::string>();
            if (rt != "default")
                throw std::runtime_error(
                    "config rope_parameters.rope_type = '" + rt + "' but "
                    "common/rope.hpp implements only the unscaled `default` "
                    "law -- this build cannot run this checkpoint");
        }

        // mrope_section is decorative under the interleave, but it still has to
        // sum to the number of frequency slots. If it does not, RoPE_DIM (which
        // came from partial_rotary_factor) and this array disagree about the
        // rotary width, and one of the two is wrong.
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array()) {
            uint32_t sum = 0;
            for (const auto& e : rp["mrope_section"]) sum += e.get<uint32_t>();
            if (sum != RoPE_DIM / 2)
                throw std::runtime_error(
                    "config rope_parameters.mrope_section sums to " +
                    std::to_string(sum) + " but partial_rotary_factor gives " +
                    std::to_string(RoPE_DIM) + " rotary dims = " +
                    std::to_string(RoPE_DIM / 2) + " frequency slots -- the two "
                    "disagree about the rotary width");
        }

        // partial_rotary_factor sits at the top level AND here. build() reads
        // the top-level one because that is the key every other engine reads;
        // if the two ever disagree, one of them is a lie.
        if (rp.contains("partial_rotary_factor")) {
            const float    prf = rp["partial_rotary_factor"].get<float>();
            const uint32_t rd  = static_cast<uint32_t>(DH * prf);
            if (rd != RoPE_DIM)
                throw std::runtime_error(
                    "config partial_rotary_factor gives rotary dim " +
                    std::to_string(RoPE_DIM) + " at the top level but " +
                    std::to_string(rd) + " under rope_parameters");
        }
    }

    /// cpu_ref_attention_layer.hpp hardcodes the attention geometry as q38::
    /// constants (it was ported from a standalone verifier). The delta-net
    /// side is runtime-configurable, the attention side is not, so a config
    /// that disagrees would be silently ignored by half the stack.
    void _check_against_cpu_ref() const {
        using namespace cpu_ref::q38;
        auto eq = [](const char* what, uint32_t got, int want) {
            if ((int)got != want)
                throw std::runtime_error(
                    std::string("config ") + what + " = " + std::to_string(got) +
                    " but cpu_ref_attention_layer.hpp is compiled for " +
                    std::to_string(want) + " -- the attention reference is not "
                    "runtime-configurable, so this build cannot run this checkpoint");
        };
        eq("hidden_size",          D,                 MODEL_DIM);
        eq("intermediate_size",    intermediate_size, INTERMEDIATE_SIZE);
        eq("num_attention_heads",  num_heads,         NUM_ATTN_HEADS);
        eq("num_key_value_heads",  num_kv_heads,      NUM_KV_HEADS);
        eq("head_dim",             DH,                HEAD_DIM);
        eq("rope dim",             RoPE_DIM,          ROPE_DIM);
        eq("full_attention_interval",
           (uint32_t)full_attention_interval,         FULL_ATTN_INTERVAL);

        // The three scalars. These are not shapes, so nothing downstream would
        // read out of bounds if they moved -- which is exactly why they need a
        // gate here:
        //
        //   rope_theta  has NO consumer. mrope_table() uses cpu_ref::ROPE_THETA,
        //               so a checkpoint that moved it would load clean and drift.
        //   rms_eps     is worse: the delta-net layers honour the config value,
        //               while the attention layers call rms_norm() without an eps
        //               and so take the q38::RMS_EPS default. A disagreement
        //               SPLITS the stack rather than being uniformly ignored.
        //   attn_scale  is derived from head_dim, which eq() already gates, so
        //               config cannot move it on its own. This line is a check on
        //               the CONSTANTS: it fires if ATTN_SCALE ever stops being
        //               1/sqrt(HEAD_DIM) because one of the two was edited alone.
        //
        // The tolerance is relative and tight. It exists to absorb the decimal
        // round trip through JSON, not to admit a different value -- a real
        // disagreement here is orders of magnitude, never one ulp.
        auto feq = [](const char* what, float got, float want) {
            const float scale = std::fabs(want) > 0.0f ? std::fabs(want) : 1.0f;
            if (std::fabs(got - want) > 1e-6f * scale) {
                std::ostringstream o;
                o << std::setprecision(9) << "config " << what << " = " << got
                  << " but this build is compiled for " << want
                  << " -- the value is read from config.json and then not used, "
                     "so honouring it would take a code change, not a reload";
                throw std::runtime_error(o.str());
            }
        };
        feq("rope_theta",                     rope_theta, cpu_ref::ROPE_THETA);
        feq("attn_scale (1/sqrt(head_dim))",  attn_scale, ATTN_SCALE);
        feq("rms_norm_eps",                   rms_eps,    RMS_EPS);
    }
};

#endif  // __QWEN3_8MTP_MODEL_DEF_HPP__
