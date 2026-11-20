// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_op_factory.hpp
/// \brief Which backend each of the tower's ops gets.
///
/// Ported from Qwen3.8-27B's tower. `vision_tower.cpp` asks a factory for its
/// 115 matmuls and its one attention op and never learns what it got.
///
/// `make_attn` always returns the device op -- the attention xclbin is required
/// to construct this factory at all. `make_mm` returns the device op once
/// `enable_mm` has been given both `vision_mm` designs, and `cpu_mm_op`
/// otherwise; the engine always enables both, so a shipping tower is "npu" or it
/// does not start. The half-configured factory exists for the verify driver.
///
/// `name()` says which halves are live: "npu" or "npu-attn", so a half-ported
/// tower cannot be read as a full one from a log line.
///
/// The arenas -- the x/y staging the matmul ops of one design share -- belong
/// here because they must outlive every op, and the engine holds the factory
/// for the tower's lifetime.
#ifndef __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__
#define __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__

#include "image/npu/vision_attn_op.hpp"
#include "image/npu/vision_attn_reconfig.hpp"
#include "image/npu/vision_mm_op.hpp"
#include "image/vision_ops.hpp"

#include "npu_utils/npu_utils.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcv_npu {

class vision_op_factory final : public mcv::vision::op_factory {
public:
    /// \param attn_mgr manager for vision_attn.xclbin, owned by the model's
    ///        npu_xclbin_manager, which outlives the tower.
    explicit vision_op_factory(npu_app_manager* attn_mgr,
                               const vision_attn_rtp_addresses& rtp = vision_attn_rtp_addresses{})
        : attn_mgr_(attn_mgr), rtp_(rtp) {
        if (!attn_mgr_)
            throw std::runtime_error(
                "vision_op_factory: null vision_attn app manager -- construct "
                "cpu_op_factory instead when the xclbin is absent, so the off-arm "
                "of an A/B is the real host path");
    }

    /// Put the matmuls on the device too. Both designs together: A serves
    /// every shape but the projector's fc2 (N 1024), which only B can.
    /// \param mm_c optional vision_mm_c.xclbin (design C, 512-row tiles): an op whose N is a multiple of 384 runs on it
    ///        when the call is big enough (see npu_vision_mm_op); null keeps every matmul on A / B.
    /// \param mm_p optional vision_mm_p.xclbin (design P, C on packed bfp16 operands): when given, every op whose N
    ///        is a multiple of 384 runs on it for every M, and C goes unused.
    void enable_mm(npu_app_manager* mm_a, npu_app_manager* mm_b,
                   const vision_mm_rtp_addresses& mm_rtp = vision_mm_rtp_addresses{},
                   npu_app_manager* mm_c = nullptr, npu_app_manager* mm_p = nullptr) {
        if (!mm_a || !mm_b)
            throw std::runtime_error(
                "vision_op_factory::enable_mm: both vision_mm designs are required "
                "(A serves every shape but merger.fc2, whose N of 1024 is not a "
                "multiple of A's tile_N of 384)");
        arena_a_ = std::make_unique<vision_mm_arena>(mm_a, vision_mm_tiles::A, mm_rtp.a, lane_.get());
        arena_b_ = std::make_unique<vision_mm_arena>(mm_b, vision_mm_tiles::B, mm_rtp.b, lane_.get());
        if (mm_c)
            arena_c_ = std::make_unique<vision_mm_arena>(mm_c, vision_mm_tiles::C, mm_rtp.c, lane_.get());
        if (mm_p)
            arena_p_ = std::make_unique<vision_mm_arena>(mm_p, vision_mm_tiles::P, mm_rtp.p, lane_.get());
    }

    bool mm_on_device() const { return arena_a_ && arena_b_; }

    std::unique_ptr<mcv::vision::mm_op> make_mm(const char* tag) override {
        if (!mm_on_device()) return std::make_unique<mcv::vision::cpu_mm_op>();
        return std::make_unique<npu_vision_mm_op>(arena_a_.get(), arena_b_.get(), tag ? tag : "", arena_c_.get(),
                                                  arena_p_.get());
    }

    std::unique_ptr<mcv::vision::attn_op> make_attn(const char*) override {
        return std::make_unique<npu_vision_attn_op>(attn_mgr_, rtp_, sc_.get());
    }

#if FLM_MINICPM_RECONFIG_SUPPORTED
    /// FLM_MINICPM_V_VISION_SINGLE_CTX=1: let the attention dispatches run in vision_mm_a's context between two in-stream
    /// reconfigurations (see vision_attn_reconfig.hpp). Call after enable_mm(), before make_attn(). Builds the transactions
    /// and validates them with a known-answer test; any failure leaves the separate-context path in charge.
    /// \param dir the model's xclbin directory
    /// \param attn the attention xclbin attn_mgr was registered from
    void enable_single_context(bool preemption_enabled, const std::string& dir, const std::string& attn) {
        if (!vision_attn_reconfig::requested() || !mm_on_device()) return;
        sc_ = std::make_unique<vision_attn_reconfig>();
        sc_->build(preemption_enabled, arena_a_->mgr(), dir, attn);
        if (sc_->built) _self_test();
    }
#endif

    const char* name() const override { return mm_on_device() ? "npu" : "npu-attn"; }

private:
#if FLM_MINICPM_RECONFIG_SUPPORTED
    /// Known-answer test of the single-context attention on the real overlays, once at load, on an NPU this process has to itself:
    /// 512 deterministic rows (16 heads x 72) through vision_attn's own context (reference) and through the round trip -- 4 round trips,
    /// one of them twice in a row, and a two-group call (the array stays in the attention configuration between its two dispatches) --
    /// each followed by a design-A matmul of fixed data. Attention outputs and matmul outputs must be
    /// bit-identical to the references (same overlays, same sequences, deterministic arithmetic), so a replay that does not restore
    /// the array is caught. Any exception, timeout or mismatch disables the path for the process.
    void _self_test() {
        const std::string ver = minicpm_reconfig::versions();
        if (minicpm_reconfig::disabled()) return;
        if (!minicpm_reconfig::npu_exclusive()) {
            minicpm_reconfig::disable("another process has hw contexts on the NPU (" + ver + ")");
            return;
        }
        constexpr int LT = 512, H = 16, DH = 72, D = H * DH;   // D = 1152 = 3 x A's tile_N 384 and K
        npu_vision_attn_op op(attn_mgr_, rtp_, sc_.get());
        try {
            uint64_t st = 0xD1B54A32D192ED03ULL;
            auto rnd = [&st] { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return (float)((int)(st >> 40) & 0xFFFF) / 32768.0f - 1.0f; };
            std::vector<float> q((size_t)LT * D), k((size_t)LT * D), v((size_t)LT * D), x((size_t)LT * D);
            std::vector<float> o_ref((size_t)LT * D), o_sc((size_t)LT * D), y_ref((size_t)LT * D), y_sc((size_t)LT * D), o_ref2((size_t)LT * D);
            const std::vector<int> two_groups = {300, 212};   // two dispatches (different lengths) in one call
            std::vector<uint16_t> w((size_t)D * D);
            for (auto* a : {&q, &k, &v, &x}) for (float& f : *a) f = rnd();
            for (uint16_t& b : w) { const float f = 0.05f * rnd(); uint32_t u; std::memcpy(&u, &f, 4); b = (uint16_t)(u >> 16); }
            auto mm = make_mm("single_ctx_selftest");
            mm->bind(w.data(), nullptr, D, D, mcv::vision::activation::none);
            op.configure(LT, H, DH, 1.0f / std::sqrt((float)DH));
            auto same = [](const std::vector<float>& a, const std::vector<float>& b) { return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; };

            mm->run(x.data(), LT, y_ref.data());
            op.set_path_mode(npu_vision_attn_op::path_mode::CONTEXT_ONLY);
            op.run(q.data(), k.data(), v.data(), LT, o_ref.data());
            op.run_views(q.data(), k.data(), v.data(), D, two_groups, o_ref2.data());
            op.set_path_mode(npu_vision_attn_op::path_mode::SINGLE_CONTEXT_ONLY);
            const int trips[4] = {1, 2, 1, 1};
            int n_trips = 0;
            for (int it = 0; it < 4; it++) {
                for (int j = 0; j < trips[it]; j++) {
                    std::memset(o_sc.data(), 0xFF, o_sc.size() * sizeof(float));
                    op.run(q.data(), k.data(), v.data(), LT, o_sc.data());
                    n_trips++;
                    if (!same(o_sc, o_ref)) throw std::runtime_error("attention output after round trip " + std::to_string(n_trips) + " differs from the separate-context result");
                }
                if (it == 1) {
                    std::memset(o_sc.data(), 0xFF, o_sc.size() * sizeof(float));
                    op.run_views(q.data(), k.data(), v.data(), D, two_groups, o_sc.data());
                    n_trips++;
                    if (!same(o_sc, o_ref2)) throw std::runtime_error("two-group attention differs from the separate-context result");
                }
                std::memset(y_sc.data(), 0xFF, y_sc.size() * sizeof(float));
                mm->run(x.data(), LT, y_sc.data());
                if (!same(y_sc, y_ref)) throw std::runtime_error("design A matmul after round trip " + std::to_string(n_trips) + " differs from the reference (array state not restored)");
            }
            op.set_path_mode(npu_vision_attn_op::path_mode::CONTEXT_ONLY);
            std::memset(o_sc.data(), 0xFF, o_sc.size() * sizeof(float));
            op.run(q.data(), k.data(), v.data(), LT, o_sc.data());
            if (!same(o_sc, o_ref)) throw std::runtime_error("attention through vision_attn's context after the round trips differs from the reference");
            std::memset(o_sc.data(), 0xFF, o_sc.size() * sizeof(float));
            op.run_views(q.data(), k.data(), v.data(), D, two_groups, o_sc.data());
            if (!same(o_sc, o_ref2)) throw std::runtime_error("two-group attention through vision_attn's context after the round trips differs from the reference");
            sc_->validated = true;
            header_print_r("INFO", "minicpm_v: vision single-context reconfiguration self-test passed (" << n_trips << " round trips + matmul checks; " << ver << ")");
        } catch (const std::exception& e) {
            minicpm_reconfig::disable(std::string("vision self-test failed: ") + e.what() + " (" + ver + ")");
            // recovery: a dispatch in vision_attn's own context makes the firmware load that PDI, so the next matmul reloads design A's
            try {
                std::vector<float> z((size_t)LT * D, 0.0f), o((size_t)LT * D);
                op.set_path_mode(npu_vision_attn_op::path_mode::CONTEXT_ONLY);
                op.configure(LT, H, DH, 1.0f / std::sqrt((float)DH));
                op.run(z.data(), z.data(), z.data(), LT, o.data());
            } catch (const std::exception& e2) {
                header_print_r("WARNING", "minicpm_v: recovery dispatch after the failed vision self-test failed too: " << e2.what());
            }
        }
    }
#endif

    npu_app_manager*                 attn_mgr_ = nullptr;
    vision_attn_rtp_addresses        rtp_;
    // declared before the arenas: it must outlive everything that submits to it
    std::unique_ptr<dispatch_lane>   lane_ = std::make_unique<dispatch_lane>();
    std::unique_ptr<vision_mm_arena> arena_a_, arena_b_, arena_c_, arena_p_;
    // declared last: destroyed first (its chains hold apps of arena A's manager)
    std::unique_ptr<vision_attn_reconfig> sc_;
};

} // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__
