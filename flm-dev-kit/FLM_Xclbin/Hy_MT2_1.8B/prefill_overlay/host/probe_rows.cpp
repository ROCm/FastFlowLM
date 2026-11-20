// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// WP12 probe: do rows the caller does not care about change the rows it does,
// in the prefill overlay's attention?
//   qpad   : q rows [L, La) random instead of zero (the padding rows of a piece)
//   kvtail : cache rows [L_begin + L, kv_rows) random instead of zero (stale rows
//            past the end of the prompt, as after restore())
// Each case runs the attention 4 times (base, qpad, kvtail, base again) and
// byte-compares the output rows [0, L).

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "npu_utils/npu_utils.hpp"
#include "utils/vm_args.hpp"
#include "hunyuan_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "hunyuan_overlay.hpp"

struct probe_case { uint32_t L_begin, L; };

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("xclbin", arg_utils::po::value<std::string>()->default_value("build/xclbins/prefill_overlay.xclbin"), "overlay xclbin");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    const int DH = 128, Heads = 16, KV_Heads = 4, DQ = Heads * DH, DK = KV_Heads * DH;
    const std::vector<probe_case> cases = {
        {1024, 476}, {1024, 480}, {1024, 512}, {0, 12}, {0, 16}, {512, 188}, {32, 37}, {32, 40}, {32, 5}, {256, 100},
    };
    uint32_t kv_rows = 0;
    for (const probe_case& c : cases) {
        const uint32_t La = (c.L + 127) / 128 * 128;
        kv_rows = std::max(kv_rows, ((c.L_begin + La + 127) / 128 + 2) * 128);
    }

    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* mgr = npu.register_xclbin(vm["xclbin"].as<std::string>());
    npu_app attn_app = mgr->create_app();
    int configured = -1;
    int bd_owner = -1;
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    buffer<bf16> q = npu.create_bo_buffer<bf16>((size_t)512 * DQ);
    buffer<bf16> o = npu.create_bo_buffer<bf16>((size_t)512 * DQ);
    buffer<bf16> cache = npu.create_bo_buffer<bf16>((size_t)kv_rows * (DK + DK));
    const size_t quadrant = cache.size() / 4;
    std::vector<bf16> q_real((size_t)512 * DQ), q_junk((size_t)512 * DQ), kv_real(cache.size()), kv_junk(cache.size());
    for (auto& v : q_real) v = bf16(nd(rng));
    for (auto& v : q_junk) v = bf16(4.0f * nd(rng));
    for (auto& v : kv_real) v = bf16(nd(rng));
    for (auto& v : kv_junk) v = bf16(8.0f * nd(rng));

    // fill q / cache for one run: real rows always, the rest zero or junk
    auto setup = [&](const probe_case& c, uint32_t La, bool q_junk_on, bool kv_junk_on) {
        for (uint32_t r = 0; r < La; r++) {
            const bool real = r < c.L;
            const bf16* src = real ? &q_real[(size_t)r * DQ] : &q_junk[(size_t)r * DQ];
            for (int e = 0; e < DQ; e++) q[(size_t)r * DQ + e] = (real || q_junk_on) ? src[e] : bf16(0.0f);
        }
        const uint32_t end = c.L_begin + c.L;
        for (int qd = 0; qd < 4; qd++)
            for (uint32_t r = 0; r < kv_rows; r++)
                for (int e = 0; e < DK / 2; e++) {
                    const size_t i = qd * quadrant + (size_t)r * (DK / 2) + e;
                    cache[i] = r < end ? kv_real[i] : (kv_junk_on ? kv_junk[i] : bf16(0.0f));
                }
        q.sync_to_device();
        cache.sync_to_device();
    };
    auto run = [&](const probe_case& c, uint32_t La) {
        hunyuan_overlay::gen_attn_run(attn_app.seq(), configured, c.L_begin, La, kv_rows, DH, Heads, KV_Heads, bd_owner);
        bd_owner = hunyuan_overlay::bd_owner_after(hunyuan_overlay::OP_ATTN, configured, bd_owner);
        configured = hunyuan_overlay::OP_ATTN;
        o.memset(bf16(0.0f));
        o.sync_to_device();
        attn_app(o, q, cache);
        o.sync_from_device();
        return std::vector<bf16>(o.data(), o.data() + (size_t)c.L * DQ);
    };
    // rows of [0, L) whose bytes differ, as "first-last" ranges
    auto diff_rows = [&](const std::vector<bf16>& a, const std::vector<bf16>& b, uint32_t L) {
        std::string s;
        int n = 0;
        int start = -1;
        for (uint32_t r = 0; r <= L; r++) {
            const bool d = r < L && memcmp(&a[(size_t)r * DQ], &b[(size_t)r * DQ], DQ * sizeof(bf16)) != 0;
            if (d) n++;
            if (d && start < 0) start = (int)r;
            if (!d && start >= 0) {
                s += " " + std::to_string(start) + (r - 1 > (uint32_t)start ? "-" + std::to_string(r - 1) : "");
                start = -1;
            }
        }
        return std::to_string(n) + " rows differ" + (n ? ":" + s : "");
    };

    for (const probe_case& c : cases) {
        const uint32_t La = (c.L + 127) / 128 * 128;
        setup(c, La, false, false);
        std::vector<bf16> base = run(c, La);
        setup(c, La, true, false);
        std::vector<bf16> qpad = run(c, La);
        setup(c, La, false, true);
        std::vector<bf16> kvtail = run(c, La);
        setup(c, La, false, false);
        std::vector<bf16> again = run(c, La);
        std::cout << "L_begin " << std::setw(5) << c.L_begin << "  L " << std::setw(4) << c.L << "  La " << std::setw(4) << La
                  << " (end " << c.L_begin + c.L << ", end % 8 = " << (c.L_begin + c.L) % 8 << ", end % 16 = " << (c.L_begin + c.L) % 16 << ")"
                  << "\n    base vs base again : " << diff_rows(base, again, c.L)
                  << "\n    q padding junk     : " << diff_rows(base, qpad, c.L)
                  << "\n    cache tail junk    : " << diff_rows(base, kvtail, c.L) << std::endl;
    }
    return 0;
}
