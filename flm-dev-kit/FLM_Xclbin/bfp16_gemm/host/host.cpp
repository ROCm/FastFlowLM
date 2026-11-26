// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// bfp16 GEMM test: random bf16 X / W / bias, packed to bfp16 on the host, Y = act(X W^T + b) on the NPU,
// checked against an fp32 product of the packed (decoded) operands, and timed.
//   ./run.exe --M 1024 --K 1024 --N 4096 [--act 1] [--iters 20]
// A run that does not complete in 10 s leaves ~/.npu_suspect (or $NPU_SUSPECT_FILE) and holds the
// hardware context 90 s, so the driver's job timeout can recover the device before the process exits.

#include <boost/program_options.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "pack_bfp16.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;
using namespace pack_bfp16;

// x: block (row / 16, kc), then [16 k tiles][2 row halves][8 rows][9 B]
static void pack_x(uint8_t* dst, const std::vector<uint16_t>& x, uint32_t M, uint32_t m_pad, uint32_t K) {
    const uint32_t KC = K / 128;
#pragma omp parallel for
    for (uint32_t r = 0; r < m_pad; r++)
        for (uint32_t k8 = 0; k8 < K / 8; k8++) {
            const uint32_t kc = k8 / 16, kt = k8 % 16;
            uint8_t* o = dst + (((size_t)(r / 16) * KC + kc) * bfp16_gemm::X_BLOCK_WORDS * 4) + kt * 144 + ((r % 16) / 8) * 72 + (r % 8) * 9;
            if (r < M) encode_block(o, &x[(size_t)r * K + 8 * k8]);
            else std::memset(o, 0, 9);
        }
}

// w: chunk (nb, column, kc, half) = [4 jp][16 k tiles][2][8 columns][9 B], then the half's 64 bf16 biases
static void pack_w(uint8_t* dst, const std::vector<uint16_t>& w, const std::vector<uint16_t>& bias, uint32_t N, uint32_t K) {
    const uint32_t KC = K / 128;
    const size_t half_bytes = bfp16_gemm::W_HALF_WORDS * 4;
#pragma omp parallel for
    for (uint32_t n = 0; n < N; n++) {
        const uint32_t nb = n / 1024, col = (n / 128) % 8, h = (n / 64) % 2, jp = (n / 16) % 4, j = (n / 8) % 2, c8 = n % 8;
        for (uint32_t kc = 0; kc < KC; kc++) {
            uint8_t* chunk = dst + ((((size_t)nb * 8 + col) * KC + kc) * 2 + h) * half_bytes;
            for (uint32_t kt = 0; kt < 16; kt++)
                encode_block(chunk + ((jp * 16 + kt) * 2 + j) * 72 + c8 * 9, &w[(size_t)n * K + kc * 128 + kt * 8]);
            std::memcpy(chunk + 9216 + 2 * (n % 64), &bias[n], 2);
        }
    }
}

// the values the cores multiply: 8 x int8 mantissa x 2^(shared exponent - 133)
static void decode_rows(std::vector<float>& dst, const std::vector<uint16_t>& src, size_t rows, uint32_t K) {
    dst.resize(rows * K);
#pragma omp parallel for
    for (size_t r = 0; r < rows; r++)
        for (uint32_t k8 = 0; k8 < K / 8; k8++) {
            uint8_t blk[9];
            encode_block(blk, &src[r * K + 8 * k8]);
            for (int i = 0; i < 8; i++) dst[r * K + 8 * k8 + i] = std::ldexp((float)(int8_t)blk[1 + i], (int)blk[0] - 133);
        }
}

static float bf16r(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// getGeluBf16_nonLUT, step by step in bf16
static float gelu_bf16(float x) {
    const float x2 = bf16r(x * x), x3 = bf16r(x * x2), inner = bf16r(x + bf16r(x3 * bf16r(0.044715f)));
    const float t = bf16r(std::tanh(inner * bf16r(0.79788456f)));
    return bf16r(x * bf16r(0.5f * bf16r(t + 1.0f)));
}

int main(int argc, char** argv) {
    uint32_t M, K, N, act, iters, seed, xcache, reuse, stream, hybrid;
    std::string xclbin;
    po::options_description desc("bfp16 GEMM test");
    desc.add_options()("help,h", "help")
        ("M", po::value<uint32_t>(&M)->default_value(256), "rows (padded to 256)")
        ("K", po::value<uint32_t>(&K)->default_value(1024), "inner size, multiple of 128 (above 3072 or with --stream: of 512)")
        ("stream", po::value<uint32_t>(&stream)->default_value(0), "1: stream the weights even if they fit")
        ("hybrid", po::value<uint32_t>(&hybrid)->default_value(1), "K <= 1536: 0 waits for the whole weight panel")
        ("N", po::value<uint32_t>(&N)->default_value(1024), "output columns, multiple of 1024")
        ("act", po::value<uint32_t>(&act)->default_value(0), "0 none, 1 GELU")
        ("iters", po::value<uint32_t>(&iters)->default_value(1), "timed calls")
        ("seed", po::value<uint32_t>(&seed)->default_value(0), "rng seed")
        ("xcache", po::value<uint32_t>(&xcache)->default_value(1), "x read cache hint: 0 no, 1 normal, 2 aggressive")
        ("reuse", po::value<uint32_t>(&reuse)->default_value(0), "1: one run object, started again for every call")
        ("xclbin", po::value<std::string>(&xclbin)->default_value("build/xclbins/gemm.xclbin"), "xclbin");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
    const uint32_t m_pad = (M + 255) / 256 * 256, R = m_pad / 256;
    bfp16_gemm::call_t call{R, K / 128, N / 1024, N, act};
    call.stream = stream != 0;
    call.hybrid = hybrid != 0;
    const bool streamed = bfp16_gemm::streams(call);
    if (K % 128 || N % 1024 || R > bfp16_gemm::MAX_ROUNDS || act > 1 ||
        (streamed && (K % 512 || call.kc * R > bfp16_gemm::MAX_QUEUE_REPEAT))) {
        std::cerr << "unsupported shape\n";
        return 2;
    }
    std::printf("M %u (pad %u, %u rounds)  K %u  N %u  act %u  iters %u  weights %s\n", M, m_pad, R, K, N, act, iters,
                streamed ? "streamed" : bfp16_gemm::hybrids(call) ? "resident (hybrid)" : "resident");

    std::vector<uint16_t> x((size_t)M * K), w((size_t)N * K), bias(N);
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dx(0.0f, 1.0f), dw(0.0f, 0.03f), db(0.0f, 0.5f);
        for (auto& v : x) v = f32_to_bf16(dx(rng));
        for (auto& v : w) v = f32_to_bf16(dw(rng));
        for (auto& v : bias) v = f32_to_bf16(db(rng));
    }

    xrt::device device(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &device, false);
    npu_app_manager* gemm_mgr = mgr.register_xclbin(xclbin);
    npu_app app = gemm_mgr->create_app();
    call.x_cache = xcache == 0 ? no_cache : xcache == 1 ? normal_cache : aggressive_cache;
    bfp16_gemm::gen_gemm(app.seq(), call);

    buffer<u8> bx = app.create_bo_buffer<u8>(bfp16_gemm::x_bytes(m_pad, K));
    buffer<u8> bw = app.create_bo_buffer<u8>(bfp16_gemm::w_bytes(N, K));
    buffer<u16> by = app.create_bo_buffer<u16>((size_t)m_pad * N);
    pack_x(bx.data(), x, M, m_pad, K);
    pack_w(bw.data(), w, bias, N, K);
    std::fill(by.data(), by.data() + by.size(), (uint16_t)0x7FC0);   // NaN: shows rows / columns never written
    bx.sync_to_device();
    bw.sync_to_device();
    by.sync_to_device();

    xrt::run kept = app.create_run(by, bx, bw);
    auto run_once = [&]() -> bool {
        xrt::run run = reuse ? kept : app.create_run(by, bx, bw);
        run.start();
        const ert_cmd_state st = run.wait(std::chrono::milliseconds(10000));
        if (st == ERT_CMD_STATE_COMPLETED) return true;
        const char* f = std::getenv("NPU_SUSPECT_FILE");
        const std::string path = f ? f : std::string(std::getenv("HOME")) + "/.npu_suspect";
        std::ofstream(path) << "bfp16_gemm run.exe: run state " << (int)st << " (M " << M << " K " << K << " N " << N << ")\n";
        std::cerr << "run did not complete (state " << (int)st << "); wrote " << path << ", holding the context 90 s\n";
        std::this_thread::sleep_for(std::chrono::seconds(90));
        return false;
    };

    std::vector<double> us;
    for (uint32_t i = 0; i < std::max(iters, 1u); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run_once()) return 3;
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    by.sync_from_device();

    // reference on the operands the cores see
    std::vector<float> xd, wd;
    decode_rows(xd, x, M, K);
    decode_rows(wd, w, N, K);
    size_t bad = 0, nan = 0, exact = 0;
    double se = 0, sr = 0, max_err = 0;
#pragma omp parallel for reduction(+ : bad, nan, exact, se, sr) reduction(max : max_err)
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0;
            const float* a = &xd[(size_t)r * K];
            const float* b = &wd[(size_t)n * K];
            for (uint32_t k = 0; k < K; k++) acc += a[k] * b[k];
            float ref = bf16r(acc + bf16_to_f32(bias[n]));
            if (act == 1) ref = gelu_bf16(ref);
            const uint16_t gb = by.data()[(size_t)r * N + n];
            if ((gb & 0x7FFF) > 0x7F80) { nan++; continue; }   // bit test: -ffast-math drops isnan
            const float got = bf16_to_f32(gb);
            const double e = std::fabs((double)got - ref);
            if (got == ref) exact++;
            if (r == M - 1 && n >= N - 4)
#pragma omp critical
                std::printf("  y[%u][%u] npu %.6g ref %.6g\n", r, n, got, ref);
            se += e * e;
            sr += (double)ref * ref;
            max_err = std::max(max_err, e);
            if (e > 0.02 * std::fabs(ref) + 0.02) bad++;
        }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    const double flop = 2.0 * m_pad * N * K;
    std::printf("NPU: min %.1f us  median %.1f us  -> %.2f TFLOPS (padded rows), %.2f (real rows)\n", us[0], med,
                flop / med * 1e-6, flop * M / m_pad / med * 1e-6);
    std::printf("vs fp32 of packed operands: rel L2 %.3e  max abs %.3e  bit-exact %.2f%%  out of tolerance %zu  unwritten %zu\n",
                std::sqrt(se / std::max(sr, 1e-30)), max_err, 100.0 * exact / ((double)M * N), bad, nan);
    const bool pass = bad == 0 && nan == 0;
    std::printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
