// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <omp.h>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"

#include "weights_desc.hpp"
#include "q4k_layout.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Fused Q4_K dequant + matmul, Qwen3.8-27B prefill projection.
//
// The array is 4 rows x 8 columns of core pairs. One MT tile is
// NPU_M_TILE x NPU_K_TILE x NPU_N_TILE = 128 x 512 x 64, x broadcasts along a
// row and the weight broadcasts down a column, so one pass of the whole array
// covers 4*128 x 512 x 8*64 = 512 x 512 x 512.
//
// The weight is Q4_K in q4nx packing: a 32 x 256 block of 4736 B, parallel size
// 32 (one packed column holds the block's whole row span). See q4k_layout.hpp.
//
// The test drives it with layer 3's q_proj and scores the NPU output against a
// CPU reference computed from the *same* quantized bytes: scalar dequant, then
// GEMM in fp32. That makes Q4_K's own lossiness common-mode -- what is left is
// the kernel's error alone, so the bar is high (cos > 0.999) rather than Q4_K's
// ~0.997 floor against a bf16 checkpoint.
// ---------------------------------------------------------------------------

/// Row blocks are interleaved `vertical_blocks` deep because the memtile forwards
/// block i of every group to core row i.
///
/// The depth is NPU_N_TILE / QXNX_ROW_BLOCK_SIZE -- a weight row block is 32 rows
/// and one core column owns NPU_N_TILE of them -- so it tracks the n geometry, not
/// the number of core rows. At n = 64 that is 2; the n = 128 build wanted 4. It is
/// the same constant as the pY multiplier in mm_aie(), so the two must move together.
inline void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2)
{
    const int block_size = Q4K_BLOCK_BYTES;
    const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;

    // Catch a stale block size here: below, the truncating division would turn a
    // wrong stride into a plausible-looking row count instead of an obvious failure.
    assert(src.size() % ((size_t)block_size * blocks_per_row) == 0);
    const int rows = src.size() / block_size / blocks_per_row;
    assert(rows % vertical_blocks == 0);

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + (size_t)i * block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
        {
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], block_size);
                dst_ptr += block_size;
                src_ptr[i] += block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (size_t)(vertical_blocks - 1) * block_size * blocks_per_row;
        }
    }
}

/// One array pass of output rows. The MLP weave below runs at this grain.
constexpr int UPGATE_WEAVE_ROWS = 512;

/// \brief Interleave mlp.up_proj and mlp.gate_proj at the array-pass grain.
///
/// The two are **woven, not concatenated**: output rows
/// [2p*512, (2p+1)*512) come from up and [(2p+1)*512, (2p+2)*512) from gate,
/// both covering intermediate columns [p*512, (p+1)*512). See
/// gated_attention_verify/host/decoding_layer.cpp, which is where this layout
/// is defined -- the GLU tile consumes one engine round at a time and a round
/// has to be all up or all gate, so a plain concatenation would hand it all
/// 17408 up rows before the first gate row ever arrived.
///
/// The grain is 512 output rows = 16 of the engine's 32-row blocks, an even
/// number, so the weave never splits a `vertical_blocks` pair: weaving first
/// and then reordering the whole matrix is byte for byte what reordering each
/// 512-row chunk in turn produces. main() asserts exactly that rather than
/// taking the comment's word for it.
///
/// The result is a well-formed (2 * intermediate_size, Din) tensor in ordinary
/// file order, so it feeds reorder_cpy() and dequant_q4k_tensor_ref() unchanged
/// and the CPU reference needs no un-weaving.
inline void weave_upgate(u8 *dst, buffer<u8> &up, buffer<u8> &gate, const int Din)
{
    assert(up.size() == gate.size());
    const size_t blocks_per_row = (size_t)Din / QXNX_COL_BLOCK_SIZE;
    const size_t chunk = (size_t)(UPGATE_WEAVE_ROWS / QXNX_ROW_BLOCK_SIZE)
                       * blocks_per_row * Q4K_BLOCK_BYTES;
    assert(up.size() % chunk == 0 && "intermediate_size must be a whole number of weave chunks");

    const size_t phases = up.size() / chunk;
    u8 *p = dst;
    for (size_t i = 0; i < phases; i++) {
        memcpy(p, up.data()   + i * chunk, chunk);  p += chunk;
        memcpy(p, gate.data() + i * chunk, chunk);  p += chunk;
    }
}

/// \brief Time `Iterations` runs of one projection and report its throughput.
static void run_projection(npu_app &app, buffer<bf16> &y, buffer<bf16> &x, buffer<u8> &w,
                           const uint32_t L, const int Dout, const int Din, const char *label)
{
    header_print("info", label);
    time_utils::time_with_unit npu_time = {0.0, "us"};
    const int Iterations = 100;
    for (int i = 0; i < Iterations; i++) {
        time_utils::time_point start = time_utils::now();
        app(y, x, w);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    y.sync_from_device();
    npu_time.first /= Iterations;

    const double flops = 2.0 * (double)L * Din * Dout;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time: " << npu_time.first << " us");
    MSG_BOX_LINE(60, "throughput: " << flops / npu_time.first / 1e6 << " TFLOP/s");
    MSG_BONDLINE(60);
}

/// \brief y[L, Dout] = x[L, Din] * w[Dout, Din]^T, accumulated in fp32.
///
/// `w` is row major over (Dout, Din) -- the same orientation the projection has
/// on disk -- so a row of w is contiguous and the reduction is a dot product.
static void cpu_ref_matmul(buffer<bf16> &y, buffer<bf16> &x, buffer<bf16> &w,
                           const int L, const int Dout, const int Din)
{
    assert(y.size() >= (size_t)L * Dout);
    assert(x.size() >= (size_t)L * Din);
    assert(w.size() >= (size_t)Dout * Din);
    assert(Din % 16 == 0);

    const bf16 *xp = x.data();
    const bf16 *wp = w.data();
    bf16 *yp = y.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < L; i++) {
        for (int o = 0; o < Dout; o++) {
            const bf16 *xr = xp + (size_t)i * Din;
            const bf16 *wr = wp + (size_t)o * Din;
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < Din; d += 16) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i *)(xr + d)));
                __m512 wv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i *)(wr + d)));
                acc = _mm512_fmadd_ps(xv, wv, acc);
            }
            yp[(size_t)i * Dout + o] = (bf16)_mm512_reduce_add_ps(acc);
        }
    }
}

/// \brief Score an NPU projection against a CPU reference built from the same bytes.
///
/// `w_quant` is the *unreordered* q4nx tensor, so the reference is indexed by the
/// tensor's own (Dout, Din) and no un-permuting is needed.
///
/// \return true if the projection passed both the accuracy bar and the control.
static bool score_projection(buffer<bf16> &y_npu, buffer<bf16> &x, buffer<u8> &w_quant,
                             const uint32_t L, const int Dout, const int Din,
                             const char *label)
{
    header_print("info", "Computing CPU reference (scalar dequant + fp32 GEMM)");
    buffer<bf16> w_ref((size_t)Dout * Din);
    dequant_q4k_tensor_ref(w_ref, w_quant, Dout, Din);

    buffer<bf16> y_ref((size_t)L * Dout);
    cpu_ref_matmul(y_ref, x, w_ref, L, Dout, Din);

    error_metrics metrics = get_error_metrics(y_npu, y_ref);
    print_error_metrics(metrics, label);

    // Negative control. A cosine near 1 on its own does not prove the output
    // mapping: any permutation the array applied consistently to both the weight
    // rows and the y columns would agree with itself just as well. Rotating the
    // reference by one core column's worth of output (NPU_N_TILE = 64) has to
    // collapse the score -- if it does not, the metric is not reading the
    // mapping at all and the PASS above means nothing.
    bool control_ok = true;
    {
        constexpr int NPU_N_TILE = 64;
        buffer<bf16> y_rot((size_t)L * Dout);
        for (int i = 0; i < (int)L; i++) {
            for (int o = 0; o < Dout; o++) {
                y_rot[(size_t)i * Dout + o] = y_ref[(size_t)i * Dout + (o + NPU_N_TILE) % Dout];
            }
        }
        error_metrics ctl = get_error_metrics(y_npu, y_rot);
        header_print("info", "control (reference rotated by one core column): cos = "
                     << ctl.CosineSimilarity);
        if (ctl.CosineSimilarity > 0.5f) {
            header_print("error", "control did not separate -- the score is not reading the output mapping");
            control_ok = false;
        }
    }

    // Both sides consume identical quantized bytes, so the only divergence is
    // accumulation order and the bf16 rounding of the dequantized weight.
    const bool ok = metrics.CosineSimilarity > 0.999f && metrics.RelativeL2 < 0.05f
                    && control_ok;
    if (ok) {
        header_print("info", "PASS: cos = " << metrics.CosineSimilarity
                     << ", relL2 = " << metrics.RelativeL2);
    } else {
        header_print("error", "FAIL: cos = " << metrics.CosineSimilarity
                     << ", relL2 = " << metrics.RelativeL2);
    }
    return ok;
}

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string weight_path = "/path/to/models/Qwen3.8-27B-NPU2/";
    Q4NX weights(weight_path);

    //////////////////////////// model parameters ////////////////////////////
    // Qwen3.8-27B (config.json): hidden 5120, 24 heads x 256 head_dim.
    const int hidden_size         = 5120;
    const int num_attention_heads = 24;
    const int head_dim            = 256;

    const int intermediate_size   = 17408;

    const int D  = hidden_size;                      // 5120 = 10 K tiles
    const int DQ = head_dim * num_attention_heads;   // 6144 = 6 array passes
    const int DI = intermediate_size;                // 17408 = 34 array passes

    // One array pass is 512 x 512 x 512, so L must be a multiple of 512 and the
    // output width a multiple of 512.
    const uint32_t L_in = 1024;

    header_print("info", "D = " << D << ", DQ = " << DQ << ", DI = " << DI
                 << ", L = " << L_in);
    header_print("info", "array pass 512 x 512 x 512 (MT tile 128 x 512 x 64, 4 x 8 cores)");

    assert(L_in % 512 == 0 && "L must cover whole 4x128-row array passes");
    assert(DQ   % 512 == 0 && "Dout must cover whole 8x64-column array passes");
    assert(D    % 512 == 0 && "Din must be a whole number of K tiles");
    // The up|gate weave alternates every UPGATE_WEAVE_ROWS output rows, so the
    // intermediate size has to be a whole number of them or the last chunk of up
    // would be short and every gate row after it would land at the wrong offset.
    assert(DI % UPGATE_WEAVE_ROWS == 0 &&
           "intermediate_size must be a whole number of weave chunks");

    //////////////////////////// setup npu xclbin ////////////////////////////
    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    std::string xclbin_name = "build/xclbins/dequant_mm.xclbin";
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app q_proj = dequant_mm->create_app();
    gen_dequant_mm(q_proj.seq(), L_in, DQ, D, 0, 0);

    buffer<u8>   quant_q_weights = q_proj.create_bo_buffer<u8>(q4_k_size((size_t)D * DQ));
    buffer<bf16> buffer_x        = q_proj.create_bo_buffer<bf16>((size_t)L_in * D);
    buffer<bf16> buffer_q        = q_proj.create_bo_buffer<bf16>((size_t)L_in * DQ);

    //////////////////////////// weights ////////////////////////////
    // Layer 3 is a full-attention layer; its q_proj fuses q and the output gate
    // on the out dim (2 * DQ), q first. qwen38.py's `(g p h) -> (p g h)`
    // de-interleave is already applied in the file, so the q half is contiguous.
    const int layer_idx = 3;
    buffer<u8> w_q;     // owns the bytes
    buffer<u8> w_q_q;   // unreordered q half, kept for the CPU reference
    {
        char w_name[128];
        snprintf(w_name, sizeof(w_name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
        weights.load_weights(w_q, w_name);

        w_q_q = buffer<u8>(w_q.data(), w_q.size() / 2);
        assert(w_q_q.size() == quant_q_weights.size());
        reorder_cpy(quant_q_weights.data(), w_q_q, D);
    }
    quant_q_weights.sync_to_device();

    //////////////////////////// input ////////////////////////////
    // A deterministic pseudo-random activation. The reference below reduces the
    // very same bytes, so real activations would not make the comparison
    // stronger -- and unlike a real prefill they cannot hide a row mix-up behind
    // a near-constant input.
    {
        bf16 *xp = buffer_x.data();
        std::mt19937 rng(1234);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < buffer_x.size(); i++) xp[i] = (bf16)dist(rng);
    }
    buffer_x.sync_to_device();

    buffer_q.memset(bf16(-1.0f));
    buffer_q.sync_to_device();

    //////////////////////////// q_proj ////////////////////////////
    run_projection(q_proj, buffer_q, buffer_x, quant_q_weights, L_in, DQ, D,
                   "Running fused dequant + q projection on the NPU");

    header_print("DEBUG", "NPU output");
    utils::print_matrix(buffer_q, DQ);

    // Dequantize the same q4nx bytes scalar, then GEMM in fp32. Q4_K's own error
    // is common to both sides, so what the metrics show is the kernel's.
    const bool q_ok = score_projection(buffer_q, buffer_x, w_q_q, L_in, DQ, D,
                                       "q_proj vs CPU reference");

    //////////////////////////// up | gate ////////////////////////////
    // The MLP's two projections are not stored the way they are consumed: the
    // engine wants them woven 512 output rows at a time (see weave_upgate).
    // This is the layout decoding_layer.cpp builds, so testing it here is what
    // makes the weave itself falsifiable rather than assumed.
    //
    // Din is D and Dout is 2 * intermediate_size = 34816, so this is a
    // 5.7x larger weight than q_proj and 68 array passes wide.
    bool ug_ok = false;
    {
        const int DUG = 2 * DI;   // 34816 = 68 array passes

        npu_app up_gate = dequant_mm->create_app();
        gen_dequant_mm(up_gate.seq(), L_in, DUG, D, 0, 0);

        buffer<u8>   quant_ug_weights = up_gate.create_bo_buffer<u8>(q4_k_size((size_t)D * DUG));
        buffer<bf16> buffer_ug        = up_gate.create_bo_buffer<bf16>((size_t)L_in * DUG);

        header_print("info", "Loading and weaving mlp.up_proj | mlp.gate_proj");
        buffer<u8> w_up, w_gate;
        char w_name[128];
        snprintf(w_name, sizeof(w_name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
        weights.load_weights(w_up, w_name);
        snprintf(w_name, sizeof(w_name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
        weights.load_weights(w_gate, w_name);
        assert(w_up.size()   == q4_k_size((size_t)D * DI));
        assert(w_gate.size() == q4_k_size((size_t)D * DI));

        // The woven tensor, still in file order -- this is the CPU reference's
        // operand as well as reorder_cpy's input.
        buffer<u8> w_ug(q4_k_size((size_t)D * DUG));
        weave_upgate(w_ug.data(), w_up, w_gate, D);

        reorder_cpy(quant_ug_weights.data(), w_ug, D);

        // The claim in weave_upgate's comment is that the weave grain is an even
        // number of row-block pairs, so weaving then reordering equals reordering
        // each 512-row chunk in place. Check it instead of trusting the comment:
        // if it were false the NPU would still run, just on shuffled rows.
        {
            const size_t chunk = (size_t)(UPGATE_WEAVE_ROWS / QXNX_ROW_BLOCK_SIZE)
                               * ((size_t)D / QXNX_COL_BLOCK_SIZE) * Q4K_BLOCK_BYTES;
            buffer<u8> piecewise(quant_ug_weights.size());
            u8 *p = piecewise.data();
            for (size_t i = 0; i < w_ug.size() / chunk; i++) {
                buffer<u8> one(w_ug.data() + i * chunk, chunk);
                reorder_cpy(p, one, D);
                p += chunk;
            }
            const bool same = memcmp(piecewise.data(), quant_ug_weights.data(),
                                     quant_ug_weights.size()) == 0;
            header_print("info", "weave/reorder commute check: "
                         << (same ? "identical" : "DIFFER"));
            assert(same && "weave grain is not an even number of row-block pairs");
        }
        quant_ug_weights.sync_to_device();

        buffer_ug.memset(bf16(-1.0f));
        buffer_ug.sync_to_device();

        run_projection(up_gate, buffer_ug, buffer_x, quant_ug_weights, L_in, DUG, D,
                       "Running fused dequant + up|gate projection on the NPU");

        header_print("DEBUG", "NPU output (up|gate)");
        utils::print_matrix(buffer_ug, DUG);

        ug_ok = score_projection(buffer_ug, buffer_x, w_ug, L_in, DUG, D,
                                 "up|gate vs CPU reference");

        // The scores above would be just as high if the weave had been dropped
        // and the two tensors concatenated, *provided* the reference were built
        // the same wrong way -- it is the same self-consistency trap the rotation
        // control guards against. So score the NPU's woven output against a
        // plainly concatenated reference too: it must collapse.
        {
            buffer<u8> w_cat(q4_k_size((size_t)D * DUG));
            memcpy(w_cat.data(),                  w_up.data(),   w_up.size());
            memcpy(w_cat.data() + w_up.size(),    w_gate.data(), w_gate.size());

            buffer<bf16> w_cat_ref((size_t)DUG * D);
            dequant_q4k_tensor_ref(w_cat_ref, w_cat, DUG, D);
            buffer<bf16> y_cat((size_t)L_in * DUG);
            cpu_ref_matmul(y_cat, buffer_x, w_cat_ref, L_in, DUG, D);

            error_metrics ctl = get_error_metrics(buffer_ug, y_cat);
            header_print("info", "control (reference concatenated, not woven): cos = "
                         << ctl.CosineSimilarity);
            if (ctl.CosineSimilarity > 0.9f) {
                header_print("error", "concat control did not separate -- the weave is untested");
                ug_ok = false;
            }
        }
    }

    const bool ok = q_ok && ug_ok;
    header_print("info", "q_proj: " << (q_ok ? "PASS" : "FAIL")
                 << ", up|gate: " << (ug_ok ? "PASS" : "FAIL"));

    return ok ? 0 : 1;
}
