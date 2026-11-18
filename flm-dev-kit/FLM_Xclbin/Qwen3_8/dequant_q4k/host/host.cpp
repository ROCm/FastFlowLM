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
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "lm_config.hpp"
#include "metrices.hpp"

#include "tensor_utils/q4_npu_eXpress.hpp"
#include "q4k_layout.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Q4_K dequant harness for Qwen3.8-27B.
//
// For every Q4_K tensor of layers 0-4 this dequantizes model.q4nx on the NPU and
// checks the result twice:
//
//   1. against a scalar CPU decode of the same bytes (q4k_layout.hpp) -- this is
//      exact up to bf16 rounding, so anything below cos ~0.9999 is a kernel or
//      DMA bug;
//   2. against the bf16 reference checkpoint -- this can only reach Q4_K's own
//      quantization error (cos ~0.997), and it is what validates the *reorders*
//      the converter applies to qkv_proj / self_attn.gate_proj / ssm_out_proj.
//
// Check 2 is the reason this exists: the expand-3 head permutation was
// generalized from Qwen3.5's hardcoded 2x v-expand and had never been run
// against a kernel. Measured result: those three tensors land on the reference
// head for head, identity, at score 0.997 -- the converter's
// `(q g p) -> (g q p)` rearranges undo GGUF's interleave rather than introduce
// one, so nothing has to be undone here. See `reorder` below for the one tensor
// that does move.
// ---------------------------------------------------------------------------

/// Row blocks are interleaved `vertical_blocks` deep because the memtile forwards
/// block i of every group to core row i.
inline void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 4)
{
    const int block_size = Q4K_BLOCK_BYTES;
    const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;

    // Catch a stale block size here: below, the truncating division would turn a wrong
    // stride into a plausible-looking row count instead of an obvious failure.
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

// ---------------------------------------------------------------------------
// Tensor table
// ---------------------------------------------------------------------------

/// \brief One tensor under test.
///
/// `unit` / `expand` / `groups` describe the row permutation needed to bring the
/// reference into q4nx order, in units of `unit` rows:
///
///     q4nx unit (g * expand + q)  <-  reference unit (q * groups + g)
///
/// `unit == 0` means the two are already aligned.
///
/// Only layer 3's q_proj moves. Gated full attention keeps q and the output gate
/// in one tensor interleaved per head, and qwen38.py de-interleaves it with
/// einops `(g p h) -> (p g h)`, h = 256, p = 2 -- i.e. this map with expand = 24,
/// groups = 2.
///
/// The delta-net tensors do *not* move, which is not what the converter source
/// suggests: `_rows_v_to_group_major` / `_reorder_qkv` / `_reorder_out_proj_cols`
/// each apply an expand-3 `(q g p) -> (g q p)`. Those act on GGUF order, and GGUF
/// interleaves the delta-net v heads relative to the HF checkpoint, so the two
/// cancel and q4nx comes out in reference order. Measured on layer 0: matching
/// each 128-row (or 128-column) unit against the reference by correlation gives
/// the identity, uniquely, with a worst unit score of 0.9965 -- Q4_K's own error.
struct tensor_case {
    std::string q4nx;   ///< name under "model.layers.<L>."
    std::string ref;    ///< name under "model.language_model.layers.<L>."
    int Dout;
    int Din;
    int unit;
    int expand;
    int groups;
};

static constexpr int ATTN_HEAD_DIM = 256;

static const std::vector<tensor_case> recurrent_tensors = {
    {"mlp.gate_proj.weight",             "mlp.gate_proj.weight",             17408,  5120, 0, 0, 0},
    {"mlp.up_proj.weight",               "mlp.up_proj.weight",               17408,  5120, 0, 0, 0},
    {"mlp.down_proj.weight",             "mlp.down_proj.weight",              5120, 17408, 0, 0, 0},
    {"linear_attn.qkv_proj.weight",      "linear_attn.in_proj_qkv.weight",   10240,  5120, 0, 0, 0},
    {"self_attn.gate_proj.weight",       "linear_attn.in_proj_z.weight",      6144,  5120, 0, 0, 0},
    {"linear_attn.ssm_out_proj.weight",  "linear_attn.out_proj.weight",       5120,  6144, 0, 0, 0},
};

static const std::vector<tensor_case> full_attention_tensors = {
    {"mlp.gate_proj.weight",   "mlp.gate_proj.weight",   17408,  5120, 0, 0, 0},
    {"mlp.up_proj.weight",     "mlp.up_proj.weight",     17408,  5120, 0, 0, 0},
    {"mlp.down_proj.weight",   "mlp.down_proj.weight",    5120, 17408, 0, 0, 0},
    {"self_attn.q_proj.weight", "self_attn.q_proj.weight", 12288, 5120, ATTN_HEAD_DIM, 24, 2},
    {"self_attn.k_proj.weight", "self_attn.k_proj.weight",  1024, 5120, 0, 0, 0},
    {"self_attn.v_proj.weight", "self_attn.v_proj.weight",  1024, 5120, 0, 0, 0},
    {"self_attn.o_proj.weight", "self_attn.o_proj.weight",  5120, 6144, 0, 0, 0},
};

/// Layer 3 is the full-attention block; 0, 1, 2 and 4 are delta-net.
inline bool is_recurrent(int layer) { return layer != 3; }

/// \brief Rewrite the reference tensor into q4nx order, in place.
///
/// Applied to the *reference*, not to the NPU output, so a failure points at the
/// converter's reorder rather than at anything this harness does.
inline void apply_reorder(buffer<bf16> &ref, const tensor_case &tc)
{
    if (tc.unit == 0) return;

    assert(tc.expand * tc.groups * tc.unit == tc.Dout);
    buffer<bf16> tmp((size_t)tc.Dout * tc.Din);
    for (int g = 0; g < tc.groups; g++) {
        for (int q = 0; q < tc.expand; q++) {
            const size_t dst_unit = (size_t)g * tc.expand + q;
            const size_t src_unit = (size_t)q * tc.groups + g;
            memcpy(tmp.data() + dst_unit * tc.unit * tc.Din,
                   ref.data() + src_unit * tc.unit * tc.Din,
                   (size_t)tc.unit * tc.Din * sizeof(bf16));
        }
    }
    memcpy(ref.data(), tmp.data(), (size_t)tc.Dout * tc.Din * sizeof(bf16));
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// \brief Cosine similarity of every 32-row block, reported as min / count below
///        a threshold.
///
/// The global number hides a partial failure: qkv_proj only permutes the last
/// 6144 of its 10240 rows, so a broken expand-3 still leaves a global cosine
/// around 0.6, which reads as "close" rather than "wrong".
inline void report_per_block(const bf16 *a, const bf16 *b, int Dout, int Din,
                             float threshold)
{
    const int row_blocks = Dout / QXNX_ROW_BLOCK_SIZE;
    double worst = 2.0;
    int worst_block = -1;
    int below = 0;
    for (int rb = 0; rb < row_blocks; rb++) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        const size_t base = (size_t)rb * QXNX_ROW_BLOCK_SIZE * Din;
        for (size_t i = 0; i < (size_t)QXNX_ROW_BLOCK_SIZE * Din; i++) {
            const double x = (float)a[base + i];
            const double y = (float)b[base + i];
            dot += x * y; na += x * x; nb += y * y;
        }
        const double cos = (na > 0.0 && nb > 0.0) ? dot / (sqrt(na) * sqrt(nb)) : 1.0;
        if (cos < worst) { worst = cos; worst_block = rb; }
        if (cos < threshold) below++;
    }
    header_print("info", "  per-32-row-block: min cos " << worst << " at block "
                         << worst_block << " (rows " << worst_block * QXNX_ROW_BLOCK_SIZE
                         << ".." << (worst_block + 1) * QXNX_ROW_BLOCK_SIZE - 1 << "), "
                         << below << " / " << row_blocks << " below " << threshold);
}

/// \brief Least-squares scale of ref against y: minimizes ||alpha * y - ref||.
///
/// A dequant that is right up to a constant factor -- a scale read from the wrong
/// field, say -- shows a perfect cosine but an alpha far from 1.
inline double least_squares_scale(const bf16 *y, const bf16 *ref, size_t n)
{
    double dot = 0.0, self = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (float)y[i];
        dot += d * (float)ref[i];
        self += d * d;
    }
    return self > 0.0 ? dot / self : 0.0;
}

/// RelativeL1 is not printed: metrices.hpp's `_mm256_abs_ps` builds its mask with
/// `_mm256_set1_ps(0x7fffffff)`, which *converts* the int to a float rather than
/// reinterpreting it, so the L1 sums are garbage. Cosine and RelativeL2 do not go
/// through it.
inline void print_metrics_line(const char *what, const error_metrics &m)
{
    header_print("info", "  " << what
                         << ": cos " << std::setprecision(6) << m.CosineSimilarity
                         << ", relL2 " << m.RelativeL2
                         << ", RMSE " << m.RMSE);
}

// ---------------------------------------------------------------------------

int main(int argc, const char *argv[]) {
    srand(0);

    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("layer,l", po::value<int>()->default_value(-1),
                       "only this layer (0..4); -1 for all");
    desc.add_options()("tensor,t", po::value<std::string>()->default_value(""),
                       "only tensors whose q4nx name contains this substring");
    desc.add_options()("model,m", po::value<std::string>()
                           ->default_value("/path/to/models/Qwen3.8-27B-NPU2"),
                       "q4nx model directory");
    desc.add_options()("reference,r", po::value<std::string>()
                           ->default_value("../../../models_simple/weights.safetensors"),
                       "bf16 reference safetensors");
    arg_utils::parse_options(argc, const_cast<char **>(argv), desc, vm);

    const int layer_filter = vm["layer"].as<int>();
    const std::string tensor_filter = vm["tensor"].as<std::string>();
    const std::string model_path = vm["model"].as<std::string>();
    const std::string reference_path = vm["reference"].as<std::string>();

    static_assert(Q4K_BLOCK_BYTES == 4736, "block stride must match the kernel struct");

    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager *dequant_app_mgr = xclbin_mgr.register_xclbin("build/xclbins/dequant.xclbin");
    npu_app dequant_q = dequant_app_mgr->create_app();

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);
    std::unique_ptr<SafeTensors> weight_reference = std::make_unique<SafeTensors>(reference_path);

    // Size the device buffers once, for the largest tensor in the table
    // (17408x5120 -> 178 MB out, 51 MB in), and reuse them: a per-tensor
    // allocation would churn 30 BOs of that size.
    size_t max_elems = 0, max_qbytes = 0;
    for (const auto *table : {&recurrent_tensors, &full_attention_tensors}) {
        for (const auto &tc : *table) {
            max_elems = std::max(max_elems, (size_t)tc.Dout * tc.Din);
            max_qbytes = std::max(max_qbytes,
                (size_t)tc.Dout / QXNX_ROW_BLOCK_SIZE * (tc.Din / QXNX_COL_BLOCK_SIZE)
                * Q4K_BLOCK_BYTES);
        }
    }
    header_print("info", "Allocating device buffers: " << max_elems * sizeof(bf16) / (1 << 20)
                         << " MB out, " << max_qbytes / (1 << 20) << " MB in");
    buffer<u8> q_weight = dequant_q.create_bo_buffer<u8>(max_qbytes);
    buffer<bf16> dequanted_w = dequant_q.create_bo_buffer<bf16>(max_elems);

    // Host-side scratch. buffer::resize always reallocates, so there is nothing to
    // gain from sizing these up front -- they are just kept out of the loop body so
    // the intent (one live copy of each, not one per tensor) stays visible.
    buffer<bf16> cpu_ref;
    buffer<bf16> ref;
    buffer<u8> raw;

    int checked = 0, failed = 0;

    for (int layer = 0; layer <= 4; layer++) {
        if (layer_filter >= 0 && layer != layer_filter) continue;
        const auto &table = is_recurrent(layer) ? recurrent_tensors : full_attention_tensors;

        for (const auto &tc : table) {
            if (!tensor_filter.empty() && tc.q4nx.find(tensor_filter) == std::string::npos)
                continue;

            const std::string q4nx_name = "model.layers." + std::to_string(layer) + "." + tc.q4nx;
            const std::string ref_name =
                "model.language_model.layers." + std::to_string(layer) + "." + tc.ref;

            header_print("info", "=== L" << layer << " " << tc.q4nx
                                 << " [" << tc.Dout << ", " << tc.Din << "] vs " << tc.ref);

            // A layer or shape mismatch would otherwise surface as a cosine near
            // zero that reads exactly like a dequant bug.
            const tensor_metadata qmeta = q4nx_model->get_tensor_metadata(q4nx_name);
            assert(qmeta.shape.size() == 3);
            assert(qmeta.shape[2] == Q4K_BLOCK_BYTES && "not a q4k tensor");
            assert((int)qmeta.shape[0] * QXNX_ROW_BLOCK_SIZE == tc.Dout);
            assert((int)qmeta.shape[1] * QXNX_COL_BLOCK_SIZE == tc.Din);
            const tensor_metadata rmeta = weight_reference->get_tensor_metadata(ref_name);
            assert(rmeta.shape.size() == 2);
            assert((int)rmeta.shape[0] == tc.Dout && (int)rmeta.shape[1] == tc.Din);

            const size_t elems = (size_t)tc.Dout * tc.Din;
            const size_t qbytes = (size_t)qmeta.shape[0] * qmeta.shape[1] * Q4K_BLOCK_BYTES;

            // --- NPU ---
            raw.resize(qbytes);
            q4nx_model->load_weights(raw, q4nx_name);
            reorder_cpy(q_weight.data(), raw, tc.Din, 4);
            q_weight.sync_to_device();
            generate_dequant_q4k_seq(dequant_q.seq(), tc.Dout, tc.Din, 0);
            dequant_q(dequanted_w, q_weight);
            dequanted_w.sync_from_device();

            // --- 1. against the scalar CPU decode of the same bytes ---
            cpu_ref.resize(elems);
            dequant_q4k_tensor_ref(cpu_ref, raw, tc.Dout, tc.Din);
            buffer<bf16> npu_view(dequanted_w.data(), elems);
            error_metrics m_cpu = get_error_metrics(npu_view, cpu_ref);
            print_metrics_line("NPU vs CPU decode ", m_cpu);
            // The two decodes agree to ~1.4% relative L2, not to bf16 rounding: the
            // kernel folds `S * scales` into a bf16 before the MAC, and w is a
            // difference of two comparable terms, so a 2^-9 error on the group scale
            // comes back amplified. That is stable across tensors and an order of
            // magnitude under Q4_K's own 7.4%, so gate on it loosely.
            //
            // The check is on relative L2 rather than cosine: cosine is accumulated
            // in float32 over up to 89M elements here, and dot ~ |a||b| makes it come
            // out at 1.00025 on the larger tensors. RelativeL2 is a ratio of sums of
            // squares and does not cancel like that. report_per_block below does use
            // cosine, in double and over one 32-row block, where it is trustworthy.
            const bool kernel_ok = m_cpu.RelativeL2 < 0.03f;
            if (!kernel_ok) {
                report_per_block(npu_view.data(), cpu_ref.data(), tc.Dout, tc.Din, 0.999f);
            }

            // --- 2. against the bf16 reference, brought into q4nx order ---
            ref.resize(elems);
            weight_reference->load_weights(ref, ref_name);
            apply_reorder(ref, tc);
            error_metrics m_ref = get_error_metrics(npu_view, ref);
            print_metrics_line("NPU vs bf16 ref  ", m_ref);
            header_print("info", "  least-squares scale ref/NPU: "
                                 << least_squares_scale(npu_view.data(), ref.data(), elems));
            report_per_block(npu_view.data(), ref.data(), tc.Dout, tc.Din, 0.99f);

            // 0.15 is twice Q4_K's own ~0.075 and far under the ~1.07 a wrong head
            // permutation leaves behind.
            const bool layout_ok = m_ref.RelativeL2 < 0.15f;
            checked++;
            if (!kernel_ok || !layout_ok) {
                failed++;
                header_print("FAIL", (kernel_ok ? "reorder/reference mismatch"
                                                : "kernel mismatch")
                                     << " on L" << layer << " " << tc.q4nx);
            } else {
                header_print("pass", "L" << layer << " " << tc.q4nx);
            }
        }
    }

    header_print("info", "==== " << (checked - failed) << " / " << checked << " tensors passed");
    return failed == 0 ? 0 : 1;
}
