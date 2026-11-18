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
#include "seq_gen.hpp"

namespace po = boost::program_options;

// Q4NX on disk : [scales 8xM bf16][qs M*256/2 B] per Mx256 block (flm_q40)
// AIE q4k_block: [scales 8xM bf16][qs M*256/2 B] per Mx256 block
// M is QXNX_ROW_BLOCK_SIZE (16), i.e. 2304B per block -- keep the geometry derived from
// the shared constants rather than spelled out, so a tile-size change cannot leave this
// copy interleaving at the wrong granularity while still passing the asserts below.
// The two layouts match now that the converter emits q40 rather than q41 (no mins slice),
// so the only remaining job is the row interleave: row blocks are interleaved
// `vertical_blocks` deep because the memtile forwards block i of every group to core row i.
inline void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 4)
{
    const int block_size = get_quantization_byte_size(
        QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q40);
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

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options

    arg_utils::parse_options(argc, const_cast<char **>(argv), desc, vm);

    int D = 3840;
    int DH = 256;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int DQ = num_attention_heads * DH;
    int DK = num_key_value_heads * DH;
    int DV = num_key_value_heads * DH;

    // The layer under test. The two checkpoints number layers the same way
    // (q4nx "model.layers.N.*" <-> reference "layer_N.*"), so both names must be
    // built from this one index -- comparing different layers silently reports a
    // cosine of ~0 that looks like a dequant bug.
    // Only sliding-window layers (N % 6 != 5) have this DH=256 / 16-head geometry;
    // the full-attention layers (5, 11, ...) are DH=512 with q_proj [8192, D].
    const int layer = 10;
    assert(layer % 6 != 5 && "full-attention layers need DH=512 / DQ=8192");

    // overide the layer xclbin
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    std::string xclbin_name = "build/xclbins/dequant.xclbin";
    npu_app_manager* dequant_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);
    npu_app dequant_q = dequant_app_mgr->create_app();

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>("/path/to/models/Gemma4-12B-IT-NPU2");
    std::string weight_reference_path = "/path/to/models/Gemma4-12B-IT-NPU2/gemma4_12b_qat_weights.safetensors";
    std::unique_ptr<SafeTensors> weight_reference = std::make_unique<SafeTensors>(weight_reference_path);

    const std::string tensor = "self_attn.q_proj.weight";
    buffer<bf16> weight_ref;
    weight_reference->load_weights(
        weight_ref, "layer_" + std::to_string(layer) + "." + tensor);
    header_print("DEBUG", "ref o weights");
    utils::print_matrix(weight_ref, D);

    {
        buffer<u8> wo_0;
        q4nx_model->load_weights(
            wo_0, "model.layers." + std::to_string(layer) + "." + tensor);

        buffer<u8> proj_weight = dequant_q.create_bo_buffer<u8>(
            get_quantization_byte_size((size_t)D * DQ, flm_q40));
        buffer<bf16> dequanted_w = dequant_q.create_bo_buffer<bf16>(D * DQ);

        reorder_cpy(proj_weight.data(), wo_0, D, 4);
        proj_weight.sync_to_device();
        generate_dequant_q4_0_seq(dequant_q.seq(), DQ, D, 0);

        dequant_q(dequanted_w, proj_weight);
        dequanted_w.sync_from_device();

        header_print("DEBUG", "q weights");
        utils::print_matrix(dequanted_w, D);
        error_metrics q_metrics = get_error_metrics(dequanted_w, weight_ref);
        print_error_metrics(q_metrics);

        // model.q4nx inherits a per-tensor scale mismatch from the source GGUF: its
        // nibbles agree with an absmax Q4_0 of the reference, but the bf16 block scales
        // are alpha x too small, alpha ~1.05 for layers 0,1 mod 6 and ~1.9 (3.5 for
        // down_proj) for the rest. That factor is in the file, not in this dequant path,
        // and it is not exactly 2 -- fit it instead of hard-coding, otherwise the metrics
        // below mix a real dequant bug with the checkpoint's scale offset.
        double dot = 0.0, self = 0.0;
        for (size_t i = 0; i < dequanted_w.size(); i++){
            const double d = (float)dequanted_w[i];
            dot += d * (float)weight_ref[i];
            self += d * d;
        }
        const double alpha = self > 0.0 ? dot / self : 1.0;
        header_print("DEBUG", "least-squares scale ref/dequant: " << alpha);
        for (size_t i = 0; i < dequanted_w.size(); i++){
            dequanted_w[i] = (bf16)((float)dequanted_w[i] * (float)alpha);
        }
        header_print("DEBUG", "after removing the per-tensor scale");
        error_metrics q_metrics_scaled = get_error_metrics(dequanted_w, weight_ref);
        print_error_metrics(q_metrics_scaled);
    }
}