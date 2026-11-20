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
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "embedding.hpp"
#include "util.hpp"
#include "weights_desc.hpp"

namespace po = boost::program_options;

int round_up_to_multiple(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    arg_utils::parse_options(argc, argv, desc, vm);

    // load reference data and weights
    std::string weight_path = "/path/to/models/Gemma4-12B-IT-NPU2";
    std::string q40_reference_path = weight_path + "/ref.safetensors";
    std::unique_ptr<SafeTensors> q40_references = std::make_unique<SafeTensors>(q40_reference_path);
    Q4NX weights(weight_path);

 //////////////////////////// model parameters size  ////////////////////////////
    int num_attention_heads = 16;
    int head_dim = 256;
    int num_key_value_heads = 8;
    int intermediate_size = 15360;

    int D = 3840;
    int DQ = head_dim * num_attention_heads;
    int DK = head_dim * num_key_value_heads;
    int DV = head_dim * num_key_value_heads;
    std::cout << "D: " << D << ", DQ: " << DQ << ", DK: " << DK << ", DV: " << DV << std::endl;
    
    uint32_t L_in = 1989;
    std::cout << "[DEBUG] L_in: " << L_in << std::endl;
    uint32_t L_begin = 0;
    uint32_t L_end = L_begin + L_in;
    uint32_t L_chunk = 512;
    uint32_t L_begin_chunked = (L_begin / L_chunk) * L_chunk;
    int L_offset = L_begin - L_begin_chunked;
    std::cout << "[DEBUG] L_offset: " << L_offset << std::endl;
    uint32_t L_padded = round_up_to_multiple(L_in, L_chunk);
    std::cout << "[DEBUG] L_padded: " << L_padded << std::endl;
    int L_effective = L_end - L_begin;
    std::cout << "[DEBUG] L_effective: " << L_effective << std::endl;
    //////////////////////////// model parameters size  ////////////////////////////
    //////////////////////////// setup npu xclbins  ////////////////////////////
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    // dequantization xclbin
    std::string xclbin_name = "build/xclbins/dequant_mm.xclbin";
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin(xclbin_name);

    // EXTRACT=1: feed the identity so that y = I * W^T = W^T.  That pulls out the
    // exact dequantised weights the NPU is actually multiplying with, which lets a
    // CPU fp32 matmul separate "the weights are lossy" from "the NPU computed the
    // wrong thing".  Diagnostic only -- not part of the normal test.
    if (getenv("EXTRACT")) {
        struct { const char* tag; int Din; int Dout; const char* w; } cfgs[] = {
            {"q", D,  DQ, "model.layers.%d.self_attn.q_proj.weight"},
            {"o", DQ, D,  "model.layers.%d.self_attn.o_proj.weight"},
        };
        for (auto& c : cfgs) {
            uint32_t L_id = round_up_to_multiple(c.Din, L_chunk);
            npu_app app = dequant_mm->create_app();
            gen_dequant_mm(app.seq(), L_id, c.Dout, c.Din, 0, 0);

            buffer<u8>   bw = app.create_bo_buffer<u8>((size_t)c.Din * c.Dout * 9 / 16);
            buffer<bf16> bx = app.create_bo_buffer<bf16>((size_t)L_id * c.Din);
            buffer<bf16> by = app.create_bo_buffer<bf16>((size_t)L_id * c.Dout);

            buffer<u8> raw;
            char w_name[128];
            snprintf(w_name, sizeof(w_name), c.w, 0);
            weights.load_weights(raw, w_name);
            buffer<u8> src(raw.data(), raw.size());
            assert(src.size() == bw.size());
            reorder_cpy(bw.data(), src, c.Din, 2);
            bw.sync_to_device();

            bx.memset(bf16(0.0f));
            for (int i = 0; i < c.Din; i++) bx.data()[(size_t)i * c.Din + i] = bf16(1.0f);
            bx.sync_to_device();
            by.memset(bf16(0.0f));
            by.sync_to_device();

            app(by, bx, bw);
            by.sync_from_device();

            std::string path = std::string("/tmp/wT_") + c.tag + ".bin";
            FILE* f = fopen(path.c_str(), "wb");
            fwrite(by.data(), sizeof(bf16), (size_t)c.Din * c.Dout, f);   // first Din rows = W^T
            fclose(f);
            std::cout << "[EXTRACT] wrote " << path << "  W^T is ("
                      << c.Din << ", " << c.Dout << ")" << std::endl;
        }
        return 0;
    }

    // test q40
    npu_app q_proj = dequant_mm->create_app();
    // The schedule covers the padded L -- the tail rows are computed and then
    // ignored, so buffer_x must be zeroed rather than left uninitialised.
    gen_dequant_mm(q_proj.seq(), L_padded, DQ, D, 0, 0);

    buffer<u8> quant_q_weights = q_proj.create_bo_buffer<u8>(D * DQ * 9 / 16);
    buffer<bf16> buffer_x = q_proj.create_bo_buffer<bf16>(L_padded * D);
    buffer<bf16> buffer_q = q_proj.create_bo_buffer<bf16>((size_t)L_padded * DQ);

    //////////////////////////// setup npu xclbins  ////////////////////////////
    //////////////////////////// Start  ////////////////////////////
    // test q40
    // load the q_proj quantized weight from disk and reorder it into the bo buffer
    int layer_idx = 0;
    {
        buffer<u8> w_q;
        char w_name[128];
        snprintf(w_name, sizeof(w_name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
        weights.load_weights(w_q, w_name);
        buffer<u8> w_q_q(w_q.data(), w_q.size());
        assert(w_q_q.size() == quant_q_weights.size());
        // vertical_blocks = 2 everywhere: the weight buffer is packed exactly the way
        // the decoding runtime packs it, and gen_dequant_mm's shim BD gathers the
        // 4-way (desired_n = 64) order the prefill core wants out of that layout.
        reorder_cpy(quant_q_weights.data(), w_q_q, D, 2);
    }
    quant_q_weights.sync_to_device();
    header_print("DEBUG", "Start Dequant qkv weights");
    buffer<bf16> x;
    q40_references->load_weights(x, "layer_0.input_layernorm");
    buffer_x.memset(bf16(0.0f));
    buffer_x.copy_from(x.data() + L_offset * D, L_effective * D);
    buffer_x.sync_to_device();

    buffer_q.memset(bf16(-1.0f));
    buffer_q.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    int Iterations = 1;
    for (int i = 0; i < Iterations; i++) {
        time_utils::time_point start = time_utils::now();
        q_proj(buffer_q, buffer_x, quant_q_weights);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    buffer_q.sync_from_device();
    
    npu_time.first /= Iterations;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    buffer<bf16> q_ref;
    buffer<bf16> q(L_effective * DQ);
    buffer<bf16> q_r(L_effective * DQ);
    q40_references->load_weights(q_ref, "layer_0.q_proj");
    q.copy_from(buffer_q.data() + L_offset * DQ, L_effective * DQ);
    q_r.copy_from(q_ref.data() + L_offset * DQ, L_effective * DQ);

    header_print("DEBUG", "Output of dequantization + q projection");
    utils::print_matrix(q, DQ);
    header_print("DEBUG", "Reference q_proj output");
    utils::print_matrix(q_r, DQ);
    // q, not buffer_q -- buffer_q is L_padded rows, q_r is only L_effective.
    error_metrics metrics = get_error_metrics(q, q_r);
    print_error_metrics(metrics);

    // v_proj has Dout = DK = 2048, so T = Dout / 256 = 8: even, like q_proj's 16,
    // so no L-chunk boundary lands in the middle of a round.  What it does cover
    // that q_proj does not is the narrow-Dout end -- only 8 weight tiles per
    // L-chunk, so the weight stream is short relative to the x stream.  The odd-T
    // seam case is o_proj below.
    {
        const int v_Din = D, v_Dout = DK;

        npu_app v_proj = dequant_mm->create_app();
        gen_dequant_mm(v_proj.seq(), L_padded, v_Dout, v_Din, 0, 0);

        buffer<u8>   v_w = v_proj.create_bo_buffer<u8>((size_t)v_Din * v_Dout * 9 / 16);
        buffer<bf16> v_x = v_proj.create_bo_buffer<bf16>((size_t)L_padded * v_Din);
        buffer<bf16> v_y = v_proj.create_bo_buffer<bf16>((size_t)L_padded * v_Dout);

        {
            buffer<u8> raw;
            char w_name[128];
            snprintf(w_name, sizeof(w_name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
            weights.load_weights(raw, w_name);
            buffer<u8> src(raw.data(), raw.size());
            assert(src.size() == v_w.size());
            reorder_cpy(v_w.data(), src, v_Din, 2);
        }
        v_w.sync_to_device();

        buffer<bf16> v_in;
        q40_references->load_weights(v_in, "layer_0.input_layernorm");
        v_x.memset(bf16(0.0f));   // the padded tail rows are computed, so zero them
        v_x.copy_from(v_in.data() + L_offset * v_Din, (size_t)L_effective * v_Din);
        v_x.sync_to_device();
        v_y.memset(bf16(-1.0f));
        v_y.sync_to_device();

        time_utils::time_point v_start = time_utils::now();
        v_proj(v_y, v_x, v_w);
        time_utils::time_point v_stop = time_utils::now();
        v_y.sync_from_device();

        buffer<bf16> v_ref_all;
        buffer<bf16> v_got(L_effective * v_Dout);
        buffer<bf16> v_ref(L_effective * v_Dout);
        q40_references->load_weights(v_ref_all, "layer_0.v_proj");
        v_got.copy_from(v_y.data() + L_offset * v_Dout, (size_t)L_effective * v_Dout);
        v_ref.copy_from(v_ref_all.data() + L_offset * v_Dout, (size_t)L_effective * v_Dout);

        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "v_proj T=" << v_Dout / 256 << " C=" << L_padded / 512
                          << " rounds=" << ((v_Dout / 256) * (L_padded / 512) + 1) / 2
                          << ": " << time_utils::duration_us(v_start, v_stop).first << " us");
        MSG_BONDLINE(60);
        error_metrics v_metrics = get_error_metrics(v_got, v_ref);
        print_error_metrics(v_metrics);

        if (const char* tag = getenv("DUMP")) {
            for (auto& p : {std::make_pair("got", &v_got), std::make_pair("ref", &v_ref)}) {
                std::string path = std::string("/tmp/v_") + p.first + "_" + tag + ".bin";
                FILE* f = fopen(path.c_str(), "wb");
                fwrite(p.second->data(), sizeof(bf16), p.second->size(), f);
                fclose(f);
            }
        }
    }
    // o_proj has Dout = 3840, so T = Dout / 256 = 15 is odd and every L-chunk
    // boundary lands in the middle of a round -- the seam rounds that q_proj
    // (T = 16) and v_proj (T = 8) never produce.  units = T * C is still even
    // here (15 * 4 = 60 at L_padded = 2048), so the half-idle tail round stays
    // uncovered; to reach it, pair this odd T with an odd C -- L_IN in
    // (1536, 2048] gives C = 4, but L_IN in (1024, 1536] gives C = 3 and
    // units = 45.
    {
        const int o_Din = DQ, o_Dout = D;

        npu_app o_proj = dequant_mm->create_app();
        gen_dequant_mm(o_proj.seq(), L_padded, o_Dout, o_Din, 0, 0);

        buffer<u8>   o_w = o_proj.create_bo_buffer<u8>((size_t)o_Din * o_Dout * 9 / 16);
        buffer<bf16> o_x = o_proj.create_bo_buffer<bf16>((size_t)L_padded * o_Din);
        buffer<bf16> o_y = o_proj.create_bo_buffer<bf16>((size_t)L_padded * o_Dout);

        {
            buffer<u8> raw;
            char w_name[128];
            snprintf(w_name, sizeof(w_name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
            weights.load_weights(raw, w_name);
            buffer<u8> src(raw.data(), raw.size());
            assert(src.size() == o_w.size());
            reorder_cpy(o_w.data(), src, o_Din, 2);
        }
        o_w.sync_to_device();

        buffer<bf16> o_in;
        q40_references->load_weights(o_in, "layer_0.attention_out");
        o_x.memset(bf16(0.0f));   // the padded tail rows are computed, so zero them
        o_x.copy_from(o_in.data() + L_offset * o_Din, (size_t)L_effective * o_Din);
        o_x.sync_to_device();
        o_y.memset(bf16(-1.0f));
        o_y.sync_to_device();

        time_utils::time_point o_start = time_utils::now();
        o_proj(o_y, o_x, o_w);
        time_utils::time_point o_stop = time_utils::now();
        o_y.sync_from_device();

        buffer<bf16> o_ref_all;
        buffer<bf16> o_got(L_effective * o_Dout);
        buffer<bf16> o_ref(L_effective * o_Dout);
        q40_references->load_weights(o_ref_all, "layer_0.aux_out");
        o_got.copy_from(o_y.data() + L_offset * o_Dout, (size_t)L_effective * o_Dout);
        o_ref.copy_from(o_ref_all.data() + L_offset * o_Dout, (size_t)L_effective * o_Dout);

        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "o_proj T=" << o_Dout / 256 << " C=" << L_padded / 512
                          << " rounds=" << ((o_Dout / 256) * (L_padded / 512) + 1) / 2
                          << ": " << time_utils::duration_us(o_start, o_stop).first << " us");
        MSG_BONDLINE(60);
        error_metrics o_metrics = get_error_metrics(o_got, o_ref);
        print_error_metrics(o_metrics);

        if (const char* tag = getenv("DUMP")) {
            for (auto& p : {std::make_pair("got", &o_got), std::make_pair("ref", &o_ref)}) {
                std::string path = std::string("/tmp/o_") + p.first + "_" + tag + ".bin";
                FILE* f = fopen(path.c_str(), "wb");
                fwrite(p.second->data(), sizeof(bf16), p.second->size(), f);
                fclose(f);
            }
        }
    }

    // Fused gate/up.  The weight buffer is built exactly the way decoding_layer.cpp
    // builds it -- alternating GLU_SLAB-row slabs of up and gate -- so the sequence
    // has to de-interleave on the way out.  y comes back as [ up | gate ], each
    // intermediate_size wide, which is what the two reference tensors are.
    {
        constexpr int GLU_SLAB = 512;                  // output rows per up/gate slab
        const int gu_Din = D, gu_Dout = 2 * intermediate_size;

        npu_app gu_proj = dequant_mm->create_app();
        gen_dequant_mm(gu_proj.seq(), L_padded, gu_Dout, gu_Din, 0, 0, GLU_SLAB);

        buffer<u8>   gu_w = gu_proj.create_bo_buffer<u8>((size_t)gu_Din * gu_Dout * 9 / 16);
        buffer<bf16> gu_x = gu_proj.create_bo_buffer<bf16>((size_t)L_padded * gu_Din);
        buffer<bf16> gu_y = gu_proj.create_bo_buffer<bf16>((size_t)L_padded * gu_Dout);

        {
            buffer<u8> w_up;
            buffer<u8> w_gate;
            char w_name[128];
            snprintf(w_name, sizeof(w_name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
            weights.load_weights(w_up, w_name);
            snprintf(w_name, sizeof(w_name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
            weights.load_weights(w_gate, w_name);

            const size_t chunk_size = (size_t)GLU_SLAB * gu_Din * 9 / 16;
            const int phases = intermediate_size / GLU_SLAB;
            assert(w_up.size() == (size_t)phases * chunk_size);
            assert(w_gate.size() == w_up.size());

            tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
            tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
            u8* w_ptr = gu_w.data();
            for (int i = 0; i < phases; i++) {
                // vertical_blocks = 2, same as the decoding runtime -- see above
                reorder_cpy(w_ptr, w_up_tensor[i], gu_Din, 2);
                w_ptr += chunk_size;
                reorder_cpy(w_ptr, w_gate_tensor[i], gu_Din, 2);
                w_ptr += chunk_size;
            }
            assert(w_ptr == gu_w.end());
        }
        gu_w.sync_to_device();

        buffer<bf16> gu_in;
        q40_references->load_weights(gu_in, "layer_0.pre_feedforward_norm");
        gu_x.memset(bf16(0.0f));   // the padded tail rows are computed, so zero them
        gu_x.copy_from(gu_in.data() + L_offset * gu_Din, (size_t)L_effective * gu_Din);
        gu_x.sync_to_device();
        gu_y.memset(bf16(-1.0f));
        gu_y.sync_to_device();

        time_utils::time_point gu_start = time_utils::now();
        gu_proj(gu_y, gu_x, gu_w);
        time_utils::time_point gu_stop = time_utils::now();
        gu_y.sync_from_device();

        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "gate/up T=" << gu_Dout / 256 << " C=" << L_padded / 512
                          << " rounds=" << ((gu_Dout / 256) * (L_padded / 512) + 1) / 2
                          << ": " << time_utils::duration_us(gu_start, gu_stop).first << " us");
        MSG_BONDLINE(60);

        // Rows are gu_Dout wide on the device but only intermediate_size wide in the
        // reference, so both halves have to be un-strided row by row.
        struct { const char* tag; size_t col_base; } halves[] = {
            {"layer_0.up_proj",   0},
            {"layer_0.gate_proj", (size_t)intermediate_size},
        };
        for (auto& h : halves) {
            buffer<bf16> ref_all;
            q40_references->load_weights(ref_all, h.tag);
            buffer<bf16> got((size_t)L_effective * intermediate_size);
            buffer<bf16> ref((size_t)L_effective * intermediate_size);
            for (int r = 0; r < L_effective; r++) {
                memcpy(got.data() + (size_t)r * intermediate_size,
                       gu_y.data() + (size_t)(L_offset + r) * gu_Dout + h.col_base,
                       (size_t)intermediate_size * sizeof(bf16));
            }
            ref.copy_from(ref_all.data() + (size_t)L_offset * intermediate_size,
                          (size_t)L_effective * intermediate_size);

            header_print("DEBUG", std::string("Output of dequantization + ") + h.tag);
            utils::print_matrix(got, intermediate_size);
            header_print("DEBUG", std::string("Reference ") + h.tag + " output");
            utils::print_matrix(ref, intermediate_size);
            error_metrics gu_metrics = get_error_metrics(got, ref);
            print_error_metrics(gu_metrics);
        }
    }

    return 0;
}