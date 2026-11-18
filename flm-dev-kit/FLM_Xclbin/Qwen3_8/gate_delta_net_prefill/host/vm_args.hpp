//===- matrix_multiplication.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the matrix multiplication
// host code, such as verifying and printing matrices.

#ifndef __VM_ARGS_H__
#define __VM_ARGS_H__

#include <algorithm>
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include "utils/debug_utils.hpp"




namespace po = boost::program_options;

/* ---------- Struct that holds every argument --------- */
struct DeltaNetPrefillArgs {
    std::string arch;
    int trace_size;
    int L; 
    int DQ, DK, DV;
    int NUM_OF_ROW, NUM_OF_COLUMN;
    int MT_L_chunk_size, CT_L_chunk_size;
    int NUM_HEAD;
    int QK_HEAD_GROUP;
    int QKV_buffer_L_head_row_major; // 1 means the qkv buffer in Lx (  DQ*NUM_Q_K_HEAD + DV*NUM_Q_K_HEAD+ DV*NUM_HEAD) row major, 
                                    // 0 means in  LxDQ*NUM_Q_K_HEAD + L*DK*NUM_Q_K_HEAD + L*DV*NUM_HEAD row major. Note that the qkv buffer is always in the order of Q K V, just the layout is different.
};

/* ---------- Argument‑parsing helper (no try/catch) ---- */
bool parse_matmul_args(int argc, char* argv[], DeltaNetPrefillArgs& args)
{
    po::options_description desc("Run matmul with command line arguments");
    desc.add_options()
        ("arch",          po::value<std::string>(&args.arch)->required(),              "Architecture type")
        ("trace_size",    po::value<int>(&args.trace_size)->required(),                "Trace buffer size")
        ("L",             po::value<int>(&args.L)->required(),                         "Sequence length")
        ("DQ",            po::value<int>(&args.DQ)->required(),                      "Dimension of Query/Key")
        ("DK",            po::value<int>(&args.DK)->required(),                      "Dimension of Key/Value")
        ("DV",            po::value<int>(&args.DV)->required(),                      "Dimension of Value")
        ("NUM_OF_ROW",    po::value<int>(&args.NUM_OF_ROW)->required(),                "Number of row tiles")
        ("NUM_OF_COLUMN", po::value<int>(&args.NUM_OF_COLUMN)->required(),             "Number of column tiles")
        ("MT_L_chunk_size", po::value<int>(&args.MT_L_chunk_size)->required(),         "L chunk size for multi-threading")
        ("CT_L_chunk_size", po::value<int>(&args.CT_L_chunk_size)->required(),         "L chunk size for each CT")
        ("NUM_HEAD",      po::value<int>(&args.NUM_HEAD)->required(),                  "Number of attention heads")
        ("QK_HEAD_GROUP", po::value<int>(&args.QK_HEAD_GROUP)->required(),             "Number of head groups that share the same Q/K")
        ("QKV_buffer_L_head_row_major", po::value<int>(&args.QKV_buffer_L_head_row_major)->required(), "data order of qkv buffer");
        




    po::positional_options_description pos;
    pos.add("arch", 1);
    pos.add("trace_size", 1);
    pos.add("L", 1);
    pos.add("DQ", 1);
    pos.add("DK", 1);
    pos.add("DV", 1);
    pos.add("NUM_OF_ROW", 1);
    pos.add("NUM_OF_COLUMN", 1);
    pos.add("MT_L_chunk_size", 1);
    pos.add("CT_L_chunk_size", 1);
    pos.add("NUM_HEAD", 1);
    pos.add("QK_HEAD_GROUP", 1);
    pos.add("QKV_buffer_L_head_row_major", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);


    const bool ok = vm.count("arch") && vm.count("trace_size") && vm.count("L") &&
                    vm.count("DQ") && vm.count("DK") && vm.count("DV") &&
                    vm.count("NUM_OF_ROW") && vm.count("NUM_OF_COLUMN") &&
                    vm.count("MT_L_chunk_size") && vm.count("CT_L_chunk_size") &&
                    vm.count("NUM_HEAD") && vm.count("QK_HEAD_GROUP") && vm.count("QKV_buffer_L_head_row_major");
    if (!ok) {
        std::cerr << "Error: Missing required arguments.\n" << desc << std::endl;
        return false;
    }
    po::notify(vm); // Assign the variables after checking all required arguments are present
    return true;
}



#endif
