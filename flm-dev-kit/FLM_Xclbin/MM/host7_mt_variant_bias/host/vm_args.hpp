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
struct MatmulArgs {
    std::string arch;
    int trace_size;
    int M_size, K_size, N_size;
    int m, k, n;
    int IS_B_ROW_MAJOR;
    int total_row, total_col;
    int r, s, t;
    int ENABLE_AXI4;
    int IS_B_COL_MAJOR_S_T_ROW_MAJOR, IS_BFLOAT16_OUT_TY, ADD_BIAS, OUTPUT_MODE;
    int OUTPUT_CLAMP;
};

/* ---------- Argument‑parsing helper (no try/catch) ---- */
bool parse_matmul_args(int argc, char* argv[], MatmulArgs& args)
{
    po::options_description desc("Run matmul with command line arguments");
    desc.add_options()
        ("arch",          po::value<std::string>(&args.arch)->required(),              "Architecture type")
        ("trace_size",    po::value<int>(&args.trace_size)->required(),                "Trace buffer size")
        ("M_size",        po::value<int>(&args.M_size)->required(),                    "Global M size")
        ("K_size",        po::value<int>(&args.K_size)->required(),                    "Global K size")
        ("N_size",        po::value<int>(&args.N_size)->required(),                    "Global N size")
        ("m",             po::value<int>(&args.m)->required(),                         "Tile‑local m")
        ("k",             po::value<int>(&args.k)->required(),                         "Tile‑local k")
        ("n",             po::value<int>(&args.n)->required(),                         "Tile‑local n")
        ("IS_B_ROW_MAJOR",po::value<int>(&args.IS_B_ROW_MAJOR)->required(),            "B matrix layout")
        ("total_row",     po::value<int>(&args.total_row)->required(),                 "Total row tiles")
        ("total_col",     po::value<int>(&args.total_col)->required(),                 "Total col tiles")
        ("r", po::value<int>(&args.r)->required(), "r")
        ("s", po::value<int>(&args.s)->required(), "s")
        ("t", po::value<int>(&args.t)->required(), "t")
        ("ENABLE_AXI4", po::value<int>(&args.ENABLE_AXI4)->required(), "Enable AXI4 interface (0 or 1)")
        ("IS_B_COL_MAJOR_S_T_ROW_MAJOR", po::value<int>(&args.IS_B_COL_MAJOR_S_T_ROW_MAJOR)->required(), "IS_B_COL_MAJOR_S_T_ROW_MAJOR 0 or 1")
        ("IS_BFLOAT16_OUT_TY", po::value<int>(&args.IS_BFLOAT16_OUT_TY)->required(), "1 if use bfloat16, else use float32")
        ("ADD_BIAS", po::value<int>(&args.ADD_BIAS)->required(), "1 if add bias, else 0")
        ("OUTPUT_MODE", po::value<int>(&args.OUTPUT_MODE)->required(), "0 if output normal, 1 if with gelu")
        
        ("OUTPUT_CLAMP", po::value<int>(&args.OUTPUT_CLAMP)->required(), "1 if clamp output, else 0");

    po::positional_options_description pos;
    pos.add("arch", 1).add("trace_size", 1).add("M_size", 1).add("K_size", 1)
       .add("N_size", 1).add("m", 1).add("k", 1).add("n", 1)
       .add("IS_B_ROW_MAJOR", 1).add("total_row", 1).add("total_col", 1).add("r",1)
       .add("s",1).add("t",1).add("ENABLE_AXI4", 1).add("IS_B_COL_MAJOR_S_T_ROW_MAJOR", 1)
       .add("IS_BFLOAT16_OUT_TY", 1).add("ADD_BIAS", 1).add("OUTPUT_MODE", 1).add("OUTPUT_CLAMP", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);

    // Manually ensure every required argument is present
    const bool ok =
        vm.count("arch")          && vm.count("trace_size") &&
        vm.count("M_size")        && vm.count("K_size")     && vm.count("N_size") &&
        vm.count("m")             && vm.count("k")          && vm.count("n") &&
        vm.count("IS_B_ROW_MAJOR")&& vm.count("total_row")  && vm.count("total_col")&&
        vm.count("r")&& vm.count("s") && vm.count("t") && vm.count("ENABLE_AXI4") && 
        vm.count("IS_B_COL_MAJOR_S_T_ROW_MAJOR") && vm.count("IS_BFLOAT16_OUT_TY")&&
        vm.count("ADD_BIAS") && vm.count("OUTPUT_MODE") && vm.count("OUTPUT_CLAMP");

    if (!ok) {
        std::cerr << "Error: missing required positional arguments.\n\n" << desc << '\n';
        return false;
    }

    po::notify(vm);   // safe because everything required is present
    return true;
}

#endif
