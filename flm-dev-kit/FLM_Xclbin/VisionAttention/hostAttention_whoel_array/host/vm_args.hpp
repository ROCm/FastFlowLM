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
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include "utils/debug_utils.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct AttentionArgs {
    std::string device;
    int r;
    int s;
    int t;
    int DH;
    int LQ_per_CT;
    int LK_per_CT;
    int lq_internal;
    int lk_lv_internal;
    int NUM_HEAD;
    int SHIM_L_SEQ;
    int SHIM_S_SEQ;
    int NUM_OF_COLUMNS;
    int NUM_OF_CT_PER_COLUMN;
    int CU_MODE;
    int LOW_PRECISION;
    int RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION;
};

inline bool parse_attention_args(int argc, const char* argv[], AttentionArgs& args) {
    po::options_description desc("Run attention with command line arguments");
    desc.add_options()
        ("device,d", po::value<std::string>(&args.device)->default_value("npu2"), "Device type")
        ("r", po::value<int>(&args.r)->default_value(8), "r")
        ("s", po::value<int>(&args.s)->default_value(8), "s")
        ("t", po::value<int>(&args.t)->default_value(8), "t")
        ("DH", po::value<int>(&args.DH)->default_value(72), "Dimension Head")
        ("LQ_per_CT", po::value<int>(&args.LQ_per_CT)->default_value(32), "LQ per CT")
        ("LK_per_CT", po::value<int>(&args.LK_per_CT)->default_value(512), "LK per CT")
        ("lq_internal", po::value<int>(&args.lq_internal)->default_value(32), "lq internal")
        ("lk_lv_internal", po::value<int>(&args.lk_lv_internal)->default_value(16), "lk lv internal")
        ("NUM_HEAD", po::value<int>(&args.NUM_HEAD)->default_value(1), "Number of Heads")
        ("SHIM_L_SEQ", po::value<int>(&args.SHIM_L_SEQ)->default_value(1024), "Shim L Seq")
        ("SHIM_S_SEQ", po::value<int>(&args.SHIM_S_SEQ)->default_value(3042), "Shim S Seq")
        ("NUM_OF_COLUMNS", po::value<int>(&args.NUM_OF_COLUMNS)->default_value(8), "Number of Columns")
        ("NUM_OF_CT_PER_COLUMN", po::value<int>(&args.NUM_OF_CT_PER_COLUMN)->default_value(4), "Number of CT per column")
        ("CU_MODE", po::value<int>(&args.CU_MODE)->default_value(0), "CU Mode")
        ("LOW_PRECISION", po::value<int>(&args.LOW_PRECISION)->default_value(1), "Low Precision")
        ("RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION", po::value<int>(&args.RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION)->default_value(1), "Runtime sequence start from L dimension")
        ("help,h", "produce help message");

    po::positional_options_description p;
    p.add("device", 1);
    p.add("r", 1);
    p.add("s", 1);
    p.add("t", 1);
    p.add("DH", 1);
    p.add("LQ_per_CT", 1);
    p.add("LK_per_CT", 1);
    p.add("lq_internal", 1);
    p.add("lk_lv_internal", 1);
    p.add("NUM_HEAD", 1);
    p.add("SHIM_L_SEQ", 1);
    p.add("SHIM_S_SEQ", 1);
    p.add("NUM_OF_COLUMNS", 1);
    p.add("NUM_OF_CT_PER_COLUMN", 1);
    p.add("CU_MODE", 1);
    p.add("LOW_PRECISION", 1);
    p.add("RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n\n";
        return false;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0);
    }

    return true;
}

#endif
