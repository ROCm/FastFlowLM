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

namespace arg_utils {

namespace po = boost::program_options;

void add_default_options(po::options_description &desc) {
    desc.add_options()("help,h", "produce help message");
    desc.add_options()("device,d", po::value<std::string>()->default_value("npu2"), "Device type, npu1 or npu2");
}

void parse_options(int argc, const char *argv[], po::options_description &desc,
    po::variables_map &vm) {
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            std::exit(1);
        }
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n\n";
        std::cerr << "Usage:\n" << desc << "\n";
        std::exit(1);
    }
}


}
#endif
