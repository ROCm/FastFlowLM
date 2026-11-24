//===- elementwise_config.h - the shape, shared by cores, host and IRON -*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//
//
// The block each core computes over, EW_N bf16 elements, and the array the
// apps use, EW_COLS columns of EW_CORES cores. The IRON design takes the same
// numbers as --n, --cols and --cores; the Makefile sets all of them, so change
// them there. The length of a launch is chosen at run time (a multiple of
// EW_N * EW_COLS * EW_CORES): the cores loop forever.
//
//===----------------------------------------------------------------------===//

#ifndef ELEMENTWISE_CONFIG_H
#define ELEMENTWISE_CONFIG_H

#ifndef EW_N
#define EW_N 256
#endif

#ifndef EW_COLS
#define EW_COLS 1
#endif

#ifndef EW_CORES
#define EW_CORES 1
#endif

// AIE2P: one bf16 vector op covers 16 lanes (as aie_kernels/aie2/add.cc of
// mlir-aie) and the loops have no tail.
#define EW_VECTOR_LEN 16

#if (EW_N % EW_VECTOR_LEN) != 0
#error "EW_N must be a multiple of EW_VECTOR_LEN (16)"
#endif

#endif // ELEMENTWISE_CONFIG_H
