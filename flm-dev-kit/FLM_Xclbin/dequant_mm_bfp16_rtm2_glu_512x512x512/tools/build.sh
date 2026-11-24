#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# build the host tools (gemm_shapes_bench, rtm_test, rtm_bench, ab_bench, ts_test) against the design's host headers and ../host_common
#   HC=<host_common dir>  NLOH=<dir that holds nlohmann/json.hpp>   override the defaults
# ts_test is built with -DDIAG_BUS_EN (the host sequence enables the core processor bus; see README "Stall split")
cd "$(dirname "$0")/.."
HC=${HC:-../host_common}
NLOH=${NLOH:-/usr/include}
for t in gemm_shapes_bench rtm_test rtm_bench ab_bench ts_test glu_test; do
  EXTRA=""; [ "$t" = ts_test ] && EXTRA="-DDIAG_BUS_EN"
  g++-13 -std=c++20 -fPIC -O2 -DUSE_AVX2=1 -mavx2 -march=native -mfma -ffast-math -fopenmp $EXTRA \
    -Ihost -I"$HC" -I"$NLOH" -I/usr/include/xrt -I/usr/include tools/$t.cpp -o tools/$t \
    -L"$HC" -L"$HC/../../../src/lib/xrt" -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -laiebu -lboost_program_options -lboost_filesystem -fopenmp || exit 1
done
