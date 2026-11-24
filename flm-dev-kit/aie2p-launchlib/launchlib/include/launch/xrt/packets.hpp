//===- packets.hpp - the control packets of a kernel, in XRT --------*- C++ -*-===//
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
// A kernel of Transport::ctrlpkt reads its control packets from a buffer of
// kernel argument `Kernel::packet_arg` (XRT argument 3 + packet_arg, one the
// runtime sequence does not use). For hosts that run the kernels themselves;
// Session (session.hpp) does it for its own.
//
//   std::vector<xrt::bo> packets = launch::packet_buffers(device, kernel, result);
//   ...
//   launch::set_packets(run, result.kernels[k], packets[k]);
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/launch.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstring>
#include <vector>

namespace launch {

/// The packets of `k` in a buffer of its packet argument's memory group (an
/// empty bo when it has none).
inline xrt::bo packet_buffer(xrt::device &device, const xrt::kernel &kernel, const Kernel &k) {
  if (k.packets.empty())
    return {};
  xrt::bo bo(device, k.packets.size() * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3 + k.packet_arg));
  std::memcpy(bo.map<void *>(), k.packets.data(), k.packets.size() * 4);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return bo;
}

/// One per kernel of `r`, in its order.
inline std::vector<xrt::bo> packet_buffers(xrt::device &device, const xrt::kernel &kernel, const Result &r) {
  std::vector<xrt::bo> out;
  for (const Kernel &k : r.kernels)
    out.push_back(packet_buffer(device, kernel, k));
  return out;
}

/// Binds `packets` (packet_buffer of `k`) to its argument of `run`; nothing
/// when `k` has no packets.
inline void set_packets(xrt::run &run, const Kernel &k, const xrt::bo &packets) {
  if (!k.packets.empty())
    run.set_arg(3 + k.packet_arg, packets);
}

} // namespace launch
