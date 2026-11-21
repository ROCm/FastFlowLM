// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// flow.hpp - the kernels of a build as XRT runs, in either flow
//
//   elf     an ELF per kernel (txn::to_elf), an xrt::ext::kernel of it; the run's
//           instruction arguments are 0 (launch::Session's default flow)
//   xclbin  the base xclbin's kernel, the instructions in a BO per kernel
//
// The buffers stay the xclbin kernel's (group_id); both flows take them.
#pragma once

#include "launch/launch.hpp"
#include "launch/txn/elf.hpp"
#include "launch/xrt/packets.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ew {

/// "elf" -> true, "xclbin" -> false.
inline bool elf_flow(const std::string &name) {
  if (name == "elf") return true;
  if (name == "xclbin") return false;
  throw std::invalid_argument("--flow " + name + ": elf or xclbin");
}

inline const char *flow_name(bool elf) { return elf ? "elf" : "xclbin"; }

class FlowKernels {
public:
  FlowKernels(xrt::device &device, xrt::hw_context &context, const xrt::kernel &kernel, const launch::Result &r,
              bool elf)
      : kernel_(kernel), result_(r), elf_(elf) {
    for (const launch::Kernel &k : r.kernels) {
      if (elf) {
        const std::vector<char> bytes = launch::txn::to_elf(k.words);
        modules_.emplace_back(xrt::elf(bytes.data(), bytes.size()));
        kernels_.push_back(xrt::ext::kernel(context, modules_.back(), "MLIR_AIE"));
        bytes_.push_back((bytes.size() + 4095) / 4096 * 4096);
      } else {
        xrt::bo &bo = instr_.emplace_back(device, k.words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        std::memcpy(bo.map<void *>(), k.words.data(), k.words.size() * 4);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    }
    packets_ = launch::packet_buffers(device, kernel, r);
  }

  /// A run of kernel k: opcode, instructions and control packets set; arguments
  /// 3 and up are the caller's.
  std::unique_ptr<xrt::run> run(size_t k) const {
    auto r = std::make_unique<xrt::run>(elf_ ? kernels_[k] : kernel_);
    r->set_arg(0, 3u); // run this instruction stream
    if (elf_) {
      r->set_arg(1, 0);
      r->set_arg(2, 0);
    } else {
      r->set_arg(1, instr_[k]);
      r->set_arg(2, (uint32_t)result_.kernels[k].words.size());
    }
    launch::set_packets(*r, result_.kernels[k], packets_[k]);
    return r;
  }

  bool elf() const { return elf_; }

  /// Device heap a run of kernel k holds (ELF flow: its own copy of the ELF, in
  /// 4 KB pages; the process has 64 MB and ~2,040 runs). 0 in the xclbin flow.
  size_t run_bytes(size_t k) const { return elf_ ? bytes_[k] : 0; }

  /// Where a runlist of these kernels must end, from `first`, to stay within
  /// `budget` bytes and `max_runs` runs (launch::Session's ELF chunks); the
  /// whole list in the xclbin flow.
  size_t chunk_end(const std::vector<size_t> &ks, size_t first, size_t budget = 24u << 20,
                   size_t max_runs = 1024) const {
    if (!elf_)
      return ks.size();
    size_t bytes = 0, i = first;
    while (i < ks.size() && (i == first || (bytes + bytes_[ks[i]] <= budget && i - first < max_runs)))
      bytes += bytes_[ks[i++]];
    return i;
  }

private:
  xrt::kernel kernel_;
  const launch::Result &result_;
  bool elf_;
  std::vector<xrt::module> modules_;
  std::vector<xrt::kernel> kernels_;
  std::vector<size_t> bytes_;
  std::vector<xrt::bo> instr_;
  std::vector<xrt::bo> packets_;
};

} // namespace ew
