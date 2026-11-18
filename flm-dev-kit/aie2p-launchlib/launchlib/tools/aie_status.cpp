//===- aie_status.cpp - dump the AIE array status from the amdxdna driver -===//
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
//   aie_status [--raw]
//
// DRM_IOCTL_AMDXDNA_GET_INFO with DRM_AMDXDNA_QUERY_AIE_METADATA (tile counts)
// and DRM_AMDXDNA_QUERY_AIE_STATUS (the firmware's status of every column of
// the open partitions). Prints the metadata and the status words per column.
// No XRT, no kernel headers: the uapi layout is copied here.
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct GetInfo {
  uint32_t param, buffer_size;
  uint64_t buffer;
};
struct TileMeta {
  uint16_t row_count, row_start, dma_channel_count, lock_count, event_reg_count, pad[3];
};
struct AieMeta {
  uint32_t col_size;
  uint16_t cols, rows;
  uint32_t major, minor;
  TileMeta core, mem, shim;
};
struct AieStatus {
  uint64_t buffer;
  uint32_t buffer_size, cols_filled;
};
constexpr uint32_t kQueryAieStatus = 0, kQueryAieMetadata = 1;
constexpr unsigned long kIoctlGetInfo = (3ul << 30) | (sizeof(GetInfo) << 16) | ('d' << 8) | (0x40 + 7);

bool get_info(int fd, uint32_t param, void *buf, uint32_t size) {
  GetInfo g{param, size, (uint64_t)(uintptr_t)buf};
  if (ioctl(fd, kIoctlGetInfo, &g) != 0) {
    std::perror(param == kQueryAieStatus ? "QUERY_AIE_STATUS" : "QUERY_AIE_METADATA");
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const bool raw = argc > 1 && std::string(argv[1]) == "--raw";
  const int fd = open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    std::perror("open");
    return 1;
  }
  AieMeta m{};
  if (!get_info(fd, kQueryAieMetadata, &m, sizeof m))
    return 1;
  std::printf("columns %u, rows %u, column status %u B, AIE %u.%u\n", m.cols, m.rows, m.col_size, m.major, m.minor);
  for (auto [name, t] : {std::pair{"core", m.core}, std::pair{"mem", m.mem}, std::pair{"shim", m.shim}})
    std::printf("  %-4s rows %u from %u: %u DMA channels, %u locks, %u event registers\n", name, t.row_count,
                t.row_start, t.dma_channel_count, t.lock_count, t.event_reg_count);
  std::vector<uint8_t> buf((size_t)m.col_size * m.cols);
  AieStatus s{(uint64_t)(uintptr_t)buf.data(), (uint32_t)buf.size(), 0};
  if (!get_info(fd, kQueryAieStatus, &s, sizeof s))
    return 1;
  std::printf("columns filled: %#x\n", s.cols_filled);
  if (raw)
    for (uint32_t c = 0; c < m.cols; c++) {
      if (!(s.cols_filled & (1u << c)))
        continue;
      std::printf("column %u:", c);
      const uint32_t *w = reinterpret_cast<const uint32_t *>(buf.data() + (size_t)c * m.col_size);
      for (uint32_t i = 0; i < m.col_size / 4; i++)
        std::printf("%s%08x", i % 8 ? " " : "\n  ", w[i]);
      std::printf("\n");
    }
  return 0;
}
