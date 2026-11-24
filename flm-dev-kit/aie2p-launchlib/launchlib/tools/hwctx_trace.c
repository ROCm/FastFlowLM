// hwctx_trace.c - log which hw_context each NPU command goes to
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//   gcc -O2 -shared -fPIC hwctx_trace.c -o hwctx_trace.so -ldl
//   HWCTX_TRACE=out.txt LD_PRELOAD=./hwctx_trace.so flm serve ...
//
// An LD_PRELOAD wrapper of ioctl() for the amdxdna driver. One line per event,
// appended to $HWCTX_TRACE (all threads, one file):
//
//   C <ns> <handle> <tiles>     a hw_context was made (DRM_AMDXDNA_CREATE_HWCTX)
//   D <ns> <handle>             it was destroyed
//   X <ns> <handle> <commands>  commands submitted to it (DRM_AMDXDNA_EXEC_CMD)
//
// <ns> is CLOCK_MONOTONIC. Consecutive X lines with different handles are where
// the firmware has to switch the array between hw_contexts (hwctx_switches.py
// counts them). Submission order, not execution order.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// From the driver's uapi (include/uapi/drm/amdxdna_accel.h).
#define AMDXDNA_NR_BASE 0x40
#define NR_CREATE_HWCTX (AMDXDNA_NR_BASE + 0)
#define NR_DESTROY_HWCTX (AMDXDNA_NR_BASE + 1)
#define NR_EXEC_CMD (AMDXDNA_NR_BASE + 6)

struct create_hwctx {
  uint64_t ext, ext_flags, qos_p;
  uint32_t umq_bo, log_buf_bo, max_opc, num_tiles, mem_size, umq_doorbell, handle;
};
struct destroy_hwctx {
  uint32_t handle;
};
struct exec_cmd {
  uint64_t ext, ext_flags;
  uint32_t hwctx, type;
  uint64_t cmd_handles, args;
  uint32_t cmd_count, arg_count;
  uint64_t seq;
};

static int (*real_ioctl)(int, unsigned long, ...);
static FILE *out;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void emit(const char *fmt, ...) {
  pthread_mutex_lock(&lock);
  if (!out) {
    const char *path = getenv("HWCTX_TRACE");
    out = fopen(path ? path : "hwctx_trace.txt", "a");
  }
  if (out) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(out, fmt, ap);
    va_end(ap);
    fflush(out);
  }
  pthread_mutex_unlock(&lock);
}

int ioctl(int fd, unsigned long req, ...) {
  va_list ap;
  va_start(ap, req);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  if (!real_ioctl)
    real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
  const unsigned type = (req >> 8) & 0xFF, nr = req & 0xFF;
  const int is_drm = type == 'd';
  if (is_drm && nr == NR_EXEC_CMD) {
    const struct exec_cmd *c = arg;
    emit("X %llu %u %u\n", (unsigned long long)now_ns(), c->hwctx, c->cmd_count);
  }
  const int ret = real_ioctl(fd, req, arg);
  if (is_drm && ret == 0 && nr == NR_CREATE_HWCTX) {
    const struct create_hwctx *c = arg;
    emit("C %llu %u %u\n", (unsigned long long)now_ns(), c->handle, c->num_tiles);
  } else if (is_drm && nr == NR_DESTROY_HWCTX) {
    const struct destroy_hwctx *c = arg;
    emit("D %llu %u\n", (unsigned long long)now_ns(), c->handle);
  }
  return ret;
}
