//===- spawn.hpp - start, stop and wait for a child process -----*- C++ -*-===//
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
// The stress test runs a second copy of itself as another NPU user: fork and
// exec on Linux, CreateProcess on Windows.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ew {

class Child {
public:
  /// Starts `exe` with `args` (not including the program name).
  Child(const std::string &exe, const std::vector<std::string> &args) {
    std::fflush(stdout);
#if defined(_WIN32)
    std::string cmd = "\"" + exe + "\"";
    for (const std::string &a : args)
      cmd += " \"" + a + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof si;
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi_))
      throw std::runtime_error("CreateProcess failed: " + cmd);
#else
    std::vector<char *> argv{const_cast<char *>(exe.c_str())};
    for (const std::string &a : args)
      argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    pid_ = fork();
    if (pid_ == 0) {
      execv(exe.c_str(), argv.data());
      _exit(127);
    }
    if (pid_ < 0)
      throw std::runtime_error("fork failed");
#endif
  }
  Child(const Child &) = delete;
  Child &operator=(const Child &) = delete;
  ~Child() {
    if (!done_) {
      stop();
      wait();
    }
  }

  /// Ends the child without waiting for it.
  void stop() {
#if defined(_WIN32)
    TerminateProcess(pi_.hProcess, 1);
#else
    if (!reaped_)
      kill(pid_, SIGTERM);
#endif
  }

  /// True while the child runs; does not wait.
  bool running() {
#if defined(_WIN32)
    return !done_ && WaitForSingleObject(pi_.hProcess, 0) == WAIT_TIMEOUT;
#else
    if (done_ || reaped_)
      return false;
    if (waitpid(pid_, &status_, WNOHANG) != pid_)
      return true;
    reaped_ = true;
    return false;
#endif
  }

  /// Waits for the child; returns its exit code.
  int wait() {
    done_ = true;
#if defined(_WIN32)
    WaitForSingleObject(pi_.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi_.hProcess, &code);
    CloseHandle(pi_.hThread);
    CloseHandle(pi_.hProcess);
    return (int)code;
#else
    if (!reaped_)
      waitpid(pid_, &status_, 0);
    reaped_ = true;
    return WIFEXITED(status_) ? WEXITSTATUS(status_) : -1;
#endif
  }

private:
#if defined(_WIN32)
  PROCESS_INFORMATION pi_{};
#else
  pid_t pid_ = -1;
  int status_ = 0;
  bool reaped_ = false;
#endif
  bool done_ = false;
};

} // namespace ew
