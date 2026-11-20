# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(gpt_oss_npu VERSION 1.0.0 LANGUAGES CXX)

# The Windows build of this test host is not maintained: it compiled the engine
# from an old source layout and linked libraries the host does not use. Build
# the host on Linux with `make` (makefiles/host.mk), or build the gpt_oss_npu
# engine on Windows from FLM_DLL/detail/gpt_oss_npu.
message(FATAL_ERROR "GPTOSS/gpt_npu_bin: the Windows (CMake) build is not supported. "
                    "Build this host on Linux with make.")
