<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# FastFlowLM (dev)

The engine libraries of FastFlowLM, one per model family (`detail/`). On Linux, build them with
`bash build.sh` (every unit) or `bash build_<model>.sh` (one engine and what it needs).

The Linux Makefiles take two settings from [`flags.mk`](flags.mk), on the `make` command line or in
the environment:

- `FLM_MARCH`: the CPU flags. The default is the AVX-512 set FastFlowLM's own engine build uses
  (`-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma`), so a library runs on any AVX-512 CPU;
  gemma, gpt_oss and whisper also need AVX512-BF16, and qwen3_asr needs BF16, VNNI and VBMI (all
  Zen 4+ Ryzen AI hosts have them). Those engines add the flags themselves.
  `FLM_MARCH=-march=native` tunes a build to the machine it is built on, which may then fault
  elsewhere.
- `XRT_ROOT`: the XRT prefix, `/opt/xilinx/xrt` by default. When that is missing and the
  distribution's `libxrt-dev` is installed (headers in `/usr/include/xrt`), `/usr` is used.

## Windows

Builds the XRT or HRX engines as `build/lib/<name>.dll` with Visual Studio's C++ tools. One-time
setup, from the FastFlowLM root in PowerShell:

```powershell
.github\script\ci\setup_windows_deps.ps1        # XRT headers, xrt_coreutil.lib and Boost headers under C:\dev
```

That is the layout FastFlowLM's own Windows build uses. It ships `xrt/detail/version.h`, which a
plain XRT checkout lacks. `aiebu_static.lib` comes from `../../src/lib/xrt/`. Then, in a Developer
PowerShell for Visual Studio:

```powershell
flm-dev-kit\FLM_DLL\build.ps1                     # every engine; priority units first
flm-dev-kit\FLM_DLL\build.ps1 -DepsRoot D:\deps   # the same layout elsewhere (add -Clean when changing it)
flm-dev-kit\FLM_DLL\build.ps1 -Runtime HRX       # HRX engines (-DFLM_USE_HRX=ON)
```

`-Runtime HRX` runs `hrx-integration\fetch-hrx-release.ps1` (downloads the pinned HRX package, or
reuses it) and configures every unit with `-DFLM_USE_HRX=ON -DCMAKE_PREFIX_PATH=<package>`, as
`src\WinSetup.ps1 -Variant HRX` does for `flm`. Its CMake trees live in `build-hrx/` rather than
`build/`, so XRT and HRX never share a CMake cache. The DLLs go to `build/lib/` for both runtimes;
`build/lib/.runtime` names the one they were built for, and switching runtime deletes the previous
runtime's DLLs and `.lib` files first.

One engine (after the priority units `q4nx mha gemm dequant lm_head`, whose `.lib` files it links
from `build/lib/`):

```powershell
cmake -S flm-dev-kit\FLM_DLL\detail\qwen2_npu -B flm-dev-kit\FLM_DLL\build\qwen2_npu -A x64
cmake --build flm-dev-kit\FLM_DLL\build\qwen2_npu --config Release
```

The XRT paths come from [`cmake/flm_xrt_win.cmake`](cmake/flm_xrt_win.cmake): override them with
`-DFLM_WIN_DEPS_ROOT=<dir>`, or `-DXRT_INCLUDE_DIR=<dir> -DXRT_LIB_DIR=<dir>`. Configuring stops
with a message when XRT, `aiebu_static.lib` or the Boost headers are missing. Only Boost's
headers are needed: XRT's `xrt/detail/any.h` uses `boost::any` under MSVC unless `/Zc:__cplusplus`
is set, and the engines build like `flm`, without it.

An engine CMakeLists only warns when a priority `.lib` is missing, and then links an incomplete DLL.
`build.ps1` stops at the first priority failure for that reason.

From WSL, `make` in an engine folder drives the same CMake through `powershell.exe`; that route was
not tested in this release.
