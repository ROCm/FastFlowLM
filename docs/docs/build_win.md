---
layout: docs
title: Build from Source (Windows)
nav_order: 3
has_children: false
---

# Building FastFlowLM from Source on Windows

This page is for building `flm.exe` yourself, for example to work on FastFlowLM.
If you only want to run models, install the packaged release instead; see
[Get Started (Windows)](../install_win/).

The build is the same one CI uses for the Windows release. The full manual
walkthrough, including every dependency and the alternative dependency routes,
is in [`src/WinSetup.md`](https://github.com/ROCm/FastFlowLM/blob/main/src/WinSetup.md)
in the repository.

---

## ⚙️ Requirements

- 🖥️ **OS:** Windows 11 x64
- 🌐 **Network:** the first build downloads codec sources and Rust crates
- ⚡ **To run the result:** an AMD Ryzen™ AI NPU with the NPU driver installed (see [Get Started (Windows)](../install_win/)). Building does not need the NPU or its driver.

---

## 🚀 Quick Start

`src\WinSetup.ps1` sets up the build machine and builds `flm.exe` in one go.
Run it from a regular 64-bit **PowerShell** window. Do not use "Developer
PowerShell for VS": that shortcut targets x86, and the script sets up its own
x64 build environment.

You need Git to clone the repository. On a fresh machine, install it with
`winget install Git.Git` and open a new PowerShell window. The script
installs everything else.

```powershell
git clone https://github.com/ROCm/FastFlowLM.git C:\dev\FastFlowLM
cd C:\dev\FastFlowLM
powershell -ExecutionPolicy Bypass -File src\WinSetup.ps1 -InstallPrereqs
```

`-InstallPrereqs` uses winget to install any tools that are missing. Visual
Studio Build Tools installs machine-wide and shows a UAC prompt, which an
administrator has to approve. If you leave the switch off, the script lists
what is missing and stops.

When the script finishes, `src\out` holds a runnable build:

```powershell
cd src\out
.\flm.exe validate
.\flm.exe run llama3.2:1b
```

Rerunning the script is safe. Steps whose output already exists are skipped,
so a rerun only rebuilds what changed.

> **Notes:**
> - Keep the clone path short, for example `C:\dev\FastFlowLM`. The Rust and codec build trees nest deeply, and a long path can exceed the Windows 260-character path limit.
> - The dependencies go under `C:\dev`, because the CMake files look for them there.

---

## 🧰 What the Script Does

1. **Checks the tools**:
   - Git
   - Visual Studio 2026 or 2022 with **Desktop development with C++** (the free Build Tools edition is enough)
   - CMake and Ninja (the copies bundled with Visual Studio work)
   - Rust through rustup, with the `stable-x86_64-pc-windows-msvc` toolchain
2. **Checks out the submodules** (`third_party/tokenizers-cpp`, `third_party/xgrammar`).
3. **Sets up the native dependencies under `C:\dev`**:
   - Boost 1.88.0 with Boost.ProgramOptions built
   - the XRT headers at a pinned commit
   - `xrt_coreutil.lib`, made from a `.def` file in the repository

   It runs the same script CI uses, `.github\script\ci\setup_windows_deps.ps1`.
   The curl headers are part of the repository, so you don't need vcpkg.
4. **Configures and builds**. It runs `cmake --preset windows-ninja` in an x64 Visual Studio developer shell.
5. **Stages the runtime files** in `src\out`: `flm.exe`, the engine DLLs, `model_list.json`, `model_info.json` and `xclbins`.

---

## 🔧 Script Options

| Option | Effect |
|---|---|
| `-Variant XRT` | The default build, the same as the release package. Includes the hip-ep (GPU) backend; pass `-NoOgaHipep` to leave it out. Builds into `src\build`. |
| `-Variant HRX` | Builds against the pinned HRX runtime release. Builds into `src\build-hrx`. |
| `-Variant Source` | Builds the Gemma 4 engine (`gemma4e_npu.dll`) from source instead of using the prebuilt DLL. Builds into `src\build-models-from-source`. |
| `-BuildDir <dir>` | Uses a different build directory under `src`. |
| `-Clean` | Deletes the build directory and `src\out` first. |
| `-Jobs <n>` | Sets the number of parallel build jobs. The default is all cores, or 2 for `Source`. |
| `-SkipBuild` | Sets up the tools and dependencies only. |
| `-NoStage` | Skips copying the runtime files into `src\out`. |

All variants stage into the same `src\out`, so it holds whichever variant
was built last.

---

## 🛠️ Building by Hand

Once the dependencies are in place, you can rebuild without the script. Open
an **x64** developer environment, for example the "x64 Native Tools Command
Prompt for VS", then run:

```powershell
cd src
cmake --preset windows-ninja -DFLM_ENABLE_OGA_HIPEP=ON
cmake --build build --parallel
```

You can also use the Visual Studio generator, which does not need a developer
shell:

```powershell
cd src
cmake --preset windows-vs18 -DFLM_ENABLE_OGA_HIPEP=ON   # VS 2026; use windows-default for VS 2022
cmake --build build --config Release --parallel
```

For the manual dependency setup, the HRX and Source variants, staging, and
the vcpkg and conda dependency routes, see
[`src/WinSetup.md`](https://github.com/ROCm/FastFlowLM/blob/main/src/WinSetup.md).

---

## 🩺 Troubleshooting

- **`LNK1104: cannot open file 'kernel32.lib'`.** The developer shell could not find the Windows SDK. This happens on machines where policy blocks `reg.exe`. `WinSetup.ps1` detects this and adds the SDK paths itself. For a manual Ninja build, use the Visual Studio generator instead.
- **"This is a 32-bit PowerShell" or "This developer shell targets x86".** Run the script from a regular 64-bit PowerShell window instead.
- **A build directory configured with another generator.** A build directory stays tied to the generator that first configured it. Rerun with `-Clean`, or pick another directory with `-BuildDir`.
