<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# FastFlowLM (dev)

The engine libraries of FastFlowLM, one per model family (`detail/`). On Linux, build them with
`bash build.sh` (every unit) or `bash build_<model>.sh` (one engine and what it needs).

> **Windows: not tested in this release.** The Windows steps below are the old notes, kept for
> reference until they are checked on a Windows machine.

This repository contains:
    - NPU xclbin code
    - NPU xclbin driver host code

# Setup instruction
## 1. Create a new folder "C:/dev"

## 1. Install visual studio 2022 Community
Inside the "C:/dev", do 
```bash
git clone https://github.com/microsoft/vcpkg.git

cd vcpkg; .\bootstrap-vcpkg.bat
.\vcpkg install curl
.\vcpkg install ffmpeg[zlib,avcodec,avformat,swscale]
```
## 2. Install boost
Down the **zip file** [1.88](https://www.boost.org/releases/1.88.0/)

Extract to "C:/dev"
Open **Developer PowerShell of VS2022**
```
cd "C:\dev\boost_xxx"

.\bootstrap.bat
# or .\bootstrap.bat vc143 for vs 2018

./b2 --build-type=complete link=static,shared
```
## 3. xrt setup
1. clone the xrt  in C:/dev
```shell
C:\dev> git clone https://github.com/Xilinx/XRT -b 2025.1_mcdm_0430

```
Follow [step4](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md)
in WSL

```shell
sudo apt update
sudo apt install mingw-w64-tools
mkdir /mnt/c/dev/xrtNPUfromDLL; cd /mnt/c/dev/xrtNPUfromDLL

cp /mnt/c/Windows/System32/xrt_coreutil.dll .

gendef xrt_coreutil.dll
```
Then follow [Step 2 of prepare for windows 11](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md)

Got to a VisutalStudio terminal
```shell
C:\dev\xrtNPUfromDLL> lib /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib
```

## 4. Cmake 
## 5. Rust for windows

# Build instruction
### 1. install
```bash
bash build.sh        # from flm-dev-kit/FLM_DLL; on Windows through WSL: wsl bash build.sh
```

### Note:
If the build stops with `Cannot open include file: 'xrt/detail/version.h'`, the XRT checkout lacks
that generated header. FastFlowLM's `.github/script/ci/setup_windows_deps.ps1` shows how CI
provides it (it copies `.github/windows-deps/xrt_version.h` into place).
