# Building FastFlowLM on Windows

This page covers building `flm.exe` from source on Windows 11 x64. The build
follows the Windows CI build (`.github/actions/win-build`). You do not need
WSL, mlir-aie, or `gendef`.

## Quick start: `WinSetup.ps1`

`src\WinSetup.ps1` performs every step below. Run it from a regular 64-bit
PowerShell window, not "Developer PowerShell for VS": the script sets up its
own x64 build environment. It can be started from any directory.

Git is needed to clone the repository. On a fresh machine, install it first
with `winget install Git.Git` and open a new window. The script installs
everything else.

```powershell
git clone https://github.com/ROCm/FastFlowLM.git C:\dev\FastFlowLM
cd C:\dev\FastFlowLM
powershell -ExecutionPolicy Bypass -File src\WinSetup.ps1 -InstallPrereqs
```

`-InstallPrereqs` installs any missing tools with winget. Some installers ask
for elevation, Visual Studio Build Tools among them. If you leave the switch
off, the script lists what is missing and stops.

When the script finishes, `src\out` contains a runnable build:

```powershell
cd src\out
.\flm.exe validate
.\flm.exe run llama3.2:1b
```

You can rerun the script at any time. Steps whose output already exists are
skipped, so a rerun just rebuilds.

To keep a log, use `Start-Transcript build.log` before running the script. Do
not pipe its output (`*>&1 | Tee-Object`). In Windows PowerShell 5.1 a
redirected stderr line from git can abort the dependency step.

| Option | Effect |
|---|---|
| `-Variant XRT` | The default build, same as the release package. Includes the hip-ep (GPU) backend; pass `-NoOgaHipep` to leave it out. Builds into `src\build`. |
| `-Variant HRX` | Builds against the pinned HRX runtime release (`hrx-integration\hrx-release.env`). Builds into `src\build-hrx`. |
| `-Variant Source` | Like the CI "models from source" package: the Gemma 4 engine (`gemma4e_npu.dll`) is built from source instead of using the prebuilt DLL, and hip-ep is not included. Builds into `src\build-models-from-source`. |
| `-BuildDir <dir>` | Use a different build directory under `src`. |
| `-Clean` | Delete the build directory and `src\out` first. |
| `-Jobs <n>` | Parallel build jobs. The default is all cores, or 2 for `Source`. |
| `-SkipBuild` | Set up tools and dependencies only. |
| `-NoStage` | Do not copy runtime files into `src\out`. |

## What gets set up

### 1. Tools

| Tool | Why | winget ID |
|---|---|---|
| Git | clone and submodules | `Git.Git` |
| Visual Studio 2026 or 2022 with **Desktop development with C++** (the Build Tools edition is enough) | MSVC, Windows SDK, `lib.exe`, `dumpbin`, and the bundled Ninja | `Microsoft.VisualStudio.BuildTools` (2026) |
| CMake 3.22 or newer (4.2 or newer with VS 2026) | build system | `Kitware.CMake` |
| Rust through rustup, toolchain `stable-x86_64-pc-windows-msvc` | tokenizers-cpp builds a Rust crate | `Rustlang.Rustup` |
| 7-Zip | unpacks the Boost archive (only needed until `C:\dev\boost_1_88_0` is built) | `7zip.7zip` |
| Ninja | generator for the `windows-ninja` preset. The copy bundled with VS works. | `Ninja-build.Ninja` |
| NASM (optional) | SIMD code in libjpeg-turbo | `NASM.NASM` |

If Visual Studio is already installed without the C++ workload, add
**Desktop development with C++** to it in the Visual Studio Installer. With
`-InstallPrereqs`, the script does this for you. To install the Build Tools by
hand with the C++ workload:

```powershell
winget install Microsoft.VisualStudio.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

cargo does not have to be on `PATH`. tokenizers-cpp also looks in
`%CARGO_HOME%\bin` and `%USERPROFILE%\.cargo\bin`.

Keep the clone path short, for example `C:\dev\FastFlowLM`. The Rust and codec
build trees nest deeply, and a long root path can exceed MAX_PATH (260
characters).

### 2. Submodules

```powershell
git submodule update --init --recursive
```

This pulls `third_party/tokenizers-cpp` (with msgpack and sentencepiece) and
`third_party/xgrammar`.

### 3. Native dependencies under `C:\dev`

`src/CMakeLists.txt` looks for these by absolute path, so the location is
fixed:

| Path | Contents |
|---|---|
| `C:\dev\vcpkg\installed\x64-windows` | curl headers and `libcurl.lib`, from a vcpkg clone with `vcpkg install curl:x64-windows` |
| `C:\dev\boost_1_88_0` | Boost 1.88.0 headers, plus `stage\lib\libboost_program_options-vc143-mt-x64-1_88.lib` |
| `C:\dev\XRT\src\runtime_src\core\include` | XRT headers at the pinned commit, plus `xrt\detail\version.h` |
| `C:\dev\xrtNPUfromDLL\xrt_coreutil.lib` | import library for the NPU driver's `xrt_coreutil.dll` |

Set up vcpkg and curl:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
C:\dev\vcpkg\bootstrap-vcpkg.bat -disableMetrics
C:\dev\vcpkg\vcpkg.exe install curl:x64-windows
```

Then run the script CI uses to set up the rest. It needs `7z` on `PATH` and
the Visual Studio C++ tools:

```powershell
$env:VCPKG_INSTALLATION_ROOT = 'C:\dev\vcpkg'
powershell -ExecutionPolicy Bypass -File .github\script\ci\setup_windows_deps.ps1
```

That script does four things:

- Downloads the Boost archive and checks its SHA-256.
- Builds Boost.ProgramOptions with `.github/windows-deps/boost_program_options`.
- Sparse-checks-out the XRT headers at the pinned commit, and adds the
  generated `version.h` from `.github/windows-deps`.
- Runs `lib.exe` on `.github/windows-deps/xrt_coreutil.def` to make
  `xrt_coreutil.lib`. The `.def` file is committed, so you need neither the
  NPU driver nor `gendef` to build.

### 4. Configure and build

Ninja and `cl` need an **x64** MSVC environment. The Start Menu "Developer
PowerShell for VS" and "Developer Command Prompt" shortcuts target **x86**,
and an x86 build fails to link against the x64 libraries. Use the
"x64 Native Tools Command Prompt for VS", or enter an x64 shell from a regular
PowerShell:

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
```

If configure then fails with `LNK1104: cannot open file 'kernel32.lib'`, the
developer shell could not find the Windows SDK. This happens when policy
blocks `reg.exe`. `WinSetup.ps1` detects this and adds the SDK paths itself;
the Visual Studio generator below is not affected.

Then build:

```powershell
cd src
cmake --preset windows-ninja -DFLM_ENABLE_OGA_HIPEP=ON
cmake --build build --parallel
```

If you prefer the Visual Studio generator, no developer shell is needed:

```powershell
cmake --preset windows-vs18 -DFLM_ENABLE_OGA_HIPEP=ON   # VS 2026; windows-default for VS 2022
cmake --build build --config Release --parallel
```

A build directory belongs to the generator it was configured with. To switch
generators, delete the directory or use another one (`-B <dir>`).

`FLM_USE_HRX`, `FLM_ENABLE_OGA_HIPEP` and `FLM_BUILD_GEMMA4E` are cached
options. To turn one off in an existing build directory, pass it as `=OFF`;
leaving it out keeps the previous value.

The first build needs internet access. It downloads the image and audio codec
sources (libjpeg-turbo, libwebp, opus, libogg, libvorbis) and the Rust crates.

The version string comes from `FLM_VERSION` in `src/CMakePresets.json`, not
from git tags.

### 5. Stage and run

The build writes `src\build\flm.exe` and copies it to `src\out\flm.exe`. Before
it can run, it needs the engine DLLs and data files next to it. The script
copies the same files as the release package:

```powershell
cd src
Copy-Item lib\xrt\*.dll out -Force
Copy-Item lib\*.dll out -Force
Copy-Item model_list.json, model_info.json out -Force
New-Item -ItemType Directory -Force out\xclbins | Out-Null
Copy-Item xclbins\* out\xclbins -Recurse -Force   # not "xclbins out": that nests on a rerun
```

For HRX, copy `lib\*.dll`, then `lib\hrx\*.dll`, then the `hrx.dll` from
`hrx-integration\.hrx-release`.

For the Source variant, also copy `build-models-from-source\engines\*.dll`
over the prebuilt engine DLLs.

To run it, you need an AMD Ryzen AI NPU with the
[NPU driver](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
installed. `flm.exe validate` checks the driver.

## Other dependency routes

`src/CMakeLists.txt` has two alternatives to the `C:\dev` layout:

- **vcpkg toolchain**
  - Install Boost and curl:
    `vcpkg install boost-program-options curl:x64-windows`
  - Configure with
    `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>\scripts\buildsystems\vcpkg.cmake`.
  - The XRT headers and `xrt_coreutil.lib` still come from `XRT_INCLUDE_DIR`
    and `XRT_LIB_DIR`, which default to the `C:\dev` paths above.
  - vcpkg's `x64-windows` Boost is a DLL. Copy
    `build\boost_program_options-*.dll` into `src\out` along with the files
    in step 5.
- **`-DFLM_WIN_DEPS_PREFIX=<prefix>`**
  - Takes Boost and curl from `<prefix>\include` and `<prefix>\lib`. In
    practice that is a conda env's `Library` dir, set up with
    `conda install -c conda-forge boost libcurl`.
  - The env's `bin` must be on `PATH` when `flm.exe` runs.
  - The XRT headers and `xrt_coreutil.lib` still come from the `C:\dev` paths
    (`XRT_INCLUDE_DIR`, `XRT_LIB_DIR`).
  - This route cannot be combined with the vcpkg toolchain.
