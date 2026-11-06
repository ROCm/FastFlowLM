<#
.SYNOPSIS
    Lays out the native build dependencies of the Windows XRT build under
    -Root (C:\dev by default), where src/CMakeLists.txt and
    src/detail/CMakeLists.txt look for them on a CI runner without a vcpkg
    toolchain.

.DESCRIPTION
    The self-hosted runner was set up by hand; this recreates the same layout
    on a GitHub-hosted windows-2025-vs2026 runner:

      boost_1_88_0\                     Boost 1.88.0 headers
      boost_1_88_0\stage\lib\           libboost_program_options-vc143-mt-x64-1_88.lib
      XRT\src\runtime_src\core\include\ XRT headers at the commit the runner has
      xrtNPUfromDLL\xrt_coreutil.lib    import library for the NPU driver's xrt_coreutil.dll

    curl needs nothing here: its headers and the .def for the shipped
    libcurl.dll are committed under third_party/curl.

    Each part is skipped when its output already exists, so a restored cache
    only fills in what is missing. Needs git, cmake, and Visual Studio with
    the C++ tools. Boost is unpacked with 7z when it is on PATH, otherwise
    with the tar.exe that ships with Windows.

.PARAMETER Root
    Directory to lay the dependencies out under. CMake expects C:\dev.
#>

[CmdletBinding()]
param(
    [string]$Root = "C:\dev"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$DepsDir = Join-Path $PSScriptRoot "..\..\windows-deps" | Resolve-Path

# Boost: GitHub release archive, pinned by the SHA-256 Boost publishes beside it.
$BoostUrl = "https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-b2-nodocs.7z"
$BoostSha256 = "6956c4283fa07db440b0cefea08a7db9a1b909b250e4f2403cd0ec52a1a1b03e"
# XRT: the commit whose headers the self-hosted runner builds against.
$XrtRepo = "https://github.com/Xilinx/XRT.git"
$XrtCommit = "77c7088d804602a53c3eb489b9cb37b709bcd751"

# Native commands run with ErrorActionPreference=Continue: when the output
# is redirected, Windows PowerShell 5.1 turns every stderr line (git
# progress, for one) into an error record, and under Stop the first one
# would abort the script. Success is judged by the exit code instead.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command, [switch]$NoThrow)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0 -and -not $NoThrow) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force $Root | Out-Null
$Root = (Resolve-Path $Root).Path
# RUNNER_TEMP is only set on a runner; Join-Path throws on a null path, so
# pick the base first (src/WinSetup.ps1 runs this on developer machines).
$TempBase = $env:RUNNER_TEMP
if (-not $TempBase) {
    $TempBase = [IO.Path]::GetTempPath()
}
$Scratch = Join-Path $TempBase "flm-windows-deps"
New-Item -ItemType Directory -Force $Scratch | Out-Null

# --- Boost 1.88.0 ----------------------------------------------------------
$BoostDir = Join-Path $Root "boost_1_88_0"
$BoostLib = Join-Path $BoostDir "stage\lib\libboost_program_options-vc143-mt-x64-1_88.lib"
if (-not (Test-Path $BoostLib)) {
    if (-not (Test-Path (Join-Path $BoostDir "boost\version.hpp"))) {
        Write-Host "Downloading Boost 1.88.0"
        $archive = Join-Path $Scratch "boost.7z"
        Invoke-WebRequest -Uri $BoostUrl -OutFile $archive
        $hash = (Get-FileHash $archive -Algorithm SHA256).Hash
        if ($hash -ne $BoostSha256) {
            throw "Boost archive SHA-256 is $hash, expected $BoostSha256"
        }
        $extract = Join-Path $Scratch "boost"
        if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
        if (Get-Command 7z -ErrorAction SilentlyContinue) {
            Invoke-Native "7z" { 7z x $archive "-o$extract" -y -bso0 -bsp0 }
        } else {
            # No 7-Zip (it needs an administrator to install): the bsdtar
            # that ships with Windows reads .7z too. It skips one Boost.Wave
            # test file with a non-ASCII name and then exits 1, so judge the
            # result by the headers instead of the exit code.
            Write-Host "7z not found; unpacking with tar.exe"
            New-Item -ItemType Directory -Force $extract | Out-Null
            Invoke-Native "tar" { & "$env:SystemRoot\System32\tar.exe" -xf $archive -C $extract } -NoThrow
            if (-not (Test-Path (Join-Path $extract "boost-1.88.0\boost\program_options.hpp"))) {
                throw "tar.exe could not unpack $archive; install 7-Zip and rerun"
            }
        }
        if (Test-Path $BoostDir) { Remove-Item $BoostDir -Recurse -Force }
        Move-Item (Join-Path $extract "boost-1.88.0") $BoostDir
    }

    Write-Host "Building Boost.ProgramOptions"
    $build = Join-Path $Scratch "boost-po-build"
    Invoke-Native "boost_program_options configure" {
        cmake -S (Join-Path $DepsDir "boost_program_options") -B $build "-DBOOST_ROOT=$BoostDir"
    }
    Invoke-Native "boost_program_options build" {
        cmake --build $build --config Release --parallel
    }
    $stage = Join-Path $BoostDir "stage\lib"
    New-Item -ItemType Directory -Force $stage | Out-Null
    Copy-Item (Join-Path $build "Release\libboost_program_options-vc143-mt-x64-1_88.lib") $stage -Force
}

# --- XRT headers -----------------------------------------------------------
# Only src/runtime_src/core/include is needed. The checkout lacks
# xrt/detail/version.h, which XRT's own build generates; the copy in
# .github/windows-deps is the one generated on the self-hosted runner.
$XrtDir = Join-Path $Root "XRT"
$XrtInclude = Join-Path $XrtDir "src\runtime_src\core\include"
if (-not (Test-Path (Join-Path $XrtInclude "xrt\detail\version.h"))) {
    Write-Host "Fetching XRT headers at $XrtCommit"
    if (Test-Path $XrtDir) { Remove-Item $XrtDir -Recurse -Force }
    New-Item -ItemType Directory -Force $XrtDir | Out-Null
    Invoke-Native "git init" { git -C $XrtDir init --quiet }
    Invoke-Native "git sparse-checkout" { git -C $XrtDir sparse-checkout set src/runtime_src/core/include }
    Invoke-Native "git fetch XRT" { git -C $XrtDir fetch --quiet --depth 1 --filter=blob:none $XrtRepo $XrtCommit }
    Invoke-Native "git checkout XRT" { git -C $XrtDir -c advice.detachedHead=false checkout --quiet FETCH_HEAD }
    $head = git -C $XrtDir rev-parse HEAD
    if ($head -ne $XrtCommit) {
        throw "Fetched XRT at $head, expected $XrtCommit"
    }
    Copy-Item (Join-Path $DepsDir "xrt_version.h") (Join-Path $XrtInclude "xrt\detail\version.h") -Force
}

# --- xrt_coreutil import library ------------------------------------------
# xrt_coreutil.dll comes with the NPU driver, so only its import library is
# needed at build time; make it from the export list.
$XrtLibDir = Join-Path $Root "xrtNPUfromDLL"
$XrtLib = Join-Path $XrtLibDir "xrt_coreutil.lib"
if (-not (Test-Path $XrtLib)) {
    Write-Host "Generating xrt_coreutil.lib"
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) {
        throw "No Visual Studio with the C++ x64 tools found"
    }
    $libExe = Get-ChildItem (Join-Path $vs "VC\Tools\MSVC\*\bin\Hostx64\x64\lib.exe") |
        Sort-Object FullName | Select-Object -Last 1
    if (-not $libExe) {
        throw "lib.exe not found under $vs"
    }
    New-Item -ItemType Directory -Force $XrtLibDir | Out-Null
    Invoke-Native "lib /def" {
        & $libExe.FullName /nologo /machine:x64 "/def:$(Join-Path $DepsDir 'xrt_coreutil.def')" "/out:$XrtLib"
    }
}

Write-Host "Windows build dependencies are ready under $Root"
