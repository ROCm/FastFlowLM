# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

<#
.SYNOPSIS
    Builds every FLM_DLL engine on Windows (XRT or HRX), the counterpart of build.sh.

.DESCRIPTION
    Builds the priority units (q4nx mha gemm dequant lm_head) first and stops
    at the first failure among them: the engine CMakeLists only warn when a
    priority .lib is missing and would then link an incomplete DLL. Then builds
    every other detail/*/ with a CMakeLists.txt, lists the failures and exits 1
    if there were any. Output: build/lib/<name>.dll and .lib.

    Run from a Developer PowerShell for Visual Studio, so that cmake knows the
    installed Visual Studio. Dependencies: the C:\dev layout from FastFlowLM's
    .github/script/ci/setup_windows_deps.ps1, or -DepsRoot pointing at the same
    layout elsewhere (see cmake/flm_xrt_win.cmake).

    -Runtime HRX first runs ..\..\hrx-integration\fetch-hrx-release.ps1, which
    downloads (or reuses) the HRX package pinned in hrx-release.env, then
    configures every unit with -DFLM_USE_HRX=ON and that package on
    CMAKE_PREFIX_PATH (cmake/flm_hrx.cmake), as src\WinSetup.ps1 -Variant HRX
    does for flm. The CMake trees go to build-hrx/<name> instead of
    build/<name>, so the two runtimes never share a CMake cache. The DLLs
    land in build/lib either way (the engine CMakeLists fix that path);
    build/lib/.runtime records which runtime they are for, and a build for
    the other runtime first deletes build/lib and every build/<name>/lib so
    an engine cannot link a priority .lib of the other runtime.

.PARAMETER Runtime
    XRT (default) or HRX.

.PARAMETER Engines
    Build only these engines (detail/ directory names, comma or space
    separated) after the priority units, like build.sh's FLM_ENGINES. Empty
    or "all" (default): every engine.

.PARAMETER DepsRoot
    Root of the Windows dependency layout (default: C:\dev). CMake caches the
    paths it derives, so add -Clean when changing it.

.PARAMETER Clean
    Delete build/ (and, with -Runtime HRX, build-hrx/) first, as build.sh
    deletes build/.

.EXAMPLE
    flm-dev-kit\FLM_DLL\build.ps1
.EXAMPLE
    flm-dev-kit\FLM_DLL\build.ps1 -Runtime HRX
#>

[CmdletBinding()]
param(
    [ValidateSet('XRT', 'HRX')]
    [string]$Runtime = 'XRT',
    [string]$Engines = '',
    [string]$DepsRoot = "C:\dev",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$priority = @("q4nx", "mha", "gemm", "dequant", "lm_head")
$root = ($DepsRoot -replace '\\', '/')
$cmakeRoot = if ($Runtime -eq 'HRX') { 'build-hrx' } else { 'build' }
$runtimeArgs = @("-DFLM_WIN_DEPS_ROOT=$root", '-DFLM_USE_HRX=OFF')

function Build-Unit([string]$name) {
    # As src\WinSetup.ps1's Invoke-Native: with Stop, Windows PowerShell 5.1
    # turns a native command's stderr line (the q4nx/dequant CMakeLists'
    # npu_utils.lib warning) into a terminating error when the script's output
    # is redirected. Function scope keeps this local; exit codes decide.
    $ErrorActionPreference = 'Continue'
    Write-Host "== $name ($Runtime)"
    # A tree configured another way (the WSL branch of detail/<name>/Makefile
    # configures build/<name> without -A, and its HRX build leaves XRT_*_DIR
    # cached empty) would fail to configure here: start it afresh.
    $tree = "$cmakeRoot/$name"
    $cache = "$tree/CMakeCache.txt"
    if (Test-Path -LiteralPath $cache) {
        $lines = Get-Content -LiteralPath $cache
        $platformOk = [bool]($lines | Where-Object { $_ -ceq 'CMAKE_GENERATOR_PLATFORM:INTERNAL=x64' })
        $hrxLine = $lines | Where-Object { $_ -match '^FLM_USE_HRX:[A-Z]+=' } | Select-Object -First 1
        $cachedHrx = [bool]($hrxLine -and (($hrxLine -split '=', 2)[1] -match '^(ON|TRUE|YES|Y|1)$'))
        if (-not $platformOk -or $cachedHrx -ne ($Runtime -eq 'HRX')) {
            Write-Host "$tree was configured differently; reconfiguring from scratch."
            Remove-Item -Recurse -Force -LiteralPath $tree
        }
    }
    # Out-Host: keep cmake's output out of the function's return value.
    & cmake -S "detail/$name" -B $tree -A x64 @runtimeArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    & cmake --build $tree --config Release --parallel | Out-Host
    return ($LASTEXITCODE -eq 0)
}

# The pinned HRX package's prefix, the way src\WinSetup.ps1 reads it: the
# fetch script prints HRX_CMAKE_PREFIX=<dir> with Write-Host (stream 6).
function Get-HrxPrefix {
    $fetch = Join-Path $PSScriptRoot '..\..\hrx-integration\fetch-hrx-release.ps1'
    if (-not (Test-Path -LiteralPath $fetch)) { throw "$fetch not found" }
    $lines = & $fetch 6>&1 | ForEach-Object { $_.ToString() }
    $lines | ForEach-Object { Write-Host $_ }
    $prefix = ($lines | Where-Object { $_ -like 'HRX_CMAKE_PREFIX=*' } |
        Select-Object -First 1) -replace '^HRX_CMAKE_PREFIX=', ''
    if (-not $prefix) { throw 'fetch-hrx-release.ps1 did not report HRX_CMAKE_PREFIX' }
    return $prefix
}

$selected = @($Engines -split '[,\s]+' | Where-Object { $_ -and $_ -ne 'all' })

# Push/Pop: the caller's shell is back in its own folder afterwards, even on failure.
Push-Location $PSScriptRoot
$code = 0
try {
    foreach ($name in $selected) {
        if ($name -notmatch '^[A-Za-z0-9_]+$' -or
            -not (Test-Path -LiteralPath (Join-Path 'detail' (Join-Path $name 'CMakeLists.txt')))) {
            throw "-Engines names $name, but detail/$name/CMakeLists.txt does not exist"
        }
    }

    if ($Clean) {
        foreach ($dir in (@('build', $cmakeRoot) | Select-Object -Unique)) {
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        }
    }

    if ($Runtime -eq 'HRX') {
        $hrxPrefix = (Get-HrxPrefix) -replace '\\', '/'
        # -U: find_package(hrx) keeps a cached hrx_DIR over a new
        # CMAKE_PREFIX_PATH, so an older package (or a from-source HRX found
        # earlier) would stay linked.
        $runtimeArgs = @('-DFLM_USE_HRX=ON', '-Uhrx_DIR', "-DCMAKE_PREFIX_PATH=$hrxPrefix")
    }

    # Outputs from the other runtime (no stamp: an XRT build from before
    # -Runtime existed) go before anything links against them.
    $stamp = 'build/lib/.runtime'
    $previous = ''
    if (Test-Path $stamp) {
        $previous = (Get-Content $stamp -Raw).Trim()
    } elseif (Test-Path 'build/lib') {
        $previous = 'XRT'
    }
    if ($previous -and $previous -ne $Runtime) {
        Write-Host "build/lib holds $previous libraries; deleting the outputs before the $Runtime build."
        Remove-Item -Recurse -Force 'build/lib'
        Get-ChildItem -Directory build -ErrorAction SilentlyContinue | ForEach-Object {
            $lib = Join-Path $_.FullName 'lib'
            if (Test-Path $lib) { Remove-Item -Recurse -Force $lib }
        }
    }
    New-Item -ItemType Directory -Force 'build/lib' | Out-Null
    Set-Content -Path $stamp -Value $Runtime

    foreach ($name in $priority) {
        if (-not (Build-Unit $name)) {
            Write-Host "Priority unit $name failed; the engines would link without it. Stopping."
            $code = 1
            break
        }
    }

    if ($code -eq 0) {
        $failed = @()
        foreach ($dir in Get-ChildItem -Directory detail) {
            if ($priority -contains $dir.Name) { continue }
            if ($selected.Count -ne 0 -and $selected -notcontains $dir.Name) { continue }
            if (-not (Test-Path (Join-Path $dir.FullName "CMakeLists.txt"))) { continue }
            if (-not (Build-Unit $dir.Name)) { $failed += $dir.Name }
        }

        if ($failed.Count -ne 0) {
            Write-Host "Build failures in:"
            $failed | ForEach-Object { Write-Host "  - detail/$_" }
            $code = 1
        } else {
            Write-Host "All $Runtime builds succeeded."
        }
    }
} catch {
    Write-Host "build.ps1: $_"
    $code = 1
} finally {
    Pop-Location
}
exit $code
