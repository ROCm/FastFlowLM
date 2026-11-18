# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

<#
.SYNOPSIS
    Builds every FLM_DLL engine on Windows (XRT), the counterpart of build.sh.

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

.PARAMETER DepsRoot
    Root of the Windows dependency layout (default: C:\dev). CMake caches the
    paths it derives, so add -Clean when changing it.

.PARAMETER Clean
    Delete build/ first, as build.sh does.
#>

[CmdletBinding()]
param(
    [string]$DepsRoot = "C:\dev",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Clean -and (Test-Path build)) { Remove-Item -Recurse -Force build }

$priority = @("q4nx", "mha", "gemm", "dequant", "lm_head")
$root = ($DepsRoot -replace '\\', '/')

function Build-Unit([string]$name) {
    Write-Host "== $name"
    # Out-Host: keep cmake's output out of the function's return value.
    & cmake -S "detail/$name" -B "build/$name" -A x64 "-DFLM_WIN_DEPS_ROOT=$root" | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    & cmake --build "build/$name" --config Release --parallel | Out-Host
    return ($LASTEXITCODE -eq 0)
}

foreach ($name in $priority) {
    if (-not (Build-Unit $name)) {
        Write-Host "Priority unit $name failed; the engines would link without it. Stopping."
        exit 1
    }
}

$failed = @()
foreach ($dir in Get-ChildItem -Directory detail) {
    if ($priority -contains $dir.Name) { continue }
    if (-not (Test-Path (Join-Path $dir.FullName "CMakeLists.txt"))) { continue }
    if (-not (Build-Unit $dir.Name)) { $failed += $dir.Name }
}

if ($failed.Count -ne 0) {
    Write-Host "Build failures in:"
    $failed | ForEach-Object { Write-Host "  - detail/$_" }
    exit 1
}
Write-Host "All builds succeeded."
