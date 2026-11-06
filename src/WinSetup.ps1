<#
.SYNOPSIS
    Sets up a Windows developer machine from scratch and builds flm.exe.

.DESCRIPTION
    Runs the same steps as the Windows CI build (.github/actions/win-build):

      1. Checks the tools (Git, CMake, Visual Studio C++ tools, Rust, 7-Zip,
         Ninja). With -InstallPrereqs, installs any that are missing via winget.
      2. Checks out the git submodules (tokenizers-cpp, xgrammar).
      3. Runs .github\script\ci\setup_windows_deps.ps1, which lays out
         C:\dev: Boost 1.88.0 (program_options built), the pinned XRT
         headers, and xrt_coreutil.lib generated from the committed .def.
         curl comes from third_party/curl and needs no setup.
      4. Enters a Visual Studio developer shell, configures with the
         windows-ninja preset and builds.
      5. Stages flm.exe plus its DLLs, model_list.json, model_info.json and
         xclbins in src\out so it can be run from there.

    Every step is skipped when its output already exists, so the script is
    safe to rerun; after the first run it is just a rebuild.

    src/CMakeLists.txt expects the dependencies under C:\dev by absolute path,
    so that location is fixed.

    Run from any directory:
        powershell -ExecutionPolicy Bypass -File src\WinSetup.ps1 -InstallPrereqs

.PARAMETER Variant
    XRT (default), HRX (the HRX runtime), or Source (the default build plus
    the Gemma 4 engine built from source instead of the prebuilt DLL).

.PARAMETER InstallPrereqs
    Install missing tools with winget. Visual Studio Build Tools and some
    other installers prompt for elevation.

.PARAMETER BuildDir
    CMake build directory under src. Defaults to build (XRT), build-hrx (HRX)
    or build-models-from-source (Source).

.PARAMETER Jobs
    Parallel build jobs. Defaults to all cores (2 for Source, whose engine
    compiles need a lot of memory).

.PARAMETER Clean
    Delete the build directory and src\out before building.

.PARAMETER NoOgaHipep
    XRT only: leave out the hip-ep (GPU) backend that CI builds in.

.PARAMETER SkipBuild
    Only set up the tools and dependencies.

.PARAMETER NoStage
    Do not copy the runtime files into src\out.
#>

[CmdletBinding()]
param(
    [ValidateSet('XRT', 'HRX', 'Source')]
    [string]$Variant = 'XRT',
    [switch]$InstallPrereqs,
    [string]$BuildDir,
    [int]$Jobs = 0,
    [switch]$Clean,
    [switch]$NoOgaHipep,
    [switch]$SkipBuild,
    [switch]$NoStage
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$DevRoot = 'C:\dev'               # hardcoded in src/CMakeLists.txt
$SrcDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $SrcDir '..')).Path
$NasmDir = Join-Path $env:ProgramFiles 'NASM'

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# Native commands run with ErrorActionPreference=Continue: when the script's
# output is redirected (e.g. *>&1 | Tee-Object), Windows PowerShell 5.1 turns
# every stderr line (git and cargo progress, compiler notes) into an error
# record, and under Stop the first one would abort the script. Success is
# judged by the exit code instead.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command, [switch]$NoThrow)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0 -and -not $NoThrow) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

# Runs a native command for its exit code only, discarding its output.
function Test-Native([scriptblock]$Command) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command *> $null } finally { $ErrorActionPreference = $old }
    return ($LASTEXITCODE -eq 0)
}

# Picks up PATH changes made by installers that ran in this session.
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Add-ToPath([string]$Dir) {
    if ($Dir -and (Test-Path $Dir) -and (($env:Path -split ';') -notcontains $Dir)) {
        $env:Path = "$Dir;$env:Path"
    }
}

$VsInstallerDir = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'

# The newest Visual Studio with the x64 C++ tools, or with -Any the newest
# Visual Studio at all (to add the C++ workload to it).
function Get-VsInstance([switch]$Any) {
    $vswhere = Join-Path $VsInstallerDir 'vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $vswhereArgs = @('-latest', '-products', '*', '-format', 'json')
    if (-not $Any) { $vswhereArgs += @('-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64') }
    $json = & $vswhere @vswhereArgs
    # Windows PowerShell 5.1 parses piped lines one by one; join them first.
    $found = ($json -join "`n") | ConvertFrom-Json
    if ($found) { return @($found)[0] }
    return $null
}

function Get-VsNinjaDir([object]$Vs) {
    if (-not $Vs) { return $null }
    $dir = Join-Path $Vs.installationPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
    if (Test-Path (Join-Path $dir 'ninja.exe')) { return $dir }
    return $null
}

function Test-Ninja([object]$Vs) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) { return $true }
    return [bool](Get-VsNinjaDir $Vs)
}

function Find-Cargo {
    $candidates = @()
    if ($env:CARGO_HOME) { $candidates += (Join-Path $env:CARGO_HOME 'bin') }
    $candidates += (Join-Path $env:USERPROFILE '.cargo\bin')
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir 'cargo.exe')) { return $dir }
    }
    $cmd = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cmd) { return (Split-Path $cmd.Source) }
    return $null
}

function Find-7z {
    $cmd = Get-Command 7z -ErrorAction SilentlyContinue
    if ($cmd) { return (Split-Path $cmd.Source) }
    foreach ($dir in @((Join-Path $env:ProgramFiles '7-Zip'), (Join-Path ${env:ProgramFiles(x86)} '7-Zip'))) {
        if (Test-Path (Join-Path $dir '7z.exe')) { return $dir }
    }
    return $null
}

function Test-Nasm {
    if (Get-Command nasm -ErrorAction SilentlyContinue) { return $true }
    return (Test-Path (Join-Path $NasmDir 'nasm.exe'))
}

function Get-CMakeVersion {
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { return $null }
    $line = (& cmake --version | Select-Object -First 1)
    if ($line -match '(\d+\.\d+(\.\d+)?)') { return [version]$Matches[1] }
    return $null
}

# VsDevCmd finds the Windows SDK with reg.exe. Where policy blocks reg.exe
# ("Registry editing has been disabled by your administrator"), it leaves
# the SDK out of INCLUDE/LIB/PATH and nothing links (LNK1104 kernel32.lib).
# Add the newest complete SDK the same way VsDevCmd would.
function Repair-WindowsSdkEnv {
    if ($env:WindowsSdkDir -and ($env:LIB -match 'Windows Kits')) { return }
    $root = $null
    $key = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -ErrorAction SilentlyContinue
    if ($key) { $root = $key.KitsRoot10 }
    if (-not $root) { $root = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\' }
    $root = $root.TrimEnd('\')
    $sdk = Get-ChildItem (Join-Path $root 'Lib') -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^\d+(\.\d+){3}$' -and
            (Test-Path (Join-Path $_.FullName 'um\x64\kernel32.lib')) -and
            (Test-Path (Join-Path $_.FullName 'ucrt\x64\ucrt.lib')) -and
            (Test-Path (Join-Path $root "Include\$($_.Name)\um\windows.h"))
        } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (-not $sdk) {
        throw "The developer shell has no Windows SDK and none was found under $root. Install the Windows 11 SDK with the Visual Studio Installer."
    }
    $v = $sdk.Name
    Write-Warning "The developer shell did not set up the Windows SDK (is reg.exe blocked by policy?). Using $v from $root."
    $inc = Join-Path $root "Include\$v"
    $env:WindowsSdkDir = "$root\"
    $env:WindowsSDKVersion = "$v\"
    $env:WindowsSDKLibVersion = "$v\"
    $env:UniversalCRTSdkDir = "$root\"
    $env:UCRTVersion = $v
    $env:WindowsSdkBinPath = "$root\bin\"
    $env:WindowsSdkVerBinPath = "$root\bin\$v\"
    $incDirs = @('ucrt', 'shared', 'um', 'winrt', 'cppwinrt') | ForEach-Object { Join-Path $inc $_ }
    $env:INCLUDE = (@($env:INCLUDE) + $incDirs | Where-Object { $_ }) -join ';'
    $env:LIB = (@($env:LIB, "$root\Lib\$v\ucrt\x64", "$root\Lib\$v\um\x64") | Where-Object { $_ }) -join ';'
    $env:LIBPATH = (@($env:LIBPATH, "$root\UnionMetadata\$v", "$root\References\$v") | Where-Object { $_ }) -join ';'
    Add-ToPath "$root\bin\$v\x64"
}

function Install-WithWinget([string]$Id, [string[]]$Extra = @()) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is not available; install $Id by hand (see src\WinSetup.md)"
    }
    Write-Host "Installing $Id with winget"
    $wingetArgs = @('install', '--id', $Id, '-e', '--source', 'winget',
        '--accept-package-agreements', '--accept-source-agreements') + $Extra
    Invoke-Native "winget install $Id" { & winget @wingetArgs } -NoThrow
    # winget exits non-zero when the package is already installed with no
    # upgrade available; the tool check after this call is what decides.
    Update-SessionPath
}

# --- 1. Tools --------------------------------------------------------------
# All installs happen in this section. Install-WithWinget resets PATH from
# the registry, so PATH is only extended after it.
Write-Step 'Checking tools'

if (-not [Environment]::Is64BitProcess) {
    # e.g. the Start Menu "Developer PowerShell for VS", which is the 32-bit
    # SysWOW64 powershell.exe: ProgramFiles points at Program Files (x86)
    # and its developer environment targets x86.
    throw ('This is a 32-bit PowerShell. Run the script from a regular 64-bit ' +
        'PowerShell window (powershell.exe or pwsh), not "Developer PowerShell for VS".')
}
if ($RepoRoot.Length -gt 60) {
    Write-Warning ("The repository path is $($RepoRoot.Length) characters long. The Rust and " +
        'codec build trees nest deeply; a short path such as C:\dev\FastFlowLM avoids ' +
        'MAX_PATH (260) failures.')
}

$missing = @()

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    if ($InstallPrereqs) { Install-WithWinget 'Git.Git' }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missing += 'Git (winget install Git.Git)' }
}

$vs = Get-VsInstance
if (-not $vs) {
    $anyVs = Get-VsInstance -Any
    if ($InstallPrereqs -and $anyVs) {
        # Visual Studio is there without the C++ tools; winget would see it
        # as installed and do nothing, so add the workload to it instead.
        $workload = 'Microsoft.VisualStudio.Workload.NativeDesktop'
        if ($anyVs.productId -like '*BuildTools*') { $workload = 'Microsoft.VisualStudio.Workload.VCTools' }
        Write-Host "Adding $workload to $($anyVs.displayName) (prompts for elevation)"
        $setupArgs = @('modify', '--installPath', "`"$($anyVs.installationPath)`"", '--add', $workload,
            '--includeRecommended', '--passive', '--norestart')
        Start-Process -FilePath (Join-Path $VsInstallerDir 'setup.exe') -ArgumentList $setupArgs -Verb RunAs -Wait
        $vs = Get-VsInstance
    } elseif ($InstallPrereqs) {
        # Visual Studio 2026 Build Tools with the C++ workload: the MSVC
        # compiler, the Windows SDK, and CMake/Ninja integration. Same major
        # version as the CI runners (windows-2025-vs2026).
        Install-WithWinget 'Microsoft.VisualStudio.BuildTools' @('--override',
            '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
        $vs = Get-VsInstance
    }
    if (-not $vs -and $anyVs) {
        $missing += ("the C++ tools in $($anyVs.displayName): in the Visual Studio Installer, Modify it " +
            'and add "Desktop development with C++"')
    } elseif (-not $vs) {
        $missing += ('Visual Studio 2022 or 2026 with "Desktop development with C++" ' +
            '(winget install Microsoft.VisualStudio.BuildTools --override "--wait --passive ' +
            '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended")')
    }
}

# Boost is the one dependency setup_windows_deps.ps1 builds: it unpacks the
# archive with 7-Zip and configures with CMake's default generator, the
# newest Visual Studio, which for VS 2026 needs CMake 4.2.
$needBoost = -not (Test-Path (Join-Path $DevRoot 'boost_1_88_0\stage\lib\libboost_program_options-vc143-mt-x64-1_88.lib'))
$minCMake = [version]'3.22'
if ($needBoost -and $vs -and ([version]$vs.installationVersion).Major -ge 18) { $minCMake = [version]'4.2' }
$cmakeVersion = Get-CMakeVersion
if (-not $cmakeVersion -or $cmakeVersion -lt $minCMake) {
    if ($InstallPrereqs) {
        Install-WithWinget 'Kitware.CMake'
        $cmakeVersion = Get-CMakeVersion
    }
    if (-not $cmakeVersion -or $cmakeVersion -lt $minCMake) {
        $missing += "CMake $minCMake or newer on PATH (winget install Kitware.CMake)"
    }
}

# The Ninja that Visual Studio bundles with its CMake component is enough.
if (-not (Test-Ninja $vs)) {
    if ($InstallPrereqs) { Install-WithWinget 'Ninja-build.Ninja' }
    if (-not (Test-Ninja $vs)) { $missing += 'Ninja (winget install Ninja-build.Ninja)' }
}

$cargoDir = Find-Cargo
if (-not $cargoDir) {
    if ($InstallPrereqs) {
        Install-WithWinget 'Rustlang.Rustup'
        $cargoDir = Find-Cargo
    }
    if (-not $cargoDir) { $missing += 'Rust via rustup (winget install Rustlang.Rustup)' }
}

$sevenZipDir = Find-7z
if (-not $sevenZipDir -and $needBoost) {
    if ($InstallPrereqs) {
        Install-WithWinget '7zip.7zip'
        $sevenZipDir = Find-7z
    }
    if (-not $sevenZipDir) { $missing += '7-Zip (winget install 7zip.7zip)' }
}

# Optional: without NASM, libjpeg-turbo builds without its SIMD code.
if (-not (Test-Nasm) -and $InstallPrereqs) { Install-WithWinget 'NASM.NASM' }

if ($missing.Count -gt 0) {
    $list = ($missing | ForEach-Object { "  - $_" }) -join "`n"
    throw "Missing tools:`n$list`nInstall them, or rerun with -InstallPrereqs."
}

Add-ToPath $cargoDir
Add-ToPath $sevenZipDir
Add-ToPath $NasmDir

# tokenizers-cpp runs cargo for the MSVC target; a rustup without a default
# toolchain has a cargo.exe that cannot build anything.
$rustup = Join-Path $cargoDir 'rustup.exe'
if (Test-Path $rustup) {
    if (-not (Test-Native { & $rustup show active-toolchain })) {
        Write-Host 'Installing the stable-x86_64-pc-windows-msvc Rust toolchain'
        Invoke-Native 'rustup default' { & $rustup default stable-x86_64-pc-windows-msvc }
    }
    Invoke-Native 'rustup target add' { & $rustup target add x86_64-pc-windows-msvc }
}

Write-Host "git:    $(git --version)"
Write-Host "cmake:  $cmakeVersion"
Write-Host "VS:     $($vs.displayName) $($vs.installationVersion)"
Write-Host "        $($vs.installationPath)"
Write-Host "cargo:  $(& (Join-Path $cargoDir 'cargo.exe') --version)"
if ($sevenZipDir) { Write-Host "7-Zip:  $sevenZipDir" }
if (Test-Nasm) {
    Write-Host 'NASM:   found'
} else {
    Write-Host 'NASM:   not found (optional; libjpeg-turbo then builds without SIMD)'
}

# --- 2. Submodules ---------------------------------------------------------
Write-Step 'Checking out submodules'
$ok = $false
for ($i = 1; $i -le 3 -and -not $ok; $i++) {
    Invoke-Native 'git submodule update' { git -C $RepoRoot submodule update --init --recursive --jobs 4 } -NoThrow
    if ($LASTEXITCODE -eq 0) {
        $ok = $true
    } else {
        Write-Host "Submodule update failed (attempt $i), retrying..."
        Start-Sleep -Seconds 10
    }
}
if (-not $ok) { throw 'git submodule update failed' }

# --- 3. Boost, XRT headers, xrt_coreutil.lib -------------------------------
# The CI script skips each part whose output exists. curl needs no setup:
# third_party/curl is committed and CMake makes its import library.
Write-Step "Setting up $DevRoot"
New-Item -ItemType Directory -Force $DevRoot | Out-Null
& (Join-Path $RepoRoot '.github\script\ci\setup_windows_deps.ps1') -Root $DevRoot

if ($SkipBuild) {
    Write-Step 'Dependencies are ready. Rerun without -SkipBuild to build.'
    return
}

# --- 4. Configure and build ------------------------------------------------
Write-Step "Entering the $($vs.displayName) developer shell"
# Everything links against x64 libraries, so the shell must target x64. The
# Start Menu developer shells default to x86; refuse those rather than stack
# a second environment on top.
if ($env:VSCMD_ARG_TGT_ARCH) {
    if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
        throw ("This developer shell targets $env:VSCMD_ARG_TGT_ARCH. Run the script from a " +
            'regular PowerShell window (it sets up an x64 environment itself) or an x64 one.')
    }
    Write-Host 'Already in an x64 developer shell'
} else {
    Import-Module (Join-Path $vs.installationPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs.installationPath -SkipAutomaticLocation `
        -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}
Repair-WindowsSdkEnv
# The developer shell rewrites PATH; keep cargo and NASM reachable, and fall
# back to the Ninja that Visual Studio bundles.
Add-ToPath $cargoDir
Add-ToPath $NasmDir
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Add-ToPath (Get-VsNinjaDir $vs)
}

if (-not $BuildDir) {
    switch ($Variant) {
        'XRT'    { $BuildDir = 'build' }
        'HRX'    { $BuildDir = 'build-hrx' }
        'Source' { $BuildDir = 'build-models-from-source' }
    }
}
$buildPath = Join-Path $SrcDir $BuildDir
$outPath = Join-Path $SrcDir 'out'

if ($Clean) {
    Write-Step "Removing $buildPath and $outPath"
    foreach ($p in @($buildPath, $outPath)) {
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
    }
}

# A tree configured by another generator (e.g. the Visual Studio one that
# src/Makefile uses) cannot be reconfigured with Ninja.
$cache = Join-Path $buildPath 'CMakeCache.txt'
if (Test-Path $cache) {
    $gen = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$' | Select-Object -First 1
    if ($gen -and $gen.Matches[0].Groups[1].Value -ne 'Ninja') {
        throw ("$buildPath was configured with '$($gen.Matches[0].Groups[1].Value)'. " +
            'Rerun with -Clean, or pick another directory with -BuildDir.')
    }
}

# The variant options are CMake cache options, so each is passed explicitly,
# OFF included; otherwise a value from an earlier configure of the same
# directory would stick. Values match .github/actions/win-build.
$hipep = 'ON'
if ($NoOgaHipep) { $hipep = 'OFF' }
$configureArgs = @('--preset', 'windows-ninja', '-B', $buildPath)
switch ($Variant) {
    'XRT' {
        $configureArgs += @('-DFLM_USE_HRX=OFF', "-DFLM_ENABLE_OGA_HIPEP=$hipep", '-DFLM_BUILD_GEMMA4E=OFF')
    }
    'HRX' {
        Write-Step 'Fetching the pinned HRX release'
        $lines = & (Join-Path $RepoRoot 'hrx-integration\fetch-hrx-release.ps1') 6>&1 |
            ForEach-Object { $_.ToString() }
        $lines | ForEach-Object { Write-Host $_ }
        $hrxPrefix = ($lines | Where-Object { $_ -like 'HRX_CMAKE_PREFIX=*' } |
            Select-Object -First 1) -replace '^HRX_CMAKE_PREFIX=', ''
        if (-not $hrxPrefix) { throw 'fetch-hrx-release.ps1 did not report HRX_CMAKE_PREFIX' }
        $configureArgs += @('-DFLM_USE_HRX=ON', '-DFLM_ENABLE_OGA_HIPEP=OFF', '-DFLM_BUILD_GEMMA4E=OFF',
            "-DCMAKE_PREFIX_PATH=$hrxPrefix")
    }
    'Source' {
        $configureArgs += @('-DFLM_USE_HRX=OFF', '-DFLM_ENABLE_OGA_HIPEP=OFF', '-DFLM_BUILD_GEMMA4E=ON')
    }
}

Write-Step "Configuring ($Variant) in $buildPath"
Push-Location $SrcDir
try {
    Invoke-Native 'cmake configure' { cmake @configureArgs }

    if ($Jobs -le 0 -and $Variant -eq 'Source') { $Jobs = 2 }
    $buildArgs = @('--build', $buildPath, '--parallel')
    if ($Jobs -gt 0) { $buildArgs += "$Jobs" }
    Write-Step 'Building (the first build also downloads codec sources and Rust crates)'
    Invoke-Native 'cmake build' { cmake @buildArgs }
} finally {
    Pop-Location
}

# --- 5. Stage the runtime files --------------------------------------------
# Same files and copy order as the package step in
# .github/workflows/win-build.yml: later copies win where names overlap.
if (-not $NoStage) {
    Write-Step "Staging runtime files in $outPath"
    New-Item -ItemType Directory -Force $outPath | Out-Null
    Copy-Item (Join-Path $buildPath 'flm.exe') $outPath -Force
    $lib = Join-Path $SrcDir 'lib'
    if ($Variant -eq 'HRX') {
        Copy-Item (Join-Path $lib '*.dll') $outPath -Force
        Copy-Item (Join-Path $lib 'hrx\*.dll') $outPath -Force
        $hrxDll = Get-ChildItem -Path (Join-Path $RepoRoot 'hrx-integration\.hrx-release') -Recurse -Filter 'hrx.dll' |
            Select-Object -First 1
        if (-not $hrxDll) { throw 'hrx.dll not found under hrx-integration\.hrx-release' }
        Copy-Item $hrxDll.FullName $outPath -Force
    } else {
        Copy-Item (Join-Path $lib 'xrt\*.dll') $outPath -Force
        Copy-Item (Join-Path $lib '*.dll') $outPath -Force
    }
    if ($Variant -eq 'Source') {
        # The source-built engines replace the prebuilt ones from lib\xrt.
        Copy-Item (Join-Path $buildPath 'engines\*.dll') $outPath -Force
    }
    foreach ($f in @('model_list.json', 'model_info.json')) {
        Copy-Item (Join-Path $SrcDir $f) $outPath -Force
    }
    # Copy the contents: copying the folder onto an existing out\xclbins
    # would nest it as out\xclbins\xclbins.
    $xclbinOut = Join-Path $outPath 'xclbins'
    New-Item -ItemType Directory -Force $xclbinOut | Out-Null
    Copy-Item (Join-Path $SrcDir 'xclbins\*') $xclbinOut -Recurse -Force
}

Write-Step 'Done'
Write-Host "flm.exe: $(Join-Path $buildPath 'flm.exe')"
if (-not $NoStage) {
    Write-Host 'To try it (needs an AMD Ryzen AI NPU and its driver):'
    Write-Host "  cd $outPath"
    Write-Host '  .\flm.exe validate'
    Write-Host '  .\flm.exe run llama3.2:1b'
}
