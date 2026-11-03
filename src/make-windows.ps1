<#
.SYNOPSIS
  The Windows side of the Makefile: build, stage and run flm.exe.

.DESCRIPTION
  The Makefile calls this with -File for every Windows target, whether make runs
  natively on Windows or inside WSL on a /mnt/<drive> checkout. Keeping the
  PowerShell here rather than inline in the recipes means no shell between make
  and PowerShell (cmd, Git's sh or WSL's sh) gets a chance to expand its `$`.

  Targets: all, hrx, run, run-hrx, serve, stage, clean, install.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('all', 'hrx', 'run', 'run-hrx', 'serve', 'stage', 'clean', 'install')]
    [string]$Target,
    [string]$RunArgs = 'llama3.2:1b -a 0',
    [string]$ServeArgs = 'qwen2.5vl-it -c 32768',
    # Compilers at once across the whole build; default every logical core.
    [int]$Jobs = [Environment]::ProcessorCount,
    # Configure preset; empty picks windows-vs18 on Visual Studio 18, else windows-default.
    [string]$Preset,
    # Script dot-sourced to set up MSVC + Windows SDK when cl.exe is not already on PATH.
    [string]$DevEnv,
    [string]$Prefix,
    # Build tree; empty means the configure preset's binaryDir (build_aie_next
    # for windows-aie-next, else build).
    [string]$BuildDir,
    [string]$OutDir = 'out'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Native {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "$($args[0]) exited with code $LASTEXITCODE" }
}

# Machines where vcvars cannot run (e.g. reg.exe blocked by policy) keep a
# script that sets the compiler environment directly; C:\dev is where the
# CMakeLists bare-runner route already expects Boost, vcpkg and XRT.
function Import-DevEnv {
    if (-not $DevEnv -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue) -and
        (Test-Path 'C:\dev\flm-devenv.ps1')) {
        $script:DevEnv = 'C:\dev\flm-devenv.ps1'
    }
    if ($DevEnv) {
        Write-Host "Loading compiler environment from $DevEnv"
        . $DevEnv
    }
}

function Get-ConfigurePreset {
    if ($Preset) { return $Preset }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $version = & $vswhere -latest -products * -property installationVersion
        if ($version -like '18.*') { return 'windows-vs18' }
    }
    return 'windows-default'
}

# The binaryDir a configure preset names (following inherits), relative to here.
function Get-PresetBinaryDir([string]$Name) {
    $presets = (Get-Content 'CMakePresets.json' -Raw | ConvertFrom-Json).configurePresets
    $seen = @{}
    while ($Name -and -not $seen.ContainsKey($Name)) {
        $seen[$Name] = $true
        $p = $presets | Where-Object { $_.name -eq $Name } | Select-Object -First 1
        if (-not $p) { break }
        if ($p.binaryDir) { return ($p.binaryDir -replace '^\$\{sourceDir\}[/\\]?', '') }
        $Name = @($p.inherits)[0]
    }
    return 'build'
}

# A preset with its own binaryDir (windows-aie-next: build_aie_next) configures there,
# so build and stage from that same tree rather than a stale or empty build\.
if (-not $BuildDir) { $BuildDir = Get-PresetBinaryDir (Get-ConfigurePreset) }

function Invoke-Build([string[]]$ExtraConfigureArgs) {
    Import-DevEnv
    New-Item -ItemType Directory -Force $BuildDir, $OutDir | Out-Null
    $env:CARGO_BUILD_JOBS = "$Jobs"
    $configure = Get-ConfigurePreset
    Write-Host "Configuring with preset $configure into $BuildDir"
    Invoke-Native cmake --preset $configure -B $BuildDir @ExtraConfigureArgs
    # '--' quoted: Windows PowerShell drops a bare -- as its own end-of-parameters marker.
    # MSBuild alone would run $Jobs projects of $Jobs cl each; the multi-tool task
    # with EnforceProcessCountAcrossBuilds holds the whole build to $Jobs compilers.
    Invoke-Native cmake --build $BuildDir --config Release --parallel $Jobs '--' `
        '/p:UseMultiToolTask=true' '/p:EnforceProcessCountAcrossBuilds=true' "/p:MultiProcMaxCount=$Jobs"
}

# Puts everything flm.exe loads at run time next to it in the out directory.
function Copy-RuntimeFiles([string]$Backend) {
    if (-not (Test-Path "$OutDir\flm.exe")) { throw "$OutDir\flm.exe not found; run make first" }
    Copy-Item "lib\$Backend\*.dll" $OutDir -Force
    Copy-Item 'lib\*.dll' $OutDir -Force
    # A core build (FLM_ENABLE_CORE) copies corelib and its model libraries beside
    # flm.exe in the build tree; corelib loads those only from its own directory.
    Copy-Item "$BuildDir\ryzenai_corelib*.dll" $OutDir -Force -ErrorAction SilentlyContinue
    # The bare-runner build links curl from the vcpkg tree under C:\dev (or
    # VCPKG_ROOT); ship the DLL from that same tree rather than lib\'s copy.
    $vcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { 'C:\dev\vcpkg' }
    $vcpkgBin = Join-Path $vcpkgRoot 'installed\x64-windows\bin'
    # zlib is z.dll in current vcpkg ports, zlib1.dll in older ones.
    foreach ($dll in 'libcurl.dll', 'z.dll', 'zlib1.dll') {
        if (Test-Path "$vcpkgBin\$dll") { Copy-Item "$vcpkgBin\$dll" $OutDir -Force }
    }
    Copy-Item 'model_list.json', 'model_info.json' $OutDir -Force
    Copy-Item 'xclbins' $OutDir -Recurse -Force
}

function Invoke-Flm([string]$Arguments) {
    Push-Location $OutDir
    try {
        Invoke-Native .\flm.exe @(-split $Arguments)
    } finally {
        Pop-Location
    }
}

switch ($Target) {
    'all' { Invoke-Build @() }
    'hrx' {
        $lines = & '..\hrx-integration\fetch-hrx-release.ps1' 6>&1 | ForEach-Object { $_.ToString() }
        $lines | ForEach-Object { Write-Host $_ }
        $hrxPrefix = ($lines | Where-Object { $_ -like 'HRX_CMAKE_PREFIX=*' } | Select-Object -First 1) -replace 'HRX_CMAKE_PREFIX=', ''
        if (-not $hrxPrefix) { throw 'failed to determine HRX_CMAKE_PREFIX from fetch-hrx-release.ps1 output' }
        Invoke-Build @('-DFLM_USE_HRX=ON', "-DCMAKE_PREFIX_PATH=$hrxPrefix")
    }
    'stage' { Copy-RuntimeFiles 'xrt' }
    'run' {
        Copy-RuntimeFiles 'xrt'
        Invoke-Flm "run $RunArgs"
    }
    'run-hrx' {
        Copy-RuntimeFiles 'hrx'
        $hrxDll = Get-ChildItem -Path '..\hrx-integration\.hrx-release' -Recurse -Filter 'hrx.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hrxDll) { Copy-Item $hrxDll.FullName $OutDir -Force }
        else { Write-Warning 'hrx.dll not found under hrx-integration\.hrx-release; run make hrx first' }
        Invoke-Flm "run $RunArgs"
    }
    'serve' {
        Copy-RuntimeFiles 'xrt'
        Invoke-Flm "serve $ServeArgs"
    }
    'clean' {
        foreach ($dir in $BuildDir, $OutDir) {
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        }
    }
    'install' {
        if (-not $Prefix) { throw 'install needs PREFIX, e.g. make install PREFIX=C:\flm' }
        Invoke-Native cmake --install $BuildDir --config Release --prefix $Prefix
    }
}
