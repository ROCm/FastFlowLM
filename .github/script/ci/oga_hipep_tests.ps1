<#
.SYNOPSIS
    Builds and runs the oga-hipep unit tests (src/test/oga_hipep and its
    frontend project) against the fake OGA C API. No GPU, no NPU.

.DESCRIPTION
    Run from the repository root in an MSVC x64 developer environment, after
    setup_windows_deps.ps1 has laid out Boost and the XRT headers under
    -DepsRoot. curl comes from third_party/curl: an import library is made
    from libcurl.def, and the shipped src/lib/libcurl.dll (with its zlib1.dll)
    is copied next to the test binaries. The cpu-tests workflow runs this;
    so can a developer, with -DepsRoot pointing at their own layout.

.PARAMETER DepsRoot
    Where setup_windows_deps.ps1 put the dependencies (C:\dev on CI).

.PARAMETER BuildRoot
    Directory for the build trees (the current directory by default).
#>

[CmdletBinding()]
param(
    [string]$DepsRoot = "C:\dev",
    [string]$BuildRoot = "."
)

$ErrorActionPreference = "Stop"

# Native commands run with ErrorActionPreference=Continue, as in
# setup_windows_deps.ps1: Windows PowerShell 5.1 can turn their stderr lines
# into error records, and under Stop the first warning would abort the run.
# Success is judged by the exit code instead.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
New-Item -ItemType Directory -Force $BuildRoot | Out-Null
$BuildRoot = (Resolve-Path $BuildRoot).Path
$DepsRoot = (Resolve-Path $DepsRoot).Path

$CurlDir = Join-Path $BuildRoot "build-oga-curl"
New-Item -ItemType Directory -Force $CurlDir | Out-Null
$CurlLib = Join-Path $CurlDir "libcurl.lib"
Invoke-Native "lib /def libcurl" {
    lib /nologo /machine:x64 "/def:$Repo\third_party\curl\libcurl.def" "/out:$CurlLib"
}

$Deps = @(
    "-DBOOST_INCLUDE_DIR=$DepsRoot\boost_1_88_0",
    "-DXRT_INCLUDE_DIR=$DepsRoot\XRT\src\runtime_src\core\include",
    "-DXRT_LIB_DIR=$DepsRoot\xrtNPUfromDLL"
)
$Projects = @(
    @{ Source = "src\test\oga_hipep"; Build = "build-oga-hipep";
       # CURL_NO_CURL_CMAKE: use these two, not a CURLConfig.cmake found on the runner's PATH
       Extra = @("-DCURL_INCLUDE_DIR=$Repo\third_party\curl\include", "-DCURL_LIBRARY=$CurlLib",
                 "-DCURL_NO_CURL_CMAKE=ON") },
    @{ Source = "src\test\oga_hipep\frontend"; Build = "build-oga-hipep-frontend"; Extra = @() }
)

foreach ($p in $Projects) {
    $build = Join-Path $BuildRoot $p.Build
    Invoke-Native "configure $($p.Source)" {
        cmake -S (Join-Path $Repo $p.Source) -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release @Deps @($p.Extra)
    }
    Invoke-Native "build $($p.Source)" { cmake --build $build --parallel }
    foreach ($dll in "libcurl.dll", "zlib1.dll") {
        Copy-Item (Join-Path $Repo "src\lib\$dll") $build -Force
    }
    Invoke-Native "ctest $($p.Source)" { ctest --test-dir $build --output-on-failure }
}
