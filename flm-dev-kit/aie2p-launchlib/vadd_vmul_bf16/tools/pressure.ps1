#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# pressure.ps1 -- tools/pressure.sh on Windows: the same parts (P1-P7), the same
# counts, on stress.exe from the CMake build (WINDOWS.md).
#
#   powershell -File tools/pressure.ps1 -Exe build/c8x4/Release/stress.exe -Package precompiled/n256-c8x4-pair
#
# -Transport (default ctrlpkt): as TRANSPORT there: P1-P4 and P6; P5 and P7 run txn.
#
# One log per part in -Out (default build/pressure); a summary line each. Exit 1
# when a part that must pass fails (P5b's full walk is expected to fail).

param(
    [Parameter(Mandatory)] [string] $Exe,
    [Parameter(Mandatory)] [string] $Package,
    [string] $Out = 'build/pressure',
    [int[]] $Seeds = @(11, 12, 13),
    [int] $Launches = 200000,
    [int] $Churn = 50,
    # leave out P5b: a full walk (warm and rerun as asked) next to another process
    [switch] $SkipFullWalk,
    [ValidateSet('txn', 'ctrlpkt')] [string] $Transport = 'ctrlpkt'
)

$Exe = (Resolve-Path $Exe -ErrorAction Stop).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$script:failed = 0

function Summary($name, $log, $status, $seconds, $mustPass = $true) {
    $stress = Select-String -Path $log -Pattern '^stress ' | Select-Object -Last 1
    $bad = Select-String -Path $log -Pattern 'unexpected: [0-9]* wrong, [0-9]* written past the end' |
        Select-Object -Last 1
    $u = if ($bad) { $bad.Matches[0].Value } else { '' }
    Write-Host "${name}: exit $status, $seconds s, $($stress.Line), $u"
    # No result line: the run did not get going (counts as a failure).
    if ($mustPass -and ($status -ne 0 -or -not $stress)) { $script:failed = 1 }
}

function Timed($name, $log, [string[]] $more, $tr = $Transport) {
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $Exe --package $Package --transport $tr @more *> $log
    $status = $LASTEXITCODE
    Summary $name $log $status ([int]$t.Elapsed.TotalSeconds)
}

# Two processes at once, txn; returns their exit codes after both end.
function Pair($log1, [string[]] $args1, $log2, [string[]] $args2) {
    $a = Start-Process -FilePath $Exe -ArgumentList (@('--package', $Package, '--transport', 'txn') + $args1) `
        -NoNewWindow -PassThru -RedirectStandardOutput $log1 -RedirectStandardError "$log1.err"
    $null = $a.Handle # keeps the exit code readable
    & $Exe --package $Package --transport txn @args2 *> $log2
    $s2 = $LASTEXITCODE
    $a.WaitForExit()
    return @($a.ExitCode, $s2)
}

foreach ($seed in $Seeds) {
    Timed "P1 soak seed $seed" "$Out/p1_seed$seed.log" @('--mode', 'random', '--seed', $seed, '--launches', $Launches)
}
Timed 'P2 long runlists' "$Out/p2.log" @('--mode', 'random', '--seed', 21, '--launches', 100000, '--max-runlist', 1000)
Timed 'P3 big lengths' "$Out/p3.log" @('--mode', 'random', '--seed', 31, '--launches', 4000, '--max-runlist', 16,
    '--lengths', '2,64,256,1024')
Timed 'P4 latency' "$Out/p4.log" @('--mode', 'latency', '--count', 2000)

# P5, txn (as P7): a ctrlpkt kernel next to another context is not tested.
$t = [Diagnostics.Stopwatch]::StartNew()
$s = Pair "$Out/p5a_1.log" @('--mode', 'random', '--robust', '--seed', 51, '--launches', 50000) `
    "$Out/p5a_2.log" @('--mode', 'random', '--robust', '--seed', 52, '--launches', 50000)
Summary 'P5a robust + robust, process 1' "$Out/p5a_1.log" $s[0] ([int]$t.Elapsed.TotalSeconds)
Summary 'P5a robust + robust, process 2' "$Out/p5a_2.log" $s[1] ([int]$t.Elapsed.TotalSeconds)
if ($SkipFullWalk) {
    Write-Host 'P5b robust next to a full walk: skipped (-SkipFullWalk)'
} else {
    $t = [Diagnostics.Stopwatch]::StartNew()
    $s = Pair "$Out/p5b_robust.log" @('--mode', 'random', '--robust', '--seed', 53, '--launches', 50000) `
        "$Out/p5b_full.log" @('--mode', 'random', '--seed', 54, '--launches', 50000)
    Summary 'P5b robust next to a full walk: robust' "$Out/p5b_robust.log" $s[0] ([int]$t.Elapsed.TotalSeconds)
    Summary 'P5b robust next to a full walk: full (state-dependent kinds expected to fail)' "$Out/p5b_full.log" $s[1] `
        ([int]$t.Elapsed.TotalSeconds) $false
}

$t = [Diagnostics.Stopwatch]::StartNew()
$bad = 0
for ($i = 1; $i -le $Churn; $i++) {
    & $Exe --package $Package --transport $Transport --mode random --seed (600 + $i) --launches 500 *> "$Out/p6_last.log"
    if ($LASTEXITCODE) { $bad++ }
}
Write-Host "P6 process churn: $Churn processes, $bad failed, $([int]$t.Elapsed.TotalSeconds) s"
if ($bad) { $script:failed = 1 }

Timed 'P7 context churn' "$Out/p7.log" @('--mode', 'churn', '--count', $Churn) 'txn'
exit $script:failed
