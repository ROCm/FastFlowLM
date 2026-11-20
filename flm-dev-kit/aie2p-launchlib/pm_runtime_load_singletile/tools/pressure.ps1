#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# pressure.ps1 -- tools/pressure.sh on Windows: the same parts and counts, on
# test.exe and stress.exe from the CMake build (build/c8/Release).
#
#   powershell -File tools/pressure.ps1 -Bin build/c8/Release -Package precompiled/c8
#
# Order: the parts that must pass first (P0-P4, P5a, P6, P7, P11), then the
# ones that hang on purpose or need the sharing guard (P5b, P8, P9, P10; on
# Windows the Session sees no other context, so P8 and P10 are informative),
# then P0 again (xclbin flow, 100 per row): the NPU is fine afterwards.
# Exit 1 when a part that must pass fails.
#
# WARNING: -Group hang (and all) took two Windows machines offline (the
# runner lost its connection during it and did not come back). Run only
# -Group must on Windows until that is understood.

param(
    [Parameter(Mandatory)] [string] $Bin,
    [Parameter(Mandatory)] [string] $Package,
    [string] $Out = 'build/pressure',
    [int[]] $Seeds = @(11, 12, 13),
    [int] $Launches = 200000,
    [int] $Churn = 50,
    # must: P0-P4, P5a, P6, P7, P11; hang: P5b, P8, P9, P10, then P0 again; all: both
    [ValidateSet('all', 'must', 'hang')] [string] $Group = 'all',
    # stop at the first part that must pass and fails (each failure can be an NPU reset on Windows)
    [switch] $StopOnFail
)

$Exe = (Resolve-Path "$Bin/stress.exe" -ErrorAction Stop).Path
$Test = (Resolve-Path "$Bin/test.exe" -ErrorAction Stop).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$script:failed = 0

function Fail {
    $script:failed = 1
    if ($StopOnFail) { Write-Host 'stopped: a part that must pass failed (-StopOnFail)'; exit 1 }
}

function Summary($name, $log, $status, $seconds, $mustPass = $true) {
    $stress = Select-String -Path $log -Pattern '^stress ' | Select-Object -Last 1
    $bad = Select-String -Path $log -Pattern 'unexpected: [0-9]* wrong, [0-9]* written past the end' |
        Select-Object -Last 1
    $u = if ($bad) { $bad.Matches[0].Value } else { '' }
    Write-Host "${name}: exit $status, $seconds s, $($stress.Line), $u"
    # No result line: the run did not get going (counts as a failure).
    if ($mustPass -and ($status -ne 0 -or -not $stress)) { Fail }
}

function Timed($name, $log, [string[]] $more, $mustPass = $true) {
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $Exe --package $Package @more *> $log
    $status = $LASTEXITCODE
    Summary $name $log $status ([int]$t.Elapsed.TotalSeconds) $mustPass
}

function HostTest($name, $log, $flow, $iters) {
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $Test --package $Package --flow $flow --iters $iters *> $log
    $status = $LASTEXITCODE
    $pass = (Select-String -Path $log -Pattern 'PASS$').Count
    $fail = (Select-String -Path $log -Pattern 'FAIL$').Count
    Write-Host "${name}: exit $status, $([int]$t.Elapsed.TotalSeconds) s, $pass rows PASS, $fail FAIL"
    if ($status -ne 0) { Fail }
}

# Two processes at once, txn (a ctrlpkt kernel next to another context is not
# tested); returns their exit codes after both end.
function Pair($log1, [string[]] $args1, $log2, [string[]] $args2) {
    $a = Start-Process -FilePath $Exe -ArgumentList (@('--package', $Package, '--transport', 'txn') + $args1) `
        -NoNewWindow -PassThru -RedirectStandardOutput $log1 -RedirectStandardError "$log1.err"
    $null = $a.Handle # keeps the exit code readable
    & $Exe --package $Package --transport txn @args2 *> $log2
    $s2 = $LASTEXITCODE
    $a.WaitForExit()
    return @($a.ExitCode, $s2)
}

if ($Group -ne 'hang') {
HostTest 'P0 host test, xclbin flow, 1000 launches per row' "$Out/p0_xclbin.log" xclbin 1000
HostTest 'P0 host test, elf flow, 1000 launches per row' "$Out/p0_elf.log" elf 1000

foreach ($seed in $Seeds) {
    Timed "P1 soak seed $seed" "$Out/p1_seed$seed.log" @('--mode', 'random', '--seed', $seed, '--launches', $Launches,
        '--max-gap', 0)
}
Timed 'P2 long runlists' "$Out/p2.log" @('--mode', 'random', '--seed', 21, '--launches', 100000, '--max-runlist', 1000,
    '--max-gap', 0)
Timed 'P3 lengths + walk' "$Out/p3.log" @('--mode', 'all', '--seed', 31, '--launches', 20000, '--max-gap', 0)
Timed 'P4 latency' "$Out/p4.log" @('--mode', 'latency', '--count', 2000)
Timed 'P4 latency, no guard' "$Out/p4_noguard.log" @('--mode', 'latency', '--count', 2000, '--no-guard')

$t = [Diagnostics.Stopwatch]::StartNew()
$s = Pair "$Out/p5a_1.log" @('--mode', 'random', '--robust', '--seed', 51, '--launches', 20000) `
    "$Out/p5a_2.log" @('--mode', 'random', '--robust', '--seed', 52, '--launches', 20000)
Summary 'P5a robust + robust, process 1' "$Out/p5a_1.log" $s[0] ([int]$t.Elapsed.TotalSeconds)
Summary 'P5a robust + robust, process 2' "$Out/p5a_2.log" $s[1] ([int]$t.Elapsed.TotalSeconds)

$t = [Diagnostics.Stopwatch]::StartNew()
$bad = 0
for ($i = 1; $i -le $Churn; $i++) {
    & $Exe --package $Package --mode random --seed (600 + $i) --launches 500 *> "$Out/p6_last.log"
    if ($LASTEXITCODE) { $bad++ }
}
Write-Host "P6 process churn: $Churn processes, $bad failed, $([int]$t.Elapsed.TotalSeconds) s"
if ($bad) { Fail }

Timed 'P7 context churn' "$Out/p7.log" @('--mode', 'churn', '--count', $Churn)
Timed 'P11 ELF walk' "$Out/p11.log" @('--mode', 'random', '--flow', 'elf', '--seed', 41, '--launches', 50000,
    '--max-runlist', 1000, '--max-gap', 0)

}

if ($Group -ne 'must') {
Write-Host '--- parts that hang on purpose or need the sharing guard'
$t = [Diagnostics.Stopwatch]::StartNew()
$s = Pair "$Out/p5b_robust.log" @('--mode', 'random', '--robust', '--seed', 53, '--launches', 20000) `
    "$Out/p5b_full.log" @('--mode', 'random', '--seed', 54, '--launches', 20000)
Summary 'P5b robust next to a full walk: robust (fails on Linux too)' "$Out/p5b_robust.log" $s[0] `
    ([int]$t.Elapsed.TotalSeconds) $false
Summary 'P5b robust next to a full walk: full (expected to fail)' "$Out/p5b_full.log" $s[1] `
    ([int]$t.Elapsed.TotalSeconds) $false
Timed 'P8 contexts (needs the guard)' "$Out/p8.log" @('--mode', 'ctx') $false
Timed 'P9 recovery' "$Out/p9.log" @('--mode', 'recover')
Timed 'P10 guard, a looping context (needs the guard)' "$Out/p10_loop.log" @('--mode', 'guard', '--child', 'loop',
    '--launches', 20000) $false
Timed 'P10 guard, contexts open and close (needs the guard)' "$Out/p10_churn.log" @('--mode', 'guard', '--child',
    'churn', '--launches', 20000) $false

HostTest 'P0 again, xclbin flow, 100 launches per row' "$Out/p0_after.log" xclbin 100
}
exit $script:failed
