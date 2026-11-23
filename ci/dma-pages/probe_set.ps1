# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# probe_set.ps1 -Dir <dist> -Backing bo,user4k,user2m -Out <file>: the #1080 cases (streams, 4 KB-apart
# rows over 64 and 1024 pages, rows 64 KB + R apart) for each backing, as whichever account runs it.
param([string]$Dir, [string]$Backing = 'bo,user4k,user2m', [string]$Out, [int]$Iters = 20)
$ErrorActionPreference = 'Continue'
$env:PROBE_LP_LOG = '1'
$names = @('stream_r256_p1024', 'scatter_r128_p1024', 'scatter_r256_p1024', 'gather_r128_p1024', 'gather_r256_p1024',
  'scatter_r128_p64', 'gather_r256_p64', 'xscatter_r128_n1024_k64', 'xgather_r256_n1024_k64')
$lines = @("account $(whoami)")
$lines += whoami /priv | Select-String SeLockMemoryPrivilege | ForEach-Object { "$_".Trim() }
$lines += 'mode,R,P,backing,median_us,min_us,GBps,Mrows_s,check,pages'
foreach ($line in Get-Content "$Dir\precompiled\cases.tsv") {
  $f = $line -split "`t"
  if ($f.Count -lt 8) { continue }
  $name, $mode, $R, $P, $ib, $ob, $per, $nch, $pitch, $cb = $f
  if ($names -notcontains $name) { continue }
  if (-not $pitch) { $pitch = 0 }
  foreach ($bk in ($Backing -split ',')) {
    $o = & "$Dir\probe.exe" "$Dir\precompiled\$name\final.xclbin" "$Dir\precompiled\$name\insts.bin" $mode $R $P $ib $ob $per $nch $bk $Iters $pitch 2>&1 |
      ForEach-Object { "$_" }
    if ($LASTEXITCODE) { $lines += "$name,$bk,FAILED,$($o -join ' ')" } else { $lines += $o }
  }
}
$lines += 'done'
$lines | Set-Content -Encoding utf8 $Out
