# run.ps1 -Exe probe.exe -Build precompiled -Backing bo,user4k,user2m
param([string]$Exe, [string]$Build, [string[]]$Backing = @('bo'), [int]$Iters = 20)
Write-Output "mode,R,P,backing,median_us,min_us,GBps,Mrows_s,check,pages"
foreach ($line in Get-Content "$Build\cases.tsv") {
  $f = $line -split "`t"
  if ($f.Count -lt 8) { continue }
  $name, $mode, $R, $P, $ib, $ob, $per, $nch, $pitch, $cb = $f
  if (-not $pitch) { $pitch = 0 }
  foreach ($bk in $Backing) {
    $out = & $Exe "$Build\$name\final.xclbin" "$Build\$name\insts.bin" $mode $R $P $ib $ob $per $nch $bk $Iters $pitch 2>&1
    if ($LASTEXITCODE) { Write-Output "$name,$bk,FAILED,$($out -join ' ')" } else { Write-Output $out }
  }
}
