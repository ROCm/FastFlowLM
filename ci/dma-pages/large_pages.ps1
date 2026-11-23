# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# large_pages.ps1 -Dir <dist> -OutDir <dir>: can the Windows NPU driver use 2 MB (large-page) user buffers,
# and do the #1080 scattered-row DMAs get faster with them? probe_set.ps1 runs three times:
#   1. as the runner account (no SeLockMemoryPrivilege: large-page allocations fall back to 4 KB);
#   2. as SYSTEM through a temporary scheduled task (SYSTEM normally holds SeLockMemoryPrivilege; no
#      policy change);
#   3. only if 2 got no large pages: SeLockMemoryPrivilege is granted to the runner account, the set runs
#      under a fresh S4U logon of that account (a new token carries the right), and the right is revoked
#      in `finally`.
# Scheduled tasks are deleted afterwards. No deliberate hangs; each set is ~1-2 min, capped at 10 min.
param([string]$Dir, [string]$OutDir)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$OutDir = (Resolve-Path $OutDir).Path
$set = Join-Path $Dir 'probe_set.ps1'

Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Security.Principal;
public static class LsaRights {
  [StructLayout(LayoutKind.Sequential)] struct LSA_UNICODE_STRING { public ushort Length, MaximumLength; public IntPtr Buffer; }
  [StructLayout(LayoutKind.Sequential)] struct LSA_OBJECT_ATTRIBUTES {
    public int Length; public IntPtr RootDirectory, ObjectName; public uint Attributes; public IntPtr SecurityDescriptor, SecurityQualityOfService; }
  [DllImport("advapi32.dll")] static extern uint LsaOpenPolicy(IntPtr system, ref LSA_OBJECT_ATTRIBUTES oa, uint access, out IntPtr h);
  [DllImport("advapi32.dll")] static extern uint LsaAddAccountRights(IntPtr h, byte[] sid, LSA_UNICODE_STRING[] rights, uint n);
  [DllImport("advapi32.dll")] static extern uint LsaRemoveAccountRights(IntPtr h, byte[] sid, bool all, LSA_UNICODE_STRING[] rights, uint n);
  [DllImport("advapi32.dll")] static extern uint LsaClose(IntPtr h);
  [DllImport("advapi32.dll")] static extern int LsaNtStatusToWinError(uint status);
  // add (or remove) one account right; returns a Win32 error code, 0 = ok
  public static int Set(string sid, string right, bool add) {
    var s = new SecurityIdentifier(sid);
    var b = new byte[s.BinaryLength];
    s.GetBinaryForm(b, 0);
    var oa = new LSA_OBJECT_ATTRIBUTES();
    oa.Length = Marshal.SizeOf(oa);
    IntPtr h;
    uint st = LsaOpenPolicy(IntPtr.Zero, ref oa, 0x00000810 /* POLICY_CREATE_ACCOUNT | POLICY_LOOKUP_NAMES */, out h);
    if (st != 0) return LsaNtStatusToWinError(st);
    var r = new LSA_UNICODE_STRING[1];
    r[0].Buffer = Marshal.StringToHGlobalUni(right);
    r[0].Length = (ushort)(right.Length * 2);
    r[0].MaximumLength = (ushort)(right.Length * 2 + 2);
    st = add ? LsaAddAccountRights(h, b, r, 1) : LsaRemoveAccountRights(h, b, false, r, 1);
    Marshal.FreeHGlobal(r[0].Buffer);
    LsaClose(h);
    return st == 0 ? 0 : LsaNtStatusToWinError(st);
  }
}
'@

# Run probe_set.ps1 as $Principal in a temporary scheduled task; wait for it, then delete the task.
function Invoke-SetAsTask([string]$Task, $Principal, [string]$Out) {
  $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$set`" -Dir `"$Dir`" -Out `"$Out`""
  $act = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $Dir
  $cfg = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  try {
    Register-ScheduledTask -TaskName $Task -Action $act -Principal $Principal -Settings $cfg -Force -ErrorAction Stop | Out-Null
    Start-ScheduledTask -TaskName $Task -ErrorAction Stop
    $t0 = Get-Date
    do {
      Start-Sleep -Seconds 5
      $state = (Get-ScheduledTask -TaskName $Task).State
      $done = (Test-Path $Out) -and (Select-String -Path $Out -Pattern '^done$' -Quiet)
    } while (-not $done -and ($state -eq 'Running' -or ((Get-Date) - $t0).TotalSeconds -lt 20) -and ((Get-Date) - $t0).TotalMinutes -lt 11)
    $info = Get-ScheduledTaskInfo -TaskName $Task
    Write-Output "task $Task state $state, last result $($info.LastTaskResult), $([int]((Get-Date) - $t0).TotalSeconds) s"
    if ($state -eq 'Running') { Stop-ScheduledTask -TaskName $Task }
  } catch {
    Write-Output "task $Task error: $_"
  } finally {
    Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue
  }
  if (Test-Path $Out) { Get-Content $Out } else { Write-Output "no output from $Task" }
}

Write-Output "== 1. runner account"
& $set -Dir $Dir -Out "$OutDir\1-runner.txt"
Get-Content "$OutDir\1-runner.txt"

Write-Output "== 2. SYSTEM (scheduled task, no policy change)"
$sys = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
Invoke-SetAsTask 'flm-large-pages-system' $sys "$OutDir\2-system.txt"

$got = (Test-Path "$OutDir\2-system.txt") -and (Select-String -Path "$OutDir\2-system.txt" -Pattern 'alloc ok' -Quiet)
if ($got) {
  Write-Output "== 3. skipped: SYSTEM got large pages, no policy change needed"
} else {
  Write-Output "== 3. runner account with SeLockMemoryPrivilege (granted, fresh S4U logon, revoked)"
  $me = [System.Security.Principal.WindowsIdentity]::GetCurrent()
  $held = whoami /priv | Select-String SeLockMemoryPrivilege -Quiet
  $added = $false
  try {
    if (-not $held) {
      $e = [LsaRights]::Set($me.User.Value, 'SeLockMemoryPrivilege', $true)
      Write-Output "grant SeLockMemoryPrivilege to $($me.Name): error $e"
      $added = $e -eq 0
    }
    $usr = New-ScheduledTaskPrincipal -UserId $me.Name -LogonType S4U -RunLevel Highest
    Invoke-SetAsTask 'flm-large-pages-user' $usr "$OutDir\3-user.txt"
  } finally {
    if ($added) {
      $e = [LsaRights]::Set($me.User.Value, 'SeLockMemoryPrivilege', $false)
      Write-Output "revoke SeLockMemoryPrivilege from $($me.Name): error $e"
    }
  }
}
Get-ScheduledTask -TaskName 'flm-large-pages-*' -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "WARNING: task left: $($_.TaskName)" }
