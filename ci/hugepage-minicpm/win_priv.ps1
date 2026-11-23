# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# win_priv.ps1: can this account get 2 MB (large) pages on Windows? Read-only: lists the token's privileges
# and admin membership, then tries to enable SeLockMemoryPrivilege in its own token (works only if the
# account already holds it) and one 2 MB VirtualAlloc(MEM_LARGE_PAGES). Changes no policy, uses no NPU.
$ErrorActionPreference = "Continue"
"runner $env:RUNNER_NAME"
"cpu $((Get-CimInstance Win32_Processor).Name)"
"os $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
"user $(whoami)"
"== privileges"
whoami /priv
"== administrators group (S-1-5-32-544) in token"
whoami /groups | Select-String "S-1-5-32-544"
"elevated: $(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class LargePage {
    [StructLayout(LayoutKind.Sequential, Pack = 4)] struct LUID { public uint Low; public int High; }
    [StructLayout(LayoutKind.Sequential, Pack = 4)] struct TOKEN_PRIVILEGES { public uint Count; public LUID Luid; public uint Attr; }
    [DllImport("kernel32.dll")] public static extern UIntPtr GetLargePageMinimum();
    [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr VirtualAlloc(IntPtr a, UIntPtr s, uint t, uint p);
    [DllImport("kernel32.dll")] static extern bool VirtualFree(IntPtr a, UIntPtr s, uint t);
    [DllImport("advapi32.dll", SetLastError = true)] static extern bool OpenProcessToken(IntPtr h, uint acc, out IntPtr tok);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool LookupPrivilegeValue(string sys, string name, out LUID luid);
    [DllImport("advapi32.dll", SetLastError = true)] static extern bool AdjustTokenPrivileges(IntPtr tok, bool all, ref TOKEN_PRIVILEGES np, uint len, IntPtr prev, IntPtr rl);
    public static string Enable() {
        IntPtr tok;
        if (!OpenProcessToken(GetCurrentProcess(), 0x28, out tok)) return "OpenProcessToken error " + Marshal.GetLastWin32Error();
        TOKEN_PRIVILEGES tp = new TOKEN_PRIVILEGES { Count = 1, Attr = 2 };
        if (!LookupPrivilegeValue(null, "SeLockMemoryPrivilege", out tp.Luid)) return "LookupPrivilegeValue error " + Marshal.GetLastWin32Error();
        bool ok = AdjustTokenPrivileges(tok, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
        int err = Marshal.GetLastWin32Error();   // 1300 = ERROR_NOT_ALL_ASSIGNED: the account lacks the privilege
        return "AdjustTokenPrivileges " + ok + ", error " + err + (err == 1300 ? " (not held by this account)" : "");
    }
    public static string Alloc() {
        UIntPtr n = GetLargePageMinimum();
        IntPtr p = VirtualAlloc(IntPtr.Zero, n, 0x20003000, 4);   // MEM_COMMIT|MEM_RESERVE|MEM_LARGE_PAGES, PAGE_READWRITE
        if (p == IntPtr.Zero) return "VirtualAlloc(MEM_LARGE_PAGES) failed, error " + Marshal.GetLastWin32Error();
        VirtualFree(p, UIntPtr.Zero, 0x8000);
        return "VirtualAlloc(MEM_LARGE_PAGES) ok";
    }
}
"@
"== large pages"
"GetLargePageMinimum $([LargePage]::GetLargePageMinimum()) bytes"
[LargePage]::Enable()
[LargePage]::Alloc()
exit 0
