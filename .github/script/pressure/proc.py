"""Memory, handle and image-path readings for the server process (stdlib only)."""

import os
import threading
import time

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _ACCESS = 0x1000 | 0x0010  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ


def stats(pid):
    """{'rss': bytes, 'private': bytes, 'handles': n} or None once the process is gone."""
    if IS_WINDOWS:
        h = _k32.OpenProcess(_ACCESS, False, pid)
        if not h:
            return None
        try:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            count = wintypes.DWORD()
            if not _k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return None
            _k32.GetProcessHandleCount(h, ctypes.byref(count))
            return {"rss": pmc.WorkingSetSize, "private": pmc.PagefileUsage, "handles": count.value}
        finally:
            _k32.CloseHandle(h)
    try:
        fields = {}
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            fields[k] = v.split()
        return {"rss": int(fields["VmRSS"][0]) * 1024, "private": int(fields.get("RssAnon", ["0"])[0]) * 1024,
                "handles": len(os.listdir(f"/proc/{pid}/fd"))}
    except (OSError, KeyError):
        return None


def image_path(pid):
    if IS_WINDOWS:
        h = _k32.OpenProcess(0x1000, False, pid)
        if not h:
            return None
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            if not _k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return None
            return buf.value
        finally:
            _k32.CloseHandle(h)
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class Sampler(threading.Thread):
    """Samples stats(pid) every `period` s into self.samples as (t, stats)."""

    def __init__(self, pid, period=5.0):
        super().__init__(daemon=True)
        self.pid, self.period, self.samples = pid, period, []
        self._halt = threading.Event()

    def run(self):
        t0 = time.monotonic()
        while not self._halt.is_set():
            s = stats(self.pid)
            if s:
                self.samples.append((round(time.monotonic() - t0, 1), s))
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)
