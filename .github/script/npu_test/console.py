"""A program in a hidden console of its own, driven the way a person drives it.

Windows only. Ctrl+C reaches every process attached to a console, so the
program gets a console nothing else shares: the signal reaches it alone, and
keys written to that console's input go to its prompt. Only a process without
a console can attach to one, so each action runs in a short-lived copy of this
file:

    python console.py <pid> ctrl-c | screen | line <text>
"""

from __future__ import annotations

import subprocess
import sys

# How a process that the default Ctrl+C handler ended exits.
STATUS_CONTROL_C_EXIT = 0xC000013A
DETACHED_PROCESS = 0x00000008


def start(command: list[str], *, stdout=None) -> subprocess.Popen:
    """Starts command in a new console with its window hidden.

    With stdout, a file, the program writes there instead of to its console.
    """
    import ctypes

    # A process inherits whether it ignores Ctrl+C; the program must not,
    # whatever started this one.
    ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0  # SW_HIDE
    streams = {}
    if stdout is not None:
        streams = {"stdin": subprocess.DEVNULL, "stdout": stdout, "stderr": subprocess.STDOUT}
    return subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE,
                            startupinfo=info, **streams)


def ctrl_c(pid: int) -> None:
    """Presses Ctrl+C in the console of process pid."""
    _act(pid, "ctrl-c")


def type_line(pid: int, text: str) -> None:
    """Types text and Enter at the prompt of process pid."""
    _act(pid, "line", text)


def screen(pid: int) -> str:
    """The text in the console of process pid, blank lines left out."""
    return _act(pid, "screen")


def _act(pid: int, *args: str) -> str:
    proc = subprocess.run([sys.executable, __file__, str(pid), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          creationflags=DETACHED_PROCESS, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"console {args[0]} for pid {pid} failed: "
                           f"{(proc.stderr or proc.stdout).strip()[-300:]}")
    return proc.stdout


def _main(argv: list[str]) -> int:
    import ctypes
    from ctypes import wintypes as w

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32")
    kernel32.CreateFileW.restype = w.HANDLE
    kernel32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD,
                                     w.DWORD, w.HANDLE]

    class COORD(ctypes.Structure):
        _fields_ = [("X", w.SHORT), ("Y", w.SHORT)]

    class SMALL_RECT(ctypes.Structure):
        _fields_ = [("Left", w.SHORT), ("Top", w.SHORT), ("Right", w.SHORT), ("Bottom", w.SHORT)]

    class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", w.WORD),
                    ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", w.BOOL), ("wRepeatCount", w.WORD), ("wVirtualKeyCode", w.WORD),
                    ("wVirtualScanCode", w.WORD), ("uChar", w.WCHAR),
                    ("dwControlKeyState", w.DWORD)]

    class INPUT_EVENT(ctypes.Union):
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("padding", ctypes.c_byte * 16)]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", w.WORD), ("Event", INPUT_EVENT)]

    def fail(what: str) -> int:
        print(f"{what} failed: error {ctypes.get_last_error()}", file=sys.stderr)
        return 1

    def open_console(name: str):
        # GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING
        handle = kernel32.CreateFileW(name, 0xC0000000, 3, None, 3, 0, None)
        return None if handle in (None, w.HANDLE(-1).value) else handle

    pid, action = int(argv[0]), argv[1]
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(pid):
        return fail(f"AttachConsole({pid})")

    if action == "ctrl-c":
        # This process shares the console now, so it must ignore the signal itself.
        kernel32.SetConsoleCtrlHandler(None, True)
        if not kernel32.GenerateConsoleCtrlEvent(0, 0):  # CTRL_C_EVENT, the whole console
            return fail("GenerateConsoleCtrlEvent")
        return 0

    if action == "screen":
        out = open_console("CONOUT$")
        info = CONSOLE_SCREEN_BUFFER_INFO()
        if out is None or not kernel32.GetConsoleScreenBufferInfo(out, ctypes.byref(info)):
            return fail("reading CONOUT$")
        width = info.dwSize.X
        cells = width * (info.dwCursorPosition.Y + 1)
        buffer = ctypes.create_unicode_buffer(cells)
        read = w.DWORD()
        if not kernel32.ReadConsoleOutputCharacterW(out, buffer, cells, COORD(0, 0),
                                                    ctypes.byref(read)):
            return fail("ReadConsoleOutputCharacterW")
        text = buffer[:read.value]
        rows = (text[i:i + width].rstrip() for i in range(0, len(text), width))
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.write("\n".join(row for row in rows if row))
        return 0

    if action == "line":
        keys = argv[2] + "\r"
        records = (INPUT_RECORD * (2 * len(keys)))()
        for i, char in enumerate(keys):
            vk = 0x0D if char == "\r" else user32.VkKeyScanW(ord(char)) & 0xFF
            for j, down in enumerate((True, False)):
                record = records[2 * i + j]
                record.EventType = 1  # KEY_EVENT
                key = record.Event.KeyEvent
                key.bKeyDown, key.wRepeatCount, key.wVirtualKeyCode = down, 1, vk
                key.wVirtualScanCode = user32.MapVirtualKeyW(vk, 0)
                key.uChar = char
        conin = open_console("CONIN$")
        written = w.DWORD()
        if conin is None or not kernel32.WriteConsoleInputW(conin, records, len(records),
                                                            ctypes.byref(written)):
            return fail("WriteConsoleInputW")
        return 0

    print(f"unknown action {action!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
