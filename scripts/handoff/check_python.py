"""Report whether the running interpreter is supported by the falls_ml handoff package.

Usage: python scripts/handoff/check_python.py [--write-exe <file>]

Prints one line   PYTHON|<ok|unsupported>|<version>|<bits>|<machine>|<executable>
and, when unsupported, a second line   REASON|<text>
Exit code 0 = supported (CPython 3.11.x or 3.13.x, 64-bit, x86-64 on Windows; arm64 accepted on macOS for developers),
3 = unsupported, 2 = usage error. With --write-exe the interpreter path (sys.executable) is written to <file> as one line.

Deliberately written in Python 2.7/3.0+ compatible syntax (no f-strings, no annotations) so that an old interpreter
reports "unsupported" instead of failing with a SyntaxError. Standard library only.
"""
from __future__ import print_function

import os
import platform
import sys

SUPPORTED = ((3, 11), (3, 13))  # the CPython series the lock files carry wheels for
REQUIRED = (3, 13)  # preferred series (development and verification)
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNSUPPORTED = 3
_WINDOWS_MACHINES = {0x8664: "AMD64", 0xAA64: "ARM64", 0x014C: "x86", 0x01C4: "ARM"}


def windows_native_machine():
    """Native OS architecture on Windows (sees through x64 emulation on ARM64), or None when unknown."""
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        func = getattr(kernel32, "IsWow64Process2", None)
        if func is None:
            return None
        process_machine, native_machine = wintypes.USHORT(0), wintypes.USHORT(0)
        if not func(kernel32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine)):
            return None
        return _WINDOWS_MACHINES.get(native_machine.value, hex(native_machine.value))
    except Exception:  # noqa: BLE001 - any ctypes problem means "unknown", never a crash
        return None


def interpreter_facts():
    """Facts about the running interpreter as a dict (all values plain strings/ints)."""
    facts = {
        "implementation": platform.python_implementation(),
        "version_info": tuple(sys.version_info[:3]),
        "version": "%d.%d.%d" % tuple(sys.version_info[:3]),
        "bits": 64 if sys.maxsize > 2 ** 32 else 32,
        "machine": platform.machine() or "unknown",
        "platform": sys.platform,
        "executable": sys.executable or "",
        "sysconfig_platform": "",
        "native_machine": None,
    }
    try:
        import sysconfig
        facts["sysconfig_platform"] = sysconfig.get_platform()
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        facts["native_machine"] = windows_native_machine()
    return facts


def evaluate(facts):
    """Return (supported: bool, reason: str or None, machine: str) for interpreter facts."""
    version = facts["version_info"]
    if facts["implementation"] != "CPython":
        return False, "%s is not supported; CPython 3.11 or 3.13 (64-bit) is required" % facts["implementation"], facts["machine"]
    if tuple(version[:2]) not in SUPPORTED:
        return False, ("Python %s found; Python 3.11.x or 3.13.x (64-bit) is required "
                       "(the lock files carry wheels for those two series only)") % facts["version"], facts["machine"]
    if facts["bits"] != 64:
        return False, "32-bit Python found; a 64-bit (x86-64) Python 3.11 or 3.13 installer is required", facts["machine"]
    if facts["platform"] == "win32":
        interp = (facts["sysconfig_platform"] or "").lower()
        native = facts["native_machine"]
        machine = native or facts["machine"]
        if native == "ARM64" or interp == "win-arm64" or (facts["machine"] or "").upper() == "ARM64":
            return False, "Windows on ARM64 is not supported; use a Windows 10/11 x64 computer", "ARM64"
        if interp and interp != "win-amd64":
            return False, "interpreter platform %s is not supported; install the Windows x86-64 (64-bit) Python 3.11 or 3.13" % interp, machine
        if (machine or "").upper() not in ("AMD64", "X86_64"):
            return False, "machine %s is not supported; Windows 10/11 x64 is required" % machine, machine
        return True, None, machine
    if facts["platform"] == "darwin":
        if (facts["machine"] or "").lower() == "arm64":
            return True, None, facts["machine"]
        return False, "macOS %s is not supported (developer mirror supports macOS arm64 only)" % facts["machine"], facts["machine"]
    return False, "platform %s/%s is not supported; the lock files cover Windows x64 (and macOS arm64 for developers)" % (
        facts["platform"], facts["machine"]), facts["machine"]


def format_lines(facts, supported, reason, machine):
    status = "ok" if supported else "unsupported"
    lines = ["PYTHON|%s|%s|%d|%s|%s" % (status, facts["version"], facts["bits"], machine, facts["executable"])]
    if not supported:
        lines.append("REASON|%s" % reason)
    return lines


def parse_line(line):
    """Parse a PYTHON|... line (used by the installer helpers and tests); returns a dict or None."""
    parts = line.strip().split("|", 5)
    if len(parts) != 6 or parts[0] != "PYTHON" or parts[1] not in ("ok", "unsupported"):
        return None
    return {"status": parts[1], "version": parts[2], "bits": parts[3], "machine": parts[4], "executable": parts[5]}


def write_exe(path, executable):
    """Write the interpreter path as one line in a form CMD's `set /p` can read back (console code page / 8.3 path)."""
    text = executable
    encoding = "utf-8"
    if sys.platform == "win32":
        encoding = "mbcs"
        try:
            import ctypes
            code_page = ctypes.windll.kernel32.GetConsoleOutputCP() or ctypes.windll.kernel32.GetOEMCP()
            encoding = "cp%d" % code_page
            "x".encode(encoding)
        except Exception:  # noqa: BLE001
            encoding = "mbcs"
        try:
            text.encode(encoding)
        except (UnicodeError, LookupError):
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(32768)
                if ctypes.windll.kernel32.GetShortPathNameW(executable, buf, 32768):
                    text = buf.value
                text.encode(encoding)
            except Exception:  # noqa: BLE001 - fall back to an ASCII-safe path; CMD reports it as not found
                text = executable.encode("ascii", "replace").decode("ascii")
                encoding = "ascii"
    handle = open(path, "wb")
    try:
        handle.write((text + os.linesep).encode(encoding, "replace"))
    finally:
        handle.close()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    exe_file = None
    if args:
        if len(args) != 2 or args[0] != "--write-exe":
            print("usage: check_python.py [--write-exe <file>]", file=sys.stderr)
            return EXIT_USAGE
        exe_file = args[1]
    facts = interpreter_facts()
    supported, reason, machine = evaluate(facts)
    for line in format_lines(facts, supported, reason, machine):
        print(line)
    if exe_file is not None:
        write_exe(exe_file, facts["executable"])
    return EXIT_OK if supported else EXIT_UNSUPPORTED


if __name__ == "__main__":
    sys.exit(main())
