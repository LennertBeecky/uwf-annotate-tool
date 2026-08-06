"""Diagnose a broken annotation-tool install.

Answers one question: why won't napari open? Checks each layer in turn —
interpreter, scientific stack, Qt binding, OpenGL, napari itself — and
prints a verdict with the fix to try. Every check is wrapped, so the
report is complete even when something explodes half way.

    python annotation_tool/doctor.py                 # writes
                                                     # uwf_annotate_diagnostic.txt
                                                     # into the current folder
    python annotation_tool/doctor.py somewhere.txt   # or a path you choose
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

QT_BINDINGS = ("PyQt6", "PyQt5", "PySide6", "PySide2")


def _line(msg: str = "") -> None:
    print(msg)


def _check(label: str, fn) -> tuple[bool, str]:
    """Run one probe. Returns (ok, detail) and never raises."""
    try:
        detail = fn()
        _line(f"  [ok]   {label}: {detail}")
        return True, str(detail)
    except Exception as exc:
        _line(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        return False, traceback.format_exc()


def _package_version(name: str):
    """Import a package out-of-process and report its version.

    Out-of-process because these imports load native code. A compiled
    extension built against the wrong runtime does not raise ImportError,
    it takes the interpreter down with it — and an in-process import then
    kills the diagnostic mid-sentence, so the report stops at a section
    heading and says nothing about why. Isolated, the crash is just a
    non-zero exit code we can name.
    """
    def probe():
        return _run_isolated(
            f"import {name} as m\n"
            f"print(getattr(m, '__version__', '(no __version__)'))\n"
        )
    return probe


def _installed_qt_bindings() -> list[str]:
    found = []
    for name in QT_BINDINGS:
        try:
            __import__(name)
            found.append(name)
        except Exception:
            pass
    return found


# Windows reports a native crash as the exception code in the exit status.
# Bare numbers cost hours to look up, and the number is often the whole
# diagnosis, so translate the ones this stack actually dies with.
_WINDOWS_CRASH_CODES = {
    0xC06D007E: "a DLL a Qt/napari component needs was not found at load "
                "time (delay-load module missing)",
    0xC06D007F: "a DLL was found but is the wrong version — it does not "
                "contain a function that was asked for (delay-load entry "
                "point missing). Usually an out-of-date Visual C++ "
                "Redistributable, or a stale DLL earlier on PATH.",
    0xC0000005: "access violation — a segfault inside a native library",
    0xC0000135: "a required DLL could not be found",
    0xC0000142: "a DLL failed to initialise",
    0xC000007B: "a 32-bit/64-bit mismatch between loaded DLLs",
    0xC0000409: "stack buffer overrun (fail-fast) inside a native library",
    0xC00000FD: "stack overflow",
}


def _describe_exit_code(code: int) -> str:
    """Human-readable cause for a native crash exit status, if we know one."""
    unsigned = code & 0xFFFFFFFF
    known = _WINDOWS_CRASH_CODES.get(unsigned)
    text = f"code {code} (0x{unsigned:08X})"
    return f"{text}: {known}" if known else text


def _dll_version(path) -> str:
    """Version string of a Windows DLL, or '' if it cannot be read."""
    import ctypes
    import ctypes.wintypes as wt

    ver = ctypes.WinDLL("version")
    size = ver.GetFileVersionInfoSizeW(ctypes.c_wchar_p(str(path)), None)
    if not size:
        return ""
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(ctypes.c_wchar_p(str(path)), 0, size, buf):
        return ""
    block = ctypes.c_void_p()
    length = wt.UINT()
    if not ver.VerQueryValueW(buf, ctypes.c_wchar_p("\\"),
                              ctypes.byref(block), ctypes.byref(length)):
        return ""

    class FixedFileInfo(ctypes.Structure):
        _fields_ = [("dwSignature", wt.DWORD), ("dwStrucVersion", wt.DWORD),
                    ("dwFileVersionMS", wt.DWORD), ("dwFileVersionLS", wt.DWORD)]

    info = ctypes.cast(block, ctypes.POINTER(FixedFileInfo)).contents
    return "%d.%d.%d.%d" % (info.dwFileVersionMS >> 16,
                            info.dwFileVersionMS & 0xFFFF,
                            info.dwFileVersionLS >> 16,
                            info.dwFileVersionLS & 0xFFFF)


# Shipped by the Visual C++ Redistributable. Every compiled Python
# extension on Windows — numpy, Qt, the lot — links against these, so an
# out-of-date copy is a fault that spans unrelated packages.
VCRUNTIME_DLLS = ("VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "MSVCP140.dll")


def _vcruntime_report() -> list[str]:
    """One line per Visual C++ runtime DLL: version, or that it is missing."""
    import pathlib

    system32 = pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    lines = []
    for name in VCRUNTIME_DLLS:
        path = system32 / name
        if not path.exists():
            lines.append(f"  [FAIL] {name}: MISSING")
            continue
        try:
            version = _dll_version(path) or "(version unreadable)"
        except Exception as exc:
            version = f"(version unreadable: {type(exc).__name__})"
        lines.append(f"  [ok]   {name}: {version}")
    return lines


def _run_isolated(code: str) -> str:
    """Run a probe in a fresh interpreter and return its last line.

    Anything that touches Qt runs out-of-process. A QApplication created
    here would collide with the one napari makes later — "QApplication::
    regClass ... class already exists" — so the diagnostic would trigger
    the very fault it is meant to report, and take itself down before
    writing anything.
    """
    import subprocess

    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=180)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = (out + "\n" + err).strip()
        raise RuntimeError(detail[-1200:] if detail
                           else f"process died with "
                                f"{_describe_exit_code(proc.returncode)} "
                                f"(no output — a hard crash, not an exception)")
    return out.splitlines()[-1] if out else "ok"


_GL_CODE = """
from qtpy.QtGui import QOffscreenSurface, QOpenGLContext
from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
surface = QOffscreenSurface(); surface.create()
ctx = QOpenGLContext()
assert ctx.create(), 'QOpenGLContext.create() failed - no usable OpenGL'
assert ctx.makeCurrent(surface), 'could not make the OpenGL context current'
fmt = ctx.format()
print('OpenGL %d.%d' % (fmt.majorVersion(), fmt.minorVersion()))
"""

_VIEWER_CODE = """
import numpy as np, napari
v = napari.Viewer(show=False)
v.add_image(np.zeros((16, 16), dtype='uint8'), name='probe')
print('viewer created with %d layer(s)' % len(v.layers))
v.close()
"""


# The probe above builds the viewer with show=False, which never puts a
# window on screen. That is the wrong thing to test on a machine where the
# complaint is "the window never appears": constructing a viewer and
# displaying one exercise different code — real window creation is where a
# GPU driver, a remote desktop session or endpoint protection interferes.
# So show a real window, let the event loop run, and close it on a timer.
_WINDOW_CODE = """
import numpy as np, napari
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer
v = napari.Viewer()
v.add_image(np.zeros((16, 16), dtype='uint8'), name='probe')
QTimer.singleShot(2500, QApplication.instance().quit)
napari.run()
print('window opened and closed cleanly')
"""


def _opengl_probe() -> str:
    """Offscreen GL context, out-of-process, reporting the driver version."""
    return _run_isolated(_GL_CODE)


def _viewer_probe() -> str:
    return _run_isolated(_VIEWER_CODE)


def _window_probe() -> str:
    """Open a real napari window briefly — what the annotator actually does."""
    return _run_isolated(_WINDOW_CODE)


def main(argv: list[str]) -> int:
    tracebacks: dict[str, str] = {}

    _line("=" * 64)
    _line("  UWF annotation tool — install diagnostic")
    _line("=" * 64)
    _line()
    _line("System")
    _line(f"  platform:   {platform.platform()}")
    _line(f"  python:     {sys.version.split()[0]}")
    _line(f"  executable: {sys.executable}")
    for var in ("QT_OPENGL", "QT_API", "QT_QPA_PLATFORM", "CONDA_PREFIX"):
        if os.environ.get(var):
            _line(f"  {var}={os.environ[var]}")
    _line()

    if sys.platform == "win32":
        _line("Visual C++ runtime")
        for entry in _vcruntime_report():
            _line(entry)
        _line()

    _line("Scientific stack")
    stack_ok = True
    for pkg in ("numpy", "PIL", "scipy", "skimage"):
        ok, detail = _check(pkg, _package_version(pkg))
        stack_ok &= ok
        if not ok:
            tracebacks[pkg] = detail
    _line()

    _line("Qt")
    bindings = _installed_qt_bindings()
    _line(f"  installed bindings: {', '.join(bindings) if bindings else 'NONE'}")
    qt_ok, detail = _check("qtpy imports a binding",
                           lambda: __import__("qtpy").API_NAME)
    if not qt_ok:
        tracebacks["qtpy"] = detail
    _line()

    _line("OpenGL")
    gl_ok, detail = _check("offscreen GL context", _opengl_probe)
    if not gl_ok:
        tracebacks["opengl"] = detail
    _line()

    _line("napari")
    napari_ok, detail = _check("import napari", _package_version("napari"))
    if not napari_ok:
        tracebacks["napari"] = detail
    napari_version_matches = True
    if napari_ok:
        try:
            installed = __import__("napari").__version__
        except Exception:
            installed = ""
        napari_version_matches = installed == TESTED_NAPARI_VERSION
        if not napari_version_matches:
            _line(f"  [warn] this is not the tested version "
                  f"({TESTED_NAPARI_VERSION}) — see the verdict below")
    viewer_ok = False
    window_ok = False
    if napari_ok:
        viewer_ok, detail = _check("open a viewer", _viewer_probe)
        if not viewer_ok:
            tracebacks["viewer"] = detail
        if viewer_ok:
            _line("  ... opening a real window for 3 seconds, please wait")
            window_ok, detail = _check("show a window", _window_probe)
            if not window_ok:
                tracebacks["window"] = detail
    _line()

    _line("=" * 64)
    if viewer_ok and not window_ok:
        _line("  VERDICT: napari builds a viewer, but putting a real window")
        _line("  on screen fails. The packages are fine — something outside")
        _line("  them is stopping the window appearing.")
        _line("  Usual causes, in order:")
        _line("    1. endpoint protection blocking the window")
        _line("    2. a remote desktop / virtual display session")
        _line("    3. the GPU driver refusing an on-screen GL surface —")
        _line("       try starting the tool with QT_OPENGL=software")
        _line("  Send the 'window' traceback below.")
    elif viewer_ok:
        _line("  VERDICT: napari works. The graphics stack is fine.")
        _line("  If the annotation tool still fails, the problem is in the")
        _line("  tool or the batch — send the error from annotate.bat.")
    elif not stack_ok:
        _line("  VERDICT: the conda environment is incomplete.")
        _line("  FIX: re-run setup and watch for errors while it builds the")
        _line("       environment.")
    elif len(bindings) > 1:
        _line(f"  VERDICT: {len(bindings)} Qt bindings installed "
              f"({', '.join(bindings)}).")
        _line("  Two Qt libraries in one process collide - the symptom is")
        _line("  'QApplication::regClass ... class already exists' and a")
        _line("  window that never appears.")
        _line("  FIX: keep one. In the uwf-annotate environment run")
        for extra in bindings[1:]:
            _line(f"    pip uninstall -y {extra}")
    elif not bindings or not qt_ok:
        _line("  VERDICT: no working Qt binding.")
        _line("  FIX: pip install \"napari[pyqt5]\"")
    elif not gl_ok:
        _line("  VERDICT: no usable OpenGL — this is why napari won't open.")
        _line("  Common on remote desktop sessions, virtual machines, and")
        _line("  PCs with the generic display driver Windows Update ships.")
        _line("  FIX, in order:")
        _line("    1. set QT_OPENGL=software     (then start the tool again)")
        _line("    2. install the GPU driver from Intel/NVIDIA/AMD directly")
        _line("    3. if this is a remote desktop, try it at the machine")
    elif not napari_version_matches:
        _line("  VERDICT: Qt and OpenGL are fine, but napari is not the")
        _line(f"  version this tool was tested against ({TESTED_NAPARI_VERSION}).")
        _line("  An untested napari is the most likely cause of a viewer")
        _line("  that dies without an error message.")
        _line("  FIX: in the uwf-annotate environment run")
        _line(f"    pip install \"napari[all]=={TESTED_NAPARI_VERSION}\"")
    else:
        _line("  VERDICT: Qt and OpenGL look fine but the viewer failed.")
        _line("  Send the traceback below.")
    _line("=" * 64)

    if tracebacks:
        _line()
        _line("Tracebacks")
        for name, tb in tracebacks.items():
            _line(f"--- {name} " + "-" * (58 - len(name)))
            _line(tb.rstrip())
    return 0 if viewer_ok else 1


DEFAULT_REPORT_NAME = "uwf_annotate_diagnostic.txt"

# The napari the tool is actually tested against. Keep in step with the pin
# in environment_clinician.yml / requirements_clinician.txt.
TESTED_NAPARI_VERSION = "0.7.0"


def resolve_report_path(requested: str | None) -> "Path":
    """Where to write the report. Defaults to the current folder.

    Do not try to be clever about the Desktop: on Windows OneDrive
    redirects it, so `%USERPROFILE%\\Desktop` frequently does not exist and
    the report vanishes. The folder the script was run from always exists
    and the annotator is already looking at it.
    """
    from pathlib import Path

    if requested is None:
        return Path.cwd() / DEFAULT_REPORT_NAME
    p = Path(requested).expanduser()
    if p.is_dir():
        return p / DEFAULT_REPORT_NAME
    if p.parent.exists():
        return p
    return Path.cwd() / p.name


class _Tee:
    """Write to the console and to the report file, flushing every line.

    The report is written as it is produced, not buffered and saved at the
    end. The failure this tool exists to diagnose — Qt or OpenGL unable to
    start — kills the process outright rather than raising, so a report
    assembled in memory would never reach disk. Written this way, the last
    line in the file is the probe that killed it.
    """

    def __init__(self, console, handle):
        self._console = console
        self._handle = handle

    def write(self, text):
        self._console.write(text)
        if self._handle is not None:
            try:
                self._handle.write(text)
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception:
                pass

    def flush(self):
        self._console.flush()


if __name__ == "__main__":
    # Always write a report, from the first line. Requiring an argument to
    # get a file is how the report went missing the first time; buffering it
    # is how it went missing the second.
    out_path = resolve_report_path(sys.argv[1] if len(sys.argv) > 1 else None)

    handle = None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(out_path, "w", encoding="utf-8", buffering=1)
    except Exception as exc:
        print(f"Could not open {out_path} for writing ({exc}).")

    real = sys.stdout
    sys.stdout = _Tee(real, handle)
    print(f"(report file: {out_path.resolve() if handle else 'NOT WRITABLE'})")

    code = 1
    try:
        code = main(sys.argv)
    except BaseException:
        print("\nThe diagnostic itself crashed:")
        print(traceback.format_exc())
    finally:
        sys.stdout = real
        if handle is not None:
            handle.close()
            print("\n" + "=" * 64)
            print("  REPORT WRITTEN TO:")
            print(f"    {out_path.resolve()}")
            print("  Send that file — or copy the VERDICT block above.")
            print("=" * 64)
    sys.exit(code)
