"""Diagnose a broken annotation-tool install.

Answers one question: why won't napari open? Checks each layer in turn —
interpreter, scientific stack, Qt binding, OpenGL, napari itself — and
prints a verdict with the fix to try. Every check is wrapped, so the
report is complete even when something explodes half way.

    python annotation_tool/doctor.py                 # print the report
    python annotation_tool/doctor.py report.txt      # also write it to a file
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
    def probe():
        mod = __import__(name)
        return getattr(mod, "__version__", "(no __version__)")
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


def _opengl_probe() -> str:
    """Create an offscreen GL context and report the driver's version.

    This is the check that usually explains a silent napari crash: remote
    desktop sessions, VMs and stock Intel drivers often expose no usable
    OpenGL, and napari renders through it.
    """
    from qtpy.QtGui import QOffscreenSurface, QOpenGLContext
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    surface = QOffscreenSurface()
    surface.create()
    ctx = QOpenGLContext()
    if not ctx.create():
        raise RuntimeError("QOpenGLContext.create() failed — no usable OpenGL")
    if not ctx.makeCurrent(surface):
        raise RuntimeError("could not make the OpenGL context current")
    try:
        fmt = ctx.format()
        version = f"OpenGL {fmt.majorVersion()}.{fmt.minorVersion()}"
    finally:
        ctx.doneCurrent()
    return version


def _viewer_probe() -> str:
    import napari

    viewer = napari.Viewer(show=False)
    try:
        import numpy as np

        viewer.add_image(np.zeros((16, 16), dtype="uint8"), name="probe")
        return f"viewer created with {len(viewer.layers)} layer(s)"
    finally:
        viewer.close()


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
    viewer_ok = False
    if napari_ok:
        viewer_ok, detail = _check("open a viewer", _viewer_probe)
        if not viewer_ok:
            tracebacks["viewer"] = detail
    _line()

    _line("=" * 64)
    if viewer_ok:
        _line("  VERDICT: napari works. The graphics stack is fine.")
        _line("  If the annotation tool still fails, the problem is in the")
        _line("  tool or the batch — send the error from annotate.bat.")
    elif not stack_ok:
        _line("  VERDICT: the conda environment is incomplete.")
        _line("  FIX: re-run setup and watch for errors while it builds the")
        _line("       environment.")
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


def resolve_report_path(requested: str) -> "Path":
    """Somewhere the report can actually be written, and be found again.

    `%USERPROFILE%\\Desktop` often does not exist on Windows: OneDrive
    redirects the Desktop to `%USERPROFILE%\\OneDrive\\Desktop`, so writing
    there fails and the annotator is told to look for a file that was never
    created. Fall back through the plausible locations, and never fail —
    a report in the wrong folder beats no report.
    """
    from pathlib import Path

    p = Path(requested).expanduser()
    if p.parent.exists():
        return p
    home = Path.home()
    for alt in (home / "OneDrive" / "Desktop", home / "Desktop", home):
        if alt.is_dir():
            return alt / p.name
    return Path.cwd() / p.name


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    if out_path:
        out_path = str(resolve_report_path(out_path))
        import io

        buf = io.StringIO()
        real = sys.stdout
        sys.stdout = _Tee = type("Tee", (), {
            "write": lambda _s, t: (real.write(t), buf.write(t)) and None,
            "flush": lambda _s: real.flush(),
        })()
        code = main(sys.argv)
        sys.stdout = real
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(buf.getvalue())
            print("\n" + "=" * 64)
            print("  REPORT WRITTEN TO:")
            print(f"    {out_path}")
            print("  Send that file back. If you cannot find it, copy the")
            print("  VERDICT block above instead.")
            print("=" * 64)
        except Exception as exc:
            print(f"\nCould not write the report ({exc}).")
            print("Copy the VERDICT block above instead.")
    else:
        code = main(sys.argv)
    sys.exit(code)
