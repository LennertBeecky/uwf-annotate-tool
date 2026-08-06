"""Keybinding registration for the napari session (no napari required).

napari resolves a keypress against the *active layer's* keymap before the
viewer's, so a viewer-only binding can be shadowed by a Labels-layer
default. The session therefore registers each A/V swap key on the viewer
AND on both label layers; these tests pin that down with fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from annotate import SWAP_KEYS, bind_key_everywhere  # type: ignore  # noqa: E402


class FakeBindable:
    def __init__(self):
        self.bound = {}

    def bind_key(self, key, func=None, *, overwrite=False):
        if not overwrite and key in self.bound:
            raise ValueError(f"{key} already bound")
        self.bound[key] = func
        return func


def test_binds_on_every_target():
    viewer, artery, veins = FakeBindable(), FakeBindable(), FakeBindable()

    def handler(_ctx):
        return "ran"

    bind_key_everywhere([viewer, artery, veins], "f", handler)
    for target in (viewer, artery, veins):
        assert target.bound["f"] is handler


def test_overwrites_an_existing_napari_default():
    layer = FakeBindable()
    layer.bind_key("f", lambda _l: "napari default")
    bind_key_everywhere([layer], "f", lambda _l: "ours")
    assert layer.bound["f"](None) == "ours"


def test_a_target_that_refuses_a_key_does_not_break_the_others():
    class Stubborn(FakeBindable):
        def bind_key(self, key, func=None, *, overwrite=False):
            raise RuntimeError("unsupported key")

    good = FakeBindable()
    bind_key_everywhere([Stubborn(), good], "u", lambda _c: None)
    assert "u" in good.bound


def test_swap_keys_are_declared_and_distinct():
    keys = [k for k, _ in SWAP_KEYS]
    assert len(keys) == len(set(keys))
    assert {"w", "Shift-W", "h", "g", "u", "Shift-X", ".", ",",
            "Meta-Z", "Control-Z"} == set(keys)


def test_undo_is_reachable_by_u_and_by_the_usual_undo_chords():
    undo_keys = {k for k, mode in SWAP_KEYS if mode == "undo"}
    assert undo_keys == {"u", "Meta-Z", "Control-Z"}


def test_swap_keys_do_not_collide_with_the_tool_s_own_keys():
    """`q`/`s`/`Tab`/`1`/`3`/`[`/`]` are already spoken for."""
    existing = {"q", "s", "Tab", "1", "3", "[", "]"}
    assert existing.isdisjoint({k for k, _ in SWAP_KEYS})


def test_swap_keys_do_not_collide_with_napari_defaults():
    """A key napari already owns may trigger its action as well as ours.

    `f` is napari's labels *fill* mode and `x` swaps the selected and
    background labels — both would fire alongside our handler.
    """
    napari_settings = pytest.importorskip("napari.settings")
    taken = {
        str(k).lower().replace("+", "-")
        for keys in napari_settings.get_settings().shortcuts.shortcuts.values()
        for k in keys
    }
    clashes = [k for k, _ in SWAP_KEYS if k.lower() in taken]
    assert not clashes, f"napari already binds {clashes}"


def test_boundary_output_is_the_default():
    """`annotate.py <img>` keeps the painted width; --skeleton opts out."""
    import inspect

    import annotate  # noqa: PLC0415

    sig = inspect.signature(annotate._open_annotation_session)
    assert sig.parameters["boundaries"].default is True


def test_pixel_output_is_never_downgraded_automatically():
    """Pixels are the deliverable. A skeleton is derivable from them.

    Nothing may flip the output to skeleton on its own — not the prefill
    kind, not the batch type. Only an explicit --skeleton does that.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "annotate.py"
    text = src.read_text()
    assert "boundaries = False" not in text
    assert "boundaries = not args.skeleton" in text


def test_the_clinician_launchers_do_not_thin_the_output():
    """The launchers are what actually runs in the field."""
    import pathlib
    import subprocess

    repo = pathlib.Path(__file__).resolve().parents[2]
    for script in ("annotate.command", "annotate.bat"):
        blob = subprocess.run(
            ["git", "show", f"clinician_setup:scripts/clinician/{script}"],
            cwd=repo, capture_output=True, text=True)
        if blob.returncode != 0:
            continue                      # branch not present in this checkout
        body = "\n".join(l for l in blob.stdout.splitlines()
                         if not l.strip().startswith(("#", "REM")))
        assert "--skeleton" not in body, f"{script} still thins the output"
