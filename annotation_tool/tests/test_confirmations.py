"""Save/skip confirmation dialogs.

[q] and [s] used to close the window on the spot — one slip of the finger
next to Tab and half an hour of correction was either saved half-done or
silently discarded. Both keys now ask a yes/no question first; answering
No keeps the session open with everything still on the canvas.

The dialog itself is Qt; these tests exercise the decision wiring through
the same napari-stubbed session as test_session_wiring, with `ask_confirm`
patched to scripted answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from test_session_wiring import (  # noqa: E402,F401  (fixtures by name)
    FakeViewer,
    _run_session,
    napari_stub,
    scene,
)


def _patch_answers(monkeypatch, answers: list[bool]) -> list[str]:
    """Feed scripted yes/no answers to annotate.ask_confirm; record the
    questions it was asked."""
    import annotate

    questions: list[str] = []

    def fake_confirm(_viewer, question: str) -> bool:
        questions.append(question)
        return answers[len(questions) - 1]

    monkeypatch.setattr(annotate, "ask_confirm", fake_confirm)
    return questions


def test_q_asks_before_saving_and_yes_saves(napari_stub, scene, monkeypatch):
    tmp_path, artery, vein = scene
    questions = _patch_answers(monkeypatch, [True])

    FakeViewer.script = lambda v: v.press("q")
    assert _run_session(tmp_path, artery, vein) is True

    assert len(questions) == 1
    assert "case1" in questions[0]          # names the image being saved
    assert (tmp_path / "out" / "case1_veins.png").exists()


def test_declining_the_save_keeps_the_session_open(napari_stub, scene,
                                                   monkeypatch):
    tmp_path, artery, vein = scene
    questions = _patch_answers(monkeypatch, [False, True])

    def script(v):
        v.press("q")
        assert not v.closed                 # No -> still annotating
        v.press("q")

    FakeViewer.script = script
    assert _run_session(tmp_path, artery, vein) is True
    assert len(questions) == 2


def test_s_asks_before_skipping_and_yes_discards(napari_stub, scene,
                                                 monkeypatch):
    tmp_path, artery, vein = scene
    questions = _patch_answers(monkeypatch, [True])

    FakeViewer.script = lambda v: v.press("s")
    assert _run_session(tmp_path, artery, vein) is False

    assert len(questions) == 1
    assert "without saving" in questions[0].lower()
    assert not (tmp_path / "out" / "case1_veins.png").exists()


def test_declining_the_skip_discards_nothing(napari_stub, scene, monkeypatch):
    """A cancelled skip must leave the session in its normal state: the
    work is still there, and closing the window saves as usual."""
    tmp_path, artery, vein = scene
    _patch_answers(monkeypatch, [False])

    def script(v):
        v.press("s")
        assert not v.closed

    FakeViewer.script = script
    assert _run_session(tmp_path, artery, vein) is True
    assert (tmp_path / "out" / "case1_veins.png").exists()


def test_ask_confirm_defaults_to_yes_without_qt(napari_stub, scene):
    """Headless (no QtWidgets in the stub): behave exactly as before the
    dialogs existed, so scripted/CI runs never hang on a prompt."""
    import annotate

    assert annotate.ask_confirm(object(), "Save?") is True
