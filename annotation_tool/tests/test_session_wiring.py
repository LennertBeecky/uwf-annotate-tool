"""End-to-end test of the annotation session with napari stubbed out.

Exercises the real `_open_annotation_session` — layer setup, keybinding
registration, a simulated [f] keypress on a mislabelled vessel, [u] undo,
and the save path — so wiring mistakes inside the napari closure are caught
on machines without napari/Qt installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


# ---- napari / qtpy stubs ----


class FakeEvent:
    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def emit(self, **kw):
        for cb in list(self._cbs):
            cb(kw)


class FakeEvents:
    def __init__(self):
        self.paint = FakeEvent()


class FakeLayer:
    def __init__(self, data, name):
        self.events = FakeEvents()
        self._history: list = []
        self.data = np.asarray(data)
        self.name = name
        self.mode = "paint"
        self.brush_size = 2
        self.selected_label = 1
        self.colormap = None
        self.editable = True
        self.keymap: dict = {}
        self.refreshes = 0

    def bind_key(self, key, func=None, overwrite=False):
        if func is None:
            def deco(f):
                self.keymap[key] = f
                return f
            return deco
        self.keymap[key] = func
        return func

    def refresh(self):
        self.refreshes += 1

    def world_to_data(self, position):
        return tuple(position)

    # -- napari Labels API used by the session --
    def paint_stroke(self, yx, value=1):
        """Simulate a brush stroke: records history and emits `paint`."""
        y, x = int(yx[0]), int(yx[1])
        self._history.append(self.data.copy())
        self.data[y - 1:y + 2, x - 1:x + 2] = value
        self.events.paint.emit()

    def undo(self):
        if self._history:
            self.data[...] = self._history.pop()


class FakeSelection:
    active = None


class FakeLayers(list):
    def __init__(self):
        super().__init__()
        self.selection = FakeSelection()


class FakeViewer:
    #  napari.run() hands control to this callback so a test can "use" the GUI
    script = None

    def __init__(self, title=""):
        self.title = title
        self.layers = FakeLayers()
        self.keymap: dict = {}
        self.mouse_drag_callbacks: list = []
        self.text_overlay = types.SimpleNamespace(
            text="", visible=False, font_size=12, color="white")
        self.cursor = types.SimpleNamespace(position=(0.0, 0.0))
        self.closed = False

    def add_image(self, data, multiscale=False, name=""):
        layer = FakeLayer(data[0] if multiscale else data, name)
        self.layers.append(layer)
        return layer

    def add_labels(self, data, name=""):
        layer = FakeLayer(data, name)
        self.layers.append(layer)
        return layer

    def bind_key(self, key, func=None, overwrite=False):
        if func is None:
            def deco(f):
                self.keymap[key] = f
                return f
            return deco
        self.keymap[key] = func
        return func

    def close(self):
        self.closed = True

    # -- test helpers --
    def press(self, key, at=None):
        if at is not None:
            self.cursor.position = at
        self.keymap[key](self)

    def layer(self, name):
        return next(x for x in self.layers if x.name == name)


@pytest.fixture
def napari_stub(monkeypatch):
    viewer_box = {}

    def _make_viewer(*a, **kw):
        v = FakeViewer(*a, **kw)
        viewer_box["viewer"] = v
        return v

    fake = types.ModuleType("napari")
    fake.Viewer = _make_viewer
    fake.run = lambda: (FakeViewer.script or (lambda _v: None))(
        viewer_box["viewer"])
    monkeypatch.setitem(sys.modules, "napari", fake)

    qtpy = types.ModuleType("qtpy")
    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.Qt = object()
    qtpy.QtCore = qtcore
    monkeypatch.setitem(sys.modules, "qtpy", qtpy)
    monkeypatch.setitem(sys.modules, "qtpy.QtCore", qtcore)

    yield viewer_box
    FakeViewer.script = None


@pytest.fixture
def scene(tmp_path):
    """An image plus a vein tree that the model mislabelled as artery."""
    Image.fromarray(np.full((100, 100, 3), 40, dtype=np.uint8)).save(
        tmp_path / "case1.png")
    artery = np.zeros((100, 100), dtype=np.uint8)
    vein = np.zeros((100, 100), dtype=np.uint8)
    artery[30, 10:61] = 1                       # trunk, correctly artery
    for i in range(25):                         # daughter that is really a vein
        artery[30 - i, 60 + i] = 1
        artery[30 + i, 60 + i] = 1
    vein[70, 10:90] = 1
    return tmp_path, artery, vein


def _run_session(tmp_path, artery, vein, **kw):
    import annotate  # imported late: needs the napari stub in place

    return annotate._open_annotation_session(
        tmp_path / "case1.png", tmp_path / "out", overwrite=True,
        prefill_source="predictions", prefill_masks=(artery, vein), **kw,
    )


def test_swap_keys_are_bound_on_viewer_and_both_label_layers(napari_stub, scene):
    from annotate import SWAP_KEYS  # noqa: PLC0415

    tmp_path, artery, vein = scene
    FakeViewer.script = lambda v: None
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    for key, _mode in SWAP_KEYS:
        assert key in v.keymap, f"{key} not bound on viewer"
        assert key in v.layer("artery").keymap, f"{key} not bound on artery"
        assert key in v.layer("veins").keymap, f"{key} not bound on veins"


def test_w_moves_the_mislabelled_branch_and_the_save_reflects_it(
        napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("w", at=(20, 70))               # hover the upper daughter

    FakeViewer.script = script
    assert _run_session(tmp_path, artery, vein) is True

    v = napari_stub["viewer"]
    assert v.layer("artery").data[20, 70] == 0
    assert v.layer("veins").data[20, 70] == 1
    assert v.layer("artery").data[30, 20] == 1  # trunk kept its class
    assert "veins" in v.text_overlay.text or "artery" in v.text_overlay.text

    saved_vein = np.asarray(Image.open(tmp_path / "out" / "case1_veins.png"))
    assert saved_vein[20, 70] > 0


def test_u_undoes_the_swap(napari_stub, scene):
    tmp_path, artery, vein = scene
    before = artery.copy()

    def script(v):
        v.press("w", at=(20, 70))
        v.press("u")

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, before)


def test_pressing_w_over_empty_background_is_a_no_op(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("w", at=(90, 5))

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, artery)
    assert "no vessel" in v.text_overlay.text


def test_shift_x_swaps_both_layers_wholesale(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("Shift-X")

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, vein)
    assert np.array_equal(v.layer("veins").data, artery)


# ---- [h]: partial-segment (range) swap ----
#
# Two presses: the first marks one end of the mislabelled stretch, the
# second marks the other and performs the swap.


def test_h_twice_moves_only_the_stretch_between_the_marks(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("h", at=(70, 30))            # first mark on the vein bar
        v.press("h", at=(70, 60))            # second mark further along

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert v.layer("artery").data[70, 45] == 1      # stretch moved to artery
    assert v.layer("veins").data[70, 45] == 0
    assert v.layer("veins").data[70, 15] == 1       # outside the marks: unchanged
    assert v.layer("veins").data[70, 85] == 1


def test_first_h_only_arms_the_range_and_changes_nothing(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("h", at=(70, 30))

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("veins").data, vein)
    assert "second" in v.text_overlay.text.lower() or \
           "end" in v.text_overlay.text.lower()


def test_h_across_two_different_vessels_is_refused(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("h", at=(70, 30))            # on the vein bar
        v.press("h", at=(30, 20))            # on the artery trunk: other vessel

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("veins").data, vein)
    assert np.array_equal(v.layer("artery").data, artery)
    assert "same vessel" in v.text_overlay.text.lower()


def test_u_undoes_a_range_swap(napari_stub, scene):
    tmp_path, artery, vein = scene

    def script(v):
        v.press("h", at=(70, 30))
        v.press("h", at=(70, 60))
        v.press("u")

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("veins").data, vein)


# ---- boundary / width mode ----
#
# For DVA the painted mask IS the annotation: thickness is vessel width, so
# it must reach disk un-thinned and the annotator must be able to adjust it.


@pytest.fixture
def thick_scene(tmp_path):
    """An image with a 5-px-wide artery and a 5-px-wide vein."""
    Image.fromarray(np.full((100, 100, 3), 40, dtype=np.uint8)).save(
        tmp_path / "case1.png")
    artery = np.zeros((100, 100), dtype=np.uint8)
    vein = np.zeros((100, 100), dtype=np.uint8)
    artery[28:33, 10:90] = 1
    vein[68:73, 10:90] = 1
    return tmp_path, artery, vein


def _run_boundary_session(tmp_path, artery, vein):
    """Boundary output is the default now; this is just the explicit form."""
    return _run_session(tmp_path, artery, vein, boundaries=True)


def test_filled_masks_are_the_default_output(napari_stub, thick_scene):
    """Width is the annotation by default; skeletons are now opt-in."""
    tmp_path, artery, vein = thick_scene
    FakeViewer.script = lambda v: None
    assert _run_session(tmp_path, artery, vein) is True

    saved = np.asarray(Image.open(tmp_path / "out" / "case1_artery.png"))
    assert set(np.unique(saved).tolist()) <= {0, 255}
    assert int((saved > 0).sum()) == int(artery.sum())      # nothing thinned
    assert int((saved[:, 50] > 0).sum()) == 5               # width survived


def test_skeleton_mode_thins_when_asked(napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene
    FakeViewer.script = lambda v: None
    _run_session(tmp_path, artery, vein, boundaries=False)
    saved = np.asarray(Image.open(tmp_path / "out" / "case1_artery.png"))
    assert int((saved > 0).sum()) < int(artery.sum()) / 3
    assert int((saved[:, 50] > 0).sum()) == 1


def test_dot_widens_and_comma_narrows_the_vessel_under_the_cursor(
        napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.press(".", at=(30, 50))
        v.press(".", at=(30, 50))

    FakeViewer.script = script
    _run_boundary_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert int((v.layer("artery").data[:, 50] > 0).sum()) == 9   # 5 + 2 + 2
    assert int((v.layer("veins").data[:, 50] > 0).sum()) == 5    # untouched


def test_comma_narrows_and_u_undoes_it(napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.press(",", at=(30, 50))
        assert int((v.layer("artery").data[:, 50] > 0).sum()) == 3
        v.press("u")

    FakeViewer.script = script
    _run_boundary_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, artery)


def test_boundary_mode_does_not_warn_about_thickness(napari_stub, thick_scene,
                                                     capsys):
    tmp_path, artery, vein = thick_scene
    FakeViewer.script = lambda v: None
    _run_boundary_session(tmp_path, artery, vein)
    out = capsys.readouterr().out
    assert "1-pixel wide" not in out
    assert "boundaries" in out.lower() or "width" in out.lower()


# ---- undo: one timeline for brush strokes and for our edits ----


def test_cmd_z_undoes_a_brush_stroke(napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.layer("artery").paint_stroke((10, 10))
        assert v.layer("artery").data[10, 10] == 1
        v.press("Meta-Z")

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert v.layer("artery").data[10, 10] == 0


def test_u_and_cmd_z_are_the_same_undo(napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.press(".", at=(30, 50))
        v.press("Meta-Z")                    # cmd+z undoes our width edit too

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, artery)


def test_undo_walks_back_through_both_kinds_in_order(napari_stub, thick_scene):
    """paint, then swap, then two undos must restore both — in order.

    Undoing the swap first and the stroke second is what stops a swapped
    brush stroke leaving orphan pixels behind on the destination layer.
    """
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.layer("artery").paint_stroke((50, 50))     # isolated new blob
        v.press("w", at=(50, 50))                    # move it to the veins
        assert v.layer("veins").data[50, 50] == 1
        v.press("u")                                 # undo the swap
        assert v.layer("artery").data[50, 50] == 1
        assert v.layer("veins").data[50, 50] == 0
        v.press("u")                                 # undo the stroke

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, artery)
    assert np.array_equal(v.layer("veins").data, vein)


def test_ctrl_z_works_too(napari_stub, thick_scene):
    tmp_path, artery, vein = thick_scene

    def script(v):
        v.press(",", at=(30, 50))
        v.press("Control-Z")

    FakeViewer.script = script
    _run_session(tmp_path, artery, vein)
    v = napari_stub["viewer"]
    assert np.array_equal(v.layer("artery").data, artery)


# ---- walking a dataset one image at a time ----


@pytest.fixture
def dataset(tmp_path):
    """Three frames plus a directory of filled A/V masks, DVA-style."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    (mask_dir / "artery").mkdir(parents=True)
    (mask_dir / "veins").mkdir(parents=True)
    stems = ["f001", "f002", "f003"]
    for i, stem in enumerate(stems):
        Image.fromarray(np.full((80, 80, 3), 30 + i, dtype=np.uint8)).save(
            img_dir / f"{stem}.png")
        art = np.zeros((80, 80), dtype=np.uint8)
        vein = np.zeros((80, 80), dtype=np.uint8)
        art[20:25, 10:70] = 255                  # 5-px artery
        vein[50:57, 10:70] = 255                 # 7-px vein
        Image.fromarray(art, mode="L").save(mask_dir / "artery" / f"{stem}.png")
        Image.fromarray(vein, mode="L").save(mask_dir / "veins" / f"{stem}.png")
    return tmp_path, img_dir, mask_dir, stems


def _walk(tmp_path, img_dir, mask_dir, monkeypatch, **kw):
    import annotate

    monkeypatch.setattr("builtins.input", lambda *_a: "")   # "next image"
    annotate._walk_directory(
        img_dir, tmp_path / "out", overwrite=True,
        prefill_source="masks", masks_dir=mask_dir, **kw,
    )


def test_walks_every_image_and_saves_filled_a_v_masks(
        napari_stub, dataset, monkeypatch):
    tmp_path, img_dir, mask_dir, stems = dataset
    FakeViewer.script = lambda v: None
    _walk(tmp_path, img_dir, mask_dir, monkeypatch)

    for stem in stems:
        art = np.asarray(Image.open(tmp_path / "out" / f"{stem}_artery.png"))
        vein = np.asarray(Image.open(tmp_path / "out" / f"{stem}_veins.png"))
        assert art.shape == (80, 80) and vein.shape == (80, 80)
        assert set(np.unique(art).tolist()) <= {0, 255}
        # widths survive the round trip, and the two classes stay distinct
        assert int((art[:, 40] > 0).sum()) == 5
        assert int((vein[:, 40] > 0).sum()) == 7
        assert not (art & vein).any()


def test_already_annotated_images_are_skipped_on_a_second_pass(
        napari_stub, dataset, monkeypatch, capsys):
    tmp_path, img_dir, mask_dir, stems = dataset
    FakeViewer.script = lambda v: None
    _walk(tmp_path, img_dir, mask_dir, monkeypatch)
    capsys.readouterr()

    import annotate
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    annotate._walk_directory(img_dir, tmp_path / "out", overwrite=False,
                             prefill_source="masks", masks_dir=mask_dir)
    out = capsys.readouterr().out
    assert out.count("already annotated") == len(stems)


def test_edits_during_the_walk_reach_the_saved_masks(
        napari_stub, dataset, monkeypatch):
    """A swap and a width edit on each frame must both survive to disk."""
    tmp_path, img_dir, mask_dir, stems = dataset

    def script(v):
        v.press("w", at=(22, 40))            # artery bar -> vein
        v.press(".", at=(53, 40))            # widen the original vein

    FakeViewer.script = script
    _walk(tmp_path, img_dir, mask_dir, monkeypatch)

    for stem in stems:
        art = np.asarray(Image.open(tmp_path / "out" / f"{stem}_artery.png"))
        vein = np.asarray(Image.open(tmp_path / "out" / f"{stem}_veins.png"))
        assert art[22, 40] == 0              # the swap happened
        assert vein[22, 40] > 0
        assert int((vein[45:65, 40] > 0).sum()) == 9   # 7 + 1 each side
