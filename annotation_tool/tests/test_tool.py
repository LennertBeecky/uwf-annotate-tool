"""Unit tests for the annotation tool that do NOT require napari/Qt.

Tests the pure-numpy utilities (loading, multiscale pyramid, skeletonise,
validation, image listing, time-log CSV, already-annotated check).

Run:
    pytest annotation_tool/tests/test_tool.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from utils import (  # type: ignore  # noqa: E402
    EDIT_LOG_FIELDS,
    AVSwapper,
    align_mask_to_image,
    already_annotated,
    apply_swap,
    load_mask_prefill,
    binarise_mask,
    branch_at,
    component_at,
    disk_at,
    nearest_foreground,
    owner_of_point,
    path_between,
    undo_swap,
    append_edit_log,
    append_time_log,
    build_multiscale_pyramid,
    list_images,
    load_image_rgb,
    neighbour_count_8,
    save_skeleton_png,
    skeleton_edit_distance,
    skeletonise_mask,
    validate_saved_skeleton,
)


# ---- load_image_rgb ----


def test_load_rgb(tmp_path: Path) -> None:
    p = tmp_path / "rgb.png"
    img = (np.random.default_rng(0).integers(0, 256, (64, 64, 3))).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(p)
    loaded = load_image_rgb(p)
    assert loaded.shape == (64, 64, 3)
    assert loaded.dtype == np.uint8


def test_load_rgba_drops_alpha(tmp_path: Path) -> None:
    p = tmp_path / "rgba.png"
    img = (np.random.default_rng(0).integers(0, 256, (32, 32, 4))).astype(np.uint8)
    Image.fromarray(img, mode="RGBA").save(p)
    loaded = load_image_rgb(p)
    assert loaded.shape == (32, 32, 3)


def test_load_grayscale_triplicates(tmp_path: Path) -> None:
    p = tmp_path / "gray.png"
    img = (np.random.default_rng(0).integers(0, 256, (32, 32))).astype(np.uint8)
    Image.fromarray(img, mode="L").save(p)
    loaded = load_image_rgb(p)
    assert loaded.shape == (32, 32, 3)
    # all three channels identical for grayscale
    assert np.array_equal(loaded[..., 0], loaded[..., 1])
    assert np.array_equal(loaded[..., 0], loaded[..., 2])


# ---- multiscale pyramid ----


def test_pyramid_small_image_single_level() -> None:
    small = np.zeros((1000, 1000, 3), dtype=np.uint8)
    levels, multiscale = build_multiscale_pyramid(small)
    assert len(levels) == 1
    assert multiscale is False


def test_pyramid_large_image_three_levels() -> None:
    big = np.zeros((4000, 3200, 3), dtype=np.uint8)
    levels, multiscale = build_multiscale_pyramid(big)
    assert len(levels) == 3
    assert multiscale is True
    assert levels[0].shape == (4000, 3200, 3)
    assert levels[1].shape == (2000, 1600, 3)   # half
    assert levels[2].shape == (1000, 800, 3)    # quarter


# ---- skeletonise ----


def test_skeletonise_empty_mask_returns_zeros() -> None:
    m = np.zeros((20, 20), dtype=np.uint8)
    out = skeletonise_mask(m)
    assert out.shape == (20, 20)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) == {0}


def test_skeletonise_thick_line_to_single_pixel_wide() -> None:
    # 3-pixel-thick horizontal line
    m = np.zeros((20, 30), dtype=np.uint8)
    m[9:12, 5:25] = 255
    out = skeletonise_mask(m)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) == {0, 255}
    # Skeletonize may trim 0-2 endpoint pixels; original line was 20 px long
    n = int((out > 0).sum())
    assert 17 <= n <= 20, f"expected ~20 skeleton pixels, got {n}"
    # Single row: neighbour count max should be 2 (linear line)
    nc = neighbour_count_8(out)
    assert int(nc.max()) <= 2


# ---- save + validation ----


def test_save_and_validate(tmp_path: Path) -> None:
    skel = np.zeros((40, 40), dtype=np.uint8)
    for i in range(5, 35):
        skel[20, i] = 255
    p = tmp_path / "out.png"
    save_skeleton_png(skel, p)
    res = validate_saved_skeleton(p, (40, 40))
    assert res.ok, f"validation failed: {res.messages}"


def test_validate_wrong_shape(tmp_path: Path) -> None:
    skel = np.zeros((40, 40), dtype=np.uint8); skel[20, 5:35] = 255
    p = tmp_path / "out.png"
    save_skeleton_png(skel, p)
    res = validate_saved_skeleton(p, (60, 60))   # wrong expected shape
    assert not res.ok
    assert any("shape" in m for m in res.messages)


def test_validate_detects_thick_blob(tmp_path: Path) -> None:
    blob = np.zeros((40, 40), dtype=np.uint8); blob[18:22, 18:22] = 255  # 4x4 solid
    p = tmp_path / "blob.png"
    save_skeleton_png(blob, p)
    res = validate_saved_skeleton(p, (40, 40))
    assert not res.ok
    assert any("1-pixel" in m for m in res.messages)


# ---- directory helpers ----


def test_list_images_directory(tmp_path: Path) -> None:
    (tmp_path / "a.png").touch()
    (tmp_path / "b.jpg").touch()
    (tmp_path / "c.txt").touch()
    (tmp_path / ".DS_Store").touch()
    imgs = list_images(tmp_path)
    names = [p.name for p in imgs]
    assert names == ["a.png", "b.jpg"]


def test_already_annotated(tmp_path: Path) -> None:
    assert not already_annotated("img01", tmp_path)
    (tmp_path / "img01_artery.png").touch()
    assert not already_annotated("img01", tmp_path)
    (tmp_path / "img01_veins.png").touch()
    assert already_annotated("img01", tmp_path)


def test_skeleton_edit_distance_both_empty() -> None:
    z = np.zeros((20, 20), dtype=np.uint8)
    m = skeleton_edit_distance(z, z)
    assert m == {"seed_px": 0, "final_px": 0, "kept_px": 0,
                 "added_px": 0, "removed_px": 0, "iou": 1.0}


def test_skeleton_edit_distance_identical() -> None:
    seed = np.zeros((20, 20), dtype=np.uint8); seed[5, 2:18] = 255
    m = skeleton_edit_distance(seed, seed.copy())
    assert m["seed_px"] == 16
    assert m["final_px"] == 16
    assert m["kept_px"] == 16
    assert m["added_px"] == 0
    assert m["removed_px"] == 0
    assert m["iou"] == 1.0


def test_skeleton_edit_distance_partial_overlap() -> None:
    # seed: row 5 cols 2..9 (8 px). final: row 5 cols 5..12 (8 px). overlap cols 5..9 (5 px).
    seed = np.zeros((20, 20), dtype=np.uint8); seed[5, 2:10] = 255
    final = np.zeros((20, 20), dtype=np.uint8); final[5, 5:13] = 255
    m = skeleton_edit_distance(seed, final)
    assert m["seed_px"] == 8
    assert m["final_px"] == 8
    assert m["kept_px"] == 5
    assert m["added_px"] == 3
    assert m["removed_px"] == 3
    # IoU = 5 / (8 + 8 - 5) = 5/11
    assert abs(m["iou"] - (5 / 11)) < 1e-3


def test_skeleton_edit_distance_seed_all_removed() -> None:
    seed = np.zeros((20, 20), dtype=np.uint8); seed[5, 2:10] = 255
    final = np.zeros((20, 20), dtype=np.uint8)
    m = skeleton_edit_distance(seed, final)
    assert m["removed_px"] == 8
    assert m["kept_px"] == 0
    assert m["added_px"] == 0
    assert m["iou"] == 0.0


def test_append_edit_log_writes_header_and_rows(tmp_path: Path) -> None:
    p = tmp_path / "annotation_edits.csv"
    append_edit_log(p, {
        "timestamp": "2026-04-21T12:00:00",
        "image_filename": "a.png",
        "duration_seconds": 120.0,
        "prefill_source": "lunet",
        "lunet_thresh": 0.5,
        "artery_seed_px": 100, "artery_final_px": 95, "artery_kept_px": 80,
        "artery_added_px": 15, "artery_removed_px": 20, "artery_iou": 0.72,
        "vein_seed_px": 110, "vein_final_px": 120, "vein_kept_px": 100,
        "vein_added_px": 20, "vein_removed_px": 10, "vein_iou": 0.78,
    })
    lines = p.read_text().strip().splitlines()
    assert lines[0].split(",") == EDIT_LOG_FIELDS
    assert "a.png" in lines[1]
    assert "lunet" in lines[1]


def test_append_time_log(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotation_times.csv"
    append_time_log(csv_path, {
        "timestamp": "2026-04-21T12:00:00",
        "image_filename": "a.png",
        "duration_seconds": 123.4,
        "artery_pixel_count": 500,
        "vein_pixel_count": 600,
    })
    append_time_log(csv_path, {
        "timestamp": "2026-04-21T12:15:00",
        "image_filename": "b.png",
        "duration_seconds": 200.0,
        "artery_pixel_count": 700,
        "vein_pixel_count": 800,
    })
    text = csv_path.read_text().strip().splitlines()
    assert text[0].startswith("timestamp,")
    assert len(text) == 3   # header + 2 rows
    assert "a.png" in text[1]
    assert "b.png" in text[2]


# ---- A/V reclassification ----


def _tree_mask() -> np.ndarray:
    """A trunk from (50,10) to (50,60) that forks into two daughters."""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[50, 10:61] = 1                       # trunk
    for i in range(30):                    # upper daughter
        m[50 - i, 60 + i] = 1
    for i in range(30):                    # lower daughter
        m[50 + i, 60 + i] = 1
    return m


def test_nearest_foreground_snaps_and_gives_up():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[10, 10] = 1
    assert nearest_foreground(m, (10, 10)) == (10, 10)
    assert nearest_foreground(m, (12, 10), radius=4) == (10, 10)
    assert nearest_foreground(m, (12, 10), radius=1) is None
    assert nearest_foreground(m, (-5, 3)) is None


def test_component_at_takes_whole_tree_not_the_other_one():
    m = _tree_mask()
    m[5, 5:40] = 1                          # unrelated separate vessel
    comp = component_at(m, (50, 30))
    assert comp is not None
    assert comp.sum() == _tree_mask().sum()
    assert not comp[5, 20]


def test_branch_at_selects_one_daughter_only():
    m = _tree_mask()
    upper = branch_at(m, (35, 75))          # on the upper daughter
    assert upper is not None
    assert upper[35, 75]
    assert not upper[65, 75]                # lower daughter untouched
    assert not upper[50, 20]                # trunk untouched
    # and it is a strict subset of the component
    assert upper.sum() < component_at(m, (35, 75)).sum()


def test_branch_at_reclaims_painted_width():
    """A thick (brush-painted) branch moves entirely, not just its skeleton."""
    m = np.zeros((60, 60), dtype=np.uint8)
    m[28:33, 5:55] = 1                      # 5-px-wide bar
    region = branch_at(m, (30, 30))
    assert region is not None
    assert region[28, 30] and region[32, 30]


def test_branch_at_falls_back_to_component_without_junctions():
    m = np.zeros((40, 40), dtype=np.uint8)
    m[20, 5:35] = 1
    region = branch_at(m, (20, 20))
    assert region is not None and region.sum() == m.sum()


def test_disk_at_is_clipped_to_bounds():
    d = disk_at((30, 30), (1, 1), 5)
    assert d[0, 0] and d[1, 5] and not d[10, 10]
    assert d.shape == (30, 30)


def test_apply_and_undo_swap_roundtrip_preserves_crossings():
    src = np.zeros((10, 10), dtype=np.uint8)
    dst = np.zeros((10, 10), dtype=np.uint8)
    src[5, 2:8] = 1
    dst[5, 5] = 1                           # crossing: already in dst
    src0, dst0 = src.copy(), dst.copy()

    region = np.zeros((10, 10), dtype=bool)
    region[5, 2:8] = True
    rec = apply_swap(src, dst, region)

    assert rec["n"] == 6
    assert src.sum() == 0
    assert dst[5, 2:8].all()

    undo_swap(src, dst, rec)
    assert np.array_equal(src, src0)
    assert np.array_equal(dst, dst0)        # the crossing pixel survived


def test_owner_of_point_picks_the_nearer_class():
    art = np.zeros((40, 40), dtype=np.uint8)
    vein = np.zeros((40, 40), dtype=np.uint8)
    art[10, 5:35] = 1
    vein[30, 5:35] = 1
    masks = {"artery": art, "veins": vein}
    assert owner_of_point(masks, (12, 20)) == "artery"
    assert owner_of_point(masks, (28, 20)) == "veins"
    assert owner_of_point(masks, (20, 20)) is None       # both out of range
    assert owner_of_point(masks, (20, 20), radius=20) in ("artery", "veins")


# ---- AVSwapper: the interaction state machine ----
#
# Written before the implementation. This is the object annotate.py drives
# from its keybindings, so the swap/undo semantics are tested without napari.


def _av_pair():
    """Artery bar at row 10, vein tree at row 30 forking at column 60."""
    art = np.zeros((100, 100), dtype=np.uint8)
    vein = np.zeros((100, 100), dtype=np.uint8)
    art[10, 5:95] = 1
    vein[30, 10:61] = 1
    for i in range(25):
        vein[30 - i, 60 + i] = 1
        vein[30 + i, 60 + i] = 1
    return art, vein


def test_swapper_moves_branch_to_the_other_class():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    res = sw.swap_at((20, 70), mode="branch")     # upper vein daughter
    assert res is not None
    assert res["src"] == "veins" and res["dst"] == "artery"
    assert res["n"] > 0
    assert art[20, 70] == 1 and vein[20, 70] == 0
    assert vein[30, 20] == 1                       # vein trunk untouched
    assert art[10, 50] == 1                        # artery bar untouched


def test_swapper_direction_follows_whichever_class_owns_the_pixel():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    res = sw.swap_at((10, 50), mode="component")
    assert res["src"] == "artery" and res["dst"] == "veins"
    assert art.sum() == 0
    assert vein[10, 50] == 1


def test_swapper_returns_none_when_nothing_is_near():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    assert sw.swap_at((80, 20), mode="branch") is None
    assert not sw.can_undo


def test_swapper_undo_restores_exactly_including_crossings():
    art, vein = _av_pair()
    art[30, 40] = 1                                # crossing: both classes
    art0, vein0 = art.copy(), vein.copy()
    sw = AVSwapper({"artery": art, "veins": vein})
    assert sw.swap_at((30, 40), mode="component") is not None
    assert sw.undo() is not None
    assert np.array_equal(art, art0)
    assert np.array_equal(vein, vein0)


def test_swapper_undo_is_lifo_and_bottoms_out():
    art, vein = _av_pair()
    art0, vein0 = art.copy(), vein.copy()
    sw = AVSwapper({"artery": art, "veins": vein})
    sw.swap_at((10, 50), mode="component")
    sw.swap_at((30, 20), mode="component")
    sw.undo()
    sw.undo()
    assert np.array_equal(art, art0) and np.array_equal(vein, vein0)
    assert sw.undo() is None


def test_swapper_disk_mode_is_bounded_by_its_radius():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    sw.swap_at((10, 50), mode="disk", radius=5)
    assert vein[10, 50] == 1 and vein[10, 47] == 1
    assert art[10, 50] == 0
    assert art[10, 30] == 1                        # far end of the bar stays


def test_swapper_swap_all_exchanges_in_place():
    """Arrays must keep their identity — napari layers hold these objects."""
    art, vein = _av_pair()
    art0, vein0 = art.copy(), vein.copy()
    sw = AVSwapper({"artery": art, "veins": vein})
    res = sw.swap_all()
    assert res["kind"] == "global"
    assert np.array_equal(art, vein0) and np.array_equal(vein, art0)
    sw.undo()
    assert np.array_equal(art, art0) and np.array_equal(vein, vein0)


def test_swapper_rejects_an_unknown_mode():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    with pytest.raises(ValueError):
        sw.swap_at((10, 50), mode="lasso")


# ---- partial-segment (range) swaps ----
#
# A mislabelled *piece* of a vessel is common: the model flips class
# part-way along a branch. Swapping the whole junction-to-junction segment
# over-corrects, so the annotator marks the two ends of the bad stretch.


def test_path_between_covers_only_the_stretch_between_the_two_points():
    m = np.zeros((60, 100), dtype=np.uint8)
    m[30, 5:95] = 1
    region = path_between(m, (30, 30), (30, 60))
    assert region is not None
    assert region[30, 30] and region[30, 45] and region[30, 60]
    assert not region[30, 20]            # before the first mark
    assert not region[30, 75]            # past the second mark


def test_path_between_snaps_both_ends_off_the_centreline():
    m = np.zeros((60, 100), dtype=np.uint8)
    m[30, 5:95] = 1
    region = path_between(m, (33, 30), (27, 60))   # cursor a few px off
    assert region is not None and region[30, 45]


def test_path_between_runs_through_a_junction():
    """Two points on different daughters: the path crosses the fork."""
    m = _tree_mask()                      # trunk row 50, fork at col 60
    upper, lower = (35, 75), (65, 75)
    region = path_between(m, upper, lower)
    assert region is not None
    assert region[35, 75] and region[65, 75]
    assert region[50, 60]                 # the fork itself
    assert not region[50, 20]             # not back down the trunk


def test_path_between_reclaims_painted_width():
    m = np.zeros((60, 100), dtype=np.uint8)
    m[28:33, 5:95] = 1                    # 5-px-wide bar
    region = path_between(m, (30, 30), (30, 60))
    assert region is not None
    assert region[28, 45] and region[32, 45]
    assert not region[30, 80]


def test_path_between_refuses_two_different_vessels():
    m = np.zeros((60, 100), dtype=np.uint8)
    m[10, 5:95] = 1
    m[50, 5:95] = 1                       # separate, unconnected vessel
    assert path_between(m, (10, 30), (50, 60)) is None


def test_path_between_returns_none_when_an_end_is_off_vessel():
    m = np.zeros((60, 100), dtype=np.uint8)
    m[30, 5:95] = 1
    assert path_between(m, (30, 30), (5, 5)) is None


def test_swapper_range_mode_moves_only_the_marked_stretch():
    art = np.zeros((60, 100), dtype=np.uint8)
    vein = np.zeros((60, 100), dtype=np.uint8)
    art[30, 5:95] = 1
    sw = AVSwapper({"artery": art, "veins": vein})
    res = sw.swap_between((30, 30), (30, 60))
    assert res is not None
    assert res["src"] == "artery" and res["dst"] == "veins"
    assert vein[30, 45] == 1 and art[30, 45] == 0
    assert art[30, 20] == 1 and art[30, 80] == 1     # both ends keep their class


def test_swapper_range_swap_is_undoable():
    art = np.zeros((60, 100), dtype=np.uint8)
    vein = np.zeros((60, 100), dtype=np.uint8)
    art[30, 5:95] = 1
    art0, vein0 = art.copy(), vein.copy()
    sw = AVSwapper({"artery": art, "veins": vein})
    sw.swap_between((30, 30), (30, 60))
    sw.undo()
    assert np.array_equal(art, art0) and np.array_equal(vein, vein0)


def test_swapper_range_swap_returns_none_across_two_vessels():
    art = np.zeros((60, 100), dtype=np.uint8)
    vein = np.zeros((60, 100), dtype=np.uint8)
    art[10, 5:95] = 1
    art[50, 5:95] = 1
    sw = AVSwapper({"artery": art, "veins": vein})
    assert sw.swap_between((10, 30), (50, 60)) is None
    assert not sw.can_undo


def test_swapper_owner_at_reports_the_class_under_a_point():
    art, vein = _av_pair()
    sw = AVSwapper({"artery": art, "veins": vein})
    assert sw.owner_at((10, 50)) == "artery"
    assert sw.owner_at((30, 20)) == "veins"
    assert sw.owner_at((55, 5)) is None


# ---- boundary / width annotation ----
#
# For DVA (and later UWF) the painted mask IS the annotation: its thickness
# is the vessel width, so it must survive the save un-thinned, and the
# annotator needs to be able to adjust it.


def _bar(width_px: int, shape=(60, 100)) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    half = width_px // 2
    m[30 - half:30 + half + 1, 5:95] = 1
    return m


def test_binarise_mask_preserves_width():
    m = _bar(5)
    out = binarise_mask(m)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert int((out > 0).sum()) == int((m > 0).sum())   # nothing thinned away


def test_validate_saved_mask_accepts_a_thick_vessel(tmp_path):
    p = tmp_path / "a.png"
    save_skeleton_png(binarise_mask(_bar(9)), p)
    thin = validate_saved_skeleton(p, (60, 100))
    thick = validate_saved_skeleton(p, (60, 100), check_thin=False)
    assert not thin.ok and any("1-pixel" in m for m in thin.messages)
    assert thick.ok and not thick.messages


def _widths(mask, col=50):
    return int((mask[:, col] > 0).sum())


def test_resize_at_widens_the_segment_under_the_cursor():
    art, vein = _bar(3), np.zeros((60, 100), dtype=np.uint8)
    sw = AVSwapper({"artery": art, "veins": vein})
    rec = sw.resize_at((30, 50), +1)
    assert rec is not None and rec["kind"] == "widen"
    assert _widths(art) == 5                      # one px added each side
    assert vein.sum() == 0                        # other class untouched


def test_resize_at_narrows_the_segment_under_the_cursor():
    art, vein = _bar(5), np.zeros((60, 100), dtype=np.uint8)
    sw = AVSwapper({"artery": art, "veins": vein})
    rec = sw.resize_at((30, 50), -1)
    assert rec is not None and rec["kind"] == "narrow"
    assert _widths(art) == 3


def test_narrowing_can_never_erase_the_vessel():
    """Repeated narrowing bottoms out at the centreline, not at nothing."""
    art, vein = _bar(3), np.zeros((60, 100), dtype=np.uint8)
    sw = AVSwapper({"artery": art, "veins": vein})
    for _ in range(6):
        sw.resize_at((30, 50), -1)
    assert art[30, 50] == 1
    assert _widths(art) >= 1


def test_narrowing_keeps_the_vessel_connected_at_a_junction():
    from scipy import ndimage
    m = np.zeros((100, 100), dtype=np.uint8)
    m[48:53, 10:61] = 1                            # thick trunk
    for i in range(30):                            # two thick daughters
        m[48 - i:53 - i, 60 + i] = 1
        m[48 + i:53 + i, 60 + i] = 1
    sw = AVSwapper({"artery": m, "veins": np.zeros_like(m)})
    before = ndimage.label(m, structure=np.ones((3, 3)))[1]
    sw.resize_at((50, 30), -1)                     # narrow the trunk
    after = ndimage.label(m, structure=np.ones((3, 3)))[1]
    assert after == before                         # no vessel split off


def test_resize_at_off_vessel_returns_none():
    art, vein = _bar(3), np.zeros((60, 100), dtype=np.uint8)
    sw = AVSwapper({"artery": art, "veins": vein})
    assert sw.resize_at((5, 5), +1) is None
    assert not sw.can_undo


def test_resize_is_undoable_both_directions():
    art, vein = _bar(5), np.zeros((60, 100), dtype=np.uint8)
    art0 = art.copy()
    sw = AVSwapper({"artery": art, "veins": vein})
    sw.resize_at((30, 50), +1)
    sw.resize_at((30, 50), -1)
    sw.undo()
    sw.undo()
    assert np.array_equal(art, art0)


def test_widening_may_overlap_the_other_class_at_a_crossing():
    art = _bar(3)
    vein = np.zeros((60, 100), dtype=np.uint8)
    vein[20:40, 50] = 1                            # a vein crossing the artery
    sw = AVSwapper({"artery": art, "veins": vein})
    sw.resize_at((30, 20), +1)
    assert art[28, 50] == 1 and vein[28, 50] == 1  # both classes keep the pixel


# ---- mask prefill (DVA layout) ----


def test_align_mask_same_shape_is_unchanged():
    m = np.zeros((60, 80), dtype=np.uint8)
    m[10, 10] = 1
    out = align_mask_to_image(m, (60, 80))
    assert np.array_equal(out, m)


def test_align_mask_centre_crops_a_padded_mask():
    """DVA_maastricht: 720x720 masks against 576x720 frames."""
    m = np.zeros((720, 720), dtype=np.uint8)
    m[72:648, :] = 1                       # the real frame inside the padding
    out = align_mask_to_image(m, (576, 720))
    assert out.shape == (576, 720)
    assert out.all()                       # exactly the un-padded content


def test_align_mask_centre_pads_a_smaller_mask():
    m = np.ones((576, 720), dtype=np.uint8)
    out = align_mask_to_image(m, (720, 720))
    assert out.shape == (720, 720)
    assert out[0, 0] == 0 and out[360, 360] == 1
    assert int(out.sum()) == 576 * 720


def test_align_mask_rejects_a_mismatched_width():
    m = np.ones((576, 500), dtype=np.uint8)
    with pytest.raises(ValueError):
        align_mask_to_image(m, (576, 720))


def test_mask_prefill_reads_the_dva_subdirectory_layout(tmp_path):
    for cls, row in (("artery", 10), ("veins", 20)):
        d = tmp_path / cls
        d.mkdir()
        arr = np.zeros((60, 80), dtype=np.uint8)
        arr[row, 5:70] = 255
        save_skeleton_png(arr, d / "case1.png")
    art, vein = load_mask_prefill(tmp_path, "case1", (60, 80))
    assert art[10, 30] == 1 and art.sum() == 65
    assert vein[20, 30] == 1


def test_mask_prefill_reads_the_flat_layout(tmp_path):
    """Our own output layout, so a saved annotation can be reopened."""
    for cls, row in (("artery", 10), ("veins", 20)):
        arr = np.zeros((60, 80), dtype=np.uint8)
        arr[row, 5:70] = 255
        save_skeleton_png(arr, tmp_path / f"case1_{cls}.png")
    art, vein = load_mask_prefill(tmp_path, "case1", (60, 80))
    assert art[10, 30] == 1 and vein[20, 30] == 1


def test_mask_prefill_returns_none_when_absent(tmp_path):
    assert load_mask_prefill(tmp_path, "nope", (60, 80)) is None


def test_mask_prefill_aligns_a_padded_mask(tmp_path):
    d = tmp_path / "artery"
    d.mkdir()
    arr = np.zeros((720, 720), dtype=np.uint8)
    arr[72:648, :] = 255
    save_skeleton_png(arr, d / "case1.png")
    (tmp_path / "veins").mkdir()
    save_skeleton_png(np.zeros((720, 720), dtype=np.uint8),
                      tmp_path / "veins" / "case1.png")
    art, vein = load_mask_prefill(tmp_path, "case1", (576, 720))
    assert art.shape == (576, 720) and art.all()


def test_undo_is_the_exact_revert_for_a_width_edit():
    """Dilate-then-erode is a morphological closing, not an inverse, so
    `[u]` — not the opposite key — is what exactly reverts a width edit."""
    art = np.zeros((60, 100), dtype=np.uint8)
    art[28:33, 5:95] = 1
    art[28, 40:45] = 0                      # a notch on the vessel edge
    art0 = art.copy()
    sw = AVSwapper({"artery": art, "veins": np.zeros((60, 100), dtype=np.uint8)})
    sw.resize_at((30, 50), +1)
    sw.resize_at((30, 50), -1)
    sw.undo(); sw.undo()
    assert np.array_equal(art, art0)


def test_repeated_widening_keeps_acting_on_the_same_segment():
    """Two presses of [.] must do the same thing twice.

    Dilating a vessel can make it touch its neighbour, which changes where
    the junctions are and hence which pixels a freshly-computed segment
    covers. Without an anchor the second press would suddenly grow the
    merged neighbour too.
    """
    art = np.zeros((60, 100), dtype=np.uint8)
    art[30, 5:95] = 1                        # the vessel being widened
    art[34, 5:95] = 1                        # a neighbour 3 px away
    sw = AVSwapper({"artery": art, "veins": np.zeros((60, 100), dtype=np.uint8)})

    first = sw.resize_at((30, 50), +1)
    second = sw.resize_at((30, 50), +1)
    assert first is not None and second is not None
    # Both presses grow the same vessel by one ring: comparable pixel counts.
    assert second["n"] < first["n"] * 2
    # The neighbour keeps its own width; it was not swept into the segment.
    assert art[36, 50] == 0
