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
    already_annotated,
    append_time_log,
    build_multiscale_pyramid,
    list_images,
    load_image_rgb,
    neighbour_count_8,
    save_skeleton_png,
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
