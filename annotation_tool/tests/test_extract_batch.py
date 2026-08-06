"""Tests for unpacking a batch zip into the clinician folder layout.

Written before the implementation. Extraction used to live in shell — one
version in bash, another in cmd, each with its own quoting and its own idea
of how deep to look for `images/`. The Windows one copied nothing and said
nothing when a zip arrived with a wrapper folder, which is how an annotator
ended up staring at empty batch folders. It lives in Python now so it can
be tested on any machine.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from extract_batch import BatchLayoutError, extract_batch  # type: ignore  # noqa: E402


def _png(path: Path, value: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(path)


def _make_zip(zip_path: Path, files: dict[str, bytes | None]) -> Path:
    """Build a zip from {archive_name: content}. None means a real PNG."""
    staging = zip_path.parent / "_staging"
    for name, content in files.items():
        target = staging / name
        if content is None:
            _png(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())
    return zip_path


@pytest.fixture
def install(tmp_path):
    root = tmp_path / "uwf-annotate"
    (root / "clinician_data" / "incoming").mkdir(parents=True)
    return root


def _flat_batch() -> dict:
    return {
        "images/000003-frame-125.png": None,
        "images/000004-frame-125.png": None,
        "predictions/000003-frame-125_artery.png": None,
        "predictions/000003-frame-125_veins.png": None,
        "predictions/000004-frame-125_artery.png": None,
        "predictions/000004-frame-125_veins.png": None,
        "README.txt": b"batch notes",
    }


def test_flat_zip_lands_in_the_right_folders(install, tmp_path):
    zip_path = _make_zip(tmp_path / "batch_dva_test_2026-08-06.zip", _flat_batch())
    result = extract_batch(zip_path, install)

    assert result.batch_name == "batch_dva_test_2026-08-06"
    assert result.n_images == 2
    assert result.n_prefill == 4
    assert (install / "clinician_data/images_to_annotate/batch_dva_test_2026-08-06"
            / "000003-frame-125.png").exists()
    assert (install / "clinician_data/predictions/batch_dva_test_2026-08-06"
            / "000003-frame-125_artery.png").exists()


def test_wrapper_folder_is_seen_through(install, tmp_path):
    """A zip re-compressed in transit gains a folder level. This is the
    case that silently produced empty batch folders on Windows."""
    wrapped = {f"batch_dva_test_2026-08-06/{k}": v
               for k, v in _flat_batch().items()}
    zip_path = _make_zip(tmp_path / "batch_dva_test_2026-08-06.zip", wrapped)
    result = extract_batch(zip_path, install)
    assert result.n_images == 2 and result.n_prefill == 4


def test_two_wrapper_folders_are_seen_through(install, tmp_path):
    wrapped = {f"outer/inner/{k}": v for k, v in _flat_batch().items()}
    zip_path = _make_zip(tmp_path / "batch_x.zip", wrapped)
    result = extract_batch(zip_path, install)
    assert result.n_images == 2 and result.n_prefill == 4


def test_macos_zip_artefacts_are_ignored(install, tmp_path):
    noisy = dict(_flat_batch())
    noisy["__MACOSX/images/._000003-frame-125.png"] = b"resource fork"
    noisy["images/._000003-frame-125.png"] = b"resource fork"
    noisy["images/.DS_Store"] = b"finder junk"
    zip_path = _make_zip(tmp_path / "batch_x.zip", noisy)
    result = extract_batch(zip_path, install)

    assert result.n_images == 2
    out = install / "clinician_data/images_to_annotate/batch_x"
    assert not any(p.name.startswith(".") for p in out.iterdir())


def test_predictions_may_be_absent(install, tmp_path):
    images_only = {k: v for k, v in _flat_batch().items()
                   if not k.startswith("predictions/")}
    zip_path = _make_zip(tmp_path / "batch_x.zip", images_only)
    result = extract_batch(zip_path, install)
    assert result.n_images == 2
    assert result.n_prefill == 0


def test_artery_vein_subdirectories_are_accepted_as_prefill(install, tmp_path):
    """The DVA ground-truth layout, in case a batch is built from it."""
    layout = {
        "images/000003-frame-125.png": None,
        "predictions/artery/000003-frame-125.png": None,
        "predictions/veins/000003-frame-125.png": None,
    }
    zip_path = _make_zip(tmp_path / "batch_x.zip", layout)
    result = extract_batch(zip_path, install)
    assert result.n_images == 1 and result.n_prefill == 2
    assert (install / "clinician_data/predictions/batch_x/artery"
            / "000003-frame-125.png").exists()


def test_a_zip_with_no_images_folder_raises_with_its_contents(install, tmp_path):
    zip_path = _make_zip(tmp_path / "batch_x.zip",
                         {"some_folder/photo.png": None, "notes.txt": b"hi"})
    with pytest.raises(BatchLayoutError) as excinfo:
        extract_batch(zip_path, install)
    message = str(excinfo.value)
    assert "images" in message
    assert "some_folder/photo.png" in message or "photo.png" in message


def test_loose_images_with_no_images_folder_are_still_found(install, tmp_path):
    """Someone zips the images directly, with no folder around them."""
    zip_path = _make_zip(tmp_path / "batch_x.zip",
                         {"000003-frame-125.png": None,
                          "000004-frame-125.png": None})
    result = extract_batch(zip_path, install)
    assert result.n_images == 2


def test_re_extracting_the_same_batch_is_safe(install, tmp_path):
    zip_path = _make_zip(tmp_path / "batch_x.zip", _flat_batch())
    first = extract_batch(zip_path, install)
    second = extract_batch(zip_path, install)
    assert first.n_images == second.n_images == 2
    out = install / "clinician_data/images_to_annotate/batch_x"
    assert len(list(out.iterdir())) == 2


def test_non_image_files_do_not_count_as_images(install, tmp_path):
    layout = dict(_flat_batch())
    layout["images/notes.txt"] = b"not an image"
    zip_path = _make_zip(tmp_path / "batch_x.zip", layout)
    result = extract_batch(zip_path, install)
    assert result.n_images == 2


def test_returns_absolute_paths_the_caller_can_hand_to_annotate(install, tmp_path):
    zip_path = _make_zip(tmp_path / "batch_x.zip", _flat_batch())
    result = extract_batch(zip_path, install)
    assert result.images_dir.is_absolute() and result.images_dir.is_dir()
    assert result.prefill_dir.is_absolute() and result.prefill_dir.is_dir()
    assert list(result.images_dir.glob("*.png"))


def test_a_missing_zip_is_reported_clearly(install, tmp_path):
    with pytest.raises(BatchLayoutError) as excinfo:
        extract_batch(tmp_path / "nope.zip", install)
    assert "nope.zip" in str(excinfo.value)
