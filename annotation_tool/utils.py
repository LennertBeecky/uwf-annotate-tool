"""Helpers for the napari annotation tool.

Pure-NumPy/scikit-image utilities — no napari dependency. Unit-testable
on any machine whether or not napari/Qt is installed.
"""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.transform import downscale_local_mean

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

ARTERY_COLOR_INDEX = 1
VEIN_COLOR_INDEX = 1   # both layers use label value 1; napari colour comes from the layer, not the value


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB numpy array.

    Drops the alpha channel if present. Handles grayscale by triplicating
    across channels so napari's RGB display path is used consistently.
    """
    img = Image.open(str(image_path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def build_multiscale_pyramid(image: np.ndarray, max_side_for_single: int = 2000
                             ) -> tuple[list[np.ndarray], bool]:
    """Return a napari-ready multiscale list when the image is large.

    For images smaller than `max_side_for_single` on both sides, returns
    `[image]` with `multiscale=False` semantics (single-level). For larger
    images, returns `[full, half, quarter]` so napari can swap in the
    downscaled versions when zoomed out, keeping the UI snappy.

    The label layers should always be attached at full resolution; only
    the BACKGROUND image layer benefits from multiscale.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side_for_single:
        return [image], False
    # Factor-2 and factor-4 downscales. Keep channel dim (factor 1).
    half = downscale_local_mean(image, (2, 2, 1)).astype(np.uint8)
    quarter = downscale_local_mean(image, (4, 4, 1)).astype(np.uint8)
    return [image, half, quarter], True


def skeletonise_mask(mask: np.ndarray) -> np.ndarray:
    """Binarise (>0) → `skeletonize` → uint8 {0, 255}.

    Returns a 1-pixel-wide skeleton PNG-ready array of the same HxW as the
    input mask.
    """
    bin_mask = mask > 0
    if not bin_mask.any():
        return np.zeros(mask.shape[:2], dtype=np.uint8)
    skel = skeletonize(bin_mask)
    return (skel.astype(np.uint8) * 255)


def save_skeleton_png(skeleton_uint8: np.ndarray, path: Path) -> None:
    """Save a (H, W) uint8 {0, 255} mask as a lossless PNG."""
    if skeleton_uint8.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {skeleton_uint8.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(skeleton_uint8, mode="L").save(str(path))


def neighbour_count_8(mask: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-neighbour foreground pixels (uint8)."""
    from scipy.signal import convolve2d
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    m = (mask > 0).astype(np.uint8)
    return convolve2d(m, k, mode="same", boundary="fill", fillvalue=0).astype(np.uint8) * m


@dataclass
class SaveValidation:
    ok: bool
    messages: list[str]


def validate_saved_skeleton(png_path: Path, expected_shape: tuple[int, int]
                            ) -> SaveValidation:
    """Post-hoc sanity check on a saved skeleton PNG.

    - File exists and is readable
    - Shape matches `expected_shape`
    - dtype is uint8 with unique values subset of {0, 255}
    - Skeleton is 1-pixel wide: max 8-neighbour count ≤ 3 (Y/T junctions
      are allowed; 4 means an unthinned blob survived).
    """
    msgs: list[str] = []
    if not png_path.exists():
        return SaveValidation(ok=False, messages=[f"file missing: {png_path}"])
    try:
        arr = np.asarray(Image.open(str(png_path)))
    except Exception as exc:
        return SaveValidation(ok=False, messages=[f"cannot read {png_path}: {exc}"])

    ok = True
    if arr.ndim != 2 or arr.shape != expected_shape:
        msgs.append(f"shape {arr.shape} ≠ expected {expected_shape}")
        ok = False
    if arr.dtype != np.uint8:
        msgs.append(f"dtype {arr.dtype} ≠ uint8")
        ok = False
    uniques = set(np.unique(arr).tolist())
    if not uniques.issubset({0, 255}):
        msgs.append(f"unique values {sorted(uniques)} contain values ≠ {{0, 255}}")
        ok = False
    # Skeleton thickness probe
    if (arr > 0).any():
        nc = neighbour_count_8(arr)
        max_nc = int(nc.max()) if nc.any() else 0
        if max_nc > 3:
            msgs.append(f"skeleton may not be 1-pixel wide (max 8-neighbours = {max_nc})")
            ok = False
    return SaveValidation(ok=ok, messages=msgs)


def list_images(path: Path) -> list[Path]:
    """Return the images in `path` (file → [path], dir → sorted list)."""
    if path.is_file():
        if path.suffix.lower() in IMG_EXTS:
            return [path]
        raise ValueError(f"{path} is not a supported image ({IMG_EXTS})")
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS
                      and not p.name.startswith("."))
    raise FileNotFoundError(path)


def already_annotated(image_stem: str, output_dir: Path) -> bool:
    a = output_dir / f"{image_stem}_artery.png"
    v = output_dir / f"{image_stem}_veins.png"
    return a.exists() and v.exists()


def append_time_log(csv_path: Path, row: dict) -> None:
    """Append one row to the annotation_times.csv log, creating the file + header lazily."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    fields = ["timestamp", "image_filename", "duration_seconds",
              "artery_pixel_count", "vein_pixel_count"]
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def human_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
