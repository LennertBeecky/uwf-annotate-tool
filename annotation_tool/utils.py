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


EDIT_LOG_FIELDS = [
    "timestamp", "image_filename", "duration_seconds",
    "prefill_source", "lunet_thresh",
    "artery_seed_px", "artery_final_px",
    "artery_kept_px", "artery_added_px", "artery_removed_px", "artery_iou",
    "vein_seed_px", "vein_final_px",
    "vein_kept_px", "vein_added_px", "vein_removed_px", "vein_iou",
]


def skeleton_edit_distance(seed_skel: np.ndarray, final_skel: np.ndarray) -> dict:
    """Pixel-level comparison of two binary skeletons.

    Both inputs: (H, W) arrays where >0 means skeleton pixel. Returns raw
    counts and Jaccard IoU. Safe when either is empty (IoU=1.0 if both are).
    Note: skeletons are sparse, so even small spatial shifts drop IoU
    significantly — counts are the more robust signal.
    """
    seed_bin = seed_skel > 0
    final_bin = final_skel > 0
    kept = int((seed_bin & final_bin).sum())
    seed_px = int(seed_bin.sum())
    final_px = int(final_bin.sum())
    union = int((seed_bin | final_bin).sum())
    iou = (kept / union) if union > 0 else 1.0
    return {
        "seed_px": seed_px,
        "final_px": final_px,
        "kept_px": kept,
        "added_px": final_px - kept,
        "removed_px": seed_px - kept,
        "iou": round(float(iou), 4),
    }


def append_edit_log(csv_path: Path, row: dict) -> None:
    """Append one row to the annotation_edits.csv audit log."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDIT_LOG_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def lunet_prefill_masks(
    image_path: Path,
    model_path: Path,
    thresh: float = 0.5,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LUNet A/V segmentation (cached) and return binary seed masks.

    Returns (artery_mask, vein_mask) as uint8 {0, 1} at the image's native
    resolution. Cached probabilities are stored to
    `cache_dir/{stem}_probs.npz` as float16 so re-opens skip inference.

    Lazy imports cv2/onnxruntime so the utils test suite stays light.
    """
    import cv2  # noqa: PLC0415

    stem = image_path.stem
    cache_path = cache_dir / f"{stem}_probs.npz" if cache_dir is not None else None

    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as z:
            art_probs = z["artery"].astype(np.float32)
            vein_probs = z["vein"].astype(np.float32)
    else:
        import sys as _sys  # noqa: PLC0415

        _src_dir = Path(__file__).resolve().parents[1] / "src"
        if _src_dir.exists() and str(_src_dir) not in _sys.path:
            _sys.path.insert(0, str(_src_dir))
        from uwf_zonal_extraction.segmentation.lunet import (  # noqa: PLC0415
            LunetSegmenter,
        )

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"cv2 could not read {image_path}")
        seg = LunetSegmenter(model_path)
        probs = seg.predict_tiled(image_bgr)
        art_probs = probs[..., 0]
        vein_probs = probs[..., 1]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                artery=art_probs.astype(np.float16),
                vein=vein_probs.astype(np.float16),
            )

    art_mask = np.where(np.isfinite(art_probs), art_probs > thresh, False).astype(np.uint8)
    vein_mask = np.where(np.isfinite(vein_probs), vein_probs > thresh, False).astype(np.uint8)
    return art_mask, vein_mask


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def human_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
