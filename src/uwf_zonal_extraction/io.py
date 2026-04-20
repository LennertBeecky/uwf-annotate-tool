"""Image / mask loaders and sidecar readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_mask(path: str | Path) -> np.ndarray:
    """Load an A/V mask image.

    Convention: 0=bg, 1=artery, 2=vein. Accepts either a 3-channel image
    (interpreted via common RGB colouring: red→1, blue→2), a grayscale
    image with pixel values 0/1/2, or a grayscale with 0/85/170 style.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load mask: {path}")

    if img.ndim == 3:
        # BGR; red = artery, blue = vein (matches hybrid_pipeline.py convention)
        b, g, r = cv2.split(img)
        mask = np.zeros(b.shape, dtype=np.uint8)
        mask[r > 128] = 1
        mask[b > 128] = 2
        return mask

    # Grayscale
    if img.max() <= 2:
        return img.astype(np.uint8)
    # 0/85/170/255-style: partition on the high byte
    out = np.zeros_like(img, dtype=np.uint8)
    out[(img > 32) & (img < 128)] = 1
    out[img >= 128] = 2
    return out


def load_meta_sidecar(image_path: str | Path) -> dict[str, Any]:
    """Read `{image_stem}.meta.json` if present; return {} otherwise.

    Expected keys (all optional):
        laterality:       'OD' | 'OS'
        axial_length_mm:  float
        device:           str
        patient_id:       str
        visit_id:         str
        image_datetime:   ISO-8601 str
    """
    p = Path(image_path)
    sidecar = p.parent / f"{p.stem}.meta.json"
    if not sidecar.exists():
        return {}
    with sidecar.open() as f:
        return json.load(f)
