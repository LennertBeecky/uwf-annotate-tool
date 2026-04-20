"""Fovea detection — arcade-apex heuristic with manual-sidecar override.

v1 implements the simple arcade-minimum heuristic. For cases where this
fails, a sidecar JSON next to the image (`{stem}.landmarks.json` with
`{"fovea": [y, x]}`) takes precedence.

The heuristic looks for a low-vessel-density point roughly 2.5 DD
temporal from the OD along the mean-arcade curvature axis. This is good
enough for healthy UWF but unreliable when:
  - the arcades are asymmetric (e.g. pathological vessel displacement)
  - the macula has an advanced lesion shadowing the whole region
  - the fovea is near the frame edge (steered Optos mode)

Improvements (v2 backlog):
  - dedicated fovea CNN
  - investigate LUNet-ODC bg_a/bg_b channels (may correlate with macula)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def detect_fovea_from_probs(
    av_probs: np.ndarray,                  # (H, W, 2) p_artery, p_vein
    od_center: tuple[float, float],
    od_radius_px: float,
    laterality: str = "OD",
    search_radius_dd: float = 3.5,
    expected_distance_dd: float = 2.5,
) -> tuple[Optional[tuple[float, float]], float]:
    """Heuristic fovea detection.

    Searches a temporal annulus of radius 2.0–3.5 DD from the OD
    (temporal direction inferred from laterality) for a low-vessel-density
    minimum.

    Returns:
        ((x, y), confidence) or (None, 0.0).
    """
    h, w = av_probs.shape[:2]
    vessel_density = av_probs[:, :, 0] + av_probs[:, :, 1]
    vessel_density = cv2.GaussianBlur(vessel_density, (31, 31), sigmaX=10)

    cx, cy = od_center
    dd = 2 * od_radius_px
    if dd <= 0:
        return None, 0.0

    # Temporal direction: +x for OD (right eye), -x for OS (left eye).
    sign = +1.0 if laterality.upper() in {"OD", "R", "RIGHT"} else -1.0

    # Search ROI: annulus around expected fovea point.
    fx = cx + sign * expected_distance_dd * dd
    fy = cy
    roi_radius = 0.75 * dd

    y0 = int(max(0, fy - roi_radius))
    y1 = int(min(h, fy + roi_radius))
    x0 = int(max(0, fx - roi_radius))
    x1 = int(min(w, fx + roi_radius))
    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return None, 0.0

    patch = vessel_density[y0:y1, x0:x1]
    # Mask out regions outside the search annulus.
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - fx) ** 2 + (yy - fy) ** 2)
    valid = dist <= roi_radius
    if not valid.any():
        return None, 0.0

    masked = np.where(valid, patch, np.inf)
    min_idx = np.unravel_index(np.argmin(masked), masked.shape)
    foy = y0 + min_idx[0]
    fox = x0 + min_idx[1]

    # Confidence: relative contrast between minimum and the search-ROI median.
    med = float(np.median(patch[valid]))
    mn = float(masked[min_idx])
    confidence = float(np.clip((med - mn) / (med + 1e-6), 0.0, 1.0))
    return (float(fox), float(foy)), confidence


def load_sidecar_fovea(image_path: str | Path) -> Optional[tuple[float, float]]:
    """Return (x, y) from `{image_stem}.landmarks.json` if present."""
    p = Path(image_path)
    sidecar = p.with_suffix("").with_suffix(".landmarks.json")
    if not sidecar.exists():
        sidecar = p.parent / f"{p.stem}.landmarks.json"
    if not sidecar.exists():
        return None
    with sidecar.open() as f:
        data = json.load(f)
    fov = data.get("fovea")
    if fov is None or len(fov) != 2:
        return None
    # Stored as [y, x] for numpy consistency; return (x, y).
    y, x = fov
    return (float(x), float(y))
