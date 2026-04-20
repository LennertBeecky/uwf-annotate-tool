"""Geometry and interpolation utilities shared across modules."""

from __future__ import annotations

import numpy as np


def tangent_pca(
    points: np.ndarray, idx: int, window: int = 7
) -> np.ndarray:
    """Estimate the unit tangent at `points[idx]` via PCA on a local window.

    Args:
        points: (N, 2) array of (y, x) skeleton coordinates in order.
        idx:    central index.
        window: total window length (odd); default 7 ⇒ 3 points each side.

    Returns:
        (2,) unit tangent as (ty, tx). Arbitrary sign convention.
    """
    half = window // 2
    a = max(0, idx - half)
    b = min(len(points), idx + half + 1)
    pts = points[a:b].astype(float)
    if len(pts) < 3:
        if len(pts) < 2:
            return np.array([0.0, 1.0])
        d = pts[-1] - pts[0]
        n = np.hypot(*d)
        return d / n if n else np.array([0.0, 1.0])
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, -1]                         # largest-eigenvalue eigenvector
    n = np.hypot(*v)
    return v / n if n else np.array([0.0, 1.0])


def perpendicular(tangent: np.ndarray) -> np.ndarray:
    """Rotate a unit 2-vector (ty, tx) by 90°."""
    ty, tx = tangent
    return np.array([-tx, ty])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest v such that cum(w(≤v)) >= 0.5 * total_w."""
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)
    if vals.size == 0:
        return float("nan")
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cum = np.cumsum(wts)
    total = cum[-1]
    if total <= 0:
        return float(np.median(vals))
    target = total / 2.0
    idx = int(np.searchsorted(cum, target))
    idx = min(idx, len(vals) - 1)
    return float(vals[idx])


def mad(x: np.ndarray) -> float:
    """Median absolute deviation."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def mad_trimmed_median(x: np.ndarray, k: float = 2.0) -> float:
    """Median of `x` after removing points > k·MAD from the median."""
    x = np.asarray(x, dtype=float)
    m = np.median(x)
    d = mad(x)
    if d == 0:
        return float(m)
    mask = np.abs(x - m) <= k * d
    return float(np.median(x[mask])) if mask.any() else float(m)
