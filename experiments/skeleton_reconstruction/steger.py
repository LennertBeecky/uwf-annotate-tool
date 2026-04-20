"""Steger 1998 — Hessian-based sub-pixel ridge detection.

Direct scipy implementation (the ridge-detection PyPI package is translated
from a Java plugin and is fiddly; 50 lines of scipy does the job).

Two public entry points:

1. `hessian_components(image, sigma)` — computes Ix, Iy, Ixx, Iyy, Ixy at
   scale σ using derivatives-of-Gaussian kernels. Run once per image.

2. `steger_snap_points(image, points, sigma=1.5, max_offset=2.0,
   dark_ridge=True)` — refines a list of skeleton pixels to sub-pixel
   ridge positions, returning refined (y, x), Hessian-derived normals,
   and a validity mask.

Key equations (Steger 1998, eq. 15):

    H = [[Iyy, Ixy],             # 2D Hessian in (row, col) = (y, x) order
         [Ixy, Ixx]]
    eigenvalues λ₁, λ₂ of H; for a DARK ridge the maximum-|λ| eigenvalue
    is POSITIVE (the ridge is a local minimum → convex in the normal
    direction).
    n = eigenvector of that eigenvalue  (unit, 2-D)
    Taylor offset along n:
        t = -(n · ∇I) / (nᵀ H n)
    If |t| ≤ max_offset pixels, refined position = (y, x) + t · n.
    Otherwise the point is flagged as NOT on a ridge (flat noise, off-line
    starting point, etc.) and should be passed through as the raw skeleton
    pixel with a PCA-derived tangent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class HessianComponents:
    """First and second derivatives of a Gaussian-smoothed image."""

    Iy: np.ndarray
    Ix: np.ndarray
    Iyy: np.ndarray
    Ixx: np.ndarray
    Ixy: np.ndarray
    sigma: float


def hessian_components(image: np.ndarray, sigma: float) -> HessianComponents:
    """Compute first and second partial derivatives of the Gaussian-smoothed image.

    Uses `scipy.ndimage.gaussian_filter` with the `order=` arg to get
    derivatives-of-Gaussian directly (not a separate smooth + finite
    difference, which would double the effective blur).

    Image axes: (y, x) — `order=(dy, dx)` in scipy convention.
    """
    img = image.astype(np.float32, copy=False)
    Iy = gaussian_filter(img, sigma=sigma, order=(1, 0))
    Ix = gaussian_filter(img, sigma=sigma, order=(0, 1))
    Iyy = gaussian_filter(img, sigma=sigma, order=(2, 0))
    Ixx = gaussian_filter(img, sigma=sigma, order=(0, 2))
    Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1))
    return HessianComponents(Iy=Iy, Ix=Ix, Iyy=Iyy, Ixx=Ixx, Ixy=Ixy, sigma=sigma)


def steger_refine_single(
    y: int,
    x: int,
    H: HessianComponents,
    dark_ridge: bool = True,
    max_offset: float = 2.0,
) -> tuple[float, float, np.ndarray, float, bool]:
    """Refine ONE point (y, x) to sub-pixel ridge position.

    Returns (refined_y, refined_x, normal_yx, ridge_strength, valid).

    - `normal_yx` is the 2-unit vector perpendicular to the ridge (direction
      of max curvature).
    - `ridge_strength` is the magnitude of the dominant Hessian eigenvalue
      (higher = more ridge-like).
    - `valid = True` iff the Taylor offset is within `max_offset` pixels.
      When False, the returned refined position is the original integer
      pixel and the normal is the Hessian principal direction but the
      caller should treat the result as non-authoritative.
    """
    h, w = H.Iyy.shape
    if not (0 <= y < h and 0 <= x < w):
        # Out of bounds — pass through
        return float(y), float(x), np.array([0.0, 1.0]), 0.0, False

    Iyy = float(H.Iyy[y, x])
    Ixx = float(H.Ixx[y, x])
    Ixy = float(H.Ixy[y, x])
    Iy = float(H.Iy[y, x])
    Ix = float(H.Ix[y, x])

    # Hessian in (y, x) order: H = [[Iyy, Ixy], [Ixy, Ixx]]
    trace = Iyy + Ixx
    det = Iyy * Ixx - Ixy * Ixy
    disc = max(trace * trace / 4.0 - det, 0.0)
    sqrt_disc = np.sqrt(disc)
    lam1 = trace / 2.0 + sqrt_disc
    lam2 = trace / 2.0 - sqrt_disc

    # Pick eigenvalue with LARGEST ABSOLUTE value; for a DARK ridge on bright
    # background the max-|λ| eigenvalue is POSITIVE (profile is a local
    # minimum, second derivative positive). For bright ridges it's negative.
    if abs(lam1) >= abs(lam2):
        lam_dom = lam1
    else:
        lam_dom = lam2

    if dark_ridge and lam_dom <= 0.0:
        # Point is not a dark-ridge structure (could be a bright streak
        # or flat background noise).
        return float(y), float(x), np.array([0.0, 1.0]), 0.0, False
    if not dark_ridge and lam_dom >= 0.0:
        return float(y), float(x), np.array([0.0, 1.0]), 0.0, False

    # Eigenvector for λ_dom — analytic 2x2 form.
    # (H - λ I) v = 0 → v proportional to (Ixy, λ - Iyy) or (λ - Ixx, Ixy)
    if abs(Ixy) > 1e-9:
        ny, nx = (lam_dom - Ixx), Ixy
    elif abs(Iyy - lam_dom) > abs(Ixx - lam_dom):
        # Hessian diagonal: vertical ridge → normal is +x
        ny, nx = 0.0, 1.0
    else:
        ny, nx = 1.0, 0.0

    norm = np.hypot(ny, nx)
    if norm < 1e-12:
        return float(y), float(x), np.array([0.0, 1.0]), 0.0, False
    ny /= norm
    nx /= norm
    normal = np.array([ny, nx], dtype=float)

    # Taylor offset along the normal:
    #     t = -(n · ∇I) / (nᵀ H n)
    grad_dot_n = ny * Iy + nx * Ix
    n_H_n = ny * (ny * Iyy + nx * Ixy) + nx * (ny * Ixy + nx * Ixx)
    if abs(n_H_n) < 1e-9:
        return float(y), float(x), normal, float(abs(lam_dom)), False
    t = -grad_dot_n / n_H_n

    if abs(t) > max_offset:
        # Sub-pixel offset too large → the point is too far off the true
        # ridge for reliable Taylor refinement.
        return float(y), float(x), normal, float(abs(lam_dom)), False

    refined_y = float(y) + t * ny
    refined_x = float(x) + t * nx
    return refined_y, refined_x, normal, float(abs(lam_dom)), True


@dataclass
class StegerSnapResult:
    """Per-point output of `steger_snap_points`."""

    refined_points: np.ndarray     # (N, 2) float, (y, x) sub-pixel
    normals: np.ndarray            # (N, 2) float, (ny, nx) unit vectors
    strengths: np.ndarray          # (N,) float, |dominant Hessian eigenvalue|
    valid: np.ndarray              # (N,) bool, True if Taylor snap succeeded


def steger_snap_points(
    image: np.ndarray,
    points_yx: np.ndarray,
    sigma: float = 1.5,
    max_offset: float = 2.0,
    dark_ridge: bool = True,
    hessian: HessianComponents | None = None,
) -> StegerSnapResult:
    """Refine a list of skeleton points to sub-pixel ridge positions.

    Args:
        image: (H, W) 2-D grayscale (float or uint8). Caller usually passes
            the green channel of the CFP. Polarity: `dark_ridge=True`
            expects the VESSEL to be DARKER than the background.
        points_yx: (N, 2) int or float array of (y, x) skeleton coordinates.
        sigma: Gaussian smoothing scale for the Hessian. Best matched to
            the *expected* vessel radius. 1.5 px is a reasonable default
            for CFP main vessels; small capillaries are better analysed
            with σ ≈ 1.0.
        max_offset: |t| cap for the Taylor refinement (pixels). Points with
            larger offsets are flagged invalid and returned unchanged.
        dark_ridge: True for dark-on-bright (green channel vessels).
        hessian: precomputed HessianComponents for this image + σ. If None,
            computed here. Pass this when calling in a loop to avoid
            recomputing the full Hessian per call.

    Returns:
        StegerSnapResult with refined positions, normals, strengths, validity.
    """
    H = hessian if hessian is not None else hessian_components(image, sigma)
    n = len(points_yx)
    refined = np.empty((n, 2), dtype=np.float64)
    normals = np.empty((n, 2), dtype=np.float64)
    strengths = np.empty(n, dtype=np.float64)
    valid = np.empty(n, dtype=bool)

    for i in range(n):
        y, x = int(round(float(points_yx[i, 0]))), int(round(float(points_yx[i, 1])))
        ry, rx, n_yx, s, ok = steger_refine_single(
            y, x, H, dark_ridge=dark_ridge, max_offset=max_offset
        )
        refined[i, 0] = ry
        refined[i, 1] = rx
        normals[i] = n_yx
        strengths[i] = s
        valid[i] = ok
    return StegerSnapResult(
        refined_points=refined,
        normals=normals,
        strengths=strengths,
        valid=valid,
    )
