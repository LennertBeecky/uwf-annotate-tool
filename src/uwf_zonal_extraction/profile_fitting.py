"""Convolved step model for sub-pixel vessel-diameter extraction.

Vein model:
    I(x) = B - (A/2) * [ erf((x - c + w/2) / (sigma*sqrt(2)))
                       - erf((x - c - w/2) / (sigma*sqrt(2))) ]

Artery model (single-Gaussian reflex by default):
    I(x) = I_vein(x) + R * exp( -(x - c)^2 / (2 * sigma_r^2) )

`w` (diameter) is explicitly decoupled from `sigma` (optical PSF blur).
PSF `sigma` is later post-processed into a radial polynomial (stage 04).

v1 implements the core per-profile fit; the radial-sigma pass is a TODO
that operates on a pool of first-pass fits (see docs/04).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erf

from uwf_zonal_extraction.config import ProfileFitCfg
from uwf_zonal_extraction.utils import mad_trimmed_median

ReflexModel = Literal["none", "single_gauss", "twin_gauss"]


def fwhm_width_from_profile(
    t: np.ndarray,
    profile: np.ndarray,
    min_depth: float = 3.0,
) -> float | None:
    """Crude FWHM width estimate from an intensity profile, mask-free.

    Used as `w_init` for the LM fit. Decouples width estimation from the
    (possibly thin-edged) LUNet vessel mask: once we know roughly where a
    vessel is (via the skeleton), the full-width-at-half-maximum of its
    perpendicular intensity profile is a reliable first-order width estimate.

    Returns None if the profile has no meaningful dip.
    """
    tail = max(int(0.25 * len(profile)), 3)
    baseline_pool = np.concatenate([profile[:tail], profile[-tail:]])
    B = float(np.median(baseline_pool))
    depth = B - float(np.min(profile))
    if depth < min_depth:
        return None
    half_level = B - 0.5 * depth
    below = profile < half_level
    if not below.any():
        return None
    first_idx = int(np.argmax(below))
    last_idx = int(len(below) - 1 - np.argmax(below[::-1]))
    fwhm = float(t[last_idx] - t[first_idx])
    return max(1.0, fwhm)


@dataclass
class ProfileFit:
    w: float                          # vessel diameter (pixels)
    sigma: float                      # PSF blur (pixels)
    c: float                          # centre along the profile axis (pixels)
    B: float                          # background
    A: float                          # absorption amplitude
    R: float                          # reflex amplitude (0 if vein)
    sigma_r: float                    # reflex sigma (0 if vein)
    reflex_model: ReflexModel
    rmse_rel: float                   # RMSE / A  (normalized quality)
    sigma_w: float                    # 1-σ SE on w from Jacobian (pixels)
    success: bool
    t: np.ndarray                     # profile axis positions
    profile: np.ndarray               # profile intensities
    residual: np.ndarray              # fit residuals


# --- model functions --------------------------------------------------


def _convolved_step(x: np.ndarray, B: float, A: float, c: float, w: float, sigma: float) -> np.ndarray:
    s = sigma * math.sqrt(2.0)
    return B - (A / 2.0) * (erf((x - c + w / 2.0) / s) - erf((x - c - w / 2.0) / s))


def _vein_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    B, A, c, w, sigma = params
    return _convolved_step(x, B, A, c, w, sigma)


def _artery_single_reflex(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    B, A, c, w, sigma, R, sigma_r = params
    reflex = R * np.exp(-((x - c) ** 2) / (2.0 * sigma_r**2))
    return _convolved_step(x, B, A, c, w, sigma) + reflex


def _artery_twin_reflex(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    B, A, c, w, sigma, R, sigma_r, d = params
    reflex = R * (
        np.exp(-((x - c - d) ** 2) / (2.0 * sigma_r**2))
        + np.exp(-((x - c + d) ** 2) / (2.0 * sigma_r**2))
    )
    return _convolved_step(x, B, A, c, w, sigma) + reflex


# --- profile extraction ----------------------------------------------


def extract_profile(
    image: np.ndarray,                # (H, W) float32/float64 — MUST be pre-cast
    center_yx: tuple[float, float],
    perp_yx: np.ndarray,              # unit vector (perp_y, perp_x)
    half_width_px: int,
    step_px: float = 0.25,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Bicubic-sampled perpendicular profile.

    IMPORTANT: `image` must already be float32 or float64. Casting a
    4000×4000 uint8 array to float inside this hot loop costs ~128 MB
    per call; callers MUST cast once upstream.

    Returns:
        (t, profile): t in pixel units along perp; profile in image intensity.
    """
    if image.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"extract_profile expects float image (pre-cast by caller); "
            f"got dtype={image.dtype}. Cast once outside the loop."
        )
    t = np.arange(-half_width_px, half_width_px + step_px, step_px, dtype=float)
    cy, cx = center_yx
    sample_y = cy + t * perp_yx[0]
    sample_x = cx + t * perp_yx[1]
    coords = np.stack([sample_y, sample_x])
    profile = ndimage.map_coordinates(image, coords, order=order, mode="reflect")
    return t, profile


# --- fitting ---------------------------------------------------------


def fit_vessel_profile(
    t: np.ndarray,
    profile: np.ndarray,
    vessel_type: Literal["artery", "vein"],
    w_init: float,
    cfg: ProfileFitCfg,
    sigma_fixed: float | None = None,
) -> ProfileFit:
    """Fit the convolved-step model to one perpendicular profile.

    If `sigma_fixed` is set, `sigma` is clamped to that value (second pass).
    """
    # Initial guesses. Compute an intensity-driven FWHM width up front and
    # prefer it over the mask-derived `w_init` — the LUNet mask tends to be
    # too thin for veins, so its EDT-derived width is biased low. The
    # intensity profile is the direct physical observable of vessel width.
    tail = int(round(cfg.baseline_tail_fraction * len(profile)))
    tail = max(tail, 3)
    baseline_pool = np.concatenate([profile[:tail], profile[-tail:]])
    B0 = mad_trimmed_median(baseline_pool)
    A0 = max(B0 - float(np.min(profile)), 1.0)
    c0 = float(t[int(np.argmin(profile))])

    w_intensity = fwhm_width_from_profile(t, profile, min_depth=3.0)
    if w_intensity is not None:
        w0 = w_intensity
    else:
        w0 = max(float(w_init), 1.0)
    sigma0 = sigma_fixed if sigma_fixed is not None else max(w0 / 4.0, 0.5)

    half = float(t[-1])
    # Tighten w bounds to a multiplicative range around w_init. The medial-axis
    # estimate is a reliable first-order visual width; letting the LM wander
    # to arbitrarily small w lets it collapse to "flat baseline + tiny step"
    # local minima that look low-residual but return A ≈ 0. Bounds of
    # [0.4·w_init, 3·w_init] keep the fit anchored to the skeleton estimate
    # while allowing for medial-axis under-estimation on thick or tilted vessels.
    w_lo = max(0.5, 0.4 * w0)
    w_hi = min(half, max(3.0 * w0, w0 + 2.0))  # always > w_lo
    bounds_vein = (
        [0.0, 0.0, float(t[0]), w_lo, 0.1],
        [255.0, 255.0, float(t[-1]), w_hi, max(half / 2.0, 0.2)],
    )

    def _run(residuals_fn, p0, bounds):
        return least_squares(
            residuals_fn,
            p0,
            bounds=bounds,
            method="trf",
            loss=cfg.lm_loss,
            f_scale=max(A0 / cfg.lm_f_scale_divisor, 1.0),
            max_nfev=2000,
        )

    # Fit vein model first (baseline for artery residual diagnostic).
    def _residuals_vein(p):
        return _vein_model(p, t) - profile

    try:
        res_vein = _run(_residuals_vein, [B0, A0, c0, w0, sigma0], bounds_vein)
    except Exception:
        return _failed_fit(t, profile)

    if vessel_type == "vein":
        return _finalize(t, profile, res_vein, _vein_model, "none", 4, sigma_fixed)

    # Arteries: single-Gaussian reflex as default; twin fallback if needed.
    R0 = 0.3 * A0
    sigma_r0 = max(w0 / 4.0, 0.5)
    bounds_single = (
        [0.0, 0.0, float(t[0]), w_lo, 0.1, 0.0, 0.1],
        [255.0, 255.0, float(t[-1]), w_hi, max(half / 2.0, 0.2), 255.0, max(half / 4.0, 0.2)],
    )

    def _residuals_single(p):
        return _artery_single_reflex(p, t) - profile

    try:
        res_single = _run(
            _residuals_single, [B0, A0, c0, w0, sigma0, R0, sigma_r0], bounds_single
        )
    except Exception:
        return _finalize(t, profile, res_vein, _vein_model, "none", 4, sigma_fixed)

    single_fit = _finalize(t, profile, res_single, _artery_single_reflex, "single_gauss", 4, sigma_fixed)

    # Twin-Gauss fallback if the single-reflex residual has a central dip.
    centre_band = (t >= single_fit.c - 1.0) & (t <= single_fit.c + 1.0)
    mad_r = float(np.median(np.abs(single_fit.residual - np.median(single_fit.residual)))) or 1e-6
    if np.any(single_fit.residual[centre_band] < -cfg.twin_reflex_trigger * mad_r):
        bounds_twin = (
            [0.0, 0.0, float(t[0]), w_lo, 0.1, 0.0, 0.1, 0.0],
            [
                255.0, 255.0, float(t[-1]), w_hi, max(half / 2.0, 0.2),
                255.0, max(half / 4.0, 0.2), max(half, 1.0),
            ],
        )

        def _residuals_twin(p):
            return _artery_twin_reflex(p, t) - profile

        try:
            res_twin = _run(
                _residuals_twin,
                [B0, A0, c0, w0, sigma0, R0, sigma_r0, max(w0 / 2.0, 0.5)],
                bounds_twin,
            )
            twin_fit = _finalize(t, profile, res_twin, _artery_twin_reflex, "twin_gauss", 4, sigma_fixed)
            if twin_fit.rmse_rel < single_fit.rmse_rel:
                return twin_fit
        except Exception:
            pass

    return single_fit


# --- helpers ---------------------------------------------------------


def _finalize(
    t: np.ndarray,
    profile: np.ndarray,
    res,
    model_fn,
    reflex: ReflexModel,
    w_index: int,
    sigma_fixed: float | None,
) -> ProfileFit:
    params = res.x
    fitted = model_fn(params, t)
    residual = profile - fitted
    A = float(params[1]) if params[1] > 1.0 else 1.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    rmse_rel = rmse / A

    # Jacobian covariance → σ_w for parameter index 3 (w).
    # Reduced χ² = rss / (n - p); cov = pinv(J.T J) * reduced_χ².
    try:
        jt_j = res.jac.T @ res.jac
        n = len(residual)
        p = len(params)
        dof = max(n - p, 1)
        rss = float(np.sum(residual**2))
        reduced_chi2 = rss / dof
        cov = np.linalg.pinv(jt_j) * reduced_chi2
        sigma_w = float(np.sqrt(max(cov[3, 3], 0.0)))
    except Exception:
        sigma_w = float("nan")

    B, A_fit, c, w, sigma = (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
        float(params[4]),
    )
    R_val = float(params[5]) if len(params) > 5 else 0.0
    sigma_r = float(params[6]) if len(params) > 6 else 0.0

    success = res.success and not np.isnan(sigma_w) and rmse_rel < 1.0
    return ProfileFit(
        w=w,
        sigma=sigma,
        c=c,
        B=B,
        A=A_fit,
        R=R_val,
        sigma_r=sigma_r,
        reflex_model=reflex,
        rmse_rel=rmse_rel,
        sigma_w=sigma_w,
        success=bool(success),
        t=t,
        profile=profile,
        residual=residual,
    )


def _failed_fit(t: np.ndarray, profile: np.ndarray) -> ProfileFit:
    return ProfileFit(
        w=float("nan"),
        sigma=float("nan"),
        c=float("nan"),
        B=float("nan"),
        A=float("nan"),
        R=0.0,
        sigma_r=0.0,
        reflex_model="none",
        rmse_rel=float("nan"),
        sigma_w=float("nan"),
        success=False,
        t=t,
        profile=profile,
        residual=profile * 0.0,
    )
