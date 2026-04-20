"""Skeleton-driven mask reconstruction for the validation experiment.

Given a binary GT mask (artery OR vein), skeletonize it, walk along every
skeleton pixel, extract a perpendicular profile from the raw image's green
channel, fit the convolved-step model (reusing uwf_zonal_extraction's
`fit_vessel_profile`), and paint the fitted vessel cross-section onto a
reconstructed mask.

Two outputs per A/V class:
    recon_hard  : uint8 binary mask (the vessel occupies [c - w/2, c + w/2] along the normal)
    recon_soft  : float32 [0, 1] per-pixel "is-vessel" score derived from the
                  fitted model (P = (B − I_model) / A clipped to [0, 1])

This mirrors what an annotator drawing only the centerline would deliver,
and validates whether the physics-informed fit can recover the full mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.special import erf
from skimage.morphology import (
    closing,
    disk,
    skeletonize,
)
from skimage.measure import label as _sklabel

from uwf_zonal_extraction.config import ProfileFitCfg
from uwf_zonal_extraction.profile_fitting import (
    ProfileFit,
    extract_profile,
    fit_vessel_profile,
    fwhm_width_from_profile,
)
from uwf_zonal_extraction.utils import mad_trimmed_median

# Default σ_PSF for 1444×1444 CFP (≈ 0.7–1.0 px from imaging calibrations).
# Overridden per-image by two-phase calibration when enough successful
# convolved-step fits are available.
DEFAULT_SIGMA_PSF_PX = 0.8

# FWHM = 2√(2 ln 2) · σ ≈ 2.355·σ for a Gaussian
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _gaussian_profile(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    B, A, c, sigma = params
    return B - A * np.exp(-0.5 * ((x - c) / max(sigma, 1e-3)) ** 2)


def fit_thin_vessel_gaussian(
    t: np.ndarray,
    profile: np.ndarray,
    sigma_psf: float,
    min_depth: float = 1.0,
) -> ProfileFit | None:
    """Fit a pure-Gaussian absorption profile for sub-resolution vessels.

    When the vessel width is smaller than ~2·σ_PSF, the convolved-step model
    becomes ill-conditioned (w and σ trade off). In that regime the observed
    profile IS a Gaussian whose variance is σ_vessel² + σ_PSF². Fitting a
    Gaussian and then deconvolving the PSF gives an image-derived width that
    does not rely on the GT mask (protects the "skeleton + image → mask"
    scientific claim).

    Returns a ProfileFit-compatible object so _paint_fit can consume it, or
    None when the profile has no meaningful dip.
    """
    tail = max(int(0.25 * len(profile)), 3)
    baseline_pool = np.concatenate([profile[:tail], profile[-tail:]])
    B0 = float(mad_trimmed_median(baseline_pool))
    depth = B0 - float(np.min(profile))
    if depth < min_depth:
        return None

    c0 = float(t[int(np.argmin(profile))])
    sigma0 = max(sigma_psf * 1.2, 0.8)
    A0 = max(depth, 1.0)

    # Keep c tight — a loose c bound lets the fit chase reflex side-lobes.
    c_lo = max(float(t[0]), c0 - 2.0)
    c_hi = min(float(t[-1]), c0 + 2.0)
    bounds = (
        [0.0, 0.5, c_lo, max(sigma_psf * 0.5, 0.25)],
        [255.0, 255.0, c_hi, 6.0],
    )

    try:
        res = least_squares(
            lambda p: _gaussian_profile(p, t) - profile,
            [B0, A0, c0, sigma0],
            bounds=bounds,
            method="trf",
            loss="soft_l1",
            f_scale=max(A0 / 5.0, 1.0),
            max_nfev=500,
        )
    except Exception:
        return None

    B, A, c, sigma = res.x
    residual = _gaussian_profile(res.x, t) - profile
    rmse = float(np.sqrt(np.mean(residual**2)))
    rmse_rel = rmse / max(A, 1.0)

    # Deconvolve PSF and recover vessel width as FWHM of the image-only Gaussian.
    sigma_vessel_sq = max(sigma**2 - sigma_psf**2, 1e-3)
    sigma_vessel = float(np.sqrt(sigma_vessel_sq))
    w = max(_FWHM_PER_SIGMA * sigma_vessel, 1.0)

    success = bool(res.success) and rmse_rel < 0.35 and not np.isnan(sigma)

    return ProfileFit(
        w=w,
        sigma=max(sigma, sigma_psf),
        c=float(c),
        B=float(B),
        A=float(A),
        R=0.0,
        sigma_r=0.0,
        reflex_model="none",
        rmse_rel=rmse_rel,
        sigma_w=float("nan"),
        success=success,
        t=t,
        profile=profile,
        residual=residual,
    )
from uwf_zonal_extraction.skeleton import trace_segments
from uwf_zonal_extraction.utils import perpendicular, tangent_pca

VesselType = Literal["artery", "vein"]

# --- configuration ------------------------------------------------------


@dataclass
class ReconstructCfg:
    half_width_px: int = 20
    profile_step_px: float = 0.25
    profile_interp_order: int = 3              # bicubic
    sample_stride_px: float = 2.0              # 2 px walk; paint-stripe covers the gap
    skeleton_min_length_px: int = 5
    skeleton_closing_radius: int = 1           # mild gap-bridging before skeletonize
    bifurcation_skip_px: float = 2.0           # skip samples very near a junction only
    crossing_skip_px: float = 4.0              # skip samples very near the other class's skeleton
    thin_vessel_fallback_px: float = 2.0       # if fit fails and mask width < this, use distance transform
    post_closing_radius: int = 0               # morph close; Fix #4 makes it mostly redundant
    fwhm_min_depth: float = 1.5                # lower = accepts thinner/shallower vessels
    junction_disc_search_px: float = 8.0       # radius to sample adjacent widths for disc paint
    junction_disc_enabled: bool = False        # disc-paint at junctions (off — overshoots boundaries)
    junction_disc_scale: float = 0.7           # if enabled, paint radius = scale × max-DT
    # Fix #4: per-pixel model-threshold painting. A pixel is painted if the
    # fitted erf-step model predicts absorption > paint_threshold_frac × A.
    # 0.50 = half-max of the erf profile (equivalent to hard |d| ≤ w/2 for a
    # SINGLE sample); 0.35 empirically matches manual annotator contours
    # according to the morphometry panel.
    paint_threshold_frac: float = 0.35
    profile_fit: ProfileFitCfg = field(default_factory=ProfileFitCfg)


# --- skeleton helpers ---------------------------------------------------


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), ddepth=cv2.CV_8U, kernel=k)


def _prune_spurs(skel: np.ndarray, min_length_px: int) -> np.ndarray:
    """Remove connected components shorter than `min_length_px` from the
    1-px skeleton. Hand-rolled with `skimage.measure.label` + bincount to
    sidestep `skimage.morphology.remove_small_objects` — deprecated in 0.26.
    """
    skel = skel.astype(bool)
    if min_length_px <= 1:
        return skel
    lbl = _sklabel(skel, connectivity=2)
    counts = np.bincount(lbl.ravel())
    small_labels = np.flatnonzero(counts < min_length_px)
    # label 0 is background; mask it out explicitly
    small_labels = small_labels[small_labels != 0]
    if small_labels.size == 0:
        return skel
    out = skel.copy()
    out[np.isin(lbl, small_labels)] = False
    return out


def skeletonize_mask(mask: np.ndarray, cfg: ReconstructCfg) -> np.ndarray:
    """Binary mask → 1-px skeleton, cleaned of short spurs."""
    m = mask.astype(bool)
    if cfg.skeleton_closing_radius > 0:
        m = closing(m, disk(cfg.skeleton_closing_radius))
    skel = skeletonize(m)
    skel = _prune_spurs(skel, cfg.skeleton_min_length_px)
    return skel.astype(np.uint8)


def junction_mask(skel: np.ndarray) -> np.ndarray:
    """Bool mask of skeleton pixels with 3+ neighbours (bifurcations + crossings)."""
    nc = _neighbour_count(skel.astype(np.uint8)) * skel.astype(np.uint8)
    return nc >= 3


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Distance transform: distance (px) from every pixel to the nearest mask pixel."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~mask).astype(np.float32)


# --- single-segment reconstruction --------------------------------------


def _paint_fit(
    recon_hard: np.ndarray,
    recon_soft: np.ndarray,
    center_yx: tuple[float, float],
    tangent_yx: np.ndarray,
    perp_yx: np.ndarray,
    fit: ProfileFit,
    label: int,
    tangent_half_px: float,
    threshold_frac: float = 0.5,
) -> None:
    """Paint an oriented 2D region aligned with (tangent, perp). A pixel is
    painted iff the fitted model predicts absorption exceeding
    `threshold_frac × A` at its perpendicular offset. Overlapping samples
    keep the maximum absorption (naturally blends adjacent fits).

    Soft: `P(t) = clip((B − I_model(t)) / A, 0, 1)` per-pixel.
    """
    half_w = max(fit.w / 2.0, 0.5)
    step = 0.25
    # Clamp σ for painting to the plausible CFP PSF range. The convolved-step
    # fitter often inflates σ on thick vessels (w↔σ trade-off), which would
    # soften the painted edge and shift the absorption=threshold contour far
    # from the nominal w/2 boundary. We paint as if the PSF is ~0.8 px so
    # the threshold crosses cleanly at d = w/2.
    sigma_paint = float(np.clip(fit.sigma, 0.4, 1.2))
    # Extent covering the full soft shoulder.
    t_perp_ext = half_w + 3.0 * sigma_paint
    t_perp = np.arange(-t_perp_ext, t_perp_ext + step, step)
    t_par = np.arange(-tangent_half_px, tangent_half_px + step, step)
    tt_par, tt_perp = np.meshgrid(t_par, t_perp, indexing="ij")
    cy, cx = center_yx
    ys = cy + tt_par * tangent_yx[0] + tt_perp * perp_yx[0]
    xs = cx + tt_par * tangent_yx[1] + tt_perp * perp_yx[1]

    h, w = recon_hard.shape
    yi = np.rint(ys).astype(int).ravel()
    xi = np.rint(xs).astype(int).ravel()
    tt_perp_flat = tt_perp.ravel()
    in_bounds = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)

    # Model profile using the CLAMPED σ.
    s = sigma_paint * np.sqrt(2.0)
    I_model = fit.B - (fit.A / 2.0) * (
        erf((tt_perp_flat + half_w) / s) - erf((tt_perp_flat - half_w) / s)
    )
    depth = np.clip((fit.B - I_model) / max(fit.A, 1.0), 0.0, 1.0).astype(np.float32)

    # Hard: pixel is painted when model absorption crosses threshold_frac.
    # This replaces the hard rectangle `|t_perp| ≤ w/2` and naturally
    # follows the fitted σ (the erf transition width).
    hard_mask = in_bounds & (depth > threshold_frac)
    recon_hard[yi[hard_mask], xi[hard_mask]] = label

    # Soft channel: per-pixel absorption, max-blended with existing content.
    ch = label - 1
    idx_y = yi[in_bounds]
    idx_x = xi[in_bounds]
    depth_in = depth[in_bounds]
    existing = recon_soft[idx_y, idx_x, ch]
    recon_soft[idx_y, idx_x, ch] = np.maximum(existing, depth_in)


@dataclass
class ReconstructionStats:
    n_segments: int = 0
    n_sample_attempts: int = 0
    n_sample_fits_ok: int = 0
    n_sample_fits_failed: int = 0
    n_sample_skipped_junction: int = 0
    n_sample_skipped_crossing: int = 0
    n_sample_fallback_thin: int = 0
    n_gaussian_rescue: int = 0
    sigma_psf_used: float = 0.0
    r_squared: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    sigma_pool: list[float] = field(default_factory=list)


def reconstruct_one_class(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    other_skel: np.ndarray | None,
    vessel_type: VesselType,
    cfg: ReconstructCfg,
    recon_hard: np.ndarray,
    recon_soft: np.ndarray,
) -> ReconstructionStats:
    """Reconstruct ONE A/V class (artery or vein) into the passed recon_hard/soft buffers.

    `other_skel` is the skeleton of the OTHER class (used to detect A/V crossings
    and skip samples where the two classes' skeletons run close together).
    """
    stats = ReconstructionStats()
    label = 1 if vessel_type == "artery" else 2

    green = image_bgr[:, :, 1].astype(np.float32, copy=False)

    skel = skeletonize_mask(mask, cfg)
    if not skel.any():
        return stats

    junc = junction_mask(skel)
    junc_dist = distance_to_mask(junc)
    if other_skel is not None and other_skel.any():
        cross_dist = distance_to_mask(other_skel.astype(bool))
    else:
        cross_dist = np.full(skel.shape, np.inf, dtype=np.float32)

    # (Previously we computed distance_transform_edt(mask) here to seed a
    # fallback width — that was a scientific-claim leak since the EDT
    # encodes the GT mask's width. All width estimation is now image-derived
    # via FWHM init + Phase-2 Gaussian fallback.)

    segments = trace_segments(skel.astype(bool), min_length_px=cfg.skeleton_min_length_px)
    stats.n_segments = len(segments)

    # --- Phase 1: run standard convolved-step fit on every sample. Samples
    # that fail are deferred to Phase 2, which uses a Gaussian thin-vessel
    # model with a PSF σ calibrated from the pool of Phase-1 successes.
    # Samples that were skipped (junction/crossing neighbourhoods) are kept
    # in `skipped` for Phase 3 interpolation.
    deferred: list[tuple] = []
    # Per-segment painted records: seg_idx -> list of (arc_s, width, cy, cx,
    # tangent, perp). Used by Phase 3 to interpolate across gaps.
    painted_per_seg: dict[int, list[tuple]] = {}
    # Per-segment skipped records awaiting Phase 3: seg_idx -> list of
    # (arc_s, cy, cx, tangent, perp).
    skipped_per_seg: dict[int, list[tuple]] = {}
    # Deferred records also need to know their segment so Phase 3 can pick
    # them up if the Gaussian rescue also fails.
    # deferred entries are extended with (seg_idx, arc_s).

    tangent_half = 0.5 * cfg.sample_stride_px + 0.5

    for seg_idx, seg in enumerate(segments):
        pts = seg.astype(float)
        diffs = np.diff(pts, axis=0)
        seg_arc = np.concatenate(([0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))))
        total_arc = float(seg_arc[-1])
        if total_arc < cfg.skeleton_min_length_px:
            continue
        painted_per_seg[seg_idx] = []
        skipped_per_seg[seg_idx] = []

        sample_positions = np.arange(0.0, total_arc, cfg.sample_stride_px)
        for s in sample_positions:
            j = int(np.searchsorted(seg_arc, s))
            if j >= len(pts):
                j = len(pts) - 1
            y_i, x_i = int(pts[j, 0]), int(pts[j, 1])
            stats.n_sample_attempts += 1

            tangent = tangent_pca(pts, j, window=7)
            perp = perpendicular(tangent)
            cy, cx = float(pts[j, 0]), float(pts[j, 1])

            # Samples near junctions or class crossings have corrupted profiles.
            # Record them for Phase-3 interpolation rather than painting now.
            if junc_dist[y_i, x_i] < cfg.bifurcation_skip_px:
                stats.n_sample_skipped_junction += 1
                skipped_per_seg[seg_idx].append((float(s), cy, cx, tangent.copy(), perp.copy()))
                continue
            if cross_dist[y_i, x_i] < cfg.crossing_skip_px:
                stats.n_sample_skipped_crossing += 1
                skipped_per_seg[seg_idx].append((float(s), cy, cx, tangent.copy(), perp.copy()))
                continue

            t, profile = extract_profile(
                green, (cy, cx), perp,
                half_width_px=cfg.half_width_px,
                step_px=cfg.profile_step_px,
                order=cfg.profile_interp_order,
            )
            w_fwhm = fwhm_width_from_profile(t, profile, min_depth=cfg.fwhm_min_depth)
            w0 = w_fwhm if w_fwhm is not None else 3.0
            fit = fit_vessel_profile(t, profile, vessel_type, w0, cfg.profile_fit)

            denom = float(np.var(profile) * len(profile)) + 1e-9
            r2 = 1.0 - float(np.sum(fit.residual**2)) / denom

            fit_is_good = fit.success and fit.rmse_rel <= 0.25 and r2 >= 0.3
            if fit_is_good:
                stats.n_sample_fits_ok += 1
                stats.r_squared.append(r2)
                stats.widths.append(fit.w)
                if fit.w >= 3.0 and r2 >= 0.7:
                    stats.sigma_pool.append(fit.sigma)
                c_offset = float(np.clip(fit.c, -1.0, 1.0))
                paint_y = cy + c_offset * perp[0]
                paint_x = cx + c_offset * perp[1]
                _paint_fit(
                    recon_hard, recon_soft,
                    (paint_y, paint_x), tangent, perp,
                    fit, label, tangent_half,
                    threshold_frac=cfg.paint_threshold_frac,
                )
                painted_per_seg[seg_idx].append(
                    (float(s), float(fit.w), paint_y, paint_x, tangent.copy(), perp.copy())
                )
            else:
                stats.n_sample_fits_failed += 1
                deferred.append(
                    (seg_idx, float(s), y_i, x_i, cy, cx,
                     tangent.copy(), perp.copy(), t, profile)
                )

    # --- Phase 2: Gaussian fallback for deferred samples ----------------
    # PSF σ is the LOWER BOUND of the fitted σ distribution (by physics:
    # σ_fit² = σ_PSF² + σ_vessel_width²). So take a low percentile, not
    # the median. Clamp to the plausible CFP range [0.3, 1.5] px to guard
    # against the convolved-step fitter trading w↔σ on thick vessels and
    # returning inflated σ.
    if len(stats.sigma_pool) >= 20:
        raw = float(np.percentile(stats.sigma_pool, 10.0))
        sigma_psf = float(np.clip(raw, 0.3, 1.5))
    else:
        sigma_psf = DEFAULT_SIGMA_PSF_PX
    stats.sigma_psf_used = sigma_psf

    for (seg_idx, arc_s, y_i, x_i, cy, cx, tangent, perp, t_arr, profile) in deferred:
        g_fit = fit_thin_vessel_gaussian(t_arr, profile, sigma_psf, min_depth=1.0)
        if g_fit is not None and g_fit.success and g_fit.w >= 0.8:
            stats.n_gaussian_rescue += 1
            stats.widths.append(g_fit.w)
            c_offset = float(np.clip(g_fit.c, -1.0, 1.0))
            paint_y = cy + c_offset * perp[0]
            paint_x = cx + c_offset * perp[1]
            _paint_fit(
                recon_hard, recon_soft,
                (paint_y, paint_x), tangent, perp,
                g_fit, label, tangent_half,
            )
            if seg_idx in painted_per_seg:
                painted_per_seg[seg_idx].append(
                    (arc_s, float(g_fit.w), paint_y, paint_x, tangent.copy(), perp.copy())
                )
        else:
            # No image-based width available — defer to Phase-3 interpolation.
            if seg_idx in skipped_per_seg:
                skipped_per_seg[seg_idx].append((arc_s, cy, cx, tangent.copy(), perp.copy()))

    # --- Phase 3: interpolate widths across skipped / unrescued samples
    # within each segment, then paint. Uses only nearest-neighbour painted
    # widths along the segment arc — no GT mask info.
    _phase3_fill(
        recon_hard, recon_soft,
        painted_per_seg, skipped_per_seg,
        label, tangent_half, stats,
        threshold_frac=cfg.paint_threshold_frac,
    )

    return stats


def _phase3_fill(
    recon_hard: np.ndarray,
    recon_soft: np.ndarray,
    painted_per_seg: dict[int, list[tuple]],
    skipped_per_seg: dict[int, list[tuple]],
    label: int,
    tangent_half: float,
    stats: ReconstructionStats,
    threshold_frac: float = 0.5,
) -> None:
    """For every skipped sample in a segment, linearly interpolate width from
    the two nearest painted neighbours along the segment's arc, then paint
    an oriented rectangle at the skipped location.
    """
    for seg_idx, skipped_records in skipped_per_seg.items():
        if not skipped_records:
            continue
        painted = painted_per_seg.get(seg_idx, [])
        if not painted:
            # No anchor on this segment — fall back to 1.5-px paint.
            for (arc_s, cy, cx, tangent, perp) in skipped_records:
                stats.n_sample_fallback_thin += 1
                _paint_fit(
                    recon_hard, recon_soft,
                    (cy, cx), tangent, perp,
                    _SyntheticFit.from_width(1.5),
                    label, tangent_half,
                    threshold_frac=threshold_frac,
                )
            continue
        # Sort anchors by arc for fast neighbour lookup
        painted_sorted = sorted(painted, key=lambda r: r[0])
        arcs = [r[0] for r in painted_sorted]
        widths = [r[1] for r in painted_sorted]
        for (arc_s, cy, cx, tangent, perp) in skipped_records:
            w_interp = _interp_width(arcs, widths, arc_s)
            if w_interp is None:
                continue
            stats.n_gaussian_rescue += 0
            stats.widths.append(w_interp)
            # No sub-pixel c information for interpolated samples — paint at
            # the skeleton pixel. Junction samples tend to be near the true
            # vessel centre anyway; boundary error is dominated by width.
            _paint_fit(
                recon_hard, recon_soft,
                (cy, cx), tangent, perp,
                _SyntheticFit.from_width(w_interp),
                label, tangent_half,
                threshold_frac=threshold_frac,
            )


def _disc_paint_junctions(
    recon_hard: np.ndarray,
    skel: np.ndarray,
    mask: np.ndarray,
    label: int,
    search_px: float = 8.0,
    scale: float = 0.7,
) -> None:
    """At every junction (3+ neighbour skeleton pixel), paint a disc of
    radius = max(reconstructed widths) within `search_px` of the node.

    Width is recovered from the already-painted `recon_hard` by taking the
    local run of painted-label pixels perpendicular to a nominal direction.
    Here we cheat slightly by using the maximum inscribed radius at each
    junction: the distance transform of `recon_hard != label` at the
    junction pixel gives the half-width of the widest painted vessel
    passing through it. That distance is an image-derived upper bound
    on what the local vessel should be.
    """
    junc = junction_mask(skel)
    if not junc.any():
        return
    # Distance from each pixel to the "not-this-label" region in recon_hard.
    # Where the junction sits in painted vessel, this tells us the current
    # inscribed radius; we then paint a disc of that radius to close any
    # pinch.
    inside = (recon_hard == label)
    if not inside.any():
        return
    dt = distance_transform_edt(inside).astype(np.float32)
    js, xs = np.where(junc)
    for y, x in zip(js, xs):
        # Sample a small neighbourhood and take the max DT — this is the
        # widest painted vessel near the junction.
        y0 = max(0, int(y) - int(search_px))
        y1 = min(skel.shape[0], int(y) + int(search_px) + 1)
        x0 = max(0, int(x) - int(search_px))
        x1 = min(skel.shape[1], int(x) + int(search_px) + 1)
        patch = dt[y0:y1, x0:x1]
        r = float(patch.max()) * scale
        if r < 1.0:
            continue  # nothing painted near this junction, skip
        cv2.circle(recon_hard, (int(x), int(y)), int(round(r)), label, thickness=-1)


def _interp_width(arcs: list[float], widths: list[float], target_s: float) -> float | None:
    """Linear interpolation of width at arc-position target_s; clamp to nearest
    anchor at the segment ends."""
    if not arcs:
        return None
    if target_s <= arcs[0]:
        return widths[0]
    if target_s >= arcs[-1]:
        return widths[-1]
    # bisect
    import bisect
    i = bisect.bisect_right(arcs, target_s)
    a_lo, a_hi = arcs[i - 1], arcs[i]
    w_lo, w_hi = widths[i - 1], widths[i]
    if a_hi - a_lo <= 1e-6:
        return 0.5 * (w_lo + w_hi)
    frac = (target_s - a_lo) / (a_hi - a_lo)
    return float(w_lo * (1.0 - frac) + w_hi * frac)


class _SyntheticFit:
    """Trivial stand-in for ProfileFit used by the thin-vessel fallback."""

    def __init__(self, w: float):
        self.w = float(max(w, 1.0))
        self.sigma = 0.5
        self.c = 0.0
        self.B = 128.0
        self.A = 64.0

    @classmethod
    def from_width(cls, w: float) -> "_SyntheticFit":
        return cls(w)


# --- top-level reconstruction ------------------------------------------


@dataclass
class Reconstruction:
    recon_hard: np.ndarray           # (H, W) uint8; 0=bg, 1=artery, 2=vein, 3=crossing
    recon_soft: np.ndarray           # (H, W, 2) float32 — (artery_soft, vein_soft)
    artery_skeleton: np.ndarray      # (H, W) bool
    vein_skeleton: np.ndarray        # (H, W) bool
    artery_stats: ReconstructionStats
    vein_stats: ReconstructionStats


def reconstruct_from_gt(
    image_bgr: np.ndarray,
    artery_mask: np.ndarray,
    vein_mask: np.ndarray,
    cfg: ReconstructCfg | None = None,
) -> Reconstruction:
    """End-to-end: skeletonise each GT mask, fit, paint reconstruction.

    The reconstruction uses ONLY the skeleton (centerline topology) and the raw
    image — the GT binary extents are not consulted during fitting.
    """
    cfg = cfg or ReconstructCfg()
    h, w = artery_mask.shape
    recon_hard = np.zeros((h, w), dtype=np.uint8)
    recon_soft = np.zeros((h, w, 2), dtype=np.float32)

    art_skel = skeletonize_mask(artery_mask, cfg)
    vn_skel = skeletonize_mask(vein_mask, cfg)

    artery_stats = reconstruct_one_class(
        image_bgr, artery_mask, vn_skel, "artery", cfg, recon_hard, recon_soft
    )
    vein_stats = reconstruct_one_class(
        image_bgr, vein_mask, art_skel, "vein", cfg, recon_hard, recon_soft
    )

    if cfg.junction_disc_enabled:
        # Disc-paint at junction nodes to close the "pinch" where the
        # perpendicular-profile approach is degenerate. Scale factor <1
        # guards against overshoot in tapered regions.
        _disc_paint_junctions(recon_hard, art_skel, artery_mask, label=1,
                              search_px=cfg.junction_disc_search_px,
                              scale=cfg.junction_disc_scale)
        _disc_paint_junctions(recon_hard, vn_skel, vein_mask, label=2,
                              search_px=cfg.junction_disc_search_px,
                              scale=cfg.junction_disc_scale)

    # Post-closing: morphological close each class's hard mask to fill residual
    # gaps (<= closing_radius px) left by skipped junction/crossing samples.
    if cfg.post_closing_radius > 0:
        from skimage.morphology import closing as _sk_closing
        se = disk(cfg.post_closing_radius)
        art_hard = _sk_closing(recon_hard == 1, se)
        vn_hard = _sk_closing(recon_hard == 2, se)
        recon_hard = np.zeros_like(recon_hard)
        recon_hard[art_hard] = 1
        recon_hard[vn_hard] = 2

    # Crossings: where both soft channels are significant
    both = (recon_soft[:, :, 0] > 0.25) & (recon_soft[:, :, 1] > 0.25)
    recon_hard[both] = 3

    return Reconstruction(
        recon_hard=recon_hard,
        recon_soft=recon_soft,
        artery_skeleton=art_skel.astype(bool),
        vein_skeleton=vn_skel.astype(bool),
        artery_stats=artery_stats,
        vein_stats=vein_stats,
    )
