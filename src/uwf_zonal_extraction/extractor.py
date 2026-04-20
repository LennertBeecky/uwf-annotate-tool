"""ZonalDiameterExtractor — orchestrates the full pipeline.

Two public entry points:
  - `extract_caliber(image, av_mask, av_probs, od, fovea, config)`:
      scientific core, mask-in. No ONNX dependency. Used in tests.
  - `extract_caliber_from_image(image_path, config, bundle)`:
      end-to-end. Requires a SegmentationBundle (ONNX).

v1 wires together dewarp -> segmentation -> coords -> skeleton ->
profile fit -> Knudtson -> bootstrap-CI. Some pieces (fovea detection,
radial sigma, longitudinal matching, viz) are stubbed; see the
individual modules for TODOs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from uwf_zonal_extraction.config import ExtractionConfig
from uwf_zonal_extraction.coordinate_system import RetinalCoordinateSystem, Quadrant
from uwf_zonal_extraction.dewarp import dewarp_optos
from uwf_zonal_extraction.knudtson import aggregate_knudtson
from uwf_zonal_extraction.models import (
    DebugArtefacts,
    ExtractionResult,
    VesselMeasurement,
    VesselSegment,
    ZonalResult,
)
from uwf_zonal_extraction.profile_fitting import (
    ProfileFit,
    extract_profile,
    fit_vessel_profile,
)
from uwf_zonal_extraction.segmentation import SegmentationBundle
from uwf_zonal_extraction.segmentation.fovea import (
    detect_fovea_from_probs,
    load_sidecar_fovea,
)
from uwf_zonal_extraction.skeleton import build_vessel_segments
from uwf_zonal_extraction.utils import perpendicular, tangent_pca, weighted_median


# --- scientific core -----------------------------------------------------


def extract_caliber(
    image_bgr: np.ndarray,
    av_mask: np.ndarray,
    av_probs: np.ndarray,
    od_center: tuple[float, float],
    od_radius_px: float,
    fovea_center: Optional[tuple[float, float]],
    fovea_confidence: float,
    config: ExtractionConfig,
    laterality: str = "OD",
    image_id: str = "image",
    mm_per_px_map: Optional[np.ndarray] = None,
    dewarp_applied: bool = False,
    run_metadata: Optional[dict] = None,
    retina_mask: Optional[np.ndarray] = None,
    return_debug: bool = False,
) -> ExtractionResult:
    """Mask-in scientific core. No ONNX dependency."""

    _validate_inputs(image_bgr, av_mask, av_probs, od_radius_px)
    # Cast green to float32 ONCE. extract_profile is in a hot loop and will
    # refuse uint8 input — casting inside it copies the full 4000×4000 array
    # per sample (~128 MB each, >600 GB cumulative).
    green = image_bgr[:, :, 1].astype(np.float32, copy=False)

    coords = RetinalCoordinateSystem.from_landmarks(
        od_center=od_center,
        od_radius_px=od_radius_px,
        fovea_center=fovea_center,
        config=config,
        laterality=laterality,
    )

    # --- 1. Skeleton + segments ---------------------------------------
    if return_debug:
        segments, skeleton_map, distance_map = build_vessel_segments(
            av_mask, av_probs, coords, config, return_maps=True
        )
    else:
        segments = build_vessel_segments(av_mask, av_probs, coords, config)
        skeleton_map = distance_map = None

    # --- 2. Per-cell sampling and profile fitting ---------------------
    funnel: dict[tuple[int, str], dict[str, int]] = {}
    measurements = _measure_all(green, segments, coords, config, funnel)

    # --- 3. DD / μm conversion ----------------------------------------
    _convert_units(measurements, segments, coords, config, mm_per_px_map)

    # --- 4. Knudtson per cell -----------------------------------------
    zonal = _aggregate_zonal(measurements, config, dewarp_applied=dewarp_applied)

    meta = dict(run_metadata) if run_metadata else {}
    meta.setdefault("config", config.to_dict())
    meta.setdefault("dewarp_applied", dewarp_applied)
    meta.setdefault("package_version", _package_version())
    meta.setdefault("numpy_version", np.__version__)
    try:
        import scipy
        meta.setdefault("scipy_version", scipy.__version__)
    except ImportError:  # pragma: no cover
        pass
    try:
        import skimage
        meta.setdefault("scikit_image_version", skimage.__version__)
    except ImportError:  # pragma: no cover
        pass

    debug = None
    if return_debug:
        n_total = len(segments)
        n_uncertain = sum(1 for s in segments if s.uncertain)
        n_nonuncertain = n_total - n_uncertain
        n_measurements = len(measurements)
        overall_funnel = {
            "segments_total": n_total,
            "segments_uncertain": n_uncertain,
            "segments_nonuncertain": n_nonuncertain,
            "measurements_accepted": n_measurements,
        }
        debug = DebugArtefacts(
            image_bgr=image_bgr,
            av_mask=av_mask,
            av_probs=av_probs,
            retina_mask=retina_mask
            if retina_mask is not None
            else np.ones_like(av_mask, dtype=np.uint8),
            skeleton=skeleton_map if skeleton_map is not None else np.zeros_like(av_mask, dtype=bool),
            distance_map=distance_map
            if distance_map is not None
            else np.zeros_like(av_mask, dtype=np.float32),
            funnel=overall_funnel,
            cell_funnel=funnel,
        )

    return ExtractionResult(
        image_id=image_id,
        laterality=laterality,
        segments=segments,
        measurements=measurements,
        zonal=zonal,
        od_center=od_center,
        od_radius_px=od_radius_px,
        fovea_center=fovea_center,
        fovea_confidence=fovea_confidence,
        dewarp_device=config.dewarp.device,
        axial_length_mm=config.dewarp.axial_length_mm_default,
        mm_per_px_map=mm_per_px_map,
        run_metadata=meta,
        debug=debug,
    )


# --- end-to-end entry point ---------------------------------------------


def extract_caliber_from_image(
    image_path: str | Path,
    config: Optional[ExtractionConfig] = None,
    bundle: Optional[SegmentationBundle] = None,
    laterality: Optional[str] = None,
    return_debug: bool = False,
) -> ExtractionResult:
    """Image-in entry point. Needs a SegmentationBundle."""
    config = config or ExtractionConfig()
    if bundle is None:
        bundle = SegmentationBundle.from_model_dir("models")

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # --- Dewarp --------------------------------------------------------
    dewarp_res = dewarp_optos(image_bgr, config.dewarp)
    image_for_seg = dewarp_res.image

    # --- Segmentation --------------------------------------------------
    seg = bundle.run(image_for_seg)

    # --- Fovea ---------------------------------------------------------
    fov = load_sidecar_fovea(image_path)
    fov_conf = 1.0 if fov is not None else 0.0
    if fov is None:
        fov, fov_conf = detect_fovea_from_probs(
            seg.av_probs,
            seg.od.center,
            seg.od.radius_px,
            laterality=laterality or config.laterality_default,
        )

    run_metadata = {
        "image_path": str(image_path),
        "lunet_sha256": _file_sha256(bundle.lunet.model_path),
        "odc_sha256": _file_sha256(bundle.odc.model_path),
        "dewarp_applied": dewarp_res.applied,
        "dewarp_device": dewarp_res.device,
        "axial_length_mm": dewarp_res.axial_length_mm,
    }

    return extract_caliber(
        image_bgr=image_for_seg,
        av_mask=seg.av_mask,
        av_probs=seg.av_probs,
        od_center=seg.od.center,
        od_radius_px=seg.od.radius_px,
        fovea_center=fov,
        fovea_confidence=fov_conf,
        config=config,
        laterality=laterality or config.laterality_default,
        image_id=image_path.stem,
        mm_per_px_map=dewarp_res.mm_per_px,
        dewarp_applied=dewarp_res.applied,
        run_metadata=run_metadata,
        retina_mask=seg.retina_mask,
        return_debug=return_debug,
    )


# --- internals -----------------------------------------------------------


def _measure_all(
    green: np.ndarray,
    segments: list[VesselSegment],
    coords: RetinalCoordinateSystem,
    config: ExtractionConfig,
    funnel: Optional[dict[tuple[int, str], dict[str, int]]] = None,
) -> list[VesselMeasurement]:
    """Walk every sub-segment, fit profiles, aggregate → VesselMeasurement list.

    If `funnel` is given, per-(zone, quadrant) stage counts are recorded:
      n_subsegments        — count of sub-segments in this cell
      n_subseg_nonempty    — after exclusion mask filter (had sample positions)
      n_fits_accepted      — profile fits that passed RMSE_rel threshold
      n_measurements       — cells with >=1 accepted fit (always 0 or 1 per sub-segment)
    """
    out: list[VesselMeasurement] = []
    cfg = config.profile_fit
    track = funnel is not None

    def _bump(cell, key):
        if not track:
            return
        funnel.setdefault(cell, {}).setdefault(key, 0)
        funnel[cell][key] += 1

    for seg in tqdm(segments, desc="measuring segments", leave=False):
        if seg.uncertain:
            continue
        dd_px = coords.dd_px
        for (zone, quad), idxs in seg.sub_segments.items():
            cell = (int(zone), str(quad))
            _bump(cell, "n_subsegments")

            fits, sample_pts, short_arc = _sample_and_fit(
                green, seg, idxs, cfg, dd_px, return_short_flag=True
            )
            if sample_pts is not None and len(sample_pts) > 0:
                _bump(cell, "n_subseg_nonempty")
            if not fits:
                continue
            _bump(cell, "n_fits_accepted")
            _bump(cell, "n_measurements")
            if short_arc and len(fits) == 1:
                _bump(cell, "n_midpoint_fallback")

            diameters = np.array([f.w for f in fits], dtype=float)
            sigmas = np.array([f.sigma_w for f in fits], dtype=float)
            rmse_rel = np.array([f.rmse_rel for f in fits], dtype=float)

            weights = np.where(sigmas > 0, 1.0 / np.maximum(sigmas**2, 1e-12), 1.0)
            med = weighted_median(diameters, weights)
            arc_dd = _arc_length_dd(seg.full_points, idxs, dd_px)

            # Tortuosity = arc / chord
            pts = seg.full_points[idxs]
            chord = float(np.hypot(*(pts[-1] - pts[0]))) if len(pts) > 1 else 1.0
            tort = float(arc_dd * dd_px / max(chord, 1e-6))

            resolution_limited = med < 3.0

            out.append(
                VesselMeasurement(
                    vessel_id=seg.segment_id,
                    vessel_type=seg.vessel_type,
                    zone_index=int(zone),
                    quadrant=quad,  # type: ignore[arg-type]
                    median_diameter_px=float(med),
                    diameters_px=diameters,
                    sigma_w_px=sigmas,
                    rmse_rel=rmse_rel,
                    arc_length_dd=float(arc_dd),
                    n_samples=len(fits),
                    reflex_model=fits[0].reflex_model,
                    sample_points=np.asarray(sample_pts, dtype=float),
                    example_fit=fits[len(fits) // 2],  # midpoint — representative
                    tortuosity=tort,
                    resolution_limited=resolution_limited,
                )
            )
    return out


def _sample_and_fit(
    green: np.ndarray,
    seg: VesselSegment,
    idxs: list[int],
    cfg,
    dd_px: float,
    return_points: bool = False,
    return_short_flag: bool = False,
):
    """Walk the sub-segment at regular arc-length steps and fit each profile.

    Returns:
        default:             list[ProfileFit]
        return_points=True:  (fits, sample_points_accepted_list)
        return_short_flag=True: (fits, sample_points_accepted_list, short_arc_bool)
    """
    # Arc length along full_points
    pts = seg.full_points
    full_arc = np.zeros(len(pts), dtype=float)
    for i in range(1, len(pts)):
        full_arc[i] = full_arc[i - 1] + np.hypot(*(pts[i] - pts[i - 1]))

    sub_arc = full_arc[idxs] - full_arc[idxs[0]]
    step_px = max(cfg.sample_interval_dd * dd_px, cfg.min_sample_spacing_px)

    # End buffer scales with local vessel width (panel recommendation): junction
    # artefacts extend ~1.5·w from a bifurcation, regardless of DD or step_px.
    local_w = float(np.mean(seg.w_init[idxs]))
    end_buffer = max(cfg.min_sample_spacing_px, cfg.end_buffer_w_factor * local_w)

    arc_len = float(sub_arc[-1])
    if arc_len < cfg.min_sample_spacing_px:
        # Genuinely too short — below pixel resolution.
        if return_short_flag:
            return [], [], False
        if return_points:
            return [], []
        return []
    short_arc = arc_len <= 2 * end_buffer
    if short_arc:
        # Short arc: take ONE sample at the midpoint (as far from both junction
        # ends as possible). The inverse-variance weight from the LM fit
        # naturally downweights this single-sample measurement in the zonal
        # median against vessels that have many samples.
        sample_positions = np.array([arc_len / 2.0])
    else:
        sample_positions = np.arange(end_buffer, arc_len - end_buffer + 1e-6, step_px)

    fits: list[ProfileFit] = []
    accepted_points: list[tuple[float, float]] = []
    for pos in sample_positions:
        j_local = int(np.searchsorted(sub_arc, pos))
        if j_local >= len(idxs):
            continue
        i_full = idxs[j_local]
        # For short arcs the midpoint may fall inside the skeleton-level
        # exclusion buffer around the nearby bifurcations — if we apply the
        # mask, nothing ever passes. Bypass the mask in that specific case;
        # the midpoint is still the best-available measurement and the LM
        # fit's covariance will reflect any junction-induced residual.
        if seg.exclusion_mask[i_full] and not short_arc:
            continue

        tangent = tangent_pca(pts.astype(float), i_full, window=7)
        perp = perpendicular(tangent)
        w_init = max(float(seg.w_init[i_full]), 1.0)
        # Half-width: fixed generous window unless half_width_init_factor > 0.
        # A fixed window guarantees the outer 25% of every profile is pure
        # tissue for baseline estimation, which matters more for thick veins
        # than adapting to a (possibly underestimated) mask-derived w_init.
        if cfg.half_width_init_factor > 0:
            half_w = max(cfg.half_width_min_px, int(np.ceil(cfg.half_width_init_factor * w_init)))
        else:
            half_w = cfg.half_width_min_px

        t, profile = extract_profile(
            green,
            (float(pts[i_full, 0]), float(pts[i_full, 1])),
            perp,
            half_width_px=half_w,
            step_px=cfg.profile_sampling_step_px,
            order=cfg.profile_interp_order,
        )
        fit = fit_vessel_profile(
            t, profile, seg.vessel_type, w_init, cfg, sigma_fixed=None
        )
        if fit.success and fit.rmse_rel < cfg.rmse_rel_accept:
            fits.append(fit)
            accepted_points.append((float(pts[i_full, 0]), float(pts[i_full, 1])))
    if return_short_flag:
        return fits, accepted_points, short_arc
    if return_points:
        return fits, accepted_points
    return fits


def _arc_length_dd(full_points: np.ndarray, idxs: list[int], dd_px: float) -> float:
    pts = full_points[idxs].astype(float)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])) / max(dd_px, 1e-6))


def _convert_units(
    measurements: list[VesselMeasurement],
    segments: list[VesselSegment],
    coords: RetinalCoordinateSystem,
    config: ExtractionConfig,
    mm_per_px_map: Optional[np.ndarray],
) -> None:
    """Populate DD and μm columns.

    When `mm_per_px_map` is provided (post-dewarp), sample it at each
    vessel's mean (y, x). Otherwise fall back to the population-DD-derived
    scalar (`population_dd_um / dd_px`).
    """
    dd_px = coords.dd_px
    fallback_um_per_px = config.population_dd_um / max(dd_px, 1e-6)

    segments_by_id = {s.segment_id: s for s in segments}

    for m in measurements:
        m.median_diameter_dd = m.median_diameter_px / max(dd_px, 1e-6)

        if mm_per_px_map is not None:
            seg = segments_by_id.get(m.vessel_id)
            if seg is not None and (m.zone_index, m.quadrant) in seg.sub_segments:
                idxs = seg.sub_segments[(m.zone_index, m.quadrant)]
                pts = seg.full_points[idxs]
                y = int(np.clip(pts[:, 0].mean(), 0, mm_per_px_map.shape[0] - 1))
                x = int(np.clip(pts[:, 1].mean(), 0, mm_per_px_map.shape[1] - 1))
                um_per_px = float(mm_per_px_map[y, x] * 1000.0)
            else:
                um_per_px = fallback_um_per_px
        else:
            um_per_px = fallback_um_per_px

        m.median_diameter_um = m.median_diameter_px * um_per_px
        m.diameters_um = m.diameters_px * um_per_px


# --- Knudtson aggregation ----------------------------------------------


def _aggregate_zonal(
    measurements: list[VesselMeasurement],
    config: ExtractionConfig,
    dewarp_applied: bool = False,
) -> list[ZonalResult]:
    zonal: list[ZonalResult] = []
    # Flag peripheral zones when dewarp was skipped. Any zone whose outer
    # boundary is >= 3 DD is sensitive to stereographic distortion.
    dewarp_flag_zones = set()
    if not dewarp_applied:
        for z_idx, (_, outer) in enumerate(config.zones_dd):
            if outer >= 3.0:
                dewarp_flag_zones.add(z_idx)

    # Enumerate all cells present in the zone table × 4 quadrants
    for z_idx, _ in enumerate(config.zones_dd):
        if z_idx == 0:  # Z0 is excluded
            continue
        for quad in ("ST", "SN", "IT", "IN"):
            arts = [
                m.median_diameter_px
                for m in measurements
                if m.zone_index == z_idx and m.quadrant == quad
                and m.vessel_type == "artery" and not m.resolution_limited
            ]
            vns = [
                m.median_diameter_px
                for m in measurements
                if m.zone_index == z_idx and m.quadrant == quad
                and m.vessel_type == "vein" and not m.resolution_limited
            ]
            crae, n_a, flags_a = aggregate_knudtson(arts, "artery", config.knudtson)
            crve, n_v, flags_v = aggregate_knudtson(vns, "vein", config.knudtson)
            flags = set(flags_a) | set(flags_v)
            if n_a == 0:
                flags.add("no_arteries")
            if n_v == 0:
                flags.add("no_veins")
            if z_idx in dewarp_flag_zones:
                flags.add("dewarp_skipped")

            zonal.append(
                ZonalResult(
                    zone_index=z_idx,
                    quadrant=quad,  # type: ignore[arg-type]
                    crae_px=crae,
                    crve_px=crve,
                    avr=(crae / crve) if (crae and crve) else None,
                    n_arteries=n_a,
                    n_veins=n_v,
                    artery_diameters_px=list(arts),
                    vein_diameters_px=list(vns),
                    flags=flags,
                )
            )
    return zonal


# --- validation / metadata helpers --------------------------------------


def _validate_inputs(
    image_bgr: np.ndarray,
    av_mask: np.ndarray,
    av_probs: np.ndarray,
    od_radius_px: float,
) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError(
            f"image_bgr must be a 3-channel BGR image, got shape {image_bgr.shape}"
        )
    if av_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"av_mask shape {av_mask.shape} does not match image {image_bgr.shape[:2]}"
        )
    if av_probs.shape[:2] != image_bgr.shape[:2] or av_probs.shape[2] < 2:
        raise ValueError(
            f"av_probs must be (H, W, ≥2), got {av_probs.shape} for image {image_bgr.shape[:2]}"
        )
    if od_radius_px <= 0:
        raise ValueError(
            f"od_radius_px must be positive (got {od_radius_px}); "
            "OD segmentation likely failed. Provide OD landmarks via sidecar."
        )
    h = image_bgr.shape[0]
    min_reasonable = max(10.0, 0.005 * h)  # 0.5% of image height
    if od_radius_px < min_reasonable:
        raise ValueError(
            f"od_radius_px={od_radius_px:.1f} is implausibly small for an "
            f"image of height {h}; minimum plausible is ~{min_reasonable:.0f} px."
        )


def _package_version() -> str:
    try:
        from uwf_zonal_extraction import __version__
        return __version__
    except ImportError:  # pragma: no cover
        return "unknown"


def _file_sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
