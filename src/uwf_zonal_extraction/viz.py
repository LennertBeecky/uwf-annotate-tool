"""Quality-control visualisations.

Two entry points:
  - `overlay_zones_and_skeleton(...)` : quick one-shot overlay (no matplotlib)
  - `save_debug_pack(result, output_dir)` : render the full debug artefact set

matplotlib is imported lazily inside functions so the core package still
imports without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np

from uwf_zonal_extraction.coordinate_system import RetinalCoordinateSystem
from uwf_zonal_extraction.models import ExtractionResult

# BGR colour conventions (matches legacy hybrid_pipeline.py)
COLOUR_ARTERY = (0, 0, 255)        # red
COLOUR_VEIN = (255, 0, 0)          # blue
COLOUR_ZONE_RING = (0, 255, 255)   # yellow
COLOUR_OD = (0, 255, 0)            # green
COLOUR_FOVEA = (255, 255, 0)       # cyan
COLOUR_AXIS = (255, 255, 0)        # cyan
COLOUR_SAMPLE_ACCEPT = (0, 255, 0)
COLOUR_SAMPLE_REJECT = (0, 0, 255)


# --- basic helpers ----------------------------------------------------


def _thickness_for(image: np.ndarray) -> int:
    return max(1, int(min(image.shape[:2]) / 800))


def overlay_zones_and_skeleton(
    image_bgr: np.ndarray,
    coords: RetinalCoordinateSystem,
    skeleton: np.ndarray | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Render zone rings + OD-fovea axis + optional skeleton."""
    vis = image_bgr.copy()
    t = _thickness_for(vis)
    cx, cy = coords.od_center
    dd = coords.dd_px
    for _inner, outer in coords.zones_dd:
        cv2.circle(vis, (int(cx), int(cy)), int(outer * dd), COLOUR_ZONE_RING, t)
    cv2.circle(vis, (int(cx), int(cy)), int(dd / 2), COLOUR_OD, t * 2)
    if coords.fovea_center is not None:
        fx, fy = coords.fovea_center
        cv2.line(vis, (int(cx), int(cy)), (int(fx), int(fy)), COLOUR_AXIS, t * 2)
        cv2.circle(vis, (int(fx), int(fy)), 6 * t, COLOUR_FOVEA, -1)
    if skeleton is not None:
        ys, xs = np.where(skeleton)
        vis[ys, xs] = [255, 255, 255]
    if output_path is not None:
        cv2.imwrite(str(output_path), vis)
    return vis


# --- debug pack -------------------------------------------------------


def save_debug_pack(
    result: ExtractionResult,
    output_dir: str | Path,
) -> Path:
    """Render a 10-artefact debug pack + data tables into `output_dir`.

    Requires `result.debug` to be populated (call extract_caliber* with
    `return_debug=True`).

    Returns the output directory path.
    """
    if result.debug is None:
        raise ValueError(
            "result.debug is None; re-run with return_debug=True to populate it."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dbg = result.debug
    image = dbg.image_bgr

    coords = RetinalCoordinateSystem.from_landmarks(
        od_center=result.od_center,
        od_radius_px=result.od_radius_px,
        fovea_center=result.fovea_center,
        config=_make_dummy_config(result),
        laterality=result.laterality,
    )

    _save_01_image(image, out)
    _save_02_av_overlay(image, dbg.av_mask, out)
    _save_03_retina_mask(image, dbg.retina_mask, out)
    _save_04_od_overlay(image, result, out)
    _save_05_zones(image, coords, out)
    _save_06_skeleton(image, dbg, result, out)
    _save_07_exclusion(image, dbg, result, out)
    _save_08_sample_points(image, result, out)
    _save_09_example_profiles(result, out)
    _save_10_funnel(result, dbg, out)
    _save_tables_and_metadata(result, out)

    return out


# --- individual renderers --------------------------------------------


def _save_01_image(image: np.ndarray, out: Path) -> None:
    cv2.imwrite(str(out / "01_image.jpg"), image)


def _save_02_av_overlay(image: np.ndarray, av_mask: np.ndarray, out: Path) -> None:
    overlay = image.copy()
    overlay[av_mask == 1] = COLOUR_ARTERY
    overlay[av_mask == 2] = COLOUR_VEIN
    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    cv2.imwrite(str(out / "02_av_overlay.jpg"), blended)


def _save_03_retina_mask(image: np.ndarray, retina_mask: np.ndarray, out: Path) -> None:
    tinted = image.copy()
    tinted[retina_mask == 0] = (tinted[retina_mask == 0] * 0.25).astype(np.uint8)
    cv2.imwrite(str(out / "03_retina_mask.jpg"), tinted)


def _save_04_od_overlay(image: np.ndarray, result: ExtractionResult, out: Path) -> None:
    vis = image.copy()
    t = _thickness_for(vis)
    cx, cy = result.od_center
    cv2.circle(vis, (int(cx), int(cy)), int(result.od_radius_px), COLOUR_OD, t * 2)
    cv2.drawMarker(
        vis, (int(cx), int(cy)), COLOUR_OD, cv2.MARKER_CROSS, markerSize=20 * t, thickness=t * 2
    )
    if result.fovea_center is not None:
        fx, fy = result.fovea_center
        cv2.drawMarker(
            vis, (int(fx), int(fy)), COLOUR_FOVEA,
            cv2.MARKER_TILTED_CROSS, markerSize=20 * t, thickness=t * 2,
        )
        cv2.line(vis, (int(cx), int(cy)), (int(fx), int(fy)), COLOUR_AXIS, t)
        cv2.putText(
            vis, f"fovea conf={result.fovea_confidence:.2f}",
            (int(fx) + 10, int(fy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, COLOUR_FOVEA, t,
        )
    cv2.imwrite(str(out / "04_od_overlay.jpg"), vis)


def _save_05_zones(image: np.ndarray, coords: RetinalCoordinateSystem, out: Path) -> None:
    vis = overlay_zones_and_skeleton(image, coords, skeleton=None)
    cv2.imwrite(str(out / "05_zones.jpg"), vis)


def _save_06_skeleton(
    image: np.ndarray, dbg, result: ExtractionResult, out: Path
) -> None:
    """Skeleton coloured by A/V label. Junctions + endpoints annotated."""
    vis = image.copy()
    # Paint by segment label
    for seg in result.segments:
        col = COLOUR_ARTERY if seg.vessel_type == "artery" else COLOUR_VEIN
        if seg.uncertain:
            col = (150, 150, 150)
        pts = seg.full_points
        ys, xs = pts[:, 0], pts[:, 1]
        inb = (ys >= 0) & (ys < vis.shape[0]) & (xs >= 0) & (xs < vis.shape[1])
        vis[ys[inb], xs[inb]] = col
    # Mark bifurcations + crossings
    for seg in result.segments:
        for i, ptype in enumerate(seg.point_types):
            if ptype in ("bifurcation", "crossing"):
                y, x = int(seg.full_points[i, 0]), int(seg.full_points[i, 1])
                colour = (0, 255, 255) if ptype == "bifurcation" else (255, 0, 255)
                cv2.circle(vis, (x, y), 3, colour, 1)
    cv2.imwrite(str(out / "06_skeleton.jpg"), vis)


def _save_07_exclusion(
    image: np.ndarray, dbg, result: ExtractionResult, out: Path
) -> None:
    """Render which skeleton points are INSIDE the bifurcation-exclusion buffer.

    Dark red = excluded (will not be sampled). Light green = available for sampling.
    """
    vis = image.copy()
    for seg in result.segments:
        if seg.uncertain:
            continue
        pts = seg.full_points
        for i in range(len(pts)):
            y, x = int(pts[i, 0]), int(pts[i, 1])
            if not (0 <= y < vis.shape[0] and 0 <= x < vis.shape[1]):
                continue
            vis[y, x] = (30, 30, 180) if seg.exclusion_mask[i] else (30, 220, 30)
    cv2.imwrite(str(out / "07_exclusion.jpg"), vis)


def _save_08_sample_points(
    image: np.ndarray, result: ExtractionResult, out: Path
) -> None:
    """Every ACCEPTED sample location, coloured by zone index."""
    vis = image.copy()
    t = _thickness_for(vis)
    # Blue gradient over zones
    zone_colours = {
        0: (120, 120, 120),
        1: (255, 180, 0),      # Z1 prominent cyan-ish
        2: (255, 120, 0),
        3: (255, 60, 0),
        4: (200, 0, 120),
        5: (140, 0, 180),
        6: (80, 0, 200),
        7: (0, 0, 200),
    }
    for m in result.measurements:
        if m.sample_points is None or len(m.sample_points) == 0:
            continue
        col = zone_colours.get(m.zone_index, (255, 255, 255))
        for y, x in m.sample_points:
            cv2.circle(vis, (int(x), int(y)), 2 * t, col, -1)
    cv2.imwrite(str(out / "08_sample_points.jpg"), vis)


def _save_09_example_profiles(result: ExtractionResult, out: Path) -> None:
    """12 example profile fits across zones/quadrants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Pick a spread: try to cover many (zone, quadrant, vessel_type) combos
    picks = []
    seen: set = set()
    # Priority: non-Z0 cells with fits
    for m in sorted(
        result.measurements, key=lambda mm: (mm.zone_index, mm.quadrant)
    ):
        if m.example_fit is None:
            continue
        key = (m.zone_index, m.quadrant, m.vessel_type)
        if key in seen:
            continue
        seen.add(key)
        picks.append(m)
        if len(picks) >= 12:
            break

    if not picks:
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, m in enumerate(picks):
        ax = axes[idx // 4, idx % 4]
        ax.set_visible(True)
        fit = m.example_fit
        ax.plot(fit.t, fit.profile, "k.", markersize=2, label="profile")
        # Evaluate the fit at fine resolution
        ax.plot(fit.t, fit.profile - fit.residual, "r-", linewidth=1.2, label="fit")
        ax.set_title(
            f"Z{m.zone_index} {m.quadrant} {m.vessel_type}\n"
            f"w={fit.w:.2f}px  sigma={fit.sigma:.2f}  RMSE/A={fit.rmse_rel:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("px from vessel centre")
        ax.set_ylabel("intensity")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Example profile fits — {result.image_id}")
    fig.tight_layout()
    fig.savefig(out / "09_example_profiles.png", dpi=130)
    plt.close(fig)


def _save_10_funnel(result: ExtractionResult, dbg, out: Path) -> None:
    """Bar chart + JSON of per-stage drop counts across zones × quadrants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None  # type: ignore[assignment]

    # Build a flat table from cell_funnel keyed on (zone, quadrant).
    # Include every key the extractor put in cell_funnel (n_midpoint_fallback
    # is recorded there; keeping this dynamic avoids viz/extractor drift).
    funnel = dbg.cell_funnel
    cells = sorted(funnel.keys(), key=lambda k: (k[0], k[1]))
    base_stages = ["n_subsegments", "n_subseg_nonempty", "n_fits_accepted", "n_measurements"]
    extra_keys = sorted({k for c in funnel.values() for k in c.keys() if k not in base_stages})
    stages = base_stages + extra_keys

    records = []
    for (z, q) in cells:
        row = {"zone": z, "quadrant": q}
        for s in stages:
            row[s] = int(funnel[(z, q)].get(s, 0))
        records.append(row)

    json_path = out / "10_funnel.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "overall": dbg.funnel,
                "cells": records,
            },
            f,
            indent=2,
        )

    if plt is None or not cells:
        return

    # One bar chart per stage, zones on x-axis, quadrant-grouped bars
    zones = sorted({z for z, _ in cells})
    quads = ["ST", "SN", "IT", "IN"]
    fig, axes = plt.subplots(len(stages), 1, figsize=(10, 2.4 * len(stages)), sharex=True)
    x = np.arange(len(zones))
    width = 0.2
    for ax, stage in zip(axes, stages):
        for i, qname in enumerate(quads):
            heights = [
                funnel.get((z, qname), {}).get(stage, 0) for z in zones
            ]
            ax.bar(x + (i - 1.5) * width, heights, width=width, label=qname)
        ax.set_ylabel(stage)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"Z{z}" for z in zones])
    axes[0].legend(ncol=4, fontsize=9)
    fig.suptitle(f"Stage funnel — {result.image_id}")
    fig.tight_layout()
    fig.savefig(out / "10_funnel.png", dpi=130)
    plt.close(fig)


# --- tables + metadata -----------------------------------------------


def _save_tables_and_metadata(result: ExtractionResult, out: Path) -> None:
    try:
        result.measurements_dataframe().to_parquet(out / "measurements.parquet")
        result.zonal_dataframe().to_parquet(out / "zonal.parquet")
    except ImportError:
        # pandas missing; skip silently
        pass

    # Always dump a JSON copy of the zonal table — no pandas dependency.
    zonal_rows = []
    for z in sorted(result.zonal, key=lambda z: (z.zone_index, z.quadrant)):
        zonal_rows.append(
            {
                "zone": z.zone_index,
                "quadrant": z.quadrant,
                "crae_px": z.crae_px,
                "crve_px": z.crve_px,
                "avr": z.avr,
                "crae_um": z.crae_um,
                "crve_um": z.crve_um,
                "n_arteries": z.n_arteries,
                "n_veins": z.n_veins,
                "flags": sorted(z.flags),
            }
        )
    with (out / "zonal.json").open("w") as f:
        json.dump(zonal_rows, f, indent=2, default=str)
    with (out / "run_metadata.json").open("w") as f:
        json.dump(_json_safe(result.run_metadata), f, indent=2, default=str)

    with (out / "summary.txt").open("w") as f:
        f.write(f"image: {result.image_id}\n")
        f.write(f"laterality: {result.laterality}\n")
        f.write(
            f"OD: ({result.od_center[0]:.1f}, {result.od_center[1]:.1f})  "
            f"r={result.od_radius_px:.1f}px\n"
        )
        f.write(
            f"fovea: {result.fovea_center}  conf={result.fovea_confidence:.2f}\n"
        )
        f.write(f"dewarp: device={result.dewarp_device}  applied="
                f"{result.run_metadata.get('dewarp_applied', False)}\n")
        f.write(f"segments: {len(result.segments)}  "
                f"measurements: {len(result.measurements)}  "
                f"zonal cells: {len(result.zonal)}\n\n")
        f.write("--- Z1 (Zone B) ---\n")
        for z in sorted(result.zonal, key=lambda z: (z.zone_index, z.quadrant)):
            if z.zone_index != 1:
                continue
            f.write(
                f"  Z1 {z.quadrant}: "
                f"CRAE={_fmt(z.crae_px)}px ({z.n_arteries}a)  "
                f"CRVE={_fmt(z.crve_px)}px ({z.n_veins}v)  "
                f"flags={','.join(sorted(z.flags)) or '-'}\n"
            )


def _fmt(v) -> str:
    return f"{v:.1f}" if v is not None else "NaN"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    return obj


def _make_dummy_config(result: ExtractionResult):
    """Reconstruct a minimal config for coord-system (only zones_dd needed)."""
    from uwf_zonal_extraction.config import ExtractionConfig

    cfg = result.run_metadata.get("config")
    if isinstance(cfg, dict) and "zones_dd" in cfg:
        return ExtractionConfig(zones_dd=tuple(tuple(z) for z in cfg["zones_dd"]))
    return ExtractionConfig()
