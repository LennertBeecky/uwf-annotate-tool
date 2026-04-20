"""In-memory result types. All arrays are numpy; `to_dataframe` produces pandas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# pandas is only required at serialization time; keep it lazy so the core
# scientific pipeline imports on a numpy-only environment.

Quadrant = Literal["ST", "SN", "IT", "IN"]
VesselType = Literal["artery", "vein"]
ReflexModel = Literal["none", "single_gauss", "twin_gauss"]
PointType = Literal["interior", "bifurcation", "crossing", "endpoint"]


@dataclass
class VesselSegment:
    """One unbranched vessel arc, with A/V label and per-cell sub-segments."""

    segment_id: int
    vessel_type: VesselType
    label_confidence: float                       # |mean(p_a - p_v)|, [0, 1]
    uncertain: bool                               # label_confidence below threshold
    full_points: np.ndarray                       # (N, 2) ordered (y, x)
    w_init: np.ndarray                            # (N,) 2*EDT[skel]
    point_types: np.ndarray                       # (N,) str
    exclusion_mask: np.ndarray                    # (N,) bool; True = skip sampling
    sub_segments: dict[tuple[int, str], list[int]] = field(default_factory=dict)
    # maps (zone_index, quadrant) -> list of indices into full_points


@dataclass
class VesselMeasurement:
    """Per-vessel, per-cell zonal median diameter."""

    vessel_id: int
    vessel_type: VesselType
    zone_index: int
    quadrant: Quadrant
    median_diameter_px: float
    diameters_px: np.ndarray
    sigma_w_px: np.ndarray
    rmse_rel: np.ndarray
    arc_length_dd: float
    n_samples: int
    reflex_model: ReflexModel
    # Per-accepted-sample position in image coords (y, x); parallel to diameters_px.
    sample_points: np.ndarray | None = None        # (n_samples, 2) float
    # One example ProfileFit kept for QC. Not intended for downstream analysis.
    example_fit: Any | None = None
    median_diameter_dd: float | None = None
    median_diameter_um: float | None = None
    diameters_um: np.ndarray | None = None
    resolution_limited: bool = False
    tortuosity: float = 1.0                       # arc_length / chord_length


@dataclass
class ZonalResult:
    """Knudtson-aggregated (zone, quadrant) cell."""

    zone_index: int
    quadrant: Quadrant
    crae_px: float | None = None
    crve_px: float | None = None
    avr: float | None = None
    crae_um: float | None = None
    crve_um: float | None = None
    crae_ci_low: float | None = None
    crae_ci_high: float | None = None
    crve_ci_low: float | None = None
    crve_ci_high: float | None = None
    n_arteries: int = 0
    n_veins: int = 0
    artery_diameters_px: list[float] = field(default_factory=list)
    vein_diameters_px: list[float] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)


@dataclass
class DebugArtefacts:
    """Intermediate tensors kept for QC / debug rendering.

    Populated when `return_debug=True` is passed to extract_caliber*. Not
    part of the scientific output — downstream code should not rely on it.
    """

    image_bgr: np.ndarray                         # (H, W, 3) dewarped input
    av_mask: np.ndarray                           # (H, W) 0/1/2
    av_probs: np.ndarray                          # (H, W, 2)
    retina_mask: np.ndarray                       # (H, W) 0/1
    skeleton: np.ndarray                          # (H, W) bool
    distance_map: np.ndarray                      # (H, W) float32 (EDT in vessel pixels)
    funnel: dict[str, int] = field(default_factory=dict)
    # per-(zone,quadrant) diagnostic counts for the stage funnel
    cell_funnel: dict[tuple[int, str], dict[str, int]] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Top-level output. Holds the full stack + the aggregated matrix."""

    image_id: str
    laterality: str
    segments: list[VesselSegment]
    measurements: list[VesselMeasurement]
    zonal: list[ZonalResult]                      # one per (zone, quadrant)
    od_center: tuple[float, float]
    od_radius_px: float
    fovea_center: tuple[float, float] | None
    fovea_confidence: float
    dewarp_device: str
    axial_length_mm: float
    mm_per_px_map: np.ndarray | None = None       # (H, W) float32, optional
    run_metadata: dict[str, Any] = field(default_factory=dict)
    debug: DebugArtefacts | None = None
    # run_metadata keys: config_hash, lunet_sha256, odc_sha256, package_version,
    # numpy_version, scipy_version, scikit_image_version, timestamp, etc.

    # -- DataFrame views --------------------------------------------------

    def measurements_dataframe(self):
        """Long-format table of per-vessel, per-cell median diameters."""
        import pandas as pd
        rows = []
        for m in self.measurements:
            rows.append(
                {
                    "image_id": self.image_id,
                    "laterality": self.laterality,
                    "vessel_id": m.vessel_id,
                    "vessel_type": m.vessel_type,
                    "zone": m.zone_index,
                    "quadrant": m.quadrant,
                    "median_diameter_px": m.median_diameter_px,
                    "median_diameter_dd": m.median_diameter_dd,
                    "median_diameter_um": m.median_diameter_um,
                    "n_samples": m.n_samples,
                    "arc_length_dd": m.arc_length_dd,
                    "reflex_model": m.reflex_model,
                    "tortuosity": m.tortuosity,
                    "resolution_limited": m.resolution_limited,
                }
            )
        return pd.DataFrame(rows)

    def zonal_dataframe(self):
        """Long-format table of per-cell CRAE/CRVE/AVR with CIs."""
        import pandas as pd
        rows = []
        for z in self.zonal:
            rows.append(
                {
                    "image_id": self.image_id,
                    "laterality": self.laterality,
                    "zone": z.zone_index,
                    "quadrant": z.quadrant,
                    "crae_px": z.crae_px,
                    "crve_px": z.crve_px,
                    "crae_um": z.crae_um,
                    "crve_um": z.crve_um,
                    "avr": z.avr,
                    "crae_ci_low": z.crae_ci_low,
                    "crae_ci_high": z.crae_ci_high,
                    "crve_ci_low": z.crve_ci_low,
                    "crve_ci_high": z.crve_ci_high,
                    "n_arteries": z.n_arteries,
                    "n_veins": z.n_veins,
                    "flags": ",".join(sorted(z.flags)),
                }
            )
        return pd.DataFrame(rows)

    def to_dataframe(self):
        """Alias for `zonal_dataframe()` — the primary output table."""
        return self.zonal_dataframe()
