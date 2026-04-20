"""ExtractionConfig — frozen dataclass with YAML loader.

All pipeline tunables live here. Defaults are grep-able in code; YAML
overrides support per-experiment tuning without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# yaml is only needed by `from_yaml`; import lazily to keep the core
# scientific pipeline importable without pyyaml.


# -- Zone table (DD units) --------------------------------------------------

DEFAULT_ZONES_DD: tuple[tuple[float, float], ...] = (
    (0.0, 0.5),   # Z0  — disc margin (excluded)
    (0.5, 1.0),   # Z1  — classic Zone B
    (1.0, 1.5),   # Z2  — classic Zone C
    (1.5, 2.0),   # Z3
    (2.0, 2.5),   # Z4
    (2.5, 3.0),   # Z5
    (3.0, 4.0),   # Z6  — UWF only (research)
    (4.0, 5.0),   # Z7  — UWF only (research)
)

Z0_EXCLUDED = 0
Z1_ZONE_B = 1


# -- Sub-config dataclasses ------------------------------------------------


@dataclass(frozen=True)
class SegmentationCfg:
    tile_size: int = 1472
    stride: int = 736
    av_threshold: float = 0.5
    green_input: bool = True
    tta: bool = False


@dataclass(frozen=True)
class DewarpCfg:
    device: str = "daytona"           # 'daytona', 'california', 'p200_tx', or 'none'
    axial_length_mm_default: float = 23.5
    retinal_radius_offset_mm: float = 1.5   # R_ret = AL - 1.5 mm


@dataclass(frozen=True)
class SkeletonCfg:
    closing_disk_radius: int = 2
    spur_min_px: int = 5                          # floor of max(5, 0.5*w_local)
    min_segment_length_px: int = 5
    bridge_gaps: bool = False
    bridge_max_px: int = 10
    bridge_angle_tol_deg: float = 30.0
    bifurcation_exclude_factor: float = 0.8       # * local_w
    a_v_confidence_flag: float = 0.2              # |p_a - p_v| threshold


@dataclass(frozen=True)
class ProfileFitCfg:
    sample_interval_dd: float = 0.05
    min_sample_spacing_px: float = 3.0
    # End-buffer at each end of a sub-arc — junction artefacts extend
    # roughly 1.5 * local_w regardless of image resolution.
    end_buffer_w_factor: float = 1.5              # * local_w
    half_width_min_px: int = 30
    # Retained for optional adaptive scaling; set to 0.0 to keep half_width
    # fixed at half_width_min_px (the intensity-driven design wants a
    # generous fixed window so the outer 25% of every profile is reliably
    # pure tissue for baseline estimation).
    half_width_init_factor: float = 0.0           # * w_init
    profile_sampling_step_px: float = 0.25
    profile_interp_order: int = 3                 # 3 = bicubic
    baseline_tail_fraction: float = 0.25          # outer 25 %
    rmse_rel_accept: float = 0.10                 # RMSE/A threshold for acceptance
    rmse_rel_global_sigma: float = 0.05           # threshold for σ_PSF pool
    twin_reflex_trigger: float = 2.0              # MAD multiplier for twin fallback
    lm_loss: str = "soft_l1"
    lm_f_scale_divisor: float = 5.0               # f_scale = A / this
    sigma_radial_bins: int = 5
    sigma_poly_degree: int = 2


@dataclass(frozen=True)
class KnudtsonCfg:
    artery_coef: float = 0.88
    vein_coef: float = 0.95
    n_vessels_target: int = 6
    n_vessels_min_validated: int = 3
    n_vessels_min_provisional: int = 2
    legacy_zone_b: bool = False                   # single-crossing @ Z1 outer, for IVAN comparison


@dataclass(frozen=True)
class LongitudinalCfg:
    bootstrap_iters: int = 1000
    bootstrap_seed: int = 42


@dataclass(frozen=True)
class ExtractionConfig:
    zones_dd: tuple[tuple[float, float], ...] = DEFAULT_ZONES_DD
    segmentation: SegmentationCfg = field(default_factory=SegmentationCfg)
    dewarp: DewarpCfg = field(default_factory=DewarpCfg)
    skeleton: SkeletonCfg = field(default_factory=SkeletonCfg)
    profile_fit: ProfileFitCfg = field(default_factory=ProfileFitCfg)
    knudtson: KnudtsonCfg = field(default_factory=KnudtsonCfg)
    longitudinal: LongitudinalCfg = field(default_factory=LongitudinalCfg)

    # Population defaults
    population_dd_um: float = 1800.0       # average human DD in microns
    laterality_default: str = "OD"         # fallback if sidecar absent

    # Output
    output_units: tuple[str, ...] = ("px", "dd", "um")

    # -- IO --------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExtractionConfig":
        import yaml
        with Path(path).open() as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        def _asdict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, tuple):
                return [_asdict(x) for x in obj]
            return obj

        return _asdict(self)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> "ExtractionConfig":
        sub = {
            "segmentation": SegmentationCfg(**raw.get("segmentation", {})),
            "dewarp":       DewarpCfg(**raw.get("dewarp", {})),
            "skeleton":     SkeletonCfg(**raw.get("skeleton", {})),
            "profile_fit":  ProfileFitCfg(**raw.get("profile_fit", {})),
            "knudtson":     KnudtsonCfg(**raw.get("knudtson", {})),
            "longitudinal": LongitudinalCfg(**raw.get("longitudinal", {})),
        }
        scalar_keys = {
            "zones_dd", "population_dd_um", "laterality_default", "output_units",
        }
        top = {k: v for k, v in raw.items() if k in scalar_keys}
        if "zones_dd" in top:
            top["zones_dd"] = tuple(tuple(z) for z in top["zones_dd"])
        if "output_units" in top:
            top["output_units"] = tuple(top["output_units"])
        return cls(**top, **sub)
