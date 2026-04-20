"""RetinalCoordinateSystem — OD-fovea anchored coordinates in DD units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math

import numpy as np

from uwf_zonal_extraction.config import ExtractionConfig

Quadrant = Literal["ST", "SN", "IT", "IN"]


@dataclass
class RetinalCoordinateSystem:
    """Patient-anchored coordinate system.

    All inputs in pixel coordinates of the (dewarped) image. `od_center`
    is (x, y). Zones and quadrants derive from:
      - DD = 2 * od_radius_px   (disc diameter, from ellipse major axis)
      - axis_angle = atan2(-(y_f - y_od), (x_f - x_od))
    """

    od_center: tuple[float, float]
    od_radius_px: float
    fovea_center: Optional[tuple[float, float]]
    zones_dd: tuple[tuple[float, float], ...]
    laterality: str = "OD"

    # Cached axis angle (None if fovea is missing)
    axis_angle: Optional[float] = None

    def __post_init__(self) -> None:
        if self.fovea_center is None:
            # Fallback: temporal = +x for OD, -x for OS
            self.axis_angle = 0.0 if self.laterality.upper() in {"OD", "R", "RIGHT"} else math.pi
        else:
            fx, fy = self.fovea_center
            cx, cy = self.od_center
            self.axis_angle = math.atan2(-(fy - cy), (fx - cx))

    # -- Basic geometry -------------------------------------------------

    @property
    def dd_px(self) -> float:
        return 2.0 * self.od_radius_px

    def distance_dd(self, y: float, x: float) -> float:
        cx, cy = self.od_center
        return math.hypot(x - cx, y - cy) / max(self.dd_px, 1e-6)

    def distance_dd_array(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        cx, cy = self.od_center
        return np.hypot(xs - cx, ys - cy) / max(self.dd_px, 1e-6)

    # -- Zones ----------------------------------------------------------

    def point_to_zone(self, y: float, x: float) -> int:
        """Return zone index (0..len(zones_dd)-1), or -1 if outside the outermost zone."""
        d = self.distance_dd(y, x)
        for z, (inner, outer) in enumerate(self.zones_dd):
            if inner <= d < outer:
                return z
        return -1

    def point_to_zone_array(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        d = self.distance_dd_array(ys, xs)
        out = np.full(d.shape, -1, dtype=np.int8)
        for z, (inner, outer) in enumerate(self.zones_dd):
            out[(d >= inner) & (d < outer)] = z
        return out

    # -- Quadrants -----------------------------------------------------

    def point_to_quadrant(self, y: float, x: float) -> Quadrant:
        cx, cy = self.od_center
        angle = math.atan2(-(y - cy), (x - cx)) - (self.axis_angle or 0.0)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        if 0.0 <= angle < math.pi / 2:
            return "ST"
        if math.pi / 2 <= angle <= math.pi:
            return "SN"
        if -math.pi / 2 <= angle < 0.0:
            return "IT"
        return "IN"

    def point_to_quadrant_array(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        cx, cy = self.od_center
        angle = np.arctan2(-(ys - cy), (xs - cx)) - (self.axis_angle or 0.0)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        out = np.empty(angle.shape, dtype="U2")
        out[(angle >= 0) & (angle < np.pi / 2)] = "ST"
        out[(angle >= np.pi / 2) & (angle <= np.pi)] = "SN"
        out[(angle >= -np.pi / 2) & (angle < 0)] = "IT"
        out[(angle >= -np.pi) & (angle < -np.pi / 2)] = "IN"
        return out

    def point_to_cell(self, y: float, x: float) -> tuple[int, Quadrant]:
        return self.point_to_zone(y, x), self.point_to_quadrant(y, x)

    # -- Construction helpers ------------------------------------------

    @classmethod
    def from_landmarks(
        cls,
        od_center: tuple[float, float],
        od_radius_px: float,
        fovea_center: tuple[float, float] | None,
        config: ExtractionConfig,
        laterality: str = "OD",
    ) -> "RetinalCoordinateSystem":
        return cls(
            od_center=(float(od_center[0]), float(od_center[1])),
            od_radius_px=float(od_radius_px),
            fovea_center=(
                (float(fovea_center[0]), float(fovea_center[1]))
                if fovea_center is not None
                else None
            ),
            zones_dd=config.zones_dd,
            laterality=laterality,
        )
