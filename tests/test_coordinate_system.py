"""Unit tests for RetinalCoordinateSystem."""

from __future__ import annotations

import math

import numpy as np
import pytest

from uwf_zonal_extraction.config import ExtractionConfig
from uwf_zonal_extraction.coordinate_system import RetinalCoordinateSystem


def _cs(fovea_center=(500, 250), od_center=(250, 250), od_radius=50):
    return RetinalCoordinateSystem.from_landmarks(
        od_center=od_center,
        od_radius_px=od_radius,
        fovea_center=fovea_center,
        config=ExtractionConfig(),
        laterality="OD",
    )


def test_axis_angle_fovea_right_of_od_is_zero():
    cs = _cs(fovea_center=(500, 250), od_center=(250, 250))
    assert cs.axis_angle == pytest.approx(0.0)


def test_point_to_zone_at_boundary():
    cs = _cs()
    # DD = 100 px; Z1 = 0.5-1.0 DD = 50-100 px from OD centre
    # A point at (250+75, 250) is 75px = 0.75 DD → Z1
    assert cs.point_to_zone(250, 325) == 1
    # Point at (250+60, 250) = 60 px = 0.6 DD → Z1
    assert cs.point_to_zone(250, 310) == 1
    # Point at (250+40, 250) = 40 px = 0.4 DD → Z0
    assert cs.point_to_zone(250, 290) == 0
    # Point at (250+600, 250) = 600 px = 6 DD → outside outermost zone (5 DD)
    assert cs.point_to_zone(250, 850) == -1


def test_point_to_quadrant_standard_90deg_sectors():
    cs = _cs()  # OD at (250,250), fovea at (500,250) → axis angle 0, temporal +x
    # Up-temporal (smaller y, larger x) → ST
    assert cs.point_to_quadrant(200, 300) == "ST"
    # Down-temporal (larger y, larger x) → IT
    assert cs.point_to_quadrant(300, 300) == "IT"
    # Up-nasal (smaller y, smaller x) → SN
    assert cs.point_to_quadrant(200, 200) == "SN"
    # Down-nasal (larger y, smaller x) → IN
    assert cs.point_to_quadrant(300, 200) == "IN"


def test_distance_dd():
    cs = _cs()
    assert cs.distance_dd(250, 250) == pytest.approx(0.0)
    # 100px horizontal = 1.0 DD
    assert cs.distance_dd(250, 350) == pytest.approx(1.0)


def test_rotation_invariance_of_distance():
    cs = _cs()
    r = 73.0
    for theta in np.linspace(0, 2 * math.pi, 16):
        y = 250 + r * math.sin(theta)
        x = 250 + r * math.cos(theta)
        assert cs.distance_dd(y, x) == pytest.approx(r / 100.0, abs=1e-9)


def test_point_to_zone_array_matches_scalar():
    cs = _cs()
    ys = np.array([250, 250, 250, 250], dtype=float)
    xs = np.array([290, 310, 400, 850], dtype=float)
    arr = cs.point_to_zone_array(ys, xs)
    scalars = [cs.point_to_zone(float(y), float(x)) for y, x in zip(ys, xs)]
    assert list(arr) == scalars


def test_asymmetric_od_x_y_convention():
    """Catch silent (x, y) <-> (y, x) swap with an asymmetric landmark config.

    OD at (x=300, y=200), fovea at (x=600, y=200): axis points +x (temporal
    right). A point up-temporal (smaller y, larger x) should be ST; a point
    down-nasal (larger y, smaller x) should be IN. With an x/y swap these
    would come out wrong.
    """
    cs = RetinalCoordinateSystem.from_landmarks(
        od_center=(300, 200),              # (x, y)
        od_radius_px=40,
        fovea_center=(600, 200),
        config=ExtractionConfig(),
        laterality="OD",
    )
    # Distance: (300+40, 200) is 40 px = 1 DD → Z1 outer edge → Z-lookup is 1
    assert cs.point_to_zone(200, 340) == 1
    # Up-temporal: (y=150, x=350) is above-right of OD → ST
    assert cs.point_to_quadrant(150, 350) == "ST"
    # Down-nasal: (y=250, x=250) is below-left → IN
    assert cs.point_to_quadrant(250, 250) == "IN"
    # Down-temporal: (y=250, x=350) → IT
    assert cs.point_to_quadrant(250, 350) == "IT"
    # Up-nasal: (y=150, x=250) → SN
    assert cs.point_to_quadrant(150, 250) == "SN"
