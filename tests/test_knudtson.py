"""Unit tests for Knudtson aggregation."""

from __future__ import annotations

import math

import pytest

from uwf_zonal_extraction.config import KnudtsonCfg
from uwf_zonal_extraction.knudtson import aggregate_knudtson, pair_knudtson


CFG = KnudtsonCfg()


def test_pair_knudtson_artery():
    # 0.88 * sqrt(100^2 + 60^2) = 0.88 * sqrt(13600) ≈ 0.88 * 116.619 ≈ 102.624
    got = pair_knudtson(100.0, 60.0, "artery", CFG)
    assert got == pytest.approx(0.88 * math.sqrt(100**2 + 60**2))


def test_pair_knudtson_vein():
    got = pair_knudtson(100.0, 60.0, "vein", CFG)
    assert got == pytest.approx(0.95 * math.sqrt(100**2 + 60**2))


def test_pair_order_independent():
    assert pair_knudtson(80.0, 120.0, "artery", CFG) == pytest.approx(
        pair_knudtson(120.0, 80.0, "artery", CFG)
    )


def test_aggregate_single_vessel_returns_raw():
    val, n, flags = aggregate_knudtson([123.4], "artery", CFG)
    assert val == pytest.approx(123.4)
    assert n == 1
    assert "single_vessel" in flags


def test_aggregate_two_vessels_one_round():
    val, n, flags = aggregate_knudtson([100, 60], "artery", CFG)
    assert val == pytest.approx(0.88 * math.sqrt(100**2 + 60**2))
    assert n == 2
    assert f"n<{CFG.n_vessels_target}" in flags


def test_aggregate_six_vessels():
    # Known sequence: 6 diameters 160..110 step -10
    ds = [160, 150, 140, 130, 120, 110]
    val, n, _ = aggregate_knudtson(ds, "artery", CFG)
    # Round 1: pair (160,110), (150,120), (140,130)
    a = 0.88 * math.sqrt(160**2 + 110**2)
    b = 0.88 * math.sqrt(150**2 + 120**2)
    c = 0.88 * math.sqrt(140**2 + 130**2)
    # Sorted desc
    rnd1 = sorted([a, b, c], reverse=True)
    # Round 2: pair (rnd1[0], rnd1[2]), carry rnd1[1]
    d = 0.88 * math.sqrt(rnd1[0] ** 2 + rnd1[2] ** 2)
    e = rnd1[1]
    rnd2 = sorted([d, e], reverse=True)
    expected = 0.88 * math.sqrt(rnd2[0] ** 2 + rnd2[1] ** 2)
    assert val == pytest.approx(expected)
    assert n == 6


def test_aggregate_more_than_six_takes_top_six():
    ds = [200, 180, 160, 140, 120, 100, 80, 60]
    val, n, _ = aggregate_knudtson(ds, "vein", CFG)
    val6, n6, _ = aggregate_knudtson(ds[:6], "vein", CFG)
    assert val == pytest.approx(val6)
    assert n == 6 == n6


def test_aggregate_empty():
    val, n, flags = aggregate_knudtson([], "artery", CFG)
    assert val is None
    assert n == 0
    assert "no_vessels" in flags


def test_aggregate_nans_ignored():
    val1, n1, _ = aggregate_knudtson([100, 60, math.nan], "artery", CFG)
    val2, n2, _ = aggregate_knudtson([100, 60], "artery", CFG)
    assert val1 == pytest.approx(val2)
    assert n1 == n2
