"""Synthetic-phantom tests for the convolved-step profile fit."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from uwf_zonal_extraction.config import ProfileFitCfg
from uwf_zonal_extraction.profile_fitting import fit_vessel_profile


def make_vein(t, B, A, c, w, sigma):
    s = sigma * math.sqrt(2.0)
    return B - (A / 2.0) * (erf((t - c + w / 2.0) / s) - erf((t - c - w / 2.0) / s))


@pytest.mark.parametrize(
    "w_true, sigma_true",
    [
        (4.0, 1.0),
        (8.0, 1.5),
        (12.0, 2.0),
        (16.0, 2.5),
    ],
)
def test_vein_width_recovery_noiseless(w_true, sigma_true):
    cfg = ProfileFitCfg()
    t = np.arange(-30, 30.25, 0.25)
    profile = make_vein(t, B=200.0, A=100.0, c=0.0, w=w_true, sigma=sigma_true)
    fit = fit_vessel_profile(t, profile, "vein", w_init=w_true * 0.8, cfg=cfg)
    assert fit.success
    # Noiseless: recover w to <1% of the profile-sampling step (0.25)
    assert abs(fit.w - w_true) < 0.25


def test_vein_width_recovery_with_noise():
    cfg = ProfileFitCfg()
    rng = np.random.default_rng(0)
    t = np.arange(-30, 30.25, 0.25)
    profile = make_vein(t, B=200.0, A=100.0, c=0.0, w=8.0, sigma=1.5)
    profile = profile + rng.normal(scale=5.0, size=profile.shape)
    fit = fit_vessel_profile(t, profile, "vein", w_init=7.0, cfg=cfg)
    assert fit.success
    # 5% tolerance at noise σ=5 (profile depth = 100)
    assert abs(fit.w - 8.0) / 8.0 < 0.05


def test_artery_with_single_reflex_recovers_width():
    cfg = ProfileFitCfg()
    rng = np.random.default_rng(0)
    t = np.arange(-30, 30.25, 0.25)
    base = make_vein(t, B=200.0, A=100.0, c=0.0, w=10.0, sigma=1.5)
    reflex = 25.0 * np.exp(-(t**2) / (2.0 * 2.0**2))
    profile = base + reflex + rng.normal(scale=2.0, size=t.shape)
    fit = fit_vessel_profile(t, profile, "artery", w_init=9.0, cfg=cfg)
    assert fit.success
    assert abs(fit.w - 10.0) / 10.0 < 0.05
    assert fit.reflex_model in {"single_gauss", "twin_gauss"}
