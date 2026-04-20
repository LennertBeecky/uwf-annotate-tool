"""Cross-visit delta + bootstrap CIs (v1)."""

from __future__ import annotations

import numpy as np

from uwf_zonal_extraction.config import KnudtsonCfg
from uwf_zonal_extraction.knudtson import aggregate_knudtson
from uwf_zonal_extraction.models import ExtractionResult


def bootstrap_crae(
    diameters: list[float],
    vessel_type: str,
    cfg: KnudtsonCfg,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (median, ci_low, ci_high) from bootstrap resampling."""
    rng = np.random.default_rng(seed)
    values = [float(d) for d in diameters if d is not None and not np.isnan(d)]
    if len(values) < cfg.n_vessels_min_provisional:
        return (float("nan"),) * 3

    n_sample = min(len(values), cfg.n_vessels_target)
    samples = np.empty(n_iter, dtype=float)
    arr = np.asarray(values, dtype=float)
    for i in range(n_iter):
        resamp = rng.choice(arr, size=n_sample, replace=True)
        val, _, _ = aggregate_knudtson(resamp.tolist(), vessel_type, cfg)
        samples[i] = val if val is not None else np.nan

    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return (float("nan"),) * 3
    return (
        float(np.median(samples)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def compare_visits(
    baseline: ExtractionResult,
    followup: ExtractionResult,
    cfg: KnudtsonCfg | None = None,
):
    """Return delta table joining baseline vs follow-up at each (zone, quadrant)."""
    import pandas as pd  # noqa: F401
    df_v1 = baseline.zonal_dataframe().rename(columns=lambda c: f"{c}_v1" if c not in {"zone", "quadrant"} else c)
    df_vk = followup.zonal_dataframe().rename(columns=lambda c: f"{c}_vk" if c not in {"zone", "quadrant"} else c)
    df = df_v1.merge(df_vk, on=["zone", "quadrant"], how="outer")

    df["delta_crae"] = df["crae_px_vk"] - df["crae_px_v1"]
    df["delta_crve"] = df["crve_px_vk"] - df["crve_px_v1"]
    df["delta_avr"] = df["avr_vk"] - df["avr_v1"]
    df["pct_delta_crae"] = df["delta_crae"] / df["crae_px_v1"] * 100
    df["pct_delta_crve"] = df["delta_crve"] / df["crve_px_v1"] * 100
    return df
