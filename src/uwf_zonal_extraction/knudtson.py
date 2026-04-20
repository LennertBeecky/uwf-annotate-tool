"""Knudtson revised (2003) CRAE/CRVE aggregation.

Canonical coefficients:
    arteries: 0.88 * sqrt(w1^2 + w2^2)
    veins:    0.95 * sqrt(w1^2 + w2^2)

Iterative pairing: sort descending, pair largest-with-smallest, carry the
middle element (if any), repeat until one value remains.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

from uwf_zonal_extraction.config import KnudtsonCfg

VesselType = Literal["artery", "vein"]


def pair_knudtson(w1: float, w2: float, vessel_type: VesselType, cfg: KnudtsonCfg) -> float:
    """Apply the Knudtson revised pair formula to two diameters."""
    if w1 < w2:
        w1, w2 = w2, w1
    coef = cfg.artery_coef if vessel_type == "artery" else cfg.vein_coef
    return coef * math.sqrt(w1 * w1 + w2 * w2)


def aggregate_knudtson(
    diameters: Iterable[float],
    vessel_type: VesselType,
    cfg: KnudtsonCfg | None = None,
) -> tuple[float | None, int, list[str]]:
    """Aggregate per-vessel diameters into a single CRAE / CRVE value.

    Returns:
        (value, n_used, flags)
        - value is None if fewer than `n_vessels_min_provisional` diameters supplied
        - flags contains provenance: {'n<6','provisional','single_pair',...}
    """
    if cfg is None:
        cfg = KnudtsonCfg()

    vals = sorted((float(d) for d in diameters if d is not None and not math.isnan(d)), reverse=True)
    n = len(vals)
    flags: list[str] = []

    if n == 0:
        return None, 0, ["no_vessels"]
    if n < cfg.n_vessels_min_provisional:
        # n == 1: report the raw diameter but flag single-vessel
        flags.append("single_vessel")
        return vals[0], 1, flags

    # Take up to the 6 largest (or whatever n_vessels_target is).
    if n > cfg.n_vessels_target:
        vals = vals[: cfg.n_vessels_target]
        n_used = cfg.n_vessels_target
    else:
        n_used = n
        if n_used < cfg.n_vessels_target:
            flags.append(f"n<{cfg.n_vessels_target}")
        if n_used < cfg.n_vessels_min_validated:
            flags.append("provisional")

    # Iterative pairing: largest-with-smallest, carry middle, recurse.
    current = list(vals)
    while len(current) > 1:
        current.sort(reverse=True)
        next_round: list[float] = []
        left, right = 0, len(current) - 1
        while left < right:
            next_round.append(pair_knudtson(current[left], current[right], vessel_type, cfg))
            left += 1
            right -= 1
        if left == right:
            next_round.append(current[left])
        current = next_round

    return current[0], n_used, flags
