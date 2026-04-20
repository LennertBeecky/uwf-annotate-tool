# 05 — Knudtson Aggregation

## Purpose

Aggregate per-vessel zonal median diameters into a single summary index
(CRAE for arteries, CRVE for veins) per (zone, quadrant) cell using
the Knudtson revised formulas. This produces the clinically standard
biomarkers comparable with published population studies.

## Background

The Knudtson et al. 2003 revision of the Parr–Hubbard formula uses
iterative pairing of vessel diameters based on Murray's-law branching
coefficients. It replaced earlier formulas (Hubbard 1999, Parr-Hubbard
1994) and is the current standard in ARIC, BMES, Rotterdam, and most
CVD-retinal studies (SIVA, IVAN, VAMPIRE).

## Canonical formulas (Knudtson 2003)

For two vessels with diameters `w₁ ≥ w₂`:

**Arteries:**
```
w_parent = 0.88 × √(w₁² + w₂²)
```

**Veins:**
```
w_parent = 0.95 × √(w₁² + w₂²)
```

These are the canonical Knudtson coefficients used across the
literature. (An earlier critique of the draft spec confused these with
the pre-Knudtson Brinchmann-Hansen / Hubbard 1999 form
`√(a·w₁² + b·w₂²)`. Knudtson collapsed those to a single coefficient
times `√(w₁² + w₂²)` as a deliberate simplification.)

Veins have a higher coefficient because the venular branching exponent
is larger than the arteriolar one — consistent with in-vivo observations
in the ARIC cohort.

## Iterative aggregation procedure

1. Collect all artery (or vein) `median_diameter` values in the cell.
2. Sort descending. Select the **6 largest**. If fewer than 6 are
   available, use all; if fewer than 2, return `NaN`.
3. Iteratively pair **largest-with-smallest**, apply the formula:

   ```
   With 6 diameters [d1 ≥ d2 ≥ ... ≥ d6]:
     Round 1: pair (d1, d6) → A;  pair (d2, d5) → B;  pair (d3, d4) → C
              → 3 values [A, B, C] (sort descending)
     Round 2: pair (A, C) → D;    carry B → E
              → 2 values [D, E]   (sort descending)
     Round 3: pair (D, E) → final CRAE
   ```

With fewer starting diameters the same "largest-with-smallest" rule
applies, just fewer rounds:
- 5 diameters: pair (d1, d5) + (d2, d4), carry d3 → 3 values → recurse.
- 4 diameters: pair (d1, d4) + (d2, d3) → 2 values → 1 round.
- 3 diameters: pair (d1, d3), carry d2 → 2 values → 1 round.
- 2 diameters: 1 round.

(All three cases reduce to the same procedure — "pair largest with
smallest, sort, recurse until one value remains".)

### Why 6 vessels

Knudtson showed that using exactly 6 vessels minimises inter-grader
variability while capturing the major vessels. Using more adds smaller
vessels that are harder to measure reliably; using fewer loses
information. This convention is preserved across IVAN / VAMPIRE / SIVA.

For the classic Zone B (0.5–1.0 DD), 6 is typical. In outer zones of
UWF images, there may be many more vessel crossings due to branching —
still select the 6 largest for consistency with the literature.

## Per-cell vs global aggregation

### Per-cell aggregation (v1 default)

Compute CRAE and CRVE independently for each (zone, quadrant) cell.
Produces:

```
CRAE[zone][quadrant]   — arteriolar caliber matrix
CRVE[zone][quadrant]   — venular caliber matrix
AVR[zone][quadrant]    — = CRAE / CRVE
```

### Global CRAE/CRVE (backward compatibility with published studies)

To produce a single CRAE / CRVE value comparable to IVAN / VAMPIRE /
SIVA published norms: use only Z1 (0.5–1.0 DD), pool all quadrants,
take the 6 largest arterioles and 6 largest venules, apply Knudtson.

### `legacy_zone_b` mode

Classic studies do NOT measure zonal-median diameters — they measure a
**single crossing-point diameter** at the Z1 outer boundary (or the
"most clearly defined" point within Z1, grader-dependent). The zonal
median is methodologically better but will produce **systematically
different absolute μm values** than IVAN / VAMPIRE / SIVA on the same
image, because the vessel tapers within 0.5 DD.

For direct Bland-Altman agreement studies, enable
`ExtractionConfig.legacy_zone_b = True`:

- For each vessel, sample ONE profile at the point nearest the Z1 outer
  boundary (r = 1.0 DD).
- Fit the convolved step model at that single point.
- Feed the single-diameter values into Knudtson (6 largest, global
  pool).
- Expected outcome: r > 0.90 correlation with published IVAN / SIVA on
  the same images, though absolute agreement is cohort-dependent.

The v1 scientific contribution is the **zonal-median matrix**. The
`legacy_zone_b` mode exists for validation and is published alongside
the new metric.

## Handling sparse cells

Outer zones and individual quadrants may have fewer than 6 vessels of
one type:

- **≥ 3 vessels**: apply Knudtson with available vessels. Flag
  `n_vessels < 6` in the output.
- **2 vessels**: single pairing round. Flag `n=2`; large uncertainty.
- **1 vessel**: report raw diameter as `crae_estimate` but do NOT apply
  Knudtson. Flag `n=1`, CI = ±∞.
- **0 vessels**: NaN.

v1 default: require `n_vessels >= 3` for a "validated" CRAE/CRVE output;
n=2 is reported as provisional. This is stricter than the legacy spec's
`n >= 2` because Z6–Z7 with only 2 vessels is noisy.

## Output per cell

```python
@dataclass
class ZonalResult:
    zone_index:         int
    quadrant:           Literal['ST','SN','IT','IN']
    crae_px:            float | None       # None ≡ NaN
    crve_px:            float | None
    avr:                float | None       # crae_px / crve_px
    crae_um:            float | None       # after mm/px
    crve_um:            float | None
    n_arteries:         int                # vessels used in aggregation
    n_veins:            int
    artery_diameters_px: list[float]       # per-vessel medians, for bootstrap
    vein_diameters_px:   list[float]
    flags:              set[str]           # e.g. {'n_arteries<6', 'dewarp_skipped'}
```

## Derived features for CVD

From the zone × quadrant matrix:

1. **Radial profile**: CRAE(distance_dd), CRVE(distance_dd), averaged
   across quadrants or per-quadrant. Healthy vessels taper gradually;
   accelerated peripheral narrowing is a CVD signal.

2. **Quadrant asymmetry**: max(|quadrant_i − quadrant_j|) at each zone,
   plus a hemifield difference (superior vs inferior — per Ikram /
   SIVA). Focal narrowing in one quadrant is a stronger CVD risk
   marker than global narrowing.

3. **AVR gradient**: slope of AVR against zone index. Steepening AVR
   toward the periphery may indicate early microvascular dysfunction.

4. **Zone B reference**: global CRAE / CRVE from Z1 (all quadrants)
   with `legacy_zone_b=True` for direct literature comparison.

## Validation against published norms

- Target correlation with IVAN / VAMPIRE / SIVA on Z1: **r ≥ 0.90**.
  (The legacy spec targeted r ≥ 0.95; published inter-software
  agreement (Yip 2016) is r ≈ 0.85–0.92, so 0.95 is unrealistic.)
- Target Bland-Altman bias: within ±10 μm of IVAN on Z1.
- Test-retest SD on same-day duplicate images: < 5 μm (≈ 3 % of CRAE).
