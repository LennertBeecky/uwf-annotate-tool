# 04 — Diameter Extraction

## Purpose

Extract sub-pixel vessel diameters along each vessel sub-segment within
a (zone, quadrant) cell, and aggregate into a single robust "zonal
median diameter" per vessel per cell.

## Why zonal averaging, not point sampling

A single crossing-point measurement (where a vessel intersects the zone
boundary) has problems:

1. **Angle sensitivity** — depends on the vessel's angle relative to
   the zone-boundary circle at that exact pixel.
2. **Single-point noise** — one bad profile fit (artefact, shadow,
   residual crossing) corrupts the measurement.
3. **Wasted information** — the entire vessel arc within the zone
   carries information; we use all of it.

The `legacy_zone_b` comparator mode (see 05_KNUDTSON_AGGREGATION.md)
additionally provides single-crossing measurements at the Z1 outer
boundary for direct Bland-Altman agreement with IVAN / VAMPIRE / SIVA.

## Profile sampling

For each sub-segment:

1. Compute cumulative arc length along `full_points` in pixels, convert
   to DD units.
2. Define sample positions at intervals of
   `sample_interval_dd` (default 0.05 DD) along the sub-segment, with a
   minimum pixel spacing of 3 px. Skip the first and last interval to
   avoid boundary artefacts.
3. Skip samples inside the exclusion buffer (`VesselSegment.exclusion_mask`)
   around bifurcations/crossings.
4. At each valid sample, find the nearest skeleton point index, estimate
   the tangent (PCA, 7-point window on `full_points`), and extract a
   perpendicular intensity profile.

## Tangent estimation — PCA, not chord

```python
window = 7
start  = max(0, idx - window // 2)
end    = min(len(full_points), idx + window // 2 + 1)
pts    = full_points[start:end].astype(float)
pts   -= pts.mean(axis=0)
cov    = np.cov(pts, rowvar=False)
_, eigvecs = np.linalg.eigh(cov)
tangent = eigvecs[:, -1]                   # principal axis
```

(PCA is robust on tortuous peripheral vessels; endpoint subtraction
returns a chord direction, which is biased.)

Perpendicular: rotate tangent by 90°:

```python
# tangent = (ty, tx); perpendicular = (-tx, ty) or (tx, -ty)
perp = np.array([-tangent[1], tangent[0]])
```

**Sign convention to verify in tests**: the sign of `perp` does not
matter for diameter estimation (the profile is symmetric) but matters
for QC overlay alignment — pick a convention and stick to it.

## Perpendicular profile extraction

```python
half_width_px = max(15, ceil(1.8 * w_init[idx]))
t             = np.arange(-half_width_px, half_width_px + 0.25, 0.25)
sample_yx     = center + t[:, None] * perp[None, :]
profile       = scipy.ndimage.map_coordinates(
                    green_channel,
                    sample_yx.T,
                    order=3,              # bicubic — bilinear over-smooths peaks
                    mode='reflect',
                )
```

`half_width_px` is **adaptive**: major arcades near the OD can be
20–25 px wide; `half_width = 15` (the legacy default) leaves the
baseline-tail region inside the vessel and biases the baseline estimate
`B` downward, inflating `w`. Scaling with `1.8 × w_init` guarantees the
outer 25% of the profile is pure tissue.

For colour images use the **green channel** (best vessel contrast in
CFP / Optos). For the dewarped UWF the pipeline extracts green before
stage 04.

## Convolved step model fitting

### Vein model (no central reflex)

```
I(x) = B − (A/2) · [ erf((x − c + w/2) / (σ √2)) − erf((x − c − w/2) / (σ √2)) ]
```

Parameters: `B` (background), `A` (absorption depth), `c` (centre),
`w` (diameter — the parameter we want), `σ` (PSF blur).

`w` is the vessel diameter and is explicitly decoupled from `σ`.

### Artery model (single-Gaussian reflex by default)

```
I(x) = I_vein(x) + R · exp( −(x − c)² / (2 σ_r²) )
```

Additional parameters: `R` (reflex amplitude), `σ_r` (reflex width).

A **twin-Gaussian fallback** (two symmetric bright peaks flanking the
dark core, typical of young/hypertensive arterioles with brisk reflex):

```
I(x) = I_vein(x) + R · [ exp(−(x − c − d)²/(2σ_r²)) + exp(−(x − c + d)²/(2σ_r²)) ]
```

The twin model is fitted only if the single-Gaussian residuals show a
central over-correction (centre residual < −2 · MAD(residuals)).

### Fitting procedure

1. **Initialize from raw profile + w_init**:
   - `B` = median of outer 25 % of profile (MAD-rejected: discard pixels
     > 2 MAD from the median before recomputing).
   - `A` = `B − min(profile)`.
   - `c` = position of minimum intensity.
   - `w` = `w_init` (from `medial_axis` EDT).
   - `σ` = initial value from the radial-σ map (see below); if no map
     yet, `σ = w / 4`.
   - For arteries: `R = 0.3 · A`, `σ_r = w / 4`.

2. **Fit via `scipy.optimize.least_squares`** with `method='trf'`,
   `loss='soft_l1'`, `f_scale = A / 5`. Bounds:
   - `B`: [0, 255]
   - `A`: [0, 255]
   - `c`: [profile start, profile end]
   - `w`: [0.5, `half_width_px`]
   - `σ`: [0.1, `half_width_px / 2`]
   - `R`: [0, 255]; `σ_r`: [0.1, `half_width_px / 4`]

   Soft-L1 loss is Huber-like and is robust to the single-pixel bright
   spike that occasionally occurs from Optos laser artefacts or vein
   wall reflexes.

3. **Quality metric**: use `RMSE_rel = RMSE / A` (normalized residual
   depth) AND the parameter standard error `σ_w` from the Jacobian
   covariance. `RMSE_rel < 0.1` (good fit) is the acceptance threshold;
   low-contrast profiles are not penalized the way an R²-threshold
   penalizes them (R²-denominator explodes when the profile is nearly
   constant, mis-ranking fits).

4. **Compute fit SE**: `σ_w = sqrt(diag(inv(J.T @ J))[index_w]) · RMSE`.
   This is the weight used in zonal-median aggregation.

## σ_PSF as a radial field (NOT a global constant)

The legacy spec proposed fixing `σ` to the median of high-R² fits. On
Optos UWF this is unsafe: peripheral blur (stereographic + off-axis
optics) makes `σ` vary by ≥2× between the posterior pole and Z6–Z7.
Even after dewarping the PSF is not isotropic.

v1 strategy:

1. First pass: fit all profiles with `σ` free, using `RMSE_rel` as
   quality.
2. Keep fits with `RMSE_rel < 0.05` (high-confidence).
3. Bin high-confidence fits by distance from **image centre** (5 radial
   bins equally spaced from 0 to max radius). Compute median `σ` per
   bin.
4. Fit a smooth polynomial (degree 2) to `σ(radius)`. This is the
   radial σ map.
5. Second pass: re-fit every profile with `σ` fixed to
   `σ_map(radius(y, x))`. This eliminates one DOF and improves
   diameter precision, especially for thin vessels in the periphery
   where the original σ estimate is noisy.

## Zonal median computation

After sampling N profiles along a vessel sub-segment:

```python
diameters = [w_1, w_2, ..., w_N]
weights   = [1/σ_w_1², 1/σ_w_2², ..., 1/σ_w_N²]

zonal_median = weighted_median(diameters, weights)
```

**Weighted median** by inverse-variance weights: sort by diameter,
compute cumulative weight, find the value where cumulative weight
crosses 50 %. Inverse-variance is strictly better than `R²`-weighting:
the range of `R²` among accepted fits is compressed (0.7–0.99), so
`R²` weights act almost uniformly; `1/σ_w²` spans multiple orders of
magnitude and gives substantially more weight to high-confidence fits.

### Why median over mean

- Robust to outliers from residual vessel crossings or bifurcation tails.
- A single corrupted profile that passes the quality threshold but is
  biased doesn't pull the estimate.

Also store `n_samples`, `arc_length_dd`, the full `diameters` and
`sigma_w` arrays (for bootstrap CIs in stage 06), and the residuals
(for QC).

## Output per vessel per cell

```python
@dataclass
class VesselMeasurement:
    vessel_id:            int
    vessel_type:          Literal['artery', 'vein']
    zone_index:           int
    quadrant:             Literal['ST', 'SN', 'IT', 'IN']
    median_diameter_px:   float             # inverse-variance-weighted median
    diameters_px:         np.ndarray        # all per-sample diameters
    sigma_w_px:           np.ndarray        # per-sample LM SE
    rmse_rel:             np.ndarray        # per-sample quality
    arc_length_dd:        float
    n_samples:            int
    reflex_model:         Literal['none','single_gauss','twin_gauss']
    # In μm (after per-pixel mm/px map is applied):
    median_diameter_um:   float
    diameters_um:         np.ndarray
```

These feed the Knudtson aggregation.

## Known failure modes and mitigations

- **Hemorrhages/cotton-wool spots** near the vessel darken the
  profile shoulder and bias `B` downward → overestimated `w`. Mitigated
  by MAD-based outlier rejection in the baseline estimate.
- **Vessel tortuosity.** Sub-segments in highly tortuous vessels have
  samples that are not independent (spatial correlation), but the
  median is robust. Tortuosity itself is flagged per sub-segment
  (arc_length / end-to-end distance ratio) and included in the output.
- **Thin peripheral vessels** (w < 3 px in Z6–Z7) are at the resolution
  limit of the convolved step model. Rejected as
  `resolution_limited=True` and excluded from Knudtson.
