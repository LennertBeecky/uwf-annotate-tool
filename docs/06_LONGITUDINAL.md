# 06 — Longitudinal Comparison and Robustness

## Purpose

Compare CRAE / CRVE / AVR matrices across patient visits to detect
clinically meaningful vascular changes, and characterize the measurement
noise floor so that real changes can be distinguished from pipeline
variability.

v1 scope: **bootstrap-CI-based delta**. Full vessel-graph matching is
scoped out and lives as a stub in `longitudinal/matching.py` for future
v2.

## Cross-visit comparison (v1: bootstrap CI)

### Delta computation

For visits V1 (baseline) and Vk (follow-up):

```
ΔCRAE[zone][quad] = CRAE_Vk − CRAE_V1
ΔCRVE[zone][quad] = CRVE_Vk − CRVE_V1
ΔAVR[zone][quad]  = AVR_Vk  − AVR_V1

%ΔCRAE = ΔCRAE / CRAE_V1 · 100
%ΔCRVE = ΔCRVE / CRVE_V1 · 100
```

### Bootstrap CI (v1 default)

At each visit independently:

1. Extract all vessel diameters per cell.
2. Apply Knudtson with 6 largest (or all if < 6).
3. Resample 6 of N vessels **with replacement** 1000 times; compute
   CRAE each time. Report **median and 95% percentile CI**.

Cross-visit difference is **significant** if the CIs do not overlap
(equivalently `|ΔCRAE| > sqrt(CI_V1² + CI_Vk²)`).

The bootstrap also produces per-cell CIs for single-visit reports —
every `ZonalResult` in the output DataFrame includes
`crae_ci_low, crae_ci_high, crve_ci_low, crve_ci_high`.

### Clinical significance thresholds

- Published test-retest SD for CRAE ≈ 3–5 % (≈ 5–7 μm).
- 2 × SD (≈ 10–14 μm) is a reasonable single-vessel clinical-change
  threshold.
- Effect-size tied to CVD outcome: McGeechan ARIC 2009 showed 1 SD
  lower CRAE = 1.17× stroke risk → the minimum clinically-interesting
  change is ~1 SD (≈ 10 μm) over a year.

## Vessel-graph matching (v2 STUB)

The most reliable longitudinal comparison uses the **same anatomical
vessels** at each visit, which eliminates set-membership noise
(different vessels entering/leaving the top-6 at different visits).

Sketch for v2:

1. At each visit, build a graph: nodes = bifurcation points, edges =
   vessel segments with attributes `(length_dd, mean_w, A/V)`.
2. Anchor at the OD. Extract the first 3 bifurcations of each major
   artery/vein as a topology descriptor.
3. Match descriptors across visits using bipartite matching on
   `(topology_distance, spatial_distance_DD)` cost.
4. Measure matched vessels at both visits and apply Knudtson to the
   same set.

`longitudinal/matching.py` contains a placeholder interface but
raises `NotImplementedError` in v1. Install with `[matching]` extra
(networkx) before enabling.

## Robustness testing

Three categories of perturbation to characterize the pipeline's noise
floor. Implemented in `tests/test_robustness.py` using synthetic and
(optionally) real images.

### 1. Geometric invariance

Pipeline should produce identical CRAE / CRVE under affine transforms
(the image changed, the retina didn't).

- Rotation: 0°, 5°, 10°, 15°, 20°
- Translation: ±5, ±10, ±20 px
- Scale: 0.95×, 1.0×, 1.05×

For each perturbation: re-run full pipeline (OD detection, coord
system, extraction, Knudtson). Report coefficient of variation across
perturbations.

**Targets:** CV < 2 % for rotation and translation; CV < 3 % for scale.

**Known failure mode:** OD detector shifts by 1–2 px under rotation,
moving zone boundaries and reassigning borderline vessel crossings.
Mitigation: detect OD on the original, transform the OD centre with
the same affine, define zones in the transformed space.

### 2. Noise robustness

- Add Gaussian noise at σ = 0, 2, 5, 10, 15, 20, 30 to the raw image.
- Re-extract diameters (keep mask fixed — isolates profile fitting).

**Expected:** flat up to σ ≈ 10 (convolved step model absorbs moderate
noise into σ_PSF), smooth degradation beyond. Compare against a
single-Gaussian baseline — the convolved-step model should degrade
later.

**Target:** < 2 % CRAE change at σ = 10 (typical UWF noise level).

### 3. Segmentation sensitivity

Tests what happens when the A/V mask is imperfect (the realistic case).

- **Pixel dropout** — randomly drop N % of vessel pixels (N = 5, 10,
  20, 30, 50). Simulates segmentation gaps.
- **Segment-level label flip** — randomly swap A ↔ V labels on M % of
  segments (M = 5, 10, 20). Simulates A/V misclassification.
- **Class-systematic erosion** — morphological erosion of veins *only*
  in zones ≥ 4 (simulates LUNet under-segmentation of small peripheral
  veins).
- **Tile-edge A/V flip stripe** — a 10-px-wide A-to-V swap band along
  a simulated tile boundary (simulates tile-boundary flip in LUNet
  tiled inference).

**Expected behaviour:**
- Pixel dropout is well-tolerated (zonal averaging + Knudtson absorb
  local loss).
- Label flips are dangerous: one artery misclassified as a vein moves
  a diameter from CRAE to CRVE, biasing both. At 10 % segment-level
  flip rate expect ~5–8 % CRAE error.
- Class-systematic erosion biases CRVE upward (small veins gone, only
  large ones remain).

**Key metric:** at what A/V accuracy does CRAE error exceed 5 μm?

### Reporting format

Per perturbation type, a table:

```
Perturbation | Level | CRAE_mean | CRAE_std | ΔCRAE_% | CRVE_mean | CRVE_std | ΔCRVE_%
Rotation     |   5°  | 152.3     | 1.2      | 0.8 %   | 228.1     | 1.5      | 0.7 %
...
```

Plus per-quadrant breakdowns (some quadrants, especially sparse ones,
are more sensitive than others).

## Radial profile analysis

Beyond single-zone CRAE / CRVE, the radial profile is a richer
longitudinal feature:

```
distance_dd[z]    = (inner + outer) / 2 for zone z
crae_profile[z]   = CRAE at zone z (per quadrant or averaged)
crve_profile[z]   = CRVE at zone z
```

Fit a model (linear, exponential, or power-law taper) to the radial
profile and track the fitted slope across visits. A steepening slope
(faster peripheral narrowing) may be an early CVD biomarker that
single-zone CRAE misses.

## Clinical validation path

1. **Internal consistency** — test-retest on same-day duplicate images:
   CRAE / CRVE should agree within 3 %.
2. **External consistency** — compare Z1 CRAE / CRVE with IVAN /
   VAMPIRE / SIVA on the same images using `legacy_zone_b=True` mode;
   target r > 0.90.
3. **Clinical utility** — in a cohort with known CVD outcomes, test
   whether peripheral-zone CRAE / CRVE or radial-profile slope add
   predictive value beyond Zone B alone. This is the scientific
   contribution of the project.
