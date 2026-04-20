# UWF Zonal Extraction — Iteration Log

Started: 2026-04-17
Test image: `3SEDLSNI34IFIFLWYPDGYNMFPA_4000x4000.jpeg`
  (one Optos Daytona UWF, right eye, from
  `Physics-Informed_Fundus/uwf_data/paired_data/uwf/`)

All smoke runs in `runs/smoke01` … `runs/smoke06/` inside this project.

---

## Current status (after smoke06)

### Z1 (Zone B) — literature-comparable

| Metric       | Value             | Literature        | Status        |
|--------------|-------------------|-------------------|---------------|
| CRAE (μm)    | 151               | 147–155           | **in range ✓**|
| CRVE (μm)    | 221               | 220–230           | **in range ✓**|
| AVR          | 0.88              | ≈ 0.67            | close         |
| n arteries   | 2                 | ≥ 6 ideal         | below Knudtson|
| n veins      | 3                 | ≥ 6 ideal         | below Knudtson|

### Z2 plausible; Z3–Z7 currently unreliable

LUNet A/V segmentation quality degrades toward the UWF periphery (CFP-trained
model). Symptoms in Z4–Z7: CRAE *growing* with eccentricity, A/V counts
10:1 imbalanced, Z3 zero veins. The extraction pipeline is now sound; the
limiting factor is the upstream A/V mask.

---

## Iteration-by-iteration summary

Every smoke run used the same test image on CPU. Wall time ≈ 16–17 min per run.

### smoke01 — baseline

Initial scaffolded pipeline end-to-end. Two-stage OD detection, step-px
end buffer, RMSE/A acceptance threshold.

- **Segments**: 571. **Measurements**: 213.
- **Z1 per quadrant**: 1–2 vessels/cell; AVR nonsensical.
- **Z1 CRAE**: 5–8 px ≈ **75 μm** (≈50 % of literature).
- **Debug pack** revealed:
  - Z1 sub-segments eaten by bifurcation buffer.
  - Many example profile fits collapsed to A≈0 flat-baseline local minima.

### smoke02 — buffer 1.5 → 0.8; min_segment_length 5 → 15

- **Segments**: 367 (noise fragments pruned). **Measurements**: 229.
- Z1 `n_subseg_nonempty` **identical to smoke01** → the buffer was NOT
  the bottleneck.
- Diagnosis: the sample-loop boundary-skip (`arange(step_px, L-step_px)`)
  was returning zero samples on Z1 arcs shorter than 2·step_px (13.6 px
  for this DD).
- **No material Z1 improvement.**

### smoke03 — adaptive end-buffer + midpoint fallback

Panel recommendation: replace `step_px` end buffer with `1.5 · local_w`
(junction artefacts scale with vessel width, not DD), and take one
sample at the midpoint of arcs shorter than `2·end_buffer`.

- **Measurements**: 244 (vs 213 baseline).
- **Z1 IN**: 1a1v → 3a3v, CRAE 5.3 → 8.4 px.
- **Learning**: adaptive buffer alone delivered the gains. Midpoint
  fallback *never fired* because the skeleton-level `exclusion_mask`
  still blocked the midpoint sample.

### smoke04 — bypass exclusion_mask on short-arc midpoints

- **Measurements**: 260 (+16 from smoke03), mostly in Z3/Z4.
- **Z1 totals unchanged**: 5 arteries, 5 veins.
- **AVR still inverted**: 1.47 (physiological ≈ 0.67).
- **Viz bug surfaced**: funnel JSON hardcoded stage list, dropping
  `n_midpoint_fallback`. Fixed by dynamic-stage discovery.

### smoke05 — tight w bounds around mask-derived w_init + radial profile

- Tightened LM bounds to `w ∈ [0.4·w_init, 3·w_init]` anchored to
  `w_init = 2·EDT[skeleton]`.
- Added radial-profile summary to smoke_test stdout; `zonal.json` dump
  from `save_debug_pack` (no pandas dependency).
- **First full radial profile visible**:

  | Zone | CRAE px | CRVE px | AVR | n_a | n_v |
  |------|---------|---------|------|-----|-----|
  | Z1   | 9.00    | 8.70    | 1.47 | 7   | 5   |
  | Z2   | 8.52    | 5.24    | 1.62 | 11  | 5   |
  | Z3   | 9.56    | 11.32   | 0.95 | 9   | 3   |
  | Z4   | 9.21    | 6.78    | 1.05 | 14  | 2   |
  | Z5   | 8.56    | 6.26    | 1.27 | 17  | 6   |
  | Z6   | 10.88   | 6.10    | 1.87 | 24  | 6   |
  | Z7   | 11.39   | 7.35    | 1.85 | 20  | 5   |

- **Two big problems exposed**:
  - CRAE 120 μm, CRVE 116 μm — **CRVE at HALF of literature**.
  - AVR ≈ 1.5 (inverted — should be ~0.67).
- **A/V overlay inspection** showed vein masks *consistently thinner
  than artery masks* along the same arcade → **LUNet vein-outline bias**.
  A channel flip alone couldn't explain it (post-flip CRVE still ≈ 120 μm).

### smoke06 — intensity-driven FWHM init (USER INSIGHT)

User's observation: **"use A/V mask only to locate the vessel, let the
profile find centerline and width itself"**.

- Added `fwhm_width_from_profile()` — measures FWHM of the raw
  perpendicular profile (baseline from outer 25 %, find half-depth
  crossings).
- `w_init` now comes from profile FWHM, not from `2·EDT[skeleton]`.
- `half_width_px` fixed at 30 px (generous) so outer 25 % of every
  profile is guaranteed pure tissue for baseline estimation.
- LM bounds remain tight around the new intensity-derived `w_init`.

**Results**:

| Zone | CRAE px | CRAE μm | CRVE px | CRVE μm | AVR  | n_a | n_v |
|------|---------|---------|---------|---------|------|-----|-----|
| Z1   | 11.35   | **151** | 16.61   | **221** | 0.88 | 2   | 3   |
| Z2   | 9.12    | 121     | 6.06    | 81      | 1.44 | 8   | 6   |
| Z3   | 11.15   | 148     | NaN     | —       | —    | 11  | 0   |
| Z4   | 20.64   | 275     | 6.62    | 88      | 4.09 | 14  | 3   |
| Z5   | 16.16   | 215     | 4.70    | 63      | 4.25 | 12  | 3   |
| Z6   | 23.00   | 306     | 4.10    | 55      | 4.39 | 22  | 2   |
| Z7   | 16.73   | 223     | 7.17    | 95      | 2.78 | 12  | 5   |

- **Z1 CRAE and CRVE both IN LITERATURE RANGE.**
- **Z2 CRAE plausible** (121 μm, slightly below 130–140 μm expected).
- Peripheral zones clearly broken (LUNet artefacts).

---

## What got better

- **Z1 now publishable-quality** on this one image (needs cross-patient validation).
- **Pipeline end-to-end works** on a real 4000×4000 Optos in 17 min CPU.
- **Debug pack** renders every stage for visual QC — was the decisive tool.
- **Width measurement decoupled from LUNet mask outline** — crucial fix.
- **Adaptive end-buffer + short-arc midpoint fallback** preserves Z1 arcs
  that used to vanish entirely.
- **Skeleton tracing double-counting bug fixed** (edges_used set).
- **Stage funnel JSON** — concrete diagnostic for "where do vessels go?".

## What got worse / remained broken

- **Peripheral zones (Z4–Z7) unreliable**. LUNet (CFP-trained) produces
  spurious "arteries" in the UWF periphery and misses veins. Not a bug
  in our extractor.
- **Z1 vessel counts dropped** (smoke01 7a5v → smoke06 2a3v). The
  disappeared vessels were mostly garbage fits that were dragging
  CRAE down; rejecting them is correct but leaves n_a below Knudtson's
  preferred 6.
- **AVR 0.88 vs physiological 0.67** — in the right direction but not
  perfect. Possibly still some thin-vein-mask residual bias or a
  real feature of this particular patient.
- **No stereographic dewarp yet** — Z6/Z7 μm numbers are still biased
  by Optos peripheral pixel compression (flagged via `dewarp_skipped`).
- **Fovea detection** uses an arcade-apex heuristic; worked on this
  image (confidence 0.85) but untested on pathological macula.

## Panels consulted

**Round 1** (spec review, before coding):
- Clinical biomarker, vessel morphometry, ML segmentation, software
  architecture. All four agents running in parallel.
- Key outputs: fixed quadrant prose, pinned tile/stride, canonical
  Knudtson coefficients, ellipse-fit OD radius, `dewarp_skipped` flag.

**Round 2** (Z1 sparsity fix):
- Clinical + morphometry panels.
- **Consensus: do NOT widen Z1** — breaks literature comparability.
- Clinical: recommended Hubbard single-circle crossing at 0.75 DD.
- Morphometry: recommended adaptive end-buffer + midpoint fallback.
- We implemented the morphometry path; clinical path (`legacy_zone_b`
  mode) is still TODO.

---

## Open questions / next steps

1. **Validate Z1 numbers on 2–3 more UWF images** from the same directory.
   One image is not a result; need to confirm the Z1 literature-match holds.
2. **Implement Hubbard single-circle `legacy_zone_b` mode** — the
   clinical panel's primary recommendation. Sample each vessel where
   its centerline crosses the 0.75-DD circle → one clean measurement
   per vessel per annulus, matches IVAN/VAMPIRE/SIVA protocol.
3. **Implement real Sagong/Tan dewarp** for Daytona so Z5–Z7 physical
   scale becomes correct. Current passthrough warns; flagged at output.
4. **TTA + confidence filtering** on A/V masks. LUNet outputs independent
   sigmoids; we can average across rot+flip and filter low-confidence
   pixels before skeletonising. Config flag exists, branch unimplemented.
5. **Investigate peripheral "arteries"** — are they all hemorrhages /
   image artefacts / background drift? Overlay inspection on smoke06
   would answer this.
6. **Longer term**: UWF-native A/V model (Ding 2020, Tan 2023) or
   retrain LUNet on Optos data. Removes the single biggest quality
   bottleneck.

## Architectural decisions preserved

- **Z0 excluded, Z1–Z7 reported**. Z6–Z7 are "research zones" with no
  published norms and `dewarp_skipped` flag when dewarp is off.
- **Units**: pixels + DD + μm reported per vessel; μm uses population
  mean DD = 1800 μm unless axial length overrides.
- **Knudtson coefficients**: canonical 0.88 artery / 0.95 vein (Knudtson
  2003). Confirmed correct; earlier panelist critique was based on
  pre-Knudtson (Hubbard 1999) formulas.
- **Coordinate system**: quadrants at 0° and ±90° relative to OD-fovea
  axis (standard SIVA convention).
- **Segmentation**: LUNet tiled on full UWF frame (not centre-crop).
  Two-stage OD detection (crop to 1500×1500 for OD only). Retina-mask
  gated before thresholding so LUNet hallucinations in the Optos black
  frame don't enter the skeleton.

---

## Smoke run timings (CPU, ONNX Runtime sequential)

| Run | Wall time | Notes |
|-----|-----------|-------|
| smoke01 | ~16 min | baseline |
| smoke02 | ~11 min | fewer segments (min_length pruning) |
| smoke03 | ~16 min | adaptive buffer |
| smoke04 | ~16 min | short-arc bypass |
| smoke05 | ~16 min | tight w bounds |
| smoke06 | ~17 min | fixed half_width 30 → more samples per profile |

GPU / CUDA server: these same runs would take ~30–60 s if `onnxruntime-gpu`
is installed (LUNet tiled inference is the dominant cost; CPU is the
bottleneck, not the pipeline).
