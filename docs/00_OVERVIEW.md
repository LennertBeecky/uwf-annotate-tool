# UWF Zonal Vessel Diameter Extraction Pipeline

## Goal

Take an ultra-wide-field (UWF) retinal image, produce clinically meaningful
vascular-caliber biomarkers (CRAE, CRVE, AVR) per concentric zone and
per quadrant around the optic disc. These biomarkers should be comparable
across patient visits for longitudinal CVD-risk tracking.

## Pipeline stages

```
Input: UWF image (Optos 200°, raw 4000×4000 or similar)
  │
  ├─ 0. Stereographic dewarping              (see 01_COORDINATE_SYSTEM.md §Dewarp)
  │     Optos stereographic → equirectangular using emmetropic model eye
  │     (23.5 mm default, axial length override from sidecar if available).
  │
  ├─ 1. Landmark detection                   (see 01_COORDINATE_SYSTEM.md)
  │     Auto-detect OD (LUNet-ODC + ellipse fit major semi-axis → DD/2)
  │     Auto-detect fovea (arcade-apex heuristic, with manual sidecar override)
  │
  ├─ 2. A/V segmentation                     (see 02_SEGMENTATION.md)
  │     LUNet (CFP-trained) on REPLICATED-GREEN input, tiled (1472, 736)
  │     with reflect-pad at right/bottom edges → (H, W) A/V mask at native
  │     resolution + (H, W, 2) sigmoid probability map.
  │
  ├─ 3. Skeleton + segments                  (see 03_SKELETON_AND_SEGMENTS.md)
  │     medial_axis(return_distance=True) → 1-px skeleton + free w_init.
  │     Trace segments between endpoints/bifurcations.
  │     Assign A/V label from mean (p_a − p_v) along segment pixels.
  │     Exclude bifurcation/crossing zones (arc buffer = 1.5 × local_w).
  │     Split each segment into sub-segments by (zone, quadrant) cell.
  │
  ├─ 4. Zonal diameter extraction            (see 04_DIAMETER_EXTRACTION.md)
  │     Walk each sub-segment at 0.05 DD arc-length steps.
  │     Extract perpendicular intensity profile (bicubic, 0.25 px sampling,
  │     half_width adaptive = max(15, ceil(1.8 × w_init))).
  │     Fit convolved step model (vein) or convolved step + reflex (artery).
  │     σ_PSF fitted as a radial polynomial over the image (NOT a single
  │     global constant) to accommodate Optos peripheral blur.
  │     Take diameter-SE-weighted median per vessel per cell.
  │
  ├─ 5. Knudtson aggregation                 (see 05_KNUDTSON_AGGREGATION.md)
  │     Iterative pairing (largest-with-smallest), using canonical Knudtson
  │     coefficients: 0.88 (arteries) and 0.95 (veins) on √(w₁² + w₂²).
  │     Use 6 largest arterioles AND 6 largest venules per cell, aggregated
  │     independently.
  │
  └─ 6. Longitudinal bootstrap CI             (see 06_LONGITUDINAL.md)
        Bootstrap 1000× (resample-6-of-N); report CRAE/CRVE with 95% CI.
        Cross-visit Δ significant if CIs non-overlapping. Vessel-graph
        matching is scoped out of v1 (optional extra in a future release).

Output:
  • per-vessel measurements (DataFrame)
  • CRAE/CRVE/AVR matrix [zone × quadrant] (DataFrame) in px, DD, and μm
  • radial profile CRAE(distance_dd), CRVE(distance_dd) per quadrant
  • bootstrap 95% CIs
  • QC overlays (zone rings, skeleton, fit examples, per-profile residuals)
  • run metadata (config hash, model SHA256, package version, numpy/scipy
    versions, dewarp parameters, laterality)
```

## Key design decisions

- **Segment on the full UWF frame, not a central crop.** The original
  `process_uwf.py` centre-cropped 1500×1500 (~3 DD reach); Z6/Z7 (3–5 DD)
  require the full periphery. Tiled LUNet inference at (1472, 736) covers
  the entire frame with reflect-padding on the trailing edge.

- **Stereographic dewarping is a first-class pipeline stage, not a
  downstream assumption.** Optos exports are typically not fully dewarped;
  applying a Sagong 2014 / Tan 2021 stereographic→equirectangular warp with
  an emmetropic (or per-patient) model eye keeps pixel scale approximately
  isotropic across the field, so DD-based zones map to constant physical
  arc and perpendicular-to-skeleton profiles are not anisotropically
  distorted in the periphery.

- **Zones in DD units** (fraction of disc diameter) so the pipeline
  transfers across image resolutions, magnifications, and modalities (UWF,
  standard fundus, DRVA) without rescaling heuristics. DD is derived from
  the major semi-axis of an ellipse fit to the optic-disc contour, not
  from √(area/π), because the latter collapses under tilt and partial
  occlusion.

- **Quadrants** (ST/SN/IT/IN) are the four standard 90° sectors defined by
  the OD-fovea axis and its perpendicular. Boundaries are at 0° and ±90°
  relative to the axis (NOT at ±45° and ±135° as the first draft of the
  spec erroneously wrote).

- **Zonal averaging along vessel arcs** rather than single crossing-point
  measurements. Each vessel contributes its SE-weighted median diameter
  within a zone. A backward-compatible `legacy_zone_b` mode is provided
  which emulates IVAN/VAMPIRE's single-crossing measurement at the Z1
  outer boundary for Bland-Altman agreement studies.

- **Convolved step model** (box-PSF + erf) for sub-pixel diameter
  extraction; artery model adds a central light-reflex term (single
  Gaussian by default; twin-Gaussian fallback if single-Gaussian residuals
  show central over-correction). Vessel width `w` is an explicit fit
  parameter, decoupled from optical blur `σ`.

- **σ_PSF as a radial polynomial** across the image, fit from
  high-confidence (R² > 0.85) profiles, binned by distance from image
  centre. A single global `σ` bias peripheral diameters upward in Z5–Z7.

- **Knudtson revised (2003) aggregation** with canonical coefficients
  (0.88 for arteries, 0.95 for veins on √(w₁² + w₂²)) using the 6 largest
  vessels per type per cell, consistent with ARIC / BMES / Rotterdam /
  SIVA / IVAN literature.

- **Quadrant separation** (ST/SN/IT/IN) because focal narrowing in one
  quadrant is a stronger CVD signal than global narrowing (Ikram/SIVA,
  Rotterdam).

- **Units reported in px, DD, and μm.** μm conversion uses a population
  mean DD = 1800 μm by default; when axial length is supplied via sidecar,
  a per-patient Littmann correction is applied. All three columns appear
  in the output DataFrame so downstream users pick the right one.

## Dependencies

Required: `numpy`, `scipy`, `scikit-image`, `opencv-python-headless`,
`pandas`, `pyyaml`, `tqdm`.

Extras:
- `[segmentation]` → `onnxruntime` (CPU) or `onnxruntime-gpu`
- `[viz]` → `matplotlib`
- `[cli]` → `typer`
- `[matching]` → `networkx` (future v2 vessel-graph matching)
- `[dev]` → `pytest`, `pytest-regressions`, `hypothesis`

## File layout

```
project-root/
├── pyproject.toml
├── README.md
├── docs/                                  # the 7 spec files (this one + 01..06)
├── src/uwf_zonal_extraction/
│   ├── __init__.py                        # public API re-exports
│   ├── config.py                          # ExtractionConfig (frozen dataclass)
│   ├── models.py                          # VesselSegment, VesselMeasurement, ZonalResult, ExtractionResult
│   ├── io.py                              # image/mask loaders, laterality, sidecars
│   ├── segmentation/
│   │   ├── lunet.py                       # LunetSegmenter, re-authored
│   │   ├── lunet_odc.py                   # OpticDiscSegmenter, re-authored
│   │   ├── fovea.py                       # arcade-apex heuristic + manual override
│   │   └── bundle.py                      # SegmentationBundle (LUNet + ODC + fovea + dewarp)
│   ├── dewarp.py                          # Optos stereographic → equirectangular
│   ├── coordinate_system.py               # RetinalCoordinateSystem
│   ├── skeleton.py                        # medial_axis, segments, A/V labels, cell split
│   ├── profile_fitting.py                 # convolved step model, radial σ
│   ├── knudtson.py                        # iterative pairing, canonical 0.88/0.95
│   ├── extractor.py                       # ZonalDiameterExtractor orchestrator
│   ├── longitudinal/
│   │   ├── delta.py                       # matrix subtraction, bootstrap CI
│   │   └── matching.py                    # STUB — vessel-graph matching (v2)
│   ├── viz.py                             # overlays, profile plots, matrices
│   ├── utils.py                           # interpolation, tangent, geometry
│   └── cli.py                             # typer entry point
├── models/                                # ONNX files; .gitignored
├── scripts/
│   └── fetch_models.py                    # hash-verified downloader (Zenodo-ready)
├── tests/
│   ├── conftest.py
│   ├── data/                              # synthetic + (later) tiny real
│   ├── test_synthetic.py
│   ├── test_coordinate_system.py
│   ├── test_knudtson.py
│   ├── test_profile_fitting.py
│   └── test_robustness.py
└── examples/
    └── run_single_uwf.py                  # end-to-end demo on one image
```
