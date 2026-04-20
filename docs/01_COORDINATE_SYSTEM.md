# 01 — Retinal Coordinate System

## Purpose

Define a patient-anchored coordinate system centred on the optic disc so
that all downstream measurements (diameters, zones, quadrants) are in
physical units (disc diameters, DD) and invariant to image resolution,
gaze angle, and head tilt.

## Landmarks

Two landmarks are required:

1. **Optic disc centre `(y_od, x_od)` and radius `r_od`** in pixels.
   The disc diameter `DD = 2 · r_od` is the fundamental length unit.
2. **Fovea centre `(y_f, x_f)`** in pixels. The OD-fovea axis defines
   the temporal direction and breaks rotational ambiguity.

### OD detection — two-stage crop + ellipse fit

The CFP-trained `lunet_odc` ONNX returns a disc-probability map from a
512×512 input. On a raw 4000×4000 Optos frame the disc occupies ~2.5%
of the width — at 512×512 that's only ~13 px, below the CFP training
distribution, and the model returns an empty mask. Two-stage detection
avoids this:

1. **Crop** a `crop_size × crop_size` (default 1500) square centred on
   the image centre. In non-steered Optos mode the OD is always near
   the image centre; this is a safe default. (For steered frames, an
   optional `crop_center` override is accepted.)
2. **Run LUNet-ODC** on the crop. At 1500→512 the disc occupies ~6–7%
   ≈ 33–40 px, well inside the CFP training distribution.
3. **Threshold** `p_disc > 0.5` → binary mask.
4. Extract the largest contour (`cv2.findContours`).
5. **Fit an ellipse** (`cv2.fitEllipse`). Use the **major semi-axis**
   as `r_od` (DD/2). Centre comes from the ellipse.
6. **Remap** centre and ellipse axes back to full-frame coordinates by
   adding the crop origin offset.
7. Fallback to `sqrt(area / π)` only if (a) `cv2.fitEllipse` fails, or
   (b) the axis ratio exceeds 2 (non-disc-like shape).

Two-stage is triggered automatically when `max(H, W) >= 2000` pixels
(`SegmentationBundle.od_crop_size_trigger_px`). Standard CFP inputs
(≤ 1600 px) use the direct single-stage path.

Rationale: in UWF the disc is tilted and partially occluded at oblique
gaze; `sqrt(area/π)` under-estimates `r_od` in these cases, which
rescales every zone boundary smaller and silently corrupts longitudinal
comparability. LUNet A/V segmentation still runs on the **full**
uncropped UWF frame — only OD detection uses the crop.

### Fovea detection — arcade-apex heuristic

For UWF both landmarks are usually visible. v1 implements a geometric
heuristic:

1. Expected fovea lies ≈ 2.5 DD temporal from OD centre along the
   mean vessel-arcade curvature axis.
2. Compute the vessel density map (sum of A + V probability) in a
   circular ROI of radius 3 DD centred on the OD.
3. The fovea minimum density is approximately at the **arcade apex**:
   fit a parabola to the superior and inferior vascular-arcade
   centroids, intersect at apex.
4. If confidence < 0.5 (arcade fit poor, or apex outside the retina
   mask), return `None` and emit a warning.

Manual override: if a sidecar JSON is present alongside the image
(`{image_stem}.landmarks.json`) with `{"fovea": [y, x]}`, use it in
preference to auto-detection. The detected-vs-manual choice is logged
in run metadata.

For standard fundus (non-UWF) inputs the fovea may be at the frame edge
or absent. The pipeline should still proceed: if fovea is `None`, set
`axis_angle = 0` (images assumed horizontally aligned, OS/OD laterality
from sidecar) and flag quadrant assignments as `low_confidence=True`.

## Stereographic dewarping (Optos-specific)

**This is a first-class pipeline stage, not a downstream assumption.**

Optos 200° UWF images are stereographic projections of the retinal sphere
onto a 2-D plane about an off-axis virtual centre. Without correction,
pixel scale varies by ≈ 2× between the posterior pole and Z6–Z7 periphery,
and vessel widths measured perpendicular to the skeleton are
anisotropically distorted (radial vs tangential).

### Model

Implement Sagong et al. 2014 / Tan et al. 2021 dewarp:

1. Model eye: emmetropic reduced schematic eye, axial length `AL = 23.5 mm`
   by default. Retinal radius `R_ret ≈ AL − 1.5 mm ≈ 22 mm`.
2. Per-patient override: if a sidecar JSON provides `axial_length_mm`,
   apply a Littmann-like scaling `R_ret_patient = R_ret · (AL_patient / 23.5)`.
3. Compute the forward stereographic map: each pixel `(y, x)` in the raw
   Optos frame corresponds to a retinal arc `(θ, φ)` via the known device
   projection (Optos uses an off-axis ellipsoidal mirror; the published
   projection matrix is device-model dependent — see §Device config).
4. Warp the raw frame to an equirectangular (arc-preserving) grid such
   that `mm_per_px` is approximately constant across the field.
5. Return both the warped image AND a per-pixel `mm_per_px(y, x)` map for
   use in pixel-to-μm conversions on the warped image.

### Device config

The exact stereographic parameters differ between Optos P200 / Daytona /
California / Silverstone. v1 ships with:
- `daytona` (default, most common clinical Optos)
- `california`
- `p200_tx`

Selected via `ExtractionConfig.device = "daytona"`. If the device is
unknown or non-Optos, the user can set `device = "none"` to skip the
warp — zones in Z4+ are then flagged as `dewarp_skipped=True` and should
be interpreted with caution.

### Ordering

**Dewarp BEFORE LUNet inference.** A CFP-trained segmenter is calibrated
to approximately constant mm/px — feeding it raw stereographic UWF
confuses the receptive field and degrades A/V classification in the
periphery. All coordinates downstream of dewarping are in the warped
(equirectangular) frame.

## Concentric zones

Zones are annular rings centred on the OD. Boundaries are multiples of DD:

| Zone | Inner (DD) | Outer (DD) | Notes |
|------|-----------|-----------|-------|
| Z0   | 0.0       | 0.5       | Disc margin — excluded (vessels too thick, reflex-corrupted). |
| Z1   | 0.5       | 1.0       | Standard Zone B (classic CRAE/CRVE zone). Comparator for IVAN/VAMPIRE/SIVA. |
| Z2   | 1.0       | 1.5       | Standard Zone C. |
| Z3   | 1.5       | 2.0       | Near-periphery. |
| Z4   | 2.0       | 2.5       | Mid-periphery. |
| Z5   | 2.5       | 3.0       | Mid-periphery. |
| Z6   | 3.0       | 4.0       | Far-periphery (UWF only). **Research zone — no published norms yet.** |
| Z7   | 4.0       | 5.0       | Far-periphery (UWF only). **Research zone — no published norms yet.** |

Zone boundaries in pixels: `boundary_px = boundary_dd · 2 · r_od`.

Z0 is excluded because vessels near the disc margin are too thick,
overlap, are poorly resolved, and are corrupted by the disc-rim shadow.
Z1 corresponds to the classic "Zone B" — keeping it allows direct
comparison with published CRAE/CRVE norms when combined with the
`legacy_zone_b` aggregation mode (see 05_KNUDTSON_AGGREGATION.md).

For standard fundus images, zones Z6–Z7 will be empty (field of view
≈ 2.5 DD radius). The pipeline returns NaN gracefully.

Z6–Z7 caveats:
- Vessel calibre approaches the imaging resolution limit (`w ≈ 2 · σ_PSF`),
  where the convolved-step model is ill-conditioned. A quality gate
  `w_init > 3 px` rejects fits that are resolution-limited.
- No normative data exists. Present as **research features**, not as
  validated biomarkers, until a normative cohort is published.

## Quadrants

The OD-fovea axis angle:

```python
axis_angle = atan2(-(y_f - y_od), (x_f - x_od))
```

(The y-inversion accounts for image y increasing downward.)

Quadrants are the four standard **90° sectors**, boundaries at 0° and
±90° relative to the axis:

| Quadrant          | Label | Angular range (relative to axis) |
|-------------------|-------|----------------------------------|
| Superior-temporal | ST    | 0° to +90°                       |
| Superior-nasal    | SN    | +90° to +180°                    |
| Inferior-temporal | IT    | 0° to −90°                       |
| Inferior-nasal    | IN    | −90° to −180°                    |

(The previous draft's prose erroneously wrote "boundaries at ±45° and
±135°"; that contradicted the table. The table is correct — standard
SIVA/IVAN convention.)

A point's quadrant:

```python
angle = atan2(-(y - y_od), (x - x_od)) - axis_angle
angle = normalize_to_[-pi, pi](angle)

if   0   <= angle < pi/2:   ST
elif pi/2 <= angle < pi:     SN
elif -pi/2 <= angle < 0:     IT
else:                         IN
```

## Eye laterality

For a **right eye (OD)**, temporal is to the right of the disc in the
standard scanning-laser display; for a **left eye (OS)**, temporal is to
the left. The OD-fovea axis handles this automatically — temporal is
always toward the fovea regardless of which eye.

Laterality metadata comes from:
1. Sidecar JSON (`{image_stem}.meta.json` with `{"laterality": "OD"}`),
2. Filename convention if configured (`ExtractionConfig.laterality_regex`),
3. Auto-inferred from fovea position relative to OD (fallback).

## Pixel-to-μm conversion

After dewarping, the per-pixel map `mm_per_px(y, x)` is approximately
constant across the field. For downstream per-vessel diameter conversion:

```python
diameter_mm = diameter_px * mm_per_px(vessel_y, vessel_x)
diameter_um = diameter_mm * 1000
```

When axial length is NOT supplied, use DD = 1800 μm (population mean) as
a sanity cross-check — the pipeline's implied μm/px and the DD-derived
μm/px should agree within 10%. Large discrepancies are flagged as a
warning (typically indicates wrong device profile or misdetected OD).

## Implementation notes

- Store `od_center`, `od_radius_px`, `od_ellipse`, `fovea_center`,
  `fovea_confidence`, `mm_per_px_map`, `device`, `axial_length_mm` as the
  canonical coordinate-system state. Everything else derives from these.
- Provide `point_to_zone(y, x) → int`, `point_to_quadrant(y, x) → str`,
  `point_to_cell(y, x) → (int, str)`, and `distance_dd(y, x) → float`
  methods on the `RetinalCoordinateSystem` object.
- Coordinates downstream of dewarping are in the warped frame; raw-frame
  coordinates are retained in a separate `RawFrameMap` for QC overlays on
  the original image.
