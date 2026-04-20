# 03 — Skeleton Tracing and Segment Splitting

## Purpose

Convert the binary vessel mask into ordered, A/V-labelled vessel
segments that can be walked for diameter sampling. Each segment is a
single unbranched vessel arc split into sub-segments per
(zone, quadrant) cell.

## Input

- `av_mask` (H, W) uint8, 0/1/2 as per 02_SEGMENTATION.md
- `av_probs` (H, W, 2) float32 — required for segment-level A/V
  confidence (spec 03 §Label assignment)
- `RetinalCoordinateSystem` instance
- `ExtractionConfig`

## Step 1 — Skeletonisation

Use `skimage.morphology.medial_axis(binary_vessel_mask, return_distance=True)`.

**Rationale for medial_axis over Zhang-Suen `skeletonize`:**
- `medial_axis` returns, for free, the Euclidean distance from each
  skeleton pixel to the nearest background pixel. Twice that distance is
  a **cheap initial width estimate** (`w_init = 2 · EDT[skel]`) which
  the profile-fitting stage (04) uses to bootstrap Levenberg-Marquardt
  and to adaptively size the profile half-width.
- `medial_axis` is centred on even-width vessels; `skeletonize`
  (Zhang-Suen) can shift by 0.5 px on even widths, biasing diameter at
  the sub-pixel scale.

Before skeletonisation: morphological closing with a disk SE of radius
2 on the combined vessel mask (`av_mask > 0`) to bridge 1–2 px gaps
from LUNet sub-threshold pixels.

Cleanup:
- Remove isolated pixels (0 neighbours).
- Remove spurs shorter than `max(5 px, 0.5 · local_diameter)` — the
  length scale scales with local diameter, so a 5-px threshold is too
  harsh for peripheral thin vessels and too lenient for major arcades
  near the OD.

## Step 2 — Segment tracing

Trace the skeleton into ordered point sequences, splitting at
bifurcations and crossings:

1. Classify skeleton points by 8-connected neighbour count:
   - **Endpoints**: 1 neighbour
   - **Interior**: 2 neighbours
   - **Bifurcations/crossings**: 3 or 4 neighbours (no distinction made at
     this step; step 4 separates true bifurcations from A-over-V crossings)
2. Walk from each endpoint and from each bifurcation along the skeleton,
   recording `(y, x)` in order, until hitting another endpoint or
   bifurcation.
3. Discard segments shorter than 5 pixels (noise fragments).

Output: list of `segment_points` (each a list of `(y, x)`).

### Gap bridging (optional, with guardrails)

The legacy spec proposed endpoint-tangent matching (10 px radius, 30°
angle tolerance) to bridge gaps. This is fragile at crossings. v1
applies the bridge **only if all three guardrails pass**:

- Both endpoints share the same majority A/V label on a 3 × 3
  neighbourhood (no A-over-V bridging).
- The straight line between endpoints traverses pixels with mean
  vessel-probability (`max(p_a, p_v)`) > 0.3.
- The bridge length is < 1.5 × the mean `w_init` of the two endpoints.

Bridging is off by default in v1 (`ExtractionConfig.bridge_gaps =
False`); the skeleton pipeline produces shorter, more fragmented
segments but no spurious A/V fusion. Turn on only for images where the
default LUNet mask has obvious, isolated gaps.

## Step 3 — A/V label assignment (continuous confidence)

Each segment gets a single A/V label from the **mean signed probability
difference** along its constituent pixels:

```python
p_a = av_probs[pixel_ys, pixel_xs, 0]      # shape (N,)
p_v = av_probs[pixel_ys, pixel_xs, 1]
signed_conf = (p_a - p_v).mean()           # in [-1, +1]

label      = 'artery' if signed_conf > 0 else 'vein'
confidence = abs(signed_conf)              # 0 = uncertain, 1 = perfect
```

If a skeleton pixel falls on background (thinning artefact), dilate the
lookup: take the maximum of `p_a − p_v` over a 3 × 3 neighbourhood.

Segments with `confidence < 0.2` are flagged as `uncertain`. These are
included in the output DataFrame with a flag but excluded from Knudtson
aggregation.

Rationale: majority-vote on a hard binary mask (the legacy approach)
discards the continuous confidence signal. LUNet outputs independent
sigmoids per class, and `p_a − p_v` is the natural discriminant.

## Step 4 — Bifurcation and crossing exclusion

A true bifurcation (Y-split) and an A-over-V crossing (X-shape) are
both 3+-neighbour skeleton points but have different physics. v1
handles both with an arc-length exclusion buffer:

1. **Detect crossings** by mask overlap: a 3+-neighbour skeleton point
   where `av_mask == 1` and `av_mask == 2` both appear within a 3-px
   dilation — or equivalently where both `p_a > 0.3` and `p_v > 0.3` in
   the 5 × 5 neighbourhood. These are A-over-V crossings (or unclear
   crossings). Labelled `point_type = "crossing"`.
2. **Detect true bifurcations**: 3+-neighbour points where only one
   A/V class is present. Labelled `point_type = "bifurcation"`.
3. **Exclude an arc-length buffer** around every bifurcation/crossing
   equal to `1.5 · local_w` (where `local_w = 2 · EDT[skel]` at that
   point) before the profile-fitting stage. Profiles within this buffer
   are corrupted by the branching / crossing vessel and should not be
   sampled.

## Step 5 — Split by zone-quadrant cell

For each segment, assign each point to its `(zone_index, quadrant)` cell
using the coordinate system. Group consecutive same-cell points into
sub-segments.

A single vessel segment may span multiple zones (it radiates outward
from the OD) and may curve through a quadrant boundary. Each sub-segment
is the portion of the vessel within one (zone, quadrant) cell.

Discard sub-segments shorter than 5 **samples** (not 5 points — "samples"
is after applying the 0.05 DD sample-interval rule in stage 04), which
corresponds to ~0.25 DD of arc length.

## Output data structure

```python
@dataclass
class VesselSegment:
    segment_id:          int
    vessel_type:         Literal['artery', 'vein']
    label_confidence:    float        # abs mean (p_a - p_v); [0, 1]
    uncertain:           bool         # label_confidence < 0.2
    full_points:         np.ndarray   # (N, 2) ordered (y, x)
    w_init:              np.ndarray   # (N,) 2·EDT[skel] for each point
    point_types:         np.ndarray   # (N,) str: 'interior','bifurcation','crossing','endpoint'
    exclusion_mask:      np.ndarray   # (N,) bool: True = inside buffer, exclude from sampling
    sub_segments:        dict[tuple[int, str], list[int]]
                                      # (zone, quadrant) -> list of indices into full_points
```

The tangent at an internal point of a sub-segment is always computed
from **full_points** (with a 7-point PCA window, see 04), so that
tangent estimation near cell boundaries has context from both sides.

## Notes

- Tangent estimation uses PCA on a 7-point window (not endpoint
  subtraction — chord ≠ tangent on curved UWF peripheral vessels).
- `w_init` from `medial_axis(return_distance=True)` is a per-skeleton-point
  initial diameter estimate. Stage 04 uses it both to initialize the LM
  fit AND to adaptively size the profile half-width.
- This module is pure NumPy / scikit-image and has no ONNX dependency.
