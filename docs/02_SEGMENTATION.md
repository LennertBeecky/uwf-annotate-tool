# 02 — A/V Segmentation

## Purpose

Produce an A/V mask and the underlying per-pixel probability maps at
the native resolution of the (dewarped) UWF frame. These feed the
skeleton and diameter stages.

## Upstream contract

The segmentation layer returns, for each input image:

```python
SegmentationResult(
    av_mask:    np.ndarray,  # (H, W) uint8:  0=bg, 1=artery, 2=vein
    av_probs:   np.ndarray,  # (H, W, 2) float32: ch0 = p_artery, ch1 = p_vein
    retina_mask:np.ndarray,  # (H, W) uint8:  1 inside the valid retinal ROI
    od_probs:   np.ndarray,  # (H, W) float32: p_disc from LUNet-ODC
)
```

- Probabilities are independent sigmoids, **not** a softmax — a single
  pixel can be both artery and vein (crossing). Downstream code must
  respect this and not call `argmax`.
- `av_mask` is derived as: background=0; artery if
  `p_a > 0.5 and p_a > p_v`; vein if `p_v > 0.5 and p_v > p_a`.
- Confidence at each pixel is `p_a − p_v` (signed): useful for segment-level
  A/V confidence (see 03_SKELETON_AND_SEGMENTS.md §Label assignment).

## Model

v1 uses LUNetV2 (Fhima et al.) via ONNX Runtime. Two ONNX files are
required:

- `lunetv2Large.onnx` (810 MB) — A/V vessel segmentation, native input
  1472×1472 RGB, output (1472, 1472, 4) with ch0 = artery, ch1 = vein.
- `lunetv2_odc.onnx` (125 MB) — optic disc/cup segmentation, native
  input 512×512 RGB, output (512, 512, 4) with ch0 = disc, ch1 = cup.

Both are copied into `models/` at project root (gitignored;
`scripts/fetch_models.py` handles distribution for public releases).

## Input channel convention — REPLICATED GREEN

LUNet was trained on CFPs. The vessel-contrast channel for Optos UWF
is the green channel (Optos uses a 532 nm laser); feeding raw BGR into
LUNet substantially degrades A/V separation at the periphery.

**Required:** extract the green channel from the dewarped UWF image
and replicate it three times to form a 3-channel tensor:

```python
green = image_bgr[:, :, 1]
tensor_bgr = cv2.merge([green, green, green])   # "RGB" to LUNet = all-green
```

This is the same trick used in the legacy `hybrid_pipeline.py` and
`vessel_diameter.py` — we now require it in the spec.

## Tiled inference on full UWF

The UWF frame is 4000×4000 (raw) or similar post-dewarping. LUNet's
native input is 1472; running LUNet once on a resized image throws away
peripheral vessel resolution. We run tiled inference with Gaussian
blending.

**Canonical configuration (v1 default):**

- `tile_size = 1472` — matches LUNet native, no up/down-sampling artefact
- `stride = 736` — 50 % overlap
- **reflect-pad** the right and bottom edges of the input image up to the
  next multiple of `stride` before tiling. Accumulator weights below
  `1e-3` are treated as invalid (produce NaN in the prob map) instead
  of being clamped — this prevents the "weight floor amplifies noise at
  uncovered pixels" bug in the legacy wrapper.
- Gaussian weight `exp(-(d/σ)²/2)` with `σ = tile/4`, applied per tile,
  normalized at accumulator output.

**Critical:** do NOT centre-crop the UWF before segmentation. The
legacy `process_uwf.py` crop 1500×1500 only reaches ~3 DD and destroys
the Z6/Z7 story. The dewarped full frame is the correct input.

## LUNet wrapper API (re-authored)

```python
from uwf_zonal_extraction.segmentation import LunetSegmenter, OpticDiscSegmenter

lunet = LunetSegmenter(model_path="models/lunetv2Large.onnx")
odc   = OpticDiscSegmenter(model_path="models/lunetv2_odc.onnx")

# Full-frame tiled (UWF):
av_probs = lunet.predict_tiled(
    image_bgr,              # replicated-green applied internally if requested
    tile_size=1472,
    stride=736,
    green_input=True,       # replicates green channel before inference
    tta=False,              # optional 4-rot + hflip TTA (v1: off by default)
)                           # → (H, W, 4); ch0=artery, ch1=vein

od_probs = odc.predict(image_bgr)          # → (H, W, 4); ch0=disc, ch1=cup
```

The `SegmentationBundle` dataclass groups both models, the fovea
detector, and the dewarp into a single injection point for
`ZonalDiameterExtractor`.

## Known failure modes and mitigations

- **Tile-edge A/V flips.** A vessel straddling a tile boundary can be
  labelled artery on one side and vein on the other. Mitigated by
  Gaussian blending + 50 % overlap, but not eliminated. The skeleton
  stage (03) computes `label_confidence` from `p_a − p_v` along each
  segment — segments with |confidence| < 0.2 are flagged.
- **Systematic under-segmentation of small peripheral veins.** LUNet's
  CFP training set underrepresents the low-contrast small-calibre
  vessels typical of Optos far-periphery. Robustness (06_LONGITUDINAL.md)
  tests this with class-systematic erosion, not just random dropout.
- **Optos red-laser contamination in "green".** Optos separates red
  (633 nm) and green (532 nm) laser channels into R and G, but there is
  some crosstalk. This is an accepted limitation of v1; a Optos-native
  A/V model would remove it but no open-source ONNX is available.

## Test-time augmentation (optional)

`tta=True` averages probability maps across {identity, rot90, rot180,
rot270, hflip, hflip+rot90, hflip+rot180, hflip+rot270}. Typically cuts
A/V flip rate by ~30–50% at ~8× inference cost. Off by default in v1;
enable via `ExtractionConfig.segmentation.tta = True` when accuracy
matters more than throughput.

## Alternative / future work

- **Optos-native A/V segmenter.** Literature candidates (Ding 2020 UCLA,
  Tan 2023 Singapore, VAMPIRE-UWF) would outperform a CFP-trained LUNet
  on 200° UWF, but none is currently available as a redistributable
  ONNX. This is v2 work.
- **Fovea from LUNet-ODC `bg_a`/`bg_b`.** The optic-disc model has two
  "background region" channels whose semantics are unclear; preliminary
  inspection suggests one may correlate with the macula region. Not
  used in v1 but flagged for investigation.
