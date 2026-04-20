# UWF Zonal Vessel-Diameter Extraction

Extract CRAE / CRVE / AVR vascular-caliber biomarkers per concentric
zone and quadrant around the optic disc, from ultra-wide-field (UWF)
retinal images.

## Design

See `docs/00_OVERVIEW.md` for the full pipeline and the motivation
behind each design decision. Briefly:

1. Stereographic dewarp (Sagong 2014 / Tan 2021) BEFORE segmentation
2. LUNet tiled A/V segmentation on replicated-green input
3. Ellipse-fit optic disc + arcade-apex fovea → retinal coordinate system
4. Medial-axis skeleton with free initial width estimate (`EDT · 2`)
5. Bifurcation/crossing exclusion + per-cell splitting
6. Adaptive-width perpendicular profiles, convolved-step-model fit
   (soft-L1 loss), radial σ_PSF map
7. Inverse-variance-weighted zonal medians → Knudtson (0.88 / 0.95)
8. Bootstrap-CI longitudinal delta (v1); vessel-graph matching is v2

Outputs in pixels, DD (disc-diameters), and μm with per-patient axial
length override.

## Install

```bash
conda activate physics-informed_fundus      # or any Python >= 3.10 env
cd /path/to/project-root    # local: ~/Documents/PHD/Papers/UWF_Zonal_Extraction; brigand: /esat/biomeddata/users/lbeeckma/uwf
pip install -e ".[segmentation,viz,cli]"
# or, for full dev setup:
pip install -e ".[segmentation,viz,cli,dev]"
```

## Models

ONNX weights are not in git (too large). Copy from the parent project:

```bash
cp ../Physics-Informed_Fundus/lunet/lunetv2Large.onnx  models/
cp ../Physics-Informed_Fundus/lunet/lunetv2_odc.onnx   models/
```

Or fetch via the (stub) download script once Zenodo distribution is
set up:

```bash
python scripts/fetch_models.py
```

## Quick start

```bash
uwf-extract run --image path/to/uwf.jpeg --output-dir runs/demo01
```

Or from Python:

```python
from uwf_zonal_extraction import extract_caliber_from_image

result = extract_caliber_from_image("path/to/uwf.jpeg")
df = result.to_dataframe()
print(df.head())
```

## Status

v1, alpha. Stable pieces: coordinate system, Knudtson aggregation,
config. Work in progress: stereographic dewarp, fovea detector, σ_PSF
radial polynomial, CLI, visualisations, robustness tests.
