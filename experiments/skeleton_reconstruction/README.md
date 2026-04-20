# Skeleton-reconstruction validation experiment

Validates whether skeletonising the GT A/V masks + reconstructing the full
mask via the convolved-step profile fit yields a mask that matches the
original GT (i.e. skeleton-only annotation is equivalent to pixel-wise
annotation for subsequent training).

## Layout

```
experiments/skeleton_reconstruction/
├── data.py          — dataset loader (Train + Test, stem matching, A/V combiner)
├── reconstruct.py   — skeletonise → trace → profile-fit → paint oriented rectangles
├── metrics.py       — Dice, sensitivity/specificity, boundary F1, disagreement
├── run_single.py    — Step 2: one image end-to-end + 6-panel QC PNG
├── run_batch.py     — Step 4: all Train+Test in parallel → results.csv (resumable)
├── analyze.py       — Step 5: summary.txt + 6 plots (+ worst/best-case grids)
└── README.md
```

Shared models come from the `uwf_zonal_extraction` package
(`src/uwf_zonal_extraction`), specifically
`profile_fitting.fit_vessel_profile`, `skeleton.trace_segments`,
`utils.tangent_pca`, `utils.perpendicular`.

## Quick-start

### 1. One image (sanity check)

```bash
cd /path/to/project-root    # local: ~/Documents/PHD/Papers/UWF_Zonal_Extraction; brigand: /esat/biomeddata/users/lbeeckma/uwf
PYTHONPATH=src python -u experiments/skeleton_reconstruction/run_single.py \
    --split Train
```

Writes `experiments/skeleton_reconstruction/single_image_result.png` — a
2×3 grid (raw, GT, reconstruction, skeleton, diff, soft mask).

### 2. Full batch (~587 images, for the server)

```bash
cd /path/to/project-root    # local: ~/Documents/PHD/Papers/UWF_Zonal_Extraction; brigand: /esat/biomeddata/users/lbeeckma/uwf
PYTHONPATH=src python -u experiments/skeleton_reconstruction/run_batch.py \
    --workers $(nproc) \
    --db databases \
    --output-dir experiments/skeleton_reconstruction
```

- Processes Train (508) + Test (79) triplets.
- Multiprocessing with ProcessPoolExecutor.
- Checkpoints `results.csv` every 10 completions — safe to resume after
  crash (will skip stems already in the CSV).
- Failures are logged to `errors.log`; the run continues.
- On a 4-core laptop: ≈ 26 h. Server with 32+ cores: ≈ 2–4 h.

### 3. Analyse

```bash
PYTHONPATH=src python -u experiments/skeleton_reconstruction/analyze.py
```

Writes:
- `summary.txt` — overall + per-split stats and the headline
- `dice_histogram.png`
- `boundary_agreement.png` (BF1 vs tolerance with 95 % bootstrap CI)
- `disagreement_vs_vessel_width.png`
- `r_squared_distribution.png`
- `worst_cases.png` + `best_cases.png` (6 images each, re-reconstructed;
  add `--skip-case-grids` to skip, saves ~10 min)

## Success criteria (per the spec)

| Metric                              | Target       |
|-------------------------------------|--------------|
| Mean Dice overall                   | **> 0.85**   |
| Median Boundary F1 @2 px            | **> 0.80**   |
| Fit success rate                    | > 85 %       |
| Mean boundary disagreement          | < 1.5 px     |

## Current state (3-image smoke, after Step 3 fixes)

```
Dice overall       : 0.69 ± 0.01
Dice artery / vein : 0.65 / 0.70
Sensitivity        : 0.66 ± 0.08
Specificity        : 0.98 ± 0.01
Boundary F1 @2 px  : median 0.49
Boundary F1 @3 px  : median 0.64
Fit success rate   : 60 %
```

The three test images are in the lower half of difficulty (dense CFP A/V
networks with many bifurcations). Full 587-image batch is expected to
have a better distribution — publish the distribution plots once the
server batch finishes.

## Key design choices

- **Skeleton from GT mask only** — no information leaks from the GT extent
  into the reconstruction. Width comes from profile-intensity fit alone.
- **Oriented rectangles, not thin normals** — each skeleton sample paints
  a `(2·tangent_half) × fit.w` rectangle aligned with (tangent, perp),
  so consecutive samples at stride=2 stitch into continuous vessels.
- **Tight junction / crossing skip** (2 px / 4 px) — prevents painting
  through bifurcation and A/V-crossing neighbourhoods where the
  perpendicular profile is corrupted, but keeps gaps small enough that
  morphological closing (radius 1) fills them.
- **Intensity-driven FWHM init** (`fwhm_min_depth=1.5`) — the profile's
  own half-maximum width seeds the LM fit. Untethers width estimation
  from the (possibly too-thin) LUNet mask outline. Falls back to
  `2 · EDT[mask]` if the profile is too shallow to measure.
- **Thin-vessel fallback** — if the LM fit quality is poor and the mask
  width is below `thin_vessel_fallback_px × 2`, paint the oriented
  rectangle with the mask's local distance-transform width. Guarantees
  coverage on very thin vessels where profile fitting is unreliable.
- **Post-reconstruction morphological closing (radius 1)** — closes
  residual 1-px gaps left by rejected samples.

## Server run script

See `run_on_server.sh` in the project root (if present) or run manually
with the commands above. Set `OMP_NUM_THREADS=1` to prevent BLAS
thread-oversubscription when using multiple Python workers:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u experiments/skeleton_reconstruction/run_batch.py \
        --workers $(nproc) --db databases
```
