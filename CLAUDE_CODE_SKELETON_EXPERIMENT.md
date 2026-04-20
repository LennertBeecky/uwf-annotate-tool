# Claude Code: Skeleton Reconstruction Validation Experiment

## Context

We have ~600 retinal fundus images with pixel-wise A/V segmentation ground truth masks. We want to validate that skeleton-only annotation + physics-based boundary reconstruction can reproduce the full mask.

The diameter extraction code already exists in `uwf_zonal_extraction.py` — specifically the `fit_vessel_profile`, `convolved_step_profile`, `convolved_step_with_reflex`, `extract_perpendicular_profile`, `estimate_tangent`, and `trace_skeleton_segments` functions. Reuse these directly.

## Command for Claude Code

```
Implement a skeleton reconstruction validation experiment. The goal: prove that skeletonizing ground truth A/V masks and reconstructing full masks via convolved step model profile fitting produces masks that match the original ground truth.

### Step 1: Setup and data loading

The dataset is in `databases/` with this structure:
```
databases/
├── Train/
│   ├── images/       # raw fundus images
│   ├── artery/       # binary artery masks
│   └── veins/        # binary vein masks (note: "veins" not "vein")
└── Test/
    ├── images/
    ├── artery/
    └── veins/
```

Load BOTH Train and Test sets for this experiment (we're not training a model yet — just validating the reconstruction method on all available GT). Match files across the three subdirectories by filename stem. Inspect a few masks to confirm the binary encoding (check unique values — could be 0/1 or 0/255). Print: number of matched triplets found per split, image dimensions, mask value ranges.

Combine the two masks into a single label map for processing: `combined[artery > 0] = 1`, `combined[vein > 0] = 2`. Where both masks overlap (crossing vessels), mark as 3 (crossing) — print how many pixels fall in this category.

Create `experiments/skeleton_reconstruction/` as the working directory.

### Step 2: Single-image pipeline (get this working first on ONE image before scaling)

Pick one image-mask pair. Implement this pipeline:

a) **Skeletonize the GT mask**
   - The artery and vein masks are already separate files — skeletonize each independently using `skimage.morphology.skeletonize`
   - Combine into a single skeleton image: skeleton_labeled where 1=artery skeleton, 2=vein skeleton
   - Where both skeletons overlap (at crossings), keep the label from whichever original mask had a larger area locally
   - Clean: remove isolated pixels (0 neighbors) and spurs < 5px

b) **Trace skeleton segments**
   - Reuse `trace_skeleton_segments()` from uwf_zonal_extraction.py
   - Assign A/V label per segment by majority vote on the labeled skeleton
   - Print: number of segments found, artery vs vein count, mean segment length

c) **Reconstruct mask from skeleton + raw image**
   For each skeleton segment:
   - Walk along the segment at 1-pixel intervals (dense sampling — we want full reconstruction)
   - At each point, estimate tangent and extract perpendicular profile from the RAW IMAGE (not the mask)
   - Use the green channel of the raw image for profile extraction
   - Fit `convolved_step_profile` for veins, `convolved_step_with_reflex` for arteries
   - From the fitted parameters (center c, width w, PSF sigma), compute the vessel extent: for each pixel position x along the normal, compute the model intensity and threshold at 50% of absorption depth A to get the vessel boundary
   - Better approach: use the fitted w directly — the vessel occupies [c - w/2, c + w/2] along the normal direction. Paint those pixels into the reconstructed mask with the appropriate A/V label.
   - For soft mask: instead of hard threshold, store the normalized model response as a probability: P(x) = (B - I_model(x)) / A, clipped to [0, 1]. This gives a soft label where edge pixels have values between 0 and 1.
   
   Generate TWO reconstructed masks:
   - `recon_hard`: binary mask (0/1/2) using w as the vessel width
   - `recon_soft`: float mask (0.0 to 1.0 per class) using the model profile shape

d) **Compare against original GT**
   Combine the original separate artery and vein masks into a single GT for comparison (same combination as Step 1: artery=1, vein=2, overlap=3).
   
   Compute for the hard reconstruction:
   - Dice coefficient (overall vessel, artery-only, vein-only)
   - Sensitivity (recall) and specificity
   - Boundary F1 at tolerance 1px, 2px, 3px (use `skimage` boundary detection — compare boundary pixels between GT and reconstruction)
   - Per-pixel agreement map: where do GT and reconstruction disagree?

   Print all metrics. Save a visualization showing:
   - Row 1: raw image, GT mask overlay, reconstructed mask overlay
   - Row 2: skeleton overlay, difference map (green=agree, red=recon but not GT, blue=GT but not recon), soft mask
   Save as `experiments/skeleton_reconstruction/single_image_result.png`

e) **Diagnose disagreements**
   For pixels where GT and reconstruction disagree:
   - Measure the distance from the skeleton (are disagreements at vessel edges? — they should be)
   - Compute mean disagreement distance in pixels
   - Print: "Mean boundary disagreement: X.X pixels"
   
   Also compute: what is the R² distribution of the profile fits? Low R² fits may produce bad reconstruction — print the percentage of fits with R² < 0.5.

### Step 3: Fix any issues found in Step 2

Common issues to watch for:
- Profile fitting failures near bifurcations → skip points within 5px of bifurcation nodes
- Crossing vessels producing wide fits → detect where artery and vein skeletons are within 10px of each other, skip those regions
- Very thin vessels (< 2px) where fitting is unreliable → fall back to distance transform width from the GT skeleton
- sigma initialization issues → fit sigma globally first on high-R² fits, then fix it for re-fitting

After fixing, re-run on the same image and confirm metrics improve.

### Step 4: Batch processing on full dataset

Scale to all ~600 images:

a) Run the pipeline on every image-mask pair. For each, store:
   - split ('Train' or 'Test')
   - filename
   - dice_overall, dice_artery, dice_vein
   - sensitivity, specificity  
   - boundary_f1_1px, boundary_f1_2px, boundary_f1_3px
   - n_segments, n_fit_failures, median_r_squared
   - mean_boundary_disagreement_px

b) Save per-image results to `experiments/skeleton_reconstruction/results.csv`

c) Use multiprocessing (ProcessPoolExecutor, n_workers=cpu_count-1) for speed. Add a progress bar (tqdm).

d) Handle failures gracefully — if one image fails, log the error and continue.

### Step 5: Analysis and visualization

From results.csv:

a) Print summary statistics (report overall AND per-split Train/Test):
   - Mean ± std Dice (overall, artery, vein)
   - Median boundary F1 at each tolerance
   - Percentage of images with Dice > 0.90, > 0.85, > 0.80
   - Distribution of fit R² values across all images

b) Generate plots saved to `experiments/skeleton_reconstruction/`:
   - `dice_histogram.png`: histogram of per-image Dice scores (overall, artery, vein overlaid)
   - `boundary_agreement.png`: boundary F1 vs tolerance curve (mean with 95% CI band)
   - `disagreement_vs_vessel_width.png`: scatter of mean boundary disagreement vs mean vessel width per image
   - `r_squared_distribution.png`: histogram of per-fit R² values
   - `worst_cases.png`: visualization grid of the 6 images with lowest Dice — show raw, GT, reconstruction, difference for each. These are the failure cases to inspect.
   - `best_cases.png`: same grid for the 6 images with highest Dice

c) Print a summary conclusion:
   "Reconstruction Dice: {mean}±{std} | Boundary F1@2px: {mean}±{std} | Fit success rate: {pct}%"

### Important implementation notes

- Ground truth is in `databases/Train/` and `databases/Test/`, each containing `images/`, `artery/`, `veins/` (note the "veins" spelling with an s). Match files by filename stem across the three subdirectories.
- Use BOTH Train and Test for this experiment — we are validating reconstruction, not training a model yet.
- The raw image green channel is used for profile extraction, NOT the mask
- The skeleton comes from the GT mask (simulating what an annotator would draw)
- The reconstruction uses ONLY skeleton + raw image (the GT mask is only used for evaluation)
- Profile half-width should be 15-20 pixels to capture the full vessel cross-section
- Use bilinear interpolation for sub-pixel profile sampling (already in the existing code)
- For the soft mask: the erf() profile naturally gives you the transition — normalize it to [0,1]
- Save intermediate results frequently so we don't lose progress if something crashes
- Print progress every 50 images during batch processing

### File structure
```
experiments/skeleton_reconstruction/
├── single_image_result.png      # Step 2 visualization
├── results.csv                  # Per-image metrics
├── dice_histogram.png           # Step 5 plots
├── boundary_agreement.png
├── disagreement_vs_vessel_width.png
├── r_squared_distribution.png
├── worst_cases.png
├── best_cases.png
└── summary.txt                  # Final summary statistics
```
```

## What success looks like

- **Dice > 0.85 mean**: the reconstruction closely matches manual annotation
- **Boundary F1@2px > 0.80**: edges agree within 2 pixels (typical inter-grader variability)
- **Fit success rate > 85%**: most vessel profiles can be fitted reliably
- **Boundary disagreement < 1.5 px mean**: disagreements are sub-pixel, at vessel edges

If we hit these numbers, skeleton annotation is validated as equivalent to full pixel-wise annotation for training A/V segmentation models.

## After this experiment

If results are positive, the next experiment is:
1. Split the 600 images into train/test (e.g., 480/120)
2. Train LUNetV2 on original GT masks → evaluate on test set → baseline Dice
3. Train LUNetV2 on skeleton-reconstructed soft masks → evaluate on same test set → compare
4. If comparable or better → skeleton annotation is validated for training, not just reconstruction
