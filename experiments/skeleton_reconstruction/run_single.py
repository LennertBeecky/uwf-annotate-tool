"""Step 2: run the full pipeline on a SINGLE image, render QC, print metrics.

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/run_single.py \
        [--split Train] [--stem 000036_RR_43001_04734_F_48] \
        [--db databases] [--output-dir experiments/skeleton_reconstruction]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project-local imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import (  # type: ignore  # noqa: E402
    LABEL_ARTERY,
    LABEL_CROSSING,
    LABEL_VEIN,
    combine_av_mask,
    find_triplets,
    load_triplet,
)
from metrics import compute_metrics  # type: ignore  # noqa: E402
from reconstruct import ReconstructCfg, reconstruct_from_gt  # type: ignore  # noqa: E402


def _say(msg: str) -> None:
    print(msg, flush=True)


def overlay_av(image: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Red = artery, blue = vein, magenta = crossing."""
    vis = image.copy()
    color = vis.copy()
    color[labels == LABEL_ARTERY] = [0, 0, 255]
    color[labels == LABEL_VEIN] = [255, 0, 0]
    color[labels == LABEL_CROSSING] = [255, 0, 255]
    return cv2.addWeighted(vis, 1 - alpha, color, alpha, 0)


def difference_map(gt_vessel: np.ndarray, pr_vessel: np.ndarray) -> np.ndarray:
    """Green=agree, red=recon only, blue=GT only, black=both bg."""
    h, w = gt_vessel.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    gt = gt_vessel.astype(bool)
    pr = pr_vessel.astype(bool)
    agree = gt & pr
    only_pr = ~gt & pr
    only_gt = gt & ~pr
    out[agree] = [0, 180, 0]
    out[only_pr] = [0, 0, 255]
    out[only_gt] = [255, 0, 0]
    return out


def render_six_panel(
    image: np.ndarray,
    gt_labels: np.ndarray,
    recon_labels: np.ndarray,
    skel_artery: np.ndarray,
    skel_vein: np.ndarray,
    recon_soft: np.ndarray,
    output_path: Path,
) -> None:
    """2×3 grid: image, GT, recon, skeleton, diff, soft."""
    # Row 1
    gt_overlay = overlay_av(image, gt_labels)
    recon_overlay = overlay_av(image, recon_labels)
    # Row 2
    skel_vis = image.copy()
    skel_vis[skel_artery] = [0, 0, 255]
    skel_vis[skel_vein] = [255, 0, 0]
    diff = difference_map((gt_labels > 0).astype(np.uint8), (recon_labels > 0).astype(np.uint8))
    soft_rgb = np.zeros_like(image)
    soft_rgb[:, :, 2] = (recon_soft[:, :, 0] * 255).astype(np.uint8)  # artery → red
    soft_rgb[:, :, 0] = (recon_soft[:, :, 1] * 255).astype(np.uint8)  # vein → blue

    # Thumb labels
    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return out

    row1 = np.hstack([_label(image, "raw"), _label(gt_overlay, "GT"), _label(recon_overlay, "recon hard")])
    row2 = np.hstack([_label(skel_vis, "skeleton"), _label(diff, "diff G=ok R=recon B=GT"), _label(soft_rgb, "recon soft")])
    grid = np.vstack([row1, row2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="databases", type=Path)
    p.add_argument("--split", default="Train")
    p.add_argument("--stem", default=None,
                   help="File stem. If absent, pick the first triplet in the split.")
    p.add_argument("--output-dir", default="experiments/skeleton_reconstruction", type=Path)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override ReconstructCfg.paint_threshold_frac "
                        "(0.35 = annotator contour, 0.50 = half-max).")
    args = p.parse_args()

    triplets = find_triplets(args.db, splits=[args.split])
    if not triplets:
        _say(f"no triplets found in split={args.split}")
        return 1
    if args.stem is None:
        triplet = triplets[0]
    else:
        picked = [t for t in triplets if t.stem == args.stem]
        if not picked:
            _say(f"no triplet matches stem={args.stem!r}")
            return 1
        triplet = picked[0]

    _say(f"image: {triplet.split}/{triplet.stem}")
    _say(f"  image_path: {triplet.image_path.name}")

    t0 = time.time()
    image, artery, vein = load_triplet(triplet)
    _say(f"  image shape: {image.shape}")
    _say(f"  artery pixels: {int(artery.sum()):,}   "
         f"vein pixels: {int(vein.sum()):,}   "
         f"crossing pixels: {int((artery & vein).sum()):,}")

    gt_labels = combine_av_mask(artery, vein, label_crossing=True)

    cfg = ReconstructCfg()
    if args.threshold is not None:
        from dataclasses import replace
        cfg = replace(cfg, paint_threshold_frac=args.threshold)
    _say(f"Reconstructing (half_width={cfg.half_width_px}, stride={cfg.sample_stride_px}, "
         f"threshold={cfg.paint_threshold_frac})...")
    t1 = time.time()
    recon = reconstruct_from_gt(image, artery, vein, cfg)
    dt = time.time() - t1
    _say(f"  reconstruction done in {dt:.1f}s")

    # Funnel
    a = recon.artery_stats
    v = recon.vein_stats
    _say(
        f"  artery: {a.n_segments} segments, "
        f"{a.n_sample_fits_ok}/{a.n_sample_attempts} fits OK "
        f"({a.n_sample_skipped_junction} junction-skip, "
        f"{a.n_sample_skipped_crossing} crossing-skip, "
        f"{a.n_sample_fits_failed} failed)"
    )
    _say(
        f"  vein:   {v.n_segments} segments, "
        f"{v.n_sample_fits_ok}/{v.n_sample_attempts} fits OK "
        f"({v.n_sample_skipped_junction} junction-skip, "
        f"{v.n_sample_skipped_crossing} crossing-skip, "
        f"{v.n_sample_fits_failed} failed)"
    )

    n_total_attempts = a.n_sample_attempts + v.n_sample_attempts
    n_total_ok = a.n_sample_fits_ok + v.n_sample_fits_ok
    n_gauss = a.n_gaussian_rescue + v.n_gaussian_rescue
    n_thin_fallback = a.n_sample_fallback_thin + v.n_sample_fallback_thin
    success_rate = (n_total_ok + n_gauss) / max(n_total_attempts, 1)
    _say("")
    _say(f"  Phase-1 convolved-step fits: {n_total_ok}/{n_total_attempts}")
    _say(f"  Phase-2 Gaussian rescues:    {n_gauss}")
    _say(f"  σ_PSF calibrated:            "
         f"artery={a.sigma_psf_used:.2f}px  vein={v.sigma_psf_used:.2f}px")
    _say(f"  Final 1.5-px fallback:       {n_thin_fallback}")

    metrics = compute_metrics(
        gt_artery=artery.astype(bool),
        gt_vein=vein.astype(bool),
        recon_hard=recon.recon_hard,
        split=triplet.split,
        filename=triplet.stem,
        n_segments=a.n_segments + v.n_segments,
        n_fit_failures=a.n_sample_fits_failed + v.n_sample_fits_failed,
        r_squared=a.r_squared + v.r_squared,
        widths=a.widths + v.widths,
        fit_success_rate=success_rate,
    )

    _say("")
    _say("--- Metrics ---")
    _say(f"  Dice overall : {metrics.dice_overall:.4f}")
    _say(f"  Dice artery  : {metrics.dice_artery:.4f}")
    _say(f"  Dice vein    : {metrics.dice_vein:.4f}")
    _say(f"  Sensitivity  : {metrics.sensitivity:.4f}")
    _say(f"  Specificity  : {metrics.specificity:.4f}")
    _say(f"  Boundary F1 @1px: {metrics.boundary_f1_1px:.4f}")
    _say(f"  Boundary F1 @2px: {metrics.boundary_f1_2px:.4f}")
    _say(f"  Boundary F1 @3px: {metrics.boundary_f1_3px:.4f}")
    _say(f"  Mean boundary disagreement: {metrics.mean_boundary_disagreement_px:.2f} px")
    _say(f"  Median R²: {metrics.median_r_squared:.3f}")
    _say(f"  Fit success rate: {success_rate:.1%}")
    low_r2 = sum(1 for r in (a.r_squared + v.r_squared) if r < 0.5)
    total_r2 = len(a.r_squared) + len(v.r_squared)
    _say(f"  Fits with R² < 0.5: {low_r2}/{total_r2} ({low_r2 / max(total_r2, 1):.1%})")

    out_png = args.output_dir / "single_image_result.png"
    render_six_panel(
        image=image,
        gt_labels=gt_labels,
        recon_labels=recon.recon_hard,
        skel_artery=recon.artery_skeleton,
        skel_vein=recon.vein_skeleton,
        recon_soft=recon.recon_soft,
        output_path=out_png,
    )
    _say(f"Wrote {out_png}")
    _say(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
