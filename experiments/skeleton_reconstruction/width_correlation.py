"""Build a diameter-correlation plot: GT width vs predicted width for v1 and v2.

For each of the 3 test images:
  1. Run v1 and v2 reconstructions
  2. Skeletonize the GT mask to get centerline pixels
  3. Look up GT, v1, v2 widths at each skeleton pixel via distance transform
  4. Aggregate across images

Produces:
  experiments/v2_test/width_correlation.png  — two scatter panels side by side
  experiments/v2_test/width_correlation.csv  — raw (gt, v1, v2) per skeleton point

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/width_correlation.py \
        --stems 3_A,15_A,17_A --output-dir experiments/v2_test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import find_triplets, load_triplet  # type: ignore  # noqa: E402
from reconstruct import ReconstructCfg as V1Cfg, reconstruct_from_gt as v1_recon  # type: ignore  # noqa: E402
from reconstruct_v2 import ReconstructCfg as V2Cfg, reconstruct_from_gt as v2_recon  # type: ignore  # noqa: E402


def _widths_at_skeleton(
    gt_artery: np.ndarray,
    gt_vein: np.ndarray,
    v1_hard: np.ndarray,
    v2_hard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every GT skeleton pixel, return (gt_w, v1_w, v2_w) triples.

    Width at a pixel = 2 · distance_transform(vessel_mask)[pixel].  For the
    GT, the mask is (artery ∪ vein); for v1/v2, the mask is (hard > 0).
    """
    gt_vessel = (gt_artery.astype(bool) | gt_vein.astype(bool))
    skel = skeletonize(gt_vessel).astype(bool)

    gt_dt = 2.0 * distance_transform_edt(gt_vessel).astype(np.float32)
    v1_dt = 2.0 * distance_transform_edt(v1_hard > 0).astype(np.float32)
    v2_dt = 2.0 * distance_transform_edt(v2_hard > 0).astype(np.float32)

    ys, xs = np.where(skel)
    return gt_dt[ys, xs], v1_dt[ys, xs], v2_dt[ys, xs]


def _scatter_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlim: tuple[float, float] = (0.0, 20.0),
    ylim: tuple[float, float] = (0.0, 20.0),
) -> None:
    # Density-coloured scatter via 2D histogram
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmin=1, cmap="viridis",
                  norm=matplotlib.colors.LogNorm())
    ax.plot([0, 20], [0, 20], "r--", linewidth=1.0, label="y = x")
    # Running median: bin x into integers, show median y per bin
    bin_edges = np.arange(0, 21, 1)
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    medians = []
    p25 = []
    p75 = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 5:
            medians.append(float(np.median(y[mask])))
            p25.append(float(np.percentile(y[mask], 25)))
            p75.append(float(np.percentile(y[mask], 75)))
        else:
            medians.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
    medians = np.array(medians)
    ax.plot(bin_mid, medians, "w-", linewidth=1.5, label="running median")
    ax.fill_between(bin_mid, p25, p75, color="white", alpha=0.25, label="IQR")
    ax.set_xlabel("GT width (px)")
    ax.set_ylabel("Predicted width (px)")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="databases", type=Path)
    p.add_argument("--split", default="Train")
    p.add_argument("--stems", default="3_A,15_A,17_A", type=str)
    p.add_argument("--output-dir", default="experiments/v2_test", type=Path)
    args = p.parse_args()

    triplets_all = find_triplets(args.db, splits=[args.split])
    by_stem = {t.stem: t for t in triplets_all}
    stems = [s.strip() for s in args.stems.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_widths: list[np.ndarray] = []
    v1_widths: list[np.ndarray] = []
    v2_widths: list[np.ndarray] = []
    per_image_n: dict[str, int] = {}

    for stem in stems:
        if stem not in by_stem:
            print(f"  [skip] {stem!r} not found in {args.split}")
            continue
        triplet = by_stem[stem]
        print(f"\n=== {triplet.split}/{stem} ===", flush=True)
        image, artery, vein = load_triplet(triplet)

        t0 = time.time()
        v1 = v1_recon(image, artery, vein, V1Cfg())
        print(f"  v1 done in {time.time() - t0:.1f}s")
        t0 = time.time()
        v2 = v2_recon(image, artery, vein, V2Cfg(steger_enabled=True))
        print(f"  v2 done in {time.time() - t0:.1f}s")

        gt_w, v1_w, v2_w = _widths_at_skeleton(
            artery, vein, v1.recon_hard, v2.recon_hard
        )
        print(f"  skeleton points: {len(gt_w)}")
        gt_widths.append(gt_w)
        v1_widths.append(v1_w)
        v2_widths.append(v2_w)
        per_image_n[stem] = len(gt_w)

    gt_all = np.concatenate(gt_widths)
    v1_all = np.concatenate(v1_widths)
    v2_all = np.concatenate(v2_widths)
    print(f"\nTotal skeleton points across {len(per_image_n)} images: {len(gt_all)}")

    # Dump raw CSV
    csv_path = args.output_dir / "width_correlation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_width_px", "v1_width_px", "v2_width_px"])
        # subsample to 50k rows for CSV sanity (full data → plot only)
        n = len(gt_all)
        if n > 50000:
            idx = np.random.default_rng(0).choice(n, size=50000, replace=False)
            gt_s, v1_s, v2_s = gt_all[idx], v1_all[idx], v2_all[idx]
        else:
            gt_s, v1_s, v2_s = gt_all, v1_all, v2_all
        for a, b, c in zip(gt_s, v1_s, v2_s):
            w.writerow([float(a), float(b), float(c)])
    print(f"CSV → {csv_path}  ({min(n, 50000)} rows)")

    # --- Build the figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Spearman-style robust correlation
    def _robust_corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    r_v1 = _robust_corr(gt_all, v1_all)
    r_v2 = _robust_corr(gt_all, v2_all)
    mae_v1 = float(np.mean(np.abs(v1_all - gt_all)))
    mae_v2 = float(np.mean(np.abs(v2_all - gt_all)))

    _scatter_panel(
        axes[0], gt_all, v1_all,
        f"v1 (conv-step + Gaussian rescue)\n"
        f"r = {r_v1:.3f}, MAE = {mae_v1:.2f} px, n = {len(gt_all):,}",
    )
    _scatter_panel(
        axes[1], gt_all, v2_all,
        f"v2 (Steger sub-pixel snap + rescue)\n"
        f"r = {r_v2:.3f}, MAE = {mae_v2:.2f} px, n = {len(gt_all):,}",
    )
    fig.suptitle(
        f"Diameter correlation: predicted vs GT width at every GT skeleton point  "
        f"({', '.join(stems)})",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = args.output_dir / "width_correlation.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out_png}")

    # Printed summary table
    print("\n=== Summary ===")
    print(f"{'':<12} {'r':>8} {'MAE':>10} {'bias':>10}")
    print(f"{'v1':<12} {r_v1:>8.3f} {mae_v1:>8.2f}px {float(np.mean(v1_all - gt_all)):>+8.2f}px")
    print(f"{'v2':<12} {r_v2:>8.3f} {mae_v2:>8.2f}px {float(np.mean(v2_all - gt_all)):>+8.2f}px")
    # Per-thickness breakdown
    print("\n  Per-thickness MAE breakdown:")
    thresholds = [(0, 2.5, "Thin   (<2.5)"), (2.5, 4.5, "Small  (2.5-4.5)"),
                  (4.5, 8.0, "Medium (4.5-8)"), (8.0, 999, "Thick  (>=8)")]
    for lo, hi, name in thresholds:
        mask = (gt_all >= lo) & (gt_all < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mae1 = float(np.mean(np.abs(v1_all[mask] - gt_all[mask])))
        mae2 = float(np.mean(np.abs(v2_all[mask] - gt_all[mask])))
        print(f"  {name:<20} n={n:>7,}  v1 MAE={mae1:5.2f}px  v2 MAE={mae2:5.2f}px  Δ={mae2 - mae1:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
