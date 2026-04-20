"""Stratified v1 vs v2 comparison on a fixed list of images.

For each image in `--stems`:
  1. Run v1 (reconstruct.py) and v2 (reconstruct_v2.py) on the GT masks
  2. Bin skeleton pixels by GT-derived local vessel width (thin/small/medium/thick)
  3. Per bin: dice, mean absolute width error, fit success rate, N pixels
  4. Emit a table CSV + pretty-print + save a visual comparison figure

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/compare_v1_v2.py \
        --stems 3_A,15_A,17_A \
        --output-dir experiments/v2_test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import combine_av_mask, find_triplets, load_triplet  # type: ignore  # noqa: E402
from metrics import dice as dice_fn  # type: ignore  # noqa: E402
from reconstruct import (  # type: ignore  # noqa: E402
    ReconstructCfg as V1Cfg,
    reconstruct_from_gt as v1_recon,
)
from reconstruct_v2 import (  # type: ignore  # noqa: E402
    ReconstructCfg as V2Cfg,
    reconstruct_from_gt as v2_recon,
)

# GT width bins (pixels). Thresholds matched to where the convolved-step
# model's identifiability breaks down (w ~ 2·σ_PSF).
BIN_EDGES = [0.0, 2.5, 4.5, 8.0, 1e6]
BIN_NAMES = ["Thin", "Small", "Medium", "Thick"]


def _width_map(artery: np.ndarray, vein: np.ndarray) -> np.ndarray:
    """Per-pixel GT vessel width (diameter in px).

    Uses 2 · distance_transform_edt on the OR of artery + vein masks, i.e.
    distance to the nearest background pixel; doubled gives the inscribed
    diameter at each interior pixel.
    """
    vessel = (artery.astype(bool) | vein.astype(bool))
    dt = distance_transform_edt(vessel).astype(np.float32)
    return (2.0 * dt)


def _bin_per_pixel(width_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """For each pixel in `mask`, assign a bin index (0..3) from BIN_EDGES,
    or -1 when outside the mask. Returns an int8 map of shape mask.shape.
    """
    bins = np.full(mask.shape, -1, dtype=np.int8)
    valid = mask.astype(bool)
    w = width_map[valid]
    idx = np.digitize(w, BIN_EDGES[1:-1])  # 0/1/2/3
    bins[valid] = idx.astype(np.int8)
    return bins


def _per_bin_stats(
    gt_vessel: np.ndarray,
    pred_vessel: np.ndarray,
    width_map: np.ndarray,
    gt_width: np.ndarray | None = None,
    pred_width: np.ndarray | None = None,
) -> list[dict]:
    """Compute Dice restricted to each GT width bin + a MAE of local widths.

    Dice per bin: restrict BOTH the numerator AND denominator to pixels
    whose GT width falls in that bin. Skeletonless — works on the dense
    mask.
    """
    rows = []
    bins_gt = _bin_per_pixel(width_map, gt_vessel)
    for b, name in enumerate(BIN_NAMES):
        # Pixels in this bin (inside GT + inside bin)
        bin_gt = (bins_gt == b)
        n_gt = int(bin_gt.sum())
        if n_gt == 0:
            rows.append({
                "bin": name,
                "n_gt_pixels": 0,
                "dice": float("nan"),
                "sens": float("nan"),
                "spec": float("nan"),
                "width_mae_px": float("nan"),
            })
            continue
        # Predicted vessel pixels intersected with this bin region
        bin_pred = pred_vessel.astype(bool) & bin_gt
        # Dice restricted to this bin region:
        #   TP = |GT_bin ∩ pred_bin|
        #   Dice = 2TP / (|GT_bin| + |pred_bin ∪ GT_dilated_bin|)
        # Simpler + still useful: Dice over only the bin region.
        tp = int((bin_gt & pred_vessel.astype(bool)).sum())
        # Sensitivity within the bin
        sens = tp / n_gt if n_gt else float("nan")
        # Specificity and a fair Dice need predicted pixels that fall INTO
        # the bin's "neighbourhood". Use a small dilation of bin_gt to
        # catch painting that spills ±1 px:
        from scipy.ndimage import binary_dilation as _dilate
        bin_region = _dilate(bin_gt, iterations=1)
        bin_fp = int((pred_vessel.astype(bool) & bin_region & ~bin_gt).sum())
        n_region = int(bin_region.sum())
        n_tn = n_region - n_gt - bin_fp
        spec = n_tn / max(n_region - n_gt, 1)
        dice = 2 * tp / max(2 * tp + (n_gt - tp) + bin_fp, 1)

        # Width MAE: compare predicted local width to GT local width at every
        # pixel where BOTH agree on vessel membership.
        width_mae = float("nan")
        if gt_width is not None and pred_width is not None:
            both = bin_gt & pred_vessel.astype(bool)
            if both.any():
                width_mae = float(np.mean(np.abs(pred_width[both] - gt_width[both])))

        rows.append({
            "bin": name,
            "n_gt_pixels": n_gt,
            "dice": dice,
            "sens": sens,
            "spec": spec,
            "width_mae_px": width_mae,
        })
    return rows


def _print_table(per_image: dict[str, dict[str, list[dict]]]) -> None:
    """Pretty-print a side-by-side v1 vs v2 table (one per bin, averaged across images)."""
    print("\n" + "=" * 92)
    print(f"{'Bin':<10} | {'v1: Dice / Sens / MAE':<34} | {'v2: Dice / Sens / MAE':<34}")
    print("-" * 92)
    # Aggregate per bin across images
    stats_v1 = {name: [] for name in BIN_NAMES}
    stats_v2 = {name: [] for name in BIN_NAMES}
    for stem, both in per_image.items():
        for r in both["v1"]:
            stats_v1[r["bin"]].append(r)
        for r in both["v2"]:
            stats_v2[r["bin"]].append(r)

    for name in BIN_NAMES:
        v1 = [r for r in stats_v1[name] if r["n_gt_pixels"] > 0]
        v2 = [r for r in stats_v2[name] if r["n_gt_pixels"] > 0]
        if not v1 or not v2:
            print(f"{name:<10} | (no GT pixels)")
            continue
        d1 = np.mean([r["dice"] for r in v1])
        s1 = np.mean([r["sens"] for r in v1])
        m1 = np.nanmean([r["width_mae_px"] for r in v1])
        d2 = np.mean([r["dice"] for r in v2])
        s2 = np.mean([r["sens"] for r in v2])
        m2 = np.nanmean([r["width_mae_px"] for r in v2])
        print(f"{name:<10} | Dice={d1:.3f} Sens={s1:.3f} MAE={m1:.2f}px | "
              f"Dice={d2:.3f} Sens={s2:.3f} MAE={m2:.2f}px")
    print("=" * 92)


def _run_versions(
    image: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run both v1 and v2, return (v1_hard, v2_hard, v1_soft_sum, v2_soft_sum)."""
    v1 = v1_recon(image, artery, vein, V1Cfg())
    v2 = v2_recon(image, artery, vein, V2Cfg(steger_enabled=True))
    return v1.recon_hard, v2.recon_hard, v1.recon_soft.sum(axis=-1), v2.recon_soft.sum(axis=-1)


def _estimate_pred_width(pred_hard: np.ndarray) -> np.ndarray:
    """Predicted per-pixel vessel width — distance-transform of pred mask."""
    m = pred_hard > 0
    return (2.0 * distance_transform_edt(m).astype(np.float32))


def _render_visual(
    image: np.ndarray,
    gt_labels: np.ndarray,
    v1_hard: np.ndarray,
    v2_hard: np.ndarray,
    output_path: Path,
) -> None:
    """3×3 visual: raw/GT/skel | v1/v2/diff | zoom×3."""
    from run_single import difference_map, overlay_av  # type: ignore

    gt_overlay = overlay_av(image, gt_labels)
    v1_overlay = overlay_av(image, v1_hard)
    v2_overlay = overlay_av(image, v2_hard)
    diff = difference_map(
        (gt_labels > 0).astype(np.uint8),
        (v2_hard > 0).astype(np.uint8),
    )
    # v1 vs v2 diff
    v1_v2_diff = difference_map(
        (v1_hard > 0).astype(np.uint8),
        (v2_hard > 0).astype(np.uint8),
    )

    def _label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return out

    row1 = np.hstack([
        _label(image, "raw"),
        _label(gt_overlay, "GT"),
        _label(v2_overlay, "v2 recon"),
    ])
    row2 = np.hstack([
        _label(v1_overlay, "v1 recon"),
        _label(diff, "v2-vs-GT diff"),
        _label(v1_v2_diff, "v1-vs-v2 diff"),
    ])
    # Zoom into a 300x300 central patch
    h, w = image.shape[:2]
    zy, zx = h // 2 - 200, w // 2 - 400   # shift toward a vessel-rich area
    zy = max(0, min(zy, h - 300))
    zx = max(0, min(zx, w - 300))
    zoom_v1 = v1_overlay[zy:zy + 300, zx:zx + 300]
    zoom_v2 = v2_overlay[zy:zy + 300, zx:zx + 300]
    zoom_diff = v1_v2_diff[zy:zy + 300, zx:zx + 300]
    # Resize each zoom panel to the same size as the main panels so the
    # hstacked row3 has the same total width (3·W) as row1 and row2.
    tile_h = image.shape[0]
    tile_w = image.shape[1]
    zoom_v1 = cv2.resize(zoom_v1, (tile_w, tile_h))
    zoom_v2 = cv2.resize(zoom_v2, (tile_w, tile_h))
    zoom_diff = cv2.resize(zoom_diff, (tile_w, tile_h))
    row3 = np.hstack([
        _label(zoom_v1, "zoom v1"),
        _label(zoom_v2, "zoom v2"),
        _label(zoom_diff, "zoom diff"),
    ])
    grid = np.vstack([row1, row2, row3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


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
    per_image_rows: dict[str, dict[str, list[dict]]] = {}
    # Open the CSV upfront and flush after each image so a later crash
    # doesn't lose earlier results.
    csv_path = args.output_dir / "comparison_table.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["stem", "version", "bin", "n_gt_pixels", "dice",
                         "sens", "spec", "width_mae_px"])
    csv_file.flush()

    for stem in stems:
        if stem not in by_stem:
            print(f"  [skip] stem {stem!r} not found in split {args.split}")
            continue
        triplet = by_stem[stem]
        print(f"\n=== {triplet.split}/{stem} ===", flush=True)
        image, artery, vein = load_triplet(triplet)
        gt_labels = combine_av_mask(artery, vein, label_crossing=True)
        gt_vessel = (artery.astype(bool) | vein.astype(bool))
        gt_width = _width_map(artery, vein)

        t0 = time.time()
        v1_hard, v2_hard, _, _ = _run_versions(image, artery, vein)
        dt = time.time() - t0
        print(f"  v1 + v2 combined wall: {dt:.1f}s")

        v1_width = _estimate_pred_width(v1_hard)
        v2_width = _estimate_pred_width(v2_hard)

        rows_v1 = _per_bin_stats(gt_vessel, (v1_hard > 0), gt_width, gt_width, v1_width)
        rows_v2 = _per_bin_stats(gt_vessel, (v2_hard > 0), gt_width, gt_width, v2_width)

        # Overall Dice for context
        d1 = dice_fn(gt_vessel, v1_hard > 0)
        d2 = dice_fn(gt_vessel, v2_hard > 0)
        print(f"  overall Dice  — v1: {d1:.4f} | v2: {d2:.4f}  (Δ={d2 - d1:+.4f})")

        per_image_rows[stem] = {"v1": rows_v1, "v2": rows_v2}

        # Flush this image's rows immediately so a later crash still leaves
        # per-image CSV data on disk.
        for ver, rows in (("v1", rows_v1), ("v2", rows_v2)):
            for r in rows:
                csv_writer.writerow([stem, ver, r["bin"], r["n_gt_pixels"],
                                     r["dice"], r["sens"], r["spec"], r["width_mae_px"]])
        csv_file.flush()

        # Visual comparison — guarded so any render bug doesn't destroy stats.
        try:
            out_png = args.output_dir / f"{stem}_v1_v2_comparison.png"
            _render_visual(image, gt_labels, v1_hard, v2_hard, out_png)
            print(f"  wrote {out_png}")
        except Exception as exc:  # pragma: no cover
            print(f"  [viz skipped] render error: {exc}")

    csv_file.close()
    print(f"\nCSV → {csv_path}")
    _print_table(per_image_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
