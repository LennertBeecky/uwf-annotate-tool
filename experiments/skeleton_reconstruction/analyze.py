"""Step 5: analyse results.csv — summary stats + the 6 plots + summary.txt.

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/analyze.py \
        [--results experiments/skeleton_reconstruction/results.csv] \
        [--db databases] \
        [--output-dir experiments/skeleton_reconstruction]

The `--db` argument is needed for the worst/best-case image grids, which
reload the raw images + masks and re-run reconstruction on just those 12
images (not the full 587) to render the panels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import Triplet, combine_av_mask, find_triplets, load_triplet  # type: ignore  # noqa: E402
from reconstruct import ReconstructCfg, reconstruct_from_gt  # type: ignore  # noqa: E402
from run_single import difference_map, overlay_av  # type: ignore  # noqa: E402

FLOAT_COLS = [
    "dice_overall", "dice_artery", "dice_vein",
    "sensitivity", "specificity",
    "boundary_f1_1px", "boundary_f1_2px", "boundary_f1_3px",
    "median_r_squared", "mean_boundary_disagreement_px",
    "mean_vessel_width_px", "fit_success_rate", "wall_s",
]
INT_COLS = ["n_segments", "n_fit_failures"]


# --- data loading ------------------------------------------------------


@dataclass
class Row:
    split: str
    filename: str
    dice_overall: float
    dice_artery: float
    dice_vein: float
    sensitivity: float
    specificity: float
    boundary_f1_1px: float
    boundary_f1_2px: float
    boundary_f1_3px: float
    median_r_squared: float
    mean_boundary_disagreement_px: float
    mean_vessel_width_px: float
    fit_success_rate: float
    wall_s: float
    n_segments: int
    n_fit_failures: int


def load_rows(csv_path: Path) -> list[Row]:
    out: list[Row] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = {}
            for c in FLOAT_COLS:
                v = r.get(c)
                d[c] = float(v) if v not in (None, "") else float("nan")
            for c in INT_COLS:
                v = r.get(c)
                d[c] = int(v) if v not in (None, "") else 0
            d["split"] = r["split"]
            d["filename"] = r["filename"]
            out.append(Row(**d))  # type: ignore[arg-type]
    return out


# --- summary helpers --------------------------------------------------


def _fmt_mean_std(xs: Iterable[float]) -> str:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return "NaN"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{mean(xs):.4f} ± {stdev(xs):.4f}"


def _pct_ge(xs: Iterable[float], thresh: float) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return 0.0
    return 100.0 * sum(1 for x in xs if x >= thresh) / len(xs)


def write_summary(rows: list[Row], output_path: Path) -> str:
    splits = ["overall"] + sorted({r.split for r in rows})
    lines: list[str] = []

    def _add(s: str) -> None:
        lines.append(s)

    _add(f"Skeleton-reconstruction validation — n images = {len(rows)}")
    _add("")

    for split in splits:
        if split == "overall":
            subset = rows
            header = f"== overall ({len(subset)}) =="
        else:
            subset = [r for r in rows if r.split == split]
            header = f"== {split} ({len(subset)}) =="
        if not subset:
            continue
        _add(header)
        _add(f"  Dice overall:     {_fmt_mean_std(r.dice_overall for r in subset)}")
        _add(f"  Dice artery:      {_fmt_mean_std(r.dice_artery for r in subset)}")
        _add(f"  Dice vein:        {_fmt_mean_std(r.dice_vein for r in subset)}")
        _add(f"  Sensitivity:      {_fmt_mean_std(r.sensitivity for r in subset)}")
        _add(f"  Specificity:      {_fmt_mean_std(r.specificity for r in subset)}")
        _add(f"  Boundary F1@1px:  median {median(r.boundary_f1_1px for r in subset):.4f}")
        _add(f"  Boundary F1@2px:  median {median(r.boundary_f1_2px for r in subset):.4f}")
        _add(f"  Boundary F1@3px:  median {median(r.boundary_f1_3px for r in subset):.4f}")
        _add(f"  Median R²:        {_fmt_mean_std(r.median_r_squared for r in subset)}")
        _add(f"  Fit success rate: {_fmt_mean_std(r.fit_success_rate for r in subset)}")
        _add(f"  Mean bdy disagree: {_fmt_mean_std(r.mean_boundary_disagreement_px for r in subset)} px")
        _add(f"  %images Dice ≥ 0.90: {_pct_ge([r.dice_overall for r in subset], 0.90):.1f}%")
        _add(f"  %images Dice ≥ 0.85: {_pct_ge([r.dice_overall for r in subset], 0.85):.1f}%")
        _add(f"  %images Dice ≥ 0.80: {_pct_ge([r.dice_overall for r in subset], 0.80):.1f}%")
        _add("")

    # Headline
    overall = rows
    headline = (
        f"Reconstruction Dice: {_fmt_mean_std(r.dice_overall for r in overall)} | "
        f"Boundary F1@2px: median {median(r.boundary_f1_2px for r in overall):.4f} | "
        f"Fit success: {_fmt_mean_std(r.fit_success_rate for r in overall)}"
    )
    _add(headline)

    text = "\n".join(lines) + "\n"
    output_path.write_text(text)
    return text


# --- plots ----------------------------------------------------------


def plot_dice_histogram(rows: list[Row], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist([r.dice_overall for r in rows], bins=bins, alpha=0.7, label="overall",
            color="k")
    ax.hist([r.dice_artery for r in rows], bins=bins, alpha=0.5, label="artery",
            color="red")
    ax.hist([r.dice_vein for r in rows], bins=bins, alpha=0.5, label="vein",
            color="blue")
    ax.set_xlabel("Dice coefficient")
    ax.set_ylabel("# images")
    ax.set_title(f"Per-image Dice distribution (n={len(rows)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_boundary_agreement(rows: list[Row], path: Path) -> None:
    """Mean Boundary F1 (with 95% bootstrap CI) vs tolerance (1, 2, 3 px)."""
    tols = [1, 2, 3]
    vals = np.array([
        [r.boundary_f1_1px for r in rows],
        [r.boundary_f1_2px for r in rows],
        [r.boundary_f1_3px for r in rows],
    ])
    means = vals.mean(axis=1)
    rng = np.random.default_rng(0)
    n = vals.shape[1]
    if n >= 10:
        boot = rng.choice(vals, size=(1000, vals.shape[0], n), replace=True, axis=1)
        boot_means = boot.mean(axis=2)
        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
    else:
        lo, hi = means, means  # too few for a meaningful CI

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tols, means, "-o", color="k", label="mean BF1")
    ax.fill_between(tols, lo, hi, alpha=0.2, color="k", label="95% CI")
    ax.axhline(0.80, color="green", linestyle="--", alpha=0.5, label="target 0.80")
    ax.set_xlabel("tolerance (px)")
    ax.set_ylabel("Boundary F1")
    ax.set_title(f"Boundary F1 vs tolerance (n={n})")
    ax.set_xticks(tols)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_disagreement_vs_width(rows: list[Row], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r.mean_vessel_width_px for r in rows]
    ys = [r.mean_boundary_disagreement_px for r in rows]
    ax.scatter(xs, ys, alpha=0.5, s=16)
    ax.set_xlabel("mean fitted vessel width (px)")
    ax.set_ylabel("mean boundary disagreement (px)")
    ax.set_title("Boundary disagreement vs vessel width")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_r_squared_distribution(rows: list[Row], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r.median_r_squared for r in rows]
    ax.hist(xs, bins=30, color="steelblue", edgecolor="k")
    ax.set_xlabel("per-image median R² of successful fits")
    ax.set_ylabel("# images")
    ax.set_title("Fit-quality distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- worst/best-case grids (re-reconstruct the N selected images) -----


def _triplet_for(rows: list[Row], triplets: list[Triplet], n: int, best: bool
                 ) -> list[tuple[Row, Triplet]]:
    sorted_rows = sorted(rows, key=lambda r: r.dice_overall, reverse=best)[:n]
    by_key = {(t.split, t.stem): t for t in triplets}
    out: list[tuple[Row, Triplet]] = []
    for r in sorted_rows:
        key = (r.split, r.filename)
        if key in by_key:
            out.append((r, by_key[key]))
    return out


def _render_image_panel(row: Row, triplet: Triplet, cfg: ReconstructCfg) -> np.ndarray:
    import cv2

    image, artery, vein = load_triplet(triplet)
    gt = combine_av_mask(artery, vein, label_crossing=True)
    recon = reconstruct_from_gt(image, artery, vein, cfg)
    gt_overlay = overlay_av(image, gt)
    recon_overlay = overlay_av(image, recon.recon_hard)
    diff = difference_map(
        (gt > 0).astype(np.uint8),
        (recon.recon_hard > 0).astype(np.uint8),
    )

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    raw = _label(image, f"{triplet.split}/{triplet.stem}  Dice={row.dice_overall:.2f}")
    gt_lbl = _label(gt_overlay, "GT")
    rc_lbl = _label(recon_overlay, "recon")
    df_lbl = _label(diff, "diff")
    return np.hstack([raw, gt_lbl, rc_lbl, df_lbl])


def render_case_grid(
    rows: list[Row],
    triplets: list[Triplet],
    output_path: Path,
    n: int,
    best: bool,
) -> None:
    import cv2

    picks = _triplet_for(rows, triplets, n, best)
    if not picks:
        return
    cfg = ReconstructCfg()
    panels = []
    for row, trip in picks:
        try:
            panels.append(_render_image_panel(row, trip, cfg))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] {trip.stem}: {exc}", flush=True)
    if not panels:
        return
    # Down-rescale each panel to speed up I/O
    h_target = 320
    resized = []
    for p in panels:
        scale = h_target / p.shape[0]
        p2 = cv2.resize(p, (int(p.shape[1] * scale), h_target))
        resized.append(p2)
    grid = np.vstack(resized)
    cv2.imwrite(str(output_path), grid)


# --- main ---------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiments/skeleton_reconstruction/results.csv", type=Path)
    p.add_argument("--db", default="databases", type=Path)
    p.add_argument("--output-dir", default="experiments/skeleton_reconstruction", type=Path)
    p.add_argument("--case-grid-n", default=6, type=int)
    p.add_argument("--skip-case-grids", action="store_true",
                   help="Skip the worst/best-case re-reconstruction grids (saves ~10 min).")
    args = p.parse_args()

    rows = load_rows(args.results)
    print(f"Loaded {len(rows)} rows from {args.results}", flush=True)
    if not rows:
        print("no rows — nothing to analyse")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = write_summary(rows, args.output_dir / "summary.txt")
    print("\n" + summary)

    plot_dice_histogram(rows, args.output_dir / "dice_histogram.png")
    plot_boundary_agreement(rows, args.output_dir / "boundary_agreement.png")
    plot_disagreement_vs_width(rows, args.output_dir / "disagreement_vs_vessel_width.png")
    plot_r_squared_distribution(rows, args.output_dir / "r_squared_distribution.png")
    print(f"Wrote plots → {args.output_dir}/*.png", flush=True)

    if not args.skip_case_grids:
        triplets = find_triplets(args.db, splits=["Train", "Test"])
        print(f"Rendering {args.case_grid_n} worst cases…", flush=True)
        render_case_grid(
            rows, triplets,
            args.output_dir / "worst_cases.png",
            n=args.case_grid_n, best=False,
        )
        print(f"Rendering {args.case_grid_n} best cases…", flush=True)
        render_case_grid(
            rows, triplets,
            args.output_dir / "best_cases.png",
            n=args.case_grid_n, best=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
