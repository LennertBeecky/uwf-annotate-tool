"""Step 4: batch reconstruction over every Train + Test triplet.

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/run_batch.py \
        [--db databases] [--output-dir experiments/skeleton_reconstruction] \
        [--workers 7] [--limit 0]

Per-image results are appended to `results.csv` every 10 completions so we
don't lose progress if the run crashes. Images that crash are logged to
`errors.log` and the run continues.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import Triplet, combine_av_mask, find_triplets, load_triplet  # type: ignore  # noqa: E402
from metrics import compute_metrics  # type: ignore  # noqa: E402
from reconstruct import ReconstructCfg, reconstruct_from_gt  # type: ignore  # noqa: E402


CSV_COLUMNS = [
    "split",
    "filename",
    "dice_overall",
    "dice_artery",
    "dice_vein",
    "sensitivity",
    "specificity",
    "boundary_f1_1px",
    "boundary_f1_2px",
    "boundary_f1_3px",
    "n_segments",
    "n_fit_failures",
    "median_r_squared",
    "mean_boundary_disagreement_px",
    "mean_vessel_width_px",
    "fit_success_rate",
    "wall_s",
]


def _process_one(args: tuple) -> dict | tuple[str, str, str]:
    """Return a row dict on success, or (stem, split, traceback) on failure."""
    triplet, cfg, masks_dir, save_preview = args
    t0 = time.time()
    try:
        import cv2 as _cv2
        import numpy as _np
        image, artery, vein = load_triplet(triplet)
        _ = combine_av_mask(artery, vein)
        recon = reconstruct_from_gt(image, artery, vein, cfg)
        a = recon.artery_stats
        v = recon.vein_stats
        n_attempts = a.n_sample_attempts + v.n_sample_attempts
        n_ok = a.n_sample_fits_ok + v.n_sample_fits_ok
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
            fit_success_rate=n_ok / max(n_attempts, 1),
        )
        row = metrics.as_row()
        row["wall_s"] = round(time.time() - t0, 1)

        if masks_dir is not None:
            split_dir = Path(masks_dir) / triplet.split
            split_dir.mkdir(parents=True, exist_ok=True)
            # Hard mask: uint8 {0,1,2,3}, lossless PNG (~200 KB / image)
            _cv2.imwrite(
                str(split_dir / f"{triplet.stem}_hard.png"),
                recon.recon_hard.astype(_np.uint8),
            )
            # Soft mask: float16 compressed .npz; float16 halves disk with
            # no material loss for [0, 1] values.
            _np.savez_compressed(
                str(split_dir / f"{triplet.stem}_soft.npz"),
                soft=recon.recon_soft.astype(_np.float16),
            )
            if save_preview:
                # Human-viewable BGR overlay: raw image blended 50 % with
                # red=artery, blue=vein, magenta=crossing.
                preview = image.copy()
                overlay = image.copy()
                overlay[recon.recon_hard == 1] = (0, 0, 255)      # artery  (BGR)
                overlay[recon.recon_hard == 2] = (255, 0, 0)      # vein
                overlay[recon.recon_hard == 3] = (255, 0, 255)    # crossing
                blended = _cv2.addWeighted(preview, 0.5, overlay, 0.5, 0)
                _cv2.imwrite(
                    str(split_dir / f"{triplet.stem}_preview.jpg"),
                    blended,
                    [_cv2.IMWRITE_JPEG_QUALITY, 85],
                )
        return row
    except Exception:
        return triplet.stem, triplet.split, traceback.format_exc()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="databases", type=Path)
    p.add_argument("--output-dir", default="experiments/skeleton_reconstruction", type=Path)
    p.add_argument("--workers", default=max(1, os.cpu_count() - 1), type=int)
    p.add_argument("--limit", default=0, type=int, help="max triplets to process (0 = all)")
    p.add_argument(
        "--checkpoint-every", default=10, type=int,
        help="flush rows to results.csv every N completions",
    )
    p.add_argument(
        "--shard", default=None, type=str,
        help="i/N — take every N-th triplet starting at i (0 ≤ i < N). "
             "E.g. --shard 2/5 processes triplets 2, 7, 12, ...",
    )
    p.add_argument(
        "--save-masks", action="store_true",
        help="Save reconstructed hard + soft masks per image alongside results.csv.",
    )
    p.add_argument(
        "--masks-dir", default=None, type=Path,
        help="Where to save per-image masks (default: --output-dir/masks).",
    )
    p.add_argument(
        "--splits", default="Train,Test", type=str,
        help="Comma-separated list of splits to process, e.g. 'Train' or 'Train,Test'.",
    )
    p.add_argument(
        "--save-preview-overlays", action="store_true",
        help="Save a human-viewable JPEG preview of each reconstruction "
             "(raw image blended with red=artery, blue=vein, magenta=crossing). "
             "Adds ~100 KB per image. Requires --save-masks.",
    )
    args = p.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    triplets = find_triplets(args.db, splits=splits)

    # Shard: i/N → take every N-th triplet starting at i.
    shard_tag = ""
    if args.shard:
        try:
            i_str, N_str = args.shard.split("/")
            i, N = int(i_str), int(N_str)
            if not (0 <= i < N):
                raise ValueError
        except ValueError:
            raise SystemExit(f"--shard must be i/N with 0 ≤ i < N; got {args.shard!r}")
        triplets = triplets[i::N]
        shard_tag = f"shard{i}of{N}_"
        print(f"Shard {i}/{N}: {len(triplets)} triplets assigned to this worker.", flush=True)

    # Shard outputs have shard-tagged names so parallel jobs don't collide.
    results_path = output_dir / f"{shard_tag}results.csv"
    errors_path = output_dir / f"{shard_tag}errors.log"

    if args.limit > 0:
        triplets = triplets[: args.limit]
    print(f"Processing {len(triplets)} triplets with {args.workers} workers.", flush=True)

    masks_dir = None
    if args.save_masks:
        masks_dir = args.masks_dir or (output_dir / "masks")
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"  saving masks → {masks_dir}", flush=True)

    cfg = ReconstructCfg()

    # Skip triplets already processed (resumable).
    done_stems: set[tuple[str, str]] = set()
    if results_path.exists():
        with results_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                done_stems.add((r["split"], r["filename"]))
        triplets = [t for t in triplets if (t.split, t.stem) not in done_stems]
        print(f"  resuming — {len(done_stems)} already done, {len(triplets)} remaining.", flush=True)

    if not triplets:
        print("Nothing to do.")
        return 0

    # Open csv in append mode; write header if empty
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    csv_file = results_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        csv_file.flush()

    error_file = errors_path.open("a")

    save_preview = bool(args.save_preview_overlays and args.save_masks)
    if args.save_preview_overlays and not args.save_masks:
        print("  warning: --save-preview-overlays requires --save-masks; ignoring.", flush=True)
    pending = [(t, cfg, masks_dir, save_preview) for t in triplets]
    n_done = 0
    n_errors = 0
    rows_buffer: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as ex:
        fut_to_stem = {ex.submit(_process_one, item): item[0].stem for item in pending}
        with tqdm(total=len(pending), smoothing=0.1) as bar:
            for fut in as_completed(fut_to_stem):
                try:
                    r = fut.result()
                except Exception as exc:  # pragma: no cover
                    n_errors += 1
                    error_file.write(f"{fut_to_stem[fut]}: worker crash: {exc}\n")
                    bar.update(1)
                    continue

                if isinstance(r, dict):
                    rows_buffer.append(r)
                    n_done += 1
                else:
                    stem, split, tb = r
                    error_file.write(f"=== {split}/{stem} ===\n{tb}\n")
                    n_errors += 1
                bar.update(1)

                if len(rows_buffer) >= args.checkpoint_every:
                    for row in rows_buffer:
                        writer.writerow(row)
                    csv_file.flush()
                    rows_buffer.clear()

                if (n_done + n_errors) % 50 == 0:
                    bar.write(
                        f"progress: {n_done + n_errors}/{len(pending)} "
                        f"ok={n_done} err={n_errors}"
                    )

    # Flush remaining rows
    for row in rows_buffer:
        writer.writerow(row)
    csv_file.flush()
    csv_file.close()
    error_file.close()

    print(f"Done. ok={n_done}, errors={n_errors}. results → {results_path}")
    if n_errors:
        print(f"See errors in {errors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
