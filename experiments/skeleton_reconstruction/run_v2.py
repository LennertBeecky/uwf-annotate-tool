"""Run v2 (Steger-enhanced) reconstruction on a single Train image.

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/run_v2.py \
        --split Train --stem 3_A \
        --output-dir experiments/v2_test/3_A_v2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import combine_av_mask, find_triplets, load_triplet  # type: ignore  # noqa: E402
from metrics import compute_metrics  # type: ignore  # noqa: E402
from reconstruct_v2 import ReconstructCfg, reconstruct_from_gt  # type: ignore  # noqa: E402
from run_single import render_six_panel  # type: ignore  # noqa: E402


def _say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="databases", type=Path)
    p.add_argument("--split", default="Train")
    p.add_argument("--stem", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--steger-sigma", type=float, default=1.5)
    p.add_argument("--steger-max-offset", type=float, default=2.0)
    p.add_argument("--no-steger", action="store_true",
                   help="Disable Steger — v2 behaves exactly like v1.")
    args = p.parse_args()

    triplets = find_triplets(args.db, splits=[args.split])
    picked = [t for t in triplets if t.stem == args.stem]
    if not picked:
        _say(f"No triplet matches stem={args.stem!r} in split={args.split}")
        return 1
    triplet = picked[0]

    _say(f"v2 reconstruction: {triplet.split}/{triplet.stem}")
    image, artery, vein = load_triplet(triplet)
    gt_labels = combine_av_mask(artery, vein, label_crossing=True)

    from dataclasses import replace
    cfg = ReconstructCfg()
    cfg = replace(
        cfg,
        steger_enabled=not args.no_steger,
        steger_sigma=args.steger_sigma,
        steger_max_offset=args.steger_max_offset,
    )
    _say(f"  steger_enabled={cfg.steger_enabled}  sigma={cfg.steger_sigma}  "
         f"max_offset={cfg.steger_max_offset}")

    t0 = time.time()
    recon = reconstruct_from_gt(image, artery, vein, cfg)
    dt = time.time() - t0

    a, v = recon.artery_stats, recon.vein_stats
    n_attempts = a.n_sample_attempts + v.n_sample_attempts
    n_ok = a.n_sample_fits_ok + v.n_sample_fits_ok
    n_gauss = a.n_gaussian_rescue + v.n_gaussian_rescue
    n_steger = a.n_steger_snapped + v.n_steger_snapped
    _say(f"  reconstruction done in {dt:.1f}s")
    _say(f"  samples attempted: {n_attempts}")
    _say(f"  Phase-1 convolved-step fits: {n_ok}")
    _say(f"  Phase-2 Gaussian rescues:    {n_gauss}")
    _say(f"  V2 Steger-snapped samples:   {n_steger} "
         f"({n_steger / max(n_attempts, 1):.1%} of attempts)")

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

    _say("")
    _say("--- Metrics ---")
    _say(f"  Dice vessels : {metrics.dice_overall:.4f}")
    _say(f"  Dice artery  : {metrics.dice_artery:.4f}")
    _say(f"  Dice vein    : {metrics.dice_vein:.4f}")
    _say(f"  Sensitivity  : {metrics.sensitivity:.4f}")
    _say(f"  Specificity  : {metrics.specificity:.4f}")
    _say(f"  Boundary F1 @1/2/3 px: {metrics.boundary_f1_1px:.3f} / "
         f"{metrics.boundary_f1_2px:.3f} / {metrics.boundary_f1_3px:.3f}")
    _say(f"  Mean bdy disagreement: {metrics.mean_boundary_disagreement_px:.2f} px")

    # Save 6-panel + the hard/soft masks
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    render_six_panel(
        image=image,
        gt_labels=gt_labels,
        recon_labels=recon.recon_hard,
        skel_artery=recon.artery_skeleton,
        skel_vein=recon.vein_skeleton,
        recon_soft=recon.recon_soft,
        output_path=out / f"{triplet.stem}_v2_result.png",
    )

    import cv2
    import numpy as np
    cv2.imwrite(str(out / f"{triplet.stem}_v2_hard.png"),
                recon.recon_hard.astype(np.uint8))
    np.savez_compressed(str(out / f"{triplet.stem}_v2_soft.npz"),
                        soft=recon.recon_soft.astype(np.float16))

    _say(f"Wrote → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
