"""Run all three Phase-0 conditions (A manual, C physics, G hybrid) sequentially.

Each condition reuses the exact seed/split/HPs from config.yaml — only the
label source differs. Results land in `experiments/phase0_unet/results/<name>/`.

Usage:
    python -u experiments/phase0_unet/run_all_conditions.py \
        --num-epochs 100 --wandb-project uwf-phase0

Extras:
    --sanity               → num_epochs=5, --wandb-mode disabled
    --conditions A,C,G     → subset
    --require-gpu          → fail fast if no CUDA/MPS visible
    --suffix _v2           → append to experiment names (e.g. cond_A_v2)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = {
    "A": ("manual", "cond_A"),
    "C": ("physics", "cond_C"),
    "G": ("hybrid", "cond_G"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", default="A,C,G",
                   help="Comma-separated subset of A,C,G.")
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity", action="store_true",
                   help="5 epochs, wandb disabled — fast check before full run.")
    p.add_argument("--wandb-project", default="uwf-phase0")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--suffix", default="", help="Appended to each experiment name.")
    p.add_argument("--cldice-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--cldice-iters", type=int, default=5)
    args = p.parse_args()

    if args.sanity:
        args.num_epochs = 5
        args.wandb_mode = "disabled"
        args.suffix = args.suffix or "_sanity"
        print("[sanity] 5 epochs, wandb disabled, suffix=_sanity")

    picks = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    for c in picks:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition {c!r}; use A, C, or G.")

    train_script = Path(__file__).with_name("train_unet.py")
    overall_t0 = time.time()
    for c in picks:
        mode, name = CONDITIONS[c]
        exp_name = name + args.suffix
        cmd = [
            sys.executable, "-u", str(train_script),
            "--mode", mode,
            "--experiment-name", exp_name,
            "--seed", str(args.seed),
            "--num-epochs", str(args.num_epochs),
            "--wandb-project", args.wandb_project,
            "--wandb-mode", args.wandb_mode,
            "--workers", str(args.workers),
            "--bce-weight", str(args.bce_weight),
            "--dice-weight", str(args.dice_weight),
            "--cldice-weight", str(args.cldice_weight),
            "--cldice-iters", str(args.cldice_iters),
        ]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.device:
            cmd += ["--device", args.device]
        if args.require_gpu:
            cmd += ["--require-gpu"]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
        if args.wandb_tags:
            cmd += ["--wandb-tags", args.wandb_tags]

        print(f"\n{'=' * 72}")
        print(f"  [{c}] {exp_name}  (mode={mode})")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 72}")
        t0 = time.time()
        result = subprocess.run(cmd)
        wall = time.time() - t0
        print(f"  [{c}] done in {wall:.1f}s  (exit {result.returncode})")
        if result.returncode != 0:
            print(f"  [{c}] FAILED — aborting remaining conditions.")
            return result.returncode

    print(f"\nAll conditions finished in {time.time() - overall_t0:.1f}s total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
