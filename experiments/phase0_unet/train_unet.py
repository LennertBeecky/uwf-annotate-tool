"""Phase-0 UNet training — conditions A (manual), C (physics), G (hybrid).

All three conditions share the same seed, architecture, and hyperparameters.
Only the label source varies (see dataset.py).

Usage:
    PYTHONPATH=src python -u experiments/phase0_unet/train_unet.py \
        --mode manual  --experiment-name cond_A  --seed 42
    PYTHONPATH=src python -u experiments/phase0_unet/train_unet.py \
        --mode physics --experiment-name cond_C  --seed 42
    PYTHONPATH=src python -u experiments/phase0_unet/train_unet.py \
        --mode hybrid  --experiment-name cond_G  --seed 42

Defaults are read from config.yaml; CLI flags override.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import REYIADataset  # type: ignore  # noqa: E402
from losses import compute_total_loss  # type: ignore  # noqa: E402
from unet import SmallUNet, count_params  # type: ignore  # noqa: E402

# wandb is optional; lazy-imported per experiment so missing-dep doesn't
# block CPU-only dev runs.
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------- device helpers ----------


def _pick_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- metrics ----------


def per_channel_dice(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> tuple[float, float, float]:
    """(overall, artery, vein) Dice on binary (B, 2, H, W) maps."""
    a_pred, v_pred = pred_bin[:, 0], pred_bin[:, 1]
    a_gt, v_gt = target_bin[:, 0], target_bin[:, 1]

    def _dice(p, g):
        inter = float((p & g).sum().item())
        denom = float(p.sum().item() + g.sum().item())
        return 2.0 * inter / denom if denom > 0 else 1.0

    both_p = a_pred | v_pred
    both_g = a_gt | v_gt
    return _dice(both_p, both_g), _dice(a_pred, a_gt), _dice(v_pred, v_gt)


# ---------- training ----------


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-5
    input_resolution: int = 512
    seed: int = 42
    val_split: float = 0.2
    hflip_p: float = 0.5
    db_root: str = "databases"
    physics_npz_dir: str = "experiments/skeleton_reconstruction/masks/Train"
    output_root: str = "experiments/phase0_unet/results"


def _load_config(path: Path) -> TrainConfig:
    import yaml
    with path.open() as f:
        d = yaml.safe_load(f) or {}
    # only accept known fields
    kwargs = {}
    for f in TrainConfig.__dataclass_fields__:
        if f in d:
            kwargs[f] = d[f]
    return TrainConfig(**kwargs)


def evaluate_epoch(model: SmallUNet, loader: DataLoader, device: torch.device,
                   cldice_iters: int = 5) -> dict:
    model.eval()
    total, n_batches = {"dice": 0.0, "dice_art": 0.0, "dice_vein": 0.0,
                        "loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0,
                        "loss_cldice": 0.0}, 0
    with torch.no_grad():
        for img, lbl, _ in loader:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = model(img)
            loss, components = compute_total_loss(logits, lbl, cldice_iters=cldice_iters)
            probs = torch.sigmoid(logits)
            pred_bin = (probs > 0.5)
            tgt_bin = (lbl > 0.5)
            d, da, dv = per_channel_dice(pred_bin, tgt_bin)
            total["dice"] += d
            total["dice_art"] += da
            total["dice_vein"] += dv
            total["loss"] += components["loss_total"]
            total["loss_bce"] += components["loss_bce"]
            total["loss_dice"] += components["loss_dice"]
            total["loss_cldice"] += components["loss_cldice"]
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in total.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["manual", "physics", "hybrid"])
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--config", default=str(HERE / "config.yaml"), type=Path)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda/mps/cpu (auto if None)")
    p.add_argument("--workers", type=int, default=2)
    # wandb
    p.add_argument("--wandb-project", default="uwf-phase0", type=str)
    p.add_argument("--wandb-entity", default=None, type=str,
                   help="Team/user. If None, wandb uses your default.")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", default=None, type=str,
                   help="Comma-separated tags, e.g. 'phase0,sanity'.")
    # loss weights (match spec defaults: 1 / 1 / 1 for BCE / dice / clDice)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--cldice-weight", type=float, default=1.0)
    p.add_argument("--cldice-iters", type=int, default=5,
                   help="Soft-skeletonisation iterations in clDice.")
    args = p.parse_args()

    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    _set_seed(cfg.seed)
    device = _pick_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(cfg.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log effective config for reproducibility
    cfg_dump = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    cfg_dump.update({"mode": args.mode, "experiment_name": args.experiment_name,
                     "device": str(device),
                     "bce_weight": args.bce_weight,
                     "dice_weight": args.dice_weight,
                     "cldice_weight": args.cldice_weight,
                     "cldice_iters": args.cldice_iters})
    (out_dir / "config_used.json").write_text(json.dumps(cfg_dump, indent=2))

    # ---- wandb ----
    wandb_run = None
    if args.wandb_mode != "disabled" and _WANDB_AVAILABLE:
        tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
        tags = tags or [args.mode]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            tags=tags,
            config=cfg_dump,
            mode=args.wandb_mode,
            dir=str(out_dir),
        )
    elif args.wandb_mode != "disabled" and not _WANDB_AVAILABLE:
        print("  [wandb] package not installed; run without logging. "
              "pip install wandb if you want curves.")

    train_ds = REYIADataset(
        db_root=cfg.db_root, mode=args.mode, split="train",
        input_resolution=cfg.input_resolution, val_frac=cfg.val_split,
        seed=cfg.seed, physics_npz_dir=cfg.physics_npz_dir, hflip_p=cfg.hflip_p,
    )
    val_ds = REYIADataset(
        db_root=cfg.db_root, mode=args.mode, split="val",
        input_resolution=cfg.input_resolution, val_frac=cfg.val_split,
        seed=cfg.seed, physics_npz_dir=cfg.physics_npz_dir, hflip_p=0.0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = SmallUNet(in_channels=3, out_channels=2).to(device)
    print(f"Params: {count_params(model):,}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01,
    )

    csv_path = out_dir / "val_metrics.csv"
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "epoch", "train_loss", "train_bce", "train_dice_loss", "train_cldice_loss",
        "val_loss", "val_bce", "val_dice_loss", "val_cldice_loss",
        "val_dice", "val_dice_art", "val_dice_vein", "lr", "wall_s",
    ])
    csv_f.flush()

    best_dice = -1.0
    best_path = out_dir / "checkpoint_best.pt"
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss_total": 0.0, "loss_bce": 0.0,
                   "loss_dice": 0.0, "loss_cldice": 0.0}
        for img, lbl, _ in train_loader:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(img)
            loss, components = compute_total_loss(
                logits, lbl,
                cldice_iters=args.cldice_iters,
                cldice_weight=args.cldice_weight,
                dice_weight=args.dice_weight,
                bce_weight=args.bce_weight,
            )
            loss.backward()
            optim.step()
            for k, v in components.items():
                running[k] += v
        scheduler.step()
        n_batches = max(len(train_loader), 1)
        train = {k: v / n_batches for k, v in running.items()}

        val = evaluate_epoch(model, val_loader, device, cldice_iters=args.cldice_iters)
        wall = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(
            f"[{args.experiment_name}] epoch {epoch:3d}/{cfg.num_epochs}  "
            f"train={train['loss_total']:.4f} "
            f"(bce={train['loss_bce']:.3f} dice={train['loss_dice']:.3f} "
            f"cld={train['loss_cldice']:.3f})  "
            f"val_loss={val['loss']:.4f}  "
            f"val_dice={val['dice']:.4f} (A={val['dice_art']:.3f} "
            f"V={val['dice_vein']:.3f})  lr={lr:.2e}  {wall:.1f}s",
            flush=True,
        )
        csv_w.writerow([
            epoch,
            f"{train['loss_total']:.6f}", f"{train['loss_bce']:.6f}",
            f"{train['loss_dice']:.6f}", f"{train['loss_cldice']:.6f}",
            f"{val['loss']:.6f}", f"{val['loss_bce']:.6f}",
            f"{val['loss_dice']:.6f}", f"{val['loss_cldice']:.6f}",
            f"{val['dice']:.6f}", f"{val['dice_art']:.6f}",
            f"{val['dice_vein']:.6f}", f"{lr:.6e}", f"{wall:.1f}",
        ])
        csv_f.flush()

        # wandb per-epoch log
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss_total": train["loss_total"],
                "train/loss_bce": train["loss_bce"],
                "train/loss_dice": train["loss_dice"],
                "train/loss_cldice": train["loss_cldice"],
                "val/loss_total": val["loss"],
                "val/loss_bce": val["loss_bce"],
                "val/loss_dice": val["loss_dice"],
                "val/loss_cldice": val["loss_cldice"],
                "val/dice_overall": val["dice"],
                "val/dice_artery": val["dice_art"],
                "val/dice_vein": val["dice_vein"],
                "lr": lr,
                "wall_s": wall,
            })

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": best_dice,
                "config": cfg_dump,
            }, best_path)
            if wandb_run is not None:
                wandb_run.summary["best_val_dice"] = best_dice
                wandb_run.summary["best_epoch"] = epoch

    csv_f.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nBest val dice: {best_dice:.4f}  → {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
