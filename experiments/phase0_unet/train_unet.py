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
from unet import SmallUNet, count_params  # type: ignore  # noqa: E402


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


# ---------- losses ----------


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice over (B, C, H, W) maps, averaged over channels.
    `pred` and `target` in [0, 1].
    """
    dims = (0, 2, 3)  # sum over B, H, W → per-channel
    num = 2.0 * (pred * target).sum(dim=dims)
    den = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    dice = num / den
    return 1.0 - dice.mean()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dice = soft_dice_loss(probs, targets)
    return bce + dice, {"bce": float(bce.item()), "dice_loss": float(dice.item())}


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


def evaluate_epoch(model: SmallUNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total, n_batches = {"dice": 0.0, "dice_art": 0.0, "dice_vein": 0.0, "loss": 0.0}, 0
    with torch.no_grad():
        for img, lbl, _ in loader:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = model(img)
            loss, _ = compute_loss(logits, lbl)
            probs = torch.sigmoid(logits)
            pred_bin = (probs > 0.5)
            tgt_bin = (lbl > 0.5)
            d, da, dv = per_channel_dice(pred_bin, tgt_bin)
            total["dice"] += d
            total["dice_art"] += da
            total["dice_vein"] += dv
            total["loss"] += float(loss.item())
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
                     "device": str(device)})
    (out_dir / "config_used.json").write_text(json.dumps(cfg_dump, indent=2))

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
    csv_w.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_dice_art",
                    "val_dice_vein", "lr", "wall_s"])
    csv_f.flush()

    best_dice = -1.0
    best_path = out_dir / "checkpoint_best.pt"
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for img, lbl, _ in train_loader:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(img)
            loss, _ = compute_loss(logits, lbl)
            loss.backward()
            optim.step()
            running += float(loss.item())
        scheduler.step()
        train_loss = running / max(len(train_loader), 1)

        val = evaluate_epoch(model, val_loader, device)
        wall = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"[{args.experiment_name}] epoch {epoch:3d}/{cfg.num_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"val_dice={val['dice']:.4f} (A={val['dice_art']:.3f} "
              f"V={val['dice_vein']:.3f})  lr={lr:.2e}  {wall:.1f}s",
              flush=True)
        csv_w.writerow([epoch, f"{train_loss:.6f}", f"{val['loss']:.6f}",
                        f"{val['dice']:.6f}", f"{val['dice_art']:.6f}",
                        f"{val['dice_vein']:.6f}", f"{lr:.6e}", f"{wall:.1f}"])
        csv_f.flush()

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": best_dice,
                "config": cfg_dump,
            }, best_path)

    csv_f.close()
    print(f"\nBest val dice: {best_dice:.4f}  → {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
