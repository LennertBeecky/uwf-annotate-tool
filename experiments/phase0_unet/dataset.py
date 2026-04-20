"""Dataset loader supporting Phase-0 conditions A (manual), C (physics), G (hybrid).

Key invariants:
  • Same seed across all three conditions → same train/val split
  • Hybrid split is deterministic and image-level (no mixed label types per image)
  • Both manual and physics labels are returned as float32 [0, 1] so the loss
    function treats them uniformly

Two supported physics-label sources (auto-detected per sample):
  • `databases/Train_physics/{artery,veins}/{stem}.png` — uint8 [0,255] per
    channel (spec's canonical layout)
  • `experiments/skeleton_reconstruction/masks/Train/{stem}_soft.npz` — the
    Condor batch output (float16 shape (H, W, 2), channel 0 = artery,
    channel 1 = vein). Converted on-the-fly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

Mode = Literal["manual", "physics", "hybrid"]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    stem: str
    image_path: Path
    artery_path: Path | None
    vein_path: Path | None
    physics_npz: Path | None
    label_source: Literal["manual", "physics"]  # which folder to read labels from


def _listdir_stems(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS}


def _det_split(stems: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Sorted + seeded shuffle → (train_stems, val_stems)."""
    rng = np.random.default_rng(seed)
    order = sorted(stems)
    idx = rng.permutation(len(order))
    shuffled = [order[i] for i in idx]
    n_val = int(round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def build_samples(
    db_root: Path,
    mode: Mode,
    seed: int,
    val_frac: float,
    split: Literal["train", "val"],
    physics_npz_dir: Path | None = None,
) -> list[Sample]:
    """Return the list of Samples for this (mode, split).

    All three modes share the same train/val image partition (seeded), so
    conditions A/C/G see identical image sets. Only the label source differs.

    mode='hybrid': TRAIN images split by sorted stem — first half from
    manual, second half from physics. Val images always from manual (so
    all three modes evaluate against the same reference).
    """
    db_root = Path(db_root)
    train_dir = db_root / "Train"
    manual_imgs = _listdir_stems(train_dir / "images")
    manual_art = _listdir_stems(train_dir / "artery")
    manual_vn = _listdir_stems(train_dir / "veins")
    triplet_stems = sorted(set(manual_imgs) & set(manual_art) & set(manual_vn))

    physics_dir = db_root / "Train_physics"
    physics_art = _listdir_stems(physics_dir / "artery") if (physics_dir / "artery").exists() else {}
    physics_vn = _listdir_stems(physics_dir / "veins") if (physics_dir / "veins").exists() else {}

    # Optional npz cache (Condor batch output).
    physics_npz_dir = Path(physics_npz_dir) if physics_npz_dir else None
    if physics_npz_dir and physics_npz_dir.exists():
        physics_npz_stems = {p.stem.replace("_soft", ""): p for p in physics_npz_dir.iterdir()
                             if p.name.endswith("_soft.npz")}
    else:
        physics_npz_stems = {}

    train_stems, val_stems = _det_split(triplet_stems, val_frac, seed)
    selected = train_stems if split == "train" else val_stems

    # For hybrid, partition TRAIN stems in half by sorted order.
    hybrid_partition: dict[str, Literal["manual", "physics"]] = {}
    if mode == "hybrid" and split == "train":
        sorted_train = sorted(train_stems)
        half = len(sorted_train) // 2
        for s in sorted_train[:half]:
            hybrid_partition[s] = "manual"
        for s in sorted_train[half:]:
            hybrid_partition[s] = "physics"

    samples: list[Sample] = []
    for stem in selected:
        img_path = manual_imgs[stem]   # image always from the manual dir
        # Default label source per mode
        if split == "val":
            src: Literal["manual", "physics"] = "manual"  # val always manual
        elif mode == "manual":
            src = "manual"
        elif mode == "physics":
            src = "physics"
        else:
            src = hybrid_partition[stem]

        if src == "manual":
            art_p = manual_art.get(stem)
            vn_p = manual_vn.get(stem)
            npz_p = None
        else:
            art_p = physics_art.get(stem)
            vn_p = physics_vn.get(stem)
            npz_p = physics_npz_stems.get(stem)
            if (art_p is None or vn_p is None) and npz_p is None:
                # physics labels missing for this stem; skip with a note
                continue
        samples.append(
            Sample(
                stem=stem,
                image_path=img_path,
                artery_path=art_p,
                vein_path=vn_p,
                physics_npz=npz_p,
                label_source=src,
            )
        )
    return samples


class REYIADataset(Dataset):
    def __init__(
        self,
        db_root: Path | str,
        mode: Mode,
        split: Literal["train", "val"],
        input_resolution: int = 512,
        val_frac: float = 0.2,
        seed: int = 42,
        physics_npz_dir: Path | str | None = None,
        hflip_p: float = 0.5,
    ):
        self.mode = mode
        self.split = split
        self.res = input_resolution
        self.hflip_p = hflip_p if split == "train" else 0.0
        self.samples = build_samples(
            db_root=Path(db_root),
            mode=mode,
            seed=seed,
            val_frac=val_frac,
            split=split,
            physics_npz_dir=Path(physics_npz_dir) if physics_npz_dir else None,
        )
        if not self.samples:
            raise RuntimeError(
                f"No samples found for mode={mode!r} split={split!r}. "
                f"Check that databases/Train/ and (for physics/hybrid) "
                f"databases/Train_physics/ exist."
            )
        # report label-source breakdown
        n_manual = sum(1 for s in self.samples if s.label_source == "manual")
        n_physics = sum(1 for s in self.samples if s.label_source == "physics")
        print(f"[REYIADataset {mode}/{split}] n={len(self.samples)}  "
              f"manual={n_manual}  physics={n_physics}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p))  # BGR
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img

    def _load_label_manual(self, s: Sample) -> np.ndarray:
        """Returns (2, H, W) float32 {0, 1}."""
        art = cv2.imread(str(s.artery_path), cv2.IMREAD_GRAYSCALE)
        vn = cv2.imread(str(s.vein_path), cv2.IMREAD_GRAYSCALE)
        art = cv2.resize(art, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        vn = cv2.resize(vn, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        art = (art > 127).astype(np.float32)
        vn = (vn > 127).astype(np.float32)
        return np.stack([art, vn], axis=0)

    def _load_label_physics(self, s: Sample) -> np.ndarray:
        """Returns (2, H, W) float32 [0, 1]."""
        if s.physics_npz is not None:
            d = np.load(s.physics_npz)["soft"]  # (H, W, 2) float16
            art = d[..., 0].astype(np.float32)
            vn = d[..., 1].astype(np.float32)
        else:
            art = cv2.imread(str(s.artery_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            vn = cv2.imread(str(s.vein_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        art = cv2.resize(art, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        vn = cv2.resize(vn, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        return np.stack([np.clip(art, 0, 1), np.clip(vn, 0, 1)], axis=0)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_image(s.image_path)     # (H, W, 3)
        if s.label_source == "manual":
            lbl = self._load_label_manual(s)     # (2, H, W)
        else:
            lbl = self._load_label_physics(s)

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        img_t = torch.from_numpy(np.ascontiguousarray(img))
        lbl_t = torch.from_numpy(np.ascontiguousarray(lbl))

        # Horizontal flip augmentation (train only)
        if self.hflip_p > 0 and np.random.rand() < self.hflip_p:
            img_t = torch.flip(img_t, dims=[-1])
            lbl_t = torch.flip(lbl_t, dims=[-1])

        return img_t, lbl_t, s.stem


if __name__ == "__main__":
    # Smoke test: verify the three modes produce identical train/val stem sets.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="databases")
    p.add_argument("--npz-dir", default="experiments/skeleton_reconstruction/masks/Train")
    args = p.parse_args()

    for mode in ("manual", "physics", "hybrid"):
        try:
            for split in ("train", "val"):
                ds = REYIADataset(
                    db_root=args.db, mode=mode, split=split,
                    physics_npz_dir=args.npz_dir,
                )
                img, lbl, stem = ds[0]
                print(f"  {mode}/{split} first sample: stem={stem}, "
                      f"img={tuple(img.shape)} min={img.min():.2f} max={img.max():.2f}  "
                      f"lbl={tuple(lbl.shape)} min={lbl.min():.2f} max={lbl.max():.2f}")
        except Exception as e:
            print(f"  {mode}: {e}")
