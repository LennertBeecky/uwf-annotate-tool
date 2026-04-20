"""Data loading for the skeleton-reconstruction experiment.

Database layout (relative to project root):
    databases/
      Train/{images,artery,veins}/
      Test/{images,artery,veins}/

Masks are binary {0, 255} grayscale; images are 1444×1444 BGR. Artery and
vein masks can overlap at A/V crossings (≈ 3% of vessel pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Combined label convention
LABEL_BG = 0
LABEL_ARTERY = 1
LABEL_VEIN = 2
LABEL_CROSSING = 3


@dataclass
class Triplet:
    split: str
    stem: str
    image_path: Path
    artery_path: Path
    vein_path: Path


def _listdir_stems(folder: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def find_triplets(db_root: str | Path, splits: Iterable[str] = ("Train", "Test")) -> list[Triplet]:
    """Match image/artery/veins by stem across all requested splits."""
    db_root = Path(db_root)
    out: list[Triplet] = []
    for split in splits:
        imgs = _listdir_stems(db_root / split / "images")
        arts = _listdir_stems(db_root / split / "artery")
        vns = _listdir_stems(db_root / split / "veins")
        common = sorted(set(imgs) & set(arts) & set(vns))
        for stem in common:
            out.append(
                Triplet(
                    split=split,
                    stem=stem,
                    image_path=imgs[stem],
                    artery_path=arts[stem],
                    vein_path=vns[stem],
                )
            )
    return out


def load_triplet(
    triplet: Triplet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_bgr, artery_binary, vein_binary) for one Triplet.

    Masks are uint8 {0, 1}. Image is uint8 BGR.
    """
    image = cv2.imread(str(triplet.image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image {triplet.image_path}")
    artery = cv2.imread(str(triplet.artery_path), cv2.IMREAD_GRAYSCALE)
    vein = cv2.imread(str(triplet.vein_path), cv2.IMREAD_GRAYSCALE)
    if artery is None or vein is None:
        raise FileNotFoundError(
            f"could not read masks for {triplet.stem}: artery={triplet.artery_path}, "
            f"vein={triplet.vein_path}"
        )
    artery = (artery > 127).astype(np.uint8)
    vein = (vein > 127).astype(np.uint8)
    return image, artery, vein


def combine_av_mask(
    artery: np.ndarray,
    vein: np.ndarray,
    label_crossing: bool = True,
) -> np.ndarray:
    """Fuse binary artery + vein masks into a single label map.

    Labels: 0=bg, 1=artery-only, 2=vein-only, 3=crossing (both).
    If label_crossing is False, crossings take label 1 (artery wins).
    """
    art = artery.astype(bool)
    vn = vein.astype(bool)
    out = np.zeros(art.shape, dtype=np.uint8)
    out[art] = LABEL_ARTERY
    out[vn] = LABEL_VEIN
    if label_crossing:
        out[art & vn] = LABEL_CROSSING
    return out
