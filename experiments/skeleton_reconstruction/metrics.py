"""Reconstruction vs GT evaluation metrics: Dice, boundary F1, disagreement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


# --- Dice -------------------------------------------------------------


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    num = 2.0 * float((a & b).sum())
    den = float(a.sum() + b.sum())
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return num / den


# --- Sensitivity + specificity ---------------------------------------


def sens_spec(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = float((gt & pred).sum())
    fn = float((gt & ~pred).sum())
    fp = float((~gt & pred).sum())
    tn = float((~gt & ~pred).sum())
    sens = tp / max(tp + fn, 1.0)
    spec = tn / max(tn + fp, 1.0)
    return float(sens), float(spec)


# --- Boundary F1 -----------------------------------------------------


def boundary_f1(gt_mask: np.ndarray, pred_mask: np.ndarray, tolerance_px: float) -> float:
    """Boundary F1 at a tolerance (pixels).

    A GT boundary pixel is matched if any predicted boundary pixel is within
    `tolerance_px` (Euclidean), and vice versa.
    """
    gt_b = find_boundaries(gt_mask.astype(bool), mode="inner")
    pr_b = find_boundaries(pred_mask.astype(bool), mode="inner")

    if not gt_b.any() and not pr_b.any():
        return 1.0
    if not gt_b.any() or not pr_b.any():
        return 0.0

    # distance to nearest predicted boundary pixel, evaluated at GT pixels
    pred_dt = distance_transform_edt(~pr_b)
    gt_dt = distance_transform_edt(~gt_b)

    gt_matched = pred_dt[gt_b] <= tolerance_px
    pr_matched = gt_dt[pr_b] <= tolerance_px

    recall = float(gt_matched.sum()) / max(gt_b.sum(), 1)
    precision = float(pr_matched.sum()) / max(pr_b.sum(), 1)
    if recall + precision == 0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


# --- Disagreement distance ------------------------------------------


def mean_boundary_disagreement(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> float:
    """Mean distance (px) of each disagreeing pixel to the GT vessel mask.

    A disagreement is a pixel where GT and prediction disagree on vessel
    membership. We report how far those pixels are from the nearest GT
    vessel boundary — values near 1 indicate edge-only disagreements.
    """
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    disagree = gt ^ pr
    if not disagree.any():
        return 0.0
    gt_b = find_boundaries(gt, mode="inner")
    if not gt_b.any():
        return float("inf")
    dt = distance_transform_edt(~gt_b)
    return float(dt[disagree].mean())


# --- Aggregate metrics container -----------------------------------


@dataclass
class ImageMetrics:
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
    n_segments: int
    n_fit_failures: int
    median_r_squared: float
    mean_boundary_disagreement_px: float
    mean_vessel_width_px: float
    fit_success_rate: float

    def as_row(self) -> dict:
        return self.__dict__.copy()


def compute_metrics(
    gt_artery: np.ndarray,
    gt_vein: np.ndarray,
    recon_hard: np.ndarray,
    split: str,
    filename: str,
    n_segments: int,
    n_fit_failures: int,
    r_squared: list[float],
    widths: list[float],
    fit_success_rate: float,
) -> ImageMetrics:
    """Compute all metrics for one image."""
    gt_all = (gt_artery | gt_vein).astype(bool)
    pr_all = (recon_hard > 0).astype(bool)

    gt_art = gt_artery.astype(bool)
    gt_vn = gt_vein.astype(bool)
    pr_art = (recon_hard == 1) | (recon_hard == 3)
    pr_vn = (recon_hard == 2) | (recon_hard == 3)

    sens, spec = sens_spec(gt_all, pr_all)
    return ImageMetrics(
        split=split,
        filename=filename,
        dice_overall=dice(gt_all, pr_all),
        dice_artery=dice(gt_art, pr_art),
        dice_vein=dice(gt_vn, pr_vn),
        sensitivity=sens,
        specificity=spec,
        boundary_f1_1px=boundary_f1(gt_all, pr_all, 1.0),
        boundary_f1_2px=boundary_f1(gt_all, pr_all, 2.0),
        boundary_f1_3px=boundary_f1(gt_all, pr_all, 3.0),
        n_segments=int(n_segments),
        n_fit_failures=int(n_fit_failures),
        median_r_squared=float(np.median(r_squared)) if r_squared else 0.0,
        mean_boundary_disagreement_px=mean_boundary_disagreement(gt_all, pr_all),
        mean_vessel_width_px=float(np.mean(widths)) if widths else 0.0,
        fit_success_rate=float(fit_success_rate),
    )
