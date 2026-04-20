"""LUNetV2 Optic Disc/Cup segmentation wrapper, with ellipse-fit radius.

Re-authored from the parent repo. The key refinement is `disc_geometry`,
which returns a robust disc centre + radius from an ellipse fit to the
largest disc-probability contour, not sqrt(area/π). The ellipse fit
handles tilted and partially occluded UWF discs; sqrt(area/π)
under-estimates and silently corrupts DD-normalized zones.

Channels:
    ch0 = disc, ch1 = cup, ch2/ch3 = background regions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uwf_zonal_extraction.segmentation.providers import build_session

ODC_INPUT_SIZE = 512
CH_DISC = 0
CH_CUP = 1


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))


@dataclass(frozen=True)
class DiscGeometry:
    """Robust optic-disc geometry from an ellipse fit."""

    center: tuple[float, float]           # (x, y) in pixel coords
    radius_px: float                      # major semi-axis of the ellipse fit
    ellipse: tuple[tuple[float, float], tuple[float, float], float] | None
    # ellipse = ((cx, cy), (major_axis, minor_axis), angle_deg)
    axis_ratio: float                     # major/minor; > 2 flags a fallback
    fit_method: str                       # 'ellipse' or 'area_fallback'


class OpticDiscSegmenter:
    def __init__(self, model_path: str | Path) -> None:
        self.session = build_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = str(model_path)
        self.providers = self.session.get_providers()

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (ODC_INPUT_SIZE, ODC_INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run OD/cup segmentation. Returns (H, W, 4) float32 probabilities."""
        h, w = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        probs = _sigmoid(logits)
        probs = np.transpose(probs[0], (1, 2, 0))
        if (h, w) != (ODC_INPUT_SIZE, ODC_INPUT_SIZE):
            probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
        return probs

    def disc_mask(self, image_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict(image_bgr)
        return (probs[:, :, CH_DISC] > threshold).astype(np.uint8)

    def cup_mask(self, image_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict(image_bgr)
        return (probs[:, :, CH_CUP] > threshold).astype(np.uint8)

    def disc_geometry(
        self,
        image_bgr: np.ndarray,
        threshold: float = 0.5,
    ) -> DiscGeometry:
        """Ellipse-fit optic-disc geometry (centre, radius, axes).

        Uses the largest disc-probability contour. Ellipse semi-major
        axis is DD/2. Falls back to sqrt(area/π) only if the ellipse
        fit fails or returns an axis ratio > 2 (non-disc-like).
        """
        mask = self.disc_mask(image_bgr, threshold=threshold)
        return _disc_geometry_from_mask(mask)

    def disc_geometry_uwf(
        self,
        image_bgr: np.ndarray,
        threshold: float = 0.5,
        crop_size: int = 1500,
        crop_center: tuple[int, int] | None = None,
    ) -> DiscGeometry:
        """Two-stage OD detection for large UWF frames.

        LUNet-ODC was CFP-trained; feeding it a full 4000×4000 Optos
        collapses the disc to ~13 px at the 512-px model input. This
        helper crops a `crop_size × crop_size` square centred on
        `crop_center` (image centre if None — valid for non-steered
        Optos), runs OD segmentation on the crop, then remaps the
        detected centre and radius back to full-frame coordinates.

        LUNet A/V inference is unaffected — it still runs on the full
        frame so Z6/Z7 periphery is preserved.
        """
        h, w = image_bgr.shape[:2]
        if crop_center is None:
            crop_center = (w // 2, h // 2)
        cx_full, cy_full = crop_center

        half = crop_size // 2
        x0 = max(0, cx_full - half)
        y0 = max(0, cy_full - half)
        x1 = min(w, cx_full + half)
        y1 = min(h, cy_full + half)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return DiscGeometry(
                center=(w / 2, h / 2), radius_px=0.0, ellipse=None,
                axis_ratio=float("nan"), fit_method="empty",
            )

        crop_mask = self.disc_mask(crop, threshold=threshold)
        geom = _disc_geometry_from_mask(crop_mask)
        if geom.radius_px <= 0:
            return geom

        # Translate centre back to full-frame coordinates.
        full_cx = float(geom.center[0]) + x0
        full_cy = float(geom.center[1]) + y0
        ellipse_full = None
        if geom.ellipse is not None:
            (ex, ey), axes, angle = geom.ellipse
            ellipse_full = ((float(ex) + x0, float(ey) + y0), axes, angle)
        return DiscGeometry(
            center=(full_cx, full_cy),
            radius_px=geom.radius_px,
            ellipse=ellipse_full,
            axis_ratio=geom.axis_ratio,
            fit_method=f"{geom.fit_method}_uwf_crop",
        )


def _disc_geometry_from_mask(mask: np.ndarray) -> DiscGeometry:
    """Extract DiscGeometry from a binary disc mask via ellipse fit."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        h, w = mask.shape[:2]
        return DiscGeometry(
            center=(w / 2, h / 2),
            radius_px=0.0,
            ellipse=None,
            axis_ratio=float("nan"),
            fit_method="empty",
        )

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    ellipse = None
    ratio = float("nan")
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            ellipse = None

    if ellipse is not None:
        (cx, cy), (ax1, ax2), _angle = ellipse
        major = max(ax1, ax2)
        minor = max(min(ax1, ax2), 1e-6)
        ratio = major / minor
        if ratio <= 2.0:
            return DiscGeometry(
                center=(float(cx), float(cy)),
                radius_px=float(major / 2.0),
                ellipse=ellipse,
                axis_ratio=float(ratio),
                fit_method="ellipse",
            )

    # Fallback: centroid + sqrt(area/π).
    M = cv2.moments(contour)
    if M["m00"] == 0:
        h, w = mask.shape[:2]
        cx, cy = w / 2, h / 2
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    radius = float(np.sqrt(max(area, 0.0) / np.pi))
    return DiscGeometry(
        center=(float(cx), float(cy)),
        radius_px=radius,
        ellipse=ellipse,
        axis_ratio=ratio,
        fit_method="area_fallback",
    )
