"""SegmentationBundle — one object that bundles LUNet + ODC + fovea + dewarp."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from uwf_zonal_extraction.segmentation.lunet import (
    LUNET_CH_ARTERY,
    LUNET_CH_VEIN,
    LunetSegmenter,
)
from uwf_zonal_extraction.segmentation.lunet_odc import (
    DiscGeometry,
    OpticDiscSegmenter,
)


@dataclass
class SegmentationResult:
    """Everything the skeleton/extraction stages consume from segmentation."""

    av_mask: np.ndarray              # (H, W) uint8  0=bg, 1=artery, 2=vein
    av_probs: np.ndarray             # (H, W, 2) float32: (p_artery, p_vein)
    retina_mask: np.ndarray          # (H, W) uint8
    od: DiscGeometry
    od_probs: np.ndarray             # (H, W) float32
    fovea: Optional[tuple[float, float]] = None   # (x, y) or None
    fovea_confidence: float = 0.0

    @property
    def height(self) -> int:
        return int(self.av_mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.av_mask.shape[1])


@dataclass
class SegmentationBundle:
    """Injection point for all segmentation models.

    Kept separate from ExtractionConfig so tests can inject mocks without
    loading 800 MB of ONNX.
    """

    lunet: LunetSegmenter
    odc: OpticDiscSegmenter
    tile_size: int = 1472
    stride: int = 736
    av_threshold: float = 0.5
    # OD two-stage parameters — the CFP-trained ODC collapses the disc to
    # ~13 px when fed a full 4000×4000 Optos, so we crop for OD only.
    od_crop_size: int = 1500
    od_crop_size_trigger_px: int = 2000      # use two-stage if image side >= this

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        lunet_filename: str = "lunetv2Large.onnx",
        odc_filename: str = "lunetv2_odc.onnx",
        **kwargs,
    ) -> "SegmentationBundle":
        model_dir = Path(model_dir)
        return cls(
            lunet=LunetSegmenter(model_dir / lunet_filename),
            odc=OpticDiscSegmenter(model_dir / odc_filename),
            **kwargs,
        )

    def run(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Full segmentation: OD + A/V (tiled, green-replicated) + retina mask."""
        od_probs_4 = self.odc.predict(image_bgr)
        od_probs = od_probs_4[:, :, 0]                         # disc channel

        av_probs_4 = self.lunet.predict_tiled(
            image_bgr,
            tile_size=self.tile_size,
            stride=self.stride,
            green_input=True,
        )                                                      # (H, W, 4), may contain NaN
        pa = np.nan_to_num(av_probs_4[:, :, LUNET_CH_ARTERY], nan=0.0)
        pv = np.nan_to_num(av_probs_4[:, :, LUNET_CH_VEIN], nan=0.0)

        retina_mask = _retina_mask(image_bgr)

        # Gate the probability maps by the retina mask BEFORE thresholding,
        # so LUNet hallucinations outside the Optos fundus circle (black
        # frame, reflections, eyelid) never enter the skeleton.
        outside = retina_mask == 0
        pa = np.where(outside, 0.0, pa)
        pv = np.where(outside, 0.0, pv)

        # Hard mask: a/v if above threshold AND higher than the other class.
        is_a = (pa > self.av_threshold) & (pa > pv)
        is_v = (pv > self.av_threshold) & (pv > pa)
        av_mask = np.zeros_like(pa, dtype=np.uint8)
        av_mask[is_a] = 1
        av_mask[is_v] = 2

        h, w = image_bgr.shape[:2]
        if max(h, w) >= self.od_crop_size_trigger_px:
            od = self.odc.disc_geometry_uwf(
                image_bgr, threshold=0.5, crop_size=self.od_crop_size,
            )
        else:
            od = self.odc.disc_geometry(image_bgr, threshold=0.5)

        return SegmentationResult(
            av_mask=av_mask,
            av_probs=np.stack([pa, pv], axis=-1).astype(np.float32),
            retina_mask=retina_mask,
            od=od,
            od_probs=od_probs.astype(np.float32),
        )


def _retina_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the circular retinal region (excludes black frame).

    Threshold + morphology; identical to the trick in the legacy
    hybrid_pipeline.py.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)
