"""LUNetV2 A/V segmentation wrapper.

Re-authored from the parent Physics-Informed_Fundus repo. Differences:
  - No dependency on parent-repo paths.
  - `predict_tiled` pads the input to the next multiple of stride
    (reflect-pad) and marks under-weighted output pixels as NaN rather
    than clamping at 1e-6 (the legacy behaviour amplified noise in
    unsampled regions).
  - `predict_tiled` optionally replicates the green channel across all
    three input channels (the CFP-trained LUNet works best with that
    convention on Optos UWF).

Channels (from LUNetV2 eval.py, Fhima et al.):
    ch0 = artery, ch1 = vein, ch2/ch3 = unused
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from uwf_zonal_extraction.segmentation.providers import build_session

LUNET_INPUT_SIZE = 1472
LUNET_CH_ARTERY = 0
LUNET_CH_VEIN = 1


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))


class LunetSegmenter:
    """ONNX-backed A/V vessel segmentation."""

    def __init__(self, model_path: str | Path) -> None:
        self.session = build_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = str(model_path)
        self.providers = self.session.get_providers()

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (LUNET_INPUT_SIZE, LUNET_INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Single-pass inference on a BGR image. Returns (H, W, 4) float32 probs."""
        h, w = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        probs = _sigmoid(logits)
        probs = np.transpose(probs[0], (1, 2, 0))
        if (h, w) != (LUNET_INPUT_SIZE, LUNET_INPUT_SIZE):
            probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
        return probs

    def predict_tiled(
        self,
        image_bgr: np.ndarray,
        tile_size: int = LUNET_INPUT_SIZE,
        stride: int = LUNET_INPUT_SIZE // 2,
        green_input: bool = True,
        min_weight: float = 1e-3,
    ) -> np.ndarray:
        """Tiled inference with Gaussian blending, reflect-padded edges.

        Args:
            image_bgr:   (H, W, 3) uint8 BGR.
            tile_size:   square tile side (default 1472, LUNet native).
            stride:      step between tiles (default 736, 50% overlap).
            green_input: replicate the green channel into all 3 channels
                         before inference (recommended for Optos UWF
                         with the CFP-trained LUNet).
            min_weight:  pixels whose cumulative tile weight is below
                         this are set to NaN in the output (prevents
                         the legacy 1e-6-clamp noise amplification).

        Returns:
            (H, W, 4) float32 probability map. NaN where coverage was
            insufficient.
        """
        if green_input:
            g = image_bgr[:, :, 1]
            image_bgr = cv2.merge([g, g, g])

        h0, w0 = image_bgr.shape[:2]

        # Reflect-pad so the last tile at (h-tile, w-tile) is fully inside.
        pad_h = (-(h0 - tile_size) % stride) if h0 > tile_size else tile_size - h0
        pad_w = (-(w0 - tile_size) % stride) if w0 > tile_size else tile_size - w0
        if pad_h or pad_w:
            image_bgr = cv2.copyMakeBorder(
                image_bgr, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
            )
        h, w = image_bgr.shape[:2]

        # Gaussian weight map, tile-sized.
        axis = np.linspace(-2, 2, tile_size)
        g = np.exp(-(axis**2) / 2)
        weight_tile = np.outer(g, g).astype(np.float32)

        probs_acc = np.zeros((h, w, 4), dtype=np.float32)
        weight_acc = np.zeros((h, w), dtype=np.float32)

        for ty in range(0, h - tile_size + 1, stride):
            for tx in range(0, w - tile_size + 1, stride):
                patch = image_bgr[ty : ty + tile_size, tx : tx + tile_size]
                probs = self.predict(patch)  # (tile, tile, 4)
                probs_acc[ty : ty + tile_size, tx : tx + tile_size] += (
                    probs * weight_tile[:, :, None]
                )
                weight_acc[ty : ty + tile_size, tx : tx + tile_size] += weight_tile

        # Normalize; NaN where weight too low.
        out = np.full_like(probs_acc, np.nan)
        valid = weight_acc > min_weight
        out[valid] = probs_acc[valid] / weight_acc[valid, None]

        # Undo padding.
        out = out[:h0, :w0]
        return out
