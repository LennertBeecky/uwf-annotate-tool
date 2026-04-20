"""Optos stereographic-to-equirectangular dewarping — v1 STUB.

The full implementation requires device-specific stereographic parameters
(Sagong 2014 / Tan 2021) that vary between P200 / Daytona / California /
Silverstone. v1 ships a **passthrough** implementation with a clearly
flagged TODO; the pipeline remains functional for zones Z1-Z5 (<3 DD)
where the distortion is small, and emits a `dewarp_skipped=True` flag
for Z6-Z7 output so downstream analysis can exclude them from research
claims.

The STUB interface matches the final API so callers don't change when
the real implementation lands.

TODO: implement the forward map for at least 'daytona' (most common
clinical Optos). Acceptance: mm/px isotropy within 5% across the valid
retinal ROI on a test phantom.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from uwf_zonal_extraction.config import DewarpCfg


@dataclass
class DewarpResult:
    image: np.ndarray                        # (H, W, 3) dewarped (or original if 'none')
    mm_per_px: np.ndarray | None             # (H, W) float32 or None
    device: str
    axial_length_mm: float
    applied: bool                            # False if 'none' or passthrough


def dewarp_optos(
    image_bgr: np.ndarray,
    cfg: DewarpCfg,
    axial_length_mm: float | None = None,
) -> DewarpResult:
    """Return a DewarpResult. In v1 this is passthrough for any device.

    When the real implementation lands, behaviour diverges by `cfg.device`:
      - 'none':              passthrough, no mm_per_px map
      - 'daytona'/...:       forward stereographic -> equirectangular warp
    """
    axial_length_mm = axial_length_mm or cfg.axial_length_mm_default

    if cfg.device.lower() == "none":
        return DewarpResult(
            image=image_bgr,
            mm_per_px=None,
            device="none",
            axial_length_mm=axial_length_mm,
            applied=False,
        )

    # TODO: real stereographic dewarp. Placeholder: passthrough + warn.
    warnings.warn(
        f"dewarp_optos: device='{cfg.device}' is not yet implemented in v1; "
        "running in passthrough mode. Zones Z6-Z7 (>3 DD) will be flagged "
        "'dewarp_skipped' in the output.",
        UserWarning,
        stacklevel=2,
    )
    return DewarpResult(
        image=image_bgr,
        mm_per_px=None,
        device=cfg.device,
        axial_length_mm=axial_length_mm,
        applied=False,
    )
