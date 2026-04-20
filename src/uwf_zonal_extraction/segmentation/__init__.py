"""Segmentation adapters (LUNet A/V + ODC) and the SegmentationBundle."""

from uwf_zonal_extraction.segmentation.bundle import SegmentationBundle, SegmentationResult
from uwf_zonal_extraction.segmentation.lunet import LunetSegmenter
from uwf_zonal_extraction.segmentation.lunet_odc import OpticDiscSegmenter

__all__ = [
    "LunetSegmenter",
    "OpticDiscSegmenter",
    "SegmentationBundle",
    "SegmentationResult",
]
