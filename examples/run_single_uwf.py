"""End-to-end demo: process one UWF image into CRAE/CRVE/AVR tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from uwf_zonal_extraction.config import ExtractionConfig
from uwf_zonal_extraction.extractor import extract_caliber_from_image
from uwf_zonal_extraction.segmentation import SegmentationBundle
from uwf_zonal_extraction.viz import save_extraction_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--laterality", type=str, default=None, choices=["OD", "OS", None])
    args = parser.parse_args()

    cfg = ExtractionConfig.from_yaml(args.config) if args.config else ExtractionConfig()
    bundle = SegmentationBundle.from_model_dir(
        args.models_dir,
        tile_size=cfg.segmentation.tile_size,
        stride=cfg.segmentation.stride,
        av_threshold=cfg.segmentation.av_threshold,
    )
    result = extract_caliber_from_image(args.image, cfg, bundle, laterality=args.laterality)
    save_extraction_summary(result, args.output_dir)
    print(f"Wrote {args.output_dir}/measurements.parquet and zonal.parquet")


if __name__ == "__main__":
    main()
