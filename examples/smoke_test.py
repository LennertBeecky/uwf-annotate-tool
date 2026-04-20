"""Smoke test: one UWF image through the full pipeline.

Prints timing at each stage and summary stats.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--laterality", default="OD", choices=["OD", "OS"])
    p.add_argument("--models-dir", default="models")
    p.add_argument(
        "--output-dir",
        default=None,
        help="If given, save debug visualisations + tables here.",
    )
    args = p.parse_args()

    import numpy as np

    from uwf_zonal_extraction.config import ExtractionConfig
    from uwf_zonal_extraction.extractor import extract_caliber_from_image
    from uwf_zonal_extraction.segmentation import SegmentationBundle

    cfg = ExtractionConfig()
    _say("Loading SegmentationBundle (ONNX)...")
    t0 = time.time()
    bundle = SegmentationBundle.from_model_dir(
        args.models_dir,
        tile_size=cfg.segmentation.tile_size,
        stride=cfg.segmentation.stride,
        av_threshold=cfg.segmentation.av_threshold,
    )
    _say(f"  loaded in {time.time() - t0:.1f}s")
    _say(f"  lunet providers: {bundle.lunet.providers}")
    _say(f"  odc providers:   {bundle.odc.providers}")

    _say(f"Running extract_caliber_from_image on {args.image} ...")
    t0 = time.time()
    result = extract_caliber_from_image(
        args.image, cfg, bundle, laterality=args.laterality,
        return_debug=args.output_dir is not None,
    )
    _say(f"  done in {time.time() - t0:.1f}s")

    if args.output_dir is not None:
        from uwf_zonal_extraction.viz import save_debug_pack
        _say(f"Writing debug pack to {args.output_dir} ...")
        t0 = time.time()
        save_debug_pack(result, args.output_dir)
        _say(f"  debug pack written in {time.time() - t0:.1f}s")

    _say("")
    _say(f"image_id: {result.image_id}")
    _say(f"OD: ({result.od_center[0]:.0f}, {result.od_center[1]:.0f}), "
         f"r={result.od_radius_px:.1f}px")
    _say(f"fovea: {result.fovea_center} (conf={result.fovea_confidence:.2f})")
    _say(f"segments: {len(result.segments)}")
    n_art = sum(
        1 for s in result.segments if s.vessel_type == "artery" and not s.uncertain
    )
    n_vei = sum(
        1 for s in result.segments if s.vessel_type == "vein" and not s.uncertain
    )
    n_unc = sum(1 for s in result.segments if s.uncertain)
    _say(f"  arteries: {n_art}   veins: {n_vei}   uncertain: {n_unc}")
    _say(f"measurements: {len(result.measurements)}")
    _say(f"zonal cells:  {len(result.zonal)}")

    _say("")
    _say("--- Z1 (Zone B) ---")
    for z in sorted(result.zonal, key=lambda z: (z.zone_index, z.quadrant)):
        if z.zone_index != 1:
            continue
        flags_s = ",".join(sorted(z.flags)) or "-"
        crae_s = f"{z.crae_px:.1f}" if z.crae_px else "NaN"
        crve_s = f"{z.crve_px:.1f}" if z.crve_px else "NaN"
        avr_s = f"{z.avr:.3f}" if z.avr else "NaN"
        _say(
            f"  Z1 {z.quadrant}: "
            f"CRAE={crae_s}px ({z.n_arteries}a)  "
            f"CRVE={crve_s}px ({z.n_veins}v)  "
            f"AVR={avr_s}  flags={flags_s}"
        )

    _say("")
    _say("--- Radial profile (mean across quadrants) ---")
    _say(f"{'zone':>4}  {'CRAE px':>8} {'CRVE px':>8} {'AVR':>6}  {'n_a':>4} {'n_v':>4}  notes")
    for zi in range(1, 8):
        rows = [z for z in result.zonal if z.zone_index == zi]
        craes = [z.crae_px for z in rows if z.crae_px is not None]
        crves = [z.crve_px for z in rows if z.crve_px is not None]
        avrs = [z.avr for z in rows if z.avr is not None]
        n_a = sum(z.n_arteries for z in rows)
        n_v = sum(z.n_veins for z in rows)
        crae_s = f"{sum(craes) / len(craes):8.2f}" if craes else "     NaN"
        crve_s = f"{sum(crves) / len(crves):8.2f}" if crves else "     NaN"
        avr_s = f"{sum(avrs) / len(avrs):6.3f}" if avrs else "   NaN"
        flags_all = set().union(*(z.flags for z in rows))
        notes = "dewarp_skipped" if "dewarp_skipped" in flags_all else ""
        _say(f"  Z{zi}  {crae_s} {crve_s} {avr_s}  {n_a:>4} {n_v:>4}  {notes}")

    peripheral = [z for z in result.zonal if z.zone_index >= 6]
    flagged = sum(1 for z in peripheral if "dewarp_skipped" in z.flags)
    _say("")
    _say(
        f"dewarp_skipped propagation: {flagged}/{len(peripheral)} "
        "peripheral cells flagged (expect all)."
    )

    md = result.run_metadata
    _say("")
    _say("--- Run metadata ---")
    _say(f"  package {md.get('package_version')}  numpy {md.get('numpy_version')}")
    _say(
        f"  dewarp_applied={md.get('dewarp_applied')}  "
        f"device={md.get('dewarp_device')}"
    )
    _say(f"  lunet_sha256={md.get('lunet_sha256', '')[:16]}...")
    _say(f"  odc_sha256  ={md.get('odc_sha256', '')[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
