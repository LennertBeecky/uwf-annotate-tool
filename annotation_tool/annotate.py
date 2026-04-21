"""Napari-based UWF skeleton annotation tool.

Subcommands:
    annotate <image_or_dir>   (default)   open napari, paint artery/vein skeletons
    preview <image_path>                  read-only napari view of saved skeletons

Examples:
    python annotate.py path/to/uwf.png
    python annotate.py path/to/uwf_folder/
    python annotate.py preview path/to/uwf.png
    python annotate.py path/to/uwf_folder/ --overwrite
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import (  # type: ignore  # noqa: E402
    IMG_EXTS,
    SaveValidation,
    already_annotated,
    append_time_log,
    build_multiscale_pyramid,
    human_duration,
    list_images,
    load_image_rgb,
    now_iso,
    save_skeleton_png,
    skeletonise_mask,
    validate_saved_skeleton,
)

if TYPE_CHECKING:  # keep type hints without importing napari at module load
    import napari  # noqa: F401


OUTPUT_DIR = Path("annotations_uwf")
TIMES_CSV = OUTPUT_DIR / "annotation_times.csv"


# ---- napari session ---------------------------------------------------


def _open_annotation_session(
    image_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> bool:
    """Run one annotation session. Return True if user saved, False if skipped."""
    # Import napari lazily — the utils test suite must not require it.
    import napari
    from qtpy.QtCore import Qt  # napari[all] ships qtpy

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]
    print("")
    print("=== UWF Annotation Session ===")
    print(f"Image:           {image_path.name}")
    print(f"Full resolution: {h}x{w}")
    print(f"Previously annotated: {'Yes' if already_annotated(image_path.stem, output_dir) else 'No'}")
    print(f"Start time:      {now_iso()}")
    print("")
    print("Tips:")
    print("  [3] paint | [1] pan/zoom | Tab cycle layers")
    print("  [ / ] decrease / increase brush size")
    print("  [q] save and advance   [s] skip without saving")
    print("  Target time per image: 30–45 minutes")
    print("")

    layers, multiscale = build_multiscale_pyramid(image_rgb)

    viewer = napari.Viewer(
        title=f"UWF annotation — {image_path.name}  |  [3] paint  Tab layer  [q] save [s] skip",
    )
    viewer.add_image(
        layers if multiscale else layers[0],
        multiscale=multiscale,
        name="fundus",
    )

    # Label layers at FULL resolution (labels aren't multiscale-friendly)
    artery_layer = viewer.add_labels(
        np.zeros((h, w), dtype=np.uint8),
        name="artery",
    )
    vein_layer = viewer.add_labels(
        np.zeros((h, w), dtype=np.uint8),
        name="veins",
    )
    # Force well-known colours so annotators see red/blue regardless of theme.
    try:
        artery_layer.colormap = {1: (1.0, 0.0, 0.0, 1.0), None: (0, 0, 0, 0)}
    except Exception:
        pass
    try:
        vein_layer.colormap = {1: (0.0, 0.3, 1.0, 1.0), None: (0, 0, 0, 0)}
    except Exception:
        pass

    artery_layer.selected_label = 1
    vein_layer.selected_label = 1
    artery_layer.brush_size = 2
    vein_layer.brush_size = 2

    viewer.layers.selection.active = artery_layer

    # Status-bar overlay showing active layer + brush size
    def _refresh_overlay() -> None:
        active = viewer.layers.selection.active
        brush = getattr(active, "brush_size", "—")
        name = active.name if active is not None else "—"
        viewer.text_overlay.text = f"class: {name}   brush: {brush}"
        viewer.text_overlay.visible = True

    viewer.text_overlay.font_size = 14
    viewer.text_overlay.color = "white"
    _refresh_overlay()

    state = {"should_save": True}

    @viewer.bind_key("q", overwrite=True)
    def _save_quit(_viewer):
        state["should_save"] = True
        viewer.close()

    @viewer.bind_key("s", overwrite=True)
    def _skip(_viewer):
        state["should_save"] = False
        print("  [s] skipping without saving")
        viewer.close()

    @viewer.bind_key("Tab", overwrite=True)
    def _cycle_layer(_viewer):
        # cycle between artery and veins (skip the image layer)
        label_layers = [artery_layer, vein_layer]
        active = viewer.layers.selection.active
        idx = label_layers.index(active) if active in label_layers else -1
        next_idx = (idx + 1) % len(label_layers)
        viewer.layers.selection.active = label_layers[next_idx]
        _refresh_overlay()

    @viewer.bind_key("1", overwrite=True)
    def _pan_mode(_viewer):
        active = viewer.layers.selection.active
        if hasattr(active, "mode"):
            active.mode = "pan_zoom"
        _refresh_overlay()

    @viewer.bind_key("3", overwrite=True)
    def _paint_mode(_viewer):
        active = viewer.layers.selection.active
        if hasattr(active, "mode"):
            active.mode = "paint"
        _refresh_overlay()

    @viewer.bind_key("[", overwrite=True)
    def _smaller_brush(_viewer):
        active = viewer.layers.selection.active
        if hasattr(active, "brush_size"):
            active.brush_size = max(1, int(active.brush_size) - 1)
        _refresh_overlay()

    @viewer.bind_key("]", overwrite=True)
    def _bigger_brush(_viewer):
        active = viewer.layers.selection.active
        if hasattr(active, "brush_size"):
            active.brush_size = int(active.brush_size) + 1
        _refresh_overlay()

    # Run the napari event loop (blocks until the window is closed).
    t_start = time.time()
    napari.run()
    duration = time.time() - t_start

    if not state["should_save"]:
        print(f"  [skip] session {human_duration(duration)} — nothing saved.")
        return False

    # Post-process + save
    artery_mask = np.asarray(artery_layer.data, dtype=np.uint8)
    vein_mask = np.asarray(vein_layer.data, dtype=np.uint8)

    if not (artery_mask.any() or vein_mask.any()):
        print("  [warn] both labels empty — nothing painted. Skipping save.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    art_path = output_dir / f"{stem}_artery.png"
    vein_path = output_dir / f"{stem}_veins.png"

    if (art_path.exists() or vein_path.exists()) and not overwrite:
        ans = input(f"  [confirm] overwrite existing {stem}_*.png? [y/N] ").strip().lower()
        if ans != "y":
            print("  not overwriting — skipping save.")
            return False

    artery_skel = skeletonise_mask(artery_mask)
    vein_skel = skeletonise_mask(vein_mask)
    save_skeleton_png(artery_skel, art_path)
    save_skeleton_png(vein_skel, vein_path)

    art_valid = validate_saved_skeleton(art_path, (h, w))
    vein_valid = validate_saved_skeleton(vein_path, (h, w))

    art_pixels = int((artery_skel > 0).sum())
    vein_pixels = int((vein_skel > 0).sum())

    print("")
    print(f"Saved: {art_path}  ({art_pixels} skeleton pixels)")
    for m in art_valid.messages:
        print(f"  [art warn] {m}")
    print(f"Saved: {vein_path}  ({vein_pixels} skeleton pixels)")
    for m in vein_valid.messages:
        print(f"  [vein warn] {m}")
    print(f"Duration: {human_duration(duration)}")
    print(f"=== Complete ===")

    append_time_log(
        TIMES_CSV if output_dir == OUTPUT_DIR else output_dir / "annotation_times.csv",
        {
            "timestamp": now_iso(),
            "image_filename": image_path.name,
            "duration_seconds": round(duration, 2),
            "artery_pixel_count": art_pixels,
            "vein_pixel_count": vein_pixels,
        },
    )
    return True


# ---- preview -----------------------------------------------------------


def _open_preview(image_path: Path, output_dir: Path) -> None:
    import napari
    from PIL import Image as _Image

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]
    layers, multiscale = build_multiscale_pyramid(image_rgb)

    stem = image_path.stem
    art_path = output_dir / f"{stem}_artery.png"
    vein_path = output_dir / f"{stem}_veins.png"

    viewer = napari.Viewer(title=f"UWF preview — {image_path.name} (read-only)")
    viewer.add_image(layers if multiscale else layers[0],
                     multiscale=multiscale, name="fundus")
    for label_name, color, p in (
        ("artery", (1.0, 0.0, 0.0, 1.0), art_path),
        ("veins",  (0.0, 0.3, 1.0, 1.0), vein_path),
    ):
        if p.exists():
            arr = np.asarray(_Image.open(str(p)))
            lbl = (arr > 0).astype(np.uint8)
            layer = viewer.add_labels(lbl, name=label_name)
            try:
                layer.colormap = {1: color, None: (0, 0, 0, 0)}
            except Exception:
                pass
            layer.editable = False
        else:
            print(f"  [preview] {p.name} missing — skipping layer")

    napari.run()


# ---- directory walk ----------------------------------------------------


def _walk_directory(
    src_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    images = list_images(src_dir)
    if not images:
        print(f"No images found in {src_dir}")
        return
    print(f"Found {len(images)} images in {src_dir}.")
    completed = 0

    def _sigint(_signum, _frame):
        print(f"\n[ctrl-c] completed {completed}/{len(images)} this session. exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    for i, img in enumerate(images):
        stem = img.stem
        if already_annotated(stem, output_dir) and not overwrite:
            print(f"  [{i + 1}/{len(images)}] skip (already annotated): {img.name}")
            continue
        print(f"\n--- [{i + 1}/{len(images)}] {img.name} ---")
        ok = _open_annotation_session(img, output_dir, overwrite=overwrite)
        if ok:
            completed += 1
        if i + 1 < len(images):
            input("press Enter for next image, or Ctrl+C to quit... ")
    print(f"\nSession complete: {completed}/{len(images)} images annotated.")


# ---- CLI --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="annotate.py",
        description="Napari-based UWF skeleton annotation tool.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # `preview` subcommand
    p_prev = sub.add_parser("preview", help="Open saved skeletons read-only over the image.")
    p_prev.add_argument("image", type=Path)
    p_prev.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where masks live (default {OUTPUT_DIR}).")

    # Default (positional path → annotate single image or walk directory)
    parser.add_argument("path", nargs="?", type=Path,
                        help="Image file or directory of images to annotate.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where skeleton PNGs are written (default {OUTPUT_DIR}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing skeleton files without asking.")

    args = parser.parse_args()

    # Subcommand dispatch
    if args.cmd == "preview":
        if not args.image.exists():
            print(f"image not found: {args.image}")
            return 1
        _open_preview(args.image, args.output_dir)
        return 0

    # Default: annotate
    if args.path is None:
        parser.print_help()
        return 1
    if not args.path.exists():
        print(f"path not found: {args.path}")
        return 1
    if args.path.is_dir():
        _walk_directory(args.path, args.output_dir, overwrite=args.overwrite)
    else:
        _open_annotation_session(args.path, args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
