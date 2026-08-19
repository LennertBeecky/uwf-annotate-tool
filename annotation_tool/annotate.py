"""Napari-based retinal vessel annotation tool (UWF and DVA).

Two output modes:
    default        filled vessel masks; the painted thickness IS the vessel
                   width (pixel annotations).
    --skeleton     1-px centrelines only; width is re-derived downstream by
                   profile fitting on the raw image.

Subcommands:
    annotate <image_or_dir>   (default)   open napari, paint artery/vein vessels
    preview <image_path>                  read-only napari view of saved masks

Examples:
    python annotate.py path/to/uwf.png
    python annotate.py path/to/uwf_folder/ --overwrite
    python annotate.py preview path/to/uwf.png

    # DVA: start from the filled A/V masks and keep their width
    python annotate.py databases/DVA/raw/images/ \
        --prefill masks --masks-dir databases/DVA/raw \
        --output-dir annotations_dva
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
    AVSwapper,
    SaveValidation,
    already_annotated,
    append_edit_log,
    append_time_log,
    binarise_mask,
    detect_prefill_kind,
    build_multiscale_pyramid,
    human_duration,
    list_images,
    load_image_rgb,
    load_mask_prefill,
    lunet_prefill_masks,
    now_iso,
    save_skeleton_png,
    skeleton_edit_distance,
    skeletonise_mask,
    validate_saved_skeleton,
)

if TYPE_CHECKING:  # keep type hints without importing napari at module load
    import napari  # noqa: F401


# A/V reclassification keys, as (key, swap-mode). `undo` and `global` are
# handled specially. napari checks the active layer's keymap before the
# viewer's, so these are registered on both — see `bind_key_everywhere`.
#
# Every key here must be one napari does not already claim: `f` is its
# labels fill mode and `x` swaps the selected/background label, and those
# actions fire *alongside* a bind_key handler rather than being replaced by
# it. tests/test_keybinds.py checks this against the installed napari.
SWAP_KEYS = [
    ("w", "branch"),
    ("Shift-W", "component"),
    ("h", "range"),
    ("g", "disk"),
    (".", "widen"),
    (",", "narrow"),
    ("u", "undo"),
    ("Meta-Z", "undo"),      # cmd+z on macOS
    ("Control-Z", "undo"),
    ("Shift-X", "global"),
]


def bind_key_everywhere(targets, key: str, func) -> None:
    """Bind `key` → `func` on the viewer and on every label layer.

    A viewer-level binding alone loses to a Labels-layer default of the
    same name. Targets that reject the key are skipped rather than
    aborting the rest of the registration.
    """
    for target in targets:
        try:
            target.bind_key(key, func, overwrite=True)
        except Exception as exc:  # napari version differences, odd key names
            print(f"  [warn] could not bind '{key}' on "
                  f"{getattr(target, 'name', target)}: {exc}")


OUTPUT_DIR = Path("annotations_uwf")
TIMES_CSV = OUTPUT_DIR / "annotation_times.csv"
EDITS_CSV_NAME = "annotation_edits.csv"
LUNET_CACHE_DIRNAME = "_lunet_cache"
DEFAULT_LUNET_MODEL = Path("models/lunetv2Large.onnx")


def ask_confirm(viewer, question: str) -> bool:
    """Modal yes/no dialog inside the napari session; the default button
    is No so a stray Enter cannot confirm. Without QtWidgets (headless or
    scripted runs) it answers Yes, which is the pre-dialog behaviour."""
    try:
        from qtpy.QtWidgets import QMessageBox
    except Exception:
        return True
    parent = getattr(getattr(viewer, "window", None), "_qt_window", None)
    answer = QMessageBox.question(
        parent, "Confirm", question,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
    )
    return answer == QMessageBox.Yes


# ---- napari session ---------------------------------------------------


def _open_annotation_session(
    image_path: Path,
    output_dir: Path,
    overwrite: bool,
    prefill_source: str = "none",
    prefill_masks: tuple[np.ndarray, np.ndarray] | None = None,
    lunet_thresh: float | None = None,
    boundaries: bool = True,
) -> bool:
    """Run one annotation session. Return True if user saved, False if skipped.

    If `prefill_masks` is provided, the artery/vein paint layers start
    populated with those binary masks (value 1) and edit-distance metrics
    against the seed skeleton are written to `annotation_edits.csv`.
    """
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
    if prefill_source != "none":
        extra = f"  (thresh={lunet_thresh})" if prefill_source == "lunet" else ""
        print(f"Prefill source:  {prefill_source}{extra}")
    print(f"Output:          {'filled masks (boundaries — width preserved)' if boundaries else '1-px skeletons'}")
    print(f"Start time:      {now_iso()}")
    print("")
    print("Tips:")
    print("  [3] paint | [1] pan/zoom | Tab cycle layers")
    print("  [ / ] decrease / increase brush size")
    print("  A/V fix — hover the vessel, then:")
    print("    [w] swap this segment   [W] swap the whole vessel")
    print("    [h] [h] swap only the stretch between two marks")
    print("    [g] swap a brush-sized blob   [X] swap both layers")
    if boundaries:
        print("  width — hover the vessel, then:")
        print("    [.] widen it by 1px   [,] narrow it by 1px")
    print("    [u] undo last swap   (alt+click = [w], pan/zoom mode only)")
    print("  [q] save and advance   [s] skip without saving")
    print("  Target time per image: 1 minute")
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
    if prefill_masks is not None:
        art_init = (prefill_masks[0] > 0).astype(np.uint8)
        vein_init = (prefill_masks[1] > 0).astype(np.uint8)
        if art_init.shape != (h, w) or vein_init.shape != (h, w):
            raise ValueError(
                f"prefill mask shapes {art_init.shape}/{vein_init.shape} "
                f"don't match image {(h, w)}"
            )
        _encode_seed = binarise_mask if boundaries else skeletonise_mask
        seed_artery_skel = _encode_seed(art_init)
        seed_vein_skel = _encode_seed(vein_init)
        _enc = "filled" if boundaries else "skeleton"
        print(f"Seed pixels:     artery={int(art_init.sum())}  "
              f"vein={int(vein_init.sum())}  "
              f"({_enc}: art={int((seed_artery_skel > 0).sum())} "
              f"vein={int((seed_vein_skel > 0).sum())})")
    else:
        art_init = np.zeros((h, w), dtype=np.uint8)
        vein_init = np.zeros((h, w), dtype=np.uint8)
        seed_artery_skel = np.zeros((h, w), dtype=np.uint8)
        seed_vein_skel = np.zeros((h, w), dtype=np.uint8)

    artery_layer = viewer.add_labels(art_init, name="artery")
    vein_layer = viewer.add_labels(vein_init, name="veins")
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
    default_brush = 5 if boundaries else 2
    artery_layer.brush_size = default_brush
    vein_layer.brush_size = default_brush

    viewer.layers.selection.active = artery_layer

    # Status-bar overlay showing active layer + brush size
    def _refresh_overlay(note: str = "") -> None:
        active = viewer.layers.selection.active
        brush = getattr(active, "brush_size", "—")
        name = active.name if active is not None else "—"
        suffix = f"   |   {note}" if note else ""
        viewer.text_overlay.text = f"class: {name}   brush: {brush}{suffix}"
        viewer.text_overlay.visible = True

    viewer.text_overlay.font_size = 14
    viewer.text_overlay.color = "white"
    _refresh_overlay()

    state = {"should_save": True}

    # ---- A/V reclassification ----------------------------------------
    #
    # Artery and vein live on separate layers, so fixing a swapped vessel
    # used to mean erasing on one layer and re-painting on the other. These
    # handlers move the vessel under the cursor across in one keystroke.

    swapper = AVSwapper({"artery": np.asarray(artery_layer.data),
                         "veins": np.asarray(vein_layer.data)})

    # One undo timeline for two histories. Brush strokes live in napari's
    # per-layer undo stack; swaps and width edits live in the swapper's.
    # Undoing them out of order leaves orphans — a stroke that was swapped
    # across would be removed from the source layer while its copy stayed
    # on the destination — so record the order and walk it back.
    timeline: list[tuple[str, object]] = []
    for _lyr in (artery_layer, vein_layer):
        try:
            _lyr.events.paint.connect(
                lambda _e, l=_lyr: timeline.append(("paint", l)))
        except Exception as exc:
            print(f"  [warn] no paint event on {_lyr.name}: {exc} — "
                  f"[u] will not undo brush strokes")

    def _cursor_yx() -> tuple[float, float] | None:
        try:
            pos = artery_layer.world_to_data(viewer.cursor.position)
        except Exception:
            return None
        if pos is None or len(pos) < 2:
            return None
        return float(pos[-2]), float(pos[-1])

    def _report(rec: dict | None, miss: str) -> None:
        artery_layer.refresh()
        vein_layer.refresh()
        if rec is None:
            _refresh_overlay(miss)
            return
        if rec["kind"] == "global":
            msg = "global artery ⇄ vein   [u] undo"
        else:
            msg = (f"{rec['kind']} swap: {rec['n']}px "
                   f"{rec['src']} → {rec['dst']}   [u] undo")
        print(f"  [swap] {msg}")
        _refresh_overlay(msg)

    def _do_swap(mode: str) -> None:
        yx = _cursor_yx()
        if yx is None:
            return
        radius = None
        if mode == "disk":
            active = viewer.layers.selection.active
            radius = max(4, int(getattr(active, "brush_size", 2)) * 3)
        rec = swapper.swap_at(yx, mode=mode, radius=radius)
        if rec is not None:
            timeline.append(("edit", None))
        _report(rec, "no vessel under cursor")

    # First [h] marks one end of a mislabelled stretch, second [h] the other.
    range_anchor: list[tuple[float, float]] = []

    def _do_range() -> None:
        yx = _cursor_yx()
        if yx is None:
            return
        if not range_anchor:
            if swapper.owner_at(yx) is None:
                _refresh_overlay("no vessel under cursor")
                return
            range_anchor.append(yx)
            _refresh_overlay("range start set — [h] again at the other end")
            return
        start = range_anchor.pop()
        rec = swapper.swap_between(start, yx)
        if rec is None:
            _refresh_overlay("both ends must be on the same vessel — range cleared")
            return
        timeline.append(("edit", None))
        _report(rec, "")

    def _do_resize(delta: int) -> None:
        yx = _cursor_yx()
        if yx is None:
            return
        rec = swapper.resize_at(yx, delta)
        if rec is None:
            _refresh_overlay("no vessel under cursor")
            return
        timeline.append(("edit", None))
        artery_layer.refresh()
        vein_layer.refresh()
        verb = "widened" if delta > 0 else "narrowed"
        msg = f"{verb} {rec['layer']} by {abs(delta)}px ({rec['n']}px)   [u] undo"
        print(f"  [width] {msg}")
        _refresh_overlay(msg)

    def _make_handler(mode: str):
        if mode in ("widen", "narrow"):
            delta = 1 if mode == "widen" else -1

            def _handler(_ctx):
                """Grow or shrink the vessel segment under the cursor."""
                _do_resize(delta)
            return _handler
        if mode == "range":
            def _handler(_ctx):
                """Swap only the stretch between two [h] marks."""
                _do_range()
            return _handler
        if mode == "undo":
            def _handler(_ctx):
                """Undo the last edit, whether a brush stroke or a swap."""
                if not timeline:
                    _refresh_overlay("nothing to undo")
                    return
                kind, layer = timeline.pop()
                if kind == "paint":
                    try:
                        layer.undo()
                        layer.refresh()
                    except Exception as exc:
                        _refresh_overlay(f"could not undo stroke: {exc}")
                        return
                    _refresh_overlay(f"undid brush stroke on {layer.name}")
                    return
                rec = swapper.undo()
                if rec is None:
                    _refresh_overlay("nothing to undo")
                    return
                artery_layer.refresh()
                vein_layer.refresh()
                _refresh_overlay(f"undid {rec['kind']} ({rec['n']}px)")
            return _handler
        if mode == "global":
            def _handler(_ctx):
                """Swap the artery and vein layers wholesale."""
                timeline.append(("edit", None))
                _report(swapper.swap_all(), "")
            return _handler

        def _handler(_ctx):
            """Move the vessel under the cursor to the other class."""
            _do_swap(mode)
        return _handler

    bind_targets = [viewer, artery_layer, vein_layer]
    for key, mode in SWAP_KEYS:
        bind_key_everywhere(bind_targets, key, _make_handler(mode))

    # Alt+click does the same as [w], but only in pan/zoom mode — in paint
    # mode the click would also lay down a brush stroke.
    @viewer.mouse_drag_callbacks.append
    def _alt_click_swap(_viewer, event):
        if "Alt" not in getattr(event, "modifiers", ()):
            return
        active = viewer.layers.selection.active
        if getattr(active, "mode", "pan_zoom") == "paint":
            _refresh_overlay("alt+click needs pan/zoom mode — use [w] instead")
            return
        _do_swap("branch")

    @viewer.bind_key("q", overwrite=True)
    def _save_quit(_viewer):
        if not ask_confirm(viewer, f"Save annotations for {image_path.name}?"):
            print("  [q] save cancelled — session continues")
            return
        state["should_save"] = True
        viewer.close()

    @viewer.bind_key("s", overwrite=True)
    def _skip(_viewer):
        if not ask_confirm(
                viewer, f"Skip {image_path.name} without saving?"):
            print("  [s] skip cancelled — session continues")
            return
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

    encode = binarise_mask if boundaries else skeletonise_mask
    artery_skel = encode(artery_mask)
    vein_skel = encode(vein_mask)
    save_skeleton_png(artery_skel, art_path)
    save_skeleton_png(vein_skel, vein_path)

    art_valid = validate_saved_skeleton(art_path, (h, w), check_thin=not boundaries)
    vein_valid = validate_saved_skeleton(vein_path, (h, w), check_thin=not boundaries)

    art_pixels = int((artery_skel > 0).sum())
    vein_pixels = int((vein_skel > 0).sum())

    print("")
    unit = "mask pixels" if boundaries else "skeleton pixels"
    print(f"Saved: {art_path}  ({art_pixels} {unit})")
    for m in art_valid.messages:
        print(f"  [art warn] {m}")
    print(f"Saved: {vein_path}  ({vein_pixels} {unit})")
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

    art_edits = skeleton_edit_distance(seed_artery_skel, artery_skel)
    vein_edits = skeleton_edit_distance(seed_vein_skel, vein_skel)
    if prefill_source != "none":
        print(f"Edit-distance   artery: kept={art_edits['kept_px']} "
              f"added={art_edits['added_px']} removed={art_edits['removed_px']} "
              f"IoU={art_edits['iou']}")
        print(f"Edit-distance   vein:   kept={vein_edits['kept_px']} "
              f"added={vein_edits['added_px']} removed={vein_edits['removed_px']} "
              f"IoU={vein_edits['iou']}")
    append_edit_log(
        output_dir / EDITS_CSV_NAME,
        {
            "timestamp": now_iso(),
            "image_filename": image_path.name,
            "duration_seconds": round(duration, 2),
            "prefill_source": prefill_source,
            "lunet_thresh": "" if lunet_thresh is None else lunet_thresh,
            **{f"artery_{k}": v for k, v in art_edits.items()},
            **{f"vein_{k}": v for k, v in vein_edits.items()},
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


def _compute_prefill(
    image_path: Path,
    prefill_source: str,
    lunet_model: Path,
    lunet_thresh: float,
    lunet_cache_dir: Path,
    predictions_dir: Path | None = None,
    masks_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Dispatch to the requested prefill backend. Returns None if disabled."""
    if prefill_source == "none":
        return None
    if prefill_source == "masks":
        if masks_dir is None:
            raise ValueError("--prefill masks requires --masks-dir")
        from PIL import Image as _Image  # noqa: PLC0415
        shape = np.asarray(_Image.open(str(image_path))).shape[:2]
        found = load_mask_prefill(masks_dir, image_path.stem, shape)
        if found is None:
            print(f"  [masks] no artery/vein mask for {image_path.stem} in "
                  f"{masks_dir} — continuing with empty prefill")
            return None
        art, vein = found
        print(f"  [masks] {image_path.stem}: artery={int(art.sum())}px "
              f"vein={int(vein.sum())}px (filled masks, width preserved)")
        return art, vein
    if prefill_source == "lunet":
        if not lunet_model.exists():
            raise FileNotFoundError(
                f"LUNet model not found: {lunet_model}. "
                f"Pass --lunet-model or place the ONNX file there."
            )
        t0 = time.time()
        print(f"  [lunet] running inference (cache: {lunet_cache_dir})...")
        masks = lunet_prefill_masks(
            image_path, lunet_model, thresh=lunet_thresh,
            cache_dir=lunet_cache_dir,
        )
        print(f"  [lunet] done in {human_duration(time.time() - t0)}")
        return masks
    if prefill_source == "predictions":
        if predictions_dir is None:
            raise ValueError(
                "--prefill predictions requires --predictions-dir")
        # Look for <stem>_hard.png alongside the input image's stem.
        hard_path = predictions_dir / f"{image_path.stem}_hard.png"
        if not hard_path.exists():
            print(f"  [predictions] no prediction at {hard_path} — "
                  f"continuing with empty prefill")
            return None
        import cv2  # noqa: PLC0415
        hard = cv2.imread(str(hard_path), cv2.IMREAD_UNCHANGED)
        if hard is None:
            print(f"  [predictions] failed to read {hard_path} — empty prefill")
            return None
        # hard label encoding: 0=bg, 1=artery, 2=vein, 3=A∩V crossing.
        # Crossings count as both classes (the model believes both vessels
        # pass through that pixel — the annotator decides whether to keep it).
        artery = ((hard == 1) | (hard == 3)).astype(np.uint8)
        vein = ((hard == 2) | (hard == 3)).astype(np.uint8)
        n_a = int(artery.sum())
        n_v = int(vein.sum())
        print(f"  [predictions] {hard_path.name}: "
              f"artery={n_a}px vein={n_v}px (hard label split)")
        return artery, vein
    raise ValueError(f"unknown prefill source: {prefill_source}")


def _walk_directory(
    src_dir: Path,
    output_dir: Path,
    overwrite: bool,
    prefill_source: str = "none",
    lunet_model: Path = DEFAULT_LUNET_MODEL,
    lunet_thresh: float = 0.5,
    lunet_cache_dir: Path | None = None,
    predictions_dir: Path | None = None,
    masks_dir: Path | None = None,
    boundaries: bool = True,
) -> None:
    images = list_images(src_dir)
    if not images:
        print(f"No images found in {src_dir}")
        return
    print(f"Found {len(images)} images in {src_dir}.")
    if prefill_source == "lunet":
        print(f"Prefill: lunet  (model: {lunet_model}, thresh={lunet_thresh})")
    elif prefill_source == "predictions":
        print(f"Prefill: predictions  (dir: {predictions_dir})")
    elif prefill_source == "masks":
        print(f"Prefill: masks  (dir: {masks_dir})")
    completed = 0

    def _sigint(_signum, _frame):
        print(f"\n[ctrl-c] completed {completed}/{len(images)} this session. exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    cache_dir = lunet_cache_dir or (output_dir / LUNET_CACHE_DIRNAME)

    for i, img in enumerate(images):
        stem = img.stem
        if already_annotated(stem, output_dir) and not overwrite:
            print(f"  [{i + 1}/{len(images)}] skip (already annotated): {img.name}")
            continue
        print(f"\n--- [{i + 1}/{len(images)}] {img.name} ---")
        prefill_masks = _compute_prefill(
            img, prefill_source, lunet_model, lunet_thresh, cache_dir,
            predictions_dir=predictions_dir, masks_dir=masks_dir,
        )
        ok = _open_annotation_session(
            img, output_dir, overwrite=overwrite,
            prefill_source=prefill_source,
            prefill_masks=prefill_masks,
            lunet_thresh=lunet_thresh if prefill_source == "lunet" else None,
            boundaries=boundaries,
        )
        if ok:
            completed += 1
        if i + 1 < len(images):
            input("press Enter for next image, or Ctrl+C to quit... ")
    print(f"\nSession complete: {completed}/{len(images)} images annotated.")


# ---- CLI --------------------------------------------------------------


def main() -> int:
    # Handle `preview` as a manual subcommand so the default `path` positional
    # doesn't conflict with it in argparse.
    if len(sys.argv) >= 2 and sys.argv[1] == "preview":
        p_prev = argparse.ArgumentParser(prog="annotate.py preview")
        p_prev.add_argument("image", type=Path)
        p_prev.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                            help=f"Where masks live (default {OUTPUT_DIR}).")
        prev_args = p_prev.parse_args(sys.argv[2:])
        if not prev_args.image.exists():
            print(f"image not found: {prev_args.image}")
            return 1
        _open_preview(prev_args.image, prev_args.output_dir)
        return 0

    parser = argparse.ArgumentParser(
        prog="annotate.py",
        description="Napari-based retinal vessel annotation tool (UWF and "
                    "DVA). Default output is filled masks that keep the "
                    "painted vessel width; --skeleton saves 1-px centrelines "
                    "instead. Use `annotate.py preview <image>` for a "
                    "read-only view.",
    )
    parser.add_argument("path", nargs="?", type=Path,
                        help="Image file or directory of images to annotate.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where skeleton PNGs are written (default {OUTPUT_DIR}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing skeleton files without asking.")
    parser.add_argument("--skeleton", action="store_true",
                        help="Save 1-px skeleton centrelines instead of the "
                             "default filled masks. Use when only topology "
                             "matters and width is re-derived downstream by "
                             "profile fitting.")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="Directory of filled artery/vein masks for "
                             "--prefill masks. Accepts <dir>/artery/<stem>.png "
                             "or <dir>/<stem>_artery.png.")
    parser.add_argument("--prefill",
                        choices=["none", "lunet", "predictions", "masks", "auto"],
                        default="none",
                        help="Seed the paint layers. 'lunet' runs the original "
                             "LUNet ONNX A/V segmentation. 'predictions' loads "
                             "<stem>_hard.png from --predictions-dir (use this "
                             "with the finetuned v3/v4/v5 outputs). 'masks' "
                             "loads filled A/V masks from --masks-dir, "
                             "keeping their width. 'auto' inspects the "
                             "directory and picks: filled masks (DVA, pixel "
                             "output) or <stem>_hard.png (UWF, skeleton "
                             "output).")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Directory of <stem>_hard.png files from a "
                             "fine-tuned model (used when --prefill predictions).")
    parser.add_argument("--lunet-model", type=Path, default=DEFAULT_LUNET_MODEL,
                        help=f"LUNet ONNX path (default {DEFAULT_LUNET_MODEL}).")
    parser.add_argument("--lunet-thresh", type=float, default=0.5,
                        help="Probability threshold for LUNet → binary (default 0.5).")
    parser.add_argument("--lunet-cache-dir", type=Path, default=None,
                        help=f"Cache dir for LUNet probs (default "
                             f"<output-dir>/{LUNET_CACHE_DIRNAME}).")

    args = parser.parse_args()

    if args.path is None:
        parser.print_help()
        return 1
    if not args.path.exists():
        print(f"path not found: {args.path}")
        return 1

    cache_dir = args.lunet_cache_dir or (args.output_dir / LUNET_CACHE_DIRNAME)

    if args.prefill == "predictions" and args.predictions_dir is None:
        parser.error("--prefill predictions requires --predictions-dir")
    if args.prefill == "masks" and args.masks_dir is None:
        parser.error("--prefill masks requires --masks-dir")

    # `auto` lets the batch declare where its prefill comes from, so one
    # launcher serves both datasets. It never changes the OUTPUT: pixels
    # are always what gets saved, and a skeleton can be derived from them
    # afterwards. Only an explicit --skeleton thins the output.
    boundaries = not args.skeleton
    if args.prefill == "auto":
        probe = args.masks_dir or args.predictions_dir
        if probe is None:
            parser.error("--prefill auto requires --masks-dir or --predictions-dir")
        kind = detect_prefill_kind(probe)
        args.prefill = kind
        if kind == "masks":
            args.masks_dir = probe
        elif kind == "predictions":
            args.predictions_dir = probe
        print(f"Prefill (auto):  {kind} from {probe}")
        print(f"Output (auto):   "
              f"{'filled masks — vessel width kept' if boundaries else '1-px skeletons'}")

    if args.path.is_dir():
        _walk_directory(
            args.path, args.output_dir, overwrite=args.overwrite,
            prefill_source=args.prefill,
            lunet_model=args.lunet_model,
            lunet_thresh=args.lunet_thresh,
            lunet_cache_dir=cache_dir,
            predictions_dir=args.predictions_dir,
            masks_dir=args.masks_dir,
            boundaries=boundaries,
        )
    else:
        prefill_masks = _compute_prefill(
            args.path, args.prefill, args.lunet_model, args.lunet_thresh, cache_dir,
            predictions_dir=args.predictions_dir, masks_dir=args.masks_dir,
        )
        _open_annotation_session(
            args.path, args.output_dir, overwrite=args.overwrite,
            prefill_source=args.prefill,
            prefill_masks=prefill_masks,
            lunet_thresh=args.lunet_thresh if args.prefill == "lunet" else None,
            boundaries=boundaries,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
