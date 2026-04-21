# UWF skeleton annotation tool

Napari-based tool for painting 1-pixel-wide vessel skeletons (artery and vein
classes) on UWF retinal images. Output PNGs feed the `reconstruct_v2.py`
physics pipeline that re-derives vessel width from raw image intensity.

## Install

```bash
pip install "napari[all]" scikit-image pillow numpy scipy
```

If Qt fails on your platform, try: `pip install "napari[pyqt5]"`.

## Usage

```bash
# single image
python annotation_tool/annotate.py path/to/uwf.png

# walk a directory in alphabetical order, skip already-annotated
python annotation_tool/annotate.py path/to/uwf_folder/

# overwrite even if files already exist
python annotation_tool/annotate.py path/to/uwf_folder/ --overwrite

# read-only preview of saved skeletons
python annotation_tool/annotate.py preview path/to/uwf.png
```

Outputs land in `annotations_uwf/` (override with `--output-dir`):

```
annotations_uwf/
    <basename>_artery.png        # uint8 {0, 255}, 1-px skeleton
    <basename>_veins.png         # uint8 {0, 255}, 1-px skeleton
    annotation_times.csv          # per-image duration log
```

## Keybinds

| Key | Action |
|-----|--------|
| `3` | paint |
| `1` | pan / zoom |
| `[` / `]` | decrease / increase brush size |
| `Tab` | cycle between artery and veins layers |
| `q` | save and advance to next image |
| `s` | skip without saving |

See `protocol.md` for the full annotation protocol (what to paint, how to
handle junctions / crossings / peripheries).

## Tests

```bash
pytest annotation_tool/tests/test_tool.py -v
```

13 unit tests covering loaders, multiscale pyramid, skeletonisation,
validation, CSV logging. No napari/Qt dependency for the tests.
