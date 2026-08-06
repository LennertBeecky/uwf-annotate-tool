# Retinal vessel annotation tool (UWF + DVA)

Napari-based tool for painting artery and vein vessels on UWF and DVA
retinal images. By default the painted mask is saved as-is, so its
thickness is the annotated vessel width; `--skeleton` saves 1-px
centrelines instead. Output PNGs feed the `reconstruct_v2.py` physics
pipeline.

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

### Fixing artery/vein confusions

Hover the cursor over the mislabelled vessel and press one key — the pixels
move from whichever layer currently owns them to the other one. No mode
switch, no erase-and-repaint, and it works from paint or pan/zoom mode.

| Key | Action |
|-----|--------|
| `w` | swap the vessel **segment** under the cursor (cut at junctions) |
| `W` | swap the whole **connected vessel** under the cursor |
| `h` `h` | swap only the **stretch between two marks** (press at each end) |
| `g` | swap everything in a brush-sized disk (tangles, crossings) |
| `X` | swap the artery and vein layers wholesale (inverted prediction) |
| `u` | undo the last swap, or the last width edit |
| `alt`+click | same as `w`; only in pan/zoom mode, since in paint mode the click would also draw |

`w` is the everyday one: it skeletonises the vessel you are pointing at,
cuts it at its branch points, and moves just that branch — so relabelling
one mis-assigned daughter branch does not drag its parent along.

When the model flips class only **part-way along** a branch, `w` would
over-correct. Use `h` instead: press it at one end of the bad stretch and
again at the other, and only the vessel between the two marks moves. The
two marks must be on the same vessel — the tool follows the vessel from one
to the other rather than drawing a straight line, so it tracks curves and
passes through junctions.

See `protocol.md` for the full annotation protocol (what to paint, how to
handle junctions / crossings / peripheries).

## Tests

```bash
pytest annotation_tool/tests/test_tool.py -v
```

13 unit tests covering loaders, multiscale pyramid, skeletonisation,
validation, CSV logging. No napari/Qt dependency for the tests.

## Output: pixel annotations (default) vs skeletons

**By default the painted mask is saved un-thinned** — the pixels you paint
are the annotation, and their thickness is the vessel width.

```bash
# DVA: start from the filled A/V masks and keep their width
python annotate.py databases/DVA/raw/images/ \
    --prefill masks --masks-dir databases/DVA/raw \
    --output-dir annotations_dva
```

Pass `--skeleton` for the older behaviour, where the mask is thinned to a
1-px centreline on save and width is re-derived downstream by profile
fitting. Nothing downstream requires that: `reconstruct_from_gt()`
skeletonises whatever mask it is handed, and separately uses the filled
mask for painting — so filled masks are what it was designed for.

Output is `<stem>_artery.png` / `<stem>_veins.png`, uint8 {0, 255}, filled —
the same format as the existing ground truth in `databases/DVA/raw/artery/`,
so anything that reads those reads these. The centreline and the width are
both recoverable afterwards (`skeletonize(mask)`, and
`2 * distance_transform_edt(mask)` on the centreline).

`--prefill masks` seeds the layers from filled A/V masks, so you correct a
model's boundaries instead of drawing them. It accepts either
`<dir>/artery/<stem>.png` (the DVA layout) or `<dir>/<stem>_artery.png`
(this tool's own output, so you can reopen a saved annotation). Masks stored
in the 720×720 letterboxed frame are centre-cropped to match a 576×720
image automatically.

### Setting the width

| Key | Action |
|-----|--------|
| `.` | widen the vessel segment under the cursor by 1 px |
| `,` | narrow it by 1 px |
| `[` / `]` | brush size — paint the vessel body at its true caliber |
| `u` | undo the last width edit |

Two ways to work, and you'll use both: paint at true caliber with a brush
sized to the vessel (the default brush is 5 px in this mode rather than 2),
and nudge an existing boundary with `.` / `,` when a prefilled mask is
nearly right. Each press moves the boundary about 1 px per side, so the
measured width changes by roughly 2–2.6 px depending on the vessel's angle.

Two things to know:

- **Narrowing can never destroy a vessel.** It stops at the centreline, so
  a vessel can be reduced to 1 px and no further, and it cannot be broken
  into two pieces.
- **`,` does not exactly undo `.`** Dilation followed by erosion is a
  morphological closing, not an inverse: small concavities filled by the
  widen do not come back. Use `u` to revert a width edit exactly.
