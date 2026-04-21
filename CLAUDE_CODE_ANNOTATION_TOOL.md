# Claude Code: Napari Annotation Tool for UWF Skeleton Annotation

## Context

We need a lightweight annotation tool for drawing vessel skeletons (separated into artery and vein classes) on ultra-widefield (UWF) retinal images. The tool must:

1. Handle large UWF images (often 3000x3000 or larger) without UI lag
2. Let the annotator paint skeleton centerlines for arteries and veins separately
3. Export mask files in the format our physics reconstruction pipeline expects:
   `<imagename>_artery.png` and `<imagename>_veins.png` in a dedicated output directory
4. Post-process the painted strokes into clean 1-pixel skeletons via morphological thinning
5. Support annotating many images sequentially with minimal friction

The tool will be used by us and by clinical collaborators, so it needs to be simple to run and robust to misuse (accidental closes, wrong file formats, etc.).

## Dependencies

```bash
pip install "napari[all]"
pip install scikit-image
pip install pillow numpy
```

Verify napari works: `napari` from terminal should open an empty window. If Qt errors occur, try `pip install "napari[pyqt5]"`.

## Directory layout

```
annotation_tool/
    annotate.py           # main CLI script
    utils.py              # helpers (loading, saving, skeletonizing)
    protocol.md           # annotation protocol (see step 7)
    annotations_uwf/      # output directory (auto-created)
        <filename>_artery.png
        <filename>_veins.png
```

## Step 1: Build the main annotation script

Create `annotate.py` with the following functionality:

- Command line: `python annotate.py <image_path>` to annotate a single image, or `python annotate.py <directory>` to annotate all images in a directory one by one.
- If the output files already exist for a given image, print a warning and ask (y/n) whether to overwrite or skip. This prevents accidentally clobbering finished annotations.
- For each image:
  1. Load it as a numpy array (handle RGB and RGBA, drop alpha if present)
  2. Build a multiscale pyramid if the image is larger than 2000 pixels on either side. Use `skimage.transform.downscale_local_mean` at factors 1, 2, 4 so napari can display lower resolution versions when zoomed out.
  3. Open napari with the image as the background
  4. Add two label layers named "artery" (color: red / color index 1) and "veins" (color: blue / color index 1)
  5. Set default brush size to 2 pixels
  6. Set the artery layer as initially selected
  7. Display instructions in the viewer title bar: "Paint with [3], toggle layers with Tab, save and quit with [q]"
  8. Run the napari event loop until the user closes the window

After the window closes:
- Take the `.data` attribute of each labels layer, which is at the full image resolution
- Threshold at >0 to get a binary mask
- Skeletonize each mask with `skimage.morphology.skeletonize` to reduce thick brush strokes to 1-pixel skeletons
- Convert to uint8 with values {0, 255}
- Save as `annotations_uwf/<basename>_artery.png` and `annotations_uwf/<basename>_veins.png`
- Log the annotation time (start timer when napari opens, stop when it closes) to `annotations_uwf/annotation_times.csv` with columns: timestamp, image_filename, duration_seconds, artery_pixel_count, vein_pixel_count

## Step 2: Add a sanity check after save

After saving each image's annotations, verify:
- Both PNG files exist and are readable
- Masks have the expected shape (H, W) matching the source image
- Mask dtype is uint8 with unique values subset of {0, 255}
- Skeletons are actually 1-pixel wide (max value of 8-neighbor count should be reasonable)

Print a summary:
```
Saved: annotations_uwf/<basename>_artery.png (N_artery skeleton pixels)
Saved: annotations_uwf/<basename>_veins.png (N_vein skeleton pixels)
Duration: M minutes
```

If anything fails validation, print a warning but don't crash.

## Step 3: Build the directory-walking mode

When given a directory, the script should:

1. Find all PNG/JPG/TIF images in the directory
2. For each image, check if annotations already exist in `annotations_uwf/`
3. Skip already-annotated images by default, or prompt if `--overwrite` flag is set
4. Annotate unfinished images in alphabetical order
5. After each image, pause briefly (press Enter to continue) so the user can rest or stop
6. Graceful exit on Ctrl+C with a summary of how many images were completed in this session

## Step 4: Keyboard shortcuts and UX polish

Inside napari, bind these keyboard shortcuts if possible using napari's `bind_key` decorator:
- `q`: save and quit the current image (same effect as closing the window)
- `s`: skip this image (close without saving)
- `Tab`: cycle between artery and veins layers
- `1`: activate pan/zoom (default napari)
- `3`: activate paint tool
- `[` and `]`: decrease/increase brush size

Display a text overlay with the current layer name and brush size, so annotators can see at a glance which class they're painting.

## Step 5: Handle large UWF images

For images larger than 2000 pixels on either dimension:

```python
from skimage.transform import downscale_local_mean

image_full = np.array(Image.open(image_path).convert('RGB'))
h, w = image_full.shape[:2]

if max(h, w) > 2000:
    image_half = downscale_local_mean(image_full, (2, 2, 1)).astype(np.uint8)
    image_quarter = downscale_local_mean(image_full, (4, 4, 1)).astype(np.uint8)
    viewer.add_image(
        [image_full, image_half, image_quarter],
        multiscale=True,
        name='fundus'
    )
else:
    viewer.add_image(image_full, name='fundus')
```

Label layers must be at the full resolution regardless:

```python
artery_mask = np.zeros((h, w), dtype=np.uint8)
vein_mask = np.zeros((h, w), dtype=np.uint8)
```

This way painting happens at full resolution but display is multiscale for speed.

## Step 6: Add a quick preview function

Add a `preview` command that loads an image and its saved annotations to visually verify quality:

```bash
python annotate.py preview <image_path>
```

This opens napari with the image and the saved skeleton masks overlaid (as labels layers, read-only). Useful for sanity-checking finished annotations before the physics reconstruction step.

## Step 7: Write the annotation protocol

Create `protocol.md` documenting the annotation guidelines. This will be shared with clinical collaborators. Include:

- What counts as a vessel to annotate (minimum caliber rule — e.g., "annotate all vessels confidently visible at native zoom")
- Color assignments (red = artery, blue = vein)
- How to handle bifurcations (draw through, not around)
- How to handle crossings (A/V crossing points — continue each vessel through the crossing)
- What to do with ambiguous vessels (when A/V is unclear, skip rather than guess)
- Peripheral extent rule (how far into the periphery to annotate)
- How to handle artifacts (eyelashes, reflections, illumination falloff — skip obscured vessels)
- Expected time per image (30-45 minutes for UWF)
- Saving convention and file naming

Keep this document to 1-2 pages. Clarity over completeness.

## Step 8: Add a timing sanity check

On startup, before opening napari, print:

```
=== UWF Annotation Session ===
Image: <filename>
Full resolution: <HxW>
Previously annotated: [Yes/No]
Start time: <timestamp>

Tips:
- Paint with [3], use scroll to zoom, space+drag to pan
- Press [q] to save and move to the next image
- Press [s] to skip without saving
- Target time per image: 30-45 minutes
```

When annotation completes, print:

```
=== Complete ===
Duration: <M>:<SS>
Artery pixels: <N>
Vein pixels: <N>
Saved to: annotations_uwf/<basename>_*.png
```

## Step 9: Test on a dummy image before real UWF

Before trusting the tool on real data, create a synthetic test image:

```python
# tests/test_tool.py
import numpy as np
from PIL import Image
synth = np.zeros((2048, 2048, 3), dtype=np.uint8) + 100
Image.fromarray(synth).save('test_image.png')
```

Run `python annotate.py test_image.png`. Confirm that:
- Napari opens with the gray image
- Two label layers are present
- Painting works with the paint tool
- Window closes correctly
- Two PNG masks are saved in `annotations_uwf/`
- Masks have correct shape and dtype
- Skeletonization produces 1-pixel-wide results

## Step 10: Integration with the physics reconstruction pipeline

After Claude Code builds the tool, add a one-line helper so that a finished UWF annotation can be immediately fed into the existing physics reconstruction:

```bash
python experiments/skeleton_reconstruction/reconstruct_v2.py \
    --image <image_path> \
    --artery-skeleton annotations_uwf/<basename>_artery.png \
    --vein-skeleton annotations_uwf/<basename>_veins.png \
    --output-dir databases/UWF_physics/
```

This closes the loop: annotate -> reconstruct -> get physics soft labels -> use for training.

## Important constraints

- **Do not modify the physics reconstruction pipeline.** This tool only produces skeleton PNGs; reconstruction is separate.
- **Output files must match the format already used by the pipeline.** Binary 0/255 uint8 PNGs, one per class, filename convention `<basename>_artery.png` and `<basename>_veins.png`.
- **Handle large UWF images gracefully.** Never load full-resolution into display memory if it would lag — use multiscale.
- **Save on clean exit only.** If napari crashes, the script should warn the user that annotations were lost. Consider adding a periodic autosave every 5 minutes for long annotation sessions (optional enhancement).
- **Do not require a specific Python version or OS.** Napari works on Windows, Mac, Linux. The script should too.

## Deliverable

When Claude Code is done, we should be able to:
1. Run `python annotate.py <UWF_image.png>` on a fresh machine (with dependencies installed)
2. See napari open with the image loaded and annotation layers ready
3. Paint skeleton lines for arteries and veins
4. Close napari and find correctly-formatted PNG masks in `annotations_uwf/`
5. Use those masks directly as input to the existing v2 physics reconstruction script
