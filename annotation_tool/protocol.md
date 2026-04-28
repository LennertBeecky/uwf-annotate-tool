# UWF Skeleton Annotation Protocol

**Goal**: draw 1-pixel-wide **skeleton centrelines** for every retinal vessel you
can confidently identify, labelled by class (artery vs vein). These skeletons
feed the downstream physics reconstruction, which re-derives vessel width
from the raw image intensity — so your job is **topology**, not width.

## 1. Colours

| Class | Colour in tool | Filename |
|-------|----------------|----------|
| Artery | **red** | `<basename>_artery.png` |
| Vein   | **blue** | `<basename>_veins.png` |

Never mix classes in a single layer. If unsure about a vessel's class → skip
that vessel (see §7).

## 2. Which vessels to annotate

- Annotate **every vessel confidently visible at native zoom** — no minimum
  caliber threshold, but err on the side of not annotating a vessel you are
  not sure about.
- **Zoom in before judging**: a vessel that looks like noise at fit-to-window
  may be real at 200% zoom. Trust what's visible at native resolution with
  clear tubular structure and A/V contrast.
- **Stop** at the vessel tip when the vessel fades into background noise, not
  at an arbitrary distance.

## 3. Brush size + painting

- Default brush **2 px**. Bump up (`]`) for thick central arcades, drop down
  (`[`) for capillaries. Final thickness doesn't matter — morphological
  thinning reduces to 1 px on save.
- Paint **along** the vessel, not across. One continuous stroke per vessel
  branch is ideal; many short dabs are fine too.
- Re-paint over an already-painted region to correct a mistake; the paint
  tool is additive.

## 4. Bifurcations (Y-splits)

- **Draw through**, not around. The parent trunk's skeleton continues into
  each daughter branch without a gap at the branch point.
- A small fork shape at the junction is fine — skeletonization will clean up
  thick junction blobs.

## 5. A/V crossings (one vessel passes over another)

- **Continue each vessel through the crossing.** The artery and vein
  skeletons will overlap at the crossing pixel, but that's expected — the
  downstream pipeline detects overlaps and handles them.
- Don't try to separate the two spatially at the crossing. Paint both as if
  the other vessel weren't there.

## 6. Peripheral extent

- Annotate out **as far as you can reliably identify the class**.
- At the far periphery (>3 disc diameters), vessel class often becomes
  ambiguous — if you can't tell A from V for a whole peripheral branch,
  **skip it** rather than guess.

## 7. Ambiguous / obscured vessels

Skip (don't annotate):
- Vessels obscured by eyelashes, reflections, strong illumination falloff
- Vessels in areas of severe pathology (hemorrhage, exudate covering the
  vessel outline)
- Vessels where A/V class is not distinguishable
- Capillary beds — focus on macroscopic vessels

It is **better to skip a real vessel than to paint it with the wrong class**.
Missed vessels are a recall problem the downstream model can tolerate;
wrong-class vessels systematically corrupt arterial/venular statistics.

## 8. Artefacts

- Reflex / bright central stripe along an artery: paint the skeleton down the
  middle of the reflex, ignoring the stripe's internal structure.
- Image borders or blacked-out peripheries: stop painting before you reach
  them; don't extend skeletons into no-data regions.

## 9. Expected time per image

- **UWF, ~4000×4000**: 30–45 minutes from scratch
- **UWF with `--prefill lunet`**: target much lower — you correct LUNet
  instead of tracing. Actual speedup depends on LUNet quality on the image.
- **Standard fundus, ~1444×1444**: 15–25 minutes

The `annotation_times.csv` log records per-image duration — the mean across
your session is useful for scheduling.

### LUNet prefill workflow

```bash
python annotate.py <uwf_folder>/ --prefill lunet
```

What happens per image:

1. LUNet A/V segmentation runs on the native-resolution image (tiled,
   50% overlap, green-channel input). Results are cached under
   `<output-dir>/_lunet_cache/<stem>_probs.npz` so re-opens skip inference.
2. Probabilities are thresholded at `--lunet-thresh` (default 0.5) to
   produce binary artery/vein masks. These populate the paint layers as
   your starting point.
3. You erase false positives and paint in missed vessels as usual. On `[q]`
   the final mask is skeletonised and saved exactly as in the scratch flow.
4. A per-image row is appended to `annotation_edits.csv` with
   `seed_px / final_px / kept_px / added_px / removed_px / iou` for each
   class — audit trail for how much of LUNet's output survived into GT.

Bias guard: if you worry that LUNet errors are leaking into your GT, skim
the `annotation_edits.csv` IoU column. Low IoU = you corrected heavily; IoU
near 1 on every image with non-trivial seed counts = you may have rubber-
stamped LUNet. Sample a few of the high-IoU cases in `preview` mode.

## 10. Saving

- `[q]` saves both skeleton PNGs to `annotations_uwf/<basename>_*.png` and
  moves to the next image.
- `[s]` skips this image without saving anything.
- Closing the napari window (no keybind) behaves the same as `[q]`.
- Accidentally painted on the wrong class? Use the eraser mode in napari
  (mode button in the label-layer controls) before pressing `[q]`.

## 11. File naming

The output filenames are derived automatically from the input image
filename — **don't rename them afterwards**. The downstream physics
pipeline matches by stem:

```
input:  myimage.png
output: annotations_uwf/myimage_artery.png
        annotations_uwf/myimage_veins.png
```

## 12. Quality checks (automatic)

After each save, the tool prints:

```
Saved: annotations_uwf/<basename>_artery.png (N_artery skeleton pixels)
Saved: annotations_uwf/<basename>_veins.png (N_vein skeleton pixels)
Duration: M:SS
```

A warning appears if:
- The saved PNG has the wrong shape or dtype
- The "skeleton" is not actually 1 pixel wide (a blob survived thinning)

If you see a warning, re-open the image with `preview` mode to inspect:

```bash
python annotate.py preview <image_path>
```

## 13. Using the saved annotation

Once you have `annotations_uwf/<basename>_artery.png` and `..._veins.png`,
the downstream reconstruction reads them directly:

```bash
python experiments/skeleton_reconstruction/reconstruct_v2.py \
    --image <image_path> \
    --artery-skeleton annotations_uwf/<basename>_artery.png \
    --vein-skeleton   annotations_uwf/<basename>_veins.png \
    --output-dir databases/UWF_physics/
```

No further annotator action needed — the soft physics mask is generated
automatically.
