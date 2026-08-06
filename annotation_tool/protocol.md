# UWF Skeleton Annotation Protocol

**Goal**: mark every retinal vessel you can confidently identify, labelled by
class (artery vs vein). By default you paint the **vessel body**, and the
thickness you paint is saved as the vessel width (section 9c). With
`--skeleton` you instead draw 1-px **centrelines** only, and the downstream
physics reconstruction re-derives width from raw image intensity — there
your job is topology, not width.

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

**Target: 1 minute per image.** That is the working assumption for the
prefill workflow — the model has already traced the vessels, so the job is
to correct it, and the A/V swap keys in section 9b are what make a minute
realistic. If an image is taking much longer, it is usually a sign the
prediction is bad enough to skip (`[s]`) rather than repair.

For reference, tracing from a blank layer with no prefill is a different
job entirely: 30–45 minutes for UWF at ~4000×4000, 15–25 minutes for a
standard fundus at ~1444×1444.

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

## 9b. Correcting artery/vein confusions

A/V confusion is the dominant prefill error: the vessel is traced correctly
but assigned to the wrong class. Don't erase and repaint — hover the cursor
over the offending vessel and press a key:

- `[w]` moves the **segment** you are pointing at to the other class. The
  vessel is cut at its branch points first, so a single mislabelled
  daughter branch moves without dragging its parent trunk with it.
- `[W]` moves the **entire connected vessel** — use it when a whole tree
  came out inverted.
- `[h]` twice moves **only the stretch between two marks**. Press it at one
  end of the mislabelled piece and again at the other. Use this when the
  class flips part-way along a branch, where `[w]` would over-correct and
  take the correctly-labelled half with it. Both marks must be on the same
  vessel; the path follows the vessel, so curves and junctions are fine.
- `[g]` moves everything inside a brush-sized disk. This is the escape
  hatch for tangles and crossing points where the connectivity is wrong.
- `[X]` swaps the artery and vein layers wholesale.
- `[u]` undoes the last swap (swaps only — it does not undo brush strokes).

At a **crossing**, a pixel may legitimately belong to both classes. Swapping
never deletes it from the destination: pixels are cleared from the source
layer and set on the destination, and `[u]` restores exactly the pixels
that changed. Check the crossing afterwards and repaint the missing class
if the swap took a shared pixel with it.

## 9c. Annotating vessel boundaries (the default)

**By default you are annotating the vessel boundary**, and the thickness of
what you paint is the width that gets saved. Sections 3–8 describe the
older `--skeleton` mode, where you trace centrelines only and the width is
re-derived downstream; the class, junction, crossing and periphery rules
there all still apply.

- Paint the vessel **body**, not its centreline. Set the brush to roughly
  the vessel's caliber with `[` / `]` and paint along it.
- Where a prefilled mask is close but not right, nudge it: `[.]` widens the
  segment under the cursor by 1 px, `[,]` narrows it.
- Judge the boundary at the point where the vessel wall meets the
  background, at native zoom. Do not include the bright central reflex as a
  separate structure — it is inside the vessel.
- `[,]` cannot destroy a vessel: narrowing stops at the centreline.
- `[.]` then `[,]` does **not** return you exactly where you started
  (dilate-then-erode fills small concavities). Press `[u]` to revert a
  width edit exactly.
- Everything in section 9b still applies — A/V swaps work the same way and
  preserve width.

Output is a filled mask, uint8 {0, 255}, in the same format as
`databases/DVA/raw/artery/<stem>.png`.

## 10. Saving

- `[q]` saves both skeleton PNGs to `annotations_uwf/<basename>_*.png` and
  moves to the next image.
- `[s]` skips this image without saving anything.
- Closing the napari window (no keybind) behaves the same as `[q]`.
- Accidentally painted on the wrong class? Hover it and press `[w]` to move
  it across (section 9b), or use the eraser mode in napari (mode button in
  the label-layer controls) before pressing `[q]`.

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
