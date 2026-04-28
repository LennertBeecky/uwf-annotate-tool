# UWF Vessel Annotation — Clinician Setup

This is the annotation workflow for hand-tracing artery and vein skeletons on
ultra-widefield (UWF) retinal images. The tool opens each image in a napari
viewer with the model's prediction pre-filled as a starting point, you correct
it (erase wrong vessels, add missed ones, fix A/V class), and save.

## What you'll need

- A computer with macOS or Linux (Windows works too via WSL)
- About 3 GB free disk space
- A folder of UWF images (`.jpeg`, `.jpg`, or `.png`)
- The pre-computed model predictions (provided separately — see below)

## 1. One-time setup

### 1a. Install miniconda (if you don't already have it)

Download the installer for your OS from
<https://docs.conda.io/en/latest/miniconda.html> and follow the installer.

### 1b. Clone the repository and switch to this branch

```bash
git clone <repo-url> uwf-annotate
cd uwf-annotate
git checkout clinician_workflow
```

### 1c. Create the conda environment

```bash
conda env create -f environment_clinician.yml
conda activate uwf-annotate
```

This installs napari, opencv, scikit-image, and a few other dependencies.
Takes ~5 minutes.

### 1d. Verify the install

```bash
python -c "import napari, cv2, numpy, skimage, PIL; print('OK')"
```

You should see `OK`. If not, contact Lennert.

## 2. Per-batch setup

### 2a. Receive the image batch + predictions

Lennert will give you two folders for each batch:

- `clinician_data/images_to_annotate/<batch_name>/` — the raw UWF images
- `clinician_data/predictions/<batch_name>/` — pre-computed predictions
  (`<stem>_hard.png` files)

Drop them into the matching `clinician_data/` subdirectories in your local
clone. The `clinician_data/annotations/` folder is where your output goes —
create a subfolder for each batch:

```bash
mkdir -p clinician_data/annotations/<batch_name>
```

### 2b. Open the protocol

Read `annotation_tool/protocol.md` once before your first batch. It covers the
colour conventions, what to annotate, how to handle bifurcations and
crossings, and what to skip.

## 3. Annotating

### 3a. Launch the tool with model prefill

```bash
python annotation_tool/annotate.py \
    clinician_data/images_to_annotate/<batch_name>/ \
    --output-dir clinician_data/annotations/<batch_name>/ \
    --prefill predictions \
    --predictions-dir clinician_data/predictions/<batch_name>/
```

This walks through every image in the batch directory, opens napari with the
artery + vein layers pre-filled from the prediction, and waits for you to
correct it.

### 3b. In the napari viewer

| Action | Keyboard |
|---|---|
| Toggle artery / vein layer | `1` (artery) / `2` (vein) — see top-left |
| Paint mode | `B` |
| Eraser mode | `E` |
| Pan mode | `Space` (hold) |
| Bigger / smaller brush | `]` / `[` |
| Save and move to next image | `S` |
| Skip this image (no save) | `Q` |

Exact bindings appear in the napari viewer header — check there if a key
doesn't do what you expect.

The tool saves two files per image when you press `S`:

- `<stem>_artery.png` (red layer, binarised, 1-px-thick skeleton)
- `<stem>_veins.png` (blue layer, binarised, 1-px-thick skeleton)

### 3c. Pacing

For calibration purposes, please do **one out of every ~15 images from
scratch** — i.e. without prefill. Launch with `--prefill none` for those:

```bash
python annotation_tool/annotate.py \
    clinician_data/images_to_annotate/<batch_name>/<one_image.jpeg> \
    --output-dir clinician_data/annotations/<batch_name>/ \
    --prefill none
```

This lets us measure how much the prefill biases your annotations later. Note
which images you did from scratch (a simple text file is fine).

## 4. Sending annotations back

Once you've finished a batch, zip the output and send it back:

```bash
cd clinician_data/annotations/
tar czf <batch_name>_annotations.tar.gz <batch_name>/
```

Send `<batch_name>_annotations.tar.gz` to Lennert.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `napari: command not found` after install | Re-run `conda activate uwf-annotate` |
| napari window opens but is blank | Toggle the layer visibility eye icons in the layer list (left panel) |
| Prefill is empty for an image | The prediction file may be missing — annotate from scratch and tell Lennert |
| Tool crashes on a specific image | Skip with `Q` and tell Lennert which stem |

For anything else, contact Lennert (lennert.beeckmans@gmail.com) or open an
issue on the repo.
