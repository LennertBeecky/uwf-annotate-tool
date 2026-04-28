# UWF Vessel Annotation — Clinician Setup

This is the annotation workflow for hand-tracing artery and vein skeletons on
ultra-widefield (UWF) retinal images. The tool opens each image in a napari
viewer with the model's prediction pre-filled as a starting point — you
correct it (erase wrong vessels, add missed ones, fix A/V class), and save.

This README is **PyCharm-first**. Equivalent terminal commands are included
at the end for reference.

---

## A. One-time setup (~20 min)

### A.1. Install miniconda

Required for the python environment. Download the installer for your OS from
<https://docs.conda.io/en/latest/miniconda.html> and run it with default
settings.

### A.2. Clone the repo via PyCharm

1. Open PyCharm.
2. **File → New → Project from Version Control** (or "Get from VCS" on the
   welcome screen).
3. **URL**: `https://github.com/LennertBeecky/uwf.git`
4. **Directory**: choose where to put it, e.g. `~/uwf-annotate`.
5. Click **Clone**.

### A.3. Switch to the clinician branch

The bottom-right corner of PyCharm shows the current Git branch (initially
`main`).

1. Click the branch name → **Remote Branches** → `origin/clinician_workflow`
   → **Checkout**.
2. After the checkout the project tree should show only:
    - `annotation_tool/`
    - `clinician_data/`
    - `README_CLINICIAN.md` (this file)
    - `environment_clinician.yml`

### A.4. Create the conda environment

1. **Settings** (Mac: **PyCharm → Settings**; Windows/Linux: **File →
   Settings**) → **Project: uwf-annotate** → **Python Interpreter**.
2. Click the ⚙️ icon → **Add Interpreter** → **Add Local Interpreter**.
3. Choose **Conda Environment** in the left panel.
4. Select **Create new environment**.
   - **Conda executable**: should auto-fill. If not, browse to
     `~/miniconda3/bin/conda` (Mac/Linux) or
     `C:\Users\<you>\miniconda3\Scripts\conda.exe` (Windows).
   - **Python version**: `3.11`
   - **Environment name**: `uwf-annotate`
5. Click **OK**, then **Apply**.

PyCharm now has an isolated env for this project.

### A.5. Install dependencies via PyCharm's terminal

1. Open the terminal panel: **View → Tool Windows → Terminal** (or click
   the **Terminal** tab at the bottom of PyCharm).
2. The prompt should already show `(uwf-annotate)`. If not, run:
   ```bash
   conda activate uwf-annotate
   ```
3. Install everything from the env file:
   ```bash
   conda env update -f environment_clinician.yml --prune
   ```
   (~5 min the first time.)
4. Verify:
   ```bash
   python -c "import napari, cv2, numpy, skimage, PIL; print('OK')"
   ```
   You should see `OK`. If not, contact Lennert.

You only do A.1–A.5 **once**, ever.

---

## B. Per-batch workflow

Each time Lennert sends a new batch:

### B.1. Receive the batch

You'll get three files via the shared drive (OneDrive / SharePoint / USB):

- `images.zip`           (the raw UWFs)
- `predictions.zip`      (the model's prefill, one `.png` per image)
- `README.txt`           (a small per-batch note with the exact batch name)

The batch name is the date stamp, e.g. `batch_2026-04-28`.

### B.2. Create the per-batch folders

In Finder/Explorer (or PyCharm's project tree → **right-click → New → Directory**),
make these three folders inside the project:

```
clinician_data/images_to_annotate/<batch_name>/
clinician_data/predictions/<batch_name>/
clinician_data/annotations/<batch_name>/
```

### B.3. Unzip the batch

- Double-click `images.zip` in Finder/Explorer.
- Drag the **contents** of the extracted `images/` folder into
  `clinician_data/images_to_annotate/<batch_name>/`.
- Same for `predictions.zip` → `clinician_data/predictions/<batch_name>/`.

> Note: when you extract a zip, you'll see an inner folder
> (`images/` or `predictions/`). It's the **contents** of that folder you want
> in the right place — not the folder itself.

PyCharm's project tree refreshes automatically — you should see the JPEGs
under `images_to_annotate/<batch_name>/` and the PNGs under
`predictions/<batch_name>/`.

### B.4. Set up the Run Configuration (first time only — reuse for later batches)

1. **Run → Edit Configurations** → **+** → **Python**.
2. Fill in:
    - **Name**: `Annotate batch`
    - **Script path**: `annotation_tool/annotate.py`
    - **Parameters**:
      ```
      clinician_data/images_to_annotate/<batch_name>/ --output-dir clinician_data/annotations/<batch_name>/ --prefill predictions --predictions-dir clinician_data/predictions/<batch_name>/
      ```
      Replace `<batch_name>` with the actual date, e.g.
      `clinician_data/images_to_annotate/batch_2026-04-28/ ...`
    - **Python interpreter**: `uwf-annotate` (the conda env from A.4).
    - **Working directory**: the project root (auto-filled).
3. Click **OK**.

For a new batch later, just **edit this one Run Configuration** and update
the three batch-name paths (or duplicate the config and keep one per
batch).

### B.5. Annotate

1. Click the green ▶️ at the top-right of PyCharm to launch the run config.
2. napari opens with the first un-done image:
    - The UWF in greyscale background.
    - Red overlay = predicted artery skeleton.
    - Blue overlay = predicted vein skeleton.
3. Use the in-tool tips for keys (printed in the bottom status bar):
    - `1` = pan/zoom mode
    - `3` = paint mode
    - `Tab` = switch between artery and vein layers
    - `[` / `]` = smaller / bigger brush
    - `Q` = save and advance to the next image
    - `S` = skip without saving
4. **Read `annotation_tool/protocol.md` once** before your first session — it
   covers the colour conventions, what to annotate, how to handle
   bifurcations and crossings, and what to skip.
5. Each save writes:
    - `<stem>_artery.png`  (red layer)
    - `<stem>_veins.png`   (blue layer)
   into `clinician_data/annotations/<batch_name>/`.

The walker auto-skips images that already have both files in the output
directory, so you can stop and resume any time — just hit ▶️ again later.

### B.6. From-scratch calibration

For ~1 in every 15 images, please **annotate from scratch** (no prefill) so
we can measure how much the prefill biases your tracing later. Easiest: keep
a second Run Configuration called `Annotate from scratch` with the same
parameters but `--prefill none` instead of `--prefill predictions ...`. Use
that config for the calibration images, and note in a small text file which
stems you did from scratch.

### B.7. Sending annotations back

1. Open the PyCharm terminal (View → Tool Windows → Terminal).
2. Run:
   ```bash
   cd clinician_data/annotations
   tar czf <batch_name>_annotations.tar.gz <batch_name>/
   ```
3. Upload the resulting `.tar.gz` to the shared OneDrive folder. Done.

### B.8. Updating the tool

When Lennert pushes fixes (better prefill, bug fixes, protocol clarifications),
just click **Git → Pull** in PyCharm. Your `clinician_data/` is in
`.gitignore`, so your in-progress annotations are untouched.

---

## C. Troubleshooting

| Symptom | Fix |
|---|---|
| PyCharm doesn't show the conda env after A.4 | Settings → Python Interpreter → click the ⚙️ → **Show All** → make sure `uwf-annotate` is selected |
| `napari: command not found` in terminal | Run `conda activate uwf-annotate` |
| Run config fails with "module not found" | Check the **Python interpreter** in the run config is `uwf-annotate`, not the system one |
| napari opens but is blank | Toggle the layer visibility (eye icons in the layer list, left panel) |
| Prefill is empty for a particular image | The prediction file may be missing — annotate from scratch and tell Lennert |
| Crash on a specific image | Press `S` to skip and tell Lennert which stem |
| `tar: command not found` (Windows) | Use 7-Zip or Windows Explorer's "Send to → Compressed folder" instead |

For anything else, contact Lennert (lennert.beeckmans@gmail.com) or open an
issue on the GitHub repo.

---

## D. Equivalent terminal commands

If you'd rather skip PyCharm and run from a plain shell:

```bash
# A.2 + A.3
git clone https://github.com/LennertBeecky/uwf.git uwf-annotate
cd uwf-annotate
git checkout clinician_workflow

# A.4 + A.5
conda env create -f environment_clinician.yml
conda activate uwf-annotate

# B.2 (per batch)
mkdir -p clinician_data/images_to_annotate/batch_2026-04-28
mkdir -p clinician_data/predictions/batch_2026-04-28
mkdir -p clinician_data/annotations/batch_2026-04-28

# B.3 (unzip; -j strips the inner folder)
unzip -j ~/Downloads/images.zip -d clinician_data/images_to_annotate/batch_2026-04-28/
unzip -j ~/Downloads/predictions.zip -d clinician_data/predictions/batch_2026-04-28/

# B.5
python annotation_tool/annotate.py \
    clinician_data/images_to_annotate/batch_2026-04-28/ \
    --output-dir clinician_data/annotations/batch_2026-04-28/ \
    --prefill predictions \
    --predictions-dir clinician_data/predictions/batch_2026-04-28/

# B.7
cd clinician_data/annotations
tar czf batch_2026-04-28_annotations.tar.gz batch_2026-04-28/
```
