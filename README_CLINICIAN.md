# UWF Vessel Annotation — Clinician Setup

Hand-trace artery and vein skeletons on retinal images. The tool opens each
image with a model prediction pre-filled — you correct it, save, and move on.

---

## A. One-time setup

You'll do this once. About 10 minutes total.

### A.1. Install miniconda

If you don't already have it: download the installer for your OS from
<https://docs.conda.io/en/latest/miniconda.html> and run it with default
settings. After it finishes, **close and re-open** any terminal windows so
`conda` is on your PATH.

### A.2. Run the setup script

You'll receive a `setup.command` (Mac) or `setup.bat` (Windows) file from
Lennert via OneDrive or email. Save it anywhere convenient (Desktop is fine).

**Mac**: double-click `setup.command`. If macOS warns about an unidentified
developer, right-click → **Open** → confirm.

**Windows**: double-click `setup.bat`. If SmartScreen warns, click **More
info** → **Run anyway**.

A terminal window opens. The script will:

- Check that conda is installed (errors out with a link if not)
- Clone the annotation repo into `~/uwf-annotate/`
- Create a conda environment called `uwf-annotate` (~5 minutes)
- Ask you for your **annotator ID** (a short lowercase name like `ingrid`).
  This labels every annotation you save so we know who did what.

When it says "Setup complete!" you're ready.

### A.3. Re-running setup

You can re-run `setup.command` / `setup.bat` any time. It updates the tool
to the latest version (we may push fixes during the study).

---

## B. Per-batch workflow

For each batch Lennert sends you:

### B.1. Download the batch zip

He uploads a single file like `batch_2026-04-30.zip` to your OneDrive
folder. Download it to your computer (Downloads folder is fine).

### B.2. Move the zip into the incoming folder

Drag `batch_2026-04-30.zip` into:

- **Mac**: `~/uwf-annotate/clinician_data/incoming/`
- **Windows**: `C:\Users\<you>\uwf-annotate\clinician_data\incoming\`

(In Finder/Explorer, you can also browse to `uwf-annotate` from your home
folder, then `clinician_data` → `incoming`.)

### B.3. Start annotating

Inside `uwf-annotate/scripts/clinician/`, double-click:

- **Mac**: `annotate.command`
- **Windows**: `annotate.bat`

A terminal opens, then napari. The script:

- Auto-extracts your batch zip
- Activates the conda environment
- Launches the annotation tool with the model's prediction pre-filled
  (red = artery, blue = vein)
- Walks through every image in the batch, skipping ones you've already
  finished

### B.4. While annotating

Keys (also shown in the napari status bar):

| Action | Mac/Windows |
|---|---|
| Pan / zoom | `1` |
| Paint mode | `3` |
| Switch artery / vein layer | `Tab` |
| Smaller / bigger brush | `[` / `]` |
| **Save and advance to next image** | `Q` |
| **Skip this image** (no save) | `S` |

When you're done with a session, just close napari (or press `Q` after the
last image). You can come back and run `annotate.command` again any time —
images you've already saved are skipped automatically.

**Read `annotation_tool/protocol.md` once before your first session** — it
covers the colour conventions, what to annotate, how to handle bifurcations
and crossings, and what to skip.

### B.5. From-scratch calibration

Roughly **1 in every 15 images, please annotate from scratch** (no
prefill) so we can measure prefill bias later. Lennert will tell you which
stems are flagged for the calibration set.

### B.6. Send your annotations back

When you've finished a batch (or want to send progress), double-click:

- **Mac**: `upload.command`
- **Windows**: `upload.bat`

It packages your annotations for the batch into a single archive
(e.g. `batch_2026-04-30_<your-name>_annotations.tar.gz` on Mac,
`.zip` on Windows) and opens Finder/Explorer to where it lives.

Drag that one file onto your OneDrive `returned/<your-name>/` folder.

That's the entire workflow.

---

## C. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` after install | Close all terminals, re-open. Or restart your computer. |
| Mac: "cannot be opened because it is from an unidentified developer" | Right-click the `.command` file → **Open** → confirm |
| Windows: SmartScreen blocks the `.bat` | Click **More info** → **Run anyway** |
| napari opens but is blank | In napari's left panel, click the eye icons next to "artery" and "veins" to toggle visibility |
| The terminal closes immediately | Re-open it from the script and check the printed error message |
| "No batch zips found" message | You haven't dropped a batch zip into `clinician_data/incoming/` yet |
| Prefill is empty for some images | The model has no prediction for that one. Annotate from scratch and tell Lennert |

For anything else, contact Lennert (<lennert.beeckmans@gmail.com>) or open
an issue on the GitHub repo.

---

## D. What the folders look like

After your first session:

```
uwf-annotate/
├── annotation_tool/                  # the tool (don't edit)
├── clinician_data/
│   ├── incoming/
│   │   ├── batch_2026-04-30.zip      # ← drop new batch zips here
│   │   └── processed/                # finished zips get moved here
│   ├── images_to_annotate/
│   │   └── batch_2026-04-30/
│   │       └── <stem>.jpeg           # extracted automatically
│   ├── predictions/
│   │   └── batch_2026-04-30/
│   │       └── <stem>_hard.png       # extracted automatically
│   └── annotations/
│       └── batch_2026-04-30/
│           ├── <your-name>/          # your saved annotations
│           │   ├── <stem>_artery.png
│           │   └── <stem>_veins.png
│           └── batch_2026-04-30_<your-name>_annotations.tar.gz
│                                      # ← upload this to OneDrive
├── scripts/clinician/
│   ├── setup.command / setup.bat
│   ├── annotate.command / annotate.bat
│   └── upload.command / upload.bat
└── README_CLINICIAN.md               # this file
```

You should never need to touch anything outside `clinician_data/`.
