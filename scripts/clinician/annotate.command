#!/usr/bin/env bash
# UWF Annotation Tool — start an annotation session (Mac/Linux).
# Double-click to:
#   1. find the latest batch zip dropped into clinician_data/incoming/
#   2. extract images + predictions into the right per-batch folders
#   3. launch the napari annotation tool with prefill on
#
# Re-run any time during a batch — already-annotated images are skipped.

set -e

ENV_NAME="uwf-annotate"
ID_FILE="$HOME/.uwf-annotate-id"

# Silence noisy-but-harmless OpenMP / Qt warnings on macOS.
export KMP_WARNINGS=0
export QT_LOGGING_RULES="qt.qpa.window.warning=false"

# ----------------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$INSTALL_DIR"

INCOMING="clinician_data/incoming"
PROCESSED="clinician_data/incoming/processed"
mkdir -p "$INCOMING" "$PROCESSED"

# ----------------------------------------------------------------- conda
if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found. Please re-run setup.command."
    read -p "Press Enter to close..."; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ----------------------------------------------------------------- annotator id
if [ -f "$ID_FILE" ]; then
    ANNOTATOR_ID=$(cat "$ID_FILE")
else
    echo "ERROR: no annotator ID set. Re-run setup.command first."
    read -p "Press Enter to close..."; exit 1
fi
echo "Annotator: $ANNOTATOR_ID"

# ----------------------------------------------------------------- pick batch
shopt -s nullglob
ZIPS=("$INCOMING"/batch_*.zip)
shopt -u nullglob

if [ ${#ZIPS[@]} -eq 0 ]; then
    # No new zip — maybe she already extracted one. Look for in-progress
    # batches (folders that exist but aren't fully annotated).
    shopt -s nullglob
    BATCHES=(clinician_data/images_to_annotate/batch_*)
    shopt -u nullglob
    if [ ${#BATCHES[@]} -eq 0 ]; then
        cat <<EOF
No batch zips found in $INCOMING/ and no batches in progress.

To start a new batch:
  1. Download batch_*.zip from OneDrive.
  2. Move it into: $INSTALL_DIR/$INCOMING/
  3. Re-run this script (annotate.command).
EOF
        read -p "Press Enter to close..."; exit 0
    fi
    if [ ${#BATCHES[@]} -eq 1 ]; then
        BATCH_DIR="${BATCHES[0]}"
    else
        echo "Multiple in-progress batches. Pick one:"
        for i in "${!BATCHES[@]}"; do
            n=$(ls "${BATCHES[$i]}" 2>/dev/null | wc -l | tr -d ' ')
            echo "  $((i+1)). $(basename "${BATCHES[$i]}")  ($n images)"
        done
        read -rp "Choice [1-${#BATCHES[@]}]: " CHOICE
        BATCH_DIR="${BATCHES[$((CHOICE-1))]}"
    fi
    BATCH_NAME=$(basename "$BATCH_DIR")
else
    # Have at least one new zip — extract it.
    if [ ${#ZIPS[@]} -gt 1 ]; then
        echo "Multiple batch zips found:"
        for i in "${!ZIPS[@]}"; do
            echo "  $((i+1)). $(basename "${ZIPS[$i]}")"
        done
        read -rp "Pick one [1-${#ZIPS[@]}]: " CHOICE
        ZIP="${ZIPS[$((CHOICE-1))]}"
    else
        ZIP="${ZIPS[0]}"
    fi
    BATCH_NAME=$(basename "$ZIP" .zip)
    echo "Extracting batch: $BATCH_NAME"

    mkdir -p "clinician_data/images_to_annotate/$BATCH_NAME"
    mkdir -p "clinician_data/predictions/$BATCH_NAME"

    TMPDIR=$(mktemp -d)
    unzip -q "$ZIP" -d "$TMPDIR"

    # Expect bundled layout: images/, predictions/, README.txt
    if [ -d "$TMPDIR/images" ]; then
        mv "$TMPDIR/images/"* "clinician_data/images_to_annotate/$BATCH_NAME/" 2>/dev/null || true
    fi
    if [ -d "$TMPDIR/predictions" ]; then
        mv "$TMPDIR/predictions/"* "clinician_data/predictions/$BATCH_NAME/" 2>/dev/null || true
    fi

    # Move zip to processed/ so we don't re-extract on next launch.
    mv "$ZIP" "$PROCESSED/"
    rm -rf "$TMPDIR"

    BATCH_DIR="clinician_data/images_to_annotate/$BATCH_NAME"
fi

# ----------------------------------------------------------------- output dirs
ANNOTATIONS_DIR="clinician_data/annotations/$BATCH_NAME/$ANNOTATOR_ID"
mkdir -p "$ANNOTATIONS_DIR"

N_IMG=$(ls "$BATCH_DIR" 2>/dev/null | wc -l | tr -d ' ')
N_PRED=$(ls "clinician_data/predictions/$BATCH_NAME" 2>/dev/null | wc -l | tr -d ' ')
N_DONE=$(ls "$ANNOTATIONS_DIR"/*_artery.png 2>/dev/null | wc -l | tr -d ' ')

cat <<EOF

================================================================
  Batch:        $BATCH_NAME
  Images:       $N_IMG
  Predictions:  $N_PRED
  Done so far:  $N_DONE / $N_IMG
================================================================

Launching napari... (close the window or press 'q' to save+next, 's' to skip)
EOF

# ----------------------------------------------------------------- annotate
# Disable strict mode just for the python call — napari on macOS sometimes
# hits a segfault during shared-library teardown after the user closes the
# window (numpy/Qt ABI bug, harmless because annotations are saved before
# napari quits). We don't want a 139 exit code to abort the trailing
# "Session ended" message and the read-prompt that keeps the terminal open.
set +e
python annotation_tool/annotate.py \
    "$BATCH_DIR/" \
    --output-dir "$ANNOTATIONS_DIR/" \
    --prefill predictions \
    --predictions-dir "clinician_data/predictions/$BATCH_NAME/"
PY_EXIT=$?
set -e

if [ "$PY_EXIT" -ne 0 ] && [ "$PY_EXIT" -ne 139 ]; then
    echo "WARNING: annotate.py exited with code $PY_EXIT (your saved"
    echo "         annotations are still on disk; re-run this script to"
    echo "         continue). If this keeps happening, send Lennert the"
    echo "         output above."
fi

cat <<EOF

Session ended.

When you're done with this batch, double-click upload.command to package
your annotations for return to OneDrive.

(You can re-run annotate.command any time — already-saved images are skipped.)
EOF
read -p "Press Enter to close..."
