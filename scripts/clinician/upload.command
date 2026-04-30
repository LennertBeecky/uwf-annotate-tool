#!/usr/bin/env bash
# UWF Annotation Tool — package annotations for upload (Mac/Linux).
# Double-click to:
#   1. tar each completed batch's annotations as <batch>_<annotator>_annotations.tar.gz
#   2. open Finder pointing at the folder so you can drag the tarball
#      onto your OneDrive 'returned/<your-name>/' folder

set -e

ID_FILE="$HOME/.uwf-annotate-id"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$INSTALL_DIR"

if [ -f "$ID_FILE" ]; then
    ANNOTATOR_ID=$(cat "$ID_FILE")
else
    echo "ERROR: no annotator ID set. Re-run setup.command first."
    read -p "Press Enter to close..."; exit 1
fi

ANNOTATIONS_ROOT="clinician_data/annotations"
if [ ! -d "$ANNOTATIONS_ROOT" ]; then
    echo "No annotations folder found ($ANNOTATIONS_ROOT)."
    read -p "Press Enter to close..."; exit 0
fi

TAR_OUT="$ANNOTATIONS_ROOT"

shopt -s nullglob
BATCHES=("$ANNOTATIONS_ROOT"/batch_*)
shopt -u nullglob
if [ ${#BATCHES[@]} -eq 0 ]; then
    echo "No batches with annotations yet."
    read -p "Press Enter to close..."; exit 0
fi

PACKED=0
for BATCH_DIR in "${BATCHES[@]}"; do
    [ -d "$BATCH_DIR" ] || continue
    BATCH_NAME=$(basename "$BATCH_DIR")
    ANNOTATOR_DIR="$BATCH_DIR/$ANNOTATOR_ID"
    if [ ! -d "$ANNOTATOR_DIR" ]; then
        continue
    fi
    N=$(ls "$ANNOTATOR_DIR"/*_artery.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N" -eq 0 ]; then
        continue
    fi
    TAR_NAME="${BATCH_NAME}_${ANNOTATOR_ID}_annotations.tar.gz"
    echo "Packing $TAR_NAME ($N annotated images) ..."
    tar -czf "$TAR_OUT/$TAR_NAME" -C "$BATCH_DIR" "$ANNOTATOR_ID"
    PACKED=$((PACKED + 1))
done

if [ "$PACKED" -eq 0 ]; then
    echo "Nothing to pack — no annotations from $ANNOTATOR_ID found."
    read -p "Press Enter to close..."; exit 0
fi

echo ""
echo "================================================================"
echo "  Packed $PACKED batch(es). Files ready to upload:"
echo "================================================================"
ls -lh "$TAR_OUT"/*.tar.gz 2>/dev/null
echo ""
echo "  Drag each .tar.gz onto your OneDrive 'returned/$ANNOTATOR_ID/'"
echo "  folder. The Finder window opening below has the files."
echo ""

# Open Finder/Explorer at the output location.
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$TAR_OUT"
elif command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$TAR_OUT"
fi

read -p "Press Enter to close..."
