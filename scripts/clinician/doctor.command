#!/usr/bin/env bash
# UWF Annotation Tool — diagnostic (Mac/Linux).
# Double-click when napari won't open. Writes a report to your Desktop.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT="$HOME/Desktop/uwf_annotate_diagnostic.txt"
cd "$INSTALL_DIR"

echo "================================================================"
echo "  UWF Annotation Tool -- diagnostic"
echo "================================================================"
echo ""

PY=""
for CANDIDATE in \
    "$HOME/miniconda3/envs/uwf-annotate/bin/python" \
    "$HOME/anaconda3/envs/uwf-annotate/bin/python" \
    "/opt/miniconda3/envs/uwf-annotate/bin/python" \
    "/opt/anaconda3/envs/uwf-annotate/bin/python"; do
    [ -x "$CANDIDATE" ] && PY="$CANDIDATE"
done

if [ -z "$PY" ]; then
    echo "ERROR: could not find the uwf-annotate environment."
    echo "Re-run setup.command and watch for errors."
    echo "uwf-annotate environment not found - setup did not complete." > "$REPORT"
    read -p "Press Enter to close..."
    exit 1
fi

echo "  Using: $PY"
echo ""
"$PY" annotation_tool/doctor.py "$REPORT"

echo ""
echo "Report written to: $REPORT"
read -p "Press Enter to close..."
