#!/usr/bin/env bash
# UWF Annotation Tool — one-time setup for clinicians (Mac/Linux).
# Double-click this file to install / update the annotation tool.
#
# Idempotent: re-run any time to update to the latest version.

set -e

INSTALL_DIR="$HOME/uwf-annotate"
REPO_URL="https://github.com/LennertBeecky/uwf-annotate-tool.git"
BRANCH="main"
ENV_NAME="uwf-annotate"
ID_FILE="$HOME/.uwf-annotate-id"

echo "================================================================"
echo "  UWF Vessel Annotation Tool — Setup"
echo "================================================================"
echo ""

# 1. Conda check
if ! command -v conda > /dev/null 2>&1; then
    cat <<'EOF'
ERROR: conda was not found on your PATH.

Please install miniconda first:
  https://docs.conda.io/en/latest/miniconda.html

Choose the installer for your operating system and run it with default
settings. After it finishes, open a *new* Terminal window and re-run
this setup script.
EOF
    read -p "Press Enter to close this window..."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
echo "  conda found at: $CONDA_BASE"

# 2. Repo: clone or pull
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Updating existing install at $INSTALL_DIR ..."
    cd "$INSTALL_DIR"
    git fetch --all --quiet
    git checkout "$BRANCH" --quiet
    git pull --quiet
else
    echo "  Cloning repository to $INSTALL_DIR ..."
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$BRANCH" --quiet
fi
echo "  Repo on branch: $(git branch --show-current)"

# 3. Conda env
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  Updating conda environment '$ENV_NAME' ..."
    conda env update -n "$ENV_NAME" -f environment_clinician.yml --prune --quiet
else
    echo "  Creating conda environment '$ENV_NAME' (this takes ~5 min) ..."
    conda env create -n "$ENV_NAME" -f environment_clinician.yml --quiet
fi

# 4. Annotator ID prompt (one-time)
if [ ! -f "$ID_FILE" ]; then
    echo ""
    echo "  Setting your annotator ID (used to label your annotations)."
    read -rp "  Enter your name (lowercase, no spaces, e.g. ingrid): " ANNOTATOR
    ANNOTATOR=$(echo "$ANNOTATOR" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_')
    if [ -z "$ANNOTATOR" ]; then
        echo "  No name entered — defaulting to 'anonymous'. Edit ~/.uwf-annotate-id later if you want to change."
        ANNOTATOR="anonymous"
    fi
    echo "$ANNOTATOR" > "$ID_FILE"
    echo "  Saved annotator ID: $ANNOTATOR"
else
    echo "  Annotator ID: $(cat "$ID_FILE") (from $ID_FILE)"
fi

# 5. Data folders
mkdir -p "$INSTALL_DIR/clinician_data/incoming"
mkdir -p "$INSTALL_DIR/clinician_data/incoming/processed"

echo ""
echo "================================================================"
echo "  Setup complete!"
echo "================================================================"
echo ""
echo "  Install location: $INSTALL_DIR"
echo ""
echo "  Next steps:"
echo "    1. Download a batch zip (e.g. batch_2026-04-30.zip) from OneDrive."
echo "    2. Move it into:"
echo "         $INSTALL_DIR/clinician_data/incoming/"
echo "    3. Double-click annotate.command (in scripts/clinician/) to start."
echo ""
echo "  When you're done with a batch, double-click upload.command to"
echo "  package your annotations for upload to OneDrive."
echo ""
read -p "Press Enter to close this window..."
