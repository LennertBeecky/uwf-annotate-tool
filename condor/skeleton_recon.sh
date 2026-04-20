#!/usr/bin/bash
# Per-shard wrapper for the skeleton-reconstruction Condor job.
# Usage (from the submit file): skeleton_recon.sh $(Process) 5
#
# Runs the reconstruction on its assigned shard of the selected splits,
# writes results to shard-tagged CSVs so the 5 shards can be merged later.

set -euo pipefail

SHARD_INDEX="${1:?missing shard index}"
NUM_SHARDS="${2:?missing shard count}"

# ---- configure paths here --------------------------------------------------
# Where the databases/Train,Test,... directories live. Override at submit
# time by exporting UWF_DATABASES in the submit environment.
DATABASES_DIR="${UWF_DATABASES:-/esat/biomeddata/users/lbeeckma/Physics-Informed_Fundus/databases}"
# Which splits to process. Default = Train only. Override with UWF_SPLITS.
SPLITS="${UWF_SPLITS:-Train}"
# Where shard outputs + masks land. Override with UWF_OUTPUT_DIR.
OUTPUT_DIR="${UWF_OUTPUT_DIR:-/esat/biomeddata/users/lbeeckma/uwf/experiments/skeleton_reconstruction}"
# ----------------------------------------------------------------------------

echo "=== skeleton_recon shard ${SHARD_INDEX}/${NUM_SHARDS} ==="
date
echo "DATABASES_DIR=${DATABASES_DIR}"
echo "SPLITS=${SPLITS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

export PATH="/esat/biomeddata/users/lbeeckma/miniconda3/bin:$PATH"
source /esat/biomeddata/users/lbeeckma/miniconda3/etc/profile.d/conda.sh
conda activate physics-informed_fundus

which python
python -V

cd /esat/biomeddata/users/lbeeckma/uwf
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
echo "cwd: $PWD"

# Prevent BLAS thread over-subscription when the 8 Python workers each spin
# up their own BLAS thread pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u experiments/skeleton_reconstruction/run_batch.py \
    --db "${DATABASES_DIR}" \
    --splits "${SPLITS}" \
    --output-dir "${OUTPUT_DIR}" \
    --shard "${SHARD_INDEX}/${NUM_SHARDS}" \
    --workers 8 \
    --save-masks \
    --save-preview-overlays

echo "=== shard ${SHARD_INDEX}/${NUM_SHARDS} done ==="
date
