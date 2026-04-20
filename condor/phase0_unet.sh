#!/usr/bin/bash
# Phase-0 UNet training wrapper for Condor (GPU node).
# Runs cond_A, cond_C, cond_G sequentially on the assigned GPU node.

set -euo pipefail

echo "=== phase0_unet ==="
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

export PATH="/esat/biomeddata/users/lbeeckma/miniconda3/bin:$PATH"
source /esat/biomeddata/users/lbeeckma/miniconda3/etc/profile.d/conda.sh
conda activate physics-informed_fundus

which python
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /esat/biomeddata/users/lbeeckma/uwf
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
echo "cwd: $PWD"

# Prevent BLAS thread over-subscription when DataLoader workers spin up.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Overrides via env vars (set before `condor_submit`):
#   WANDB_API_KEY=...      — wandb auth (or pre-login with `wandb login`)
#   WANDB_PROJECT=uwf-phase0
#   WANDB_ENTITY=<team>
#   WANDB_TAGS=phase0,gpu
#   PHASE0_EPOCHS=100
#   PHASE0_CONDITIONS=A,C,G
#   PHASE0_SUFFIX=""
#   PHASE0_MODE=full       # full (default) | sanity | sanity_then_full
EPOCHS="${PHASE0_EPOCHS:-100}"
CONDITIONS="${PHASE0_CONDITIONS:-A,C,G}"
SUFFIX="${PHASE0_SUFFIX:-}"
WANDB_PROJECT="${WANDB_PROJECT:-uwf-phase0}"
WANDB_TAGS="${WANDB_TAGS:-phase0,gpu}"
MODE="${PHASE0_MODE:-full}"

EXTRA_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

run_sanity() {
    echo "\n=== PHASE 0 SANITY (5 epochs, wandb disabled) ==="
    python -u experiments/phase0_unet/run_all_conditions.py \
        --conditions "${CONDITIONS}" \
        --sanity \
        --require-gpu \
        --suffix "_sanity"
}

run_full() {
    echo "\n=== PHASE 0 FULL (${EPOCHS} epochs, wandb ${WANDB_MODE:-online}) ==="
    python -u experiments/phase0_unet/run_all_conditions.py \
        --conditions "${CONDITIONS}" \
        --num-epochs "${EPOCHS}" \
        --require-gpu \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-mode "${WANDB_MODE:-online}" \
        --wandb-tags "${WANDB_TAGS}" \
        --suffix "${SUFFIX}" \
        "${EXTRA_ARGS[@]}"
}

case "${MODE}" in
    sanity)
        run_sanity
        ;;
    full)
        run_full
        ;;
    sanity_then_full)
        run_sanity
        run_full
        ;;
    *)
        echo "Unknown PHASE0_MODE=${MODE} (use sanity | full | sanity_then_full)"
        exit 1
        ;;
esac

echo "=== phase0_unet done ==="
date
