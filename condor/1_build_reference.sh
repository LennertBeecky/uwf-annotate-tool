#!/usr/bin/bash
echo "Starting job: Build reference distribution"

export PATH="/esat/biomeddata/users/lbeeckma/miniconda3/bin:$PATH"
source /esat/biomeddata/users/lbeeckma/miniconda3/etc/profile.d/conda.sh
conda activate physics-informed_fundus

which python
python -V

cd /esat/biomeddata/users/lbeeckma/Physics-Informed_Fundus
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo $PWD

python ml/encoder/build_reference.py

echo "Job finished"
