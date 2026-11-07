#!/usr/bin/env bash
# Build/repair the repo .venv on a Nibi compute node (keeps pip + torch imports off the login node).
# Usage (from repo root): sbatch src/scripts/nibi_setup_venv.sbatch
#SBATCH --account=def-majidk
#SBATCH --job-name=setup_venv
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00   # a fresh install writes ~25k small files to /scratch; 30 min was too short
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

REPO=/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
cd "$REPO"
module load StdEnv/2023 python/3.11

[ -x .venv/bin/python ] || virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-index --upgrade pip
# Wheelhouse only. wfdb is not in the wheelhouse and is not needed for pretraining.
# timm==1.0.11: newest wheelhouse timm that still has timm.optim.nadam.Nadam / radam.RAdam (used by optim_factory.py).
pip install --no-index torch torchvision "numpy<2" pandas h5py scipy scikit-learn scikit-multilearn \
    einops "timm==1.0.11" kornia peft transformers accelerate torchmetrics torcheval matplotlib tqdm pyyaml

export OMP_NUM_THREADS=1
python -c "import src.patchECG_pretrain, torch, timm, numpy; print('imports OK | torch', torch.__version__, '| timm', timm.__version__, '| numpy', numpy.__version__)"
