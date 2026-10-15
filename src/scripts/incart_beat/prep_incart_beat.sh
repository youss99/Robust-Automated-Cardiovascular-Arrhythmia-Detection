#!/usr/bin/env bash
# Build the INCART single-lead 100 Hz uV cache + beat table (CPU only, ~1-2 min).
# Usage: [LEAD=V5] sbatch src/scripts/incart_beat/prep_incart_beat.sh   (default LEAD=II)
#SBATCH --account=def-majidk
#SBATCH --job-name=incart_prep
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load python/3.10 gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATASETS="$HOME/projects/def-majidk/youss99/Project_1/Datasets"
cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p logs

python -m src.incart_beat.prepare_incart \
    --incart_dir "$DATASETS/incartdb" \
    --lead "${LEAD:-II}" \
    --out_dir "$DATASETS/incart_beat_lead${LEAD:-II}_100hz"
