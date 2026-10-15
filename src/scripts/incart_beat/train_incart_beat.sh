#!/usr/bin/env bash
# INCART lead-II central-beat runs A/B/C (sequential, resumable: resubmit to continue after a timeout).
# Usage: [LEAD=V5] [SEEDS=200,201,202] sbatch src/scripts/incart_beat/train_incart_beat.sh [extra args, e.g. --smoke | --runs B --evaluate_test]
#SBATCH --account=def-majidk
#SBATCH --job-name=incart_beat
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
# Walltime: 3 runs x 90 epochs (~15 min each) + slow /project imports, ~2x measured
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load python/3.10 cuda cudnn gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
CACHE="$HOME/projects/def-majidk/youss99/Project_1/Datasets/incart_beat_lead${LEAD:-II}_100hz"
cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
[[ -f "$CACHE/signals_uV_100hz.npy" ]] || { echo "ERROR: run prep_incart_beat.sh first" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -n "${SEEDS:-}" ]]; then  # e.g. SEEDS=200,201,202: one run per seed, sequentially on this GPU slice
    for S in ${SEEDS//,/ }; do
        python -m src.incart_beat.train --cache_dir "$CACHE" --lead "${LEAD:-II}" --seed "$S" "$@"
    done
else
    python -m src.incart_beat.train --cache_dir "$CACHE" --lead "${LEAD:-II}" "$@"
fi
