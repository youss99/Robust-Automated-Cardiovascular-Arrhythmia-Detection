#!/bin/bash
# -----------------------------------------------------------------------------
# Chapman pretrain — SMOKE TEST (1 epoch, MIG slice, 30 min budget)
#
# Goal: verify the full pipeline (data loading, model init, forward/backward,
#       checkpoint write, plot generation) works on the cluster before
#       committing to a 4-hour real run.
#
# Uses the two-lead path (most complex) so a passing smoke test implies the
# single-lead variants will work too.
#
# Output dir uses a "_smoketest" suffix so it does NOT overwrite the real
# patch_ecg_chapman_ii_v2_tta results.
#
# Submit:  sbatch src/scripts/multi-gpu/pretrain/chapman/chapman_smoketest.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=chapman_smoketest
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/chapman_smoketest-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/chapman"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

echo "==== Smoke test starting at $(date) ===="
echo "Node:       $(hostname)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:     $(which python) ($(python --version))"
echo "Torch:      $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "==========================================="

echo "Staging chapman dataset to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/chapman"
echo "Staged $(du -sh "$SLURM_TMPDIR/chapman" | cut -f1)"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=chapman \
    --root_path="$SLURM_TMPDIR/chapman" \
    --data_sub_path=training \
    --custom_lead_selection="II,V2" \
    --model_name=patch_ecg_chapman_ii_v2_smoketest \
    --min_class_size=50 \
    --context_points=5000 \
    --target_points=100 \
    --scaler=standard \
    --features=M \
    --patch_len=50 \
    --stride=50 \
    --revin_mode=BN \
    --model=vanilla_vit \
    --n_layers=6 \
    --n_heads=8 \
    --d_model=128 \
    --d_ff=512 \
    --shared_embedding \
    --dropout=0.0 \
    --head_dropout=0.0 \
    --data_augmentation=none \
    --trafos None \
    --mask_ratio=0.4 \
    --zero_noise_mixing=0.5 \
    --opt=adamw \
    --weight_decay=0.05 \
    --lr=0.0015 \
    --scheduler \
    --warmup_epochs=2 \
    --warmup_lr=1e-06 \
    --min_lr=1e-05 \
    --batch_size=100 \
    --num_workers=4 \
    --n_epochs_pretrain=1 \
    --save_every=1 \
    --plot_every_n=1 \
    --world_size=1 \
    --local_node

echo "==== Smoke test finished at $(date) ===="
ls -la "results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_v2_smoketest/" 2>/dev/null \
  && echo "PASS: smoke test produced output dir" \
  || echo "FAIL: no output dir produced"
