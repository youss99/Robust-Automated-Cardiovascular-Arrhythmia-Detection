#!/bin/bash
# -----------------------------------------------------------------------------
# Chapman pretrain — lead V2 only
#
# Reproduces params.txt from:
#   results/pre-train/saved_models/chapman/patch_ecg_chapman_v2_only/
#
# Submit:  sbatch src/scripts/multi-gpu/pretrain/chapman/chapman_v2_only.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=chapman_v2_only
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chapman_v2_only-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/chapman"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

echo "Staging chapman dataset to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/chapman"
echo "Staged $(du -sh "$SLURM_TMPDIR/chapman" | cut -f1)"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=chapman \
    --root_path="$SLURM_TMPDIR/chapman" \
    --data_sub_path=training \
    --custom_lead_selection="V2" \
    --model_name=patch_ecg_chapman_v2_only \
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
    --n_epochs_pretrain=100 \
    --save_every=20 \
    --plot_every_n=1 \
    --world_size=1 \
    --local_node
