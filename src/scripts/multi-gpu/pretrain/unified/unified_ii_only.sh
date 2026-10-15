#!/bin/bash
# -----------------------------------------------------------------------------
# Unified pretrain — lead II only, thesis Small hyperparameters
#
# Derived from chapman/chapman_ii_only.sh. Three deltas vs Chapman pretrain:
#   - dataset: chapman (45k) -> unified-no-PTB (59k records)
#   - batch:   100  -> 128
#   - epochs:  100  -> 500
# Output checkpoint:
#   results/pre-train/saved_models/unified/patch_ecg_unified_ii_only/
#
# Submit:  sbatch src/scripts/multi-gpu/pretrain/unified/unified_ii_only.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=unified_ii_only
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=logs/unified_ii_only-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/Unified_Dataset_No_PTB_Segment"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

echo "Staging unified-no-PTB dataset to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/unified"
echo "Staged $(du -sh "$SLURM_TMPDIR/unified" | cut -f1)"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=unified \
    --root_path="$SLURM_TMPDIR/unified" \
    --custom_lead_selection="II" \
    --model_name=patch_ecg_unified_ii_only \
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
    --batch_size=128 \
    --num_workers=4 \
    --n_epochs_pretrain=500 \
    --save_every=20 \
    --plot_every_n=1 \
    --world_size=1 \
    --local_node
