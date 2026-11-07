#!/usr/bin/env bash
# Nibi port of pretrain_classifier_v5.sh: same hyperparameters, Nibi paths, H100 MIG slice.
# Usage (from repo root): sbatch --job-name=<name> pretrain_classifier_v5_nibi.sh <dset: ptb-xl|unified> <leads> <model_name> [epochs] [batch]
#   e.g. sbatch --job-name=unified_v2_fixed src/scripts/multi-gpu/pretrain/pretrain_classifier_v5_nibi.sh unified V2 patch_ecg_unified_v2_fixed
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
DSET="$1"; LEADS="$2"; MODEL_NAME="$3"
EPOCHS="${4:-500}"   # optional: override epochs (use 1 for a smoke/sizing run)
BATCH="${5:-128}"    # optional: override batch (drop for long all_leads sequences if OOM)

module load StdEnv/2023 python/3.11
REPO=/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
DATASETS=/home/youss99/scratch/GIT/datasets
case "$DSET" in
  ptb-xl)  SRC="$DATASETS/ptb-xl_1.0.3";                 STAGE="ptb-xl";;
  unified) SRC="$DATASETS/Unified_Dataset_No_PTB_Segment_fixed"; STAGE="unified";;
  *) echo "unknown dset: $DSET" >&2; exit 2;;
esac

cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
# Source datasets are read-only; the staged copy must be writable in case the loader writes caches.
echo "Staging $DSET ($LEADS) -> $SLURM_TMPDIR/$STAGE"; cp -R "$SRC" "$SLURM_TMPDIR/$STAGE"; chmod -R u+w "$SLURM_TMPDIR/$STAGE"
nvidia-smi -L

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain="$DSET" --root_path="$SLURM_TMPDIR/$STAGE" \
    --custom_lead_selection="$LEADS" --model_name="$MODEL_NAME" \
    --diagnostic_class=all --min_class_size=0 \
    --context_points=5000 --target_points=100 --scaler=standard --features=M \
    --patch_len=50 --stride=50 --revin_mode=BN \
    --model=vanilla_vit --n_layers=6 --n_heads=8 --d_model=128 --d_ff=512 \
    --shared_embedding --dropout=0.0 --head_dropout=0.0 \
    --data_augmentation=none --trafos None --mask_ratio=0.4 --zero_noise_mixing=0.5 \
    --opt=adamw --weight_decay=0.05 --lr=0.0015 --scheduler \
    --warmup_epochs=2 --warmup_lr=1e-06 --min_lr=1e-05 \
    --batch_size="$BATCH" --num_workers=4 --n_epochs_pretrain="$EPOCHS" \
    --save_every=20 --plot_every_n=1 --world_size=1 --local_node
