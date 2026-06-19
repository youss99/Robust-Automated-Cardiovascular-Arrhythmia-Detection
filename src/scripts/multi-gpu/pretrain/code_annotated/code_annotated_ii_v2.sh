#!/bin/bash
# -----------------------------------------------------------------------------
# CODE-ANNOTATED pretrain — two-lead (II + V2), thesis Small hyperparameters.
#   Clean SSL pretrain corpus (242,148 signals) that is DISJOINT from your
#   Unified build, so a downstream Unified fine-tune is leakage-free.
#   MPM is label-free, so CODE's 6 label classes are irrelevant here.
#
#   Stages ONLY the 11.6 GB annotated h5 + label csv (NOT the 310 GB
#   CODE/training/ raw corpus). Load path reads the existing
#   modified_dataset/modified_annotations.csv, so training/ is never touched.
#
# Output checkpoint:
#   results/pre-train/saved_models/CODE_Annotated/patch_ecg_code_annotated_ii_v2/
# Submit:
#   sbatch src/scripts/multi-gpu/pretrain/code_annotated/code_annotated_ii_v2.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=code_ann_ii_v2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch_ecg_code_annotated_ii_v2-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/CODE"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

# Selective staging: only the annotated h5 + its label csv (~11.6 GB),
# NOT the 310 GB CODE/training/ raw corpus.
echo "Staging CODE-annotated (h5 + labels only) to $SLURM_TMPDIR ..."
mkdir -p "$SLURM_TMPDIR/code/modified_dataset"
cp "$DATA_SRC/CodeDatasetAnnotated.h5" "$SLURM_TMPDIR/code/"
cp "$DATA_SRC/modified_dataset/modified_annotations.csv" "$SLURM_TMPDIR/code/modified_dataset/"
echo "Staged $(du -sh "$SLURM_TMPDIR/code" | cut -f1)"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=CODE_Annotated \
    --root_path="$SLURM_TMPDIR/code" \
    --custom_lead_selection="II,V2" \
    --model_name=patch_ecg_code_annotated_ii_v2 \
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
    --n_epochs_pretrain=100 \
    --save_every=20 \
    --plot_every_n=1 \
    --world_size=1 \
    --local_node
