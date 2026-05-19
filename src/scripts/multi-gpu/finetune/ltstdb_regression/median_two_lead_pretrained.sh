#!/bin/bash
# -----------------------------------------------------------------------------
# LTSTDB regression fine-tune — MEDIAN target, two-lead (II+V2), PRETRAINED
#   from chapman two-lead (patch_ecg_chapman_ii_v2_tta_best.pt)
#
# Mirrors params.txt from local run:
#   ltstdb_median_full16_12_2_2_omit_train_two_lead_regression_pretrained_chapman_ii_v2_trial01/
#
# Submit:  sbatch src/scripts/multi-gpu/finetune/ltstdb_regression/median_two_lead_pretrained.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=ltst_med_pre
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ltstdb_median_two_lead_pretrained-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$REPO_ROOT/custom_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress"
PRETRAINED_CKPT="$REPO_ROOT/results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "ERROR: pretrained checkpoint missing: $PRETRAINED_CKPT" >&2
  exit 2
fi

echo "Staging median dataset to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/dataset"
echo "Staged $(du -sh "$SLURM_TMPDIR/dataset" | cut -f1)"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 \
    --lr=0.0003 \
    --save_every=20 \
    --root_path="$SLURM_TMPDIR/dataset" \
    --num_workers=4 \
    --batch_size=128 \
    --data_augmentation=none \
    --trafos None \
    --class_token=cls_token \
    --shared_embedding \
    --min_class_size=1 \
    --layer_decay=0.65 \
    --weight_decay=0.05 \
    --head_dropout=0.0 \
    --dropout=0.0 \
    --dset_finetune=patient-segment-regression \
    --classification_type=REGRESSION \
    --head_type=regression \
    --regression_loss=mse \
    --no-focal_loss \
    --model=vanilla_vit \
    --d_model=128 \
    --no-scheduler \
    --custom_lead_selection="II,V2" \
    --n_epochs_finetune_head=5 \
    --n_epochs_finetune=40 \
    --n_heads=8 \
    --n_layers=6 \
    --d_ff=512 \
    --patch_len=50 \
    --stride=50 \
    --model_name=ltstdb_median_full16_12_2_2_omit_train_two_lead_regression_pretrained_chapman_ii_v2_trial01 \
    --metric_by_class \
    --no-mentor_mix \
    --no-bootstrapping \
    --iterations=5000 \
    --model_selection_metric=valid_loss \
    --pretrained_model_path="$PRETRAINED_CKPT"
