#!/bin/bash
# -----------------------------------------------------------------------------
# Unified multi-label fine-tune — two-lead (II+V2), thesis Small, Option C.
#   PRETRAINED on CODE-annotated (clean: CODE disjoint from Unified) ->
#   fine-tune Unified. Leakage-free path to a decent Unified-label model.
#   Focal loss ON (rare ventricular/block classes).
#
# Chained to run after the CODE-annotated pretrain (afterok dependency).
# Pretrained ckpt:
#   results/pre-train/saved_models/CODE_Annotated/patch_ecg_code_annotated_ii_v2/..._best.pt
#
# Submit (with dependency):
#   sbatch --dependency=afterok:<PRETRAIN_JOBID> \
#     src/scripts/multi-gpu/finetune/multi_label/unified/patch_ecg_code2unified_thesis_small.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=code2uni_ml_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=logs/patch_ecg_code2unified_ii_v2_thesis_small-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/Unified_Dataset_No_PTB_Segment"
PRETRAINED_CKPT="$REPO_ROOT/results/pre-train/saved_models/CODE_Annotated/patch_ecg_code_annotated_ii_v2/patch_ecg_code_annotated_ii_v2_best.pt"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "ERROR: pretrained checkpoint missing: $PRETRAINED_CKPT" >&2
  exit 2
fi

echo "Staging unified-no-PTB dataset to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/unified"
echo "Staged $(du -sh "$SLURM_TMPDIR/unified" | cut -f1)"

# thesis Small fine-tune WITH pretrained backbone: head 20 (freeze, active) +
# full 70, lr 1e-3, constant LR, layer_decay 0.65 (active), wd 0.05, batch 128.
# Option C: diagnostic_class=all + min_class_size=15, no custom_class_selection.
# Focal loss ON for the rare ventricular/block classes.

torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 \
    --dset_finetune=unified \
    --classification_type=MULTI_LABEL \
    --diagnostic_class=all \
    --min_class_size=15 \
    --custom_lead_selection="II,V2" \
    --root_path="$SLURM_TMPDIR/unified" \
    --model=vanilla_vit \
    --d_model=128 \
    --n_heads=8 \
    --n_layers=6 \
    --d_ff=512 \
    --patch_len=50 \
    --stride=50 \
    --class_token=cls_token \
    --shared_embedding \
    --dropout=0.0 \
    --head_dropout=0.0 \
    --batch_size=128 \
    --num_workers=4 \
    --lr=0.001 \
    --no-scheduler \
    --weight_decay=0.05 \
    --layer_decay=0.65 \
    --n_epochs_finetune_head=20 \
    --n_epochs_finetune=70 \
    --data_augmentation=none \
    --trafos None \
    --focal_loss \
    --focal_alpha=0.25 \
    --no-mentor_mix \
    --model_selection_metric=valid_AUROC \
    --metric_by_class \
    --no-bootstrapping \
    --iterations=5000 \
    --save_every=20 \
    --pretrained_model_path="$PRETRAINED_CKPT" \
    --model_name=patch_ecg_code2unified_ii_v2_thesis_small_focal
