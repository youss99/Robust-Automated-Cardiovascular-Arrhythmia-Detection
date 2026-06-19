#!/bin/bash
# -----------------------------------------------------------------------------
# PTB-XL multi-label fine-tune (71-class) — two-lead (II+V2), thesis Small
#   PRETRAINED on Unified (patch_ecg_unified_ii_v2_best.pt) -> fine-tune PTB-XL.
#
# This reproduces thesis Section 5.1 / Table 5.1 (ViT pretrained-on-Unified ->
# PTB-XL = 0.9031 vs 0.8067 from scratch), but at two-lead II+V2 instead of
# 12-lead. LEAKAGE-CLEAN: your Unified is "No_PTB_Segment", disjoint from PTB-XL.
#
# Uses the SUGGESTED PTB-XL folds (1-8 train / 9 val / 10 test) automatically.
# With a real pretrained backbone, the freeze phase (head=20) and layer_decay
# are ACTIVE (unlike the from-scratch supervised run).
#
# Submit:
#   sbatch src/scripts/multi-gpu/finetune/multi_label/ptb_xl/patch_ecg_unified2ptbxl_thesis_small.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=u2ptb_ml_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_ecg_unified2ptbxl_ii_v2_thesis_small-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/ptb-xl_1.0.3"
PRETRAINED_CKPT="$REPO_ROOT/results/pre-train/saved_models/unified/patch_ecg_unified_ii_v2/patch_ecg_unified_ii_v2_best.pt"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "ERROR: pretrained checkpoint missing: $PRETRAINED_CKPT" >&2
  exit 2
fi

echo "Staging PTB-XL (1.0.3) to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/ptb-xl"
echo "Staged $(du -sh "$SLURM_TMPDIR/ptb-xl" | cut -f1)"

# thesis Small fine-tuning (Table A.2 Small) WITH pretrained backbone:
#   head 20 (freeze phase, now valid) + full 70, lr 1e-3, constant LR,
#   layer_decay 0.65 (ACTIVE), wd 0.05, dropout 0, batch 128, BCE, no LoRA/SAM.
#   diagnostic_class=all -> 71-class task; min_class_size=0 keeps all classes;
#   PTB-XL suggested folds used automatically.

torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 \
    --dset_finetune=ptb-xl \
    --classification_type=MULTI_LABEL \
    --diagnostic_class=all \
    --min_class_size=0 \
    --custom_lead_selection="II,V2" \
    --root_path="$SLURM_TMPDIR/ptb-xl" \
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
    --no-focal_loss \
    --no-mentor_mix \
    --model_selection_metric=valid_AUROC \
    --metric_by_class \
    --no-bootstrapping \
    --iterations=5000 \
    --save_every=20 \
    --pretrained_model_path="$PRETRAINED_CKPT" \
    --model_name=patch_ecg_unified2ptbxl_ii_v2_thesis_small
