#!/usr/bin/env bash
# Tier-3 classifier finetune (thesis Small, V5 lead modes): BCE baseline + Focal Loss + TTA + SAM.
# Sibling of finetune_classifier_v5.sh — identical backbone/optimiser; the ONLY differences are the
# three low-data/imbalance methods turned ON (see CHANGES vs v5 block below).
#
# Use this when the FINE-TUNE set is small and/or class-imbalanced (the regime where the thesis
# ablations show TTA/Focal/SAM help; on large balanced sets they are ~neutral — use v5 there).
#
# Usage: sbatch --job-name=<name> --dependency=afterok:<PID> \
#          finetune_classifier_v5_tta_focal_sam.sh <dset: unified|ptb-xl> <leads> <backbone_ckpt> <model_name>
#   e.g. sbatch --job-name=u2p_iiv5_c3 finetune_classifier_v5_tta_focal_sam.sh \
#          ptb-xl "II,V5" <ckpt> unified2ptbxl_ii_v5_tta_focal_sam
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00          # bumped vs v5 (1h): SAM doubles fwd/bwd per step, TTA adds chunk passes
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
DSET="$1"; LEADS="$2"; CKPT="$3"; MODEL_NAME="$4"

module load python/3.10 cuda cudnn gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATASETS="$HOME/projects/def-majidk/youss99/Project_1/Datasets"
case "$DSET" in
  unified) SRC="$DATASETS/Unified_Dataset_No_PTB_Segment"; STAGE="unified"; MIN=15;;
  ptb-xl)  SRC="$DATASETS/ptb-xl_1.0.3";                  STAGE="ptb-xl";  MIN=0;;
  *) echo "unknown dset: $DSET" >&2; exit 2;;
esac

cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
[[ -f "$CKPT" ]] || { echo "ERROR: backbone ckpt missing: $CKPT" >&2; exit 2; }
echo "Staging $DSET ($LEADS) -> $SLURM_TMPDIR/$STAGE  (backbone: $CKPT)"; cp -R "$SRC" "$SLURM_TMPDIR/$STAGE"

# ---- CHANGES vs finetune_classifier_v5.sh -----------------------------------------------------
#   v5 (BCE baseline):  --no-focal_loss  --data_augmentation=none  (no SAM)
#   this script:        --focal_loss --focal_alpha=0.25                 (Focal Loss; Lin et al. default alpha)
#                       --data_augmentation=test_time_aug_transformer   (TTA; chunk_size/step default 250/125)
#                       --sam --sam_rho=2 --sam_adaptive                (SAM; matches thesis Table A.2)
#   Everything else (arch, lr, layer_decay, weight_decay, epochs, seed, selection metric) is identical to v5.
#   LoRA is intentionally NOT enabled here (that is the c3 config); add --lora if you want the full stack.
# -----------------------------------------------------------------------------------------------
torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 --dset_finetune="$DSET" --classification_type=MULTI_LABEL \
    --diagnostic_class=all --min_class_size="$MIN" --custom_lead_selection="$LEADS" \
    --root_path="$SLURM_TMPDIR/$STAGE" \
    --model=vanilla_vit --d_model=128 --n_heads=8 --n_layers=6 --d_ff=512 \
    --patch_len=50 --stride=50 --class_token=cls_token --shared_embedding \
    --dropout=0.0 --head_dropout=0.0 --batch_size=128 --num_workers=4 \
    --lr=0.001 --no-scheduler --weight_decay=0.05 --layer_decay=0.65 \
    --n_epochs_finetune_head=20 --n_epochs_finetune=70 \
    --trafos None --no-mentor_mix \
    --focal_loss --focal_alpha=0.25 \
    --data_augmentation=test_time_aug_transformer \
    --sam --sam_rho=2 --sam_adaptive \
    --model_selection_metric=valid_AUROC --metric_by_class \
    --no-bootstrapping --iterations=5000 --save_every=20 \
    --pretrained_model_path="$CKPT" --model_name="$MODEL_NAME"
