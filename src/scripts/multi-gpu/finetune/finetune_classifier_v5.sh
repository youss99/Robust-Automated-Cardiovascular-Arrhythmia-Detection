#!/usr/bin/env bash
# Parameterized classifier finetune (thesis Small, V5 lead modes) from a pretrained backbone.
# Usage: sbatch --job-name=<name> --dependency=afterok:<PID> \
#          finetune_classifier_v5.sh <dset: unified|ptb-xl> <leads> <backbone_ckpt> <model_name>
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
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

torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 --dset_finetune="$DSET" --classification_type=MULTI_LABEL \
    --diagnostic_class=all --min_class_size="$MIN" --custom_lead_selection="$LEADS" \
    --root_path="$SLURM_TMPDIR/$STAGE" \
    --model=vanilla_vit --d_model=128 --n_heads=8 --n_layers=6 --d_ff=512 \
    --patch_len=50 --stride=50 --class_token=cls_token --shared_embedding \
    --dropout=0.0 --head_dropout=0.0 --batch_size=128 --num_workers=4 \
    --lr=0.001 --no-scheduler --weight_decay=0.05 --layer_decay=0.65 \
    --n_epochs_finetune_head=20 --n_epochs_finetune=70 \
    --trafos None --no-mentor_mix --no-focal_loss --data_augmentation=none \
    --model_selection_metric=valid_AUROC --metric_by_class \
    --no-bootstrapping --iterations=5000 --save_every=20 \
    --pretrained_model_path="$CKPT" --model_name="$MODEL_NAME"
