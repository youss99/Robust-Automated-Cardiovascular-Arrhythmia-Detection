#!/usr/bin/env bash
# SUPERVISED (from scratch, NO pretraining) sibling of finetune_classifier_v5.sh.
# Flags copied verbatim from the proven multi_label/unified/patch_ecg_supervised_thesis_small.sh;
# only dset / leads / model_name are parameterized, and dataset staging matches
# finetune_classifier_v5.sh (PTB-XL = ptb-xl_1.0.3, the native-100 Hz copy).
#   - n_epochs_finetune_head=0: the head-freeze phase only makes sense with a pretrained body.
#   - layer_decay=0.65 passed for parity but INERT without pretrained_model_path.
# Usage: sbatch --job-name=<name> finetune_classifier_v5_supervised.sh <dset: unified|ptb-xl> <leads> <model_name>
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
set -euo pipefail
DSET="$1"; LEADS="$2"; MODEL_NAME="$3"
module load python/3.10 cuda cudnn gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATASETS="$HOME/projects/def-majidk/youss99/Project_1/Datasets"
case "$DSET" in
  unified) SRC="$DATASETS/Unified_Dataset_No_PTB_Segment_fixed"; STAGE="unified"; MIN=15;;
  ptb-xl)  SRC="$DATASETS/ptb-xl_1.0.3";                  STAGE="ptb-xl";  MIN=0;;
  *) echo "unknown dset: $DSET" >&2; exit 2;;
esac
cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
echo "Staging $DSET ($LEADS) -> $SLURM_TMPDIR/$STAGE  (supervised: no backbone)"; cp -R "$SRC" "$SLURM_TMPDIR/$STAGE"
torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 --dset_finetune="$DSET" --classification_type=MULTI_LABEL \
    --diagnostic_class=all --min_class_size="$MIN" --custom_lead_selection="$LEADS" \
    --root_path="$SLURM_TMPDIR/$STAGE" \
    --model=vanilla_vit --d_model=128 --n_heads=8 --n_layers=6 --d_ff=512 \
    --patch_len=50 --stride=50 --class_token=cls_token --shared_embedding \
    --dropout=0.0 --head_dropout=0.0 --batch_size=128 --num_workers=4 \
    --lr=0.001 --no-scheduler --weight_decay=0.05 --layer_decay=0.65 \
    --n_epochs_finetune_head=0 --n_epochs_finetune=70 \
    --data_augmentation=none --trafos None --no-focal_loss --no-mentor_mix \
    --model_selection_metric=valid_AUROC --metric_by_class \
    --no-bootstrapping --iterations=5000 --save_every=20 \
    --model_name="$MODEL_NAME" ${EXTRA_ARGS:-}
