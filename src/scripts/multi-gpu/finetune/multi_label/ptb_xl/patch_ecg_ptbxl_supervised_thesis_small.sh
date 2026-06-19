#!/bin/bash
# -----------------------------------------------------------------------------
# PTB-XL multi-label fine-tune (71-class) — two-lead (II+V2), thesis Small
#   SUPERVISED from scratch (NO pretraining). Matched baseline for the
#   Unified-pretrained run, to quantify the pretraining LIFT on PTB-XL.
#
# CONTROLLED COMPARISON: this is byte-for-byte identical to
#   patch_ecg_unified2ptbxl_thesis_small.sh EXCEPT it does not load a
#   pretrained checkpoint. The fine-tuning protocol (head 20 + full 70,
#   lr, wd, folds) is kept identical so the ONLY variable is pretraining.
#   Thesis Table 5.1 reference: ViT no-pretrain -> PTB-XL = 0.8067.
#
# NOTE: with no checkpoint the code auto-uses the plain optimizer
#   (create_optimizer), so layer_decay=0.65 is passed but INERT. The head=20
#   "freeze" phase trains the head on a random (frozen) backbone, then unfreezes
#   -- kept for protocol parity with the pretrained run.
#
# Submit:
#   sbatch src/scripts/multi-gpu/finetune/multi_label/ptb_xl/patch_ecg_ptbxl_supervised_thesis_small.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=ptb_ml_sup_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_ecg_ptbxl_ii_v2_supervised_thesis_small-%j.out

set -uo pipefail

module load python/3.10 cuda cudnn gcc arrow

REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/ptb-xl_1.0.3"

cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

echo "Staging PTB-XL (1.0.3) to $SLURM_TMPDIR ..."
cp -R "$DATA_SRC" "$SLURM_TMPDIR/ptb-xl"
echo "Staged $(du -sh "$SLURM_TMPDIR/ptb-xl" | cut -f1)"

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
    --model_name=patch_ecg_ptbxl_ii_v2_supervised_thesis_small
