#!/bin/bash
# -----------------------------------------------------------------------------
# Unified multi-label SUPERVISED (from scratch, NO pretraining) — two-lead (II+V2)
#   thesis Small hyperparameters (Table A.2/A.3), Option C: keep ALL classes
#   passing min_class_size, filter at inference.
#
# This is the SUPERVISED BASELINE (BCE, no focal/LoRA/SAM) — the "thesis Small"
# vanilla config. It establishes the floor of the pretraining payoff:
#   - thesis Small, 12-lead, from scratch (PTB-XL)  ~0.807 macro-AUROC
#   - thesis Base,  12-lead, from scratch (Unified) ~0.898
#   This run (Small, 2-lead, from scratch) should land ~0.85-0.90.
#   True ceiling = the PRETRAINED two-lead Small (CODE pretrain), not this.
#
# Output checkpoint:
#   results/fine-tune/saved_models/.../patch_ecg_unified_ii_v2_multilabel_supervised_thesis_small/
# Submit:
#   sbatch src/scripts/multi-gpu/finetune/multi_label/unified/patch_ecg_supervised_thesis_small.sh
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=uni_ml_sup_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_ecg_unified_ii_v2_multilabel_supervised_thesis_small-%j.out

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

# NOTE on thesis-Small deviations forced by from-scratch (no checkpoint):
#   - n_epochs_finetune_head=0  (thesis used 20; that is a FREEZE phase that only
#     makes sense with a pretrained backbone -- nothing to freeze from scratch)
#   - layer_decay=0.65 is passed for parity but is INERT: with no pretrained_model_path
#     the code uses the plain optimizer (create_optimizer), which ignores layer_decay.
#   Everything else matches Table A.2 Small: lr=1e-3, constant LR, dropout=0,
#   weight_decay=0.05, batch=128, full epochs=70, no LoRA/SAM/MentorMix, BCE loss.

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
    --n_epochs_finetune_head=0 \
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
    --model_name=patch_ecg_unified_ii_v2_multilabel_supervised_thesis_small
