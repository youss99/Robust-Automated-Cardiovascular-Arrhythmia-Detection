#!/bin/bash
# CODE-backbone -> Unified fine-tune | thesis Small, two-lead II+V2 | Config 1: BCE
#   (matches the Unified->PTB-XL 0.8556 config -> clean pretrain-corpus-size compare)
#SBATCH --account=def-majidk
#SBATCH --job-name=c2u_c1_bce
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch_ecg_code2unified_ii_v2_c1_bce-%j.out
set -uo pipefail
module load python/3.10 cuda cudnn gcc arrow
REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/Unified_Dataset_No_PTB_Segment"
PRETRAINED_CKPT="$REPO_ROOT/results/pre-train/saved_models/CODE_Annotated/patch_ecg_code_annotated_ii_v2/patch_ecg_code_annotated_ii_v2_best.pt"
cd "$REPO_ROOT"; source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"; mkdir -p logs
[[ -f "$PRETRAINED_CKPT" ]] || { echo "ERROR: ckpt missing: $PRETRAINED_CKPT" >&2; exit 2; }
echo "Staging unified to $SLURM_TMPDIR ..."; cp -R "$DATA_SRC" "$SLURM_TMPDIR/unified"
torchrun --standalone --nproc_per_node=1 -m src.patchECG_finetune \
    --seed=200 --dset_finetune=unified --classification_type=MULTI_LABEL \
    --diagnostic_class=all --min_class_size=15 --custom_lead_selection="II,V2" \
    --root_path="$SLURM_TMPDIR/unified" \
    --model=vanilla_vit --d_model=128 --n_heads=8 --n_layers=6 --d_ff=512 \
    --patch_len=50 --stride=50 --class_token=cls_token --shared_embedding \
    --dropout=0.0 --head_dropout=0.0 --batch_size=128 --num_workers=4 \
    --lr=0.001 --no-scheduler --weight_decay=0.05 --layer_decay=0.65 \
    --n_epochs_finetune_head=20 --n_epochs_finetune=70 \
    --trafos None --no-mentor_mix \
    --no-focal_loss --data_augmentation=none \
    --model_selection_metric=valid_AUROC --metric_by_class \
    --no-bootstrapping --iterations=5000 --save_every=20 \
    --pretrained_model_path="$PRETRAINED_CKPT" \
    --model_name=patch_ecg_code2unified_ii_v2_c1_bce
