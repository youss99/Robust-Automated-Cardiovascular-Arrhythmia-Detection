#!/usr/bin/env bash
# Parameterized classifier-backbone pretrain (thesis Small, V5 lead modes).
# Usage: sbatch --job-name=<name> pretrain_classifier_v5.sh <dset: ptb-xl|unified> <leads> <model_name>
#   e.g. sbatch --job-name=ptbxl_ii_v5 pretrain_classifier_v5.sh ptb-xl "II,V5" patch_ecg_ptbxl_ii_v5
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
DSET="$1"; LEADS="$2"; MODEL_NAME="$3"

module load python/3.10 cuda cudnn gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATASETS="$HOME/projects/def-majidk/youss99/Project_1/Datasets"
case "$DSET" in
  ptb-xl)  SRC="$DATASETS/ptb-xl_1.0.3";                 STAGE="ptb-xl";;
  unified) SRC="$DATASETS/Unified_Dataset_No_PTB_Segment"; STAGE="unified";;
  *) echo "unknown dset: $DSET" >&2; exit 2;;
esac

cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
echo "Staging $DSET ($LEADS) -> $SLURM_TMPDIR/$STAGE"; cp -R "$SRC" "$SLURM_TMPDIR/$STAGE"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain="$DSET" --root_path="$SLURM_TMPDIR/$STAGE" \
    --custom_lead_selection="$LEADS" --model_name="$MODEL_NAME" \
    --diagnostic_class=all --min_class_size=0 \
    --context_points=5000 --target_points=100 --scaler=standard --features=M \
    --patch_len=50 --stride=50 --revin_mode=BN \
    --model=vanilla_vit --n_layers=6 --n_heads=8 --d_model=128 --d_ff=512 \
    --shared_embedding --dropout=0.0 --head_dropout=0.0 \
    --data_augmentation=none --trafos None --mask_ratio=0.4 --zero_noise_mixing=0.5 \
    --opt=adamw --weight_decay=0.05 --lr=0.0015 --scheduler \
    --warmup_epochs=2 --warmup_lr=1e-06 --min_lr=1e-05 \
    --batch_size=128 --num_workers=4 --n_epochs_pretrain=500 \
    --save_every=20 --plot_every_n=1 --world_size=1 --local_node
