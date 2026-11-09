#!/usr/bin/env bash
# PatchECG Small backbone on Unified_Dataset_No_PTB_Segment_v2, without rows where a selected lead is blank.
# Same recipe as the *_fixed backbones (500 epochs; those took 50 min single-lead, 1h21 two-lead on this profile).
# Usage (from the repo root): sbatch --job-name=<model_name> src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh <leads> <model_name> [epochs]
#   e.g. sbatch --job-name=patch_ecg_unified_ii_v2_rebuilt src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh II,V2 patch_ecg_unified_ii_v2_rebuilt
# The trainer continues from an existing snapshot, so an existing model folder stops the job: remove it to start clean.
#SBATCH --account=def-majidk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
LEADS="$1"; MODEL_NAME="$2"; EPOCHS="${3:-500}"

module load python/3.10 cuda cudnn gcc arrow
REPO="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/Unified_Dataset_No_PTB_Segment_v2"

cd "$REPO"; source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p logs
OUT="results/pre-train/saved_models/unified/$MODEL_NAME"
if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists; remove it to start clean" >&2; exit 2
fi
python src/scripts/unification/stage_unified_subset.py --source "$SRC" --leads "$LEADS" --out "$SLURM_TMPDIR/unified"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=unified --root_path="$SLURM_TMPDIR/unified" \
    --custom_lead_selection="$LEADS" --model_name="$MODEL_NAME" \
    --diagnostic_class=all --min_class_size=0 \
    --context_points=5000 --target_points=100 --scaler=standard --features=M \
    --patch_len=50 --stride=50 --revin_mode=BN \
    --model=vanilla_vit --n_layers=6 --n_heads=8 --d_model=128 --d_ff=512 \
    --shared_embedding --dropout=0.0 --head_dropout=0.0 \
    --data_augmentation=none --trafos None --mask_ratio=0.4 --zero_noise_mixing=0.5 \
    --opt=adamw --weight_decay=0.05 --lr=0.0015 --scheduler \
    --warmup_epochs=2 --warmup_lr=1e-06 --min_lr=1e-05 \
    --batch_size=128 --num_workers=4 --n_epochs_pretrain="$EPOCHS" \
    --save_every=20 --plot_every_n=1 --world_size=1 --local_node

# params.txt only records the node-local path, so keep the data provenance and the split with the model
cp "$SLURM_TMPDIR/unified/modified_dataset/modified_dataset.csv" "$OUT/pretrain_split.csv"
{ echo "dataset: Unified_Dataset_No_PTB_Segment_v2 (UnifiedDataset.h5 md5 $(cut -d' ' -f1 "$SRC/MD5SUMS" | head -1))"
  echo "leads: $LEADS; rows where a selected lead is blank removed by stage_unified_subset.py"
  echo "rows trained on (train + validation): $(($(wc -l < "$OUT/pretrain_split.csv") - 1))"; } > "$OUT/data_source.txt"
