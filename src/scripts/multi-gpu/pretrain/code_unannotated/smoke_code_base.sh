#!/bin/bash
# -----------------------------------------------------------------------------
# SMOKE TEST for pretrain_code_base.sh -- validates the CODE_Unannotated loader
# path, which has never run in this project (the only CODE backbone to date,
# patch_ecg_code_annotated_ii_v2, went through the *Annotated* loader).
#
# 3 shards, 1 epoch, batch 128, lead II. Same arg set as the real script.
# A MIG slice is deliberate: this validates code, not throughput, and queues fast.
#
# Submit:  sbatch src/scripts/multi-gpu/pretrain/code_unannotated/smoke_code_base.sh
# Delete results/pre-train/saved_models/CODE_Unannotated/smoke_code_base_ii/ after.
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=smoke_code_base
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail

SHARDS=(S0000000 S0010000 S0020000)

module load python/3.10 cuda cudnn gcc arrow
REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/CODE"
cd "$REPO_ROOT"; source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

echo "=== GPU actually allocated ==="; nvidia-smi --query-gpu=name,memory.total --format=csv

SM="$SLURM_TMPDIR/CODEsmoke"
mkdir -p "$SM/training" "$SM/modified_dataset"
for s in "${SHARDS[@]}"; do cp "$DATA_SRC/training/$s.h5" "$SM/training/"; done
echo "Staged ${#SHARDS[@]} shards."

# The loader globs the staged shards but reads EVERY manifest row, so the CSV
# must be filtered to the staged groups or it KeyErrors on the other 291.
CSV="$DATA_SRC/modified_dataset/modified_unannotated.csv"
OUT="$SM/modified_dataset/modified_unannotated.csv"
head -1 "$CSV" > "$OUT"
pat=$(IFS='|'; echo ",(${SHARDS[*]}),")
grep -E "$pat" "$CSV" >> "$OUT"
echo "Manifest rows: $(( $(wc -l < "$OUT") - 1 ))"

torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=CODE_Unannotated \
    --root_path="$SM" \
    --data_sub_path=training \
    --custom_lead_selection="II" \
    --model_name=smoke_code_base_ii \
    --model=vanilla_vit \
    --d_model=768 --n_heads=12 --n_layers=12 --d_ff=3072 \
    --shared_embedding \
    --patch_len=50 --stride=50 \
    --mask_ratio=0.4 \
    --dropout=0. --head_dropout=0. \
    --data_augmentation=none --trafos=None \
    --weight_decay=0.05 \
    --lr=0.0015 --warmup_epochs=1 \
    --batch_size=128 --num_workers=4 \
    --n_epochs_pretrain=1 --save_every=1 \
    --world_size=1 --local_node

CKPT_DIR="$REPO_ROOT/results/pre-train/saved_models/CODE_Unannotated/smoke_code_base_ii"
echo "=== checkpoints written ==="; ls -l "$CKPT_DIR" 2>&1 || true
if compgen -G "$CKPT_DIR/*.pt" > /dev/null; then
    echo "SMOKE TEST PASSED -- loader path works, checkpoint written."
else
    echo "SMOKE TEST FAILED -- no .pt produced." >&2; exit 1
fi
