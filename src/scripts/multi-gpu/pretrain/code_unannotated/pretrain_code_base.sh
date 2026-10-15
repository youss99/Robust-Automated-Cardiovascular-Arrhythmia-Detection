#!/bin/bash
# -----------------------------------------------------------------------------
# THESIS-FAITHFUL CODE pretrain: self-supervised masked-patch on the FULL CODE
# corpus (CODE_Unannotated -- the 301 GB training/ shards, ~8M ECGs), ViT-Base.
# Parameterized by lead selection so one script covers all 5 lead-sets.
#
# Recipe mirrors src/scripts/multi-gpu/pretrain/patch_ecg.sh (the thesis template):
#   ViT-Base 768/12/12/3072, patch/stride 50, mask 0.4, lr 1.5e-3, wd 0.05,
#   batch 128, 100 epochs, distributed multi-GPU -- batch 128 per Table A.3
#   (Mitchell's template used 100; single-lead is 20 patches/sample vs 240, so
#   the memory headroom is there). Only deviation from the thesis is therefore
#   a lead SUBSET instead of --custom_lead_selection=all_leads.
#
# Usage:  sbatch pretrain_code_base.sh "<leads>" <model_name>
#   sbatch pretrain_code_base.sh "II,V2" patch_ecg_code_base_ii_v2
#   sbatch pretrain_code_base.sh "II,V5" patch_ecg_code_base_ii_v5
#   sbatch pretrain_code_base.sh "II"    patch_ecg_code_base_ii
#   sbatch pretrain_code_base.sh "V5"    patch_ecg_code_base_v5
#   sbatch pretrain_code_base.sh "V2"    patch_ecg_code_base_v2
# Checkpoints -> results/pre-train/saved_models/CODE_Unannotated/<model_name>/
# -----------------------------------------------------------------------------
#SBATCH --account=def-majidk
#SBATCH --job-name=code_base_pretrain
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4       # 4x FULL A100 (40GB). The "a100:" type prefix is REQUIRED:
                                     # a bare "4" also matches Narval's MIG slices
                                     # (a100_3g.20gb:4 on ng20304), giving 4x20GB fractions.
#SBATCH --ntasks-per-node=4          # one task per GPU (SLURM-managed DDP, like the thesis)
#SBATCH --cpus-per-task=8
#SBATCH --mem=187G
#SBATCH --time=3-00:00:00            # base + full corpus is multi-day; save_every=20 lets you resume
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail

LEADS="${1:?usage: sbatch pretrain_code_base.sh \"<leads>\" <model_name>}"
MODEL_NAME="${2:?provide a model_name (e.g. patch_ecg_code_base_ii_v2)}"

module load python/3.10 cuda cudnn gcc arrow
REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/CODE"
cd "$REPO_ROOT"; source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NCCL_BLOCKING_WAIT=1
export MAIN_NODE=$(hostname)
mkdir -p logs

# ---- stage the FULL unannotated corpus to node-local NVMe (training/ + manifest) ----
# (~301 GB -> can take 30-60+ min; the loader globs <root>/training/*.h5 and reads
#  <root>/modified_dataset/modified_unannotated.csv. We skip the 11 GB annotated .h5.)
DEST="$SLURM_TMPDIR/CODE"
mkdir -p "$DEST/training" "$DEST/modified_dataset"
echo "Free space on \$SLURM_TMPDIR:"; df -h "$SLURM_TMPDIR"
# Fail in seconds, not 45 minutes, if this node's /localscratch is already full.
# Narval A100 nodes advertise TmpDisk=3450000 MB (~3.37 TB), so this normally passes.
MIN_FREE_KB=$((330 * 1024 * 1024))   # 301 GB of shards + margin
FREE_KB=$(df -Pk "$SLURM_TMPDIR" | awk 'NR==2{print $4}')
if [ "$FREE_KB" -lt "$MIN_FREE_KB" ]; then
    echo "FATAL: \$SLURM_TMPDIR has $((FREE_KB/1048576)) GB free; need >= 330 GB." >&2
    exit 1
fi
echo "Staging CODE training shards to $DEST ..."
cp "$DATA_SRC"/training/*.h5 "$DEST/training/"
cp "$DATA_SRC"/modified_dataset/modified_unannotated.csv "$DEST/modified_dataset/"
STAGED=$(ls "$DEST/training"/*.h5 | wc -l)
echo "Staged $STAGED shards."
if [ "$STAGED" -ne 294 ]; then
    echo "FATAL: expected 294 shards, staged $STAGED -- copy was incomplete." >&2
    exit 1
fi

# ---- distributed ViT-Base pretrain (SLURM srun DDP, thesis pattern) ----
srun python -m src.patchECG_pretrain \
    --init_method "tcp://${MAIN_NODE}:3456" \
    --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) \
    --distributed \
    --dset_pretrain=CODE_Unannotated \
    --root_path="$DEST" \
    --data_sub_path=training \
    --custom_lead_selection="$LEADS" \
    --model_name="$MODEL_NAME" \
    --model=vanilla_vit \
    --d_model=768 --n_heads=12 --n_layers=12 --d_ff=3072 \
    --shared_embedding \
    --patch_len=50 --stride=50 \
    --mask_ratio=0.4 \
    --dropout=0. --head_dropout=0. \
    --data_augmentation=none --trafos=None \
    --weight_decay=0.05 \
    --lr=0.0015 --warmup_epochs=2 \
    --batch_size=128 --num_workers=4 \
    --n_epochs_pretrain=100 --save_every=20
