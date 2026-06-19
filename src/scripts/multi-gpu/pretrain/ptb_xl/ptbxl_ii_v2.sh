#!/bin/bash
# PTB-XL pretrain — two-lead (II+V2), thesis Small. MATCHED recipe to
# unified_ii_v2.sh (same hyperparams, 500 epochs) so the PTB-XL<->Unified
# direction comparison is fair. Leakage-clean for a Unified downstream
# (PTB-XL disjoint from Unified-No-PTB).
# Output: results/pre-train/saved_models/ptb-xl/patch_ecg_ptbxl_ii_v2/
#SBATCH --account=def-majidk
#SBATCH --job-name=ptbxl_ii_v2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_ecg_ptbxl_ii_v2-%j.out
set -uo pipefail
module load python/3.10 cuda cudnn gcc arrow
REPO_ROOT="$HOME/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_SRC="$HOME/projects/def-majidk/youss99/Project_1/Datasets/ptb-xl_1.0.3"
cd "$REPO_ROOT"; source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"; mkdir -p logs
echo "Staging ptb-xl to $SLURM_TMPDIR ..."; cp -R "$DATA_SRC" "$SLURM_TMPDIR/ptb-xl"
torchrun --standalone --nproc_per_node=1 -m src.patchECG_pretrain \
    --dset_pretrain=ptb-xl \
    --root_path="$SLURM_TMPDIR/ptb-xl" \
    --custom_lead_selection="II,V2" \
    --model_name=patch_ecg_ptbxl_ii_v2 \
    --min_class_size=0 \
    --context_points=5000 --target_points=100 \
    --scaler=standard --features=M \
    --patch_len=50 --stride=50 --revin_mode=BN \
    --model=vanilla_vit --n_layers=6 --n_heads=8 --d_model=128 --d_ff=512 \
    --shared_embedding --dropout=0.0 --head_dropout=0.0 \
    --data_augmentation=none --trafos None \
    --mask_ratio=0.4 --zero_noise_mixing=0.5 \
    --opt=adamw --weight_decay=0.05 --lr=0.0015 --scheduler \
    --warmup_epochs=2 --warmup_lr=1e-06 --min_lr=1e-05 \
    --batch_size=128 --num_workers=4 --n_epochs_pretrain=500 \
    --save_every=20 --plot_every_n=1 --world_size=1 --local_node
