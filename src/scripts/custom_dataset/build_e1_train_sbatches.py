"""
Generate sbatch training scripts for the E1 ST-regression runs
(GIT/ST_ARM_INSTRUCTIONS.md; dataset README_E1.md):

  lead_II  cv5_optB, 5 folds x 3 seed sets   (15 jobs)
  lead_V2  cv5_optB, 5 folds x 3 seed sets   (15 jobs)
  lead_V5  leave-one-record-out, 4 folds      (4 jobs)

Recipe = Test 1 / E0 Unified-pretrained arm, unchanged (params.txt of
ltstdb_beatcenter_test1_cv5_optB_lead_II_regression_pretrained_chapman_ii_only_fold01):
MSE, 20 head + 70 full epochs, lr 1e-3, layer decay 0.65, batch 128, valid_loss.
Seeds follow the existing per-fold pattern: set 200 -> 200..204, set 300 -> 300..304, ...

The folds are already applied in the dataset (strat_fold_annotated), so no
prepare_fold_dataset.py step. Nothing is submitted; this script only writes files.
"""
import os
from pathlib import Path

# Nibi layout (code, venv and backbones under GIT/<repo>; data under GIT/datasets). Override with env vars.
REPO_ROOT = os.environ.get("E1_REPO_ROOT", "/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SCRATCH_DS_ROOT = os.environ.get("E1_DS_ROOT", "/home/youss99/scratch/GIT/datasets/ltstdb_regression_e1_nkcentred_balanced_100hz")
ACCOUNT = os.environ.get("E1_ACCOUNT", "def-majidk")
GPU = os.environ.get("E1_GPU", "--gpus=h100_3g.40gb:1")    # Narval used --gres=gpu:a100_3g.20gb:1
MODULES = "StdEnv/2023 python/3.11"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "e1_training"
LOG_OUT = Path(REPO_ROOT) / "logs"

# 90 epochs over ~86.5k train + ~60-80k validation windows: about 2x E0's per-epoch
# work and 2.25x its epochs (E0: 10-16 min for 40 epochs on a100_3g.20gb).
WALLTIME = "03:00:00"

# Same backbones as Test 1 / E0 (the "Chapman-pretrained" tag is historical: the files are Unified).
# V5 has no E0 counterpart: point it at whichever Unified V5 backbone exists on the cluster.
PRETRAINED = {
    "lead_II": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_ii_only/patch_ecg_unified_ii_only_best.pt",
    "lead_V2": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_v2_only/patch_ecg_unified_v2_only_best.pt",
    "lead_V5": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_v5_fixed/patch_ecg_unified_v5_fixed_best.pt",
}
PRE_TAG = {"lead_II": "unified_ii_only", "lead_V2": "unified_v2_only", "lead_V5": "unified_v5_fixed"}
CUSTOM_LEADS = {"lead_II": "II", "lead_V2": "V2", "lead_V5": "lead_0"}

FOLDS = {
    "lead_II": [f"fold{f:02d}" for f in range(1, 6)],
    "lead_V2": [f"fold{f:02d}" for f in range(1, 6)],
    "lead_V5": [f"fold_test_{r}" for r in ("s20571", "s20581", "s30671", "s30681")],
}
SEED_SETS = {"lead_II": (200, 300, 400), "lead_V2": (200, 300, 400), "lead_V5": (200,)}


def render(lead, fold_idx, fold, seed_set, smoke=False):
    seed = seed_set + fold_idx
    model_name = f"ltstdb_e1_{lead}_regression_pretrained_{PRE_TAG[lead]}_{fold}_s{seed_set}_thesis_small"
    job_name = f"e1_{lead[5:]}_{fold_idx + 1}_s{seed_set}"
    epochs_head, epochs_full, walltime = (20, 70, WALLTIME) if not smoke else (0, 1, "00:30:00")
    if smoke:
        model_name, job_name = f"SMOKE_{model_name}", "e1_smoke"
    dataset = f"{SCRATCH_DS_ROOT}/{lead}/{fold}"
    log_pattern = f"{LOG_OUT}/{model_name}-%j.out"
    body = f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH {GPU}
#SBATCH --output={log_pattern}
#SBATCH --error={log_pattern}
set -euo pipefail

module load {MODULES}
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"

STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
echo "Staging {lead}/{fold} from {dataset} to ${{STAGED_ROOT}} ..."
mkdir -p "${{STAGED_ROOT}}"
# -L: the fold's H5 is a symlink to ../PatientSegmentRegressionDataset.h5; materialise it locally.
cp -rL "{dataset}/." "${{STAGED_ROOT}}/"
echo "Staged $(du -sh ${{STAGED_ROOT}} | awk '{{print $1}}')"

"{PY_BIN}" -m src.patchECG_finetune \\
  --seed={seed} \\
  --lr=0.001 \\
  --save_every=20 \\
  --root_path="${{STAGED_ROOT}}" \\
  --num_workers=4 \\
  --batch_size=128 \\
  --data_augmentation=none \\
  --trafos=None \\
  --class_token=cls_token \\
  --shared_embedding \\
  --min_class_size=1 \\
  --layer_decay=0.65 \\
  --weight_decay=0.05 \\
  --head_dropout=0.0 \\
  --dropout=0.0 \\
  --dset_finetune=patient-segment-regression \\
  --classification_type=REGRESSION \\
  --head_type=regression \\
  --regression_loss=mse \\
  --no-focal_loss \\
  --model=vanilla_vit \\
  --d_model=128 \\
  --no-scheduler \\
  --custom_lead_selection="{CUSTOM_LEADS[lead]}" \\
  --n_epochs_finetune_head={epochs_head} \\
  --n_epochs_finetune={epochs_full} \\
  --n_heads=8 \\
  --n_layers=6 \\
  --d_ff=512 \\
  --patch_len=50 \\
  --stride=50 \\
  --model_name="{model_name}" \\
  --metric_by_class \\
  --no-mentor_mix \\
  --no-bootstrapping \\
  --iterations=5000 \\
  --model_selection_metric=valid_loss \\
  --pretrained_model_path="{PRETRAINED[lead]}"
"""
    return f"{model_name}.sbatch", body


def main():
    SBATCH_OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for lead, folds in FOLDS.items():
        (SBATCH_OUT / lead).mkdir(exist_ok=True)
        for seed_set in SEED_SETS[lead]:
            for i, fold in enumerate(folds):
                name, body = render(lead, i, fold, seed_set)
                (SBATCH_OUT / lead / name).write_text(body)
                names.append(f"{lead}/{name}")
    # seed set 200 first, so the deployment models finish first
    names.sort(key=lambda n: ("_s200_" not in n, n))
    submit = SBATCH_OUT / "submit_all.sh"
    submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                      + "".join(f"sbatch {n}\n" for n in names))
    submit.chmod(0o755)
    # one 1-epoch job (V5, smallest) to check the environment, GPU request and backbone before submit_all.sh
    _, body = render("lead_V5", 1, FOLDS["lead_V5"][1], 200, smoke=True)
    (SBATCH_OUT / "smoke.sbatch").write_text(body)
    print(f"Wrote {len(names)} sbatch files + submit_all.sh + smoke.sbatch under {SBATCH_OUT}")


if __name__ == "__main__":
    main()
