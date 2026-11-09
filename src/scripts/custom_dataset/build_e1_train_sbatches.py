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

Runs are bundled (one run is ~10 min, below the Alliance 15-min guideline for a job):
II / V2 get one job per fold running seed sets 200, 300, 400 back to back on one staged copy
(10 jobs); V5 gets one job running its 4 folds. Each run still logs to logs/<model_name>-<jobid>.out.
"""
import os
from pathlib import Path

# Nibi layout (code, venv and backbones under GIT/<repo>; data under GIT/datasets). Override with env vars.
REPO_ROOT = os.environ.get("E1_REPO_ROOT", "/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SCRATCH_DS_ROOT = os.environ.get("E1_DS_ROOT", "/home/youss99/scratch/GIT/datasets/ltstdb_regression_e1_nkcentred_balanced_100hz")
ACCOUNT = os.environ.get("E1_ACCOUNT", "def-majidk")
# Narval used --gres=gpu:a100_3g.20gb:1. On Nibi the 2g.20gb slice (same 20 GB) pretrained V2 in 40 min vs ~51 min on Narval.
GPU = os.environ.get("E1_GPU", "--gpus=h100_2g.20gb:1")
MODULES = "StdEnv/2023 python/3.11"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "e1_training"
LOG_OUT = Path(REPO_ROOT) / "logs"

# Smoke runs on h100_2g.20gb: ~6 s/epoch for II / V2, ~1 s/epoch for V5, so a 90-epoch run is ~10 min
# (II / V2) or ~2 min (V5) plus staging and test. Limits leave ~3x headroom over the bundled estimate.
WALLTIME = {"lead_II": "01:30:00", "lead_V2": "01:30:00", "lead_V5": "00:45:00"}

# Unified backbones pretrained on Unified_Dataset_No_PTB_Segment_fixed. Test 1 / E0 used ii_only / v2_only,
# which were pretrained on the old Unified H5 whose CinC rows (~31%) were all-zero, so E1 is not a same-backbone
# comparison with E0.
PRETRAINED = {
    "lead_II": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_ii_fixed/patch_ecg_unified_ii_fixed_best.pt",
    "lead_V2": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_v2_fixed/patch_ecg_unified_v2_fixed_best.pt",
    "lead_V5": f"{REPO_ROOT}/results/pre-train/saved_models/unified/patch_ecg_unified_v5_fixed/patch_ecg_unified_v5_fixed_best.pt",
}
PRE_TAG = {"lead_II": "unified_ii_fixed", "lead_V2": "unified_v2_fixed", "lead_V5": "unified_v5_fixed"}
CUSTOM_LEADS = {"lead_II": "II", "lead_V2": "V2", "lead_V5": "lead_0"}

FOLDS = {
    "lead_II": [f"fold{f:02d}" for f in range(1, 6)],
    "lead_V2": [f"fold{f:02d}" for f in range(1, 6)],
    "lead_V5": [f"fold_test_{r}" for r in ("s20571", "s20581", "s30671", "s30681")],
}
SEED_SETS = {"lead_II": (200, 300, 400), "lead_V2": (200, 300, 400), "lead_V5": (200,)}


def stage(lead, fold):
    dataset = f"{SCRATCH_DS_ROOT}/{lead}/{fold}"
    return f"""
echo "Staging {lead}/{fold} from {dataset} to ${{STAGED_ROOT}} ..."
rm -rf "${{STAGED_ROOT}}"; mkdir -p "${{STAGED_ROOT}}"
# -L: the fold's H5 is a symlink to ../PatientSegmentRegressionDataset.h5; materialise it locally.
cp -rL "{dataset}/." "${{STAGED_ROOT}}/"
# The source dataset is read-only; the loaders may rewrite modified_dataset.csv in the staged copy.
chmod -R u+w "${{STAGED_ROOT}}"
echo "Staged $(du -sh ${{STAGED_ROOT}} | awk '{{print $1}}')"
"""


def train(lead, fold_idx, fold, seed_set, smoke=False):
    seed = seed_set + fold_idx
    model_name = f"ltstdb_e1_{lead}_regression_pretrained_{PRE_TAG[lead]}_{fold}_s{seed_set}_thesis_small"
    epochs_head, epochs_full = (20, 70) if not smoke else (0, 1)
    if smoke:
        model_name = f"SMOKE_{model_name}"
    log = f"{LOG_OUT}/{model_name}-${{SLURM_JOB_ID}}.out"
    return f"""
echo "$(date +%T) training {model_name} -> {log}"
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
  --pretrained_model_path="{PRETRAINED[lead]}" > "{log}" 2>&1
"""


def render(job_name, walltime, runs, smoke=False):
    """One sbatch job running `runs` = [(lead, fold_idx, fold, seed_set), ...] in order; consecutive runs on
    the same fold share one staged copy."""
    body = f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH {GPU}
#SBATCH --output={LOG_OUT}/{job_name}-%j.out
set -euo pipefail

module load {MODULES}
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
"""
    staged = None
    for lead, fold_idx, fold, seed_set in runs:
        if (lead, fold) != staged:
            body += stage(lead, fold)
            staged = (lead, fold)
        body += train(lead, fold_idx, fold, seed_set, smoke)
    return f"{job_name}.sbatch", body + 'echo "$(date +%T) all runs done"\n'


def main():
    SBATCH_OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for lead in ("lead_II", "lead_V2"):
        # seed set 200 first within each fold job, so the deployment models finish first
        for i, fold in enumerate(FOLDS[lead]):
            jobs.append((lead, f"e1_{lead[5:]}_{i + 1}", [(lead, i, fold, s) for s in SEED_SETS[lead]]))
    jobs.append(("lead_V5", "e1_V5_all", [("lead_V5", i, fold, s) for s in SEED_SETS["lead_V5"]
                                          for i, fold in enumerate(FOLDS["lead_V5"])]))
    names = []
    for lead, job_name, runs in jobs:
        (SBATCH_OUT / lead).mkdir(exist_ok=True)
        name, body = render(job_name, WALLTIME[lead], runs)
        (SBATCH_OUT / lead / name).write_text(body)
        names.append(f"{lead}/{name}")
    submit = SBATCH_OUT / "submit_all.sh"
    submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                      + "".join(f"sbatch {n}\n" for n in names))
    submit.chmod(0o755)
    # one 1-epoch job per lead to check the environment, GPU request and each backbone before submit_all.sh;
    # its log must contain "successfully transferred!" (a missing backbone path silently trains from scratch)
    for lead in FOLDS:
        _, body = render(f"e1_smoke_{lead[5:]}", "00:30:00", [(lead, 0, FOLDS[lead][0], 200)], smoke=True)
        (SBATCH_OUT / f"smoke_{lead}.sbatch").write_text(body)
    n_runs = sum(len(r) for _, _, r in jobs)
    print(f"Wrote {len(names)} sbatch jobs ({n_runs} runs) + submit_all.sh + smoke_<lead>.sbatch x3 under {SBATCH_OUT}")


if __name__ == "__main__":
    main()
