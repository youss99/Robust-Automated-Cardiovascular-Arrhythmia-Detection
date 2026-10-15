"""
Generate (and optionally submit) sbatch fine-tune scripts for the PTB-XL
pretrained arm of the cv5_optB thesis_small regression experiment.

This mirrors the existing Unified-pretrained arm exactly -- same optB fold
manifests, same 15-record pool, same per-fold seeds (200..204), same
thesis_small FT recipe (20 head + 70 full FT, lr 1e-3, layer_decay 0.65) --
changing ONLY the pretrain backbone from Unified to PTB-XL.

Why pretrained-only: the `supervised` arm is trained from scratch (no
--pretrained_model_path), so it is pretrain-agnostic. The existing supervised
checkpoints stay valid as the baseline and are NOT retrained here.

  3 lead modes (two_lead, lead_II, lead_V2) x 5 folds = 15 jobs.

Recipe is taken from the params.txt of the shipped Unified runs, NOT from
build_cv5_train_sbatches.py -- that older generator encodes a superseded
recipe (lr 3e-4, 5 head + 40 FT) and points at a pretrain dir that no longer
exists.

Usage:
    python build_cv5_ptbxl_train_sbatches.py                     # write only
    python build_cv5_ptbxl_train_sbatches.py --submit            # write+submit all
    python build_cv5_ptbxl_train_sbatches.py --submit --lead_modes two_lead lead_II
"""
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SCRATCH_DS_ROOT = "/scratch/youss99/ltstdb_datasets"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "cv5_ptbxl_training"
LOG_OUT = Path(REPO_ROOT) / "logs"

# Shipped Unified runs took ~19 min wall (20+70 epochs) on a100_3g.20gb.
# 1h gives ~3x margin while keeping backfill priority reasonable.
WALLTIME = "01:00:00"

PRETRAIN_ROOT = f"{REPO_ROOT}/results/pre-train/saved_models/ptb-xl"
PRETRAINED = {
    "two_lead": f"{PRETRAIN_ROOT}/patch_ecg_ptbxl_ii_v2/patch_ecg_ptbxl_ii_v2_best.pt",
    "lead_II":  f"{PRETRAIN_ROOT}/patch_ecg_ptbxl_ii/patch_ecg_ptbxl_ii_best.pt",
    "lead_V2":  f"{PRETRAIN_ROOT}/patch_ecg_ptbxl_v2/patch_ecg_ptbxl_v2_best.pt",
}

PRE_TAG = {
    "two_lead": "ptbxl_ii_v2",
    "lead_II":  "ptbxl_ii",
    "lead_V2":  "ptbxl_v2",
}

CUSTOM_LEADS = {
    "two_lead": "II,V2",
    "lead_II":  "II",
    "lead_V2":  "V2",
}

LEAD_SHORT = {"two_lead": "2L", "lead_II": "II", "lead_V2": "V2"}

# All three lead modes are trained on the optB pool (the dataset H5 carries
# both leads; lead choice is the --custom_lead_selection flag).
DESIGN = "optB"
DS_PREFIX = "cv5_optB_fold"
SEEDS = {1: 200, 2: 201, 3: 202, 4: 203, 5: 204}
SUFFIX = "thesis_small"


def render(lead_mode, fold):
    """Return (filename, body, model_name) for one sbatch script."""
    seed = SEEDS[fold]
    fold_s = f"{fold:02d}"
    dataset = f"{SCRATCH_DS_ROOT}/{DS_PREFIX}{fold_s}"
    pre_tag = PRE_TAG[lead_mode]
    design_tag = f"{DESIGN}_{lead_mode}"

    model_name = (f"ltstdb_median_cv5_{design_tag}_regression_"
                  f"pretrained_{pre_tag}_fold{fold_s}_{SUFFIX}")
    job_name = f"ptbxl_{LEAD_SHORT[lead_mode]}_pre_f{fold_s}"
    log_pattern = f"{LOG_OUT}/{model_name}-%j.out"

    body = f"""#!/usr/bin/env bash
#SBATCH --account=def-majidk
#SBATCH --job-name={job_name}
#SBATCH --time={WALLTIME}
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --output={log_pattern}
#SBATCH --error={log_pattern}
set -euo pipefail

cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"

# The learner RESUMES from {model_name}_snapshot.pt if one exists
# (src/core/learner.py:117). A resubmit into a dirty directory would therefore
# silently continue a previous run instead of training fresh -- and the metrics
# recorder is NOT restored, which crashes plot_recorders at the end. Clear the
# output dir so every submission is a clean run.
SAVE_DIR="{REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression/{model_name}"
if [ -d "${{SAVE_DIR}}" ]; then
  echo "Clearing existing output dir for a fresh run: ${{SAVE_DIR}}"
  rm -rf "${{SAVE_DIR}}"
fi

STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
echo "Staging fold {fold_s} from {dataset} to ${{STAGED_ROOT}} ..."
mkdir -p "${{STAGED_ROOT}}"
# -L: dereference hardlinks so the H5 is materialized on local scratch.
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
  --custom_lead_selection="{CUSTOM_LEADS[lead_mode]}" \\
  --n_epochs_finetune_head=20 \\
  --n_epochs_finetune=70 \\
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
  --pretrained_model_path="{PRETRAINED[lead_mode]}"
"""
    return f"{model_name}.sbatch", body, model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="sbatch the generated scripts")
    ap.add_argument("--lead_modes", nargs="+", default=["two_lead", "lead_II", "lead_V2"],
                    choices=["two_lead", "lead_II", "lead_V2"])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    SBATCH_OUT.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(REPO_ROOT) / "results/fine-tune/saved_models/patient-segment-regression"

    written, submitted, skipped = [], [], []
    for lead_mode in args.lead_modes:
        # Pre-flight: the pretrain backbone must exist before we submit.
        pre = Path(PRETRAINED[lead_mode])
        if args.submit and not pre.exists():
            print(f"SKIP {lead_mode}: pretrain backbone missing -> {pre}")
            skipped.append(f"{lead_mode} (no backbone)")
            continue
        for fold in args.folds:
            name, body, model_name = render(lead_mode, fold)
            path = SBATCH_OUT / name
            path.write_text(body)
            path.chmod(0o755)
            written.append(name)

            if not args.submit:
                continue
            if (ckpt_root / model_name / f"{model_name}_best.pt").exists():
                print(f"SKIP {model_name}: checkpoint already exists")
                skipped.append(model_name)
                continue
            r = subprocess.run(["sbatch", "--parsable", str(path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAILED {model_name}: {r.stderr.strip()}")
                continue
            jid = r.stdout.strip()
            print(f"SUBMITTED jobid={jid}  {model_name}")
            submitted.append(f"{jid}  {model_name}")

    print(f"\n===== Summary =====\nWrote {len(written)} sbatch scripts to {SBATCH_OUT}")
    if args.submit:
        print(f"Submitted: {len(submitted)}   Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
