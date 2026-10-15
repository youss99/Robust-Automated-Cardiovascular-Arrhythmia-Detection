"""
Generate sbatch training scripts for the 5-fold CV experiment across three
fold designs:

  optB     two-lead pool (15 records, s20121 pinned to train)
  lead_II  single-lead II pool (14 records, s20091+s20121 pinned)
  lead_V2  single-lead V2 pool (12 records, 4 records pinned)

For each design we emit BOTH supervised and pretrained scripts across all 5
folds. Additionally, the optB folds are used to train all three lead modes
(two_lead, lead_II only, lead_V2 only) since the dataset H5 contains both
leads — the lead choice is a `--custom_lead_selection` flag on the trainer.

The `lead_II` and `lead_V2` *native* designs use their own (smaller) record
pools and pin only the records where their lead of interest is omitted.

Total = 3 (optB lead modes) * 2 (modes) * 5 (folds)
      + 2 (single-lead native designs) * 2 (modes) * 5 (folds)
      = 50 sbatch scripts.

Nothing is submitted; this script only writes files.
"""
from pathlib import Path

REPO_ROOT = "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SCRATCH_DS_ROOT = "/scratch/youss99/ltstdb_datasets"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "cv5_training"
LOG_OUT = Path(REPO_ROOT) / "logs"

# Observed per-job runtime on a100_3g.20gb: 10-16 min. Setting walltime to
# 45 min gives ~3x safety margin and dramatically improves backfill priority
# vs the previous 3h request.
WALLTIME = "00:45:00"

PRETRAINED = {
    "two_lead": f"{REPO_ROOT}/results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt",
    "lead_II":  f"{REPO_ROOT}/results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_only/patch_ecg_chapman_ii_only_best.pt",
    "lead_V2":  f"{REPO_ROOT}/results/pre-train/saved_models/chapman/patch_ecg_chapman_v2_only/patch_ecg_chapman_v2_only_best.pt",
}

PRE_TAG = {
    "two_lead": "chapman_ii_v2",
    "lead_II":  "chapman_ii_only",
    "lead_V2":  "chapman_v2_only",
}

CUSTOM_LEADS = {
    "two_lead": "II,V2",
    "lead_II":  "II",
    "lead_V2":  "V2",
}

# Short tag used inside SLURM --job-name (max ~17 chars after prefix).
LEAD_SHORT = {"two_lead": "2L", "lead_II": "II", "lead_V2": "V2"}

# Each entry = (design_tag, dataset_dir_prefix, lead_modes_trained_on_this_pool)
CONFIGS = [
    # optB pool trains all three lead modes
    ("optB",    "cv5_optB_fold",    ["two_lead", "lead_II", "lead_V2"]),
    # Native single-lead pools train only their own lead
    ("lead_II", "cv5_lead_II_fold", ["lead_II"]),
    ("lead_V2", "cv5_lead_V2_fold", ["lead_V2"]),
]

# Per-fold seed pattern, mirrors the existing existing-split scripts (200..204).
SEEDS = {1: 200, 2: 201, 3: 202, 4: 203, 5: 204}


def render(design, lead_mode, mode, fold, ds_dir_prefix):
    """Return (filename, file_body) for one sbatch script."""
    seed = SEEDS[fold]
    fold_s = f"{fold:02d}"
    dataset = f"{SCRATCH_DS_ROOT}/{ds_dir_prefix}{fold_s}"

    pre_tag = PRE_TAG[lead_mode]
    lead_short = LEAD_SHORT[lead_mode]

    # Avoid stuttering like "cv5_lead_II_lead_II_..." when the design tag and
    # the lead_mode are the same (single-lead native designs).
    design_tag = design if design == lead_mode else f"{design}_{lead_mode}"

    if mode == "supervised":
        model_name = f"ltstdb_median_cv5_{design_tag}_regression_supervised_fold{fold_s}"
        lr = "0.001"
        head_warmup = "0"
        pretrained_line = ""
        mode_short = "sup"
    else:
        model_name = f"ltstdb_median_cv5_{design_tag}_regression_pretrained_{pre_tag}_fold{fold_s}"
        lr = "0.0003"
        head_warmup = "5"
        pretrained_line = " \\\n  --pretrained_model_path=\"" + PRETRAINED[lead_mode] + "\""
        mode_short = "pre"

    job_name = f"ltst_cv5_{design[:4]}_{lead_short}_{mode_short}_f{fold_s}"
    sbatch_name = f"{model_name}.sbatch"
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

STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
echo "Staging fold {fold_s} from {dataset} to ${{STAGED_ROOT}} ..."
mkdir -p "${{STAGED_ROOT}}"
# -L: dereference hardlinks so the H5 is materialized on local scratch.
cp -rL "{dataset}/." "${{STAGED_ROOT}}/"
echo "Staged $(du -sh ${{STAGED_ROOT}} | awk '{{print $1}}')"

"{PY_BIN}" -m src.patchECG_finetune \\
  --seed={seed} \\
  --lr={lr} \\
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
  --n_epochs_finetune_head={head_warmup} \\
  --n_epochs_finetune=40 \\
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
  --model_selection_metric=valid_loss{pretrained_line}
"""
    return sbatch_name, body


def main():
    SBATCH_OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for design, ds_prefix, lead_modes in CONFIGS:
        design_dir = SBATCH_OUT / f"cv5_{design}"
        design_dir.mkdir(parents=True, exist_ok=True)
        for lead_mode in lead_modes:
            # For designs with a single lead mode (the native lead_II / lead_V2
            # pools), keep the design dir flat. For designs that train multiple
            # lead modes off the same pool (optB), split into per-lead sub-dirs.
            if len(lead_modes) > 1:
                target_dir = design_dir / lead_mode
                target_dir.mkdir(parents=True, exist_ok=True)
                rel_prefix = f"cv5_{design}/{lead_mode}"
            else:
                target_dir = design_dir
                rel_prefix = f"cv5_{design}"
            for mode in ("supervised", "pretrained"):
                for fold in range(1, 6):
                    name, body = render(design, lead_mode, mode, fold, ds_prefix)
                    path = target_dir / name
                    path.write_text(body)
                    written.append(f"{rel_prefix}/{name}")
    print(f"Wrote {len(written)} sbatch files under {SBATCH_OUT}")
    by_subdir = {}
    for rel in written:
        sd = rel.rsplit("/", 1)[0]
        by_subdir.setdefault(sd, 0)
        by_subdir[sd] += 1
    for sd in sorted(by_subdir):
        print(f"  {sd}/  ({by_subdir[sd]} scripts)")


if __name__ == "__main__":
    main()
