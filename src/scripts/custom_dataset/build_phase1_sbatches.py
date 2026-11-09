#!/usr/bin/env python3
"""Sbatch files for the Phase 1 runs in README_phase1_runs.md. Nothing is submitted; this only writes files.

  R1b  30 jobs  the paper setup with the old backbones on the new cluster: lead_II, lead_V2, two_lead x pretrained, supervised x 5 folds
  R2   15 jobs  the rebuilt backbones: lead_II, lead_V2, two_lead, pretrained x 5 folds (supervised reused from R1b: those runs load no weights)
  R2b   3 jobs  option 2+ validation: lead_V2, pretrained on the rebuilt backbone, folds 1, 4, 5
  R3   10 jobs  beat-centred v2 data: lead_II, lead_V2, pretrained on the rebuilt backbone x 5 folds; V2 uses the option 2+ folds for 1, 4, 5. Written only with --with_r3, once the v2 data exist.

Recipe: the published runs' params.txt (MSE, 20 head + 70 full epochs, lr 1e-3, layer decay 0.65, weight decay 0.05, batch 128, valid_loss selection, seed 200 + fold - 1).
Do not take the recipe from build_cv5_train_sbatches.py: its template (40 epochs) is not what the published params.txt recorded.
Data: phase1_folds/ in this repo (build_phase1_folds.py; run link_phase1_h5.sh once on each machine). Every run has its own folder, named by run ID, for example R2_lead_V2_pretrained_fold01.

Each job stops at once if the data or backbone is missing, or if its folder already holds a snapshot, so a resubmission cannot silently resume.
Each job writes <run folder>/job_log.txt: run ID, git commit, whether the tree was clean, seed, Slurm job, host, dataset and CSV checksum, backbone path and checksum; after training, the best epoch and its validation loss. The resolved config is params.txt in the same folder.
Overrides: P1_REPO_ROOT P1_DS_ROOT P1_DS_R3 P1_ACCOUNT P1_GPU P1_WALLTIME P1_OUT.
Usage: python3 src/scripts/custom_dataset/build_phase1_sbatches.py [--with_r3]"""
import argparse
import csv
import os
from pathlib import Path

REPO_ROOT = os.environ.get("P1_REPO_ROOT", "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DS_ROOT = os.environ.get("P1_DS_ROOT", f"{REPO_ROOT}/phase1_folds")
DS_R3 = os.environ.get("P1_DS_R3", "PLACEHOLDER/ltstdb_regression_v2_beatcentred")
ACCOUNT = os.environ.get("P1_ACCOUNT", "def-majidk")
GPU = os.environ.get("P1_GPU", "--gres=gpu:a100_3g.20gb:1")
WALLTIME = os.environ.get("P1_WALLTIME", "00:45:00")
OUT = Path(os.environ.get("P1_OUT", f"{REPO_ROOT}/logs/sbatch_scripts/phase1"))
UNI = f"{REPO_ROOT}/results/pre-train/saved_models/unified"
SAVE = f"{REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
LEADS = {"lead_II": "II", "lead_V2": "V2", "two_lead": "II,V2"}
OLD = {"lead_II": "patch_ecg_unified_ii_only", "lead_V2": "patch_ecg_unified_v2_only", "two_lead": "patch_ecg_unified_ii_v2"}
NEW = {"lead_II": "patch_ecg_unified_ii_rebuilt", "lead_V2": "patch_ecg_unified_v2_rebuilt", "two_lead": "patch_ecg_unified_ii_v2_rebuilt"}
OPT2P_FOLDS = (1, 4, 5)


def backbone(run, mode):
    n = (OLD if run == "R1b" else NEW)[mode]
    return f"{UNI}/{n}/{n}_best.pt"


def dataset(run, mode, k):
    if run == "R3":
        return f"{DS_R3}/{'leadV2_opt2p_fold' if mode == 'lead_V2' and k in OPT2P_FOLDS else 'fold'}{k:02d}"
    return f"{DS_ROOT}/{'cv5_optB_leadV2_opt2p_fold' if run == 'R2b' else 'cv5_optB_fold'}{k:02d}"


def render(run, mode, arm, k):
    seed = 200 + k - 1
    name = f"{run}_{mode}_{arm}_fold{k:02d}"
    data = dataset(run, mode, k)
    bb = backbone(run, mode) if arm == "pretrained" else ""
    log = f"{REPO_ROOT}/logs/{name}-%j.out"
    run_dir = f"{SAVE}/{name}"
    check = f'[[ -f "{bb}" ]] || {{ echo "ERROR: backbone missing: {bb}" >&2; exit 2; }}\n' if bb else ""
    pre = f' \\\n  --pretrained_model_path="{bb}"' if bb else ""
    bb_log = f'echo "backbone: {bb}"; echo "backbone_sha256: $(sha256sum "{bb}" | cut -d" " -f1)"' if bb else 'echo "backbone: none"'
    return name, f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={run}_{mode[:7]}_{arm[:3]}_{k}
#SBATCH --time={WALLTIME}
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH {GPU}
#SBATCH --output={log}
#SBATCH --error={log}
set -euo pipefail

case "{data}" in PLACEHOLDER*) echo "ERROR: set P1_DS_R3 and regenerate: {data}" >&2; exit 2;; esac
[[ -f "{data}/PatientSegmentRegressionDataset.h5" ]] || {{ echo "ERROR: dataset or H5 missing: {data}" >&2; exit 2; }}
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
{check}if compgen -G "{run_dir}/*_snapshot.pt" > /dev/null; then echo "ERROR: {name} already has a snapshot; refusing to resume" >&2; exit 3; fi

mkdir -p "{run_dir}"
{{
  echo "run_id: {name}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_uncommitted_files: $(git status --porcelain | wc -l)"
  echo "seed: {seed}"
  echo "slurm_job: ${{SLURM_JOB_ID:-none}}"
  echo "host: $(hostname)"
  echo "started: $(date +%F_%T)"
  echo "dataset: {data}"
  echo "dataset_csv_sha256: $(sha256sum "{data}/modified_dataset/modified_dataset.csv" | cut -d" " -f1)"
  {bb_log}
}} > "{run_dir}/job_log.txt"

STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
mkdir -p "${{STAGED_ROOT}}"
cp -rL "{data}/." "${{STAGED_ROOT}}/"

"{REPO_ROOT}/.venv/bin/python" -m src.patchECG_finetune \\
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
  --custom_lead_selection="{LEADS[mode]}" \\
  --n_epochs_finetune_head=20 \\
  --n_epochs_finetune=70 \\
  --n_heads=8 \\
  --n_layers=6 \\
  --d_ff=512 \\
  --patch_len=50 \\
  --stride=50 \\
  --model_name="{name}" \\
  --metric_by_class \\
  --no-mentor_mix \\
  --no-bootstrapping \\
  --iterations=5000 \\
  --model_selection_metric=valid_loss{pre}

"{REPO_ROOT}/.venv/bin/python" - >> "{run_dir}/job_log.txt" <<'PYEOF'
import glob
import pandas as pd
d = pd.read_csv(glob.glob("{run_dir}/_training_metrics_gpu0_*.csv")[0])
i = d.valid_loss.idxmin()
print("best_row_epoch:", int(d.epoch[i]), "of", len(d), "rows (epoch restarts at 0 after the head phase if the file shows it)")
print("best_valid_loss:", float(d.valid_loss[i]))
PYEOF
echo "finished: $(date +%F_%T)" >> "{run_dir}/job_log.txt"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with_r3", action="store_true", help="also write R3 (needs the v2 data)")
    a = ap.parse_args()
    jobs = [("R1b", m, arm, k) for m in LEADS for arm in ("pretrained", "supervised") for k in range(1, 6)]
    jobs += [("R2", m, "pretrained", k) for m in LEADS for k in range(1, 6)]
    jobs += [("R2b", "lead_V2", "pretrained", k) for k in OPT2P_FOLDS]
    if a.with_r3:
        jobs += [("R3", m, "pretrained", k) for m in ("lead_II", "lead_V2") for k in range(1, 6)]
    rows = []
    for run, mode, arm, k in jobs:
        d = OUT / run
        d.mkdir(parents=True, exist_ok=True)
        name, body = render(run, mode, arm, k)
        (d / f"{name}.sbatch").write_text(body)
        rows.append(dict(run=run, mode=mode, arm=arm, fold=k, seed=200 + k - 1, run_id=name, dataset=dataset(run, mode, k), backbone=backbone(run, mode) if arm == "pretrained" else "none"))
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {r: sum(x["run"] == r for x in rows) for r in dict.fromkeys(x["run"] for x in rows)}
    assert {k: v for k, v in counts.items() if k != "R3"} == {"R1b": 30, "R2": 15, "R2b": 3} and counts.get("R3", 10) == 10, counts
    print(f"{len(rows)} jobs written to {OUT}: {counts}. Nothing submitted.")


if __name__ == "__main__":
    main()
