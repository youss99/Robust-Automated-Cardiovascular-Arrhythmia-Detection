#!/usr/bin/env python3
"""Sbatch files for the final thesis runs (docs/instructions_final_results.md, steps 1, 2, 2b, 3; decisions in docs/DECISIONS.md). Nothing is submitted; this only writes files.

  step1   2 jobs  re-pretrain single-lead backbones with the fixed loss: patch_ecg_unified_{ii,v2}_lossfix, via pretrain_unified_v2.sh
                  unchanged (same recipe as the _rebuilt backbones). Pre-training seed: none (single unseeded run per backbone).
  step2  45 jobs  L1 (lead_II from ii_lossfix, lead_V2 from v2_lossfix) and L2 (two_lead from patch_ecg_unified_ii_v2_rebuilt) x 5 folds x S1, S2, S3.
                  The L1 jobs need the step 1 backbones (each job stops if its backbone is missing); L2's backbone already exists.
  step2b 10 jobs  OPTIONAL. L1-rebuilt: lead_II, lead_V2 from the _rebuilt backbones, new manifest, S1.
  step3  15 jobs  L1-sup (lead_II, lead_V2) and L2-sup (two_lead), supervised (no backbone), S1.
  step3o 30 jobs  OPTIONAL. The same at S2 and S3.

Seeds (fine-tuning): S1 = 200 + fold - 1 (the published seed, every R2 params.txt), S2 = 300 + fold - 1, S3 = 400 + fold - 1 (user, 2026-11-20).
Recipe (fine-tuning): R2's params.txt, the same flags as build_phase1_sbatches.py (20 head + 70 full epochs, lr 1e-3, layer decay 0.65,
weight decay 0.05, batch 128, MSE, valid_loss selection). dry_run_final.py checks every generated job through the real parser against R2.
Splits (docs/DECISIONS.md, 2026-11-20): one identical split for every learned model (lead_II, lead_V2, two_lead; pretrained and supervised):
  folds 1, 4, 5        phase1_folds/cv5_optB_leadV2_opt2p_fold* (the R2b folders): masked record s20081/s20081/s20241 unused for every model,
                       moved record s20231/s20071/s20141 in validation. No validation record has a masked lead.
  folds 2, 3           phase1_folds/cv5_optB_fold02, 03 (published)
The cv5_optB_leadII_opt2p_fold* folders (build_final_folds.py, an earlier per-lead split) stay in the repo, but no job uses them.
Every run has its own folder named by run ID (FIN_<pipeline>_<mode>_<seed>_fold<k>); a job stops at once if its data or backbone is missing,
or if the folder already holds a snapshot, so a resubmission cannot silently resume. job_log.txt in the run folder: run ID, git commit,
uncommitted files, seed, Slurm job, host, dataset + CSV checksum, backbone + checksum; after training, best epoch and validation loss.
The pre-training jobs log the same plus the first and last logged losses, and the new backbone's checksum.
Overrides: FIN_REPO_ROOT FIN_DS_ROOT FIN_ACCOUNT FIN_GPU FIN_PT_GPU FIN_WALLTIME FIN_PT_WALLTIME FIN_OUT.
Usage: python3 src/scripts/custom_dataset/build_final_sbatches.py   then   bash <FIN_OUT>/submit_final.sh <group>"""
import csv
import os
from pathlib import Path

REPO_ROOT = os.environ.get("FIN_REPO_ROOT", "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DS_ROOT = os.environ.get("FIN_DS_ROOT", f"{REPO_ROOT}/phase1_folds")
ACCOUNT = os.environ.get("FIN_ACCOUNT", "def-majidk")
GPU = os.environ.get("FIN_GPU", "--gres=gpu:a100_3g.20gb:1")
PT_GPU = os.environ.get("FIN_PT_GPU", "--gpus-per-node=a100_3g.20gb:1")      # as pretrain_unified_v2.sh
WALLTIME = os.environ.get("FIN_WALLTIME", "00:45:00")                        # R1b/R2 took 12-14 min single-lead, 19-20 min two-lead
PT_WALLTIME = os.environ.get("FIN_PT_WALLTIME", "02:30:00")                  # as pretrain_unified_v2.sh (_fixed single-lead: 50 min)
OUT = Path(os.environ.get("FIN_OUT", f"{REPO_ROOT}/logs/sbatch_scripts/final"))
UNI = f"{REPO_ROOT}/results/pre-train/saved_models/unified"
SAVE = f"{REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
PT_SCRIPT = "src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh"
LEADS = {"lead_II": "II", "lead_V2": "V2", "two_lead": "II,V2"}
SEED_BASE = {"S1": 200, "S2": 300, "S3": 400}
OPT2P = (1, 4, 5)
PRETRAIN = {"lead_II": ("II", "patch_ecg_unified_ii_lossfix"), "lead_V2": ("V2", "patch_ecg_unified_v2_lossfix")}
BACKBONE = {
    "L1": {"lead_II": "patch_ecg_unified_ii_lossfix", "lead_V2": "patch_ecg_unified_v2_lossfix"},
    "L2": {"two_lead": "patch_ecg_unified_ii_v2_rebuilt"},
    "L1rebuilt": {"lead_II": "patch_ecg_unified_ii_rebuilt", "lead_V2": "patch_ecg_unified_v2_rebuilt"},
    "L1sup": {"lead_II": None, "lead_V2": None},
    "L2sup": {"two_lead": None},
}


def seed(s, k):
    return SEED_BASE[s] + k - 1


def dataset(mode, k):
    """One identical split for every learned model: the folder depends on the fold only, never on `mode`."""
    return f"{DS_ROOT}/{'cv5_optB_leadV2_opt2p_fold' if k in OPT2P else 'cv5_optB_fold'}{k:02d}"


def backbone(pipe, mode):
    n = BACKBONE[pipe][mode]
    return f"{UNI}/{n}/{n}_best.pt" if n else ""


BEST_PY = """import glob
import pandas as pd
d = pd.read_csv(glob.glob("{d}/_training_metrics_gpu0_*.csv")[0])
i = d.valid_loss.idxmin()
print("best_row_epoch:", int(d.epoch[i]), "of", len(d), "rows (epoch restarts at 0 after the head phase if the file shows it)")
print("best_valid_loss:", float(d.valid_loss[i]))
{extra}"""
PT_EXTRA = """for c in ("train_loss", "valid_loss"):
    print(f"{c}: first {float(d[c].iloc[0]):.4f}, last {float(d[c].iloc[-1]):.4f}, min {float(d[c].min()):.4f} (expected: starts near 1 and falls)")"""


def render_finetune(pipe, mode, s, k):
    sd = seed(s, k)
    name = f"FIN_{pipe}_{mode}_{s}_fold{k:02d}"
    data = dataset(mode, k)
    bb = backbone(pipe, mode)
    log = f"{REPO_ROOT}/logs/final/{name}-%j.out"
    run_dir = f"{SAVE}/{name}"
    check = f'[[ -f "{bb}" ]] || {{ echo "ERROR: backbone missing: {bb}" >&2; exit 2; }}\n' if bb else ""
    pre = f' \\\n  --pretrained_model_path="{bb}"' if bb else ""
    bb_log = f'echo "backbone: {bb}"; echo "backbone_sha256: $(sha256sum "{bb}" | cut -d" " -f1)"' if bb else 'echo "backbone: none"'
    return name, f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={name}
#SBATCH --time={WALLTIME}
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH {GPU}
#SBATCH --output={log}
#SBATCH --error={log}
set -euo pipefail

[[ -f "{data}/PatientSegmentRegressionDataset.h5" ]] || {{ echo "ERROR: dataset or H5 missing: {data}" >&2; exit 2; }}
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
{check}if compgen -G "{run_dir}/*_snapshot.pt" > /dev/null; then echo "ERROR: {name} already has a snapshot; refusing to resume" >&2; exit 3; fi

mkdir -p "{run_dir}"
{{
  echo "run_id: {name}"
  echo "pipeline: {pipe}; mode: {mode}; training_seed: {s} = {sd}; fold: {k}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_uncommitted_files: $(git status --porcelain | wc -l)"
  echo "seed: {sd}"
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
  --seed={sd} \\
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
{BEST_PY.format(d=run_dir, extra="")}PYEOF
echo "finished: $(date +%F_%T)" >> "{run_dir}/job_log.txt"
"""


def render_pretrain(mode):
    leads, name = PRETRAIN[mode]
    out = f"{UNI}/{name}"
    log = f"{REPO_ROOT}/logs/final/{name}-%j.out"
    return name, f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH {PT_GPU}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time={PT_WALLTIME}
#SBATCH --output={log}
#SBATCH --error={log}
set -euo pipefail
cd "{REPO_ROOT}"
[[ -e "{out}" ]] && {{ echo "ERROR: {out} exists; never overwritten" >&2; exit 2; }}
HEAD_LOG="${{SLURM_TMPDIR}}/job_log_head.txt"
{{
  echo "run_id: {name}"
  echo "step: 1 (re-pretrain with the fixed loss); leads: {leads}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_uncommitted_files: $(git status --porcelain | wc -l)"
  echo "seed: none (single unseeded pre-training run)"
  echo "recipe: {PT_SCRIPT} sha256 $(sha256sum {PT_SCRIPT} | cut -d' ' -f1)"
  echo "slurm_job: ${{SLURM_JOB_ID:-none}}"
  echo "host: $(hostname)"
  echo "started: $(date +%F_%T)"
}} > "$HEAD_LOG"

# the recipe script unchanged (its own #SBATCH lines are ignored under bash; it stages the data and runs torchrun)
bash {PT_SCRIPT} {leads} {name}

cp "$HEAD_LOG" "{out}/job_log.txt"
{{
  echo "backbone: {out}/{name}_best.pt"
  echo "backbone_sha256: $(sha256sum "{out}/{name}_best.pt" | cut -d' ' -f1)"
}} >> "{out}/job_log.txt"
"{REPO_ROOT}/.venv/bin/python" - >> "{out}/job_log.txt" <<'PYEOF'
{BEST_PY.format(d=out, extra=PT_EXTRA + chr(10))}PYEOF
echo "finished: $(date +%F_%T)" >> "{out}/job_log.txt"
"""


SUBMIT = """#!/usr/bin/env bash
# Submit one group of the final runs. Each group is submitted once: a marker file stops a second submission.
# Groups: step1 step2_L2 step2_L1 step2b step3 step3o. Order (instructions): step1 and step3 after the approved dry run;
# step2_L2 after step 0b passes; step2_L1 after step 1's checks pass (loss near 1 and falling; backbone_masked_vs_visible.py).
set -euo pipefail
cd "$(dirname "$0")"
g="${1:?group}"
[[ -d "$g" ]] || { echo "unknown group: $g" >&2; exit 2; }
[[ -e "submitted_$g.tsv" ]] && { echo "ERROR: $g already submitted (submitted_$g.tsv)" >&2; exit 3; }
mkdir -p "%s/logs/final"
for f in "$g"/*.sbatch; do
  id=$(sbatch --parsable "$f")
  printf '%%s\\t%%s\\t%%s\\n' "$(basename "$f" .sbatch)" "$id" "$(date +%%F_%%T)" | tee -a "submitted_$g.tsv"
done
"""


def main():
    jobs = [("step1", "pretrain", m, None, None) for m in PRETRAIN]
    jobs += [("step2_L1", "L1", m, s, k) for m in ("lead_II", "lead_V2") for s in SEED_BASE for k in range(1, 6)]
    jobs += [("step2_L2", "L2", "two_lead", s, k) for s in SEED_BASE for k in range(1, 6)]
    jobs += [("step2b", "L1rebuilt", m, "S1", k) for m in ("lead_II", "lead_V2") for k in range(1, 6)]
    jobs += [("step3", p, m, "S1", k) for p, ms in (("L1sup", ("lead_II", "lead_V2")), ("L2sup", ("two_lead",))) for m in ms for k in range(1, 6)]
    jobs += [("step3o", p, m, s, k) for p, ms in (("L1sup", ("lead_II", "lead_V2")), ("L2sup", ("two_lead",))) for m in ms for s in ("S2", "S3") for k in range(1, 6)]
    rows = []
    for group, pipe, mode, s, k in jobs:
        d = OUT / group
        d.mkdir(parents=True, exist_ok=True)
        if pipe == "pretrain":
            name, body = render_pretrain(mode)
            rows.append(dict(group=group, pipeline=pipe, mode=mode, training_seed="none", seed="none", fold="", run_id=name,
                             dataset="Unified_Dataset_No_PTB_Segment_v2 (" + PRETRAIN[mode][0] + ")", backbone="(output)", optional=False))
        else:
            name, body = render_finetune(pipe, mode, s, k)
            rows.append(dict(group=group, pipeline=pipe, mode=mode, training_seed=s, seed=seed(s, k), fold=k, run_id=name,
                             dataset=dataset(mode, k), backbone=backbone(pipe, mode) or "none", optional=group in ("step2b", "step3o")))
        (d / f"{name}.sbatch").write_text(body)
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")   # LF, as every CSV in the repo (.gitattributes text=auto)
        w.writeheader()
        w.writerows(rows)
    (OUT / "submit_final.sh").write_text(SUBMIT % REPO_ROOT)
    counts = {g: sum(r["group"] == g for r in rows) for g in dict.fromkeys(r["group"] for r in rows)}
    assert counts == {"step1": 2, "step2_L1": 30, "step2_L2": 15, "step2b": 10, "step3": 15, "step3o": 30}, counts
    assert len({r["run_id"] for r in rows}) == len(rows)
    for k in range(1, 6):   # one identical split for every learned model
        assert len({r["dataset"] for r in rows if r["fold"] == k}) == 1, k
    assert not any("leadII_opt2p" in r["dataset"] for r in rows)
    print(f"{len(rows)} jobs written to {OUT}: {counts}. Nothing submitted.")


if __name__ == "__main__":
    main()
