"""
Generate sbatch training scripts for Test 1-NK (2026-10-29): Test 1's segments re-centred on the NeuroKit R-peak
(dataset README_TEST1NK.md). Unified-pretrained arm only, cv5_optB, 5 folds per mode:

  lead_II   backbone patch_ecg_unified_ii_fixed  --custom_lead_selection II
  lead_V2   backbone patch_ecg_unified_v2_fixed  --custom_lead_selection V2
  two_lead  backbone patch_ecg_unified_ii_v2     --custom_lead_selection II,V2   (optional; backbone not on Nibi yet)

Recipe = Test 1 (params.txt of ltstdb_beatcenter_test1_cv5_optB_lead_II_regression_pretrained_chapman_ii_only_fold01
diffed against the template: only name / backbone / paths differ): MSE, 20 head + 70 full epochs, lr 1e-3, layer decay 0.65,
weight decay 0.05, batch 128, RevIN default (BN), valid_loss selection, seed 200 + fold index (200..204), as Test 1.
Backbones differ from Test 1: lead_II / lead_V2 use the Unified backbones pretrained on Unified_Dataset_No_PTB_Segment_fixed
(as E1), not Test 1's ii_only / v2_only (pretrained on the old Unified H5 whose CinC rows were all-zero). So a Test 1 vs
Test 1-NK difference mixes centring and backbone. Run names carry the backbone tag (pretrained_unified_ii_fixed, ...)
with the prefix ltstdb_beatcenter_test1nk_; evaluation code keyed on Test 1's chapman_* tags needs the new tags.

Runs are bundled (one run is ~10 min, below the Alliance 15-min guideline for a job): one job per fold stages the fold
once and trains lead_II then lead_V2 (both leads are in the same H5). Each run logs to logs/<model_name>-<jobid>.out.
The folds are already applied per fold directory (strat_fold_annotated). Nothing is submitted; this only writes files.
"""
import os
from pathlib import Path

REPO_ROOT = os.environ.get("T1NK_REPO_ROOT", "/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DS_ROOT = os.environ.get("T1NK_DS_ROOT", "/home/youss99/scratch/GIT/datasets/ltstdb_regression_beatcenter_100hz_test1nk_nocompress")
ACCOUNT = os.environ.get("T1NK_ACCOUNT", "def-majidk")
GPU = os.environ.get("T1NK_GPU", "--gpus=h100_2g.20gb:1")  # E1 ran on this slice at ~6 s/epoch
MODULES = "StdEnv/2023 python/3.11"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "test1nk_training"
LOG_OUT = Path(REPO_ROOT) / "logs"
# ~10 min per 90-epoch run on h100_2g.20gb (E1: 6 s/epoch with 4-5x the validation set); ~3x headroom.
WALLTIME = {"single_lead": "01:00:00", "two_lead": "01:30:00", "smoke": "00:30:00"}
UNI = f"{REPO_ROOT}/results/pre-train/saved_models/unified"
MODES = {  # mode: (backbone file, run-name tag, lead selection)
    "lead_II": (f"{UNI}/patch_ecg_unified_ii_fixed/patch_ecg_unified_ii_fixed_best.pt", "pretrained_unified_ii_fixed", "II"),
    "lead_V2": (f"{UNI}/patch_ecg_unified_v2_fixed/patch_ecg_unified_v2_fixed_best.pt", "pretrained_unified_v2_fixed", "V2"),
    # Test 1's two-lead backbone: not on Nibi yet, so the job stops at the backbone check until it is copied over.
    "two_lead": (f"{UNI}/patch_ecg_unified_ii_v2/patch_ecg_unified_ii_v2_best.pt", "pretrained_chapman_ii_v2", "II,V2"),
}


def stage(k):
    dataset = f"{DS_ROOT}/fold{k:02d}"
    return f"""
echo "Staging fold{k:02d} from {dataset} to ${{STAGED_ROOT}} ..."
rm -rf "${{STAGED_ROOT}}"; mkdir -p "${{STAGED_ROOT}}"
# -L: the fold's H5 is a symlink to ../PatientSegmentRegressionDataset.h5; materialise it locally.
# The copy stays read-only like the source: the regression loader only writes modified_dataset.csv when the
# fold split is missing, and then it should fail rather than silently re-split.
cp -rL "{dataset}/." "${{STAGED_ROOT}}/"
echo "Staged $(du -sh ${{STAGED_ROOT}} | awk '{{print $1}}')"
"""


def train(mode, k, smoke=False):
    backbone, tag, leads = MODES[mode]; seed = 200 + k - 1
    model_name = f"ltstdb_beatcenter_test1nk_cv5_optB_{mode}_regression_{tag}_fold{k:02d}_thesis_small"
    head, full = (20, 70) if not smoke else (0, 1)
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
  --custom_lead_selection="{leads}" \\
  --n_epochs_finetune_head={head} \\
  --n_epochs_finetune={full} \\
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
  --pretrained_model_path="{backbone}" > "{log}" 2>&1
"""


def render(job_name, walltime, runs, smoke=False):
    """One sbatch job running `runs` = [(mode, fold), ...] in order; consecutive runs on a fold share one staged copy."""
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
    for backbone in dict.fromkeys(MODES[mode][0] for mode, _ in runs):
        body += f'[[ -f "{backbone}" ]] || {{ echo "ERROR: backbone missing: {backbone}" >&2; exit 2; }}\n'
    staged = None
    for mode, k in runs:
        if k != staged:
            body += stage(k)
            staged = k
        body += train(mode, k, smoke)
    return f"{job_name}.sbatch", body + 'echo "$(date +%T) all runs done"\n'


def main():
    SBATCH_OUT.mkdir(parents=True, exist_ok=True)
    jobs = {  # submit script: [(job_name, walltime, runs, smoke), ...]
        "submit_smoke.sh": [("t1nk_smoke", WALLTIME["smoke"], [("lead_II", 1), ("lead_V2", 1)], True)],
        "submit_single_lead.sh": [(f"t1nk_fold{k}", WALLTIME["single_lead"], [("lead_II", k), ("lead_V2", k)], False)
                                  for k in range(1, 6)],
        "submit_two_lead.sh": [("t1nk_two_lead", WALLTIME["two_lead"], [("two_lead", k) for k in range(1, 6)], False)],
    }
    hdr = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
    for submit, specs in jobs.items():
        lines = []
        for job_name, walltime, runs, smoke in specs:
            name, body = render(job_name, walltime, runs, smoke)
            (SBATCH_OUT / name).write_text(body)
            lines.append(f"sbatch {name}\n")
        (SBATCH_OUT / submit).write_text(hdr + "".join(lines))
        os.chmod(SBATCH_OUT / submit, 0o755)
    print(f"wrote {SBATCH_OUT}: submit_smoke.sh (1 job, II + V2 one-epoch checks), "
          f"submit_single_lead.sh (5 jobs, 10 runs), submit_two_lead.sh (1 job, 5 runs; needs the ii_v2 backbone)")


if __name__ == "__main__":
    main()
