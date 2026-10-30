"""
Generate sbatch training scripts for Test 1-NK (2026-10-29): Test 1's segments re-centred on the NeuroKit R-peak
(dataset README_TEST1NK.md). Unified-pretrained arm only, cv5_optB, 5 folds per mode:

  lead_II   backbone patch_ecg_unified_ii_only   --custom_lead_selection II      (5 jobs)
  lead_V2   backbone patch_ecg_unified_v2_only   --custom_lead_selection V2      (5 jobs)
  two_lead  backbone patch_ecg_unified_ii_v2     --custom_lead_selection II,V2   (5 jobs, optional)

Recipe = Test 1 exactly (params.txt of ltstdb_beatcenter_test1_cv5_optB_lead_II_regression_pretrained_chapman_ii_only_fold01
diffed against the template: only name / backbone / paths differ): MSE, 20 head + 70 full epochs, lr 1e-3, layer decay 0.65,
weight decay 0.05, batch 128, RevIN default (BN), valid_loss selection, seed 200 + fold index (200..204), as Test 1.
Names keep Test 1's pattern ("chapman" tag = the historical name of the Unified backbones) with the prefix
ltstdb_beatcenter_test1nk_, so the existing evaluation code finds them by swapping the prefix.
The folds are already applied per fold directory (strat_fold_annotated). Nothing is submitted; this only writes files.
"""
import os
from pathlib import Path

REPO_ROOT = os.environ.get("T1NK_REPO_ROOT", "/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DS_ROOT = os.environ.get("T1NK_DS_ROOT", "/home/youss99/scratch/GIT/datasets/ltstdb_regression_beatcenter_100hz_test1nk_nocompress")
ACCOUNT = os.environ.get("T1NK_ACCOUNT", "def-majidk")
GPU = os.environ.get("T1NK_GPU", "--gpus=h100_3g.40gb:1")
MODULES = "StdEnv/2023 python/3.11"
PY_BIN = f"{REPO_ROOT}/.venv/bin/python"
SBATCH_OUT = Path(REPO_ROOT) / "logs" / "sbatch_scripts" / "test1nk_training"
LOG_OUT = Path(REPO_ROOT) / "logs"
WALLTIME = "02:00:00"
UNI = f"{REPO_ROOT}/results/pre-train/saved_models/unified"
MODES = {  # mode: (backbone file, Test 1 arm tag, lead selection)
    "lead_II": (f"{UNI}/patch_ecg_unified_ii_only/patch_ecg_unified_ii_only_best.pt", "pretrained_chapman_ii_only", "II"),
    "lead_V2": (f"{UNI}/patch_ecg_unified_v2_only/patch_ecg_unified_v2_only_best.pt", "pretrained_chapman_v2_only", "V2"),
    "two_lead": (f"{UNI}/patch_ecg_unified_ii_v2/patch_ecg_unified_ii_v2_best.pt", "pretrained_chapman_ii_v2", "II,V2"),
}


def render(mode, k, smoke=False):
    backbone, tag, leads = MODES[mode]; seed = 200 + k - 1
    model_name = f"ltstdb_beatcenter_test1nk_cv5_optB_{mode}_regression_{tag}_fold{k:02d}_thesis_small"
    job_name = f"t1nk_{mode}_{k}"
    head, full, walltime = (20, 70, WALLTIME) if not smoke else (0, 1, "00:30:00")
    if smoke:
        model_name, job_name = f"SMOKE_{model_name}", f"t1nk_smoke_{mode}"
    dataset = f"{DS_ROOT}/fold{k:02d}"; log_pattern = f"{LOG_OUT}/{model_name}-%j.out"
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
[[ -f "{backbone}" ]] || {{ echo "ERROR: backbone missing: {backbone}" >&2; exit 2; }}

STAGED_ROOT="${{SLURM_TMPDIR}}/dataset"
echo "Staging fold{k:02d} from {dataset} to ${{STAGED_ROOT}} ..."
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
  --pretrained_model_path="{backbone}"
"""
    return f"{model_name}.sbatch", body


def main():
    SBATCH_OUT.mkdir(parents=True, exist_ok=True); lines = {}
    for mode in MODES:
        (SBATCH_OUT / mode).mkdir(exist_ok=True)
        for k in range(1, 6):
            name, body = render(mode, k); (SBATCH_OUT / mode / name).write_text(body); lines.setdefault(mode, []).append(f"sbatch {mode}/{name}\n")
        name, body = render(mode, 1, smoke=True); (SBATCH_OUT / mode / name).write_text(body); lines.setdefault(f"smoke_{mode}", []).append(f"sbatch {mode}/{name}\n")
    hdr = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
    (SBATCH_OUT / "submit_smoke.sh").write_text(hdr + "".join(lines["smoke_lead_II"] + lines["smoke_lead_V2"]))
    (SBATCH_OUT / "submit_single_lead.sh").write_text(hdr + "".join(lines["lead_II"] + lines["lead_V2"]))
    (SBATCH_OUT / "submit_two_lead.sh").write_text(hdr + "".join(lines["two_lead"]))
    for f in ("submit_smoke.sh", "submit_single_lead.sh", "submit_two_lead.sh"):
        os.chmod(SBATCH_OUT / f, 0o755)
    print(f"wrote {SBATCH_OUT}: submit_smoke.sh (2 one-epoch checks), submit_single_lead.sh (10 jobs), submit_two_lead.sh (5 jobs)")


if __name__ == "__main__":
    main()
