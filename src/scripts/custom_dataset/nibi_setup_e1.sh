#!/usr/bin/env bash
# One-time checks on Nibi for the E1 ST-regression runs, as a CPU job (torch/h5py work stays off the login node).
#   sbatch src/scripts/custom_dataset/nibi_setup_e1.sh        (from the repo root)
# The .venv must already exist: sbatch src/scripts/nibi_setup_venv.sbatch
# Expects: code in $REPO, the E1 dataset in $DS, the backbones under $REPO/results/pre-train/saved_models/unified/.
#SBATCH --account=def-majidk
#SBATCH --job-name=e1_setup
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

REPO="${E1_REPO_ROOT:-/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection}"
DS="${E1_DS_ROOT:-/home/youss99/scratch/GIT/datasets/ltstdb_regression_e1_nkcentred_balanced_100hz}"
cd "$REPO"
module load StdEnv/2023 python/3.11
export OMP_NUM_THREADS=1

echo "== 1. Python environment"
[ -x .venv/bin/python ] || { echo "MISSING .venv: run sbatch src/scripts/nibi_setup_venv.sbatch first"; exit 1; }
source .venv/bin/activate

echo "== 2. Trainer imports"
python -c "import src.patchECG_finetune, torch; print('torch', torch.__version__, '| trainer imports OK')"

echo "== 3. Dataset"
python - "$DS" <<'PY'
import sys, pathlib, h5py, pandas as pd
ds = pathlib.Path(sys.argv[1])
expect = {"lead_II": ["fold%02d" % i for i in range(1, 6)], "lead_V2": ["fold%02d" % i for i in range(1, 6)],
          "lead_V5": ["fold_test_" + r for r in ("s20571", "s20581", "s30671", "s30681")]}
for lead, folds in expect.items():
    n = h5py.File(ds / lead / "PatientSegmentRegressionDataset.h5", "r")["tracings"].shape
    for f in folds:
        root = ds / lead / f
        assert (root / "PatientSegmentRegressionDataset.h5").resolve().exists(), f"{root}: H5 link broken"
        df = pd.read_csv(root / "modified_dataset" / "modified_dataset.csv", usecols=["strat_fold_annotated"])
        assert len(df) == n[0], f"{root}: CSV rows {len(df)} != H5 rows {n[0]}"
        c = df.strat_fold_annotated.value_counts().to_dict()
        print(f"{lead}/{f}: H5 {n}  train {c.get(1, 0)}  test {c.get(2, 0)}  val {c.get(3, 0)}")
PY

echo "== 4. Backbones"
for b in patch_ecg_unified_ii_fixed patch_ecg_unified_v2_fixed patch_ecg_unified_v5_fixed; do
  f="results/pre-train/saved_models/unified/$b/${b}_best.pt"
  [ -s "$f" ] && echo "ok  $f  $(md5sum < "$f" | cut -c1-32)" || { echo "MISSING $f"; exit 1; }
done
# Same layer names and shapes as the V5 backbone (which transfers cleanly into this regression model),
# finite weights, and II / V2 not the same file. torch.load uses the same defaults as transfer_weights().
python - <<'EOF'
import torch
root = "results/pre-train/saved_models/unified"
sd = {b: torch.load(f"{root}/patch_ecg_unified_{b}/patch_ecg_unified_{b}_best.pt", map_location="cpu")["model"]
      for b in ("ii_fixed", "v2_fixed", "v5_fixed")}
ref = {k: tuple(v.shape) for k, v in sd["v5_fixed"].items()}
for b, m in sd.items():
    sig = {k: tuple(v.shape) for k, v in m.items()}
    assert sig == ref, f"{b}: layers differ from v5_fixed: {set(sig.items()) ^ set(ref.items())}"
    assert all(torch.isfinite(v).all() for v in m.values() if v.is_floating_point()), f"{b}: non-finite weights"
    print(f"ok  {b}: {len(m)} tensors, same layout as v5_fixed, all finite")
assert not all(torch.equal(sd["ii_fixed"][k], sd["v2_fixed"][k]) for k in ref), "ii_fixed and v2_fixed are identical"
print("ok  ii_fixed and v2_fixed differ")
EOF

echo "== 5. Job scripts"
python src/scripts/custom_dataset/build_e1_train_sbatches.py
echo
echo "Next: for l in lead_II lead_V2 lead_V5; do sbatch logs/sbatch_scripts/e1_training/smoke_\$l.sbatch; done"
echo "      every logs/SMOKE_*.out must contain 'successfully transferred!' and end with test metrics;"
echo "      then: bash logs/sbatch_scripts/e1_training/submit_all.sh"
