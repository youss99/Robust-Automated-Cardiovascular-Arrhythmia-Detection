#!/usr/bin/env bash
# One-time setup on Nibi for the E1 ST-regression runs (run on a LOGIN node: it needs internet).
#   bash src/scripts/custom_dataset/nibi_setup_e1.sh
# Re-runnable: an existing .venv is reused, finished steps are cheap to repeat.
# Expects: code in $REPO, the E1 dataset in $DS, the backbones under $REPO/results/pre-train/saved_models/unified/.
set -euo pipefail

REPO="${E1_REPO_ROOT:-/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection}"
DS="${E1_DS_ROOT:-/home/youss99/scratch/GIT/datasets/ltstdb_regression_e1_nkcentred_balanced_100hz}"
cd "$REPO"
module load StdEnv/2023 python/3.11

echo "== 1. Python environment"
if [ ! -x .venv/bin/python ]; then
  virtualenv --no-download .venv
fi
source .venv/bin/activate
pip install --no-index --upgrade pip
# Prefer the Alliance wheelhouse (cluster-built torch); fall back to PyPI for anything it lacks.
PKGS=(torch torchvision "numpy<2" pandas h5py scipy scikit-learn scikit-multilearn einops timm kornia
      peft transformers accelerate torchmetrics torcheval matplotlib tqdm pyyaml wfdb)
for p in "${PKGS[@]}"; do
  pip install --no-index "$p" >/dev/null 2>&1 || pip install "$p"
done

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
for b in patch_ecg_unified_ii_only patch_ecg_unified_v2_only patch_ecg_unified_v5_fixed; do
  f="results/pre-train/saved_models/unified/$b/${b}_best.pt"
  [ -s "$f" ] && echo "ok  $f" || { echo "MISSING $f"; exit 1; }
done

echo "== 5. Job scripts"
python src/scripts/custom_dataset/build_e1_train_sbatches.py
echo
echo "Next: sbatch logs/sbatch_scripts/e1_training/smoke.sbatch   (1 epoch, ~minutes)"
echo "      then, if its log ends with test metrics: bash logs/sbatch_scripts/e1_training/submit_all.sh"
