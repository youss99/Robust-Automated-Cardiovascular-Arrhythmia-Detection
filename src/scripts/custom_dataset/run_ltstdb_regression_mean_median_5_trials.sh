#!/usr/bin/env bash
set -euo pipefail

# Runs 5 trials for each LTST regression experiment:
#   1. median targets, supervised from scratch
#   2. median targets, Chapman II,V2 pretraining + fine-tuning
#   3. mean targets, supervised from scratch
#   4. mean targets, Chapman II,V2 pretraining + fine-tuning
#
# After all runs finish, the script writes aggregate and per-lead summary CSVs.

REPO_ROOT="/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PRETRAINED_MODEL_PATH="${REPO_ROOT}/results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"

MEDIAN_DATA_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_median_100hz_balanced_min5_nocompress"
MEAN_DATA_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_mean_100hz_balanced_min5_nocompress"

RESULTS_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
SUMMARY_DIR="${RESULTS_ROOT}/ltstdb_mean_median_5trial_summary"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

SEEDS=(200 201 202 203 204)
N_EPOCHS=40
BATCH_SIZE=128
NUM_WORKERS=4

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

run_trial() {
  local dataset_name="$1"
  local data_root="$2"
  local training_mode="$3"
  local trial="$4"
  local seed="$5"

  local model_name
  local lr
  local head_epochs
  local pretrained_args=()

  if [[ "${training_mode}" == "supervised" ]]; then
    model_name="ltstdb_${dataset_name}_regression_supervised_trial${trial}"
    lr="0.001"
    head_epochs="0"
  elif [[ "${training_mode}" == "pretrained" ]]; then
    model_name="ltstdb_${dataset_name}_regression_pretrained_chapman_ii_v2_trial${trial}"
    lr="0.0003"
    head_epochs="5"
    pretrained_args=(--pretrained_model_path="${PRETRAINED_MODEL_PATH}")
  else
    echo "Unknown training mode: ${training_mode}" >&2
    exit 1
  fi

  echo
  echo "======================================================================"
  echo "Dataset: ${dataset_name} | Mode: ${training_mode} | Trial: ${trial} | Seed: ${seed}"
  echo "Model name: ${model_name}"
  echo "======================================================================"

  "${PYTHON_BIN}" -m src.patchECG_finetune \
    --seed="${seed}" \
    --lr="${lr}" \
    --save_every=20 \
    --root_path="${data_root}" \
    --num_workers="${NUM_WORKERS}" \
    --batch_size="${BATCH_SIZE}" \
    --data_augmentation=none \
    --trafos=None \
    --class_token=cls_token \
    --shared_embedding \
    --min_class_size=1 \
    --layer_decay=0.65 \
    --weight_decay=0.05 \
    --head_dropout=0.0 \
    --dropout=0.0 \
    --dset_finetune=patient-segment-regression \
    --classification_type=REGRESSION \
    --head_type=regression \
    --regression_loss=mse \
    --no-focal_loss \
    --model=vanilla_vit \
    --d_model=128 \
    --no-scheduler \
    --custom_lead_selection=II,V2 \
    --n_epochs_finetune_head="${head_epochs}" \
    --n_epochs_finetune="${N_EPOCHS}" \
    --n_heads=8 \
    --n_layers=6 \
    --d_ff=512 \
    --patch_len=50 \
    --stride=50 \
    --model_name="${model_name}" \
    --metric_by_class \
    --no-mentor_mix \
    --no-bootstrapping \
    --iterations=5000 \
    --model_selection_metric=valid_loss \
    "${pretrained_args[@]}"
}

trial_idx=1
for seed in "${SEEDS[@]}"; do
  trial="$(printf "%02d" "${trial_idx}")"

  run_trial "median" "${MEDIAN_DATA_ROOT}" "supervised" "${trial}" "${seed}"
  run_trial "median" "${MEDIAN_DATA_ROOT}" "pretrained" "${trial}" "${seed}"
  run_trial "mean" "${MEAN_DATA_ROOT}" "supervised" "${trial}" "${seed}"
  run_trial "mean" "${MEAN_DATA_ROOT}" "pretrained" "${trial}" "${seed}"

  trial_idx=$((trial_idx + 1))
done

echo
echo "Aggregating trial metrics..."
mkdir -p "${SUMMARY_DIR}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

import pandas as pd

repo = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
results_root = repo / "results/fine-tune/saved_models/patient-segment-regression"
summary_dir = results_root / "ltstdb_mean_median_5trial_summary"
summary_dir.mkdir(parents=True, exist_ok=True)

run_specs = []
for dataset_name in ["median", "mean"]:
    for training_mode in ["supervised", "pretrained"]:
        for trial in range(1, 6):
            trial_id = f"{trial:02d}"
            if training_mode == "supervised":
                model_name = f"ltstdb_{dataset_name}_regression_supervised_trial{trial_id}"
            else:
                model_name = f"ltstdb_{dataset_name}_regression_pretrained_chapman_ii_v2_trial{trial_id}"
            run_specs.append((dataset_name, training_mode, trial_id, model_name, results_root / model_name))


def read_split_metrics(run_dir: Path, model_name: str, split: str) -> dict:
    file_split = "valid" if split == "validation" else split
    path = run_dir / f"_{file_split}_metrics_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0].to_dict()
    out = {}
    for key, value in row.items():
        metric = key.split("_", 1)[1] if "_" in key else key
        out[f"{split}_{metric}"] = value
    return out


def read_per_class_metrics(dataset_name: str, training_mode: str, trial_id: str, model_name: str, run_dir: Path) -> list[dict]:
    rows = []
    for split in ["train", "validation", "test"]:
        file_split = "valid" if split == "validation" else split
        path = run_dir / f"_{file_split}_metrics_{model_name}_per_class.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for row in df.to_dict("records"):
            row.update({
                "dataset": dataset_name,
                "training_mode": training_mode,
                "trial": trial_id,
                "model_name": model_name,
                "split": split,
            })
            rows.append(row)
    return rows


aggregate_rows = []
per_class_rows = []
missing = []

for dataset_name, training_mode, trial_id, model_name, run_dir in run_specs:
    if not run_dir.exists():
        missing.append(str(run_dir))
        continue

    training_path = run_dir / f"_training_metrics_gpu0_{model_name}.csv"
    if not training_path.exists():
        missing.append(str(training_path))
        continue

    training = pd.read_csv(training_path)
    best = training.loc[training["valid_loss"].idxmin()]

    row = {
        "dataset": dataset_name,
        "training_mode": training_mode,
        "trial": trial_id,
        "model_name": model_name,
        "best_epoch": int(best["epoch"]),
        "best_train_loss": best["train_loss"],
        "best_valid_loss": best["valid_loss"],
        "best_train_MSE": best["train_MSE"],
        "best_valid_MSE": best["valid_MSE"],
        "best_valid_MAE": best["valid_MAE"],
        "best_valid_RMSE": best["valid_RMSE"],
        "best_valid_R2": best["valid_R2"],
    }
    for split in ["train", "validation", "test"]:
        row.update(read_split_metrics(run_dir, model_name, split))
    aggregate_rows.append(row)
    per_class_rows.extend(read_per_class_metrics(dataset_name, training_mode, trial_id, model_name, run_dir))

if missing:
    print("Warning: missing expected run artifacts:")
    for item in missing:
        print(f"  {item}")

aggregate = pd.DataFrame(aggregate_rows).sort_values(["dataset", "training_mode", "trial"])
aggregate.to_csv(summary_dir / "all_trial_metrics.csv", index=False)

metric_cols = [
    col for col in aggregate.columns
    if col not in {"dataset", "training_mode", "trial", "model_name"}
]
summary = aggregate.groupby(["dataset", "training_mode"])[metric_cols].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(summary_dir / "aggregate_metric_summary_mean_std.csv", index=False)

if per_class_rows:
    per_class = pd.DataFrame(per_class_rows).sort_values(["dataset", "training_mode", "split", "label", "trial"])
    per_class.to_csv(summary_dir / "all_trial_per_lead_metrics.csv", index=False)

    per_class_metric_cols = [
        col for col in ["MSE", "MAE", "RMSE", "R2"]
        if col in per_class.columns
    ]
    per_class_summary = per_class.groupby(["dataset", "training_mode", "split", "label"])[per_class_metric_cols].agg(["mean", "std"])
    per_class_summary.columns = [f"{metric}_{stat}" for metric, stat in per_class_summary.columns]
    per_class_summary = per_class_summary.reset_index()
    per_class_summary.to_csv(summary_dir / "per_lead_metric_summary_mean_std.csv", index=False)

print(f"Wrote summaries to {summary_dir}")
print(summary.to_string(index=False))
PY

echo "Done."
echo "Summary directory: ${SUMMARY_DIR}"
