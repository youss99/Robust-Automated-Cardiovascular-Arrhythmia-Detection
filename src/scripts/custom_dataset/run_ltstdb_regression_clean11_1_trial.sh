#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Phase 1 (clean-11 variant) of the mean/median LTST regression study.
#
# Identical to run_ltstdb_regression_mean_median_1_trial.sh except it points
# at the clean11 datasets (built by prepare_clean11_dataset.py), which:
#   - exclude the 5 records that have any lead in MANUAL_OMIT
#     (s20081, s20091, s20121, s20221, s20241)
#   - re-stratify the remaining 11 patients into 7/2/2 train/val/test using
#     upstream .16a-derived distribution features (no peeking at GS alarms)
#
# Scenarios (4 total):
#   1. supervised + mean labels
#   2. pretrained (Chapman II,V2) + mean labels
#   3. supervised + median labels
#   4. pretrained (Chapman II,V2) + median labels
#
# Extend to trials 02..05 later:
#     SEEDS="201 202 203 204" START_TRIAL_IDX=2 ./run_ltstdb_regression_clean11_1_trial.sh
#
# Robustness identical to the original 1-trial script:
#   * skip-existing (sentinel = _test_metrics_<model>.csv)
#   * per-trial log redirected to <run_dir>/run.log
#   * set -uo pipefail (NOT -e) so one failure doesn't abort the rest
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PRETRAINED_MODEL_PATH="${REPO_ROOT}/results/pre-train/saved_models/chapman/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"

MEDIAN_DATA_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_median_100hz_clean11_nocompress"
MEAN_DATA_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_mean_100hz_clean11_nocompress"

RESULTS_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

read -r -a SEEDS <<< "${SEEDS:-200}"
START_TRIAL_IDX="${START_TRIAL_IDX:-1}"

N_EPOCHS=40
BATCH_SIZE=128
NUM_WORKERS=4

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# ---- Pre-flight checks -------------------------------------------------------
preflight_fail=0
if [[ ! -f "${PRETRAINED_MODEL_PATH}" ]]; then
  echo "ERROR: pretrained checkpoint missing: ${PRETRAINED_MODEL_PATH}" >&2
  preflight_fail=1
fi
for path in "${MEDIAN_DATA_ROOT}" "${MEAN_DATA_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: clean-11 dataset missing: ${path}" >&2
    echo "       Build it first with:" >&2
    echo "         ${PYTHON_BIN} src/scripts/custom_dataset/prepare_clean11_dataset.py" >&2
    preflight_fail=1
  fi
done
[[ "${preflight_fail}" -ne 0 ]] && exit 2

mkdir -p "${RESULTS_ROOT}"
FAILURE_LOG="${RESULTS_ROOT}/run_failures.log"

# ---- Trial runner ------------------------------------------------------------
run_trial() {
  local dataset_name="$1"
  local data_root="$2"
  local training_mode="$3"
  local trial="$4"
  local seed="$5"

  local model_name lr head_epochs
  local pretrained_args=()

  if [[ "${training_mode}" == "supervised" ]]; then
    model_name="ltstdb_${dataset_name}_regression_clean11_supervised_trial${trial}"
    lr="0.001"
    head_epochs="0"
  elif [[ "${training_mode}" == "pretrained" ]]; then
    model_name="ltstdb_${dataset_name}_regression_clean11_pretrained_chapman_ii_v2_trial${trial}"
    lr="0.0003"
    head_epochs="5"
    pretrained_args=(--pretrained_model_path="${PRETRAINED_MODEL_PATH}")
  else
    echo "Unknown training mode: ${training_mode}" >&2
    return 1
  fi

  local run_dir="${RESULTS_ROOT}/${model_name}"
  local sentinel="${run_dir}/_test_metrics_${model_name}.csv"
  local log_path="${run_dir}/run.log"

  if [[ -f "${sentinel}" ]]; then
    echo "SKIP ${model_name} (sentinel exists: ${sentinel})"
    return 0
  fi

  mkdir -p "${run_dir}"

  echo
  echo "======================================================================"
  echo "Dataset: ${dataset_name} (clean-11) | Mode: ${training_mode} | Trial: ${trial} | Seed: ${seed}"
  echo "Model name: ${model_name}"
  echo "Log:        ${log_path}"
  echo "======================================================================"

  if "${PYTHON_BIN}" -m src.patchECG_finetune \
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
      "${pretrained_args[@]}" \
      2>&1 | tee "${log_path}"; then
    echo "OK   ${model_name}"
  else
    local ts
    ts="$(date -Is)"
    echo "${ts} FAILED ${model_name} (see ${log_path})" | tee -a "${FAILURE_LOG}" >&2
  fi
}

# ---- Driver loop -------------------------------------------------------------
trial_idx="${START_TRIAL_IDX}"
for seed in "${SEEDS[@]}"; do
  trial="$(printf "%02d" "${trial_idx}")"

  run_trial "mean"   "${MEAN_DATA_ROOT}"   "supervised" "${trial}" "${seed}"
  run_trial "mean"   "${MEAN_DATA_ROOT}"   "pretrained" "${trial}" "${seed}"
  run_trial "median" "${MEDIAN_DATA_ROOT}" "supervised" "${trial}" "${seed}"
  run_trial "median" "${MEDIAN_DATA_ROOT}" "pretrained" "${trial}" "${seed}"

  trial_idx=$((trial_idx + 1))
done

echo
echo "Done."
if [[ -s "${FAILURE_LOG}" ]]; then
  echo "Failures recorded in: ${FAILURE_LOG}"
fi
echo
echo "To run trials 2-5 later (16 more runs):"
echo "  SEEDS=\"201 202 203 204\" START_TRIAL_IDX=2 \\"
echo "      ${BASH_SOURCE[0]}"
