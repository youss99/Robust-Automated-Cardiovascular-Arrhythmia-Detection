#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Full-16 (12/2/2 omit-in-train) median LTSTDB regression study.
#
# 6 model variants per trial:
#   1. two_lead, supervised
#   2. two_lead, pretrained from patch_ecg_chapman_ii_v2_tta
#   3. lead_II,  supervised
#   4. lead_II,  pretrained from patch_ecg_chapman_ii_only
#   5. lead_V2,  supervised
#   6. lead_V2,  pretrained from patch_ecg_chapman_v2_only
#
# Dataset: ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress
#   TRAIN (12): s20051, s20061, s20081, s20091, s20101, s20111, s20121,
#               s20141, s20201, s20211, s20221, s20241   (5 omit + 7 clean)
#   VAL    (2): s20011, s20231                          (clean only)
#   TEST   (2): s20071, s20131                          (clean only)
#
# Hypothesis: broader 12-patient training (matching phase0's split size) should
# narrow the alarm-level gap to phase0 chapman_pretrain (f1=0.89). Single-lead
# variants additionally test whether per-lead specialization still helps.
#
# Model name prefix: ltstdb_median_full16_12_2_2_omit_train_<lead>_regression_
# Keeps these distinct from the existing _full16_ (10/3/3) trial dirs.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

MEDIAN_DATA_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress"

PRETRAIN_ROOT="${REPO_ROOT}/results/pre-train/saved_models/chapman"
CHAPMAN_TWOLEAD_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"
CHAPMAN_II_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_only/patch_ecg_chapman_ii_only_best.pt"
CHAPMAN_V2_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_v2_only/patch_ecg_chapman_v2_only_best.pt"

RESULTS_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"

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
if [[ ! -d "${MEDIAN_DATA_ROOT}" ]]; then
  echo "ERROR: dataset missing: ${MEDIAN_DATA_ROOT}" >&2
  echo "       Build it first with:" >&2
  echo "         ${PYTHON_BIN} src/scripts/custom_dataset/prepare_full16_dataset.py \\" >&2
  echo "             --n_train 12 --n_val 2 --n_test 2 \\" >&2
  echo "             --force_omit_to_train --output_suffix 12_2_2_omit_train" >&2
  preflight_fail=1
fi
for ckpt in "${CHAPMAN_TWOLEAD_CKPT}" "${CHAPMAN_II_CKPT}" "${CHAPMAN_V2_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: pretrained checkpoint missing: ${ckpt}" >&2
    preflight_fail=1
  fi
done
[[ "${preflight_fail}" -ne 0 ]] && exit 2

mkdir -p "${RESULTS_ROOT}"
FAILURE_LOG="${RESULTS_ROOT}/run_failures.log"

# ---- Trial runner ------------------------------------------------------------
# args:
#   lead_tag        : two_lead | lead_II | lead_V2
#   lead_sel        : II,V2 | II | V2
#   training_mode   : supervised | pretrained
#   pretrained_ckpt : path to .pt (only used when training_mode == pretrained)
#   trial           : zero-padded trial id
#   seed            : numerical seed
run_trial() {
  local lead_tag="$1"
  local lead_sel="$2"
  local training_mode="$3"
  local pretrained_ckpt="$4"
  local trial="$5"
  local seed="$6"

  local model_name lr head_epochs
  local pretrained_args=()
  local base="ltstdb_median_full16_12_2_2_omit_train_${lead_tag}_regression"

  if [[ "${training_mode}" == "supervised" ]]; then
    model_name="${base}_supervised_trial${trial}"
    lr="0.001"
    head_epochs="0"
  elif [[ "${training_mode}" == "pretrained" ]]; then
    local ckpt_tag
    case "${pretrained_ckpt}" in
      *chapman_ii_v2_tta*)    ckpt_tag="chapman_ii_v2" ;;
      *chapman_ii_only*)      ckpt_tag="chapman_ii_only" ;;
      *chapman_v2_only*)      ckpt_tag="chapman_v2_only" ;;
      *)                      ckpt_tag="custom" ;;
    esac
    model_name="${base}_pretrained_${ckpt_tag}_trial${trial}"
    lr="0.0003"
    head_epochs="5"
    pretrained_args=(--pretrained_model_path="${pretrained_ckpt}")
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
  echo "Variant : ${lead_tag} (--custom_lead_selection=${lead_sel})"
  echo "Mode    : ${training_mode}"
  if [[ "${training_mode}" == "pretrained" ]]; then
    echo "Pretrain: ${pretrained_ckpt}"
  fi
  echo "Trial   : ${trial} (seed=${seed})"
  echo "Model   : ${model_name}"
  echo "Log     : ${log_path}"
  echo "======================================================================"

  if "${PYTHON_BIN}" -m src.patchECG_finetune \
      --seed="${seed}" \
      --lr="${lr}" \
      --save_every=20 \
      --root_path="${MEDIAN_DATA_ROOT}" \
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
      --custom_lead_selection="${lead_sel}" \
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

  # two-lead
  run_trial "two_lead" "II,V2" "supervised" ""                       "${trial}" "${seed}"
  run_trial "two_lead" "II,V2" "pretrained" "${CHAPMAN_TWOLEAD_CKPT}" "${trial}" "${seed}"

  # lead II
  run_trial "lead_II"  "II"    "supervised" ""                       "${trial}" "${seed}"
  run_trial "lead_II"  "II"    "pretrained" "${CHAPMAN_II_CKPT}"      "${trial}" "${seed}"

  # lead V2
  run_trial "lead_V2"  "V2"    "supervised" ""                       "${trial}" "${seed}"
  run_trial "lead_V2"  "V2"    "pretrained" "${CHAPMAN_V2_CKPT}"      "${trial}" "${seed}"

  trial_idx=$((trial_idx + 1))
done

echo
echo "Done."
if [[ -s "${FAILURE_LOG}" ]]; then
  echo "Failures recorded in: ${FAILURE_LOG}"
fi
echo
echo "To run trials 2-5 later (24 more runs):"
echo "  SEEDS=\"201 202 203 204\" START_TRIAL_IDX=2 \\"
echo "      ${BASH_SOURCE[0]}"
