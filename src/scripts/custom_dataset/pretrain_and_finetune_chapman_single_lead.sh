#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Pretrain two single-lead Chapman encoders (lead II only and lead V2 only),
# then fine-tune the matching single-lead LTSTDB regression models on the
# clean-11 median dataset.
#
# Phases (skip-existing on each):
#   1. Chapman II  pretrain  ->  patch_ecg_chapman_ii_only/
#   2. Chapman V2  pretrain  ->  patch_ecg_chapman_v2_only/
#   3. Fine-tune lead_II from chapman_ii_only ->
#        ltstdb_median_clean11_lead_II_regression_pretrained_chapman_ii_only_trial01
#   4. Fine-tune lead_V2 from chapman_v2_only ->
#        ltstdb_median_clean11_lead_V2_regression_pretrained_chapman_v2_only_trial01
#
# Pretrain hyperparameters mirror the existing patch_ecg_chapman_ii_v2 / _tta
# runs (see results/pre-train/saved_models/chapman/.../params.txt) with only
# --custom_lead_selection changed.
#
# Fine-tune hyperparameters mirror the existing clean-11 median pretrained
# variant (lr=3e-4, head_epochs=5, n_epochs_finetune=40).
#
# Expected wall time on the A16:
#   - each Chapman pretrain  ~4-6 h
#   - each LTSTDB fine-tune  ~30-45 min
#   - total                  ~10-14 h
#
# Run overnight (tmux recommended):
#   tmux new -s pretrain
#   ./pretrain_and_finetune_chapman_single_lead.sh
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# Override via env if Chapman lives elsewhere on your machine.
CHAPMAN_ROOT="${CHAPMAN_ROOT:-/home/student/Desktop/Research Data/DATASET/chapman}"

LTSTDB_MEDIAN_ROOT="${REPO_ROOT}/custom_datasets/ltstdb_regression_median_100hz_clean11_nocompress"

PRETRAIN_OUT_ROOT="${REPO_ROOT}/results/pre-train/saved_models/chapman"
FINETUNE_OUT_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

# Fine-tune seed/trial (mirrors run_ltstdb_regression_clean11_median_two_vs_single.sh)
read -r -a SEEDS <<< "${SEEDS:-200}"
START_TRIAL_IDX="${START_TRIAL_IDX:-1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# ---- Pre-flight checks -------------------------------------------------------
preflight_fail=0
if [[ ! -d "${CHAPMAN_ROOT}" ]]; then
  echo "ERROR: Chapman dataset not found at: ${CHAPMAN_ROOT}" >&2
  echo "       Override with:  CHAPMAN_ROOT=/path/to/chapman ${BASH_SOURCE[0]}" >&2
  preflight_fail=1
fi
if [[ ! -d "${LTSTDB_MEDIAN_ROOT}" ]]; then
  echo "ERROR: clean-11 median LTSTDB dataset missing: ${LTSTDB_MEDIAN_ROOT}" >&2
  echo "       Build it first with:" >&2
  echo "         ${PYTHON_BIN} src/scripts/custom_dataset/prepare_clean11_dataset.py" >&2
  preflight_fail=1
fi
[[ "${preflight_fail}" -ne 0 ]] && exit 2

mkdir -p "${PRETRAIN_OUT_ROOT}" "${FINETUNE_OUT_ROOT}"
FAILURE_LOG="${FINETUNE_OUT_ROOT}/run_failures.log"

# ---- Phase 1+2: Chapman single-lead pretrain --------------------------------
# Skip-existing sentinel: <model>_best.pt exists. Delete manually to force rerun.
run_chapman_pretrain() {
  local lead_tag="$1"        # II | V2
  local lead_sel="$2"        # II | V2 (literal value passed to --custom_lead_selection)
  local model_name="patch_ecg_chapman_${lead_tag,,}_only"
  local run_dir="${PRETRAIN_OUT_ROOT}/${model_name}"
  local sentinel="${run_dir}/${model_name}_best.pt"
  local log_path="${run_dir}/run.log"

  if [[ -f "${sentinel}" ]]; then
    echo "SKIP pretrain ${model_name} (sentinel exists: ${sentinel})"
    return 0
  fi

  mkdir -p "${run_dir}"

  echo
  echo "======================================================================"
  echo "PRETRAIN  Chapman ${lead_tag} only  (custom_lead_selection=${lead_sel})"
  echo "Model:    ${model_name}"
  echo "Log:      ${log_path}"
  echo "======================================================================"

  # patchECG_pretrain always routes through configure_dist_training; with a
  # single local GPU we need torchrun to set LOCAL_RANK/RANK/WORLD_SIZE
  # (--local_node selects the local-node code path inside the framework).
  local TORCHRUN
  TORCHRUN="$(dirname "${PYTHON_BIN}")/torchrun"
  if [[ ! -x "${TORCHRUN}" ]]; then
    TORCHRUN="torchrun"
  fi

  if "${TORCHRUN}" --standalone --nproc_per_node=1 \
      -m src.patchECG_pretrain \
      --local_node \
      --lr=0.0015 \
      --save_every=20 \
      --dset_pretrain=chapman \
      --root_path="${CHAPMAN_ROOT}" \
      --num_workers=4 \
      --batch_size=100 \
      --n_epochs_pretrain=100 \
      --warmup_epochs=2 \
      --plot_every_n=1 \
      --shared_embedding \
      --min_class_size=50 \
      --d_model=128 \
      --n_heads=8 \
      --n_layers=6 \
      --d_ff=512 \
      --patch_len=50 \
      --stride=50 \
      --mask_ratio=0.4 \
      --dropout=0.0 \
      --head_dropout=0.0 \
      --weight_decay=0.05 \
      --data_augmentation=none \
      --custom_lead_selection="${lead_sel}" \
      --trafos=None \
      --model=vanilla_vit \
      --model_name="${model_name}" \
      2>&1 | tee "${log_path}"; then
    echo "OK   pretrain ${model_name}"
  else
    local ts
    ts="$(date -Is)"
    echo "${ts} FAILED pretrain ${model_name} (see ${log_path})" | tee -a "${FAILURE_LOG}" >&2
    return 1
  fi
}

run_chapman_pretrain "II" "II"
run_chapman_pretrain "V2" "V2"

# ---- Phase 3+4: LTSTDB single-lead fine-tune from new pretrains -------------
# Skip-existing sentinel: _test_metrics_<model>.csv exists. Matches the
# convention used by the other clean-11 trial scripts.
run_single_lead_finetune() {
  local lead_tag="$1"        # II | V2 (for the LTSTDB lead selection)
  local lead_sel="$2"        # II | V2 (literal --custom_lead_selection)
  local trial="$3"
  local seed="$4"

  local chapman_model="patch_ecg_chapman_${lead_tag,,}_only"
  local pretrained_ckpt="${PRETRAIN_OUT_ROOT}/${chapman_model}/${chapman_model}_best.pt"

  if [[ ! -f "${pretrained_ckpt}" ]]; then
    echo "SKIP fine-tune (pretrain not found): ${pretrained_ckpt}"
    return 1
  fi

  local model_name="ltstdb_median_clean11_lead_${lead_tag}_regression_pretrained_chapman_${lead_tag,,}_only_trial${trial}"
  local run_dir="${FINETUNE_OUT_ROOT}/${model_name}"
  local sentinel="${run_dir}/_test_metrics_${model_name}.csv"
  local log_path="${run_dir}/run.log"

  if [[ -f "${sentinel}" ]]; then
    echo "SKIP fine-tune ${model_name} (sentinel exists: ${sentinel})"
    return 0
  fi

  mkdir -p "${run_dir}"

  echo
  echo "======================================================================"
  echo "FINE-TUNE  lead_${lead_tag} (single-lead) from ${chapman_model}"
  echo "Model:     ${model_name}"
  echo "Pretrain:  ${pretrained_ckpt}"
  echo "Log:       ${log_path}"
  echo "======================================================================"

  if "${PYTHON_BIN}" -m src.patchECG_finetune \
      --seed="${seed}" \
      --lr=0.0003 \
      --save_every=20 \
      --root_path="${LTSTDB_MEDIAN_ROOT}" \
      --num_workers=4 \
      --batch_size=128 \
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
      --n_epochs_finetune_head=5 \
      --n_epochs_finetune=40 \
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
      --pretrained_model_path="${pretrained_ckpt}" \
      2>&1 | tee "${log_path}"; then
    echo "OK   fine-tune ${model_name}"
  else
    local ts
    ts="$(date -Is)"
    echo "${ts} FAILED fine-tune ${model_name} (see ${log_path})" | tee -a "${FAILURE_LOG}" >&2
  fi
}

trial_idx="${START_TRIAL_IDX}"
for seed in "${SEEDS[@]}"; do
  trial="$(printf "%02d" "${trial_idx}")"
  run_single_lead_finetune "II" "II" "${trial}" "${seed}"
  run_single_lead_finetune "V2" "V2" "${trial}" "${seed}"
  trial_idx=$((trial_idx + 1))
done

echo
echo "Done."
if [[ -s "${FAILURE_LOG}" ]]; then
  echo "Failures recorded in: ${FAILURE_LOG}"
fi
echo
echo "Outputs:"
echo "  Pretrains:   ${PRETRAIN_OUT_ROOT}/patch_ecg_chapman_{ii,v2}_only/"
echo "  Fine-tunes:  ${FINETUNE_OUT_ROOT}/ltstdb_median_clean11_lead_{II,V2}_regression_pretrained_chapman_*_only_trial${START_TRIAL_IDX}*"
