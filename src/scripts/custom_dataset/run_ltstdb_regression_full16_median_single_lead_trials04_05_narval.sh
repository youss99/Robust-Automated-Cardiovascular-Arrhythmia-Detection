#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Narval SLURM launcher for single-lead trials 04 & 05 (extending 3 -> 5).
# Submits up to 8 jobs:
#   lead_II x {supervised, pretrained_chapman_ii_only} x {trial04 seed=203, trial05 seed=204}
#   lead_V2 x {supervised, pretrained_chapman_v2_only} x {trial04 seed=203, trial05 seed=204}
#
# Same SLURM profile + dataset staging conventions as the two-lead 5-trial
# wrapper. Sentinel skip keeps trials 01-03 untouched.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
DATA_ROOT="/scratch/youss99/ltstdb_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

PRETRAIN_ROOT="${REPO_ROOT}/results/pre-train/saved_models/chapman"
CHAPMAN_II_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_only/patch_ecg_chapman_ii_only_best.pt"
CHAPMAN_V2_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_v2_only/patch_ecg_chapman_v2_only_best.pt"

RESULTS_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
LOG_ROOT="${REPO_ROOT}/logs"
SBATCH_DIR="${LOG_ROOT}/sbatch_scripts"

ACCOUNT="def-majidk"
WALLTIME="03:00:00"
MEM="20G"
CPUS_PER_TASK="4"
GRES="gpu:a100_3g.20gb:1"

N_EPOCHS=40
BATCH_SIZE=128
NUM_WORKERS=4

DRY_RUN="${DRY_RUN:-0}"

declare -A TRIAL_SEED=( [04]=203 [05]=204 )

# Each variant: lead_tag|lead_sel|mode|pretrained_ckpt|short_jobname_prefix
VARIANTS=(
  "lead_II|II|supervised||ltst_med_II_sup"
  "lead_II|II|pretrained|${CHAPMAN_II_CKPT}|ltst_med_II_pre"
  "lead_V2|V2|supervised||ltst_med_V2_sup"
  "lead_V2|V2|pretrained|${CHAPMAN_V2_CKPT}|ltst_med_V2_pre"
)

# Pre-flight
fail=0
[[ -d "${REPO_ROOT}" ]]   || { echo "ERROR: REPO_ROOT missing: ${REPO_ROOT}" >&2; fail=1; }
[[ -x "${PYTHON_BIN}" ]]  || { echo "ERROR: python venv missing: ${PYTHON_BIN}" >&2; fail=1; }
[[ -d "${DATA_ROOT}" ]]   || { echo "ERROR: dataset missing: ${DATA_ROOT}" >&2; fail=1; }
[[ -f "${CHAPMAN_II_CKPT}" ]] || { echo "ERROR: II pretrain missing"; fail=1; }
[[ -f "${CHAPMAN_V2_CKPT}" ]] || { echo "ERROR: V2 pretrain missing"; fail=1; }
[[ "${fail}" -ne 0 ]] && exit 2

mkdir -p "${LOG_ROOT}" "${SBATCH_DIR}" "${RESULTS_ROOT}"

submitted=()
skipped=()

submit_one() {
  local lead_tag="$1" lead_sel="$2" mode="$3" ckpt="$4" short="$5" trial="$6" seed="$7"
  local base="ltstdb_median_full16_12_2_2_omit_train_${lead_tag}_regression"
  local model_name lr head_epochs ckpt_tag

  if [[ "${mode}" == "supervised" ]]; then
    model_name="${base}_supervised_trial${trial}"
    lr="0.001"
    head_epochs="0"
  else
    case "${ckpt}" in
      *chapman_ii_only*) ckpt_tag="chapman_ii_only" ;;
      *chapman_v2_only*) ckpt_tag="chapman_v2_only" ;;
      *)                 ckpt_tag="custom" ;;
    esac
    model_name="${base}_pretrained_${ckpt_tag}_trial${trial}"
    lr="0.0003"
    head_epochs="5"
  fi

  local run_dir="${RESULTS_ROOT}/${model_name}"
  local sentinel="${run_dir}/_test_metrics_${model_name}.csv"
  if [[ -f "${sentinel}" ]]; then
    skipped+=("${model_name}")
    return 0
  fi

  local sbatch_file="${SBATCH_DIR}/${model_name}.sbatch"
  local pretrained_line=""
  if [[ "${mode}" == "pretrained" ]]; then
    pretrained_line="  --pretrained_model_path=\"${ckpt}\" \\"
  fi

  cat > "${sbatch_file}" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${short}_t${trial}
#SBATCH --time=${WALLTIME}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres=${GRES}
#SBATCH --output=${LOG_ROOT}/${model_name}-%j.out
#SBATCH --error=${LOG_ROOT}/${model_name}-%j.out
set -euo pipefail

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

STAGED_ROOT="\${SLURM_TMPDIR}/dataset"
echo "Staging median dataset from ${DATA_ROOT} to \${STAGED_ROOT} ..."
mkdir -p "\${STAGED_ROOT}"
cp -rL "${DATA_ROOT}/." "\${STAGED_ROOT}/"
echo "Staged \$(du -sh \${STAGED_ROOT} | awk '{print \$1}')"

"${PYTHON_BIN}" -m src.patchECG_finetune \\
  --seed=${seed} \\
  --lr=${lr} \\
  --save_every=20 \\
  --root_path="\${STAGED_ROOT}" \\
  --num_workers=${NUM_WORKERS} \\
  --batch_size=${BATCH_SIZE} \\
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
  --custom_lead_selection="${lead_sel}" \\
  --n_epochs_finetune_head=${head_epochs} \\
  --n_epochs_finetune=${N_EPOCHS} \\
  --n_heads=8 \\
  --n_layers=6 \\
  --d_ff=512 \\
  --patch_len=50 \\
  --stride=50 \\
  --model_name="${model_name}" \\
  --metric_by_class \\
  --no-mentor_mix \\
  --no-bootstrapping \\
  --iterations=5000 \\
  --model_selection_metric=valid_loss \\
${pretrained_line}
SBATCH_EOF

  if [[ "${mode}" != "pretrained" ]]; then
    sed -i 's/--model_selection_metric=valid_loss \\$/--model_selection_metric=valid_loss/' "${sbatch_file}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY-RUN  ${sbatch_file}"
    submitted+=("DRY-RUN  ${model_name}  seed=${seed}")
    return 0
  fi

  local job_id
  job_id=$(sbatch --parsable "${sbatch_file}")
  if [[ -z "${job_id}" ]]; then
    echo "FAILED submission: ${model_name}" >&2
    return 1
  fi
  echo "SUBMITTED  jobid=${job_id}  model=${model_name}  seed=${seed}"
  submitted+=("${job_id}  ${model_name}  seed=${seed}")
}

for variant in "${VARIANTS[@]}"; do
  IFS='|' read -r lead_tag lead_sel mode ckpt short <<< "${variant}"
  for trial in 04 05; do
    seed="${TRIAL_SEED[${trial}]}"
    submit_one "${lead_tag}" "${lead_sel}" "${mode}" "${ckpt}" "${short}" "${trial}" "${seed}"
  done
done

echo
echo "===== Summary ====="
echo "Skipped (sentinel exists): ${#skipped[@]}"
for s in "${skipped[@]}"; do echo "  SKIP  ${s}"; done
echo "Submitted: ${#submitted[@]}"
for s in "${submitted[@]}"; do echo "  ${s}"; done
