#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Narval SLURM launcher for the full-16 median LTSTDB regression study.
#
# Submits ONE sbatch job per (lead_variant x training_mode x trial), 6x3 = 18
# combinations total. Idempotent: combos whose _test_metrics_<model>.csv
# sentinel already exists are skipped, so re-invoking the script only fills
# in the missing runs.
#
# Variants (all on median targets, full-16 patient split):
#   1. two_lead, supervised
#   2. two_lead, pretrained from patch_ecg_chapman_ii_v2_tta
#   3. lead_II,  supervised
#   4. lead_II,  pretrained from patch_ecg_chapman_ii_only
#   5. lead_V2,  supervised
#   6. lead_V2,  pretrained from patch_ecg_chapman_v2_only
#
# Trials -> seeds:   trial01=200, trial02=201, trial03=202
#
# SLURM profile (matches trial01 jobs 61228551-554):
#   --account=def-majidk
#   --partition=gpubase_b1
#   --time=03:00:00
#   --cpus-per-task=4 --mem=20G
#   --gres=gpu:a100_3g.20gb:1
#
# Dataset is copied to $SLURM_TMPDIR on the compute node at job start
# (~966 MB on disk -> ~700 MB after copy in prior runs).
#
# Per-job sbatch script is written under logs/sbatch_scripts/ for traceability.
# Per-job stdout/stderr go to logs/<model_name>-<jobid>.out.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
# Dataset source: the in-repo custom_datasets H5 was truncated to 0 bytes on May 19;
# the full 966M copy lives on /scratch. Stage from there.
DATA_ROOT="/scratch/youss99/ltstdb_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

PRETRAIN_ROOT="${REPO_ROOT}/results/pre-train/saved_models/chapman"
CHAPMAN_TWOLEAD_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"
CHAPMAN_II_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_only/patch_ecg_chapman_ii_only_best.pt"
CHAPMAN_V2_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_v2_only/patch_ecg_chapman_v2_only_best.pt"

RESULTS_ROOT="${REPO_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
LOG_ROOT="${REPO_ROOT}/logs"
SBATCH_DIR="${LOG_ROOT}/sbatch_scripts"

ACCOUNT="def-majidk"
# Partition omitted on purpose: Narval's scheduler auto-routes GPU jobs to the
# right partition based on --gres (the legacy gpubase_b1 alias no longer exists).
WALLTIME="03:00:00"
MEM="20G"
CPUS_PER_TASK="4"
GRES="gpu:a100_3g.20gb:1"

N_EPOCHS=40
BATCH_SIZE=128
NUM_WORKERS=4

DRY_RUN="${DRY_RUN:-0}"

declare -A TRIAL_SEED=( [01]=200 [02]=201 [03]=202 )

# Each variant: lead_tag|lead_sel|mode|pretrained_ckpt|short_jobname_prefix
VARIANTS=(
  "two_lead|II,V2|supervised||ltst_med_2L_sup"
  "two_lead|II,V2|pretrained|${CHAPMAN_TWOLEAD_CKPT}|ltst_med_2L_pre"
  "lead_II|II|supervised||ltst_med_II_sup"
  "lead_II|II|pretrained|${CHAPMAN_II_CKPT}|ltst_med_II_pre"
  "lead_V2|V2|supervised||ltst_med_V2_sup"
  "lead_V2|V2|pretrained|${CHAPMAN_V2_CKPT}|ltst_med_V2_pre"
)

# Pre-flight
fail=0
[[ -d "${REPO_ROOT}" ]]      || { echo "ERROR: REPO_ROOT missing: ${REPO_ROOT}" >&2; fail=1; }
[[ -x "${PYTHON_BIN}" ]]     || { echo "ERROR: python venv missing: ${PYTHON_BIN}" >&2; fail=1; }
[[ -d "${DATA_ROOT}" ]]      || { echo "ERROR: dataset missing: ${DATA_ROOT}" >&2; fail=1; }
for ckpt in "${CHAPMAN_TWOLEAD_CKPT}" "${CHAPMAN_II_CKPT}" "${CHAPMAN_V2_CKPT}"; do
  [[ -f "${ckpt}" ]] || { echo "ERROR: pretrained ckpt missing: ${ckpt}" >&2; fail=1; }
done
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
      *chapman_ii_v2_tta*) ckpt_tag="chapman_ii_v2" ;;
      *chapman_ii_only*)   ckpt_tag="chapman_ii_only" ;;
      *chapman_v2_only*)   ckpt_tag="chapman_v2_only" ;;
      *)                   ckpt_tag="custom" ;;
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
cp -r "${DATA_ROOT}/." "\${STAGED_ROOT}/"
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

  # Strip the trailing backslash on the last python arg if no pretrained line
  if [[ "${mode}" != "pretrained" ]]; then
    # Remove the final " \\" left by --model_selection_metric=valid_loss
    sed -i 's/--model_selection_metric=valid_loss \\$/--model_selection_metric=valid_loss/' "${sbatch_file}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY-RUN  ${sbatch_file}  (model=${model_name}, seed=${seed})"
    submitted+=("DRY-RUN  ${model_name}")
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
  for trial in 01 02 03; do
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
