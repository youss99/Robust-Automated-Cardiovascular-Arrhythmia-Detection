#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Narval SLURM launcher for 5-fold patient-grouped CV on the two-lead median
# LTSTDB regression task.
#
# Per fold: train (12 patients = 5 omit + 7 clean), val (2 clean), test (2-3 clean).
# 2 modes (supervised, pretrained_chapman_ii_v2) x 5 folds = 10 training runs.
#
# Each fold's dataset lives under /scratch/youss99/ltstdb_datasets/cv5_fold0X/
# (pre-built with prepare_fold_dataset.py; H5 hardlinked from the full16 source).
#
# 1 trial per fold (no in-fold trial averaging) - cross-fold variance is the
# new variance estimate.
#
# Sentinel skip + same SLURM profile as the 5-trial wrapper.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

PRETRAIN_ROOT="${REPO_ROOT}/results/pre-train/saved_models/chapman"
CHAPMAN_TWOLEAD_CKPT="${PRETRAIN_ROOT}/patch_ecg_chapman_ii_v2_tta/patch_ecg_chapman_ii_v2_tta_best.pt"

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

# Common seed across folds (we'll vary the data split, not the seed).
SEED="${SEED:-200}"

DRY_RUN="${DRY_RUN:-0}"

FOLDS=(01 02 03 04 05)
declare -A FOLD_ROOT
for f in "${FOLDS[@]}"; do
  FOLD_ROOT[$f]="/scratch/youss99/ltstdb_datasets/cv5_fold${f}"
done

# Variants: mode|short_jobname_prefix|ckpt_path(empty for supervised)
VARIANTS=(
  "supervised|ltst_cv5_2L_sup|"
  "pretrained|ltst_cv5_2L_pre|${CHAPMAN_TWOLEAD_CKPT}"
)

# Pre-flight
fail=0
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: python venv missing: ${PYTHON_BIN}" >&2; fail=1; }
[[ -f "${CHAPMAN_TWOLEAD_CKPT}" ]] || { echo "ERROR: pretrained ckpt missing: ${CHAPMAN_TWOLEAD_CKPT}" >&2; fail=1; }
for f in "${FOLDS[@]}"; do
  d="${FOLD_ROOT[$f]}"
  [[ -d "${d}" ]] || { echo "ERROR: fold dataset missing: ${d}" >&2; fail=1; }
  [[ -f "${d}/PatientSegmentRegressionDataset.h5" ]] || { echo "ERROR: fold H5 missing: ${d}/PatientSegmentRegressionDataset.h5" >&2; fail=1; }
done
[[ "${fail}" -ne 0 ]] && exit 2

mkdir -p "${LOG_ROOT}" "${SBATCH_DIR}" "${RESULTS_ROOT}"

submitted=()
skipped=()

submit_one() {
  local mode="$1" short="$2" ckpt="$3" fold="$4"
  local data_root="${FOLD_ROOT[$fold]}"
  local model_name lr head_epochs ckpt_tag base

  base="ltstdb_median_cv5_two_lead_regression"
  if [[ "${mode}" == "supervised" ]]; then
    model_name="${base}_supervised_fold${fold}"
    lr="0.001"
    head_epochs="0"
  else
    ckpt_tag="chapman_ii_v2"
    model_name="${base}_pretrained_${ckpt_tag}_fold${fold}"
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
#SBATCH --job-name=${short}_f${fold}
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
echo "Staging fold ${fold} from ${data_root} to \${STAGED_ROOT} ..."
mkdir -p "\${STAGED_ROOT}"
# -L: dereference symlinks/hardlinks so the H5 is materialized on local scratch.
cp -rL "${data_root}/." "\${STAGED_ROOT}/"
echo "Staged \$(du -sh \${STAGED_ROOT} | awk '{print \$1}')"

"${PYTHON_BIN}" -m src.patchECG_finetune \\
  --seed=${SEED} \\
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
  --custom_lead_selection="II,V2" \\
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
    echo "DRY-RUN  ${sbatch_file}  (fold=${fold}, mode=${mode})"
    submitted+=("DRY-RUN  ${model_name}")
    return 0
  fi

  local job_id
  job_id=$(sbatch --parsable "${sbatch_file}")
  if [[ -z "${job_id}" ]]; then
    echo "FAILED submission: ${model_name}" >&2
    return 1
  fi
  echo "SUBMITTED  jobid=${job_id}  model=${model_name}"
  submitted+=("${job_id}  ${model_name}")
}

for variant in "${VARIANTS[@]}"; do
  IFS='|' read -r mode short ckpt <<< "${variant}"
  for fold in "${FOLDS[@]}"; do
    submit_one "${mode}" "${short}" "${ckpt}" "${fold}"
  done
done

echo
echo "===== Summary ====="
echo "Skipped (sentinel exists): ${#skipped[@]}"
for s in "${skipped[@]}"; do echo "  SKIP  ${s}"; done
echo "Submitted: ${#submitted[@]}"
for s in "${submitted[@]}"; do echo "  ${s}"; done
