# Phase 1: notes for the Narval session

Read `README_phase1_runs.md` first. These are the user's decisions of 2026-11-08 and what was checked on the VM. Where the two files differ, this note wins.

**Your scope:** steps 0–3. The VM does step 4, builds the v2 data for step 5, and does the LTST record inference and all scoring. Don't start R3.

## Setup

1. Check `git status` is clean, then pull `update-initial-changes-with-Mitch`.
2. Link the published median H5 into the fold folders:
   `bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5`
   It checks the H5 and the fold CSVs against `phase1_folds/SHA256SUMS.txt`. If Narval has no matching H5, ask the user to copy the VM's: `custom_datasets/ltstdb_regression_median_100hz_nocompress/PatientSegmentRegressionDataset.h5`.
3. Account, GPU and paths: `build_phase1_sbatches.py` defaults to `def-majidk`, `a100_3g.20gb` and the `Project_1` repo path. Confirm them with the user before the first submission.

## Step 0

From the repo root, with the repo's python: `.venv/bin/python src/scripts/custom_dataset/check_rebuilt_recipe.py`

It runs the real pretraining parser on each `_only` script's flags and compares every value of `_rebuilt/params.txt` except names and paths. Replacements: `ii_only` → `ii_rebuilt`, `v2_only` → `v2_rebuilt`, `ii_v2` → `ii_v2_rebuilt`. If anything differs, stop and report.

## Steps 1–3

**Jobs:** `python3 src/scripts/custom_dataset/build_phase1_sbatches.py` writes R1b (30), R2 (15) and R2b (3) to `logs/sbatch_scripts/phase1/`, with `jobs.csv`. Overrides: `P1_REPO_ROOT`, `P1_DS_ROOT`, `P1_ACCOUNT`, `P1_GPU`, `P1_WALLTIME`, `P1_OUT`. Don't pass `--with_r3`. Each job writes `job_log.txt` in its run folder.

**Recipe:** the published `params.txt` (20 head + 70 full epochs, lr 1e-3, seed 200 + fold − 1). The generator uses it. Don't take the recipe from `build_cv5_train_sbatches.py`: its template shows 40 epochs, which is not what the published runs recorded. On the VM, the generated flags matched the published `params.txt` through the real parser, apart from names, paths and the new `dump_logits` default (off).

**Data:** `phase1_folds/`. `cv5_optB_fold01`–`05` for R1b and R2, `cv5_optB_leadV2_opt2p_fold01`, `04`, `05` for R2b. On the VM, all 30 published checkpoints reproduced their saved test MSE on these folds (worst relative difference 0.00062), and the real loader ran on them.

**Backbones:** R1b needs `patch_ecg_unified_{ii_only,v2_only,ii_v2}`, R2 and R2b need the `_rebuilt` ones, under `results/pre-train/saved_models/unified/`. Verify checksums when copying. If you write your own job scripts, keep the "backbone missing" check: `patchECG_finetune.py` silently trains from scratch when the given backbone path doesn't exist.

**Code version:** R1b runs at current HEAD. Record the commit (`job_log.txt` does). If R1b doesn't reproduce the paper, the code version or the cluster is the suspect.

**Masked pairs:** the published runs kept them in training and validation with their real labels. `MANUAL_OMIT` applies only at FSM scoring. Change nothing for this. R2b's `unused` records are already −1 in its fold CSVs.

**Supervised runs:** the published ones have `pretrained_model_path=None`, and that path is the only place weights are loaded, so R2 reuses R1b's supervised runs.

**Dry run:** as the README says, run the loader on one fold, report records per split, windows per split, label mean and SD per split and the exact job list, and wait for the user's OK before submitting.

## Predictions and what to copy back

This replaces "Predictions to save, per run" in the README.

- **Do:** let each job's normal test evaluation run at the end of training. It writes `_test_metrics_*.csv`.
- **Don't:** run inference on the LTST records or save out-of-fold prediction files. The VM does that, for the FSM, gate and alarm scoring and for the per-window regression metrics. It also checks that its test MSE matches the saved one.
- **Copy back per run, to the same relative path** `results/fine-tune/saved_models/patient-segment-regression/<run_id>/`: `*_best.pt`, `params.txt`, the `_*_metrics*.csv` files and `job_log.txt`. Leave `_snapshot.pt` and `_every_20_epoch.pt` on the cluster.
