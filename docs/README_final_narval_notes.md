# Final thesis runs: notes for the Narval session (2026-11-20)

The repo is the single source of truth. Instructions: `docs/instructions_final_results.md`. Decisions and approvals: `docs/DECISIONS.md`. Reports: `docs/reports/`. This note is the cluster part. **Your scope:** steps 0, 0b, 1, 2 and 3 (S1). `step2b` and `step3o` are not approved yet. The VM does steps 4 to 7: R-peak cache, NeuroKit pipelines, all LTST inference and scoring. Don't run inference on LTST records.

Ground rules from the instructions: ask before guessing; never overwrite data, checkpoints or results (every run has its own folder, named by its ID); log every job; report when training finishes, and flag a diverged run or a failed check straight away (see "Reports").

## Setup

1. `git status` clean, then pull `update-initial-changes-with-Mitch`. Record the commit.

   **Remote names:** the fork (github.com/youss99) is named `origin` on Narval and `my-fork` on the VM; both use branch `update-initial-changes-with-Mitch`.
2. Link the published median H5 into the 8 fold folders the jobs use:
   `bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5`. It checks the H5 and those folders' CSVs against `phase1_folds/SHA256SUMS.txt`. The three `cv5_optB_leadII_opt2p_fold*` folders stay in the repo but are not used, checked or linked.

## Step 0: self-test

`.venv/bin/python src/scripts/checks/check_training_paths.py` must print 37/37. If not, stop and report.

## Step 0b: L2's backbone

```
for s in 0 1 2; do
  .venv/bin/python src/scripts/checks/backbone_masked_vs_visible.py \
    --backbone results/pre-train/saved_models/unified/patch_ecg_unified_ii_v2_rebuilt/patch_ecg_unified_ii_v2_rebuilt_best.pt \
    --h5 <Unified_Dataset_No_PTB_Segment_v2>/UnifiedDataset.h5 --leads II,V2 --seed $s
done
```

Expected: "does not reproduce visible patches", ratio 1.70 or more (readings move up to about 21 % between draws). If it reproduces them, stop and report.

## Jobs

`python3 src/scripts/custom_dataset/build_final_sbatches.py` writes every job to `logs/sbatch_scripts/final/<group>/`, plus `jobs.csv` and `submit_final.sh`. Overrides: `FIN_REPO_ROOT`, `FIN_DS_ROOT`, `FIN_ACCOUNT`, `FIN_GPU`, `FIN_PT_GPU`, `FIN_WALLTIME`, `FIN_PT_WALLTIME`, `FIN_OUT`. With the defaults, `jobs.csv` is byte-identical to the one the VM dry run used (sha256 `2133e92037d6c52239f4357d3b81b1440c8a88815d45bfb14f688895835480f8`; copy: `docs/reports/2026-11-20_dry_run_jobs.csv`). If it isn't, say so before submitting.

| Group | Jobs | What | Approved | Submit when |
|---|--:|---|---|---|
| `step1` | 2 | `patch_ecg_unified_{ii,v2}_lossfix`: `pretrain_unified_v2.sh` unchanged, called from a wrapper that writes `job_log.txt` | yes | step 0 passed |
| `step3` | 15 | L1-sup (lead II, V2) and L2-sup (two-lead), supervised, S1 | yes | step 0 passed |
| `step2_L2` | 15 | L2: two-lead from `patch_ecg_unified_ii_v2_rebuilt`, S1, S2, S3 | yes | step 0b passed |
| `step2_L1` | 30 | L1: lead II from `ii_lossfix`, V2 from `v2_lossfix`, S1, S2, S3 | yes | step 1 checks passed (below) |
| `step2b` | 10 | optional: L1-rebuilt, `_rebuilt` backbones, S1 | not yet | only once approved in `docs/DECISIONS.md` |
| `step3o` | 30 | optional: L1-sup, L2-sup at S2, S3 | not yet | only once approved in `docs/DECISIONS.md` |

Submit with `bash logs/sbatch_scripts/final/submit_final.sh <group>`. It refuses a group that was already submitted, and records the job IDs in `submitted_<group>.tsv`.

**Seeds:** S1 = 200 + fold − 1 (published), S2 = 300 + fold − 1, S3 = 400 + fold − 1. Pre-training: none.

**Splits:** one identical split for every learned model (lead II, V2, two-lead; pretrained and supervised). Folds 1, 4 and 5 use `cv5_optB_leadV2_opt2p_fold*`: the masked record (s20081, s20081, s20241) is unused for every model, and the moved record (s20231, s20071, s20141) is in validation. Folds 2 and 3 use the published `cv5_optB_fold02`/`03`. No validation record has a masked lead, and test sets are unchanged. Gold-standard episodes per lead in each validation set (required before training): `docs/reports/2026-11-20_dry_run.md`, section 5.

**Step 1 checks, before `step2_L1`:**
- `job_log.txt` of each backbone: the logged loss starts near 1 and falls. Don't compare it with old single-lead logs (they show the mixed loss).
- `backbone_masked_vs_visible.py` on each backbone with `--leads II` (or `V2`) and `--seed` 0, 1, 2 must read "does not reproduce visible patches".
- For L1-rebuilt (step 2b) to be a one-change comparison: `diff <(tr ',' '\n' < .../patch_ecg_unified_ii_rebuilt/params.txt) <(tr ',' '\n' < .../patch_ecg_unified_ii_lossfix/params.txt)`, and the same for v2. Only `model_name`, `save_path` and `root_path` may differ. Compare `data_source.txt` too.

## Reports

Write every report to `docs/reports/<date>_<step>.md` and push it, so the VM can pull it directly:
- `<date>_step0_0b.md`: the step 0 result, the three step 0b readings, and the groups submitted with their job IDs (from `submitted_<group>.tsv`). The VM starts steps 4 and 5 once this report is pushed.
- `<date>_training.md`, once all approved training has finished: the step 1 checks (first and last logged loss; three backbone-tool readings per backbone), the `step2_L1` submission, and one table with each run's best epoch and best validation loss (from `job_log.txt`). Flag NaN losses, failed jobs, and any run far from its peers (R1b had 3 diverged supervised lead II runs).
- A failed check or a diverged run gets its own report straight away (for example `<date>_step1.md`), and the affected group stops.

Commit only the report file, then `git pull --rebase origin update-initial-changes-with-Mitch` and `git push origin update-initial-changes-with-Mitch` (on Narval the fork is `origin`).

## Copy back

Same relative paths. For each fine-tune run: `*_best.pt`, `params.txt`, `_*_metrics*.csv` and `job_log.txt`. For each backbone: `*_best.pt`, `params.txt`, `job_log.txt`, `data_source.txt`, `pretrain_split.csv` and the `_training_metrics_*.csv`. Leave snapshots on the cluster.

## step3e: end-to-end supervised baselines (added 2026-11-24)

Decision and fixed comparisons: `docs/DECISIONS.md`, 2026-11-24. VM checks: `docs/reports/2026-11-24_step3e_verification.md`. **Submit only once `DECISIONS.md` records the user's final approval of step3e.**

45 jobs, `FIN_L1supE2E_{lead_II,lead_V2}_S{1,2,3}_fold0{1..5}` and `FIN_L2supE2E_two_lead_S{1,2,3}_fold0{1..5}`. Each is its step3/step3o twin with `--n_epochs_finetune_head=0 --n_epochs_finetune=90`; same data folders (already linked for the final runs), seeds, flags, walltime and GPU.

1. `git status` clean, pull, record the commit. No training code changed (only new scripts and docs).
2. Self-test: `.venv/bin/python src/scripts/checks/check_training_paths.py` must still print 37/37.
3. `python3 src/scripts/custom_dataset/build_e2e_sup_sbatches.py` writes `logs/sbatch_scripts/final_e2e/step3e/*.sbatch`, `jobs.csv` and `submit_e2e.sh` (the existing `logs/sbatch_scripts/final/` is not touched). With the default paths, `sha256sum logs/sbatch_scripts/final_e2e/jobs.csv` must equal the sha in the verification report (copy: `docs/reports/2026-11-24_step3e_dry_run_jobs.csv`). If it doesn't, stop and report.
4. `bash logs/sbatch_scripts/final_e2e/submit_e2e.sh step3e` (refuses a second submission; job IDs in `submitted_step3e.tsv`).
5. Report `docs/reports/<date>_step3e_training.md`: commit, job IDs, and one table with each run's best epoch and best validation loss (`job_log.txt`; the epoch column runs 0–89 with no restart). Flag NaN losses, failed jobs and runs far from their peers. Code freeze as before: no commits to `src/core`, `src/patchECG_*.py`, job scripts or fold manifests until every step3e job has started.
6. Copy back as in "Copy back", with a `SHA256SUMS` file written on Narval before rsync.

## podium: Learned-1L S1 on the published folds (added 2026-11-24)

Podium-only exception, approved in `docs/DECISIONS.md` (2026-11-24). VM dry run: `docs/reports/2026-11-24_podium_dry_run.md` (copy of jobs.csv: `docs/reports/2026-11-24_podium_dry_run_jobs.csv`).

6 jobs, `POD_L1_{lead_II,lead_V2}_S1_fold0{1,4,5}`. Each is its FIN twin `FIN_L1_<mode>_S1_fold0X` with the data folder `phase1_folds/cv5_optB_fold0X` (published folds; H5 already linked by `link_phase1_h5.sh`), its own run ID, and one guard line that stops the job unless the backbone sha256 is the twin's (ii `b15f12ef…`, v2 `a62cb3ec…`). Folds 2 and 3 are not retrained.

1. `git status` clean, pull, record the commit. No training code changed (only new scripts and docs).
2. Self-test: `.venv/bin/python src/scripts/checks/check_training_paths.py` must print 37/37.
3. `python3 src/scripts/final/build_podium_sbatches.py` writes `logs/sbatch_scripts/podium/podium/*.sbatch` (6), `jobs.csv` and `submit_podium.sh`. `sha256sum logs/sbatch_scripts/podium/jobs.csv` must be `5c57435ab7dba66e3d2eaf2e4ef934a50b419a6a0b6693d0acb0035ace36c38b`. If any condition fails, stop and report.
4. `bash logs/sbatch_scripts/podium/submit_podium.sh podium` (refuses a second submission; job IDs in `submitted_podium.tsv`).
5. Code freeze: no commits to `src/core`, `src/patchECG_*.py`, job scripts or fold manifests until all 6 jobs have started.
6. Report `docs/reports/<date>_podium_training.md`: commit, job IDs, and per job the resolved config (`params.txt`), seed, backbone path + sha256, best epoch and best validation loss (`job_log.txt`). Flag NaN losses or failed jobs.
7. Copy back as in "Copy back", with a `SHA256SUMS` file written on Narval before rsync.

## pubfolds: remaining pretrained runs on the published folds (added 2026-11-25)

Exception approved in `docs/DECISIONS.md` (2026-11-25). VM dry run: `docs/reports/2026-11-25_pubfolds_dry_run.md` (copy of jobs.csv: `docs/reports/2026-11-25_pubfolds_dry_run_jobs.csv`).

21 jobs: `POD_L1_{lead_II,lead_V2}_{S2,S3}_fold0{1,4,5}` (12, `_lossfix` backbones) and `POD_L2_two_lead_{S1,S2,S3}_fold0{1,4,5}` (9, `patch_ecg_unified_ii_v2_rebuilt`). Each is its FIN twin `FIN_<pipeline>_<mode>_<seed>_fold0X` with the data folder `phase1_folds/cv5_optB_fold0X`, its own run ID, and one guard line that stops the job unless the backbone sha256 is the twin's (ii `b15f12ef…`, v2 `a62cb3ec…`, ii_v2 `f3554f14…`); built exactly like the podium jobs. Not retrained: `POD_L1_*_S1` (podium) and every FIN run on folds 2 and 3.

1. `git status` clean, pull, record the commit. No training code changed (only new scripts and docs).
2. Self-test: `.venv/bin/python src/scripts/checks/check_training_paths.py` must print 37/37.
3. `python3 src/scripts/final/build_pubfolds_sbatches.py` writes `logs/sbatch_scripts/pubfolds/pubfolds/*.sbatch` (21), `jobs.csv` and `submit_pubfolds.sh`. `sha256sum logs/sbatch_scripts/pubfolds/jobs.csv` must be `e9c5eedc0251ce3471daa1896362777e9935da4e6b5289527ee16a9ac30bcb82`. If any condition fails, stop and report.
4. `bash logs/sbatch_scripts/pubfolds/submit_pubfolds.sh pubfolds` (refuses a second submission; job IDs in `submitted_pubfolds.tsv`).
5. Code freeze: no commits to `src/core`, `src/patchECG_*.py`, job scripts or fold manifests until all 21 jobs have started.
6. Report `docs/reports/<date>_pubfolds_training.md`: commit, job IDs, and per job the resolved config (`params.txt`), seed, backbone path + sha256, best epoch and best validation loss (`job_log.txt`). Flag NaN losses, failed jobs and runs far from their FIN twin's best validation loss.
7. Copy back as in "Copy back", with a `SHA256SUMS` file written on Narval before rsync.
