# step3e training finished: end-to-end supervised baselines (Narval, 2026-11-24)

All 45 step3e jobs (`FIN_L1supE2E_*`, `FIN_L2supE2E_*`) COMPLETED. No job failed, no loss is NaN, and no run diverged. Existing `FIN_*` runs were not touched.

## 1. Before submission

- Commit `b396bed` (`git pull --ff-only`), no modified tracked files (18 untracked: old Slurm logs and `.bak` copies, as before). The pull changed no training code: `src/core`, `src/patchECG_*.py`, `build_final_sbatches.py`, `check_training_paths.py`, `src/scripts/multi-gpu` and `phase1_folds` are identical to the 92-job run (`4597922`).
- Self-test, CPU job 10370941 (01:42–01:47): `check_training_paths.py` **37 of 37 passed**.
- `build_e2e_sup_sbatches.py`: **45 jobs** in `logs/sbatch_scripts/final_e2e/step3e/`. `jobs.csv` sha256 **`2b8f9f26c0f79c5c5e7bc876820adde95406c953dfd80c074b60532f7d9c7cab`**, byte-identical to `2026-11-24_step3e_dry_run_jobs.csv`. The final runs' `jobs.csv` is unchanged (`2133e920…`).
- Extra check: each of the 45 sbatch files equals its step3/step3o twin except `--n_epochs_finetune_head=0 --n_epochs_finetune=90` and the run ID (45 of 45).

Submitted with `submit_e2e.sh step3e` at 01:47–01:48: Slurm jobs 10371324–10371415 (`submitted_step3e.tsv`; IDs per run in the table below). Ran 01:47–02:26. End-of-batch email: notifier job 10371446 (`afterany` on all 45), log `logs/final/fin_e2e_all_done-10371446.out`: "45 COMPLETED". The code freeze held: the last job started at 02:04:53; the fork's next commits (`f02ca43` 02:22, `e6765a6` 02:25) came after that, touch none of the frozen paths, and were not pulled on Narval while jobs ran.

## 2. Checks over the 45 runs

- Every `job_log.txt` matches `jobs.csv` and `submitted_step3e.tsv`: Slurm job, seed, data folder, `backbone: none`. All logged `git_commit: b396bed`.
- Every run has `_best.pt`, a `finished` line, and 90 metrics rows with epochs exactly 0–89 (no restart, no head-only phase). All train and validation losses are finite. The best epoch in `job_log.txt` matches the minimum `valid_loss` row of the CSV.
- Duration: lead II median 13.8 min (12.4–16.3), V2 13.8 min (12.8–17.2), two-lead 20.9 min (19.2–24.1).

**Seed spread** (best validation loss of S1–S3 within each pipeline × fold, 15 groups): median 1.12×, max 1.44× (L1supE2E lead II fold 4: 0.00036 / 0.00047 / 0.00032). No run is far from its peers.

**Against the twin** (same pipeline, lead, fold, seed and initial weights; 20 head-only + 70 full epochs): best validation loss ratio E2E / twin median 0.99, range 0.78–1.30. By model: L1supE2E lead II 0.99 (0.90–1.30), lead V2 1.01 (0.78–1.11), L2supE2E 0.96 (0.78–1.09). Largest differences: lead II S2 fold 4 1.31, lead II S2 fold 1 1.23, V2 S2 fold 1 0.78, two-lead S2 fold 4 0.78, two-lead S2 fold 3 0.84. Median best validation loss per fold (E2E vs twin): fold 1 0.00076 vs 0.00083, 2 0.00079 vs 0.00078, 3 0.00245 vs 0.00256, 4 0.00041 vs 0.00043, 5 0.00425 vs 0.00420. Folds 3 and 5 are high in both arms, as in every pipeline (both validate on s20141). This is validation loss only; the step3e comparisons are scored on the VM as fixed in `DECISIONS.md`.

**Early best epochs, then overfitting:** best epoch median 44 (9–87), against 62 (30–85) for the twins. After the best epoch, training loss keeps falling (to about 1e-4) while validation loss rises; for example L1supE2E lead II S1 fold 4: best 0.00036 at epoch 11, 0.00105 at epoch 89. Selection on validation loss picks the early checkpoint as specified. This fits the limitation already listed in `DECISIONS.md` (lr 1e-3, not tuned for training from scratch). No change was made.

No NaN loss, no failed job, no outlier run.

## 3. Every run

Best epoch counts 0–89. Validation loss is MSE in mV². "Last" is the validation loss at epoch 89. Twin = the step3/step3o run with 20 + 70 epochs.

| run ID | pipeline | mode | training seed | seed | fold | Slurm job | best epoch | best valid loss | last valid loss | twin | twin best epoch | twin best valid loss |
|---|---|---|---|--:|--:|---|--:|--:|--:|---|--:|--:|
| FIN_L1supE2E_lead_II_S1_fold01 | L1supE2E | lead_II | S1 | 200 | 1 | 10371324 | 24 | 0.00089 | 0.00124 | FIN_L1sup_lead_II_S1_fold01 | 53 | 0.00083 |
| FIN_L1supE2E_lead_II_S1_fold02 | L1supE2E | lead_II | S1 | 201 | 2 | 10371326 | 29 | 0.00039 | 0.00048 | FIN_L1sup_lead_II_S1_fold02 | 67 | 0.00041 |
| FIN_L1supE2E_lead_II_S1_fold03 | L1supE2E | lead_II | S1 | 202 | 3 | 10371327 | 44 | 0.00187 | 0.00225 | FIN_L1sup_lead_II_S1_fold03 | 83 | 0.00173 |
| FIN_L1supE2E_lead_II_S1_fold04 | L1supE2E | lead_II | S1 | 203 | 4 | 10371328 | 11 | 0.00036 | 0.00105 | FIN_L1sup_lead_II_S1_fold04 | 47 | 0.00040 |
| FIN_L1supE2E_lead_II_S1_fold05 | L1supE2E | lead_II | S1 | 204 | 5 | 10371330 | 36 | 0.00410 | 0.00418 | FIN_L1sup_lead_II_S1_fold05 | 82 | 0.00384 |
| FIN_L1supE2E_lead_II_S2_fold01 | L1supE2E | lead_II | S2 | 300 | 1 | 10371331 | 37 | 0.00103 | 0.00126 | FIN_L1sup_lead_II_S2_fold01 | 53 | 0.00084 |
| FIN_L1supE2E_lead_II_S2_fold02 | L1supE2E | lead_II | S2 | 301 | 2 | 10371333 | 21 | 0.00040 | 0.00046 | FIN_L1sup_lead_II_S2_fold02 | 42 | 0.00042 |
| FIN_L1supE2E_lead_II_S2_fold03 | L1supE2E | lead_II | S2 | 302 | 3 | 10371334 | 41 | 0.00164 | 0.00212 | FIN_L1sup_lead_II_S2_fold03 | 80 | 0.00163 |
| FIN_L1supE2E_lead_II_S2_fold04 | L1supE2E | lead_II | S2 | 303 | 4 | 10371335 | 14 | 0.00047 | 0.00085 | FIN_L1sup_lead_II_S2_fold04 | 39 | 0.00036 |
| FIN_L1supE2E_lead_II_S2_fold05 | L1supE2E | lead_II | S2 | 304 | 5 | 10371336 | 70 | 0.00406 | 0.00451 | FIN_L1sup_lead_II_S2_fold05 | 47 | 0.00411 |
| FIN_L1supE2E_lead_II_S3_fold01 | L1supE2E | lead_II | S3 | 400 | 1 | 10371338 | 49 | 0.00072 | 0.00099 | FIN_L1sup_lead_II_S3_fold01 | 49 | 0.00078 |
| FIN_L1supE2E_lead_II_S3_fold02 | L1supE2E | lead_II | S3 | 401 | 2 | 10371340 | 29 | 0.00035 | 0.00039 | FIN_L1sup_lead_II_S3_fold02 | 52 | 0.00038 |
| FIN_L1supE2E_lead_II_S3_fold03 | L1supE2E | lead_II | S3 | 402 | 3 | 10371342 | 81 | 0.00177 | 0.00218 | FIN_L1sup_lead_II_S3_fold03 | 72 | 0.00181 |
| FIN_L1supE2E_lead_II_S3_fold04 | L1supE2E | lead_II | S3 | 403 | 4 | 10371344 | 17 | 0.00032 | 0.00067 | FIN_L1sup_lead_II_S3_fold04 | 42 | 0.00030 |
| FIN_L1supE2E_lead_II_S3_fold05 | L1supE2E | lead_II | S3 | 404 | 5 | 10371345 | 68 | 0.00387 | 0.00438 | FIN_L1sup_lead_II_S3_fold05 | 77 | 0.00390 |
| FIN_L1supE2E_lead_V2_S1_fold01 | L1supE2E | lead_V2 | S1 | 200 | 1 | 10371346 | 9 | 0.00085 | 0.00144 | FIN_L1sup_lead_V2_S1_fold01 | 50 | 0.00098 |
| FIN_L1supE2E_lead_V2_S1_fold02 | L1supE2E | lead_V2 | S1 | 201 | 2 | 10371347 | 50 | 0.00116 | 0.00122 | FIN_L1sup_lead_V2_S1_fold02 | 57 | 0.00115 |
| FIN_L1supE2E_lead_V2_S1_fold03 | L1supE2E | lead_V2 | S1 | 202 | 3 | 10371348 | 71 | 0.00270 | 0.00316 | FIN_L1sup_lead_V2_S1_fold03 | 84 | 0.00306 |
| FIN_L1supE2E_lead_V2_S1_fold04 | L1supE2E | lead_V2 | S1 | 203 | 4 | 10371349 | 22 | 0.00043 | 0.00058 | FIN_L1sup_lead_V2_S1_fold04 | 43 | 0.00042 |
| FIN_L1supE2E_lead_V2_S1_fold05 | L1supE2E | lead_V2 | S1 | 204 | 5 | 10371351 | 59 | 0.00425 | 0.00492 | FIN_L1sup_lead_V2_S1_fold05 | 85 | 0.00400 |
| FIN_L1supE2E_lead_V2_S2_fold01 | L1supE2E | lead_V2 | S2 | 300 | 1 | 10371352 | 9 | 0.00073 | 0.00124 | FIN_L1sup_lead_V2_S2_fold01 | 39 | 0.00093 |
| FIN_L1supE2E_lead_V2_S2_fold02 | L1supE2E | lead_V2 | S2 | 301 | 2 | 10371353 | 59 | 0.00134 | 0.00140 | FIN_L1sup_lead_V2_S2_fold02 | 62 | 0.00120 |
| FIN_L1supE2E_lead_V2_S2_fold03 | L1supE2E | lead_V2 | S2 | 302 | 3 | 10371354 | 84 | 0.00262 | 0.00300 | FIN_L1sup_lead_V2_S2_fold03 | 62 | 0.00256 |
| FIN_L1supE2E_lead_V2_S2_fold04 | L1supE2E | lead_V2 | S2 | 303 | 4 | 10371355 | 24 | 0.00040 | 0.00060 | FIN_L1sup_lead_V2_S2_fold04 | 43 | 0.00046 |
| FIN_L1supE2E_lead_V2_S2_fold05 | L1supE2E | lead_V2 | S2 | 304 | 5 | 10371356 | 81 | 0.00450 | 0.00479 | FIN_L1sup_lead_V2_S2_fold05 | 70 | 0.00421 |
| FIN_L1supE2E_lead_V2_S3_fold01 | L1supE2E | lead_V2 | S3 | 400 | 1 | 10371381 | 37 | 0.00099 | 0.00120 | FIN_L1sup_lead_V2_S3_fold01 | 30 | 0.00097 |
| FIN_L1supE2E_lead_V2_S3_fold02 | L1supE2E | lead_V2 | S3 | 401 | 2 | 10371382 | 61 | 0.00116 | 0.00132 | FIN_L1sup_lead_V2_S3_fold02 | 67 | 0.00129 |
| FIN_L1supE2E_lead_V2_S3_fold03 | L1supE2E | lead_V2 | S3 | 402 | 3 | 10371383 | 64 | 0.00280 | 0.00329 | FIN_L1sup_lead_V2_S3_fold03 | 77 | 0.00280 |
| FIN_L1supE2E_lead_V2_S3_fold04 | L1supE2E | lead_V2 | S3 | 403 | 4 | 10371384 | 22 | 0.00042 | 0.00053 | FIN_L1sup_lead_V2_S3_fold04 | 36 | 0.00043 |
| FIN_L1supE2E_lead_V2_S3_fold05 | L1supE2E | lead_V2 | S3 | 404 | 5 | 10371385 | 60 | 0.00465 | 0.00485 | FIN_L1sup_lead_V2_S3_fold05 | 85 | 0.00422 |
| FIN_L2supE2E_two_lead_S1_fold01 | L2supE2E | two_lead | S1 | 200 | 1 | 10371400 | 87 | 0.00069 | 0.00082 | FIN_L2sup_two_lead_S1_fold01 | 48 | 0.00080 |
| FIN_L2supE2E_two_lead_S1_fold02 | L2supE2E | two_lead | S1 | 201 | 2 | 10371402 | 75 | 0.00073 | 0.00088 | FIN_L2sup_two_lead_S1_fold02 | 75 | 0.00080 |
| FIN_L2supE2E_two_lead_S1_fold03 | L2supE2E | two_lead | S1 | 202 | 3 | 10371403 | 76 | 0.00266 | 0.00318 | FIN_L2sup_two_lead_S1_fold03 | 75 | 0.00279 |
| FIN_L2supE2E_two_lead_S1_fold04 | L2supE2E | two_lead | S1 | 203 | 4 | 10371404 | 35 | 0.00049 | 0.00081 | FIN_L2sup_two_lead_S1_fold04 | 50 | 0.00055 |
| FIN_L2supE2E_two_lead_S1_fold05 | L2supE2E | two_lead | S1 | 204 | 5 | 10371405 | 76 | 0.00451 | 0.00508 | FIN_L2sup_two_lead_S1_fold05 | 76 | 0.00420 |
| FIN_L2supE2E_two_lead_S2_fold01 | L2supE2E | two_lead | S2 | 300 | 1 | 10371406 | 16 | 0.00076 | 0.00102 | FIN_L2sup_two_lead_S2_fold01 | 78 | 0.00072 |
| FIN_L2supE2E_two_lead_S2_fold02 | L2supE2E | two_lead | S2 | 301 | 2 | 10371407 | 57 | 0.00082 | 0.00096 | FIN_L2sup_two_lead_S2_fold02 | 48 | 0.00075 |
| FIN_L2supE2E_two_lead_S2_fold03 | L2supE2E | two_lead | S2 | 302 | 3 | 10371408 | 49 | 0.00240 | 0.00255 | FIN_L2sup_two_lead_S2_fold03 | 61 | 0.00287 |
| FIN_L2supE2E_two_lead_S2_fold04 | L2supE2E | two_lead | S2 | 303 | 4 | 10371409 | 27 | 0.00039 | 0.00057 | FIN_L2sup_two_lead_S2_fold04 | 35 | 0.00050 |
| FIN_L2supE2E_two_lead_S2_fold05 | L2supE2E | two_lead | S2 | 304 | 5 | 10371410 | 84 | 0.00426 | 0.00429 | FIN_L2sup_two_lead_S2_fold05 | 76 | 0.00440 |
| FIN_L2supE2E_two_lead_S3_fold01 | L2supE2E | two_lead | S3 | 400 | 1 | 10371411 | 21 | 0.00072 | 0.00082 | FIN_L2sup_two_lead_S3_fold01 | 73 | 0.00072 |
| FIN_L2supE2E_two_lead_S3_fold02 | L2supE2E | two_lead | S3 | 401 | 2 | 10371412 | 17 | 0.00079 | 0.00096 | FIN_L2sup_two_lead_S3_fold02 | 50 | 0.00078 |
| FIN_L2supE2E_two_lead_S3_fold03 | L2supE2E | two_lead | S3 | 402 | 3 | 10371413 | 82 | 0.00245 | 0.00294 | FIN_L2sup_two_lead_S3_fold03 | 82 | 0.00255 |
| FIN_L2supE2E_two_lead_S3_fold04 | L2supE2E | two_lead | S3 | 403 | 4 | 10371414 | 38 | 0.00041 | 0.00057 | FIN_L2sup_two_lead_S3_fold04 | 65 | 0.00047 |
| FIN_L2supE2E_two_lead_S3_fold05 | L2supE2E | two_lead | S3 | 404 | 5 | 10371415 | 70 | 0.00418 | 0.00455 | FIN_L2sup_two_lead_S3_fold05 | 73 | 0.00457 |

## 4. Copy-back

`results/SHA256SUMS_step3e_copyback.txt` was written on Narval (CPU job 10374694) before any copy: 450 files = 45 runs × 10 (`*_best.pt`, `params.txt`, `job_log.txt`, 7 `_*_metrics*.csv`), repo-relative paths, the same format as `results/SHA256SUMS_final_copyback.txt`. Snapshots and `_every_20_epoch.pt` are not in the list and stay on the cluster. The file is read-only. sha256 of the list: `14af1730c135e5f2b0ebbf7e128bcbbbc103cbf0be55157f6212ff8d5da69a47`. The VM copies the same file set to the same relative paths (`--ignore-existing`) and checks it against this list.
