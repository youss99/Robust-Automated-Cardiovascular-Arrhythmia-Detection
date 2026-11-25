# Podium retraining: dry run (2026-11-24 23:34, VM)

Podium-only exception (`docs/DECISIONS.md` 2026-11-24). Generator `src/scripts/final/build_podium_sbatches.py`, this check `src/scripts/final/dry_run_podium.py`; commit `b6f5aaf` plus the uncommitted podium files. Nothing trained or submitted.

## 1. Jobs

| run ID | mode | fold | seed | data folder | backbone (sha256) | twin |
|---|---|--:|--:|---|---|---|
| POD_L1_lead_II_S1_fold01 | lead_II | 1 | 200 | `cv5_optB_fold01` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S1_fold01 |
| POD_L1_lead_II_S1_fold04 | lead_II | 4 | 203 | `cv5_optB_fold04` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S1_fold04 |
| POD_L1_lead_II_S1_fold05 | lead_II | 5 | 204 | `cv5_optB_fold05` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S1_fold05 |
| POD_L1_lead_V2_S1_fold01 | lead_V2 | 1 | 200 | `cv5_optB_fold01` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S1_fold01 |
| POD_L1_lead_V2_S1_fold04 | lead_V2 | 4 | 203 | `cv5_optB_fold04` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S1_fold04 |
| POD_L1_lead_V2_S1_fold05 | lead_V2 | 5 | 204 | `cv5_optB_fold05` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S1_fold05 |

Reused for folds 2 and 3 (already on the published folders): FIN_L1_lead_II_S1_fold02, _fold03, FIN_L1_lead_V2_S1_fold02, _fold03.

## 2. Job vs FIN twin

Thesis generator unchanged (re-run into a temporary folder: jobs.csv sha256 `2133e92037d6c522…`, approved `2133e920…`): **PASS**.
Job text, with run ID and data folder put back, differs from the twin only by the backbone sha256 guard line: **6 of 6 PASS**.

| job | trainer flags that differ from the twin's params.txt | data folder (job text): twin → job |
|---|---|---|
| POD_L1_lead_II_S1_fold01 | model_name: fin_l1_lead_ii_s1_fold01 → pod_l1_lead_ii_s1_fold01, root_path: /localscratch/youss99.9935846.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s1_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s1_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_II_S1_fold04 | model_name: fin_l1_lead_ii_s1_fold04 → pod_l1_lead_ii_s1_fold04, root_path: /localscratch/youss99.9935849.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s1_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s1_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_II_S1_fold05 | model_name: fin_l1_lead_ii_s1_fold05 → pod_l1_lead_ii_s1_fold05, root_path: /localscratch/youss99.9935850.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s1_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s1_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L1_lead_V2_S1_fold01 | model_name: fin_l1_lead_v2_s1_fold01 → pod_l1_lead_v2_s1_fold01, root_path: /localscratch/youss99.9935863.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s1_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s1_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_V2_S1_fold04 | model_name: fin_l1_lead_v2_s1_fold04 → pod_l1_lead_v2_s1_fold04, root_path: /localscratch/youss99.9935866.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s1_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s1_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_V2_S1_fold05 | model_name: fin_l1_lead_v2_s1_fold05 → pod_l1_lead_v2_s1_fold05, root_path: /localscratch/youss99.9935867.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s1_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s1_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |

94 trainer values compared per job (defaults included). Only model_name, save_path and root_path (Narval scratch) may differ; seed and backbone must equal the twin's: **6 of 6 PASS**.

## 3. Data and loader
- fold 1: `cv5_optB_fold01` CSV sha256 `ef0a0fe6bd6f6095…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 2: `cv5_optB_fold02` CSV sha256 `e7a79c5433ac50cf…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 3: `cv5_optB_fold03` CSV sha256 `810004ab8f689298…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 4: `cv5_optB_fold04` CSV sha256 `8be5d2300ccb9b94…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 5: `cv5_optB_fold05` CSV sha256 `dbabb0a9c830f466…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- H5: one file linked into all five folders, sha256 = SHA256SUMS.txt (`e4bff370e32476bf…`): **PASS**
- reused FIN_L1_lead_II_S1_fold02: job_log dataset `cv5_optB_fold02`, CSV sha = SHA256SUMS.txt, backbone sha `b15f12ef…`, seed 201: **PASS**
- reused FIN_L1_lead_V2_S1_fold02: job_log dataset `cv5_optB_fold02`, CSV sha = SHA256SUMS.txt, backbone sha `a62cb3ec…`, seed 201: **PASS**
- reused FIN_L1_lead_II_S1_fold03: job_log dataset `cv5_optB_fold03`, CSV sha = SHA256SUMS.txt, backbone sha `b15f12ef…`, seed 202: **PASS**
- reused FIN_L1_lead_V2_S1_fold03: job_log dataset `cv5_optB_fold03`, CSV sha = SHA256SUMS.txt, backbone sha `a62cb3ec…`, seed 202: **PASS**

Loader (real `get_dls`), label = the model's own lead (mV). Excluded record–lead pairs keep their real labels in training and validation, as in the published runs.

| fold | mode | split | records | windows | label mean ± SD | GS episodes (own lead) | excluded pairs in split | records |
|--:|---|---|--:|--:|---|--:|---|---|
| 1 | lead_II | train | 11 | 87,205 | -0.0529 ± 0.0968 |  | 20091 II, 20121 II, 20121 V2, 20241 V2 | 20011 20061 20071 20091 20111 20121 20131 20141 20201 20231 20241 |
| 1 | lead_II | val | 2 | 15,405 | -0.0835 ± 0.0399 | 15 | 20081 V2 | 20081 20101 |
| 1 | lead_II | test | 3 | 23,754 | -0.0815 ± 0.1541 |  | 20221 V2 | 20051 20211 20221 |
| 1 | lead_V2 | train | 11 | 87,205 | +0.1147 ± 0.0531 |  | 20091 II, 20121 II, 20121 V2, 20241 V2 | 20011 20061 20071 20091 20111 20121 20131 20141 20201 20231 20241 |
| 1 | lead_V2 | val | 2 | 15,405 | +0.0986 ± 0.0486 | 0 | 20081 V2 | 20081 20101 |
| 1 | lead_V2 | test | 3 | 23,754 | +0.1503 ± 0.1381 |  | 20221 V2 | 20051 20211 20221 |
| 2 | lead_II | train | 11 | 86,485 | -0.0695 ± 0.1173 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20051 20061 20071 20091 20111 20121 20131 20201 20211 20221 20241 |
| 2 | lead_II | val | 2 | 16,636 | -0.0666 ± 0.0557 | 6 |  | 20011 20231 |
| 2 | lead_II | test | 3 | 23,243 | -0.0308 ± 0.0829 |  | 20081 V2 | 20081 20101 20141 |
| 2 | lead_V2 | train | 11 | 86,485 | +0.1237 ± 0.0864 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20051 20061 20071 20091 20111 20121 20131 20201 20211 20221 20241 |
| 2 | lead_V2 | val | 2 | 16,636 | +0.0918 ± 0.0408 | 6 |  | 20011 20231 |
| 2 | lead_V2 | test | 3 | 23,243 | +0.1233 ± 0.0584 |  | 20081 V2 | 20081 20101 20141 |
| 3 | lead_II | train | 11 | 85,631 | -0.0729 ± 0.1121 |  | 20081 V2, 20121 II, 20121 V2, 20221 V2 | 20051 20071 20081 20101 20111 20121 20131 20201 20211 20221 20231 |
| 3 | lead_II | val | 2 | 16,066 | -0.0177 ± 0.0953 | 5 |  | 20011 20141 |
| 3 | lead_II | test | 3 | 24,667 | -0.0527 ± 0.0815 |  | 20091 II, 20241 V2 | 20061 20091 20241 |
| 3 | lead_V2 | train | 11 | 85,631 | +0.1189 ± 0.0825 |  | 20081 V2, 20121 II, 20121 V2, 20221 V2 | 20051 20071 20081 20101 20111 20121 20131 20201 20211 20221 20231 |
| 3 | lead_V2 | val | 2 | 16,066 | +0.1149 ± 0.0665 | 5 |  | 20011 20141 |
| 3 | lead_V2 | test | 3 | 24,667 | +0.1242 ± 0.0674 |  | 20091 II, 20241 V2 | 20061 20091 20241 |
| 4 | lead_II | train | 11 | 86,148 | -0.0613 ± 0.1185 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20011 20051 20071 20091 20101 20121 20141 20201 20211 20221 20241 |
| 4 | lead_II | val | 2 | 16,372 | -0.0580 ± 0.0529 | 11 | 20081 V2 | 20061 20081 |
| 4 | lead_II | test | 3 | 23,844 | -0.0670 ± 0.0853 |  |  | 20111 20131 20231 |
| 4 | lead_V2 | train | 11 | 86,148 | +0.1298 ± 0.0891 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20011 20051 20071 20091 20101 20121 20141 20201 20211 20221 20241 |
| 4 | lead_V2 | val | 2 | 16,372 | +0.1029 ± 0.0420 | 0 | 20081 V2 | 20061 20081 |
| 4 | lead_V2 | test | 3 | 23,844 | +0.0934 ± 0.0319 |  |  | 20111 20131 20231 |
| 5 | lead_II | train | 11 | 87,148 | -0.0682 ± 0.1094 |  | 20081 V2, 20091 II, 20121 II, 20121 V2, 20221 V2 | 20051 20061 20081 20091 20101 20111 20121 20141 20211 20221 20231 |
| 5 | lead_II | val | 2 | 15,861 | -0.0630 ± 0.1057 | 10 | 20241 V2 | 20131 20241 |
| 5 | lead_II | test | 3 | 23,355 | -0.0380 ± 0.0906 |  |  | 20011 20071 20201 |
| 5 | lead_V2 | train | 11 | 87,148 | +0.1210 ± 0.0841 |  | 20081 V2, 20091 II, 20121 II, 20121 V2, 20221 V2 | 20051 20061 20081 20091 20101 20111 20121 20141 20211 20221 20231 |
| 5 | lead_V2 | val | 2 | 15,861 | +0.1432 ± 0.0760 | 0 | 20241 V2 | 20131 20241 |
| 5 | lead_V2 | test | 3 | 23,355 | +0.0976 ± 0.0405 |  |  | 20011 20071 20201 |

## 4. Self-test

`check_training_paths.py`: **37 of 37 checks passed** (exit 0): **PASS**

## 5. The checks fail on injected faults
- text: lr changed: flagged
- text: epochs 70 -> 90: flagged
- flags: weight_decay changed: flagged
- flags: seed changed: flagged

## 6. Time estimate

Twins took 12–14 min each (1.3 GPU-hours for the 6); walltime 45 min.

## Result: ALL PASS

jobs.csv sha256: `5c57435ab7dba66e3d2eaf2e4ef934a50b419a6a0b6693d0acb0035ace36c38b` (Narval must reproduce it before submitting). A byte copy is next to this report.
