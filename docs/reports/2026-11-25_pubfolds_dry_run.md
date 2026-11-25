# Published-folds extension: dry run (2026-11-25 05:46, VM)

Exception to the plan freeze (`docs/DECISIONS.md` 2026-11-25): the remaining pretrained runs on the published folds. Generator `src/scripts/final/build_pubfolds_sbatches.py`, this check `src/scripts/final/dry_run_pubfolds.py`; commit `828ceaf` plus the uncommitted files. Nothing trained or submitted.

## 1. Jobs

| run ID | pipeline | mode | training seed | fold | seed | data folder | backbone (sha256) | twin |
|---|---|---|---|--:|--:|---|---|---|
| POD_L1_lead_II_S2_fold01 | L1 | lead_II | S2 | 1 | 300 | `cv5_optB_fold01` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S2_fold01 |
| POD_L1_lead_II_S2_fold04 | L1 | lead_II | S2 | 4 | 303 | `cv5_optB_fold04` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S2_fold04 |
| POD_L1_lead_II_S2_fold05 | L1 | lead_II | S2 | 5 | 304 | `cv5_optB_fold05` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S2_fold05 |
| POD_L1_lead_II_S3_fold01 | L1 | lead_II | S3 | 1 | 400 | `cv5_optB_fold01` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S3_fold01 |
| POD_L1_lead_II_S3_fold04 | L1 | lead_II | S3 | 4 | 403 | `cv5_optB_fold04` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S3_fold04 |
| POD_L1_lead_II_S3_fold05 | L1 | lead_II | S3 | 5 | 404 | `cv5_optB_fold05` | `patch_ecg_unified_ii_lossfix` (`b15f12ef…`) | FIN_L1_lead_II_S3_fold05 |
| POD_L1_lead_V2_S2_fold01 | L1 | lead_V2 | S2 | 1 | 300 | `cv5_optB_fold01` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S2_fold01 |
| POD_L1_lead_V2_S2_fold04 | L1 | lead_V2 | S2 | 4 | 303 | `cv5_optB_fold04` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S2_fold04 |
| POD_L1_lead_V2_S2_fold05 | L1 | lead_V2 | S2 | 5 | 304 | `cv5_optB_fold05` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S2_fold05 |
| POD_L1_lead_V2_S3_fold01 | L1 | lead_V2 | S3 | 1 | 400 | `cv5_optB_fold01` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S3_fold01 |
| POD_L1_lead_V2_S3_fold04 | L1 | lead_V2 | S3 | 4 | 403 | `cv5_optB_fold04` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S3_fold04 |
| POD_L1_lead_V2_S3_fold05 | L1 | lead_V2 | S3 | 5 | 404 | `cv5_optB_fold05` | `patch_ecg_unified_v2_lossfix` (`a62cb3ec…`) | FIN_L1_lead_V2_S3_fold05 |
| POD_L2_two_lead_S1_fold01 | L2 | two_lead | S1 | 1 | 200 | `cv5_optB_fold01` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S1_fold01 |
| POD_L2_two_lead_S1_fold04 | L2 | two_lead | S1 | 4 | 203 | `cv5_optB_fold04` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S1_fold04 |
| POD_L2_two_lead_S1_fold05 | L2 | two_lead | S1 | 5 | 204 | `cv5_optB_fold05` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S1_fold05 |
| POD_L2_two_lead_S2_fold01 | L2 | two_lead | S2 | 1 | 300 | `cv5_optB_fold01` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S2_fold01 |
| POD_L2_two_lead_S2_fold04 | L2 | two_lead | S2 | 4 | 303 | `cv5_optB_fold04` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S2_fold04 |
| POD_L2_two_lead_S2_fold05 | L2 | two_lead | S2 | 5 | 304 | `cv5_optB_fold05` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S2_fold05 |
| POD_L2_two_lead_S3_fold01 | L2 | two_lead | S3 | 1 | 400 | `cv5_optB_fold01` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S3_fold01 |
| POD_L2_two_lead_S3_fold04 | L2 | two_lead | S3 | 4 | 403 | `cv5_optB_fold04` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S3_fold04 |
| POD_L2_two_lead_S3_fold05 | L2 | two_lead | S3 | 5 | 404 | `cv5_optB_fold05` | `patch_ecg_unified_ii_v2_rebuilt` (`f3554f14…`) | FIN_L2_two_lead_S3_fold05 |

Reused, not retrained: POD_L1_lead_{II,V2}_S1_fold0{1,4,5} (podium, trained 2026-11-25) and FIN_{L1_lead_II, L1_lead_V2, L2_two_lead}_{S1,S2,S3}_fold0{2,3} (already on the published folders). With these, every pretrained pipeline has 5 folds × S1–S3 on the published split (45 runs; 21 new).

## 2. Job vs FIN twin

Thesis generator unchanged (re-run into a temporary folder: jobs.csv sha256 `2133e92037d6c522…`, approved `2133e920…`): **PASS**.
Podium generator unchanged (re-run: jobs.csv sha256 `5c57435ab7dba66e…`, approved `5c57435a…`): **PASS**. The new generator asserts that its render of the 6 podium jobs (L1 S1) is byte-identical to `build_podium_sbatches.render`.
Job text, with run ID and data folder put back, differs from the twin only by the backbone sha256 guard line: **21 of 21 PASS**.

| job | trainer flags that differ from the twin's params.txt | data folder (job text): twin → job |
|---|---|---|
| POD_L1_lead_II_S2_fold01 | model_name: fin_l1_lead_ii_s2_fold01 → pod_l1_lead_ii_s2_fold01, root_path: /localscratch/youss99.9935851.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s2_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s2_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_II_S2_fold04 | model_name: fin_l1_lead_ii_s2_fold04 → pod_l1_lead_ii_s2_fold04, root_path: /localscratch/youss99.9935854.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s2_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s2_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_II_S2_fold05 | model_name: fin_l1_lead_ii_s2_fold05 → pod_l1_lead_ii_s2_fold05, root_path: /localscratch/youss99.9935855.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s2_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s2_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L1_lead_II_S3_fold01 | model_name: fin_l1_lead_ii_s3_fold01 → pod_l1_lead_ii_s3_fold01, root_path: /localscratch/youss99.9935858.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s3_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s3_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_II_S3_fold04 | model_name: fin_l1_lead_ii_s3_fold04 → pod_l1_lead_ii_s3_fold04, root_path: /localscratch/youss99.9935861.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s3_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s3_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_II_S3_fold05 | model_name: fin_l1_lead_ii_s3_fold05 → pod_l1_lead_ii_s3_fold05, root_path: /localscratch/youss99.9935862.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_ii_s3_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_ii_s3_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L1_lead_V2_S2_fold01 | model_name: fin_l1_lead_v2_s2_fold01 → pod_l1_lead_v2_s2_fold01, root_path: /localscratch/youss99.9935868.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s2_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s2_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_V2_S2_fold04 | model_name: fin_l1_lead_v2_s2_fold04 → pod_l1_lead_v2_s2_fold04, root_path: /localscratch/youss99.9935871.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s2_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s2_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_V2_S2_fold05 | model_name: fin_l1_lead_v2_s2_fold05 → pod_l1_lead_v2_s2_fold05, root_path: /localscratch/youss99.9935872.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s2_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s2_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L1_lead_V2_S3_fold01 | model_name: fin_l1_lead_v2_s3_fold01 → pod_l1_lead_v2_s3_fold01, root_path: /localscratch/youss99.9935873.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s3_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s3_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L1_lead_V2_S3_fold04 | model_name: fin_l1_lead_v2_s3_fold04 → pod_l1_lead_v2_s3_fold04, root_path: /localscratch/youss99.9935876.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s3_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s3_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L1_lead_V2_S3_fold05 | model_name: fin_l1_lead_v2_s3_fold05 → pod_l1_lead_v2_s3_fold05, root_path: /localscratch/youss99.9935877.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l1_lead_v2_s3_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l1_lead_v2_s3_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L2_two_lead_S1_fold01 | model_name: fin_l2_two_lead_s1_fold01 → pod_l2_two_lead_s1_fold01, root_path: /localscratch/youss99.9935828.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s1_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s1_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L2_two_lead_S1_fold04 | model_name: fin_l2_two_lead_s1_fold04 → pod_l2_two_lead_s1_fold04, root_path: /localscratch/youss99.9935831.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s1_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s1_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L2_two_lead_S1_fold05 | model_name: fin_l2_two_lead_s1_fold05 → pod_l2_two_lead_s1_fold05, root_path: /localscratch/youss99.9935832.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s1_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s1_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L2_two_lead_S2_fold01 | model_name: fin_l2_two_lead_s2_fold01 → pod_l2_two_lead_s2_fold01, root_path: /localscratch/youss99.9935833.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s2_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s2_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L2_two_lead_S2_fold04 | model_name: fin_l2_two_lead_s2_fold04 → pod_l2_two_lead_s2_fold04, root_path: /localscratch/youss99.9935836.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s2_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s2_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L2_two_lead_S2_fold05 | model_name: fin_l2_two_lead_s2_fold05 → pod_l2_two_lead_s2_fold05, root_path: /localscratch/youss99.9935839.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s2_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s2_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |
| POD_L2_two_lead_S3_fold01 | model_name: fin_l2_two_lead_s3_fold01 → pod_l2_two_lead_s3_fold01, root_path: /localscratch/youss99.9935840.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s3_fold01/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s3_fold01/ | `cv5_optB_leadV2_opt2p_fold01` → `cv5_optB_fold01` |
| POD_L2_two_lead_S3_fold04 | model_name: fin_l2_two_lead_s3_fold04 → pod_l2_two_lead_s3_fold04, root_path: /localscratch/youss99.9935843.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s3_fold04/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s3_fold04/ | `cv5_optB_leadV2_opt2p_fold04` → `cv5_optB_fold04` |
| POD_L2_two_lead_S3_fold05 | model_name: fin_l2_two_lead_s3_fold05 → pod_l2_two_lead_s3_fold05, root_path: /localscratch/youss99.9935844.0/dataset → /tmp/none, save_path: results/fine-tune/saved_models/patient-segment-regression/fin_l2_two_lead_s3_fold05/ → results/fine-tune/saved_models/patient-segment-regression/pod_l2_two_lead_s3_fold05/ | `cv5_optB_leadV2_opt2p_fold05` → `cv5_optB_fold05` |

94 trainer values compared per job (defaults included). Only model_name, save_path and root_path (Narval scratch) may differ; seed and backbone must equal the twin's: **21 of 21 PASS**.

## 3. Data and loader
- fold 1: `cv5_optB_fold01` CSV sha256 `ef0a0fe6bd6f6095…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 2: `cv5_optB_fold02` CSV sha256 `e7a79c5433ac50cf…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 3: `cv5_optB_fold03` CSV sha256 `810004ab8f689298…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 4: `cv5_optB_fold04` CSV sha256 `8be5d2300ccb9b94…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- fold 5: `cv5_optB_fold05` CSV sha256 `dbabb0a9c830f466…` = SHA256SUMS.txt, split column = published manifest, loader records = manifest: **PASS**
- H5: one file linked into all five folders, sha256 = SHA256SUMS.txt (`e4bff370e32476bf…`): **PASS**
- reused runs (job_log.txt: dataset = published folder, CSV sha = SHA256SUMS.txt, backbone sha = the pipeline's, seed = base + fold − 1, finished with a best validation loss): **24 of 24 PASS** (18 FIN on folds 2–3, 6 POD_L1 S1 on folds 1, 4, 5)

Loader (real `get_dls`), labels = the model's own lead(s); two_lead uses both. Excluded record–lead pairs keep their real labels in training and validation, as in the published runs.

| fold | mode | split | records | windows | label mean ± SD (mV) | GS episodes (model's leads) | excluded pairs in split | records |
|--:|---|---|--:|--:|---|--:|---|---|
| 1 | lead_II | train | 11 | 87,205 | II -0.0529 ± 0.0968 |  | 20091 II, 20121 II, 20121 V2, 20241 V2 | 20011 20061 20071 20091 20111 20121 20131 20141 20201 20231 20241 |
| 1 | lead_II | val | 2 | 15,405 | II -0.0835 ± 0.0399 | 15 | 20081 V2 | 20081 20101 |
| 1 | lead_II | test | 3 | 23,754 | II -0.0815 ± 0.1541 |  | 20221 V2 | 20051 20211 20221 |
| 1 | lead_V2 | train | 11 | 87,205 | V2 +0.1147 ± 0.0531 |  | 20091 II, 20121 II, 20121 V2, 20241 V2 | 20011 20061 20071 20091 20111 20121 20131 20141 20201 20231 20241 |
| 1 | lead_V2 | val | 2 | 15,405 | V2 +0.0986 ± 0.0486 | 0 | 20081 V2 | 20081 20101 |
| 1 | lead_V2 | test | 3 | 23,754 | V2 +0.1503 ± 0.1381 |  | 20221 V2 | 20051 20211 20221 |
| 1 | two_lead | train | 11 | 87,205 | II -0.0529 ± 0.0968 / V2 +0.1147 ± 0.0531 |  | 20091 II, 20121 II, 20121 V2, 20241 V2 | 20011 20061 20071 20091 20111 20121 20131 20141 20201 20231 20241 |
| 1 | two_lead | val | 2 | 15,405 | II -0.0835 ± 0.0399 / V2 +0.0986 ± 0.0486 | 15 | 20081 V2 | 20081 20101 |
| 1 | two_lead | test | 3 | 23,754 | II -0.0815 ± 0.1541 / V2 +0.1503 ± 0.1381 |  | 20221 V2 | 20051 20211 20221 |
| 2 | lead_II | train | 11 | 86,485 | II -0.0695 ± 0.1173 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20051 20061 20071 20091 20111 20121 20131 20201 20211 20221 20241 |
| 2 | lead_II | val | 2 | 16,636 | II -0.0666 ± 0.0557 | 6 |  | 20011 20231 |
| 2 | lead_II | test | 3 | 23,243 | II -0.0308 ± 0.0829 |  | 20081 V2 | 20081 20101 20141 |
| 2 | lead_V2 | train | 11 | 86,485 | V2 +0.1237 ± 0.0864 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20051 20061 20071 20091 20111 20121 20131 20201 20211 20221 20241 |
| 2 | lead_V2 | val | 2 | 16,636 | V2 +0.0918 ± 0.0408 | 6 |  | 20011 20231 |
| 2 | lead_V2 | test | 3 | 23,243 | V2 +0.1233 ± 0.0584 |  | 20081 V2 | 20081 20101 20141 |
| 2 | two_lead | train | 11 | 86,485 | II -0.0695 ± 0.1173 / V2 +0.1237 ± 0.0864 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20051 20061 20071 20091 20111 20121 20131 20201 20211 20221 20241 |
| 2 | two_lead | val | 2 | 16,636 | II -0.0666 ± 0.0557 / V2 +0.0918 ± 0.0408 | 12 |  | 20011 20231 |
| 2 | two_lead | test | 3 | 23,243 | II -0.0308 ± 0.0829 / V2 +0.1233 ± 0.0584 |  | 20081 V2 | 20081 20101 20141 |
| 3 | lead_II | train | 11 | 85,631 | II -0.0729 ± 0.1121 |  | 20081 V2, 20121 II, 20121 V2, 20221 V2 | 20051 20071 20081 20101 20111 20121 20131 20201 20211 20221 20231 |
| 3 | lead_II | val | 2 | 16,066 | II -0.0177 ± 0.0953 | 5 |  | 20011 20141 |
| 3 | lead_II | test | 3 | 24,667 | II -0.0527 ± 0.0815 |  | 20091 II, 20241 V2 | 20061 20091 20241 |
| 3 | lead_V2 | train | 11 | 85,631 | V2 +0.1189 ± 0.0825 |  | 20081 V2, 20121 II, 20121 V2, 20221 V2 | 20051 20071 20081 20101 20111 20121 20131 20201 20211 20221 20231 |
| 3 | lead_V2 | val | 2 | 16,066 | V2 +0.1149 ± 0.0665 | 5 |  | 20011 20141 |
| 3 | lead_V2 | test | 3 | 24,667 | V2 +0.1242 ± 0.0674 |  | 20091 II, 20241 V2 | 20061 20091 20241 |
| 3 | two_lead | train | 11 | 85,631 | II -0.0729 ± 0.1121 / V2 +0.1189 ± 0.0825 |  | 20081 V2, 20121 II, 20121 V2, 20221 V2 | 20051 20071 20081 20101 20111 20121 20131 20201 20211 20221 20231 |
| 3 | two_lead | val | 2 | 16,066 | II -0.0177 ± 0.0953 / V2 +0.1149 ± 0.0665 | 10 |  | 20011 20141 |
| 3 | two_lead | test | 3 | 24,667 | II -0.0527 ± 0.0815 / V2 +0.1242 ± 0.0674 |  | 20091 II, 20241 V2 | 20061 20091 20241 |
| 4 | lead_II | train | 11 | 86,148 | II -0.0613 ± 0.1185 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20011 20051 20071 20091 20101 20121 20141 20201 20211 20221 20241 |
| 4 | lead_II | val | 2 | 16,372 | II -0.0580 ± 0.0529 | 11 | 20081 V2 | 20061 20081 |
| 4 | lead_II | test | 3 | 23,844 | II -0.0670 ± 0.0853 |  |  | 20111 20131 20231 |
| 4 | lead_V2 | train | 11 | 86,148 | V2 +0.1298 ± 0.0891 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20011 20051 20071 20091 20101 20121 20141 20201 20211 20221 20241 |
| 4 | lead_V2 | val | 2 | 16,372 | V2 +0.1029 ± 0.0420 | 0 | 20081 V2 | 20061 20081 |
| 4 | lead_V2 | test | 3 | 23,844 | V2 +0.0934 ± 0.0319 |  |  | 20111 20131 20231 |
| 4 | two_lead | train | 11 | 86,148 | II -0.0613 ± 0.1185 / V2 +0.1298 ± 0.0891 |  | 20091 II, 20121 II, 20121 V2, 20221 V2, 20241 V2 | 20011 20051 20071 20091 20101 20121 20141 20201 20211 20221 20241 |
| 4 | two_lead | val | 2 | 16,372 | II -0.0580 ± 0.0529 / V2 +0.1029 ± 0.0420 | 11 | 20081 V2 | 20061 20081 |
| 4 | two_lead | test | 3 | 23,844 | II -0.0670 ± 0.0853 / V2 +0.0934 ± 0.0319 |  |  | 20111 20131 20231 |
| 5 | lead_II | train | 11 | 87,148 | II -0.0682 ± 0.1094 |  | 20081 V2, 20091 II, 20121 II, 20121 V2, 20221 V2 | 20051 20061 20081 20091 20101 20111 20121 20141 20211 20221 20231 |
| 5 | lead_II | val | 2 | 15,861 | II -0.0630 ± 0.1057 | 10 | 20241 V2 | 20131 20241 |
| 5 | lead_II | test | 3 | 23,355 | II -0.0380 ± 0.0906 |  |  | 20011 20071 20201 |
| 5 | lead_V2 | train | 11 | 87,148 | V2 +0.1210 ± 0.0841 |  | 20081 V2, 20091 II, 20121 II, 20121 V2, 20221 V2 | 20051 20061 20081 20091 20101 20111 20121 20141 20211 20221 20231 |
| 5 | lead_V2 | val | 2 | 15,861 | V2 +0.1432 ± 0.0760 | 0 | 20241 V2 | 20131 20241 |
| 5 | lead_V2 | test | 3 | 23,355 | V2 +0.0976 ± 0.0405 |  |  | 20011 20071 20201 |
| 5 | two_lead | train | 11 | 87,148 | II -0.0682 ± 0.1094 / V2 +0.1210 ± 0.0841 |  | 20081 V2, 20091 II, 20121 II, 20121 V2, 20221 V2 | 20051 20061 20081 20091 20101 20111 20121 20141 20211 20221 20231 |
| 5 | two_lead | val | 2 | 15,861 | II -0.0630 ± 0.1057 / V2 +0.1432 ± 0.0760 | 10 | 20241 V2 | 20131 20241 |
| 5 | two_lead | test | 3 | 23,355 | II -0.0380 ± 0.0906 / V2 +0.0976 ± 0.0405 |  |  | 20011 20071 20201 |

## 4. Self-test

`check_training_paths.py`: **37 of 37 checks passed** (exit 0): **PASS**

## 5. The checks fail on injected faults
- text: lr changed: flagged
- text: epochs 70 -> 90: flagged
- flags: weight_decay changed: flagged
- flags: seed changed: flagged
- flags: L2 backbone swapped for ii_lossfix: flagged
- text: L2 guard sha changed: flagged

## 6. Time estimate

- L1: twins took 12–14 min each (2.5 GPU-hours for the 12)
- L2: twins took 17–20 min each (2.7 GPU-hours for the 9)
- walltime in every job: 00:45:00

## Result: ALL PASS

jobs.csv sha256: `e9c5eedc0251ce3471daa1896362777e9935da4e6b5289527ee16a9ac30bcb82` (Narval must reproduce it before submitting). A byte copy is next to this report.
