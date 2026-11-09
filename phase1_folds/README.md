# Phase 1 fold folders (README_phase1_runs.md)

Rebuilt from the manifests by `src/scripts/custom_dataset/build_phase1_folds.py`, so the VM and the cluster use the same copy.

| Folder | Manifests | Used by |
|---|---|---|
| `cv5_optB_fold01`–`05` | `ptbxl_models_v2/cv5_optB_fold_manifests` | R1b, R2 |
| `cv5_optB_leadV2_opt2p_fold01`, `04`, `05` | `ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests` (`unused` = -1) | R2b |

Each folder holds `modified_dataset/modified_dataset.csv` and needs the published median H5 (1 GB, not in git). On each machine, once, from the repo root:

    bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5

That checks the H5 and the CSVs against `SHA256SUMS.txt` and links the H5 into every folder. Source of the H5 and CSV on the VM: `custom_datasets/ltstdb_regression_median_100hz_nocompress`.

Check on the VM: all 30 published cv5_optB checkpoints reproduce their saved test MSE on these folders (worst relative difference 0.00062).
