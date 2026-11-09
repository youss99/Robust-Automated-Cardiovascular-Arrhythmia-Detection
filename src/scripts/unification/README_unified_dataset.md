# Unified dataset: known issues and the corrected version

**Use `Unified_Dataset_No_PTB_Segment_v2`.** Do not train on `Unified_Dataset_No_PTB_Segment` or `Unified_Dataset_No_PTB_Segment_fixed`: in both, about a third of the rows hold a signal that does not belong to the record named on that row.

Last updated 2026-11-06.

## What was wrong

The dataset should hold 40,507 Chapman rows followed by 18,437 CinC rows (CPSC 2018, CPSC-extra, Georgia). The H5 file did not match its label CSV:

| Rows | Labelled as | Delivered file | `_fixed` (2026-10-01) | `v2` |
|---|---|---|---|---|
| 0–18,436 | Chapman | CinC signals | CinC signals | Correct Chapman |
| 18,437–40,506 | Chapman | Correct Chapman | Correct Chapman | Correct Chapman |
| 40,507–58,943 | CinC | Blank | Correct CinC | Correct CinC |

There were two separate problems:

1. **Blank rows (found 2026-10-01).** The last 18,437 rows were all zeros. `_fixed` restored them from `cinc-2020/Cinc2020Dataset.h5`.
2. **Wrong signals under Chapman labels (found 2026-11-06).** Rows 0–18,436 are exact copies of the CinC rows, in the same order, under Chapman filenames and labels. The label matches the signal's true label in only 448 of these rows, by chance. `_fixed` did not touch these rows.

As a result, 18,437 Chapman rows never reached the dataset, and every CinC signal appeared twice in `_fixed`.

**Likely cause:** CinC appears to have been written from row 0 instead of after Chapman, overwriting the start and leaving the end blank. This is inferred from the row layout. The current `unified_preprocessing.py` does not reset its row counter, so the cause has not been reproduced from code.

**Not a transfer error:** a fresh import of the original files (`Datasets/New import from mitchell datasets/`) is byte-identical to our Unified, Chapman and CinC copies.

## How each version scores

Every row was compared with `ChapmanDataset.h5` or `Cinc2020Dataset.h5` by filename.

| Version | Rows | Correct signal | CinC signal under Chapman label | Blank on all 12 leads | Distinct signals |
|---|---|---|---|---|---|
| Delivered file | 58,944 | 22,072 | 18,437 | 18,439 | 39,879 |
| `_fixed` | 58,944 | 40,507 | 18,437 | 4 | 39,879 |
| `v2` | 58,942 | 58,942 | 0 | 0 | 58,265 |

Six raw PhysioNet recordings (3 Chapman from the affected rows, 3 CinC) were also compared, after resampling to 100 Hz. All six match `v2` exactly. In `_fixed`, the 3 Chapman rows hold unrelated signals (r near 0). The other rows were checked against the local source files only, not raw PhysioNet.

## What changed in v2

- Every Chapman row is taken from `chapman/ChapmanDataset.h5` by filename. CinC rows are unchanged from `_fixed`.
- Two records blank on all 12 leads in the source were removed: `cpsc_2018_extra/g1/Q0400` and `cpsc_2018_extra/g3/Q2961`.
- Row order and labels are otherwise unchanged. Every label equals its source label.

`UnifiedDataset.h5` md5: `ac56fd0edfb6b63da06c6c2498f0346a`. The folder is read-only.

## Remaining limitations in v2

- **Some leads are blank in individual rows,** as they are in the sources: II 0, V2 1,099, V5 994 of 58,942. `v2` keeps these rows. `stage_unified_subset.py` removes them at the start of each pretraining job.
- **677 rows repeat another row's signal.** Each still matches its own source row, so these appear to be duplicates in the sources. They can fall on both sides of a random train/validation split.

## Which models are affected

| Models | Status | Why |
|---|---|---|
| Unified-target models: `ptbxl2unified_*`, `unified_*_supervised_*`, `code2unified_*`, including `*_fixed`, and Homer's `patch_ecg_ptbxl2unified_ii_v2_c2_focal_tta` | **Invalid; retrain** | Trained and scored with wrong labels on about 31% of rows |
| Older backbones: `patch_ecg_unified_{ii,v2,v5,ii_v5,ii_only,v2_only,ii_v2,all_leads}` | Do not use for new work | Pretraining uses no labels, but 31% of rows were blank and many Chapman records were missing |
| `*_fixed` backbones | Usable, not as intended | Real ECGs and no labels used, but CinC counted twice and 18,437 Chapman rows missing. Duplicates may also cross the pretraining train/validation split |
| Models fine-tuned from those backbones with their own labels: `unified2ptbxl_*`, INCART, multi-database, E1, Test 1-NK, LTST regression | Labels and evaluation unaffected | They inherit only the pretrained weights. Models on the older backbones inherit the worse pretraining. The effect has not been measured |

Results in existing summaries that use Unified labels, such as `src/scripts/multi-gpu/finetune/CLASSIFICATION_RESULTS_SUMMARY.md`, should not be quoted.

## Current work

Backbones are being retrained on `v2`, with blank-lead rows removed, as `patch_ecg_unified_{ii,v2,v5,ii_v5,ii_v2}_rebuilt`. Each model folder records its data in `data_source.txt` and `pretrain_split.csv`. Deciding what else to retrain is deferred.

## Scripts

Run from the repo root:

```bash
# Check a dataset row by row
python src/scripts/unification/verify_unified_no_blanks.py --datasets ../Datasets --dataset Unified_Dataset_No_PTB_Segment_v2

# Rebuild v2 (refuses to overwrite an existing folder)
python src/scripts/unification/rebuild_unified_v2.py --datasets ../Datasets

# Pretrain a backbone on v2
sbatch --job-name=<model_name> src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh <leads> <model_name>
```

The first two load whole datasets (about 12 GB), so run them as Slurm jobs.

**Scripts that still point to `_fixed`:** `pretrain_classifier_v5.sh`, `pretrain_classifier_v5_nibi.sh`, `finetune_classifier_v5.sh`, `finetune_classifier_v5_tta_focal_sam.sh` and `finetune_classifier_v5_supervised.sh`. Update them before using them for new training.
