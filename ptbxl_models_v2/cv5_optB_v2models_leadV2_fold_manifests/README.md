# cv5_optB folds for the v2 V2 single-lead models (option 2+)

**Date:** 2026-11-06

"v2" here means the planned v2 dataset and retraining (`phase1_revision_plan.md`, section D), not lead V2, and not the corrected checkpoints that `ptbxl_models_v2/README.md` calls v2.

**Chosen before any v2 results**, using only V2 labels and gold-standard episode counts, not model performance. Earlier non-v2 results on these folds had already been seen.

## Rule
For the V2 single-lead models, in folds 1, 4 and 5 only:
1. Mark the masked V2 record in validation as `unused`: not train, not validation, not test.
2. Move one unmasked record with V2 episodes from training into validation.
3. Change nothing else.

**Why:** in these three folds, one of the two cv5_optB validation records is masked on V2, and the other (s20101, s20061, s20131) has no V2 gold-standard episodes. The masked record cannot add any, because the gold standard skips masked pairs, although its V2 labels are not quiet.

## Changes against cv5_optB
| Fold | Becomes `unused` | Moves from training to validation |
|--:|---|---|
| 1 | s20081 | s20231 |
| 4 | s20081 | s20071 |
| 5 | s20241 | s20141 |

## Result
| Fold | Test | Validation | Unused | Training records | V2 episodes, validation | V2 episodes, training |
|--:|---|---|---|--:|--:|--:|
| 1 | s20051 s20211 s20221 | s20101 s20231 | s20081 | 10 | 4 | 20 |
| 4 | s20111 s20131 s20231 | s20061 s20071 | s20081 | 10 | 5 | 18 |
| 5 | s20011 s20071 s20201 | s20131 s20141 | s20241 | 10 | 3 | 20 |

V2 episode counts cover unmasked records only, including in training, where masked records remain.

## Unchanged
- All test sets.
- Folds 2 and 3, which are byte-identical to cv5_optB here.
- Every v2 lead II model: `../cv5_optB_v2models_leadII_fold_manifests/` is a byte-identical copy of cv5_optB.
- Every v2 two_lead model, which stays on cv5_optB and uses `../cv5_optB_fold_manifests/`.
- The published fold files in `../cv5_optB_fold_manifests/`.

## Masked V2 records that stay
Only the masked record in validation was removed. These stay where cv5_optB puts them:

| Fold | Masked V2 in training | Masked V2 in test |
|--:|---|---|
| 1 | s20121 s20241 | s20221 |
| 2 | s20121 s20221 s20241 | s20081 |
| 3 | s20081 s20121 s20221 | s20241 |
| 4 | s20121 s20221 s20241 | - |
| 5 | s20081 s20121 s20221 | - |

## Using these files
- **V2 models must have their own fold folders.** Test 1-NK uses one `foldNN` folder for lead II, V2 and two_lead (`build_test1nk_train_sbatches.py`). Copying that layout with these manifests would silently remove the `unused` record from lead II and two_lead and change their validation sets.
- **Format:** same columns as cv5_optB (`patient_id,split`), with one new `split` value, `unused`.
- **In a fold's dataset CSV:** write `unused` as `strat_fold_annotated = -1`, and keep those rows in the CSV. Rows must stay in place because the loader finds each window's H5 row from its CSV position. `-1` is used rather than `0` so it cannot be read as `Split.ALL`, whose value is 0.
- **Loader:** in regression mode, which these models use, it keeps only rows equal to the requested split (1, 2 or 3), so it does not return `-1` rows. Pretraining mode ignores this column, so do not point it at these CSVs. `--reset_strat_folds` must stay off, because it would regenerate the splits and overwrite the CSV.
- **Builder:** `src/scripts/custom_dataset/prepare_fold_dataset.py` stops with an error on `unused`. The v2 builder must map it to `-1`.

## Reporting decisions
- Report regression both with and without masked leads.
- Compare the fixed-validation run with v2 on the original validation (cv5_optB), not with the published models.

## Evidence
Everything is in `GIT/Tasks/folds/option2plus/`, which is outside this repository. The dry run and table used the Test 1-NK dataset, because v2 is not built yet.

**Re-run the counts on v2 once it is built:**
```
python3 fold_table.py --dataset <v2 root>
.venv/bin/python dry_run_loader.py --dataset <v2 root> --scratch <empty folder>
```
Output goes to files named after the v2 folder, so the Test 1-NK evidence stays. Both scripts assume the Test 1-NK layout (a two-channel H5, a root CSV, and `foldNN` folders holding cv5_optB), so a different v2 layout means adapting them.

| File | What it is |
|---|---|
| `build_option2plus.py` | builds both manifest folders |
| `diff_vs_cv5_optB.txt` | line-by-line diff against cv5_optB |
| `dry_run_test1nk.txt` (from `dry_run_loader.py`) | real loader, window counts before and after |
| `before_after_test1nk.md` / `.csv` (from `fold_table.py`) | records, windows, V2 episodes, V2 SD and masked leads per fold and split |
