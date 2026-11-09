# Session handover: Narval, October 13 to November 8, 2026

For the next Claude Code session on this project. It records what was done, what is true now, and what is still open. Paths are relative to `Project_1/` unless noted.

## Read this first

1. **Use `Datasets/Unified_Dataset_No_PTB_Segment_v2`.** The older `Unified_Dataset_No_PTB_Segment` and `_fixed` have wrong signals under a third of their labels. Details: `src/scripts/unification/README_unified_dataset.md`.
2. **Before submitting anything, follow these rules.**
   - Run heavy work through Slurm, not the login node.
   - Fine-tune with the thesis recipe (Table A.2). Raise any deviation before making it.
   - Split by patient or record, never by segment.
   - Touch a test set only once the configuration is fixed.
3. **Phase 1 is planned but not pulled here.** GitHub is 4 commits ahead of Narval (`fc68f85`), and they add `README_phase1_runs.md` and `README_phase1_narval_notes.md`. Those notes say Narval does steps 0–3, starting from a clean `git status`. Six new files are still uncommitted here (see "Git"), so deal with them first. Those two READMEs were read but not executed.

## Machines and how they stay in sync

| Machine | Role |
|---|---|
| Narval | Main cluster: `/lustre06/project/6039801/youss99/Project_1` (same place as `~/projects/def-majidk/youss99/Project_1`) |
| Nibi | Used while Narval was down: repo at `/home/youss99/scratch/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection` |
| VM (`SCS-GPU-2024-5`) | Patient inference, scoring, dataset builds |

- **Code moves only through git,** on branch `update-initial-changes-with-Mitch`. Do not work on `master`.
- **Data and `.pt` files move only through `rsync`.** `*.pt`, `results/` and `logs/` are ignored by git.
- **This session has no GitHub push credentials.** The user pushes from their own terminal.
- **Do not use `git add -A`:** the repo has `.bak-*` files and `slurm-*.out` logs.

## Slurm profile that works on Narval

`--account=def-majidk --gpus-per-node=a100_3g.20gb:1 --cpus-per-task=4 --mem=20G`, with the time limit at about twice the measured run time. Run from the repo root, with `module load python/3.10 cuda cudnn gcc arrow` and `source .venv/bin/activate`. Imports from `/project` are occasionally very slow (about 11 minutes once).

## Datasets added (`Datasets/`, all read-only)

| Folder | Contents |
|---|---|
| `incartdb` | Raw INCART: 75 records, 32 patients, 257 Hz. Checksums verified |
| `incart_beat_lead{II,V5}_100hz` | Reference-centred beat caches built here (not locked) |
| `multi_II_nk` | NeuroKit-centred beats: INCART, MIT-BIH, SVDB, CPSC2021 (built on the VM) |
| `multi_V5_nk` | NeuroKit-centred beats: INCART, EDB (built on the VM) |
| `ltstdb_regression_e1_nkcentred_balanced_100hz` | LTST E1 regression data, II/V2/V5 |
| `ltstdb_regression_beatcenter_100hz_test1nk_nocompress` | LTST Test 1-NK regression data |
| `Unified_Dataset_No_PTB_Segment_v2` | Corrected Unified: 58,942 rows, H5 md5 `ac56fd0edfb6b63da06c6c2498f0346a` |
| `New import from mitchell datasets` | Fresh copy of the original Chapman, CinC-2020 and Unified files |

**Known problem:** the fold folders in the E1 and Test 1-NK datasets lost their H5 links in transfer, so they can't be used as `--root_path` on Narval until the links are recreated. That needs a temporary unlock.

**Locked datasets:** `cp -R` keeps them read-only, and the trainer writes its split to the staged CSV. Staged copies must be writable.

## What was done

### 1. INCART central-beat classifier (background / PAC / PVC), PatchECG Small

Code: `src/incart_beat/`. Patient split: validation 3, 6, 16, 24, 31; test 2, 10, 17, 22, 27. Input is µV at 100 Hz, in 10 s windows centred on the beat. Thesis epochs (20 head + 70 full). Scripts: `src/scripts/incart_beat/`. Outputs: `results/fine-tune/saved_models/incart/`.

Validation over 3 seeds, mean PAC/PVC AUPRC at the selected epoch:

| Run | Setup | Score |
|---|---|---|
| A | Batch-dependent normalization, cross-entropy | 0.758 ± 0.006 |
| **B (chosen)** | Per-window normalization, cross-entropy | 0.758 ± 0.006 |
| C | Per-window normalization, focal loss | 0.764 ± 0.013 |
| D | B, trained on 7 original symbols | 0.742 ± 0.007 |

- A and B are equally accurate, but A's probabilities shift by up to 0.38 depending on batch mates.
- Focal loss shows no reliable gain.
- D is lower: PAC AUPRC falls by 0.043 (p = 0.01).
- An independent audit passed 89 of 89 checks (`checks/audit.py`).

Test set, run once for B:

| Lead | PAC AUPRC / AUROC | PVC AUPRC / AUROC |
|---|---|---|
| II | 0.572 / 0.956 | 0.904 / 0.974 |
| V5 | 0.498 / 0.937 | 0.948 / 0.993 |

The argmax cut-off does not carry across patients. 97% of training PACs come from patients 15 and 30, and the test PAC result is essentially patient 10.

### 2. Multi-database NeuroKit-centred models

Config B, with and without ±50 ms training jitter. Outputs: `results/fine-tune/saved_models/incart_multi_nk{,_jit50}/`. Scored on the reference-centred INCART validation windows, for comparison with the models above:

| Model | II: PAC / PVC AUPRC | V5: PAC / PVC AUPRC |
|---|---|---|
| INCART-only B | 0.587 / 0.940 | 0.561 / 0.914 |
| Multi-database | 0.795 / 0.928 | 0.558 / 0.944 |
| Multi-database, jitter | **0.892 / 0.986** | **0.674 / 0.969** |

Jitter makes the model tolerate shifted beat centres, so the jittered models were recommended for patient testing. These are one seed each. The VM added test scripts (`vm/eval_multi_*.py`); those results were not reviewed in this session.

### 3. Homer patient inference

`src/incart_beat/predict_patient.py` converts units, resamples and classifies each beat. In `aligned_leads_8103_8123_first5h.csv` the chest column (`V2`) is flat for the first 8.8 minutes and over 10 mV in 24.5% of 10 s blocks. Find beats on lead II and pass them with `--beats` for the chest-lead model, and exclude low-quality windows before scoring.

### 4. Unified dataset bug, and the rebuild

Rows 0–18,436 of the old files hold CinC signals under Chapman labels. The rebuilt v2 matches its source files on every row and label, and 6 raw PhysioNet records match it exactly. A fresh import is byte-identical to our original, so the fault came with the delivered file.

### 5. Backbones rebuilt on v2 (finished 2026-11-06)

`results/pre-train/saved_models/unified/patch_ecg_unified_{ii,v2,v5,ii_v5,ii_v2}_rebuilt`. Rows with a blank selected lead were removed at job start by `stage_unified_subset.py`; the training code is unchanged. Each folder has `data_source.txt` and `pretrain_split.csv`. Script: `src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh`. Do not compare validation losses with `_fixed`: the validation sets differ. Two-lead runs log losses about 100 times smaller than single-lead runs.

### 6. Models imported from Nibi (2026-11-06)

Both are in folders named `Imported models from NIBI cluster`: 5 backbones (only `v2_fixed` is new) and 49 regression runs (E1 and Test 1-NK, including 5 smoke tests). All used `_fixed` backbones.

## Status of existing models

| Models | Status |
|---|---|
| `ptbxl2unified_*`, `unified_*_supervised_*`, `code2unified_*`, including `*_fixed`, and Homer's `patch_ecg_ptbxl2unified_ii_v2_c2_focal_tta` | **Invalid:** trained and scored on wrong labels |
| June backbones (`ii_only`, `v2_only`, `ii_v2`, …) | Do not use for new work; Phase 1 run R1b reuses them only to reproduce the published setup |
| `*_fixed` backbones and models fine-tuned from them | Own labels are fine; pretraining was weaker. The effect is not measured |

## Open items (the user deferred these)

1. Retrain the Unified-target classifiers on v2, if that label space is still used.
2. Fine-tuned models on old backbones. Earlier suggestion: refit one E1 fold per lead from the rebuilt backbones, and redo the rest only if results move beyond the seed spread. Phase 1 may supersede this.
3. Decide whether two-lead regression stays in the plan.
4. Recreate the fold links in the E1 and Test 1-NK datasets.
5. Point these at v2 before reuse: `pretrain_classifier_v5.sh`, `pretrain_classifier_v5_nibi.sh`, `finetune_classifier_v5.sh`, `finetune_classifier_v5_tta_focal_sam.sh`, `finetune_classifier_v5_supervised.sh`.
6. Copy v2 and the rebuilt backbones to the VM or Nibi when needed, and verify checksums.
7. `src/scripts/multi-gpu/finetune/CLASSIFICATION_RESULTS_SUMMARY.md` still shows Unified-label results without a warning.

## Git, as of 2026-11-08

Narval is at `4c40c6d`, 4 commits behind GitHub (`fc68f85`). These new files are uncommitted, and none overlap with the incoming commits:

```bash
git add README_session_handover_2026-11.md src/incart_beat/eval_cache.py src/scripts/unification/verify_unified_no_blanks.py src/scripts/unification/rebuild_unified_v2.py src/scripts/unification/stage_unified_subset.py src/scripts/unification/README_unified_dataset.md src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh
git commit -m "Unified v2 rebuild and verifier, backbone pretraining without blank-lead rows, session handover"
git pull --rebase origin update-initial-changes-with-Mitch
git push origin update-initial-changes-with-Mitch
```

## Traps that cost time

- Do not filter rows inside the dataset class: the trainer treats CSV row number as H5 row number.
- The trainer continues from any existing snapshot, which caused a stale checkpoint once. `pretrain_unified_v2.sh` refuses to start if the model folder exists.
- `patchECG_finetune.py` silently trains from scratch if the backbone path is missing (from the Phase 1 notes).
- Folder names with spaces need quoting: `Imported models from NIBI cluster`, `New import from mitchell datasets`.
