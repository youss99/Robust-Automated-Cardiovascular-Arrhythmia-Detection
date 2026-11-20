# Final thesis runs: notes for the Narval session (2026-11-20)

The user's instructions are `instructions (3).md` (VM, `GIT/`); this note is the cluster part. **Your scope:** steps 0, 0b, 1, 2 (and 2b, 3 optional groups only if the user says). The VM does steps 4 to 7: R-peak cache, NeuroKit pipelines, all LTST inference and scoring. Don't run inference on LTST records.

Ground rules from the instructions: ask before guessing; never overwrite data, checkpoints or results (every run has its own folder, named by its ID); log every job; report when training finishes and flag a diverged run or a failed check straight away.

## Setup

1. `git status` clean, then pull `update-initial-changes-with-Mitch`. Record the commit.
2. Link the published median H5 into the fold folders (now 11, three new lead II folders):
   `bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5`. It checks every CSV and the H5 against `phase1_folds/SHA256SUMS.txt`.

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

`python3 src/scripts/custom_dataset/build_final_sbatches.py` writes every job to `logs/sbatch_scripts/final/<group>/`, plus `jobs.csv` and `submit_final.sh`. Overrides: `FIN_REPO_ROOT`, `FIN_DS_ROOT`, `FIN_ACCOUNT`, `FIN_GPU`, `FIN_PT_GPU`, `FIN_WALLTIME`, `FIN_PT_WALLTIME`, `FIN_OUT`. With the defaults, `jobs.csv` is byte-identical to the one the VM dry run used (sha256 `7c5d2588bc9eb2c8f00e7ed66ea77d4c33eb130bedbd83795a699df88cc97a4d`); if it isn't, say so before submitting.

| Group | Jobs | What | Submit when |
|---|--:|---|---|
| `step1` | 2 | `patch_ecg_unified_{ii,v2}_lossfix`: `pretrain_unified_v2.sh` unchanged, called from a wrapper that writes `job_log.txt` | user approved the dry run, step 0 passed |
| `step3` | 15 | L1-sup (lead II, V2) and L2-sup (two-lead), supervised, S1 | user approved the dry run, step 0 passed |
| `step2_L2` | 15 | L2: two-lead from `patch_ecg_unified_ii_v2_rebuilt`, S1, S2, S3 | step 0b passed |
| `step2_L1` | 30 | L1: lead II from `ii_lossfix`, V2 from `v2_lossfix`, S1, S2, S3 | step 1 checks passed (below) |
| `step2b` | 10 | optional: L1-rebuilt, `_rebuilt` backbones, S1 | only if the user says |
| `step3o` | 30 | optional: L1-sup, L2-sup at S2, S3 | only if the user says |

Submit with `bash logs/sbatch_scripts/final/submit_final.sh <group>`. It refuses a group that was already submitted, and records the job IDs in `submitted_<group>.tsv`.

**Seeds:** S1 = 200 + fold − 1 (published), S2 = 300 + fold − 1, S3 = 400 + fold − 1. Pre-training: none.

**Validation manifest:** folds 2 and 3 use `cv5_optB_fold02`/`03` for every model. In folds 1, 4 and 5, lead II uses `cv5_optB_leadII_opt2p_fold*` (moved record added to validation). V2 and two-lead use `cv5_optB_leadV2_opt2p_fold*` (masked V2 record unused, moved record in validation; the two-lead model uses the V2 split by the user's decision).

**Step 1 checks, before `step2_L1`:**
- `job_log.txt` of each backbone: the logged loss starts near 1 and falls. Don't compare it with old single-lead logs (they show the mixed loss).
- `backbone_masked_vs_visible.py` on each backbone with `--leads II` (or `V2`) and `--seed` 0, 1, 2 must read "does not reproduce visible patches".
- For L1-rebuilt (step 2b) to be a one-change comparison: `diff <(tr ',' '\n' < .../patch_ecg_unified_ii_rebuilt/params.txt) <(tr ',' '\n' < .../patch_ecg_unified_ii_lossfix/params.txt)`, and the same for v2. Only `model_name`, `save_path` and `root_path` may differ. Compare `data_source.txt` too.

## Report and copy back

- **When training finishes:** one table with each run's best epoch and best validation loss (from `job_log.txt`). Flag NaN losses, failed jobs, and any run far from its peers (R1b had 3 diverged supervised lead II runs).
- **Copy back, same relative paths:** for each fine-tune run, `*_best.pt`, `params.txt`, `_*_metrics*.csv` and `job_log.txt`. For each backbone, `*_best.pt`, `params.txt`, `job_log.txt`, `data_source.txt`, `pretrain_split.csv` and the `_training_metrics_*.csv`. Leave snapshots on the cluster.
