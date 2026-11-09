# Phase 1 retraining runs: steps 0–5

Instructions for Claude Code, on the VM and the cluster. These are the runs that follow the rebuilt backbones:
1. a rerun of the paper setup
2. the backbone swap
3. the V2 validation fix
4. the R-peak detector freeze
5. the beat-centred v2 dataset

## Ground rules

- **Ask before guessing.** If anything here is ambiguous, or conflicts with the code, the configs or files already on the VM, stop and ask me. Don't fill gaps yourself.
- **One change per run.** Each run differs from its comparison run in exactly one thing (see "Runs at a glance"). Copy everything else from the comparison run's config:
  - data
  - folds and validation sets
  - preprocessing
  - hyperparameters and epochs
  - seed
- **Don't touch existing files.** Never overwrite published data, checkpoints, predictions or results. Write each run to its own new folder, named by run ID.
- **Dry run first.** Before submitting a step to the cluster:
  - run the data loader on one fold
  - report records per split, windows per split, label mean and SD per split, and the exact job list
  - wait for my OK
- **Log every job:**
  - the resolved config
  - git commit
  - seed
  - backbone checkpoint path with its checksum
  - best epoch and validation loss
- **Cluster details come from me:** account, GPU type and paths. Ask before the first submission.
- **Report after each step:**
  - what ran
  - any deviation from this file
  - results next to the comparison run

## Shared setup

**Records:** the 16 Option B LTST records. s20121 is in training in every fold.

**Folds:** `cv5_optB`. Test sets are the same in every run. Training is every record not in a fold's test or validation set.

| Fold | Test | Validation (published) |
|---|---|---|
| 1 | s20051, s20211, s20221 | s20081, s20101 |
| 2 | s20081, s20101, s20141 | s20011, s20231 |
| 3 | s20061, s20091, s20241 | s20011, s20141 |
| 4 | s20111, s20131, s20231 | s20061, s20081 |
| 5 | s20011, s20071, s20201 | s20131, s20241 |

**Masked pairs (`MANUAL_OMIT`):**
- **Which:** s20081 V2, s20091 II, s20121 II and V2, s20221 V2, s20241 V2.
- **In training:** handle them exactly as the published runs did. Tell me how that is before step 1.

**Model variants:**
- **Architectures:** `lead_II` and `lead_V2` are single-lead models. `two_lead` is one network that reads both leads and has two outputs.
- **Training type:** each is either **pretrained** (fine-tuned from a backbone) or **supervised** (trained from scratch).
- **Headline:** the paper's main result ("union") is the two single-lead models.

**Data for steps 1–3:** the published data, unchanged:
- non-overlapping 10 s windows on a fixed grid
- resampled from 250 to 100 Hz
- label = the median `.16a` `ee` (J+60) over the beats in the window

**Seed:** one per job. Use the published seed if it's recorded, otherwise ask. Use the same seed in every run.

**Predictions to save, per run:** out-of-fold predictions for clean and noisy (NSTDB `em`, seed 42) signals, in the published format, so the existing scoring scripts run unchanged.

**Reference results:** `PAPER_RESULTS_TWO_MODELS.md` (published runs) and `C_neurokit_baseline.md` (§10c R-peak screening, §12 finalists).

## Runs at a glance

| Step | Run | Where | Jobs | Compare with | The one change |
|---|---|---|---|---|---|
| 0 | – | Cluster | – | – | Check the rebuilt backbones |
| 1 | R1b | Cluster | 30 | R1 (published) | Rerun on the new cluster |
| 2 | R2 | Cluster | 15 | R1b | Rebuilt backbones |
| 3 | R2b | Cluster | 3 | R2 (V2, folds 1, 4, 5) | Honest V2 validation (option 2+) |
| 4 | – | VM | – | – | Freeze the R-peak detector |
| 5 | R3 | VM, then cluster | 10 | R2 single-lead, with R2b's V2 folds 1, 4, 5 | Beat-centred v2 data |

Order:
- Steps 1–3 can run in parallel once step 0 passes.
- Step 4 runs on the VM at the same time.
- Step 5 waits for step 4.

## Step 0: check the rebuilt backbones

Run `check_rebuilt_recipe.py` on the cluster. Everything except the pretraining data must match the old recipe:
- architecture
- patch size
- masking
- epochs
- optimizer and schedule
- batch size
- sampling rate
- normalization

Also confirm which rebuilt backbone replaces each published one. The published single-lead models used `ii_only` and `v2_only`, but the rebuilt set is named `patch_ecg_unified_{ii,v2,ii_v2}_rebuilt`. If anything differs, stop and report.

## Step 1: R1b, the paper setup with the old backbones on the new cluster

- **Jobs:** 30 = `lead_II`, `lead_V2` and `two_lead` × pretrained and supervised × 5 folds.
- **Setup:** exactly the published configs and checkpoints. Copy the old backbones to the cluster and verify their checksums.
- **Purpose:** to check that the published results reproduce, and to give the noise floor for every later comparison.
- **Report:** the differences from the published results.

## Step 2: R2, the rebuilt backbones

- **Jobs:** 15 = `lead_II`, `lead_V2` and `two_lead`, pretrained only, × 5 folds. Each is fine-tuned from its rebuilt backbone.
- **Supervised runs:** these don't load a backbone, so R2 reuses R1b's supervised runs. Confirm that in the code first. If they load any pretrained weights, ask.

## Step 3: R2b, the V2 validation fix (option 2+)

**Jobs:** 3 = `lead_V2`, pretrained (rebuilt backbone), folds 1, 4 and 5 only. Use the option 2+ fold manifests already prepared on the VM.

**What changes,** for V2 single-lead models only:

| Fold | Masked V2 record (set to unused) | Moved from training to validation | New V2 validation |
|---|---|---|---|
| 1 | s20081 | s20231 | s20101, s20231 |
| 4 | s20081 | s20071 | s20061, s20071 |
| 5 | s20241 | s20141 | s20131, s20141 |

**What doesn't change:**
- The masked record does not go into training.
- Test sets are unchanged.
- Lead II models, two-lead models and folds 2–3 are unchanged.

If the manifests differ from this table, stop and ask.

## Step 4: freeze the R-peak detector (VM)

**Run:**
- noise seed 43 (same protocol as seed 42) through detection and delineation
- for rodrigues2021, khamis2016 and NeuroKit
- leave out `neurokit_cleaned`

**Report,** per detector and lead, with each detector's own fold offsets:
- DMY MAE, p100, bias and RMSE
- the worst record-lead and its MAE
- spurious R-peak rate inside noise bursts, for seeds 42 and 43 (khamis2016 is missing from §10c)
- which records khamis2016's 51 clean V2 false alarms fall on

**Apply the rule fixed in advance:**
- **Primary:** DMY MAE and p100 per lead, including the worst record-lead.
- **Tie-break:** the lower spurious rate in noise.
- **Alarms:** report them, but never use them to choose.

**Also report** whether the choice holds when it's made within each fold, using only that fold's training records' DMY beats.

Report the result. I confirm the freeze.

## Step 5: R3, beat-centred v2

**Build the v2 dataset (VM).** Rerun Test 1-NK's dataset builder with only these changes:
1. **R-peaks:** take them from the frozen detector (expected: rodrigues2021) instead of the NeuroKit `ecg_clean` path. Rerun the window selection; don't shift the old windows onto new peaks.
2. **Window selection:** same rule as Test 1-NK: the first usable beat at least 10 s after the previous window's beat.
3. **Centring:** match each annotated beat to the nearest detector R-peak within ±100 ms. The window is 10 s, centred on that R-peak.
4. **Label:** that beat's `.16a` `ee` for the lead.
5. **DMY beats:** keep the `DMY` lines:
   - strip the prefix and parse the same fields
   - add an `is_dmy` column and carry it into the window index
   - use it only for evaluation
6. **Exclusions:** keep the published exclusions.

**Report before training:**
- whether DMY beats appear in the published labels and in Test 1-NK's windows (expected: no, because the parser skips `DMY` lines)
- windows per record-lead, and the total (Test 1-NK had 119,885 windows; the paper had 119,126)
- beats with no R-peak within ±100 ms, per record-lead
- how many windows are centred on a DMY beat

**Train (cluster):**
- **Jobs:** 10 = `lead_II` and `lead_V2`, pretrained (rebuilt backbone), 5 folds.
- **Settings:** same hyperparameters, epochs and seed as R2. With the window count within about 1% of the paper's, the number of training steps is about the same.
- **Validation:** V2 uses the option 2+ sets from step 3. Lead II uses the published ones.

**Predict:**
- **For alarms:** out-of-fold predictions, clean and noisy, in the form the alarm pipeline needs. Reuse Test 1-NK's inference and aggregation code, and tell me what it does before running.
- **For DMY scoring:** a window centred on the detector R-peak matched to every DMY beat (±100 ms) in each held-out record, as Test 1-NK was scored in C §9.

**Compare with R2,** with R2b's three V2 fold models swapped in. Don't compare with Test 1-NK, which also differs in detector and backbone.

## Scoring (every run)

Use the scripts that scored the published results.

**Regression:**
- clean signal, out-of-fold, per lead, with and without the masked pairs
- metrics: MAE, bias, RMSE, calibration slope, r, and p100 (share of errors larger than 0.1 mV), with R² as a secondary column
- the same metrics on DMY beats:
  - grid runs: use the grid window that contains each DMY beat, as in C §9
  - R3: use the DMY-centred windows

**Alarms:**
- PROSE FSM against the published gold standard (101 episodes)
- conditions: clean, noisy, and gated at 6 dB
- gate with `.atr` R-peaks as published, and also with the frozen detector's R-peaks once step 4 is done
- metrics: TP, FP, FN, IoU and false-alarm time

Report each run next to its comparison run from "Runs at a glance".
