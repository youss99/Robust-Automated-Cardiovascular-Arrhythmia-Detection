# Pre-training loss fix for single-lead models, and PatchTST support

November 2026. Concerns commit `813c715` and everything before it. The fix belongs to branch `update-initial-changes-with-Mitch`; a checkout has it when the self-test of Section 8 passes.

This note explains a defect in the self-supervised pre-training of single-lead models: what it is, what it affects, the code changes that fix it, how they agree with the thesis the models follow, and how the fix was checked. It also records what PatchTST needs for pre-training and for regression fine-tuning. Section 8 gives the steps for retraining.

Terms used here:
- **PatchECG** is `--model vanilla_vit`. **PatchTST** is `--model patch_tst`.
- A **window** is one 10 s segment of one or two leads. A **patch** is 0.5 s of one lead (50 samples at 100 Hz).
- **Old code** is commit `813c715`. **New code** is the fixed code.
- **The thesis** is Chatterjee, *Toward Robust Automated Cardiovascular Arrhythmia Detection using Self-supervised Learning and 1-Dimensional Vision Transformers*. Page numbers are the thesis's own.

## 1. Summary

**What was wrong**

| | Problem | Where |
|---|---|---|
| 1 | Pre-training on **one lead** minimised the wrong loss. The loss should score only the masked patches. It also scored visible patches, with about 40 % of its weight, and its value was about one batch size too large. | Every single-lead backbone, PatchECG and PatchTST |
| 2 | Logged pre-training losses were the true loss divided by the batch size. With one lead this cancelled the factor from problem 1, so those logs looked normal. | The logs of every masked-patch pre-training run |
| 3 | PatchTST crashed when two leads were given as one sequence. | PatchTST with two leads and `--data_augmentation none` |
| 4 | Two mistakes when loading a backbone were not reported. A backbone path that does not exist was ignored, and training went on from random weights. A two-lead PatchECG backbone loaded into a one-lead model without a message, with the same vector in every position. The other size mismatches did stop, with an unrelated error. | Fine-tuning with a wrong backbone |

**What was not wrong**

- **Pre-training on two or more leads.** Its loss was already the intended one, and the fix does not change it.
- **Regression fine-tuning.** Loss, metrics, layer-wise learning-rate decay and weight loading are correct. Evaluated again with the new code, the published R2 fold-1 checkpoints give their published test metrics exactly, and the same seeded fine-tune gives identical numbers on the old and the new code.

**Agreement with the thesis**

- The thesis defines the pre-training loss over the masked patches only. The fix makes single-lead pre-training do that. With two or more leads the code already did.
- For the thesis's own configurations (PatchECG on 12 leads in one sequence, PatchTST with leads fed separately) the fix changes nothing in the training itself: the same training steps give the same losses and the same weights before and after (Section 5.2). Only the logged loss changes.
- Three options discussed in this note would depart from the thesis. They are optional and are listed in Section 5.4.

**What it means for existing results**

- The defect is in pre-training only. Regression fine-tuning and its evaluation were re-checked and are correct, so the reported regression scores are right for the models that were trained.
- Models fine-tuned from a **single-lead** backbone started from weights trained on a mixed objective (reconstruct masked patches and reproduce visible ones), not on the thesis objective. Section 7 lists them.
- Supervised models, models fine-tuned from two-lead backbones (including the R2 two-lead model) and the NeuroKit pipelines are not affected.
- Whether a corrected backbone changes accuracy is not known. Finding out needs the retraining in Section 8.

**PatchTST**

- With the fix, PatchTST pre-trains and fine-tunes for regression in every lead layout: one lead, two leads fed separately (the thesis's way) and two leads in one sequence.
- **Whether PatchTST learned depended on the input normalisation.** In 30-epoch runs on lead II with the thesis's batch statistics, its masked-patch error stayed at 0.96 or above in each of four runs, where 1.0 is the error of predicting zero. With per-window normalisation as the only change to the thesis's settings it reached 0.40; PatchECG reached 0.57 (Section 6.4). 30 epochs do not show what the thesis's 500 would do.
- With batch statistics, a PatchTST prediction also depends on the other windows in its batch. PatchECG's does not (Section 6.3).
- Per-window normalisation for PatchTST departs from the thesis (Section 5.4). It is recommended here for one or two leads, for those two measured reasons. The two-lead PatchTST runs of the final checks all used it (Section 6.1).

**What to do**

1. Run the self-test wherever the code will train (Section 8).
2. Pre-train the single-lead backbones again with the same recipe.
3. Fine-tune again from the new backbones and compare with the existing runs.

## 2. The issue, explained

### 2.1 What pre-training should do

Pre-training is masked patch modelling. Each lead of a window is cut into 20 patches. 40 % of them (8 per lead) are replaced by a learned mask token, and the model reconstructs the normalised signal. The loss is the mean squared error over the **masked patches only** (thesis Section 3.3.2, p. 24). The model is never scored on patches it can see.

The thesis uses three kinds of mask, each on a third of the batch (Section 4.4.2.1, p. 42). In the code the batch is split into three groups (two with `per_lead_aug`), and all windows of a group share one mask. With one lead, a batch therefore holds three masks.

### 2.2 What the code did with one lead

`PatchMaskSupport.before_forward` in `src/core/support/patch_mask.py` builds the mask with shape [batch, leads, patches] and then called `.squeeze()`, which removes every axis of length 1.

- With two or more leads, no axis is removed.
- With one lead, the lead axis is removed and the mask becomes [batch, patches].

The loss multiplies the error per patch, shape [batch, 1, patches], by that mask. With one lead the two shapes no longer line up, so PyTorch broadcasts them to [batch, batch, patches]: every window's error is multiplied by every window's mask.

Written out, with `e(i,p)` the error of window `i` at patch `p`, `m(i,p)` equal to 1 when that patch is masked, and `M` the number of masked patches in the batch:

| | Loss |
|---|---|
| Intended | `(1/M) · Σ_i Σ_p m(i,p) · e(i,p)` |
| One lead, old code | `(1/M) · Σ_i Σ_p c(p) · e(i,p)`, where `c(p) = Σ_j m(j,p)` counts the windows of the batch that mask position `p` |

A window's error at position `p` counted whenever *any* window of the batch masked `p`, whether or not this window did.

**Example.** Two windows, three patches. Window A masks patch 1, window B masks patch 2.
- Intended loss: `(e(A,1) + e(B,2)) / 2`.
- Old code: `(e(A,1) + e(B,1) + e(A,2) + e(B,2)) / 2`. B can see its patch 1 and A can see its patch 2, and both were scored anyway.

Two consequences:

- **Visible patches were scored.** At batch size 128, 0.40 of the loss's weight fell on visible patches on average (0.31 to 0.47 in 9 of 10 batches, depending on the masks drawn). The model was trained partly to reproduce patches it can see. With `per_lead_aug` a batch holds two masks and the average share is 0.30. The project's single-lead pre-training scripts use the default.
- **The value was about one batch size too large.** When the error is the same everywhere, the loss is exactly the batch size times the intended value. A constant factor matters little to AdamW, which scales each update by the running size of the gradient. What the loss scores does matter.

### 2.3 Why two or more leads were not affected

`.squeeze()` only removes axes of length 1. With two or more leads the mask keeps its shape and the loss is the intended one. Old and new code give identical masks and identical loss values for 2 and 12 leads (Section 3.5).

### 2.4 Where it came from

- The two mask lines are original upstream code (commit `c4dcc5d`). Upstream, one-lead input does not run at all: it stops with an error in the mask code, so the case was never exercised. The thesis experiments use 12 leads.
- To let one lead run, a `.squeeze()` was removed from `create_patch` in the same file. The `.squeeze()` on the mask stayed. With that edit single-lead pre-training ran, with the loss above.
- The edit was committed as `0ae886d` on 2026-05-18 and was in use before that: `patch_ecg_chapman_ii_only` is dated 2026-05-15, and classifiers fine-tuned from single-lead Chapman backbones are dated April 2026.

### 2.5 Why it was not noticed

- Training ran, the loss fell, and the fine-tuned models worked.
- **The logged values looked normal.** The tracker divided every pre-training loss by the batch size (problem 2), which cancelled the factor of one batch size in the single-lead loss. In the existing Chapman logs, the single-lead backbones log 0.79 falling to 0.39, plausible for a normalised signal. The two-lead backbone `patch_ecg_chapman_ii_v2` logs 0.010 falling to 0.0044, which is its true loss divided by the batch size of 100.
- **The fine-tuned models did not look broken.** In R1b, the lead II models fine-tuned from the single-lead backbone had a lower test MAE than the supervised models in all 5 folds (mean 0.030 against 0.037 mV). For lead V2 it was lower in 3 of 5 folds (mean 0.0457 against 0.0465 mV).

## 3. Evidence

Every run, log and script is in `GIT/Tasks/pretrain_loss_fix/` on the VM. `FINAL_CHECK.md`, `UNIFIED_CHECK.md` and `PROBE.md` there list each check and run.

### 3.1 The loss on inputs where the right answer is known

`PatchMaskSupport.loss` on synthetic data, one lead, batch size 128 (from the self-test, Section 10):

| Error placed on | Intended loss | Old code | New code |
|---|--:|--:|--:|
| every patch, at random | 2.00 | 255.4 | 2.00 |
| masked patches only | 1 | 78.1 | 1 |
| visible patches only | 0 | 49.9 | 0 |

With 2 and 12 leads, the old and the new code both give the intended value in every case tested.

### 3.2 The same training run before and after the fix

Lead II, 30 epochs, once with the old code and once with the new. On Unified v2 the command is the one in `pretrain_unified_v2.sh`, with 30 epochs in place of 500. "Masked" and "visible" are the backbone's mean squared error on the masked and on the visible patches of validation windows, measured with `src/scripts/checks/backbone_masked_vs_visible.py`.

| Data | Model | Code | Logged best validation loss | Masked | Visible | Visible / masked |
|---|---|---|--:|--:|--:|--:|
| Chapman | PatchECG | old | 0.438 | 0.61 | 0.20 | 0.33 |
| Chapman | PatchECG | new | 0.574 | 0.58 | 0.81 | 1.39 |
| Unified v2 | PatchECG | old | 0.425 | 0.56 | 0.21 | 0.38 |
| Unified v2 | PatchECG | new | 0.558 | 0.57 | 0.73 | 1.29 |
| Chapman | PatchTST | old | 0.591 | 0.99 | 0.025 | 0.026 |
| Chapman | PatchTST | new | 0.987 | 0.99 | 1.39 | 1.40 |

- **Old code: the backbone reproduces what it can see.** PatchECG's error on visible patches is about a third of its error on masked patches (0.33 on Chapman, 0.38 on Unified v2).
- **New code: it does not.** The error on visible patches is the larger one (1.39 and 1.29 times the masked-patch error), as for the two-lead backbones in Section 3.3.
- **The logged loss now means something.** With the new code it is the masked-patch error. With the old code it was neither of the two errors: PatchECG on Chapman logged 0.438 with a masked-patch error of 0.61 and a visible-patch error of 0.20.
- **Masked-patch error after 30 epochs:** new code 0.58 (Chapman) and 0.57 (Unified v2); old code 0.61 and 0.56. This is one run each.
- **PatchTST with these settings did not learn to reconstruct masked patches**, on either code. On the old code it did learn to reproduce the visible ones (visible-patch error 0.025); on the new code it learned neither. See Section 6.4 before reading anything else into its rows.

### 3.3 Backbones that already exist

The same measurement on every fully trained pre-training backbone that is on the VM, on validation windows of the local Chapman split and of the staged Unified v2 split. For a backbone trained on other data or on another split, some of these windows may have been training data. A 1-epoch PatchTST smoke checkpoint of May 2026 is left out.

| Backbone | Leads | Chapman: masked | visible | visible / masked | Unified v2: masked | visible | visible / masked |
|---|---|--:|--:|--:|--:|--:|--:|
| `patch_ecg_chapman_ii_only` | II | 0.54 | 0.19 | 0.35 | 0.50 | 0.17 | 0.35 |
| `patch_ecg_chapman_v2_only` | V2 | 0.52 | 0.21 | 0.41 | 0.46 | 0.18 | 0.38 |
| `patch_ecg_unified_ii_fixed` | II | 0.49 | 0.17 | 0.35 | 0.44 | 0.15 | 0.33 |
| `patch_ecg_unified_v5_fixed` | V5 | 0.57 | 0.24 | 0.43 | 0.54 | 0.21 | 0.39 |
| `patch_ecg_ptbxl_ii` | II | 0.63 | 0.26 | 0.42 | 0.63 | 0.29 | 0.46 |
| `patch_ecg_ptbxl_v2` | V2 | 0.58 | 0.27 | 0.47 | 0.55 | 0.28 | 0.51 |
| `patch_ecg_chapman_ii_v2` | II,V2 | 0.44 | 0.78 | 1.76 | 0.46 | 0.78 | 1.70 |
| `patch_ecg_chapman_ii_v2_tta` | II,V2 | 0.40 | 0.80 | 1.97 | 0.41 | 0.80 | 1.94 |
| `patch_ecg_chapman_ii_v2_no_tta` | II,V2 | 0.40 | 0.99 | 2.47 | 0.41 | 1.01 | 2.48 |
| `patch_ecg_ptbxl_ii_v2` | II,V2 | 0.46 | 1.11 | 2.40 | 0.50 | 1.10 | 2.20 |

- **Every single-lead backbone reproduces visible patches.** Its error on visible patches is 0.33 to 0.51 times its error on masked patches.
- **No two-lead backbone does.** There the error on visible patches is 1.70 to 2.48 times the error on masked patches.
- **The table is one draw of the random masks.** With five draws on the same windows, a ratio differed by up to 21 % between two draws. Over all draws the single-lead ratios stayed between 0.32 and 0.53 and the two-lead ratios between 1.70 and 2.74 (`SEED_SPREAD.md` in the evidence folder). The gap between the two groups is far larger than that.

### 3.4 What the logs show

For each of the 16 pre-training runs of the final checks, the best logged validation loss was compared with a recomputation on the same validation windows, with separate code and freshly drawn masks.

- **New code, 12 runs** (PatchECG and PatchTST, one and two leads, Chapman and Unified v2): the logged value is the masked-patch MSE per window, within 5 % in every run.
- **Old code, two leads, 1 run:** the logged value is the masked-patch MSE divided by the batch size, within 5 %.
- **Old code, one lead, 3 runs:** the logged value is the old single-lead loss divided by the batch size, within 5 % in every run. For PatchECG on Chapman that is 0.438 logged against 0.447 recomputed, where the masked-patch MSE was 0.61.
- The recomputation draws other masks than the training run did, so the two cannot agree exactly: between two draws of the masks alone, a masked-patch error differed by up to 9 % (Section 3.3).

### 3.5 Nothing else changed

- **Loss with 2 and 12 leads.** The old and the new mask code give identical masks and identical loss values (5 batch sizes, both kinds of mask).
- **The thesis's configurations.** See Section 5.2.
- **PatchTST with one lead and with leads fed separately.** The old and the new model code give identical outputs.
- **Regression fine-tuning.** The same seeded fine-tune was run on the old and on the new code from the same backbone, for PatchECG lead II, PatchECG two-lead, PatchTST lead II and PatchTST with two leads fed separately. In all four, every logged value is identical (48 values each: the training log and the train, validation and test metrics). The old code run twice also gives identical values, so a change would have shown.
- **Published models.** Evaluated again with the new code (no training), the published checkpoints of `R2_lead_II_pretrained_fold01`, `R2_lead_V2_pretrained_fold01` and `R2_two_lead_pretrained_fold01` give their published test metrics exactly, at the 6 decimals stored.

### 3.6 Independent review

Before the fix was written, a second pass traced the pre-training loss and the logging without being given these conclusions, using its own reference code. It arrived at the same formula for one lead, found the loss exact for 2 and 12 leads, reproduced the logging error, and found that the upstream code cannot run one lead at all. Its scripts and outputs are in `earlier_rounds/independent_review/` in the evidence folder. The independent reading of this note itself is described in Section 10.

## 4. Code changes

Four source files changed, 27 lines added and 6 removed. Two check scripts were added.

| File | Change | Effect | Does it change training? |
|---|---|---|---|
| `src/core/support/patch_mask.py` | The mask is taken with `[:, :, :, 0]` in place of `[:, :, :, :1].squeeze()`. | The lead axis is kept. The single-lead loss is the intended one (problem 1). | **Yes, single-lead pre-training.** No change with two or more leads. |
| `src/core/support/patch_mask.py` | The loss function carries `reduction = 'mean'`. | The tracker weights each batch by its size, so `train_loss` and `valid_loss` are the mean loss per window (problem 2). | No. It changes logged values only. The best epoch is still the one with the lowest validation loss; the last, smaller batch now counts by its size. |
| `src/core/models/patchTST.py` | After two leads are flattened into one sequence, the number of CLS tokens follows the new shape. | PatchTST with two leads in one sequence runs (problem 3). | No change for one lead or for leads fed separately. |
| `src/core/utils/learner_utils.py` | `transfer_weights` stops with an error when the backbone's positional embedding has a different size. | Before, a two-lead PatchECG backbone loaded into a one-lead model got the same vector in every position and reported success. The other mismatches (a one-lead PatchECG backbone into a two-lead model, PatchTST either way) stopped with an unrelated `IndexError` (problem 4). | No effect on a valid run. |
| `src/patchECG_finetune.py` | A `--pretrained_model_path` that does not exist stops the run. | Before, training went on from random weights, with the optimiser of a pretrained run (problem 4). | No effect on a valid run. |
| `src/scripts/checks/check_training_paths.py` | New. | Self-test of the pre-training and regression paths, about a minute on CPU. | – |
| `src/scripts/checks/backbone_masked_vs_visible.py` | New. | Measures whether a backbone was trained to reproduce visible patches. | – |

Not changed: the fine-tuning loss and metrics, the optimiser, data loading, the models' architecture, and every job script.

## 5. Consistency with the thesis

Each change and each recommendation in this note was checked against the thesis, read from the pages cited.

### 5.1 What the thesis specifies, and what the code does after the fix

| The thesis | Where | The code after the fix | Agrees |
|---|---|---|---|
| The pre-training loss is the MSE over the masked patches. | Section 3.3.2, p. 24 | `PatchMaskSupport.loss`, for any number of leads | **Yes. Before the fix, only with two or more leads.** |
| The reconstruction target is the normalised signal. | Section 3.3.3, p. 25 | The target is normalised before the loss. | Yes, unchanged |
| A special token is prepended. It is left out of the loss and becomes the classification token in fine-tuning. | Section 4.2.1.2, p. 39 | Both models | Yes, unchanged |
| Three kinds of mask, each on a third of the batch. PatchTST does not use the kind drawn across all leads. | Section 4.4.2.1, p. 42 | Three mask groups by default, two with `per_lead_aug` | Yes, unchanged |
| The input is normalised with the mean and variance of each lead over the batch. | Section 4.1.2, p. 38 | `--revin_mode=BN`, the default | Yes, unchanged |
| The best model is the one with the lowest validation loss. | Section 4.4.2, p. 42 | `_best.pt` is saved on the lowest `valid_loss` | Yes. The logged value is now the true loss. |
| PatchECG joins the leads into one sequence. | Table A.3, p. 73 | `--data_augmentation=none` | Yes, unchanged |
| PatchTST takes each lead separately. Leads never interact. | Section 3.3.2, p. 23; Table A.5, p. 75 | `--data_augmentation=per_lead_aug` | Yes, unchanged |
| PatchECG pre-training settings, small model. | Table A.3, p. 73 | `pretrain_unified_v2.sh` | Yes, with the two exceptions below |
| Fine-tuning settings, with layer-wise learning-rate decay. | Section 4.2.1.3, p. 39; Table A.2 for PatchECG; Table A.4 gives PatchTST the same values | The `params.txt` of the R2 runs, which are all PatchECG | Yes for PatchECG. No PatchTST model has been fine-tuned for a result yet. |

Two entries of Table A.3 are not what the code does, before or after the fix:

- **The "minimal learning rate" of 1e-5.** The schedule the trainer builds ends near 6e-9. The flags `--min_lr`, `--warmup_epochs` and `--warmup_lr` are read but not used; the warm-up is the first 30 % of the schedule.
- **"Adam β (0.9, 0.999)".** The schedule moves the first β between 0.85 and 0.95 during training. Table A.5 lists the same β for PatchTST.

Both are in the original upstream code (commit `c4dcc5d`) and were left alone.

### 5.2 The thesis's own configurations are unchanged by the fix

The thesis trains PatchECG on 12 leads in one sequence and PatchTST on 12 leads fed separately. For both, 4 training steps from the same seed give the same losses and the same weights on the old and the new code, with no difference at all. The same procedure with one lead does differ, as it should. For these configurations only the logged loss changes: it is no longer divided by the batch size, and the last, smaller batch of an epoch counts by its size. In a close call that can change which epoch is saved as the best. The steps were run on synthetic data; a full training run was not compared.

### 5.3 Where this project already departs from the thesis

These are design choices of the project, older than this fix:

- **One or two leads** in place of 12. This is where the defect was.
- **Regression.** The thesis fine-tunes for classification only. The regression head and the MSE loss on the ST level are this project's. For PatchTST, the regression head gives one output per lead, where the thesis's PatchTST classifier averages the per-lead predictions (Section 4.2.2, p. 39).
- **The data.** Pre-training uses the corrected Unified v2.
- **Per-window normalisation in the beat classifiers.** The INCART and multi-database beat classifiers have their own training scripts and normalise each window by itself (`norm_stats: instance`). The mode was added to the repo for them on 2026-10-15.

### 5.4 Options in this note that would depart from the thesis

None of these is part of the fix. The two-lead PatchTST runs of the final checks all used the first one; Section 6.1 says which run used what.

| Option | What the thesis does | Discussed in |
|---|---|---|
| `--revin_mode=instance` for PatchTST | Batch statistics for PatchECG and PatchTST (Section 4.1.2; MLE-ViT is the thesis's exception). Figure 3.4, taken from the PatchTST paper, shows instance normalisation, but the thesis's code had no such mode. It was added to this repo on 2026-10-15 (Section 5.3). | Sections 6.3 and 6.4 |
| PatchTST with two leads in one sequence | PatchTST never joins leads. | Section 6.2 |
| PatchTST with PatchECG's settings | PatchTST has its own settings (Table A.5). | Section 6.4 |

## 6. What works now

### 6.1 Every configuration

"Old" is what commit `813c715` does: from the self-test run on that commit, and from old-code runs where the configuration ran. "New" is from the runs listed in `FINAL_CHECK.md` and `UNIFIED_CHECK.md`: each configuration was pre-trained, on Chapman and on Unified v2, and fine-tuned for regression on phase-1 fold 1 from the Chapman backbone. These runs used PatchECG's settings for both models (patch length 50, no dropout).

| Configuration | Flags | Pre-training, old → new | Regression fine-tuning, old → new |
|---|---|---|---|
| PatchECG, one lead | `--model=vanilla_vit --custom_lead_selection=II` | wrong loss → correct | correct → unchanged |
| PatchECG, two leads in one sequence | `--model=vanilla_vit --custom_lead_selection=II,V2` | correct → unchanged | correct → unchanged |
| PatchTST, one lead | `--model=patch_tst --custom_lead_selection=II --data_augmentation=per_lead_aug` | wrong loss → correct | ran → unchanged |
| PatchTST, two leads fed separately | `--model=patch_tst --custom_lead_selection=II,V2 --data_augmentation=per_lead_aug` | correct → unchanged | ran → unchanged |
| PatchTST, two leads in one sequence | `--model=patch_tst --custom_lead_selection=II,V2` | crashed → runs | crashed → runs |

- `per_lead_aug` is the thesis's flag for PatchTST. With one lead it changes one thing only: a batch holds two masks instead of three. The one-lead PatchTST runs of the final checks used the default; the probe's runs with the thesis's settings used `per_lead_aug`.
- **Which normalisation the PatchTST runs used.** PatchECG ran with batch statistics throughout. PatchTST with one lead ran with batch statistics and with `instance` on Chapman, and with `instance` on Unified v2. PatchTST with two leads, in both layouts, ran with `instance`. The thesis's own PatchTST configuration (leads fed separately, batch statistics, the settings of Table A.5) went through the trainer in separate runs on Chapman: one lead for 30 epochs and two leads for 5 epochs, each followed by a regression fine-tune (Section 6.4).
- Fine-tuning takes the same flags plus `--pretrained_model_path=<backbone>`. The backbone must come from the same model and have the same number of patch positions. A backbone with another number of positions (one lead against two leads in one sequence, or another patch length) now stops with an error, and a backbone of the other model stops as it did before. A one-lead backbone and a leads-fed-separately backbone have the same positions and load into each other, because each lead passes through the model alone. A backbone with more layers than the model is not caught (Section 9).
- "Runs" and "correct" say that the code does what it should. They say nothing about how good the trained model is: see Section 6.4 for PatchTST.

### 6.2 Two ways to give PatchTST two leads

| | Leads fed separately (`per_lead_aug`) | Leads in one sequence (the default) |
|---|---|---|
| What the model sees | Each lead alone (20 patches at a patch length of 50) | Both leads joined (40 patches) |
| Outputs | Each output reads only its own lead | Both outputs read both leads |
| In the thesis | Yes. This is the thesis's lead layout for PatchTST (Section 3.3.2, Table A.5). Table A.5 also sets a patch length of 20, which gives 50 patches per lead. | No. It is how PatchECG takes two leads (Table A.3). |
| Old code | Ran | Crashed |

"Fed separately" is the thesis's layout; it behaves like two single-lead models that share all their weights. "In one sequence" departs from the thesis. It differs from two-lead PatchECG in the model only.

### 6.3 PatchTST and the input normalisation

The thesis normalises the input with batch statistics (`--revin_mode=BN`, Section 4.1.2, p. 38), and every PatchECG model trained with the repo's trainer does the same. The beat classifiers have their own scripts and normalise per window (Section 5.3).

The measurements below use the 23,754 lead II test windows of phase-1 fold 1 and smoke models (3 epochs of fine-tuning), so only the changes matter, not the errors.

- **PatchECG: the batch does not reach the output.** PatchECG normalises every patch again as its first step. Its predictions changed by at most 0.00003 mV between batches of 128, single windows and shuffled batches, and by at most 0.00001 mV between `BN` and `instance`.
- **PatchTST: with batch statistics, a window's prediction depends on the other windows in its batch.** PatchTST has no per-patch step.

  | PatchTST with `BN`, test windows given as | Test MSE | Change of a prediction: median | largest |
  |---|--:|--:|--:|
  | batches of 128 in file order (the trainer's test pass) | 0.0061 | – | – |
  | single windows | 0.0062 | 0.004 mV | 0.18 mV |
  | batches of 128, shuffled | 0.0033 | 0.062 mV | 0.40 mV |

  A batch in file order holds consecutive windows of one record. A shuffled batch mixes the test records, as a training batch mixes the training records. Two different shufflings agree with each other far more closely than either agrees with the file order: their median difference is 0.004 mV (largest 0.11 mV).
- **With `--revin_mode=instance`, PatchTST's predictions do not depend on the batch.** Each window is normalised by its own mean and SD. The largest change across the same batchings was 0.00000 mV.

What to do with PatchTST is a choice:

- **Keep the thesis's batch statistics.** Then use one way of batching for training, testing and scoring, and state it, because the numbers depend on it. The VM scoring uses batches of 1,024 windows where the trainer uses 128.
- **Use `instance`, in pre-training and in fine-tuning.** Predictions no longer depend on the batch. This departs from the thesis (Section 5.4). It removes each window's overall offset and amplitude; differences between the patches of a window stay visible to PatchTST.

If agreement with the thesis comes first, run PatchTST with batch statistics as the main configuration, and `instance` as a comparison that changes one thing.

### 6.4 The settings for PatchTST

The validation loss here is the masked-patch MSE on a normalised signal. A model that predicts zero scores 1.0.

| Run (lead II, Chapman, 30 epochs, new code) | Normalisation | Peak learning rate | Patch | Dropout | Validation loss, epoch 1 | best | epoch 30 |
|---|---|--:|--:|--:|--:|--:|--:|
| PatchECG, its own settings (thesis Table A.3) | batch | 0.0015 | 50 | 0 | 1.00 | 0.57 | 0.58 |
| PatchTST, PatchECG's settings | batch | 0.0015 | 50 | 0 | 0.99 | 0.99 | 1.00 |
| PatchTST, the same at a lower learning rate | batch | 0.0005 | 50 | 0 | 1.00 | 0.96 | 0.97 |
| PatchTST, thesis Table A.5 as written (`per_lead_aug`) | batch | 0.0001 | 20 | 0.1 | 1.01 | 0.98 | 0.98 |
| PatchTST, thesis Table A.5 with the learning rate of the thesis's script | batch | 0.0005 | 20 | 0.1 | 1.00 | 0.97 | 0.98 |
| PatchTST, thesis Table A.5 and script learning rate, per-window normalisation | per window | 0.0005 | 20 | 0.1 | 0.99 | 0.40 | 0.41 |
| PatchTST, PatchECG's settings, per-window normalisation | per window | 0.0015 | 50 | 0 | 0.99 | 0.43 | 0.43 |
| PatchTST, PatchECG's settings, per-window normalisation, lower learning rate | per window | 0.0005 | 50 | 0 | 0.99 | 0.52 | 0.52 |
| PatchTST, PatchECG's settings, per-window normalisation, lowest learning rate | per window | 0.0001 | 50 | 0 | 1.00 | 0.95 | 0.95 |

- **With the thesis's batch statistics, PatchTST's validation loss stayed near 1.0 for all 30 epochs, in each of the four runs.** Its best value was 0.99 and 0.96 with PatchECG's other settings, and 0.98 and 0.97 with the thesis's own PatchTST settings (Table A.5: patch length 20, dropout 0.1, leads fed separately), at the table's learning rate and at its script's.
- **With per-window normalisation it learned, also with the thesis's own settings.** With Table A.5's settings and the script's learning rate, changing only the normalisation took the best validation loss from 0.97 to 0.40. With PatchECG's settings it reached 0.43 at a learning rate of 1.5e-3 and 0.52 at 5e-4, against 0.99 and 0.96 with batch statistics. PatchECG itself reached 0.57 on the same data.
- **The learning rate matters as well.** The runs that learned stayed near 1.0 at first and were below 0.9 only from epoch 8, 10 and 13 on (PatchECG: from epoch 5). At 1e-4, per-window normalisation was still at 0.95 after 30 epochs, falling slowly.

What this does and does not show:

- These are single runs of 30 epochs on lead II of Chapman. They show which settings start to learn within 30 epochs, not which are best.
- **They do not show what the thesis's 500 epochs would do with batch statistics.** A run can stay near 1.0 for several epochs and then fall, as the per-window runs did. Whether the batch-statistics runs would fall later was not tested.
- The thesis trained PatchTST for 500 epochs on 12 leads with batch statistics, and reports it as the weakest of its three models and as unstable in pre-training (p. 47 to 48). Nothing here contradicts the thesis; these runs are a different setting.
- The thesis's PatchTST path runs end to end with batch statistics: pre-training and a regression fine-tune with the backbone fully loaded, for one lead and for two leads fed separately (patch length 20, 3 epochs of fine-tuning). The backbones those fine-tunes started from had not learned.
- With two leads fed separately, the per-window runs of the final checks used PatchECG's settings and 5 epochs. Their validation loss fell to 0.78 on Chapman and 0.66 on Unified v2.

**Recommendation.** For PatchTST on one or two leads, keep the thesis's PatchTST settings (Table A.5, leads fed separately, the script's learning rate of 5e-4) and change one thing: per-window normalisation (`--revin_mode=instance`), in pre-training and in fine-tuning. It is the one change to the thesis's settings that learned here, and it removes the dependence on the batch (Section 6.3). It departs from the thesis's Section 4.1.2 and should be reported as a departure; it is the normalisation that the PatchTST design in the thesis's Figure 3.4 shows. To keep batch statistics as in the thesis, watch the validation loss: it starts near 1.0, and for as long as it stays there the model has not started to learn.

For reference, the thesis's settings for the two models. Its table and its script for PatchTST disagree in two of them:

| Setting | PatchECG, thesis Table A.3 | PatchTST, thesis Table A.5 | PatchTST, thesis script |
|---|---|---|---|
| Patch length and stride | 50 | 20 | 20 |
| Dropout and head dropout | 0 | 0.1 | 0.1 |
| Peak learning rate | 1.5e-3 | 1e-4 | 5e-4 |
| Batch size | 128 | 128 | 256 |
| Leads | in one sequence | fed separately | fed separately |

The script is `src/scripts/multi-gpu/pretrain/transformer_comparison/patch_tst_vit.sh`. Table A.5 also lists 20 patches per lead, which a patch length of 20 does not give (1,000 samples make 50 patches); the script uses a patch length of 20. The runs here used a batch size of 128. The fine-tuning settings are the same for both models (thesis Tables A.2 and A.4).

## 7. What this means for existing results

**Which backbones**

| Backbones | Leads | Pre-training loss | How this is known |
|---|---|---|---|
| `patch_ecg_chapman_ii_only`, `patch_ecg_chapman_v2_only`, `patch_ecg_unified_ii_fixed`, `patch_ecg_unified_v5_fixed`, `patch_ecg_ptbxl_ii`, `patch_ecg_ptbxl_v2` | 1 | Old single-lead loss | Measured (Section 3.3) |
| `patch_ecg_unified_ii_only`, `patch_ecg_unified_v2_only`, `patch_ecg_unified_v2_fixed`, `patch_ecg_unified_ii_rebuilt`, `patch_ecg_unified_v2_rebuilt`, `patch_ecg_unified_ii`, `patch_ecg_unified_v5`, `patch_ecg_ptbxl_v5` | 1 | Old single-lead loss | From the code. These files are not on the VM. The two mask lines are the same in every commit up to `813c715`, so every single-lead run used them, unless the copy of the code that trained it carried an uncommitted edit to that file. The backbone tool (Section 8, step 2) can confirm it wherever the files are. |
| The Chapman single-lead backbones of April 2026 (`patch_ecg_chapman_single_lead_*`) | 1 | Not known | Not on the VM, and older than the commit that lets one lead run, so the code that trained them is not in the repository. |
| `patch_ecg_chapman_ii_v2`, `patch_ecg_chapman_ii_v2_tta`, `patch_ecg_chapman_ii_v2_no_tta`, `patch_ecg_ptbxl_ii_v2` | 2 | Intended | Measured (Section 3.3) |
| `patch_ecg_unified_ii_v2`, `patch_ecg_unified_ii_v5`, `patch_ecg_ptbxl_ii_v5`, `patch_ecg_unified_ii_v2_rebuilt` | 2 | Intended | From the code (files not on the VM) |

"Unless" in the second row is not only a formality. The VM has run code that was never committed: single-lead backbones existed in April 2026, before commit `0ae886d`, and a PatchTST smoke checkpoint of 2026-05-20 has two leads in one sequence, which no committed version before this fix can run.

**Which models were fine-tuned from a single-lead backbone** (read from each run's `params.txt`, and from the checkpoints of the beat classifiers):

| Work | Runs | Backbones |
|---|---|---|
| LTST ST regression, single-lead models | R1 and R1b (lead II, lead V2), R2, R2b, R3, Test 1 (both arms), Test 1-NK, E1, the single-lead runs in `ptbxl_models_v2`, and the earlier single-lead `ltstdb_median_*` runs | `_only`, `_rebuilt`, `_fixed`, `ptbxl_ii`, `ptbxl_v2`, `chapman_ii_only`, `chapman_v2_only` |
| Single-lead classifiers on PTB-XL labels | `unified2ptbxl_ii_*`, `unified2ptbxl_v5_*` (not the two-lead `unified2ptbxl_ii_v5_*`) | `patch_ecg_unified_ii`, `patch_ecg_unified_v5` and their `_fixed` versions |
| Beat classifiers | INCART (`incart_vm*`) and multi-database (`incart_multi_nk*`) | `patch_ecg_unified_ii_fixed`, `patch_ecg_unified_v5_fixed` |
| Single-lead classifiers on Unified labels | `ptbxl2unified_ii_*`, `ptbxl2unified_v5_*` (not the two-lead `ptbxl2unified_ii_v5_*`) | `patch_ecg_ptbxl_ii`, `patch_ecg_ptbxl_v5`. These models are already invalid for another reason (`src/scripts/unification/README_unified_dataset.md`). |
| Earlier single-lead binary classifiers | The 4 pretrained runs in `~/Desktop/Extra/single-lead-binary/` (April 2026) | The Chapman single-lead backbones of April 2026. Their loss is not known (table above). |

**Not affected**

- Supervised models: they load no backbone.
- Models fine-tuned from a two-lead backbone: the two-lead regression models of R1, R1b, R2 and Test 1, the two-lead runs in `ptbxl_models_v2`, the earlier two-lead `ltstdb_*` runs, and the two-lead classifiers (`*_ii_v5_*`).
- The NeuroKit pipelines.
- 12-lead backbones: the thesis's own, and the project's `patch_ecg_unified_all_leads`.

**How to read existing comparisons**

- **The regression scores stand.** Each is what that model does on that data. Fine-tuning and its metrics were not affected.
- **"Pretrained against supervised" for single-lead models** compares supervised training with a backbone trained on the mixed loss. For lead II this backbone helped (all 5 R1b folds); for lead V2 the result was mixed (3 of 5 folds). Neither says how much a correctly pre-trained backbone helps.
- **Single-lead against two-lead** (for example the R2 single-lead and two-lead pipelines in `PHASE1_MAIN_RESULTS.md`) now differs in two things: the model and the backbone's loss. Until a corrected single-lead backbone is tested, the difference cannot be put down to the model alone.
- **R1b against R2 for single-lead models** is not changed by this: both backbones were trained with the old single-lead loss.
- **Wording.** Do not describe the single-lead backbones as pre-trained with the thesis's masked patch modelling. An accurate description: the loss also scored the visible patches, with about 40 % of its weight.

## 8. Retraining

The steps are the same on the cluster and on the VM. The VM now has the corrected Unified v2 data (`~/Desktop/Research Data/DATASET/Unified_Dataset_No_PTB_Segment_v2`, checksums verified) and a GPU.

1. **Get the fixed code onto the machine that will train, and run the self-test there** from the repo root. For the cluster, the fix has to be committed and pushed first. The self-test needs no data and no GPU:
   ```bash
   .venv/bin/python src/scripts/checks/check_training_paths.py
   ```
   It must end with `ALL CHECKS PASSED` (37 of 37). On the old code it reports 30 of 37.
2. **Optional: confirm an old backbone that is not on the VM**, for example on the cluster:
   ```bash
   .venv/bin/python src/scripts/checks/backbone_masked_vs_visible.py \
       --backbone results/pre-train/saved_models/unified/patch_ecg_unified_ii_rebuilt/patch_ecg_unified_ii_rebuilt_best.pt \
       --h5 <Unified_Dataset_No_PTB_Segment_v2>/UnifiedDataset.h5 --leads II
   ```
   Expected reading for a single-lead backbone trained with the old code: `reproduces visible patches`. The ratio it prints depends on the masks drawn (Section 3.3); `--seed` draws others.
3. **Pre-train the single-lead backbones again.** Same script and recipe as before, with new model names (the script refuses a folder that exists):
   ```bash
   sbatch --job-name=<new_name> src/scripts/multi-gpu/pretrain/unified/pretrain_unified_v2.sh II <new_name>
   ```
   Repeat for V2 and V5 as needed. Two-lead backbones do not need retraining. Without Slurm, the two commands inside that script (staging, then `src.patchECG_pretrain`) run as they are; `run_unified_check.sh` in the evidence folder did so on the VM for 30 epochs.
4. **Check each new backbone.**
   - Its logged loss is now the masked-patch MSE per window. On a normalised signal it starts near 1 and falls from there. Do not compare it with old single-lead logs, which show the mixed loss. Old two-lead logs can be compared after multiplying them by the batch size, which is close but not exact because the last, smaller batch was counted like a full one.
   - The backbone tool of step 2 should read `does not reproduce visible patches`.
5. **Fine-tune again** with the unchanged recipe, from the new backbones. `build_phase1_sbatches.py` holds the backbone names in its `OLD` and `NEW` tables (lines 32–33); the new names have to be added there.
6. **Compare one change at a time.** R2 with the new single-lead backbones against R2 as published has the same data, folds, seeds and recipe. Only the backbone's loss differs.

Keep in mind when reading the comparison:
- Pre-training has no seed, and each backbone is one run. Each fold is one fine-tuning run. A difference smaller than the R1b-to-R2 differences cannot be put down to the fix.
- The published R2 single-lead rows are in `Tasks/phase1/PHASE1_RESULTS.md` on the VM.

**For PatchTST**, also needed before a run:
- Settings that train it (Section 6.4) and a decision on the normalisation (Section 6.3).
- The job scripts set `--model=vanilla_vit` themselves: `pretrain_unified_v2.sh` (line 36), `build_phase1_sbatches.py` (line 114) and `build_v2_chain_sbatches.py` (line 95, not yet committed). A PatchTST job needs `--model=patch_tst` and the flags of Section 6.1.
- The VM scoring code builds PatchECG only (`load_patchecg_model` in `Neurokit2_Pipeline/ltst_pipeline/ml_inference.py`). A PatchTST checkpoint cannot go through the FSM and alarm scoring until that loader can build PatchTST and apply its normalisation.

## 9. Known limits that were not changed

- **Leftover snapshots are resumed.** The trainer continues from any `*_snapshot.pt` in the model folder. The phase-1 job generator refuses to start when one exists. A resumed fine-tune would also apply the layer-wise learning-rate decay a second time (read from `Learner.__init__` and `Learner.fine_tune`; not run).
- **Pre-training has no seed.** Two pre-training runs with the same settings give different backbones.
- **PatchECG cannot see a patch's offset or amplitude.** It normalises every patch (LayerNorm) before anything else. This is by design and was already known from the lead V2 analysis. PatchTST has no such step.
- **PatchTST keeps updating its BatchNorm statistics during the head-only epochs**, although the backbone weights are frozen. Whether that matters was not tested.
- **PatchECG with `--data_augmentation per_lead_aug` does not do regression.** Its output has one row per lead for every target.
- **`--revin_mode ptb-xl` works for 12 leads only.** With one lead it silently returns 12 channels; with two it stops with an error.
- **Rows without a target are not dropped when a fold column already exists.** The phase-1 and v2 fold files have no such rows.
- **`transfer_weights` still only prints a warning** for other layers whose shape does not match (for example a different feed-forward size). Only the positional embedding now stops the run.
- **A backbone with more layers than the model loads without any message.** `transfer_weights` goes through the model's own layers, so the extra layers of the backbone are dropped (tried with a 6-layer backbone and a 4-layer model).
- **CPC pre-training logs are still divided by the batch size.** The logging change is on the masked-patch loss. The CPC loss (`--model cpc`) does not have it (read, not run).

## 10. How this was checked, and what was not checked

**Checked**

1. **Self-test** (`src/scripts/checks/check_training_paths.py`). 37 checks with the repo's own classes on synthetic data: the loss against a separately written reference for 1, 2 and 12 leads; forward passes for 8 lead layouts, including the two the thesis trains; the logged loss; backbone loading; layer-wise learning rates; dependence on the batch. New code: 37 of 37. Old code: 30 of 37. The 7 that fail on the old code are the single-lead loss with both kinds of mask, PatchTST with two leads in one sequence in pre-training and in regression, the logged pre-training loss, and the two wrong-backbone loads.
2. **Final check on Chapman** (`run_final_check.sh`, report `FINAL_CHECK.md`). 36 runs with frozen copies of the old and the new code; 70 checks, all passed.
3. **Check on Unified v2** (`run_unified_check.sh`, report `UNIFIED_CHECK.md`). The cluster's staging and pre-training command on the corrected data: lead II for 30 epochs on the old and on the new code, the other lead layouts for 5 epochs on the new code, and two regression fine-tunes. 8 runs; 22 checks, all passed.
4. **PatchTST probe** (`run_tst_lr_probe.sh` and `run_tst_probe2.sh` to `run_tst_probe4.sh`, report `PROBE.md`). 8 pre-training runs (7 of 30 epochs on lead II, 1 of 5 epochs on two leads) and 2 regression fine-tunes.
5. **The thesis.** The pages cited were read from the PDF, and the audit looked every citation up again in the thesis text.
6. **Other uses of `.squeeze()`.** No other bare `.squeeze()` is on the pre-training or regression path of either model. The remaining ones are in the classification test-time augmentation, the CPC model's branches, the bootstrap function and the LSTM baseline.
7. **Number audit of this note** (`audit_readme.py`, report `audit_report.md`). Every number here was recomputed from the raw outputs with separately written code, or read at its source when it is a setting, and each claim marked there was tested. The backbone errors of Sections 3.2 and 3.3 depend on the random masks, so they were checked in two steps: each value was reproduced by running the tool again with the same seed, and the tool was compared with separately written code over five other draws of the masks. Averaged over the draws, every masked-patch error, visible-patch error and ratio agrees within 8 %.
8. **Independent reading.** A reader who was not given the audit, and who was asked to find what is wrong, unsupported or one-sided, checked an earlier version of this note against the code, the raw outputs and the thesis with its own code. It confirmed the loss formula and the worked example, the numbers of Section 3, the unchanged fine-tuning and the self-test counts. It reported 13 points. Each was reproduced, and each is corrected in this version. The main ones: two single-lead backbones listed as missing are on the VM and are now measured (Section 3.3); most PatchTST runs had used per-window normalisation without the note saying so (Section 6.1); only one of the wrong-backbone loads had been silent (Section 4); a ratio moves more with the masks drawn than the note said (Section 3.3); lead V2 does not show the benefit that lead II shows (Section 2.5); the fix was described as committed (header, Section 8); and several citations claimed more than the thesis passage says (Sections 2.1, 5.1, 5.4 and 6.2). Its report and scripts are in `independent_read/` in the evidence folder. Section 6.4 was not finished when that reading was done. Neither it nor the corrections have had a second independent reading.

**Not checked**

- **Full-length training.** The runs here are 5 to 30 epochs of pre-training and 3 epochs of fine-tuning. Their scores carry no meaning.
- **The effect of the fix on accuracy.** That is what the retraining is for.
- **The cluster.** Slurm, modules and paths were not exercised. The pre-training command of `pretrain_unified_v2.sh` was run on the VM.
- **The backbones that are not on the VM** (Section 7).
- **Pre-training on more than one GPU.**
- **Classification fine-tuning.** None of the changed lines is on its path, apart from the two backbone-loading checks.
- **Everything after the trainer:** the FSM, the gate and the alarm scoring were not touched and not re-checked.
- **A second independent reading.** Section 6.4 and the corrections made after the independent reading were audited number by number, but were not read independently again.
