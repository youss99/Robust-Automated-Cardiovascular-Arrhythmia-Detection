# step3e verification (VM, 2026-11-24)

step3e = 45 end-to-end supervised jobs (decision and fixed comparisons: `docs/DECISIONS.md`, 2026-11-24). Nothing was trained beyond one epoch per check case, nothing was submitted, and no training code was changed. Detailed outputs: `2026-11-24_step3e_dry_run.md` (with `2026-11-24_step3e_dry_run_jobs.csv`) and `2026-11-24_step3e_gpu_check.md`.

**Status: all checks pass. Waiting for the user's final approval; not committed or submitted.**

## Files

| file | what |
|---|---|
| `src/scripts/custom_dataset/build_e2e_sup_sbatches.py` (new) | Writes the 45 jobs by rendering each step3/step3o twin with the unchanged `build_final_sbatches.render_finetune`, then swapping only the epoch lines (20 + 70 → 0 + 90) and the run ID. It asserts this for every job. Output: `logs/sbatch_scripts/final_e2e/` |
| `src/scripts/final/dry_run_e2e.py` (new) | Dry run: checks 1–4 below, plus fault injection |
| `src/scripts/checks/check_e2e_supervised.py` (new) | GPU check: the real trainer, the generated jobs' flags, one epoch |
| `docs/DECISIONS.md`, `docs/instructions_final_results.md`, `docs/README_final_narval_notes.md` | step3e decision, fixed comparisons and metrics, Narval steps |

No change to `src/core`, `src/patchECG_*.py`, `build_final_sbatches.py`, the fold manifests or the fold folders.

## 1. Same seed, same initial weights

- **On the VM, bit for bit** (`check_e2e_supervised.py`, real trainer, fold 1, seed 200): the model state when `Learner.fine_tune` starts is identical for each step3e job and its twin, in lead II, lead V2 and two-lead (79 tensors; 1,197,541 values single-lead, 1,200,230 two-lead).
  - *Control:* seed 999 gives different weights.
- **Against the completed Narval runs.** The VM run of each twin reproduces the twin's epoch-0 row of `_training_metrics_gpu0_*.csv` from Narval exactly (difference 0.00e+00 at the CSV's 6 decimals): train/valid loss, MAE and R², all three modes.

  | mode | train loss | valid loss | valid R² |
  |---|--:|--:|--:|
  | lead II | 0.010149 | 0.004262 | −1.255742 |
  | lead V2 | 0.006907 | 0.002103 | −0.165234 |
  | two-lead | 0.012964 | 0.003197 | −0.735767 |

  The step3e jobs share the twins' initial weights on the VM, so they also share the Narval twins' initial weights and first-epoch batch order.
  - *Limit:* the starting state was compared bit for bit for fold 1, seed 200 only. Other folds and seeds run the same code path with a different seed number.

## 2. Every regression parameter is trained from the first epoch

Real trainer with the step3e flags, all three modes:
- **Optimizer when training starts:** one AdamW group holding all 79 parameter tensors, lr 1e-3, weight decay 0.05, `lr_scale` 1. With 0 head epochs there is no `freeze()` and no checkpoint reload, and `fine_tune` is called with (`freeze_epochs` 0, `n_epochs` 90).
- **First optimizer step:**
  - 78 of 79 tensors require a gradient and have a finite, non-zero gradient in every value. This covers the patch embedding (both LayerNorms and the linear layer), the positional embedding, the CLS token, all 6 attention and feed-forward blocks, and the head.
  - The exception is `mask_token`, which is used only in pre-training: it has no gradient in any regression run.
- **After epoch 0:** every value of those 78 tensors has changed; `mask_token` is unchanged.
- **Control** (the twin's first epoch, head-only): only the 4 `mlp_head.*` tensors get gradients and change. This shows the check can tell a frozen encoder from a trained one.

## 3. Jobs differ from their twins only in epochs and identifiers

- **Job text:** 45 of 45 differ from the twin's text in exactly two lines, `--n_epochs_finetune_head=20 → 0` and `--n_epochs_finetune=70 → 90`. The run ID and the pipeline label are excluded from the comparison.
- **Flags:** 45 of 45 go through the real parser and are compared value by value (94 values) with the twin's `params.txt` from Narval.
  - The only differing keys are `model_name`, `save_path`, `root_path` (Narval's per-job scratch path) and the two epoch flags.
  - The seed equals the twin's. There is no backbone in either.
- **Existing generator unchanged:** `build_final_sbatches.py` still writes the approved `jobs.csv` (sha256 `2133e920…`).
- **Fault injection:** each of 7 injected faults is flagged: changed lr, head epochs left at 20, weight decay 0, 70 full epochs, an added backbone, a wrong fold folder, and a wrong logged CSV checksum.
- **Self-test:** `check_training_paths.py` still prints 37/37.

## 4. Same datasets and fold assignments (checksums)

For all 45 jobs, three checksums agree:
- the data folder is the twin's (its `job_log.txt`);
- the fold CSV here has the sha256 the twin logged on Narval (`dataset_csv_sha256`);
- the same sha256 is in `phase1_folds/SHA256SUMS.txt`.

The H5 in all 5 folders matches `SHA256SUMS.txt` (`e4bff370…`). In every fold, the CSV's split column assigns exactly the manifest's records to train / validation / test / unused (folds 1, 4, 5: 10 / 2 / 3 / 1; folds 2, 3: 11 / 2 / 3 / 0).

On Narval, `link_phase1_h5.sh` already checked the H5 there against the same `SHA256SUMS.txt` for the final runs.

## 5. Weight decay in pretrained fine-tuning: LayerNorm weights are decayed

- **Code.** Pretrained fine-tuning builds its optimizer with `prepare_optimizer` → `create_optimizer(filter_bias_and_bn=True)` → `get_parameter_groups`.
  - The no-decay rule there is `name.endswith(".bias") or name in skip_list` (`optim_factory.py:66`).
  - `skip_list` is empty, because the ViT defines no `no_weight_decay()` (nothing in `src/core` does).
  - Only biases are excluded. Despite the flag's name, normalisation weights are not filtered.
- **Run** (real trainer, L1 lead II pretrained job, stopped when training starts):
  - 16 groups (8 layer ids × decay / no decay), `lr_scale` 0.049–1.0;
  - weight decay 0 on the 35 bias tensors;
  - weight decay 0.05 on the other 44, including all 15 LayerNorm weights, `pos_embedding`, `cls_token` and `mask_token`.
  - The supervised arm, for comparison: 1 group, weight decay 0.05 on all 79.
- **L2** uses the same code path and parameter names, but was not run here because its backbone (`ii_v2_rebuilt`) is only on Narval.
- **Documents.** The 2026-11-20 regression-path reports ("except biases") are correct. Wording in VM chat answers ("not on biases/norms") was wrong. No committed document says norms are excluded.
- **Thesis wording:** weight decay 0.05 on all parameters except biases (pretrained fine-tuning); on all parameters (supervised fine-tuning and pre-training).

**Related issues in the appendix hyperparameter table** (`appendix_hyperparams.tex`, generated from `params.txt` values; not changed here). The table shows the values passed to the training script, and some of them are never applied:
- layer-wise decay 0.65 is shown for the no-PT columns but is not applied there (no backbone → `create_optimizer`);
- scaler "standard" is never used;
- warm-up and minimum LR do nothing without a scheduler;
- betas "None" means PyTorch's defaults (0.9, 0.999);
- the weight-decay row doesn't say which parameters it covers.

These are worth a separate fix to `fin_step7_appendix.py` if approved.

## 6. Compute

The twins took 11.2 GPU-hours in total (median 13.9 / 12.2 / 18.6 min for L1-II / L1-V2 / L2; longest 21.3 min).
- Head-only epochs skip the encoder backward pass, so 90 full epochs cost more.
- The upper bound is 90/70 of the twin: about 14.4 GPU-hours on `a100_3g.20gb`, longest job about 27 min (walltime 45 min, unchanged).

## Scratch left on the VM (git-ignored)

- `results/fine-tune/saved_models/patient-segment-regression/_VMCHECK_e2e_*` (10 folders: 7 check cases, 3 weight-decay dumps), each holding at most an epoch-0 checkpoint and snapshot. These are not results.
- The check's work folder is in the session scratchpad.
