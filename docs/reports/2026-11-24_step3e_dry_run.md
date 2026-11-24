# step3e dry run (2026-11-24 00:29, VM)

Generated folder: `/tmp/claude-1000/-home-student-GIT/dfa42d17-c0b1-4a95-a5f7-13cc2acca044/scratchpad/e2e/gen`. Commit: `89f0641` plus the uncommitted step3e files. 45 jobs: L1supE2E lead_II 15, L1supE2E lead_V2 15, L2supE2E two_lead 15.

## 1. The existing final generator is unchanged

Re-run into a temporary folder: jobs.csv sha256 `2133e92037d6c522...`; approved 2026-11-20 copy `2133e92037d6c522...`; expected `2133e92037d6c522...`. **PASS**

## 2. Job text vs the twin's job text

Twins: step3o 30, step3 15. Changed lines must be exactly the two epoch lines. **45 of 45 PASS**.

## 3. Every flag through the real parser vs the twin's params.txt

94 values per job compared. Allowed: model_name, root_path, save_path; epoch flags must read 20 -> 0 and 70 -> 90; seed must equal the twin's and jobs.csv; backbone None in both. Keys that differed in at least one job: model_name, n_epochs_finetune, n_epochs_finetune_head, root_path, save_path. **45 of 45 PASS**.

Effective settings (job 1, from the parser): model=vanilla_vit, d_model=128, n_layers=6, n_heads=8, d_ff=512, patch_len=50, stride=50, class_token=cls_token, revin_mode=bn, custom_lead_selection=ii, opt=adamw, lr=0.001, weight_decay=0.05, layer_decay=0.65, scheduler=false, clip_grad=none, sam=false, batch_size=128, regression_loss=mse, data_augmentation=none, dropout=0.0, head_dropout=0.0, n_epochs_finetune_head=0, n_epochs_finetune=90, model_selection_metric=valid_loss, pretrained_model_path=none. layer_decay is read only by prepare_optimizer, which runs only with a backbone (src/patchECG_finetune.py:304-305).

## 4. Datasets and fold assignment

Data folder = twin's, CSV sha here = twin's logged sha (Narval) = SHA256SUMS.txt: **45 of 45 PASS**. H5 in all 5 folders = SHA256SUMS.txt (`e4bff370e32476bf...`): **PASS**.

| fold | data folder | CSV sha256 | train / val / test / unused records (CSV = manifest) |
|---|---|---|---|
| 1 | `cv5_optB_leadV2_opt2p_fold01` | `d55c0ca9ce138b1c...` | 10 / 2 / 3 / 1 (same) |
| 2 | `cv5_optB_fold02` | `e7a79c5433ac50cf...` | 11 / 2 / 3 / 0 (same) |
| 3 | `cv5_optB_fold03` | `810004ab8f689298...` | 11 / 2 / 3 / 0 (same) |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | `9de60987697f6453...` | 10 / 2 / 3 / 1 (same) |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | `4b4c1a3001fe55f3...` | 10 / 2 / 3 / 1 (same) |

Split column = manifest in every fold: **PASS**.

## 5. The checks fail on injected faults

- text: lr changed in the job: flagged
- text: head epochs left at 20: flagged
- flags: weight_decay 0.05 -> 0.0: flagged
- flags: n_epochs_finetune 90 -> 70: flagged
- flags: backbone added: flagged
- data: fold-2 folder in a fold-1 job: flagged
- data: logged CSV sha altered: flagged

## 6. Time estimate

Twins (20 head-only + 70 full) took, in minutes: L1supE2E lead_II: median 13.9, max 17.5; L1supE2E lead_V2: median 12.2, max 14.8; L2supE2E two_lead: median 18.6, max 21.3. Total 11.2 GPU-hours. A head-only epoch has no backward pass through the encoder, so 90 full epochs cost more; upper bound 90/70 of the twin: about 14.4 GPU-hours, longest job about 27 min (walltime 45 min).

## Result: ALL PASS

`2026-11-24_step3e_dry_run_jobs.csv` next to this report is a byte copy of the generated jobs.csv, sha256 `2b8f9f26c0f79c5c5e7bc876820adde95406c953dfd80c074b60532f7d9c7cab`. Narval must reproduce this sha before submitting.
