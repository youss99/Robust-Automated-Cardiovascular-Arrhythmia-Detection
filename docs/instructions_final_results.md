# Final thesis results: instructions for Claude Code

Where this file and DECISIONS.md differ, DECISIONS.md wins.

20 November 2026 (lean version). **Goal:** one frozen set of results for the thesis, collected in `RESULTS_FINAL.md`. **Scope:** the ST-deviation estimator and PROSE alarms on LTST only.

## Ground rules

- **Ask before guessing.** If anything here is ambiguous or conflicts with the code, configs or files, stop and ask.
- **Never overwrite** published data, checkpoints, predictions or results. Each run gets its own folder, named by its ID.
- **One dry run for the whole batch.** Run the loader on one fold, list every job, and estimate the time. I approve once; then submit everything.
- **Log every job:** resolved config, git commit, seed (pre-training: "none", since each backbone is a single run that can't be repeated), backbone path with checksum, best epoch and validation loss.
- **Report twice:** when training finishes, and when scoring is done. Flag a diverged run or a failed check straight away.

## 1. Frozen setup (every main pipeline)

| Item | Setting |
|---|---|
| Records, folds, masked pairs | As in `README_phase1_runs.md` (Shared setup): `cv5_optB`, published test sets, `MANUAL_OMIT` handled as published |
| Splits | One identical split for every learned model (lead II, V2, two-lead; pretrained and supervised): the V2 option 2+ split (`cv5_optB_leadV2_opt2p_fold*`) in folds 1, 4, 5, published `cv5_optB_fold02/03` in folds 2, 3. In folds 1, 4, 5 the masked record (s20081, s20081, s20241) is unused for every model and the moved record (s20231, s20071, s20141) is in validation. No validation record has a masked lead. Test sets unchanged. NeuroKit is unaffected (it fits on all non-test records) |
| Windows and labels | Published grid: 10 s non-overlapping windows, 250 → 100 Hz, label = median `.16a` `ee` |
| Gold standard | Published per-beat gold standard, 101 episodes. Call them "ST-deviation episodes" |
| Noise | NSTDB `em`, published protocol. Only the seed changes |
| R-peaks | rodrigues2021, detected per record, lead and signal (clean, and each noisy draw). Compute once into a cache; every pipeline reads the cache |
| SQI gate | Published SQI, computed on the cached rodrigues2021 peaks, threshold 6 dB |
| Alarm rules | Published PROSE FSM (criteria, 30 s cooldown). Gated and missing windows both freeze the FSM |
| Code | Fixed pre-training code. `check_training_paths.py` must pass 37/37 on every machine that trains |

## 2. Pipelines

| ID | Role | ST estimate | Training / backbone | Delineation peaks | Gate peaks |
|---|---|---|---|---|---|
| MON | Main (reference) | Commercial monitor | – | – | As published |
| NK-PR | Main | NeuroKit paper rule: PR-median baseline, NeuroKit J point, windowed as M1 | – | rodrigues2021 | rodrigues2021 |
| NK-FP | Main | NeuroKit fixed point: baseline at R − *a*, ST at R + *b* + 60 ms. *a*, *b* refit per fold and lead on training DMY beats, with rodrigues2021 peaks | – | rodrigues2021 | rodrigues2021 |
| L1 | Main | Single-lead pair: lead II model + V2 model, alarms combined as a union | Pretrained, new single-lead backbones (step 1) | – | rodrigues2021 |
| L2 | Main | Two-lead model | Pretrained, `patch_ecg_unified_ii_v2_rebuilt` (R2's two-lead backbone, from its `params.txt`; not affected by the bug) | – | rodrigues2021 |
| L1-sup, L2-sup | Ablation A1 | As L1, L2 | Supervised, no backbone | – | rodrigues2021 |
| L1-old | Ablation A2 (published reference) | CMBEC model (R1), existing predictions. Differs from L1 in backbone and folds | `_only` backbones: old data, old loss | – | rodrigues2021 |
| L1-rebuilt | Ablation A2 (optional) | Single-lead pair, re-fine-tuned under the new manifest (step 2b). Differs from L1 in the pre-training loss (see note) | `_rebuilt` backbones: fixed data, old loss | – | rodrigues2021 |
| NK-FP-nk | Ablation A3 | As NK-FP | – | NeuroKit's own peaks | rodrigues2021 |
| NK-PR-pub | Ablation A3 | NK-PR exactly as published (M1) | – | NeuroKit's own peaks | `.atr` |

- **MON:** copy the published numbers (TP 76, FP 54, FN 23; 99 episodes). Don't recompute.
- **L1-rebuilt vs L1 (only if step 2b runs):** a one-change comparison only if `params.txt` confirms the `_rebuilt` backbones used the same data, recipe and epochs as step 1. Even then, it also contains pre-training run-to-run noise (one unseeded backbone per arm); state this in the thesis.
- **A4 (gate ablation),** for L1, L2 and NK-FP: (a) `.atr` peaks for the gate instead of rodrigues2021; (b) the SQI threshold sweep, −20 to +25 dB.
- **Out of scope:** PatchTST, beat-centred windows (R3), test-time augmentation, the arrhythmia models, the local Holter patients and the amplitude-aware model (D4). Don't run any of these unless I ask.

## 3. Steps

**Order:** 0 → 0b → 1 → 2. Steps 3, 4 and 5 (and 2b, if run) start right after 0. Step 6 waits for 2 and 3. Step 7 comes last.

| Step | Where | What | Jobs | Check before moving on |
|---|---|---|---|---|
| 0 | Cluster | Commit check (HEAD = pushed commit; no modified tracked files) + self-test (`check_training_paths.py`) | – | 37/37 |
| 0b | Cluster | Run `backbone_masked_vs_visible.py` on L2's backbone only (`patch_ecg_unified_ii_v2_rebuilt`), with 2–3 `--seed` values | – | The pass rule in `docs/DECISIONS.md` (2026-11-20, step 0). If it fails, stop and report |
| 1 | Cluster | Re-pretrain single-lead backbones for II and V2: `pretrain_unified_v2.sh`, same recipe, new names (e.g. `patch_ecg_unified_{ii,v2}_lossfix`) | 2 | Logged loss starts near 1 and falls; don't compare it with old single-lead logs, which show the mixed loss. `backbone_masked_vs_visible.py` (2–3 seeds) reads "does not reproduce visible patches" |
| 2 | Cluster | Fine-tune L1 (II, V2) from the step 1 backbones, and L2. Same recipe as R2 | 3 models × 5 folds × 3 seeds = 45 | No resumed snapshots. Report any diverged run. No reproduction run needed: the fix doesn't touch fine-tuning (verified on the VM) |
| 2b | Cluster | **Optional, only if the cluster has spare capacity:** re-fine-tune L1-rebuilt (II, V2) from the `_rebuilt` backbones under the new manifest, seed S1 | 10 | As step 2 |
| 3 | Cluster | Train L1-sup and L2-sup, seeds S1–S3 | 45 | Report any diverged run (R1b had 3 diverged supervised lead II runs) |
| 4 | VM | R-peak cache: rodrigues2021 on clean signal and on noisy seeds 42, 43, 44 | – | Spot-check against `.atr`. Reuse existing seed 42/43 outputs only if the setup is identical |
| 5 | VM | NK-PR and NK-FP: clean + seeds 42–44. NK-FP-nk and NK-PR-pub: seed 42 | – | Report the fitted *a*, *b* per fold and lead. Time one seed first |
| 6 | VM | Inference: out-of-fold predictions, clean and noisy | – | Window counts match the published runs |
| 7 | VM | Scoring, then `RESULTS_FINAL.md` | – | First re-score R1 with its published setup: it must match `PAPER_RESULTS_TWO_MODELS.md` exactly |


## 4. Seeds

| Pipelines | Training seeds | Noise seeds |
|---|---|---|
| L1, L2 | S1 (the published seed), S2, S3 | 42 for all three; 42–44 for S1 (robustness) |
| L1-sup, L2-sup | S1–S3 | 42 |
| NK-PR, NK-FP | – (deterministic) | 42–44 |
| All other ablations | Published seed only | 42 |

- **Each training seed is a full pipeline:** 5 fold models, one out-of-fold trace per record, then alarms.
- **Never choose a seed by its test results.** Single-model figures use S1.
- **Record S2 and S3.** If S1 isn't recorded, ask.

## 5. What goes in `RESULTS_FINAL.md`

| # | Table or figure | Pipelines | Condition | Content |
|---|---|---|---|---|
| T1 | Regression, all windows | L1, L2, NK-PR, NK-FP | Clean | Per lead, masked pairs excluded: MAE, bias, RMSE, calibration slope, r, p100; R² as a secondary column. Coverage (% windows with a value) for NK |
| T2 | Regression, DMY beats | Same | Clean | Same metrics, matched to DMY beats as in C §9 |
| T3 | Error by true ST band | Same | Clean | MAE and bias in < 0.05, 0.05–0.10, 0.10–0.20 and > 0.20 mV |
| T4 | Noise robustness | Same | Noisy, ungated and gated | MAE overall and inside noise bursts |
| Fig. 1 | Bland–Altman | L1, L2 (S1), NK-FP | Clean, DMY beats | Per lead: bias, 95% limits of agreement, slope of difference against mean |
| T5 | Alarms, main table | MON, NK-PR, NK-FP, L1, L2 | Clean / noisy / gated | TP, FP, FN, precision, recall, F1, IoU, false-alarm hours, duration Se and +P, redundant alarms (n/a for MON). Learned rows: mean ± SD over the training seeds |
| T6 | Alarm robustness | NK-PR, NK-FP, L1 (S1), L2 (S1) | Gated, seeds 42–44 | Mean ± SD of TP, FP, FN, F1 and false-alarm hours |
| T7 | Paired differences | L2 − L1, L2 − NK-FP, L1 − NK-FP, L2 − NK-PR; pretrained − supervised (L1, L2) | Gated, seed 42 | Record-level paired bootstrap, 5,000 resamples, training seeds averaged inside each resample. 95% CI for ΔF1, ΔFP and ΔMAE |
| T8 | Alarm transitions | L1, L2 (S1), NK-FP | Clean → noisy → gated | Persists / noise-created / removed by gate / gate-created, separately for TP and FP |
| T9 | Amplitude check | L1, L2 (S1) | Clean | Change in prediction when the input is scaled ×2. Expected to be near zero by design (PatchECG normalises every patch): report it as a confirmed design property that explains the compression, not as a new finding |
| T10 | Ablations A1–A4 | As in §2 | Seed 42 (A1: the training seeds that were run) | T1 and T5 metrics, each ablation row next to the row it's compared with |

**Also save:**
- per-record CSVs (TP, FP, FN and MAE per record × lead × pipeline × seed), so every table can be recomputed
- `provenance.csv`: run ID → config, git commit, checkpoint, checksum, seed
- every figure with the script that made it
