# Decisions (final thesis results)
2026-11-19  Scope: ST-deviation estimator + PROSE alarms on LTST only. Out of scope: PatchTST, beat-centred windows (R3), TTA, arrhythmia models, local Holter patients, amplitude-aware model (D4).
2026-11-19  R-peaks: rodrigues2021 frozen. Used for NeuroKit delineation and the SQI gate in every main pipeline, cached per record, lead and signal (clean and each noisy draw). NK-PR-pub keeps NeuroKit peaks and the .atr gate (continuity with CMBEC).
2026-11-19  Backbones: re-pretrain single-lead II and V2 with the fixed loss (*_lossfix). L2 uses patch_ecg_unified_ii_v2_rebuilt (two-lead, unaffected). Pre-training has no seed.
2026-11-20  Splits: one identical split for every learned model. Folds 1, 4, 5 use cv5_optB_leadV2_opt2p (masked record s20081/s20081/s20241 unused for all models; moved record s20231/s20071/s20141 in validation). Folds 2, 3 published. Test sets unchanged. No validation record has a masked lead.
2026-11-20  Seeds: S1 = 200+fold-1 (published), S2 = 300+fold-1, S3 = 400+fold-1. L1 and L2: S1-S3. Supervised: S1 (S2, S3 optional). Noise: 42 for everything; 42-44 for the main pipelines' robustness check.
2026-11-20  NK-FP offsets: fit on all 13 non-test records; DMY beats on masked pairs excluded (confirm s20121 contributes none).
2026-11-20  Checks (lean): self-test on the cluster; backbone tool on L2's backbone and the new backbones (3 seeds); R1 re-score must match the published numbers. No reproduction runs. One dry-run approval; two reports (training done, scoring done).
2026-11-20  Before training: report gold-standard episodes per lead in each fold's validation set.
2026-11-20  Approved for Narval: steps 0, 0b, 1, step2_L2, step2_L1, step3 (S1). Not approved yet: step2b, step3o.
2026-11-20  Order: cluster submissions first; VM steps 4-5 (including NK-PR with rodrigues2021 on clean and seeds 42-44) in the background after that.
2026-11-20  Terminology: "ST-deviation episodes", not "ischemic episodes".
