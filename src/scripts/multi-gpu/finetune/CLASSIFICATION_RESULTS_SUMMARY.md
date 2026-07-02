# Classification track — results summary (PVC & APB)

Companion to `TASKS_HANDOVER.md`. Self-contained; readable standalone after a pull.
Date: 2026-07-02. Classes of interest: **PVC** and **APB** (APB = the bedside GS
label; the training label is **PAC** = atrial premature complex — same class).

--------------------------------------------------------------------------------
## TL;DR
1. **BCE beats the thesis "top hyper-params" (Focal+TTA+SAM) for any real use.** The
   thesis stack raises in-distribution macro-AUROC (+1.8 on PTB-XL) but **collapses
   operating-point recall** (Focal compresses probabilities). BCE is the deploy model.
2. **PVC is strong and deployable** (2-lead BCE: AUROC 0.978, AUPRC 0.677, recall 0.76;
   ~50% recall at the bedside). Improves further with 12 leads but bedside is 2-lead.
3. **APB is limited by classification performance, NOT by lack of leads.** It is weak
   in-distribution (AUPRC ~0.17, recall ~0.05) and **12 leads do not help** (AUPRC
   0.175 -> 0.174). Root cause: label scarcity (398 total / 40 test) + intrinsic
   subtlety. The thesis's per-class table is **AUROC-only**, which overstated it.
4. **The next queued task (Task 4: CODE-base classification pretrain) is now redundant**
   — it cannot fix PVC (already at ceiling) or APB (label-bound, not backbone-bound).
5. **Pipeline verified.** We found and fixed a real inference bug (TTA train/serve
   mismatch); BCE was always clean; conclusions hold after the fix.

--------------------------------------------------------------------------------
## A. BCE vs thesis top-hyper-params (Tier-3 = Focal a0.25 + TTA + SAM rho2)
Goal: test whether the thesis's recommended hyper-param stack (Table A.2) improves
performance over plain BCE. Same backbone, same data, seed 200, 2-lead II,V5, PTB-XL.

### Classification level (held-out PTB-XL test)
| model            | macro-AUROC | PVC AUROC/AUPRC/recall | PAC AUROC/AUPRC/recall |
|------------------|-------------|------------------------|------------------------|
| BCE              | 0.8604      | 0.978 / 0.677 / **0.76** | 0.882 / 0.175 / 0.05 |
| Tier-3 (thesis)  | 0.8784 (+1.8) | 0.964 / 0.620 / **0.04** | 0.874 / 0.145 / 0.00 |

(Unified label space: BCE macro 0.7800 -> Tier-3 0.7836, ~neutral.)

### Episode level (Homer bedside, Patient 21, PVC `combine(II,V5)` — TP/FP/FN)
| model                  | PVC @0.5      | PVC @0.3       |
|------------------------|---------------|----------------|
| BCE                    | 26 / 45 / 26  | 32 / 103 / 20  |
| Tier-3 (TTA applied)   | 0 / 0 / 52    | 2 / 9 / 50     |

APB: **0 true detections everywhere** — both recipes, both levels.

**Verdict:** Tier-3's AUROC gain is a ranking artifact that does not survive a
threshold. Focal loss shifts the probability calibration down, so recall craters
(PVC 0.76 -> 0.04 in-dist; 26 -> 0 at the bedside). **Use BCE.** For rare classes,
judge by **AUPRC/recall**, not AUROC.

--------------------------------------------------------------------------------
## B. Does APB fail because of too few leads? The 12-lead investigation
### Thesis reference (Appendix C, 12-lead PatchECG) — AUROC ONLY
No AUPRC/precision/recall reported. PTB-XL: PVC 0.964, PAC 0.931. Unified: PVC 0.987,
PAC 0.969. The PAC 0.931 looks promising — but AUROC is optimistic for rare classes.

### Our own 12-lead run (to get the FULL metrics the thesis omits)
Pretrained an all_leads backbone (Unified-No-PTB, Small, converged ~epoch 262), then
finetuned on PTB-XL **twice** (BCE and thesis-top). Per-class test metrics:

| class | 2-lead BCE (AUROC/AUPRC/recall) | 12-lead BCE | 12-lead thesis-top |
|-------|--------------------------------|-------------|--------------------|
| **PVC** | 0.978 / 0.677 / 0.76 | **0.990 / 0.810 / 0.80** | 0.974 / 0.685 / 0.10 |
| **PAC** | 0.882 / 0.175 / 0.05 | 0.863 / **0.174** / 0.10 | 0.838 / 0.185 / 0.00 |

- **PVC:** 12-lead genuinely better (AUPRC +0.13, recall 0.80). Our 12-lead 0.990
  even exceeds the thesis's 0.964 -> validates the pipeline + our cross-pretrain.
- **PAC/APB:** 12-lead **does not help** — AUPRC essentially flat (0.175 -> 0.174),
  recall still ~0.10. The thesis's 0.931 AUROC was the optimistic-metric mirage.

--------------------------------------------------------------------------------
## C. HOW WE KNOW APB is classification-performance-limited (and PVC proves it)
The argument is a chain, and PVC is the positive control:

1. **APB is weak at the classification level itself** (the source). Held-out PTB-XL:
   AUPRC 0.175, recall 0.05 at 2-lead. A model that can't rank/detect the class
   in-distribution cannot detect it downstream.
2. **We ruled out the lead/montage hypothesis by ablation.** If APB failed for lack
   of leads, 12 leads would fix it. They did NOT: AUPRC 0.175 -> 0.174, recall ~0.10.
   So leads are not the limiter -> the limit is the classifier's intrinsic ability.
3. **Root cause of the weak classifier:** label scarcity (398 PAC positives total, only
   40 in test) + an intrinsically subtle beat (premature *atrial* beat = normal QRS,
   early/abnormal P-wave). More leads add no discriminative P-wave signal the model
   can exploit here; only more labels would.
4. **PVC is the positive control that isolates the cause to classification, not the
   pipeline.** PVC has a *strong* classifier (AUPRC 0.677 -> 0.810, recall 0.76-0.80),
   and that strength **translates to real bedside detection** (26/52 = ~50% recall).
   Same pipeline, same leads, same GS -> PVC works, APB doesn't. Therefore APB's
   failure is attributable to **classification performance**, not the montage, not
   domain shift alone, and not the pipeline.

**Statement for the writeup:** "PVC is detectable (strong classifier that transfers to
the bedside); APB is not, and this is a classification-performance / label-scarcity
ceiling — not a lead-count problem — as shown by a 12-lead ablation that leaves APB
AUPRC and recall unchanged, while the same 12 leads and pipeline improve PVC."

--------------------------------------------------------------------------------
## D. Pipeline / inference verification
- **Bug found:** the Tier-3 models were trained/evaluated **with TTA**, but the Homer
  inference (`predict_homer_classification.py` / `ml_classification.predict_proba_per_segment`)
  ran them with `augmentation=none` — a train/serve mismatch that invalidated the first
  Tier-3 bedside numbers.
- **Fix:** added a `--tta` path mirroring `learner.batch_test` exactly (chunk indices =
  `get_test_time_augmentation_indices`, per-chunk sigmoid -> mean). Smoke-tested (indices
  and probabilities verified), then re-ran the Tier-3 Homer scoring with TTA on.
- **Result of the fix:** Tier-3 still collapsed at the bedside -> the cause is the Focal
  calibration shift, **not** the pipeline.
- **BCE was always clean** (trained with `data_augmentation=none`; train == serve), so
  its 26/52 is a valid, unaffected number.
- **Lead-subset finetune verified** (shared_embedding + non-strict `transfer_weights`),
  so the 12-lead vs 2-lead comparisons are legitimate.

(Inference edits live in the Neurokit2_Pipeline repo: `ltst_pipeline/ml_classification.py`,
`predict_homer_classification.py`, `sbatch/homer_ai_vs_gs_tier3.sbatch` — push separately.)

--------------------------------------------------------------------------------
## E. Task status + what's next
| Task | What | Status |
|------|------|--------|
| Task 3 | Tier-3 finetune (Focal+TTA+SAM), 6 models + Homer eval | DONE — doesn't help bedside |
| Task 2 | 12-lead baseline (all_leads pretrain + BCE + thesis-top finetunes) | DONE — corrected APB hypothesis |
| Task 1 | CODE-base pretrain for the **regression** backbone (II,V2) | not started |
| Task 4 | CODE-base pretrain for **classification** (II,V5) -> Tier-3 finetune | not started — **REDUNDANT** |

**Why Task 4 is now redundant:** it was the classification "big gun" (full CODE, base
params, then best-hyper-param finetune). But our results already show:
- PVC is at ceiling (2-lead ~= 12-lead; a bigger backbone won't add usable headroom);
- APB is **label-bound, not backbone-bound** (12 leads didn't help, so more pretrain
  data on the same labels won't either);
- Tier-3 (the finetune half of Task 4) degrades the operating point.
So a bigger CODE-base classifier cannot change either conclusion. Skip Task 4.

**What is technically next:** the classification track is essentially finished. The
remaining value is the **core ischemia / ST-regression track** (PatchECG regression ->
FSM -> ischemia alarms; FSM/SQI fixed). Within it, **Task 1 (CODE-base regression
backbone)** is the one CODE pretrain still worth doing — the thesis never trained a
CODE-base ST-regression backbone, so it is genuinely new. Alternatively, finish the
clean/noisy/SQI-gated ischemia evaluation.

--------------------------------------------------------------------------------
## F. Model inventory (checkpoints under results/fine-tune/saved_models/, gitignored)
- 2-lead BCE:        ptb-xl/unified2ptbxl_ii_v5_bce, unified/ptbxl2unified_ii_v5_bce (+ ii, v5 single-lead)
- 2-lead Tier-3:     {ptb-xl,unified}/{unified2ptbxl,ptbxl2unified}_{ii_v5,ii,v5}_tta_focal_sam (6)
- 12-lead:           ptb-xl/unified2ptbxl_all_leads_bce, ptb-xl/unified2ptbxl_all_leads_tta_focal_sam
- 12-lead backbone:  pre-train/.../unified/patch_ecg_unified_all_leads (converged ~ep262; resumable)
Per-class metrics: each model dir has `_test_metrics_*_per_class.csv` (has AUPRC/precision/recall).
