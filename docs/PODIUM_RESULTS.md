# Podium results (CMBEC48 talk): paper configuration, one run per system

Generated 2026-11-24 21:50:28 by `src/scripts/final/podium_results.py` (VM). **Podium-only exception** (`docs/DECISIONS.md` 2026-11-24): these numbers are for the conference talk and are not thesis results; no thesis output was changed and nothing was retrained.

**Configuration (the paper's):** noise seed 42 (NSTDB `em`); one run per system. **Fiducial:** NeuroKit R-peaks and NeuroKit dwt delineation, ST level = 5-sample median (±8 ms) at J+60 minus the PR-segment median, per-beat values rounded to 1e-6 mV (removes the float residue at the PROSE thresholds that the paper's stored segment files carry), 10 s windows with ≥ 5 usable beats. **Learned-1L:** pretrained, one model per lead, training seed S1, alarms of both leads merged. **Gate:** the paper's SQI (0.67 Hz high-pass, 30 s, 25th percentile of per-beat SNR) from `.atr` beats, 6 dB, every gated row. Same PROSE FSM and scorer as the thesis; union of II and V2, all 101 episodes (Monitor: 99, copied from the paper); excluded record–lead pairs left out.

Controls: all passed — Learned-1L gated = A4a S1 .atr row: (76, 25, 25) vs (76, 25, 25); Fiducial artifact = 67/43/34: (67, 43, 34) vs (67, 43, 34); Fiducial gated = 64/27/37: (64, 27, 37) vs (64, 27, 37); definition check: published R1 (paper's learned arm), .atr gate: 74/28/27, coverage (eq. 4) 1.213 (paper 74/28/27, 1.213); merged across leads: alarms only 1.059, alarms and GS 1.112 (paper ~1.06).

## (a) Detection of ischemic episodes (paper Table 1 layout)

P = TP/(TP+FP); R = TP/(TP+FN). Redundant = later alarms inside an already-detected episode (not counted as FP). Paper column: the published TP/FP/FN for comparison.

| System | Condition | TP | FP | FN | Redundant | P | R | F1 | paper TP/FP/FN |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| Monitor [1] | baseline | 78 | 66 | 21 | n/a | 0.542 | 0.788 | 0.642 | 78/66/21 |
|  | artifact | 77 | 171 | 22 | n/a | 0.310 | 0.778 | 0.444 | 77/171/22 |
|  | gated | 76 | 54 | 23 | n/a | 0.585 | 0.768 | 0.664 | 76/54/23 |
| Fiducial | baseline | 67 | 34 | 34 | 13 | 0.663 | 0.663 | 0.663 | 66/32/35 |
|  | artifact | 67 | 43 | 34 | 15 | 0.609 | 0.663 | 0.635 | 66/40/35 |
|  | gated | 64 | 27 | 37 | 7 | 0.703 | 0.634 | 0.667 | 64/24/37 |
| Learned-1L (S1) | baseline | 75 | 40 | 26 | 19 | 0.652 | 0.743 | 0.694 | 75/40/26 |
|  | artifact | 74 | 49 | 27 | 57 | 0.602 | 0.733 | 0.661 | 73/57/28 |
|  | gated | 76 | 25 | 25 | 11 | 0.752 | 0.752 | 0.752 | 74/28/27 |

## (b) Temporal extent of the alarms (paper Table 2 layout)

Sens. = duration sensitivity TPt/(TPt+FNt); PPV = TPt/(TPt+FPt); IoU = TPt/(TPt+FPt+FNt); Cov. = coverage, paper eq. 4, (TPt+FPt)/(TPt+FNt), per record–lead. Cov. merged = each record's II and V2 alarms merged into one timeline before measuring alarm time, over the same gold-standard time (the paper's "merging alarms across leads"). Paper values in brackets.

| System | Condition | Sens. | PPV | IoU | Cov. | Cov. merged |
|---|---|--:|--:|--:|--:|--:|
| Fiducial | artifact | 0.456 (0.455) | 0.725 (0.727) | 0.389 (0.389) | 0.628 (0.627) | 0.625 |
|  | gated | 0.498 (0.496) | 0.715 (0.721) | 0.415 (0.416) | 0.696 (0.688) | 0.693 |
| Learned-1L (S1) | artifact | 0.613 (0.644) | 0.693 (0.674) | 0.482 (0.491) | 0.884 (0.955) | 0.829 |
|  | gated | 0.680 (0.749) | 0.623 (0.618) | 0.482 (0.512) | 1.091 (1.213) | 1.019 |

Definition check for Cov. merged: the paper's own learned arm (published models, `.atr` gate) gives Cov. 1.213 (paper 1.213) and Cov. merged 1.059 (paper: "reduces the value to 1.06"). Merging the gold-standard time across leads as well would give 1.112, so the paper's 1.06 is the alarms-only merge used here.

## (c) Gated TP per lead

Each lead's alarms scored against that lead's episodes (69 on II, 32 on V2).

| System | II TP | V2 TP | paper II / V2 |
|---|--:|--:|---|
| Fiducial | 59 of 69 | 5 of 32 | – / 5 of 32 |
| Learned-1L (S1) | 56 of 69 | 20 of 32 | – / 22 of 32 |

## (d) Fiducial: baseline windows with no value

10 s windows of the NeuroKit grid in which fewer than 5 beats have a usable ST deviation (the estimator supplies no value and the FSM freezes). Paper: 20.8 % (II) / 12.2 % (V2).

| Denominator | II: no value / windows | V2: no value / windows |
|---|--:|--:|
| 15 records, excluded pairs left out (thesis convention) | 22,305 / 111,142 (20.1 %) | 10,971 / 94,816 (11.6 %) |
| 15 records, all record-lead pairs | 24,850 / 119,245 (20.8 %) | 14,571 / 119,245 (12.2 %) |

Differences from the paper's numbers:
- Fiducial: the paper's rows were scored from segment files that store values such as −0.0999999999999999 mV where the true median is −0.1, so the FSM's threshold tests failed on exact ties (about one episode per condition). The paper's own segment files, rounded to 1e-6 mV, give 67/32/34 (3-sample baseline), 67/43/34 and 64/27/37; the baseline row here uses 5 samples (67/34/34).
- Learned-1L: S1 of the final (retrained) models, not the published models; the gate is the same `.atr` gate. The baseline counts equal the paper's (75/40/26) by coincidence: the predictions differ (mean |Δ| 0.015 mV II, 0.022 mV V2) and so do redundant alarms and IoU.

<!-- published_folds_nk -->
## Published folds, NeuroKit gate (final podium configuration)

Generated 2026-11-25 00:26:01 by `src/scripts/final/podium_published_folds.py` (VM). **Final podium configuration** (`docs/DECISIONS.md` 2026-11-24; decided before any result: this is what the podium talk and the corrected CMBEC paper use; thesis outputs are not affected).

**Learned-1L (S1):** pretrained (`_lossfix` backbones), training seed S1, one model per lead, on the **published folds** `cv5_optB_fold01–05` with their published validation sets: POD_L1_lead_{II,V2}_S1_fold01/04/05 (Narval, `docs/reports/2026-11-25_podium_training.md`) and FIN_L1_lead_{II,V2}_S1_fold02/03 (already on the published folders). Everything else as the thesis runs. **Fiducial:** NeuroKit R-peaks and dwt delineation, 5-sample ST level at J+60 minus the PR-segment median, values rounded to 1e-6 mV. **Gate:** the paper's SQI from **NeuroKit R-peaks**, 6 dB, every gated row. Noise: NSTDB `em`, seed 42. Same PROSE FSM and scorer; union of II and V2, 101 episodes (Monitor: 99, copied from the paper); excluded record–lead pairs left out. The published V2 validation sets of folds 1, 4 and 5 hold no V2 gold-standard episode (a property of the published folds).

Controls: Fiducial baseline = RESULTS_NK Phase A: (67, 34, 34) vs (67, 34, 34); Fiducial artifact = RESULTS_NK Phase A: (67, 43, 34) vs (67, 43, 34); Fiducial gated = RESULTS_NK Phase A: (65, 30, 36) vs (65, 30, 36). Inference checks (`Tasks/final/runs/podium/published_folds/CHECKS_infer.txt`): each POD run reproduces its saved test MSE; folds 2–3 predictions equal the FIN S1 predictions; window counts equal the published runs; checkpoints match the copy-back list.

### (a) Detection of ischemic episodes (paper Table 1 layout)

| System | Condition | TP | FP | FN | Redundant | P | R | F1 | paper TP/FP/FN (P / R / F1) |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| Monitor [1] | baseline | 78 | 66 | 21 | n/a | 0.542 | 0.788 | 0.642 | same |
|  | artifact | 77 | 171 | 22 | n/a | 0.310 | 0.778 | 0.444 | same |
|  | gated | 76 | 54 | 23 | n/a | 0.585 | 0.768 | 0.664 | same |
| Fiducial | baseline | 67 | 34 | 34 | 13 | 0.663 | 0.663 | 0.663 | 66/32/35 (0.673 / 0.653 / 0.663) |
|  | artifact | 67 | 43 | 34 | 15 | 0.609 | 0.663 | 0.635 | 66/40/35 (0.623 / 0.653 / 0.638) |
|  | gated | 65 | 30 | 36 | 6 | 0.684 | 0.644 | 0.663 | 64/24/37 (0.727 / 0.634 / 0.677) |
| Learned-1L (S1) | baseline | 81 | 46 | 20 | 15 | 0.638 | 0.802 | 0.711 | 75/40/26 (0.652 / 0.743 / 0.694) |
|  | artifact | 79 | 56 | 22 | 59 | 0.585 | 0.782 | 0.669 | 73/57/28 (0.562 / 0.723 / 0.632) |
|  | gated | 77 | 34 | 24 | 10 | 0.694 | 0.762 | 0.726 | 74/28/27 (0.726 / 0.733 / 0.729) |

### (b) Temporal extent of the alarms (paper Table 2 layout)

Sens. = TPt/(TPt+FNt); PPV = TPt/(TPt+FPt); IoU = TPt/(TPt+FPt+FNt); Cov. = (TPt+FPt)/(TPt+FNt) (paper eq. 4), per record–lead; Cov. merged = each record's II and V2 alarms merged into one timeline (the paper's 1.06 definition, checked in the section above). Paper values in brackets.

| System | Condition | Sens. | PPV | IoU | Cov. | Cov. merged |
|---|---|--:|--:|--:|--:|--:|
| Fiducial | artifact | 0.456 (0.455) | 0.725 (0.727) | 0.389 (0.389) | 0.628 (0.627) | 0.625 |
|  | gated | 0.510 (0.496) | 0.739 (0.721) | 0.432 (0.416) | 0.691 (0.688) | 0.687 |
| Learned-1L (S1) | artifact | 0.761 (0.644) | 0.682 (0.674) | 0.562 (0.491) | 1.116 (0.955) | 1.007 |
|  | gated | 0.734 (0.749) | 0.632 (0.618) | 0.514 (0.512) | 1.162 (1.213) | 1.034 (1.06) |

### (c) Gated TP per lead (NeuroKit gate)

Each lead's alarms against that lead's episodes (69 on II, 32 on V2).

| System | II TP | V2 TP | paper V2 |
|---|--:|--:|---|
| Fiducial | 60 of 69 | 5 of 32 | 5 of 32 |
| Learned-1L (S1) | 51 of 69 | 26 of 32 | 22 of 32 |

### (d) Learned-1L (S1) training: best epoch and best validation loss per fold

Epochs 0–89 (0–19 head only, 20–89 full fine-tuning); validation loss = MSE (mV²) on the published validation set. Folds 2 and 3: the reused FIN runs.

| Fold | II run | II best epoch | II best valid loss | V2 run | V2 best epoch | V2 best valid loss |
|--:|---|--:|--:|---|--:|--:|
| 1 | POD_L1_lead_II_S1_fold01 | 33 | 0.000760 | POD_L1_lead_V2_S1_fold01 | 19 | 0.000992 |
| 2 | FIN_L1_lead_II_S1_fold02 | 31 | 0.000231 | FIN_L1_lead_V2_S1_fold02 | 73 | 0.001451 |
| 3 | FIN_L1_lead_II_S1_fold03 | 50 | 0.001273 | FIN_L1_lead_V2_S1_fold03 | 51 | 0.002584 |
| 4 | POD_L1_lead_II_S1_fold04 | 29 | 0.000319 | POD_L1_lead_V2_S1_fold04 | 24 | 0.001008 |
| 5 | POD_L1_lead_II_S1_fold05 | 31 | 0.001537 | POD_L1_lead_V2_S1_fold05 | 30 | 0.001545 |

V2 fold 1 selects epoch 19, the last head-only epoch (as every FIN V2 fold-1 run did); the selection rule is unchanged.
<!-- /published_folds_nk -->

<!-- published_folds_atr -->
## Published folds, .atr gate (sensitivity check)

Generated 2026-11-25 00:45:21 by `src/scripts/final/podium_published_folds.py score_atr` (VM). Same as the final section above (Learned-1L S1 on the published folds; Fiducial on NeuroKit delineation, 5 samples, rounded; noise seed 42), except the SQI gate: `.atr` beats at 6 dB instead of NeuroKit R-peaks. Gated rows only.

Controls: Fiducial gated = the existing .atr section: (64, 27, 37) vs (64, 27, 37); Learned-1L baseline = the final section: (81, 46, 20) vs (81, 46, 20); Learned-1L artifact = the final section: (79, 56, 22) vs (79, 56, 22).

| System | TP | FP | FN | Redundant | P | R | F1 | Sens. | PPV | IoU | Cov. | Cov. merged | II TP | V2 TP |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Fiducial | 64 | 27 | 37 | 7 | 0.703 | 0.634 | 0.667 | 0.498 | 0.715 | 0.415 | 0.696 | 0.693 | 59 of 69 | 5 of 32 |
| Learned-1L (S1) | 76 | 25 | 25 | 9 | 0.752 | 0.752 | 0.752 | 0.750 | 0.610 | 0.507 | 1.229 | 1.093 | 50 of 69 | 26 of 32 |
<!-- /published_folds_atr -->
