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
