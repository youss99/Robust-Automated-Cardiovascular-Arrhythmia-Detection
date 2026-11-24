# Final thesis results

Frozen setup: `docs/instructions_final_results.md` and `docs/DECISIONS.md`. Provenance: `Tasks/final/runs/step6/provenance.csv` (VM). Run log: `docs/reports/2026-11-21_vm_steps4_7.md`.

**Naming (DECISIONS.md 2026-11-24):** thesis names throughout: Monitor; Fiducial (paper rule on rodrigues2021 beats, 5-sample ST level); Learned-1L (pretrained) / Learned-2L (pretrained); Learned-1L (supervised-only) / Learned-2L (supervised-only) = the end-to-end runs (0 + 90 epochs); Learned-1L (published); baseline / artifact / gated; ischemic episodes; excluded record–lead pairs. The fixed-point fiducial variants and ablation A3 are not reported (archived).

<!-- tier1 -->
## Tier 1 (T1, T5, T7)

Generated 2026-11-24 04:16:08 by `src/scripts/final/fin_step7_tier1.py` (VM). 15 records, 101 ischemic episodes (Monitor: 99, as published). Learned rows: mean ± SD (ddof 1) over training seeds S1–S3; per-seed values in `Tasks/final/runs/step7/`. Noise: NSTDB `em`, seed 42. Gate: the paper's SQI from rodrigues2021 peaks on the artifact signal, 6 dB.

Control (published R1 predictions through this alarm path, `.atr` gate): all 6 conditions reproduce the paper.

### T1. Regression, all windows (baseline)

Set A: the 10 s grid windows, label = median `.16a` `ee` of the window's beats; excluded pairs not scored; each pipeline on its own windows with a value (coverage = share of labelled windows with a value). Slope = OLS of prediction on label.

| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| II | Learned-1L (pretrained) | 100.0 ± 0.0 | 110783 ± 0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| II | Learned-2L (pretrained) | 100.0 ± 0.0 | 110783 ± 0 | 0.0290 ± 0.0018 | 0.0020 ± 0.0023 | 0.0444 ± 0.0016 | 0.661 ± 0.011 | 0.931 ± 0.008 | 5.1 ± 0.3 | 0.817 ± 0.013 |
| II | Fiducial | 80.7 | 89372 | 0.0346 | -0.0309 | 0.0638 | 0.935 | 0.851 | 7.5 | 0.561 |
| V2 | Learned-1L (pretrained) | 100.0 ± 0.0 | 94475 ± 0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| V2 | Learned-2L (pretrained) | 100.0 ± 0.0 | 94475 ± 0 | 0.0290 ± 0.0011 | 0.0020 ± 0.0007 | 0.0368 ± 0.0011 | 0.389 ± 0.037 | 0.659 ± 0.027 | 0.7 ± 0.3 | 0.427 ± 0.036 |
| V2 | Fiducial | 75.0 | 70883 | 0.0441 | -0.0415 | 0.0587 | 0.670 | 0.663 | 8.7 | -0.358 |

### T5. Alarms, main table

Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record–lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN). Duration Se / +P = time sensitivity / time PPV; redundant = Ignored (extra alarms inside an already-detected episode). Monitor copied from the paper (episode counts only).

| condition | pipeline | TP | FP | FN | precision | recall | F1 | IoU | duration Se | duration +P | redundant |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | Monitor | 78 | 66 | 21 | 0.542 | 0.788 | 0.642 | – | – | – | n/a |
| baseline | Fiducial | 72.0 | 26.0 | 29.0 | 0.735 | 0.713 | 0.724 | 0.445 | 0.523 | 0.749 | 12.0 |
| baseline | Learned-1L (pretrained) | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 0.738 ± 0.022 | 0.687 ± 0.028 | 18.0 ± 2.6 |
| baseline | Learned-2L (pretrained) | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 0.694 ± 0.034 | 0.720 ± 0.051 | 20.3 ± 0.6 |
| artifact | Monitor | 77 | 171 | 22 | 0.310 | 0.778 | 0.444 | – | – | – | n/a |
| artifact | Fiducial | 67.0 | 51.0 | 34.0 | 0.568 | 0.663 | 0.612 | 0.392 | 0.462 | 0.719 | 13.0 |
| artifact | Learned-1L (pretrained) | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 0.614 ± 0.021 | 0.709 ± 0.019 | 72.7 ± 13.8 |
| artifact | Learned-2L (pretrained) | 73.3 ± 2.1 | 66.7 ± 18.1 | 27.7 ± 2.1 | 0.530 ± 0.061 | 0.726 ± 0.021 | 0.610 ± 0.034 | 0.402 ± 0.027 | 0.454 ± 0.022 | 0.780 ± 0.074 | 109.7 ± 1.5 |
| gated | Monitor | 76 | 54 | 23 | 0.585 | 0.768 | 0.664 | – | – | – | n/a |
| gated | Fiducial | 66.0 | 21.0 | 35.0 | 0.759 | 0.653 | 0.702 | 0.452 | 0.526 | 0.763 | 7.0 |
| gated | Learned-1L (pretrained) | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 0.724 ± 0.020 | 0.660 ± 0.023 | 11.7 ± 2.5 |
| gated | Learned-2L (pretrained) | 75.7 ± 1.2 | 20.0 ± 3.5 | 25.3 ± 1.2 | 0.791 ± 0.032 | 0.749 ± 0.011 | 0.770 ± 0.021 | 0.503 ± 0.019 | 0.670 ± 0.016 | 0.668 ± 0.022 | 14.7 ± 1.2 |

### T7. Paired differences (gated, seed 42)

Record-clustered paired bootstrap, 5,000 resamples (seed 20260805); training seeds averaged inside each resample. dMAE: set A windows the gate keeps (gate at the learned anchor, same for both), artifact predictions against the baseline labels, excluded pairs left out, each pipeline on its own windows. * = 95 % CI excludes 0.

| comparison | metric | A | B | A − B | 95 % CI |
|---|---|--:|--:|--:|---|
| Learned-2L (pretrained) - Learned-1L (pretrained) | dF1 (union) | 0.770 | 0.752 | +0.018 | [-0.039, +0.078] |
| Learned-2L (pretrained) - Learned-1L (pretrained) | dFP (union) | 20.0 | 24.0 | -4.0 | [-14.3, +7.3] |
| Learned-2L (pretrained) - Learned-1L (pretrained) | dMAE II (mV) | 0.0292 | 0.0288 | +0.0004 | [-0.0027, +0.0038] |
| Learned-2L (pretrained) - Learned-1L (pretrained) | dMAE V2 (mV) | 0.0303 | 0.0359 | -0.0056 | [-0.0118, +0.0007] |
| Learned-2L (pretrained) - Fiducial | dF1 (union) | 0.770 | 0.702 | +0.068 | [-0.085, +0.204] |
| Learned-2L (pretrained) - Fiducial | dFP (union) | 20.0 | 21.0 | -1.0 | [-19.3, +25.0] |
| Learned-2L (pretrained) - Fiducial | dMAE II (mV) | 0.0292 | 0.0341 | -0.0049 | [-0.0318, +0.0182] |
| Learned-2L (pretrained) - Fiducial | dMAE V2 (mV) | 0.0303 | 0.0439 | -0.0137 | [-0.0276, +0.0004] |
| Learned-1L (pretrained) - Learned-1L (supervised-only) | dF1 (union) | 0.752 | 0.685 | +0.067 * | [+0.018, +0.130] |
| Learned-1L (pretrained) - Learned-1L (supervised-only) | dFP (union) | 24.0 | 21.7 | +2.3 | [-2.3, +7.0] |
| Learned-1L (pretrained) - Learned-1L (supervised-only) | dMAE II (mV) | 0.0288 | 0.0367 | -0.0079 * | [-0.0125, -0.0040] |
| Learned-1L (pretrained) - Learned-1L (supervised-only) | dMAE V2 (mV) | 0.0359 | 0.0338 | +0.0021 | [-0.0025, +0.0085] |
| Learned-2L (pretrained) - Learned-2L (supervised-only) | dF1 (union) | 0.770 | 0.698 | +0.072 * | [+0.001, +0.148] |
| Learned-2L (pretrained) - Learned-2L (supervised-only) | dFP (union) | 20.0 | 17.0 | +3.0 | [-4.7, +11.7] |
| Learned-2L (pretrained) - Learned-2L (supervised-only) | dMAE II (mV) | 0.0292 | 0.0366 | -0.0074 * | [-0.0144, -0.0024] |
| Learned-2L (pretrained) - Learned-2L (supervised-only) | dMAE V2 (mV) | 0.0303 | 0.0357 | -0.0054 * | [-0.0100, -0.0007] |
<!-- /tier1 -->

<!-- t5_temporal -->
### T5 note: temporal metrics, Learned-1L (pretrained) vs Learned-2L (pretrained) (written by hand, 2026-11-22)

From the T5 rows above (mean ± SD over S1–S3). Temporal metrics: IoU, duration Se, duration +P.

| condition | IoU Learned-1L (pretrained) / Learned-2L (pretrained) | duration Se Learned-1L (pretrained) / Learned-2L (pretrained) | duration +P Learned-1L (pretrained) / Learned-2L (pretrained) |
|---|---|---|---|
| baseline | 0.552 ± 0.030 / 0.546 ± 0.033 | 0.738 ± 0.022 / 0.694 ± 0.034 | 0.687 ± 0.028 / 0.720 ± 0.051 |
| artifact | 0.491 ± 0.021 / 0.402 ± 0.027 | 0.614 ± 0.021 / 0.454 ± 0.022 | 0.709 ± 0.019 / 0.780 ± 0.074 |
| gated | 0.528 ± 0.025 / 0.503 ± 0.019 | 0.724 ± 0.020 / 0.670 ± 0.016 | 0.660 ± 0.023 / 0.668 ± 0.022 |

- Learned-1L (pretrained) covers more of each true episode in time: higher duration Se in every condition, and higher IoU (tie on baseline).
- Learned-2L (pretrained)'s alarms are more conservative: higher duration +P (about equal when gated).
- No significance test was run on any temporal metric (T7 covers F1, FP and MAE only). Gated, the IoU gap (0.025) is about one seed SD,
  so Learned-1L (pretrained)'s temporal advantage is not established there; the artifact-condition gaps are several SDs. On episode F1, Learned-2L (pretrained) is ahead (gated 0.770 vs 0.752,
  95 % CI of the difference includes 0, T7).
<!-- /t5_temporal -->

<!-- t10 -->
## T10. Ablations A1, A2 and A4 (A3 not reported, DECISIONS.md 2026-11-24)

Generated 2026-11-24 04:18:47 by `src/scripts/final/fin_step7_t10_fig1.py` (VM). Noise seed 42. Same alarm path, scorer, labels and gate tables as tier 1; rows marked S1–S3 are mean ± SD over the training seeds (A1, and Learned-1L (pretrained) in A2), others one run. Each ablation row sits under the row it is compared with. L1-rebuilt (A2, optional) was not run: step 2b was not submitted.

Controls: published M1 and M2-R through this alarm path reproduce phase 1's stored counts (5 / 5); the 6 dB point of the sweep equals tier 1 (3 / 3).

### T10a. Regression (T1 metrics, baseline)

| ablation | lead | pipeline | seeds | coverage % | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | II | Learned-1L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| A1 | II | Learned-1L (supervised-only) | S1–S3 | 100.0 ± 0.0 | 0.0365 ± 0.0015 | 0.0068 ± 0.0028 | 0.0556 ± 0.0017 | 0.626 ± 0.011 | 0.858 ± 0.014 | 8.0 ± 0.7 | 0.714 ± 0.018 |
| A1 | II | Learned-2L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0290 ± 0.0018 | 0.0020 ± 0.0023 | 0.0444 ± 0.0016 | 0.661 ± 0.011 | 0.931 ± 0.008 | 5.1 ± 0.3 | 0.817 ± 0.013 |
| A1 | II | Learned-2L (supervised-only) | S1–S3 | 100.0 ± 0.0 | 0.0361 ± 0.0010 | 0.0034 ± 0.0017 | 0.0586 ± 0.0027 | 0.560 ± 0.014 | 0.848 ± 0.023 | 7.0 ± 0.2 | 0.682 ± 0.029 |
| A2 | II | Learned-1L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| A2 | II | Learned-1L (published) | published | 100.0 | 0.0307 | 0.0014 | 0.0465 | 0.655 | 0.918 | 6.4 | 0.801 |
| A1 | V2 | Learned-1L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| A1 | V2 | Learned-1L (supervised-only) | S1–S3 | 100.0 ± 0.0 | 0.0336 ± 0.0013 | 0.0050 ± 0.0021 | 0.0426 ± 0.0009 | 0.429 ± 0.028 | 0.550 ± 0.006 | 2.2 ± 0.2 | 0.236 ± 0.031 |
| A1 | V2 | Learned-2L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0290 ± 0.0011 | 0.0020 ± 0.0007 | 0.0368 ± 0.0011 | 0.389 ± 0.037 | 0.659 ± 0.027 | 0.7 ± 0.3 | 0.427 ± 0.036 |
| A1 | V2 | Learned-2L (supervised-only) | S1–S3 | 100.0 ± 0.0 | 0.0349 ± 0.0016 | 0.0004 ± 0.0035 | 0.0436 ± 0.0019 | 0.358 ± 0.035 | 0.498 ± 0.046 | 2.3 ± 0.7 | 0.197 ± 0.068 |
| A2 | V2 | Learned-1L (pretrained) | S1–S3 | 100.0 ± 0.0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| A2 | V2 | Learned-1L (published) | published | 100.0 | 0.0384 | 0.0154 | 0.0466 | 0.316 | 0.473 | 2.6 | 0.085 |

### T10b. Alarms (T5 metrics, union, FULL)

Gate: rodrigues2021 peaks, 6 dB, except the A4a `.atr` rows. Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record–lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN).

| ablation | condition | pipeline | seeds | gate | TP | FP | FN | precision | recall | F1 | IoU | redundant |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | baseline | Learned-1L (pretrained) | S1–S3 | – | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 18.0 ± 2.6 |
| A1 | artifact | Learned-1L (pretrained) | S1–S3 | – | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 72.7 ± 13.8 |
| A1 | gated | Learned-1L (pretrained) | S1–S3 | rodrigues2021 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 11.7 ± 2.5 |
| A1 | baseline | Learned-1L (supervised-only) | S1–S3 | – | 63.7 ± 4.0 | 45.0 ± 4.6 | 37.3 ± 4.0 | 0.586 ± 0.035 | 0.630 ± 0.040 | 0.607 ± 0.035 | 0.471 ± 0.025 | 22.3 ± 3.8 |
| A1 | artifact | Learned-1L (supervised-only) | S1–S3 | – | 61.7 ± 4.7 | 62.0 ± 5.0 | 39.3 ± 4.7 | 0.499 ± 0.030 | 0.611 ± 0.047 | 0.549 ± 0.035 | 0.369 ± 0.038 | 121.0 ± 10.8 |
| A1 | gated | Learned-1L (supervised-only) | S1–S3 | rodrigues2021 | 64.0 ± 4.4 | 21.7 ± 1.5 | 37.0 ± 4.4 | 0.747 ± 0.026 | 0.634 ± 0.043 | 0.685 ± 0.036 | 0.439 ± 0.043 | 13.0 ± 2.6 |
| A1 | baseline | Learned-2L (pretrained) | S1–S3 | – | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 20.3 ± 0.6 |
| A1 | artifact | Learned-2L (pretrained) | S1–S3 | – | 73.3 ± 2.1 | 66.7 ± 18.1 | 27.7 ± 2.1 | 0.530 ± 0.061 | 0.726 ± 0.021 | 0.610 ± 0.034 | 0.402 ± 0.027 | 109.7 ± 1.5 |
| A1 | gated | Learned-2L (pretrained) | S1–S3 | rodrigues2021 | 75.7 ± 1.2 | 20.0 ± 3.5 | 25.3 ± 1.2 | 0.791 ± 0.032 | 0.749 ± 0.011 | 0.770 ± 0.021 | 0.503 ± 0.019 | 14.7 ± 1.2 |
| A1 | baseline | Learned-2L (supervised-only) | S1–S3 | – | 64.3 ± 2.5 | 30.3 ± 7.0 | 36.7 ± 2.5 | 0.682 ± 0.055 | 0.637 ± 0.025 | 0.658 ± 0.033 | 0.369 ± 0.024 | 20.7 ± 5.0 |
| A1 | artifact | Learned-2L (supervised-only) | S1–S3 | – | 62.3 ± 3.2 | 45.7 ± 2.5 | 38.7 ± 3.2 | 0.577 ± 0.019 | 0.617 ± 0.032 | 0.596 ± 0.023 | 0.321 ± 0.030 | 75.7 ± 8.6 |
| A1 | gated | Learned-2L (supervised-only) | S1–S3 | rodrigues2021 | 63.3 ± 4.0 | 17.0 ± 3.0 | 37.7 ± 4.0 | 0.789 ± 0.021 | 0.627 ± 0.040 | 0.698 ± 0.019 | 0.346 ± 0.039 | 15.0 ± 4.6 |
| A2 | baseline | Learned-1L (pretrained) | S1–S3 | – | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 18.0 ± 2.6 |
| A2 | artifact | Learned-1L (pretrained) | S1–S3 | – | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 72.7 ± 13.8 |
| A2 | gated | Learned-1L (pretrained) | S1–S3 | rodrigues2021 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 11.7 ± 2.5 |
| A2 | baseline | Learned-1L (published) | published | – | 75.0 | 40.0 | 26.0 | 0.652 | 0.743 | 0.694 | 0.538 | 21.0 |
| A2 | artifact | Learned-1L (published) | published | – | 73.0 | 57.0 | 28.0 | 0.562 | 0.723 | 0.632 | 0.491 | 99.0 |
| A2 | gated | Learned-1L (published) | published | rodrigues2021 | 75.0 | 27.0 | 26.0 | 0.735 | 0.743 | 0.739 | 0.539 | 14.0 |

### T10c. A4a: gate peaks, `.atr` vs rodrigues2021 (gated, 6 dB)

Gated MAE: artifact predictions against the baseline labels on the set A windows each gate keeps (gate at the learned grid anchor), excluded pairs left out.

| pipeline | gate | TP | FP | FN | F1 | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Learned-1L (pretrained, S1) | rodrigues2021 | 78 | 29 | 23 | 0.750 | 78.3 | 0.0291 | 77.0 | 0.0360 |
| Learned-1L (pretrained, S1) | .atr | 76 | 25 | 25 | 0.752 | 72.5 | 0.0299 | 78.0 | 0.0360 |
| Learned-2L (pretrained, S1) | rodrigues2021 | 75 | 22 | 26 | 0.758 | 78.3 | 0.0311 | 77.0 | 0.0314 |
| Learned-2L (pretrained, S1) | .atr | 75 | 24 | 26 | 0.750 | 72.5 | 0.0320 | 78.0 | 0.0318 |

### T10d. A4b: SQI threshold sweep (rodrigues2021 gate)

Every 5 dB shown; all 46 thresholds (−20 to +25 dB, 1 dB steps) in `Tasks/final/runs/step7_t10/T10_A4b_sweep.csv` and `docs/thesis_figures/FigA4b_gate_threshold_sweep.pdf`.

| threshold (dB) | pipeline | TP | FP | FN | F1 | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| -20 | Learned-1L (pretrained, S1) | 74 | 49 | 27 | 0.661 | 100.0 | 0.0352 | 100.0 | 0.0355 |
| -20 | Learned-2L (pretrained, S1) | 75 | 86 | 26 | 0.573 | 100.0 | 0.0368 | 100.0 | 0.0315 |
| -15 | Learned-1L (pretrained, S1) | 74 | 49 | 27 | 0.661 | 100.0 | 0.0351 | 99.9 | 0.0355 |
| -15 | Learned-2L (pretrained, S1) | 75 | 86 | 26 | 0.573 | 100.0 | 0.0368 | 99.9 | 0.0314 |
| -10 | Learned-1L (pretrained, S1) | 74 | 45 | 27 | 0.673 | 98.5 | 0.0344 | 99.9 | 0.0355 |
| -10 | Learned-2L (pretrained, S1) | 75 | 84 | 26 | 0.577 | 98.5 | 0.0361 | 99.9 | 0.0314 |
| -5 | Learned-1L (pretrained, S1) | 74 | 39 | 27 | 0.692 | 91.7 | 0.0318 | 99.4 | 0.0355 |
| -5 | Learned-2L (pretrained, S1) | 75 | 61 | 26 | 0.633 | 91.7 | 0.0339 | 99.4 | 0.0314 |
| +0 | Learned-1L (pretrained, S1) | 76 | 35 | 25 | 0.717 | 86.2 | 0.0298 | 96.9 | 0.0353 |
| +0 | Learned-2L (pretrained, S1) | 75 | 41 | 26 | 0.691 | 86.2 | 0.0319 | 96.9 | 0.0312 |
| +5 | Learned-1L (pretrained, S1) | 77 | 29 | 24 | 0.744 | 81.7 | 0.0290 | 82.5 | 0.0360 |
| +5 | Learned-2L (pretrained, S1) | 76 | 25 | 25 | 0.752 | 81.7 | 0.0311 | 82.5 | 0.0314 |
| +6 (main) | Learned-1L (pretrained, S1) | 78 | 29 | 23 | 0.750 | 78.3 | 0.0291 | 77.0 | 0.0360 |
| +6 (main) | Learned-2L (pretrained, S1) | 75 | 22 | 26 | 0.758 | 78.3 | 0.0311 | 77.0 | 0.0314 |
| +10 | Learned-1L (pretrained, S1) | 66 | 10 | 35 | 0.746 | 41.9 | 0.0278 | 49.6 | 0.0355 |
| +10 | Learned-2L (pretrained, S1) | 68 | 10 | 33 | 0.760 | 41.9 | 0.0265 | 49.6 | 0.0305 |
| +15 | Learned-1L (pretrained, S1) | 45 | 1 | 56 | 0.612 | 10.9 | 0.0237 | 12.6 | 0.0326 |
| +15 | Learned-2L (pretrained, S1) | 43 | 1 | 58 | 0.593 | 10.9 | 0.0247 | 12.6 | 0.0298 |
| +20 | Learned-1L (pretrained, S1) | 3 | 0 | 98 | 0.058 | 2.5 | 0.0160 | 0.1 | 0.0275 |
| +20 | Learned-2L (pretrained, S1) | 7 | 0 | 94 | 0.130 | 2.5 | 0.0253 | 0.1 | 0.0398 |
| +25 | Learned-1L (pretrained, S1) | 0 | 0 | 101 | 0.000 | 0.0 | nan | 0.0 | nan |
| +25 | Learned-2L (pretrained, S1) | 0 | 0 | 101 | 0.000 | 0.0 | nan | 0.0 | nan |
<!-- /t10 -->

<!-- fig1 -->
## Fig. 1. Bland–Altman (baseline, DMY beats)

Generated 2026-11-24 04:18:48 by `src/scripts/final/fin_step7_t10_fig1.py` (VM). Figure: `docs/thesis_figures/Fig1_bland_altman.pdf`. Label = the DMY beat's expert ST deviation; estimate = the 10 s window containing the beat (as phase 1's set B and C §9); excluded pairs left out. LoA = bias ± 1.96 SD of the difference; slope = OLS of the difference on the mean (negative = compression of large deviations).

| lead | pipeline | n beats | bias (mV) | SD (mV) | 95 % LoA (mV) | slope (diff on mean) | MAE (mV) |
|---|---|--:|--:|--:|---|--:|--:|
| II | Learned-1L (pretrained, S1) | 1702 | +0.0013 | 0.0465 | [-0.0898, +0.0924] | -0.253 | 0.0310 |
| II | Learned-2L (pretrained, S1) | 1702 | +0.0002 | 0.0495 | [-0.0968, +0.0971] | -0.337 | 0.0334 |
| V2 | Learned-1L (pretrained, S1) | 1425 | +0.0106 | 0.0399 | [-0.0677, +0.0889] | -0.342 | 0.0324 |
| V2 | Learned-2L (pretrained, S1) | 1425 | +0.0033 | 0.0368 | [-0.0689, +0.0754] | -0.498 | 0.0295 |
<!-- /fig1 -->

<!-- t2 -->
## T2. Regression, DMY beats (baseline)

Generated 2026-11-24 04:18:51 by `src/scripts/final/fin_step7_rest.py` (VM). DMY beats (label = the beat's expert ST deviation); estimate = the 10 s window containing the beat (C §9, phase 1's set B); excluded pairs not scored; T1's metrics; coverage = share of DMY beats with a value. Learned-1L (pretrained), Learned-2L (pretrained): mean ± SD over S1–S3.

| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| II | Learned-1L (pretrained) | 100.0 ± 0.0 | 1702 ± 0 | 0.0303 ± 0.0008 | 0.0030 ± 0.0017 | 0.0464 ± 0.0014 | 0.704 ± 0.016 | 0.915 ± 0.006 | 4.4 ± 0.4 | 0.814 ± 0.011 |
| II | Learned-2L (pretrained) | 100.0 ± 0.0 | 1702 ± 0 | 0.0311 ± 0.0020 | 0.0020 ± 0.0019 | 0.0483 ± 0.0015 | 0.664 ± 0.009 | 0.914 ± 0.008 | 4.5 ± 0.3 | 0.799 ± 0.012 |
| II | Fiducial | 83.5 | 1421 | 0.0382 | -0.0345 | 0.0700 | 0.939 | 0.851 | 8.4 | 0.552 |
| V2 | Learned-1L (pretrained) | 100.0 ± 0.0 | 1425 ± 0 | 0.0322 ± 0.0003 | 0.0101 ± 0.0005 | 0.0410 ± 0.0002 | 0.445 ± 0.008 | 0.578 ± 0.005 | 1.8 ± 0.3 | 0.252 ± 0.009 |
| V2 | Learned-2L (pretrained) | 100.0 ± 0.0 | 1425 ± 0 | 0.0283 ± 0.0010 | 0.0037 ± 0.0004 | 0.0360 ± 0.0011 | 0.430 ± 0.030 | 0.656 ± 0.028 | 1.0 ± 0.5 | 0.424 ± 0.036 |
| V2 | Fiducial | 75.2 | 1071 | 0.0450 | -0.0420 | 0.0582 | 0.731 | 0.674 | 9.0 | -0.487 |

DMY beats per lead (excluded pairs left out): II 1,702, V2 1,425.
<!-- /t2 -->

<!-- t3 -->
## T3. Error by true ST band (baseline)

Generated 2026-11-24 04:18:58 by `src/scripts/final/fin_step7_rest.py` (VM). Set A windows (as T1), excluded pairs not scored, banded by |label| (lower bound inclusive). Bias = estimate − label; shrinkage = mean of −sign(label) × (estimate − label), positive when the estimate is closer to 0 than the label. Bands partition the labelled windows: yes. Learned-1L (pretrained), Learned-2L (pretrained): mean ± SD over S1–S3.

| lead | \|ST\| band (mV) | windows (% of lead) | pipeline | n with value | MAE (mV) | bias (mV) | shrinkage (mV) |
|---|---|--:|---|--:|--:|--:|--:|
| II | < 0.05 | 46,966 (42.4 %) | Learned-1L (pretrained) | 46966 ± 0 | 0.0147 ± 0.0004 | -0.0019 ± 0.0006 | -0.0009 ± 0.0006 |
|  |  |  | Learned-2L (pretrained) | 46966 ± 0 | 0.0137 ± 0.0020 | -0.0068 ± 0.0036 | 0.0013 ± 0.0003 |
|  |  |  | Fiducial | 37724 | 0.0234 | -0.0203 | -0.0042 |
| II | 0.05–0.10 | 35,237 (31.8 %) | Learned-1L (pretrained) | 35237 ± 0 | 0.0259 ± 0.0009 | -0.0131 ± 0.0010 | 0.0069 ± 0.0010 |
|  |  |  | Learned-2L (pretrained) | 35237 ± 0 | 0.0259 ± 0.0023 | -0.0160 ± 0.0032 | 0.0086 ± 0.0020 |
|  |  |  | Fiducial | 28225 | 0.0355 | -0.0312 | -0.0128 |
| II | 0.10–0.20 | 19,252 (17.4 %) | Learned-1L (pretrained) | 19252 ± 0 | 0.0323 ± 0.0042 | 0.0086 ± 0.0020 | 0.0193 ± 0.0024 |
|  |  |  | Learned-2L (pretrained) | 19252 ± 0 | 0.0312 ± 0.0005 | 0.0112 ± 0.0021 | 0.0207 ± 0.0014 |
|  |  |  | Fiducial | 17114 | 0.0494 | -0.0458 | -0.0291 |
| II | ≥ 0.20 | 9,328 (8.4 %) | Learned-1L (pretrained) | 9328 ± 0 | 0.1033 ± 0.0075 | 0.0807 ± 0.0093 | 0.1026 ± 0.0076 |
|  |  |  | Learned-2L (pretrained) | 9328 ± 0 | 0.1128 ± 0.0032 | 0.0953 ± 0.0029 | 0.1115 ± 0.0030 |
|  |  |  | Fiducial | 6309 | 0.0577 | -0.0523 | 0.0203 |
| V2 | < 0.05 | 17,948 (19.0 %) | Learned-1L (pretrained) | 17948 ± 0 | 0.0416 ± 0.0012 | 0.0406 ± 0.0013 | -0.0148 ± 0.0008 |
|  |  |  | Learned-2L (pretrained) | 17948 ± 0 | 0.0396 ± 0.0019 | 0.0379 ± 0.0034 | -0.0155 ± 0.0021 |
|  |  |  | Fiducial | 13461 | 0.0255 | -0.0218 | 0.0180 |
| V2 | 0.05–0.10 | 38,283 (40.5 %) | Learned-1L (pretrained) | 38283 ± 0 | 0.0275 ± 0.0005 | 0.0196 ± 0.0008 | -0.0194 ± 0.0008 |
|  |  |  | Learned-2L (pretrained) | 38283 ± 0 | 0.0202 ± 0.0013 | 0.0132 ± 0.0012 | -0.0130 ± 0.0011 |
|  |  |  | Fiducial | 27562 | 0.0395 | -0.0360 | 0.0357 |
| V2 | 0.10–0.20 | 35,661 (37.7 %) | Learned-1L (pretrained) | 35661 ± 0 | 0.0359 ± 0.0014 | -0.0134 ± 0.0014 | 0.0135 ± 0.0014 |
|  |  |  | Learned-2L (pretrained) | 35661 ± 0 | 0.0293 ± 0.0015 | -0.0220 ± 0.0016 | 0.0220 ± 0.0016 |
|  |  |  | Fiducial | 27542 | 0.0528 | -0.0515 | 0.0515 |
| V2 | ≥ 0.20 | 2,583 (2.7 %) | Learned-1L (pretrained) | 2583 ± 0 | 0.1049 ± 0.0040 | -0.1045 ± 0.0040 | 0.1049 ± 0.0040 |
|  |  |  | Learned-2L (pretrained) | 2583 ± 0 | 0.0834 ± 0.0030 | -0.0831 ± 0.0030 | 0.0834 ± 0.0030 |
|  |  |  | Fiducial | 2318 | 0.1027 | -0.1017 | 0.1018 |
<!-- /t3 -->

<!-- t4 -->
## T4. Noise robustness (NSTDB `em`, seed 42)

Generated 2026-11-24 04:19:08 by `src/scripts/final/fin_step7_rest.py` (VM). Artifact estimates against the baseline labels, set A windows, excluded pairs not scored. Gated = windows the 6 dB SQI gate (rodrigues2021 peaks, at the learned grid anchor) keeps, the same windows for every pipeline (as T7). Bursts = windows entirely inside a burst of the seed-42 contamination log. n = windows with a value. Learned-1L (pretrained), Learned-2L (pretrained): mean ± SD over S1–S3.

| lead | pipeline | ungated MAE (mV) | n | ungated burst MAE (mV) | n | gated MAE (mV) | n | gated burst MAE (mV) | n |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| II | Learned-1L (pretrained) | 0.0349 ± 0.0003 | 110,783 | 0.0729 ± 0.0012 | 14,754 | 0.0288 ± 0.0004 | 86,707 | 0.0613 ± 0.0029 | 11 |
| II | Learned-2L (pretrained) | 0.0352 ± 0.0015 | 110,783 | 0.0742 ± 0.0006 | 14,754 | 0.0292 ± 0.0018 | 86,707 | 0.0607 ± 0.0023 | 11 |
| II | Fiducial | 0.0666 | 91,875 | 0.2339 | 14,202 | 0.0341 | 69,532 | 0.1238 | 8 |
| V2 | Learned-1L (pretrained) | 0.0354 ± 0.0002 | 94,475 | 0.0349 ± 0.0007 | 12,586 | 0.0359 ± 0.0003 | 72,704 | 0.0417 ± 0.0014 | 1,801 |
| V2 | Learned-2L (pretrained) | 0.0304 ± 0.0009 | 94,475 | 0.0391 ± 0.0012 | 12,586 | 0.0303 ± 0.0010 | 72,704 | 0.0571 ± 0.0021 | 1,801 |
| V2 | Fiducial | 0.0462 | 73,823 | 0.0564 | 12,147 | 0.0439 | 53,999 | 0.0701 | 1,620 |

Labelled windows: II 110,783 (bursts 14,754), gate keeps 86,707 (bursts 11); V2 94,475 (bursts 12,586), gate keeps 72,704 (bursts 1,801).
<!-- /t4 -->

<!-- t6 -->
## T6. Alarm robustness across noise seeds (gated)

Generated 2026-11-24 04:19:23 by `src/scripts/final/fin_step7_rest.py` (VM). NSTDB `em` seeds 42, 43, 44, each with its own artifact signal, predictions, NeuroKit values and SQI gate (rodrigues2021 peaks, 6 dB). Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record–lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN). Mean ± SD (ddof 1) over the three noise seeds; Learned-1L (pretrained), Learned-2L (pretrained): training seed S1. Check: seed 42 equals the tier-1 gated rows (3 / 3).

| pipeline | TP | FP | FN | F1 | per seed 42 / 43 / 44: TP / FP / FN |
|---|--:|--:|--:|--:|---|
| Learned-1L (pretrained, S1) | 75.7 ± 2.1 | 29.0 ± 3.0 | 25.3 ± 2.1 | 0.736 ± 0.018 | 78/29/23; 74/32/27; 75/26/26 |
| Learned-2L (pretrained, S1) | 75.3 ± 0.6 | 20.7 ± 1.2 | 25.7 ± 0.6 | 0.765 ± 0.007 | 75/22/26; 76/20/25; 75/20/26 |
| Fiducial | 68.3 ± 2.1 | 21.0 ± 0.0 | 32.7 ± 2.1 | 0.718 ± 0.014 | 66/21/35; 69/21/32; 70/21/31 |
<!-- /t6 -->

<!-- data -->
## Data

Generated 2026-11-24 04:23:16 by `src/scripts/final/fin_step7_data_reference.py` (VM). Pre-training: Chapman–Shaoxing and CinC 2020 (`Unified_Dataset_No_PTB_Segment_v2`, 58,942 10 s windows at 100 Hz from Chapman and CinC 2020, md5 `ac56fd0e…`); rows where a selected lead is blank are removed. Fine-tuning: the fold dataset CSVs (`phase1_folds/`), one split for every learned model; a window is a 10 s segment of the median LTST signal with both targets (every row has both). Excluded record–lead pairs are in the training rows (their targets are present) and are excluded from scoring; gold-standard episodes: 69 lead II + 32 lead V2 = 101, none on a excluded pair.

### Pre-training set

| backbone | leads | used by | rows | train | validation | Chapman | CinC 2020 |
|---|---|---|--:|--:|--:|--:|--:|
| `patch_ecg_unified_ii_lossfix` | II | Learned-1L (pretrained) lead II model | 58,942 | 47,154 | 11,788 | 40,507 | 18,435 |
| `patch_ecg_unified_v2_lossfix` | V2 | Learned-1L (pretrained) lead V2 model | 57,843 | 46,274 | 11,569 | 39,415 | 18,428 |
| `patch_ecg_unified_ii_v2_rebuilt` | II + V2 | Learned-2L (pretrained) | 57,843 | 46,274 | 11,569 | 39,415 | 18,428 |

The two-lead backbone's counts are from a read-only check on Narval (2026-11-22: its `pretrain_split.csv` and `data_source.txt`, `_best.pt` sha256 `f3554f14…`; the same check reproduced the two single-lead rows above). It holds the same 57,843 windows as the V2 backbone (lead II is never blank, so the II + V2 filter equals the V2 filter), with its own random 80/20 train / validation draw: the totals per split and per source are equal, the assignment differs (39,179 of 57,843 windows, 68 %, share a split; Narval's count).

### Fine-tuning folds

| fold | dataset | split | records | windows | II windows (no excluded) | V2 windows (no excluded) | II episodes | V2 episodes | record IDs (excluded: II / V2) |
|--:|---|---|--:|--:|--:|--:|--:|--:|---|
| 1 | `cv5_optB_leadV2_opt2p_fold01` | train | 10 | 78,797 | 63,201 | 62,892 | 42 | 20 | s20011 s20061 s20071 s20091 s20111 s20121 s20131 s20141 s20201 s20241 (excluded: s20091 s20121 / s20121 s20241) |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | validation | 2 | 15,609 | 15,609 | 15,609 | 12 | 4 | s20101 s20231 |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | test | 3 | 23,754 | 23,754 | 15,959 | 10 | 8 | s20051 s20211 s20221 (excluded: – / s20221) |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | unused | 1 | 8,204 | 8,204 | 0 | 5 | 0 | s20081 (excluded: – / s20081) |
| 2 | `cv5_optB_fold02` | train | 11 | 86,485 | 70,889 | 62,785 | 47 | 23 | s20051 s20061 s20071 s20091 s20111 s20121 s20131 s20201 s20211 s20221 s20241 (excluded: s20091 s20121 / s20121 s20221 s20241) |
| 2 | `cv5_optB_fold02` | validation | 2 | 16,636 | 16,636 | 16,636 | 6 | 6 | s20011 s20231 |
| 2 | `cv5_optB_fold02` | test | 3 | 23,243 | 23,243 | 15,039 | 16 | 3 | s20081 s20101 s20141 (excluded: – / s20081) |
| 3 | `cv5_optB_fold03` | train | 11 | 85,631 | 78,130 | 62,131 | 56 | 24 | s20051 s20071 s20081 s20101 s20111 s20121 s20131 s20201 s20211 s20221 s20231 (excluded: s20121 / s20081 s20121 s20221) |
| 3 | `cv5_optB_fold03` | validation | 2 | 16,066 | 16,066 | 16,066 | 5 | 5 | s20011 s20141 |
| 3 | `cv5_optB_fold03` | test | 3 | 24,667 | 16,572 | 16,263 | 8 | 3 | s20061 s20091 s20241 (excluded: s20091 / s20241) |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | train | 10 | 78,565 | 62,969 | 54,865 | 40 | 18 | s20011 s20051 s20091 s20101 s20121 s20141 s20201 s20211 s20221 s20241 (excluded: s20091 s20121 / s20121 s20221 s20241) |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | validation | 2 | 15,751 | 15,751 | 15,751 | 7 | 5 | s20061 s20071 |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | test | 3 | 23,844 | 23,844 | 23,844 | 17 | 9 | s20111 s20131 s20231 |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | unused | 1 | 8,204 | 8,204 | 0 | 5 | 0 | s20081 (excluded: – / s20081) |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | train | 10 | 79,310 | 63,714 | 55,810 | 40 | 20 | s20051 s20061 s20081 s20091 s20101 s20111 s20121 s20211 s20221 s20231 (excluded: s20091 s20121 / s20081 s20121 s20221) |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | validation | 2 | 15,295 | 15,295 | 15,295 | 9 | 3 | s20131 s20141 |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | test | 3 | 23,355 | 23,355 | 23,355 | 18 | 9 | s20011 s20071 s20201 |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | unused | 1 | 8,404 | 8,404 | 0 | 2 | 0 | s20241 (excluded: – / s20241) |

Checks: test records over the 5 folds = the 15 scored records, each once: yes; fold 1: episodes over splits = 69 II / 32 V2: yes; fold 1: 16 records over splits: yes; fold 2: episodes over splits = 69 II / 32 V2: yes; fold 2: 16 records over splits: yes; fold 3: episodes over splits = 69 II / 32 V2: yes; fold 3: 16 records over splits: yes; fold 4: episodes over splits = 69 II / 32 V2: yes; fold 4: 16 records over splits: yes; fold 5: episodes over splits = 69 II / 32 V2: yes; fold 5: 16 records over splits: yes; pre-training total rows in the cluster report = 58,942 (Chapman–Shaoxing and CinC 2020): yes.
Note: the CSVs hold the windows the trainer used (step 6 reproduced every saved test MSE on them); the prediction grid of the trainer's H5 has 263 more 10 s segments over the 15 records (119,126 vs 118,863), which have no CSV row. The scored set A grid (T1) is a separate 10 s grid from sample 0.
<!-- /data -->

<!-- reference_window -->
## Reference-window benchmark (baseline)

Generated 2026-11-24 04:20:34 by `src/scripts/final/fin_step7_data_reference.py` (VM). The reference window values (set A: median `.16a` `ee` of each 10 s grid window, anchors 2500k + 2499) through the same FSM and scorer as T5, baseline condition (no noise, no gate), excluded pairs not scored. Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record–lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN). It is the score of a perfect window-level estimator under this FSM. Control: equals phase 1's oracle union (yes).

| pipeline | TP | FP | FN | precision | recall | F1 | IoU | duration Se | duration +P | redundant |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| reference window values | 100 | 9 | 1 | 0.917 | 0.990 | 0.952 | 0.920 | 0.997 | 0.923 | 3 |
| Fiducial (T5) | 72.0 | 26.0 | 29.0 | 0.735 | 0.713 | 0.724 | 0.445 | 0.523 | 0.749 | 12.0 |
| Learned-1L (pretrained) (T5) | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 0.738 ± 0.022 | 0.687 ± 0.028 | 18.0 ± 2.6 |
| Learned-2L (pretrained) (T5) | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 0.694 ± 0.034 | 0.720 ± 0.051 | 20.3 ± 0.6 |

With the true labels as input, the 1 missed episode and 9 false alarms reflect differences between the FSM on 10 s window medians and the way the gold-standard episodes were built; their cause was not examined.
<!-- /reference_window -->

<!-- reference_window_inputs -->
### Reference-window note: how each system feeds the FSM (written by hand, 2026-11-22; updated 2026-11-24)

Checked in the code (`src/scripts/final/fin_step5_nk.py`, `Tasks/C_neurokit/c13_fp_gap.segments`, `c20b_paper_rule.paper_beats`,
`ltst_pipeline/old_fsm_gs.run_fsm_on_record`). Every final system feeds the FSM **one value per 10 s window, never per beat**, on the same grid
(windows [2500k, 2500k + 2500)), so the reference-window benchmark applies to all three.

| | Fiducial | learned systems (pretrained and supervised-only) | Reference window values |
|---|---|---|---|
| per-beat value | median of 5 samples (±8 ms) at J+60 in every condition minus the median of the PR segment (P offset to R onset); rodrigues2021 delineation | – (one prediction per window) | `.16a` `ee` of each annotated beat |
| beat → window | J+60 sample | – | J+60 sample |
| window value | median | model output | median (rounded to 4 decimals) |
| minimum beats | ≥ 5 usable beats (finite ST and baseline), else no value | – | ≥ 1 beat, else no value |
| FSM anchor | 2500k + 2500 | 2500k + 2499 | 2500k + 2499 |

- A window with no value is skipped by the FSM: its state is frozen and the skipped time does not count toward warm-up durations (the same for
  gated windows). The reference window value is the learned models' training target (phase 1's oracle control).
- What Ch4 should state for the fiducial system: (1) it needs ≥ 5 usable detected beats per window, the reference windows ≥ 1 annotated beat; on the
  baseline signal, Fiducial has a value on 80.7 % (II) / 75.0 % (V2) of labelled windows (T1); (2) it uses detected beats
  (rodrigues2021), not annotated ones; (3) its anchor is one sample (4 ms) later, which does
  not change the window a beat belongs to.
- Not measured: how much the ≥ 5-beat rule alone would change the reference-window benchmark (it would need a benchmark run with ≥ 5 annotated beats per window).
<!-- /reference_window_inputs -->
