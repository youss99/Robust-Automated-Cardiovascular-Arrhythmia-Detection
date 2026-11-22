# Final thesis results

Frozen setup: `docs/instructions_final_results.md` and `docs/DECISIONS.md`. Provenance: `Tasks/final/runs/step6/provenance.csv` (VM). Run log: `docs/reports/2026-11-21_vm_steps4_7.md`.

<!-- tier1 -->
## Tier 1 (T1, T5, T7)

Generated 2026-11-21 22:45:34 by `Tasks/final/fin_step7_tier1.py` (VM). 15 records, 101 ST-deviation episodes (MON: 99, as published). Learned rows: mean ± SD (ddof 1) over training seeds S1–S3; per-seed values in `Tasks/final/runs/step7/`. Noise: NSTDB `em`, seed 42. Gate: the paper's SQI from rodrigues2021 peaks on the noisy signal, 6 dB.

Control (published R1 predictions through this alarm path, `.atr` gate): all 6 conditions reproduce the paper.

### T1. Regression, all windows (clean)

Set A: the 10 s grid windows, label = median `.16a` `ee` of the window's beats; masked pairs excluded; each pipeline on its own windows with a value (coverage = share of labelled windows with a value). Slope = OLS of prediction on label.

| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| II | L1 | 100.0 ± 0.0 | 110783 ± 0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| II | L2 | 100.0 ± 0.0 | 110783 ± 0 | 0.0290 ± 0.0018 | 0.0020 ± 0.0023 | 0.0444 ± 0.0016 | 0.661 ± 0.011 | 0.931 ± 0.008 | 5.1 ± 0.3 | 0.817 ± 0.013 |
| II | NK-PR | 80.7 | 89372 | 0.0347 | -0.0309 | 0.0639 | 0.935 | 0.851 | 7.5 | 0.561 |
| II | NK-FP | 100.0 | 110783 | 0.0127 | -0.0039 | 0.0181 | 0.948 | 0.986 | 0.0 | 0.970 |
| V2 | L1 | 100.0 ± 0.0 | 94475 ± 0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| V2 | L2 | 100.0 ± 0.0 | 94475 ± 0 | 0.0290 ± 0.0011 | 0.0020 ± 0.0007 | 0.0368 ± 0.0011 | 0.389 ± 0.037 | 0.659 ± 0.027 | 0.7 ± 0.3 | 0.427 ± 0.036 |
| V2 | NK-PR | 75.0 | 70883 | 0.0441 | -0.0415 | 0.0588 | 0.672 | 0.663 | 8.8 | -0.361 |
| V2 | NK-FP | 100.0 | 94475 | 0.0109 | -0.0043 | 0.0197 | 1.041 | 0.936 | 0.1 | 0.837 |

### T5. Alarms, main table

Union of II and V2 alarms, FULL scope. Duration Se / +P = time sensitivity / time PPV; redundant = Ignored (extra alarms inside an already-detected episode); false-alarm hours = time in false alarms. MON copied from the paper (episode counts only).

| condition | pipeline | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | duration Se | duration +P | redundant |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clean | MON | 78 | 66 | 21 | 0.542 | 0.788 | 0.642 | – | – | – | – | n/a |
| clean | NK-PR | 71.0 | 26.0 | 30.0 | 0.732 | 0.703 | 0.717 | 0.445 | 34.3 | 0.524 | 0.749 | 10.0 |
| clean | NK-FP | 87.0 | 10.0 | 14.0 | 0.897 | 0.861 | 0.879 | 0.849 | 15.3 | 0.916 | 0.921 | 11.0 |
| clean | L1 | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 65.6 ± 6.6 | 0.738 ± 0.022 | 0.687 ± 0.028 | 18.0 ± 2.6 |
| clean | L2 | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 53.3 ± 13.8 | 0.694 ± 0.034 | 0.720 ± 0.051 | 20.3 ± 0.6 |
| noisy | MON | 77 | 171 | 22 | 0.310 | 0.778 | 0.444 | – | – | – | – | n/a |
| noisy | NK-PR | 67.0 | 51.0 | 34.0 | 0.568 | 0.663 | 0.612 | 0.392 | 35.2 | 0.462 | 0.719 | 13.0 |
| noisy | NK-FP | 87.0 | 30.0 | 14.0 | 0.744 | 0.861 | 0.798 | 0.773 | 15.7 | 0.835 | 0.912 | 34.0 |
| noisy | L1 | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 49.1 ± 3.5 | 0.614 ± 0.021 | 0.709 ± 0.019 | 72.7 ± 13.8 |
| noisy | L2 | 73.3 ± 2.1 | 66.7 ± 18.1 | 27.7 ± 2.1 | 0.530 ± 0.061 | 0.726 ± 0.021 | 0.610 ± 0.034 | 0.402 ± 0.027 | 25.6 ± 11.4 | 0.454 ± 0.022 | 0.780 ± 0.074 | 109.7 ± 1.5 |
| gated | MON | 76 | 54 | 23 | 0.585 | 0.768 | 0.664 | – | – | – | – | n/a |
| gated | NK-PR | 66.0 | 21.0 | 35.0 | 0.759 | 0.653 | 0.702 | 0.452 | 31.8 | 0.526 | 0.763 | 7.0 |
| gated | NK-FP | 84.0 | 6.0 | 17.0 | 0.933 | 0.832 | 0.880 | 0.822 | 24.3 | 0.925 | 0.881 | 6.0 |
| gated | L1 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 72.9 ± 5.5 | 0.724 ± 0.020 | 0.660 ± 0.023 | 11.7 ± 2.5 |
| gated | L2 | 75.7 ± 1.2 | 20.0 ± 3.5 | 25.3 ± 1.2 | 0.791 ± 0.032 | 0.749 ± 0.011 | 0.770 ± 0.021 | 0.503 ± 0.019 | 65.1 ± 5.6 | 0.670 ± 0.016 | 0.668 ± 0.022 | 14.7 ± 1.2 |

### T7. Paired differences (gated, seed 42)

Record-clustered paired bootstrap, 5,000 resamples (seed 20260805); training seeds averaged inside each resample. dMAE: set A windows the gate keeps (gate at the learned anchor, same for both), noisy predictions against the clean labels, masked pairs out, each pipeline on its own windows. * = 95 % CI excludes 0.

| comparison | metric | A | B | A − B | 95 % CI |
|---|---|--:|--:|--:|---|
| L2 - L1 | dF1 (union) | 0.770 | 0.752 | +0.018 | [-0.039, +0.078] |
| L2 - L1 | dFP (union) | 20.0 | 24.0 | -4.0 | [-14.3, +7.3] |
| L2 - L1 | dMAE II (mV) | 0.0292 | 0.0288 | +0.0004 | [-0.0027, +0.0038] |
| L2 - L1 | dMAE V2 (mV) | 0.0303 | 0.0359 | -0.0056 | [-0.0118, +0.0007] |
| L2 - NK-FP | dF1 (union) | 0.770 | 0.880 | -0.110 * | [-0.198, -0.022] |
| L2 - NK-FP | dFP (union) | 20.0 | 6.0 | +14.0 | [-2.3, +37.0] |
| L2 - NK-FP | dMAE II (mV) | 0.0292 | 0.0125 | +0.0167 * | [+0.0069, +0.0316] |
| L2 - NK-FP | dMAE V2 (mV) | 0.0303 | 0.0110 | +0.0193 * | [+0.0122, +0.0269] |
| L1 - NK-FP | dF1 (union) | 0.752 | 0.880 | -0.127 * | [-0.217, -0.050] |
| L1 - NK-FP | dFP (union) | 24.0 | 6.0 | +18.0 * | [+1.7, +37.7] |
| L1 - NK-FP | dMAE II (mV) | 0.0288 | 0.0125 | +0.0163 * | [+0.0075, +0.0290] |
| L1 - NK-FP | dMAE V2 (mV) | 0.0359 | 0.0110 | +0.0249 * | [+0.0171, +0.0334] |
| L2 - NK-PR | dF1 (union) | 0.770 | 0.702 | +0.068 | [-0.085, +0.204] |
| L2 - NK-PR | dFP (union) | 20.0 | 21.0 | -1.0 | [-19.3, +25.0] |
| L2 - NK-PR | dMAE II (mV) | 0.0292 | 0.0341 | -0.0049 | [-0.0318, +0.0182] |
| L2 - NK-PR | dMAE V2 (mV) | 0.0303 | 0.0439 | -0.0137 | [-0.0276, +0.0004] |
| L1 - L1sup | dF1 (union) | 0.752 | 0.693 | +0.059 * | [+0.013, +0.117] |
| L1 - L1sup | dFP (union) | 24.0 | 25.3 | -1.3 | [-10.3, +6.0] |
| L1 - L1sup | dMAE II (mV) | 0.0288 | 0.0368 | -0.0080 * | [-0.0141, -0.0031] |
| L1 - L1sup | dMAE V2 (mV) | 0.0359 | 0.0332 | +0.0027 | [-0.0030, +0.0100] |
| L2 - L2sup | dF1 (union) | 0.770 | 0.698 | +0.072 * | [+0.010, +0.141] |
| L2 - L2sup | dFP (union) | 20.0 | 19.7 | +0.3 | [-7.3, +7.3] |
| L2 - L2sup | dMAE II (mV) | 0.0292 | 0.0363 | -0.0071 * | [-0.0143, -0.0016] |
| L2 - L2sup | dMAE V2 (mV) | 0.0303 | 0.0357 | -0.0054 * | [-0.0101, -0.0009] |
<!-- /tier1 -->

<!-- t5_temporal -->
### T5 note: temporal metrics, L1 vs L2 (written by hand, 2026-11-22)

From the T5 rows above (mean ± SD over S1–S3). Temporal metrics: IoU, duration Se, duration +P, false-alarm hours.

| condition | IoU L1 / L2 | duration Se L1 / L2 | duration +P L1 / L2 | false-alarm h L1 / L2 |
|---|---|---|---|---|
| clean | 0.552 ± 0.030 / 0.546 ± 0.033 | 0.738 ± 0.022 / 0.694 ± 0.034 | 0.687 ± 0.028 / 0.720 ± 0.051 | 65.6 ± 6.6 / 53.3 ± 13.8 |
| noisy | 0.491 ± 0.021 / 0.402 ± 0.027 | 0.614 ± 0.021 / 0.454 ± 0.022 | 0.709 ± 0.019 / 0.780 ± 0.074 | 49.1 ± 3.5 / 25.6 ± 11.4 |
| gated | 0.528 ± 0.025 / 0.503 ± 0.019 | 0.724 ± 0.020 / 0.670 ± 0.016 | 0.660 ± 0.023 / 0.668 ± 0.022 | 72.9 ± 5.5 / 65.1 ± 5.6 |

- L1 covers more of each true episode in time: higher duration Se in every condition, and higher IoU (tie on clean).
- L2's alarms are more conservative: fewer false-alarm hours in every condition and higher duration +P (about equal when gated).
  T6 agrees across noise seeds 42–44, gated: false-alarm hours L1 (S1) 78.4 ± 3.3, L2 (S1) 70.1 ± 1.9.
- No significance test was run on any temporal metric (T7 covers F1, FP and MAE only). Gated, the IoU gap (0.025) is about one seed SD,
  so L1's temporal advantage is not established there; the noisy-condition gaps are several SDs. On episode F1, L2 is ahead (gated 0.770 vs 0.752,
  95 % CI of the difference includes 0, T7).
<!-- /t5_temporal -->

<!-- t10 -->
## T10. Ablations A1–A4

Generated 2026-11-21 22:55:54 by `Tasks/final/fin_step7_t10_fig1.py` (VM). Noise seed 42. Same alarm path, scorer, labels and gate tables as tier 1; rows marked S1–S3 are mean ± SD over the training seeds (A1, and L1 in A2), others one run. Each ablation row sits under the row it is compared with. L1-rebuilt (A2, optional) was not run: step 2b was not submitted.

Controls: published M1 and M2-R through this alarm path reproduce phase 1's stored counts (5 / 5); the 6 dB point of the sweep equals tier 1 (3 / 3).

### T10a. Regression (T1 metrics, clean)

| ablation | lead | pipeline | seeds | coverage % | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | II | L1 | S1–S3 | 100.0 ± 0.0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| A1 | II | L1-sup | S1–S3 | 100.0 ± 0.0 | 0.0366 ± 0.0007 | 0.0070 ± 0.0008 | 0.0573 ± 0.0014 | 0.603 ± 0.014 | 0.849 ± 0.013 | 8.0 ± 0.2 | 0.696 ± 0.015 |
| A1 | II | L2 | S1–S3 | 100.0 ± 0.0 | 0.0290 ± 0.0018 | 0.0020 ± 0.0023 | 0.0444 ± 0.0016 | 0.661 ± 0.011 | 0.931 ± 0.008 | 5.1 ± 0.3 | 0.817 ± 0.013 |
| A1 | II | L2-sup | S1–S3 | 100.0 ± 0.0 | 0.0358 ± 0.0009 | 0.0020 ± 0.0009 | 0.0585 ± 0.0034 | 0.567 ± 0.018 | 0.846 ± 0.028 | 6.6 ± 0.6 | 0.684 ± 0.037 |
| A2 | II | L1 | S1–S3 | 100.0 ± 0.0 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 4.7 ± 0.8 | 0.830 ± 0.004 |
| A2 | II | L1-old | published | 100.0 | 0.0307 | 0.0014 | 0.0465 | 0.655 | 0.918 | 6.4 | 0.801 |
| A3 | II | NK-FP | - | 100.0 | 0.0127 | -0.0039 | 0.0181 | 0.948 | 0.986 | 0.0 | 0.970 |
| A3 | II | NK-FP-nk | - | 100.0 | 0.0126 | -0.0046 | 0.0190 | 0.937 | 0.985 | 0.1 | 0.967 |
| A3 | II | NK-PR | - | 80.7 | 0.0347 | -0.0309 | 0.0639 | 0.935 | 0.851 | 7.5 | 0.561 |
| A3 | II | NK-PR-pub | - | 79.9 | 0.0355 | -0.0317 | 0.0645 | 0.936 | 0.845 | 7.6 | 0.531 |
| A1 | V2 | L1 | S1–S3 | 100.0 ± 0.0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| A1 | V2 | L1-sup | S1–S3 | 100.0 ± 0.0 | 0.0331 ± 0.0003 | 0.0046 ± 0.0004 | 0.0425 ± 0.0004 | 0.405 ± 0.017 | 0.540 ± 0.001 | 2.4 ± 0.2 | 0.238 ± 0.016 |
| A1 | V2 | L2 | S1–S3 | 100.0 ± 0.0 | 0.0290 ± 0.0011 | 0.0020 ± 0.0007 | 0.0368 ± 0.0011 | 0.389 ± 0.037 | 0.659 ± 0.027 | 0.7 ± 0.3 | 0.427 ± 0.036 |
| A1 | V2 | L2-sup | S1–S3 | 100.0 ± 0.0 | 0.0349 ± 0.0006 | -0.0010 ± 0.0014 | 0.0441 ± 0.0007 | 0.336 ± 0.018 | 0.480 ± 0.021 | 2.7 ± 0.4 | 0.181 ± 0.027 |
| A2 | V2 | L1 | S1–S3 | 100.0 ± 0.0 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 2.6 ± 0.4 | 0.168 ± 0.006 |
| A2 | V2 | L1-old | published | 100.0 | 0.0384 | 0.0154 | 0.0466 | 0.316 | 0.473 | 2.6 | 0.085 |
| A3 | V2 | NK-FP | - | 100.0 | 0.0109 | -0.0043 | 0.0197 | 1.041 | 0.936 | 0.1 | 0.837 |
| A3 | V2 | NK-FP-nk | - | 100.0 | 0.0323 | 0.0174 | 0.1176 | 0.922 | 0.360 | 4.7 | -4.832 |
| A3 | V2 | NK-PR | - | 75.0 | 0.0441 | -0.0415 | 0.0588 | 0.672 | 0.663 | 8.8 | -0.361 |
| A3 | V2 | NK-PR-pub | - | 88.4 | 0.0433 | -0.0405 | 0.0582 | 0.626 | 0.637 | 8.8 | -0.387 |

### T10b. Alarms (T5 metrics, union, FULL)

Gate: rodrigues2021 peaks, 6 dB, except NK-PR-pub (`.atr`, as published) and the A4a `.atr` rows.

| ablation | condition | pipeline | seeds | gate | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | redundant |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | clean | L1 | S1–S3 | – | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 65.6 ± 6.6 | 18.0 ± 2.6 |
| A1 | noisy | L1 | S1–S3 | – | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 49.1 ± 3.5 | 72.7 ± 13.8 |
| A1 | gated | L1 | S1–S3 | rodrigues2021 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 72.9 ± 5.5 | 11.7 ± 2.5 |
| A1 | clean | L1-sup | S1–S3 | – | 65.7 ± 1.5 | 50.3 ± 3.1 | 35.3 ± 1.5 | 0.566 ± 0.019 | 0.650 ± 0.015 | 0.605 ± 0.016 | 0.452 ± 0.004 | 59.9 ± 1.1 | 18.3 ± 2.5 |
| A1 | noisy | L1-sup | S1–S3 | – | 62.3 ± 1.5 | 69.7 ± 2.3 | 38.7 ± 1.5 | 0.472 ± 0.009 | 0.617 ± 0.015 | 0.535 ± 0.010 | 0.378 ± 0.008 | 40.7 ± 5.2 | 99.7 ± 9.5 |
| A1 | gated | L1-sup | S1–S3 | rodrigues2021 | 67.0 ± 1.0 | 25.3 ± 4.5 | 34.0 ± 1.0 | 0.727 ± 0.034 | 0.663 ± 0.010 | 0.693 ± 0.014 | 0.422 ± 0.011 | 69.4 ± 0.9 | 12.7 ± 0.6 |
| A1 | clean | L2 | S1–S3 | – | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 53.3 ± 13.8 | 20.3 ± 0.6 |
| A1 | noisy | L2 | S1–S3 | – | 73.3 ± 2.1 | 66.7 ± 18.1 | 27.7 ± 2.1 | 0.530 ± 0.061 | 0.726 ± 0.021 | 0.610 ± 0.034 | 0.402 ± 0.027 | 25.6 ± 11.4 | 109.7 ± 1.5 |
| A1 | gated | L2 | S1–S3 | rodrigues2021 | 75.7 ± 1.2 | 20.0 ± 3.5 | 25.3 ± 1.2 | 0.791 ± 0.032 | 0.749 ± 0.011 | 0.770 ± 0.021 | 0.503 ± 0.019 | 65.1 ± 5.6 | 14.7 ± 1.2 |
| A1 | clean | L2-sup | S1–S3 | – | 65.3 ± 1.2 | 32.3 ± 4.5 | 35.7 ± 1.2 | 0.670 ± 0.031 | 0.647 ± 0.011 | 0.658 ± 0.017 | 0.389 ± 0.026 | 48.7 ± 3.4 | 22.3 ± 7.2 |
| A1 | noisy | L2-sup | S1–S3 | – | 62.7 ± 1.5 | 50.0 ± 14.7 | 38.3 ± 1.5 | 0.562 ± 0.064 | 0.620 ± 0.015 | 0.588 ± 0.032 | 0.321 ± 0.007 | 41.2 ± 2.7 | 71.7 ± 4.7 |
| A1 | gated | L2-sup | S1–S3 | rodrigues2021 | 64.7 ± 2.1 | 19.7 ± 4.0 | 36.3 ± 2.1 | 0.768 ± 0.032 | 0.640 ± 0.021 | 0.698 ± 0.009 | 0.360 ± 0.015 | 58.0 ± 4.4 | 15.0 ± 1.7 |
| A2 | clean | L1 | S1–S3 | – | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 65.6 ± 6.6 | 18.0 ± 2.6 |
| A2 | noisy | L1 | S1–S3 | – | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 0.491 ± 0.021 | 49.1 ± 3.5 | 72.7 ± 13.8 |
| A2 | gated | L1 | S1–S3 | rodrigues2021 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 0.528 ± 0.025 | 72.9 ± 5.5 | 11.7 ± 2.5 |
| A2 | clean | L1-old | published | – | 75.0 | 40.0 | 26.0 | 0.652 | 0.743 | 0.694 | 0.538 | 81.0 | 21.0 |
| A2 | noisy | L1-old | published | – | 73.0 | 57.0 | 28.0 | 0.562 | 0.723 | 0.632 | 0.491 | 60.9 | 99.0 |
| A2 | gated | L1-old | published | rodrigues2021 | 75.0 | 27.0 | 26.0 | 0.735 | 0.743 | 0.739 | 0.539 | 85.1 | 14.0 |
| A3 | clean | NK-FP | - | – | 87.0 | 10.0 | 14.0 | 0.897 | 0.861 | 0.879 | 0.849 | 15.3 | 11.0 |
| A3 | noisy | NK-FP | - | – | 87.0 | 30.0 | 14.0 | 0.744 | 0.861 | 0.798 | 0.773 | 15.7 | 34.0 |
| A3 | gated | NK-FP | - | rodrigues2021 | 84.0 | 6.0 | 17.0 | 0.933 | 0.832 | 0.880 | 0.822 | 24.3 | 6.0 |
| A3 | clean | NK-FP-nk | - | – | 85.0 | 16.0 | 16.0 | 0.842 | 0.842 | 0.842 | 0.848 | 12.0 | 13.0 |
| A3 | noisy | NK-FP-nk | - | – | 86.0 | 84.0 | 15.0 | 0.506 | 0.851 | 0.635 | 0.749 | 17.6 | 49.0 |
| A3 | gated | NK-FP-nk | - | rodrigues2021 | 82.0 | 8.0 | 19.0 | 0.911 | 0.812 | 0.859 | 0.814 | 15.2 | 11.0 |
| A3 | clean | NK-PR | - | – | 71.0 | 26.0 | 30.0 | 0.732 | 0.703 | 0.717 | 0.445 | 34.3 | 10.0 |
| A3 | noisy | NK-PR | - | – | 67.0 | 51.0 | 34.0 | 0.568 | 0.663 | 0.612 | 0.392 | 35.2 | 13.0 |
| A3 | gated | NK-PR | - | rodrigues2021 | 66.0 | 21.0 | 35.0 | 0.759 | 0.653 | 0.702 | 0.452 | 31.8 | 7.0 |
| A3 | clean | NK-PR-pub | - | – | 67.0 | 32.0 | 34.0 | 0.677 | 0.663 | 0.670 | 0.418 | 34.9 | 13.0 |
| A3 | noisy | NK-PR-pub | - | – | 67.0 | 43.0 | 34.0 | 0.609 | 0.663 | 0.635 | 0.389 | 33.7 | 15.0 |
| A3 | gated | NK-PR-pub | - | .atr | 64.0 | 27.0 | 37.0 | 0.703 | 0.634 | 0.667 | 0.415 | 38.7 | 7.0 |

### T10c. A4a: gate peaks, `.atr` vs rodrigues2021 (gated, 6 dB)

Gated MAE: noisy predictions against the clean labels on the set A windows each gate keeps (gate at the learned grid anchor), masked pairs out.

| pipeline | gate | TP | FP | FN | F1 | false-alarm h | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L1 (S1) | rodrigues2021 | 78 | 29 | 23 | 0.750 | 75.4 | 78.3 | 0.0291 | 77.0 | 0.0360 |
| L1 (S1) | .atr | 76 | 25 | 25 | 0.752 | 80.3 | 72.5 | 0.0299 | 78.0 | 0.0360 |
| L2 (S1) | rodrigues2021 | 75 | 22 | 26 | 0.758 | 70.9 | 78.3 | 0.0311 | 77.0 | 0.0314 |
| L2 (S1) | .atr | 75 | 24 | 26 | 0.750 | 68.3 | 72.5 | 0.0320 | 78.0 | 0.0318 |
| NK-FP | rodrigues2021 | 84 | 6 | 17 | 0.880 | 24.3 | 78.3 | 0.0125 | 77.0 | 0.0110 |
| NK-FP | .atr | 80 | 9 | 21 | 0.842 | 21.2 | 72.5 | 0.0129 | 78.0 | 0.0103 |

### T10d. A4b: SQI threshold sweep (rodrigues2021 gate)

Every 5 dB shown; all 46 thresholds (−20 to +25 dB, 1 dB steps) in `Tasks/final/runs/step7_t10/T10_A4b_sweep.csv` and `docs/thesis_figures/FigA4b_gate_threshold_sweep.pdf`.

| threshold (dB) | pipeline | TP | FP | FN | F1 | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| -20 | L1 (S1) | 74 | 49 | 27 | 0.661 | 100.0 | 0.0352 | 100.0 | 0.0355 |
| -20 | L2 (S1) | 75 | 86 | 26 | 0.573 | 100.0 | 0.0368 | 100.0 | 0.0315 |
| -20 | NK-FP | 87 | 30 | 14 | 0.798 | 100.0 | 0.0395 | 100.0 | 0.0140 |
| -15 | L1 (S1) | 74 | 49 | 27 | 0.661 | 100.0 | 0.0351 | 99.9 | 0.0355 |
| -15 | L2 (S1) | 75 | 86 | 26 | 0.573 | 100.0 | 0.0368 | 99.9 | 0.0314 |
| -15 | NK-FP | 87 | 30 | 14 | 0.798 | 100.0 | 0.0394 | 99.9 | 0.0139 |
| -10 | L1 (S1) | 74 | 45 | 27 | 0.673 | 98.5 | 0.0344 | 99.9 | 0.0355 |
| -10 | L2 (S1) | 75 | 84 | 26 | 0.577 | 98.5 | 0.0361 | 99.9 | 0.0314 |
| -10 | NK-FP | 85 | 22 | 16 | 0.817 | 98.5 | 0.0367 | 99.9 | 0.0138 |
| -5 | L1 (S1) | 74 | 39 | 27 | 0.692 | 91.7 | 0.0318 | 99.4 | 0.0355 |
| -5 | L2 (S1) | 75 | 61 | 26 | 0.633 | 91.7 | 0.0339 | 99.4 | 0.0314 |
| -5 | NK-FP | 85 | 8 | 16 | 0.876 | 91.7 | 0.0230 | 99.4 | 0.0137 |
| +0 | L1 (S1) | 76 | 35 | 25 | 0.717 | 86.2 | 0.0298 | 96.9 | 0.0353 |
| +0 | L2 (S1) | 75 | 41 | 26 | 0.691 | 86.2 | 0.0319 | 96.9 | 0.0312 |
| +0 | NK-FP | 86 | 9 | 15 | 0.878 | 86.2 | 0.0151 | 96.9 | 0.0131 |
| +5 | L1 (S1) | 77 | 29 | 24 | 0.744 | 81.7 | 0.0290 | 82.5 | 0.0360 |
| +5 | L2 (S1) | 76 | 25 | 25 | 0.752 | 81.7 | 0.0311 | 82.5 | 0.0314 |
| +5 | NK-FP | 85 | 10 | 16 | 0.867 | 81.7 | 0.0127 | 82.5 | 0.0114 |
| +6 (main) | L1 (S1) | 78 | 29 | 23 | 0.750 | 78.3 | 0.0291 | 77.0 | 0.0360 |
| +6 (main) | L2 (S1) | 75 | 22 | 26 | 0.758 | 78.3 | 0.0311 | 77.0 | 0.0314 |
| +6 (main) | NK-FP | 84 | 6 | 17 | 0.880 | 78.3 | 0.0125 | 77.0 | 0.0110 |
| +10 | L1 (S1) | 66 | 10 | 35 | 0.746 | 41.9 | 0.0278 | 49.6 | 0.0355 |
| +10 | L2 (S1) | 68 | 10 | 33 | 0.760 | 41.9 | 0.0265 | 49.6 | 0.0305 |
| +10 | NK-FP | 74 | 1 | 27 | 0.841 | 41.9 | 0.0104 | 49.6 | 0.0109 |
| +15 | L1 (S1) | 45 | 1 | 56 | 0.612 | 10.9 | 0.0237 | 12.6 | 0.0326 |
| +15 | L2 (S1) | 43 | 1 | 58 | 0.593 | 10.9 | 0.0247 | 12.6 | 0.0298 |
| +15 | NK-FP | 38 | 0 | 63 | 0.547 | 10.9 | 0.0068 | 12.6 | 0.0151 |
| +20 | L1 (S1) | 3 | 0 | 98 | 0.058 | 2.5 | 0.0160 | 0.1 | 0.0275 |
| +20 | L2 (S1) | 7 | 0 | 94 | 0.130 | 2.5 | 0.0253 | 0.1 | 0.0398 |
| +20 | NK-FP | 11 | 0 | 90 | 0.196 | 2.5 | 0.0042 | 0.1 | 0.0055 |
| +25 | L1 (S1) | 0 | 0 | 101 | 0.000 | 0.0 | nan | 0.0 | nan |
| +25 | L2 (S1) | 0 | 0 | 101 | 0.000 | 0.0 | nan | 0.0 | nan |
| +25 | NK-FP | 0 | 0 | 101 | 0.000 | 0.0 | nan | 0.0 | nan |
<!-- /t10 -->

<!-- fig1 -->
## Fig. 1. Bland–Altman (clean, DMY beats)

Generated 2026-11-21 22:55:54 by `Tasks/final/fin_step7_t10_fig1.py` (VM). Figure: `docs/thesis_figures/Fig1_bland_altman.pdf`. Label = the DMY beat's expert ST deviation; estimate = the 10 s window containing the beat (as phase 1's set B and C §9); masked pairs out. LoA = bias ± 1.96 SD of the difference; slope = OLS of the difference on the mean (negative = compression of large deviations).

| lead | pipeline | n beats | bias (mV) | SD (mV) | 95 % LoA (mV) | slope (diff on mean) | MAE (mV) |
|---|---|--:|--:|--:|---|--:|--:|
| II | L1 (S1) | 1702 | +0.0013 | 0.0465 | [-0.0898, +0.0924] | -0.253 | 0.0310 |
| II | L2 (S1) | 1702 | +0.0002 | 0.0495 | [-0.0968, +0.0971] | -0.337 | 0.0334 |
| II | NK-FP | 1702 | -0.0041 | 0.0182 | [-0.0398, +0.0316] | -0.027 | 0.0134 |
| V2 | L1 (S1) | 1425 | +0.0106 | 0.0399 | [-0.0677, +0.0889] | -0.342 | 0.0324 |
| V2 | L2 (S1) | 1425 | +0.0033 | 0.0368 | [-0.0689, +0.0754] | -0.498 | 0.0295 |
| V2 | NK-FP | 1425 | -0.0040 | 0.0173 | [-0.0380, +0.0299] | +0.109 | 0.0115 |
<!-- /fig1 -->

<!-- t2 -->
## T2. Regression, DMY beats (clean)

Generated 2026-11-22 00:24:16 by `Tasks/final/fin_step7_rest.py` (VM). DMY beats (label = the beat's expert ST deviation); estimate = the 10 s window containing the beat (C §9, phase 1's set B); masked pairs excluded; T1's metrics; coverage = share of DMY beats with a value. L1, L2: mean ± SD over S1–S3.

| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| II | L1 | 100.0 ± 0.0 | 1702 ± 0 | 0.0303 ± 0.0008 | 0.0030 ± 0.0017 | 0.0464 ± 0.0014 | 0.704 ± 0.016 | 0.915 ± 0.006 | 4.4 ± 0.4 | 0.814 ± 0.011 |
| II | L2 | 100.0 ± 0.0 | 1702 ± 0 | 0.0311 ± 0.0020 | 0.0020 ± 0.0019 | 0.0483 ± 0.0015 | 0.664 ± 0.009 | 0.914 ± 0.008 | 4.5 ± 0.3 | 0.799 ± 0.012 |
| II | NK-PR | 83.5 | 1421 | 0.0384 | -0.0347 | 0.0702 | 0.939 | 0.851 | 8.9 | 0.550 |
| II | NK-FP | 100.0 | 1702 | 0.0134 | -0.0041 | 0.0187 | 0.959 | 0.986 | 0.0 | 0.970 |
| V2 | L1 | 100.0 ± 0.0 | 1425 ± 0 | 0.0322 ± 0.0003 | 0.0101 ± 0.0005 | 0.0410 ± 0.0002 | 0.445 ± 0.008 | 0.578 ± 0.005 | 1.8 ± 0.3 | 0.252 ± 0.009 |
| V2 | L2 | 100.0 ± 0.0 | 1425 ± 0 | 0.0283 ± 0.0010 | 0.0037 ± 0.0004 | 0.0360 ± 0.0011 | 0.430 ± 0.030 | 0.656 ± 0.028 | 1.0 ± 0.5 | 0.424 ± 0.036 |
| V2 | NK-PR | 75.2 | 1071 | 0.0450 | -0.0419 | 0.0582 | 0.732 | 0.675 | 8.7 | -0.483 |
| V2 | NK-FP | 100.0 | 1425 | 0.0115 | -0.0040 | 0.0178 | 1.052 | 0.946 | 0.1 | 0.859 |

DMY beats per lead (masked pairs out): II 1,702, V2 1,425.
<!-- /t2 -->

<!-- t3 -->
## T3. Error by true ST band (clean)

Generated 2026-11-22 00:24:59 by `Tasks/final/fin_step7_rest.py` (VM). Set A windows (as T1), masked pairs excluded, banded by |label| (lower bound inclusive). Bias = estimate − label; shrinkage = mean of −sign(label) × (estimate − label), positive when the estimate is closer to 0 than the label. Bands partition the labelled windows: yes. L1, L2: mean ± SD over S1–S3.

| lead | \|ST\| band (mV) | windows (% of lead) | pipeline | n with value | MAE (mV) | bias (mV) | shrinkage (mV) |
|---|---|--:|---|--:|--:|--:|--:|
| II | < 0.05 | 46,966 (42.4 %) | L1 | 46966 ± 0 | 0.0147 ± 0.0004 | -0.0019 ± 0.0006 | -0.0009 ± 0.0006 |
|  |  |  | L2 | 46966 ± 0 | 0.0137 ± 0.0020 | -0.0068 ± 0.0036 | 0.0013 ± 0.0003 |
|  |  |  | NK-PR | 37724 | 0.0234 | -0.0204 | -0.0042 |
|  |  |  | NK-FP | 46966 | 0.0106 | -0.0017 | 0.0014 |
| II | 0.05–0.10 | 35,237 (31.8 %) | L1 | 35237 ± 0 | 0.0259 ± 0.0009 | -0.0131 ± 0.0010 | 0.0069 ± 0.0010 |
|  |  |  | L2 | 35237 ± 0 | 0.0259 ± 0.0023 | -0.0160 ± 0.0032 | 0.0086 ± 0.0020 |
|  |  |  | NK-PR | 28225 | 0.0355 | -0.0312 | -0.0128 |
|  |  |  | NK-FP | 35237 | 0.0152 | -0.0091 | 0.0072 |
| II | 0.10–0.20 | 19,252 (17.4 %) | L1 | 19252 ± 0 | 0.0323 ± 0.0042 | 0.0086 ± 0.0020 | 0.0193 ± 0.0024 |
|  |  |  | L2 | 19252 ± 0 | 0.0312 ± 0.0005 | 0.0112 ± 0.0021 | 0.0207 ± 0.0014 |
|  |  |  | NK-PR | 17114 | 0.0495 | -0.0459 | -0.0292 |
|  |  |  | NK-FP | 19252 | 0.0131 | -0.0029 | 0.0032 |
| II | ≥ 0.20 | 9,328 (8.4 %) | L1 | 9328 ± 0 | 0.1033 ± 0.0075 | 0.0807 ± 0.0093 | 0.1026 ± 0.0076 |
|  |  |  | L2 | 9328 ± 0 | 0.1128 ± 0.0032 | 0.0953 ± 0.0029 | 0.1115 ± 0.0030 |
|  |  |  | NK-PR | 6309 | 0.0577 | -0.0524 | 0.0202 |
|  |  |  | NK-FP | 9328 | 0.0128 | 0.0031 | 0.0053 |
| V2 | < 0.05 | 17,948 (19.0 %) | L1 | 17948 ± 0 | 0.0416 ± 0.0012 | 0.0406 ± 0.0013 | -0.0148 ± 0.0008 |
|  |  |  | L2 | 17948 ± 0 | 0.0396 ± 0.0019 | 0.0379 ± 0.0034 | -0.0155 ± 0.0021 |
|  |  |  | NK-PR | 13461 | 0.0257 | -0.0221 | 0.0181 |
|  |  |  | NK-FP | 17948 | 0.0190 | -0.0149 | 0.0066 |
| V2 | 0.05–0.10 | 38,283 (40.5 %) | L1 | 38283 ± 0 | 0.0275 ± 0.0005 | 0.0196 ± 0.0008 | -0.0194 ± 0.0008 |
|  |  |  | L2 | 38283 ± 0 | 0.0202 ± 0.0013 | 0.0132 ± 0.0012 | -0.0130 ± 0.0011 |
|  |  |  | NK-PR | 27562 | 0.0396 | -0.0360 | 0.0358 |
|  |  |  | NK-FP | 38283 | 0.0075 | -0.0014 | 0.0014 |
| V2 | 0.10–0.20 | 35,661 (37.7 %) | L1 | 35661 ± 0 | 0.0359 ± 0.0014 | -0.0134 ± 0.0014 | 0.0135 ± 0.0014 |
|  |  |  | L2 | 35661 ± 0 | 0.0293 ± 0.0015 | -0.0220 ± 0.0016 | 0.0220 ± 0.0016 |
|  |  |  | NK-PR | 27542 | 0.0528 | -0.0515 | 0.0515 |
|  |  |  | NK-FP | 35661 | 0.0096 | -0.0007 | 0.0008 |
| V2 | ≥ 0.20 | 2,583 (2.7 %) | L1 | 2583 ± 0 | 0.1049 ± 0.0040 | -0.1045 ± 0.0040 | 0.1049 ± 0.0040 |
|  |  |  | L2 | 2583 ± 0 | 0.0834 ± 0.0030 | -0.0831 ± 0.0030 | 0.0834 ± 0.0030 |
|  |  |  | NK-PR | 2318 | 0.1027 | -0.1017 | 0.1018 |
|  |  |  | NK-FP | 2583 | 0.0224 | -0.0215 | 0.0217 |
<!-- /t3 -->

<!-- t4 -->
## T4. Noise robustness (NSTDB `em`, seed 42)

Generated 2026-11-22 00:25:19 by `Tasks/final/fin_step7_rest.py` (VM). Noisy estimates against the clean labels, set A windows, masked pairs excluded. Gated = windows the 6 dB SQI gate (rodrigues2021 peaks, at the learned grid anchor) keeps, the same windows for every pipeline (as T7). Bursts = windows entirely inside a burst of the seed-42 contamination log. n = windows with a value. L1, L2: mean ± SD over S1–S3.

| lead | pipeline | ungated MAE (mV) | n | ungated burst MAE (mV) | n | gated MAE (mV) | n | gated burst MAE (mV) | n |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| II | L1 | 0.0349 ± 0.0003 | 110,783 | 0.0729 ± 0.0012 | 14,754 | 0.0288 ± 0.0004 | 86,707 | 0.0613 ± 0.0029 | 11 |
| II | L2 | 0.0352 ± 0.0015 | 110,783 | 0.0742 ± 0.0006 | 14,754 | 0.0292 ± 0.0018 | 86,707 | 0.0607 ± 0.0023 | 11 |
| II | NK-PR | 0.0666 | 91,875 | 0.2339 | 14,202 | 0.0341 | 69,532 | 0.1238 | 8 |
| II | NK-FP | 0.0395 | 110,783 | 0.2076 | 14,754 | 0.0125 | 86,707 | 0.0553 | 11 |
| V2 | L1 | 0.0354 ± 0.0002 | 94,475 | 0.0349 ± 0.0007 | 12,586 | 0.0359 ± 0.0003 | 72,704 | 0.0417 ± 0.0014 | 1,801 |
| V2 | L2 | 0.0304 ± 0.0009 | 94,475 | 0.0391 ± 0.0012 | 12,586 | 0.0303 ± 0.0010 | 72,704 | 0.0571 ± 0.0021 | 1,801 |
| V2 | NK-PR | 0.0462 | 73,823 | 0.0564 | 12,147 | 0.0439 | 53,999 | 0.0701 | 1,620 |
| V2 | NK-FP | 0.0140 | 94,475 | 0.0333 | 12,586 | 0.0110 | 72,704 | 0.0297 | 1,801 |

Labelled windows: II 110,783 (bursts 14,754), gate keeps 86,707 (bursts 11); V2 94,475 (bursts 12,586), gate keeps 72,704 (bursts 1,801).
<!-- /t4 -->

<!-- t6 -->
## T6. Alarm robustness across noise seeds (gated)

Generated 2026-11-22 00:26:08 by `Tasks/final/fin_step7_rest.py` (VM). NSTDB `em` seeds 42, 43, 44, each with its own noisy signal, predictions, NeuroKit values and SQI gate (rodrigues2021 peaks, 6 dB). Union, FULL scope. Mean ± SD (ddof 1) over the three noise seeds; L1, L2: training seed S1. Check: seed 42 equals the tier-1 gated rows (4 / 4).

| pipeline | TP | FP | FN | F1 | false-alarm h | per seed 42 / 43 / 44: TP / FP / FN |
|---|--:|--:|--:|--:|--:|---|
| L1 (S1) | 75.7 ± 2.1 | 29.0 ± 3.0 | 25.3 ± 2.1 | 0.736 ± 0.018 | 78.4 ± 3.3 | 78/29/23; 74/32/27; 75/26/26 |
| L2 (S1) | 75.3 ± 0.6 | 20.7 ± 1.2 | 25.7 ± 0.6 | 0.765 ± 0.007 | 70.1 ± 1.9 | 75/22/26; 76/20/25; 75/20/26 |
| NK-PR | 68.3 ± 2.1 | 21.0 ± 0.0 | 32.7 ± 2.1 | 0.718 ± 0.014 | 33.5 ± 1.7 | 66/21/35; 69/21/32; 70/21/31 |
| NK-FP | 84.0 ± 1.0 | 7.0 ± 1.0 | 17.0 ± 1.0 | 0.875 ± 0.005 | 21.6 ± 3.0 | 84/6/17; 85/8/16; 83/7/18 |
<!-- /t6 -->

<!-- data -->
## Data

Generated 2026-11-22 01:32:48 by `Tasks/final/fin_step7_data_ceiling.py` (VM). Pre-training: Unified v2 (`Unified_Dataset_No_PTB_Segment_v2`, 58,942 10 s windows at 100 Hz from Chapman and CinC 2020, md5 `ac56fd0e…`); rows where a selected lead is blank are removed. Fine-tuning: the fold dataset CSVs (`phase1_folds/`), one split for every learned model; a window is a 10 s segment of the median LTST signal with both targets (every row has both). Masked record–lead pairs are in the training rows (their targets are present) and are excluded from scoring; gold-standard episodes: 69 lead II + 32 lead V2 = 101, none on a masked pair.

### Pre-training set

| backbone | leads | used by | rows | train | validation | Chapman | CinC 2020 |
|---|---|---|--:|--:|--:|--:|--:|
| `patch_ecg_unified_ii_lossfix` | II | L1 lead II model | 58,942 | 47,154 | 11,788 | 40,507 | 18,435 |
| `patch_ecg_unified_v2_lossfix` | V2 | L1 lead V2 model | 57,843 | 46,274 | 11,569 | 39,415 | 18,428 |
| `patch_ecg_unified_ii_v2_rebuilt` | II + V2 | L2 | 57,843 | 46,274 | 11,569 | 39,415 | 18,428 |

The two-lead backbone's counts are from a read-only check on Narval (2026-11-22: its `pretrain_split.csv` and `data_source.txt`, `_best.pt` sha256 `f3554f14…`; the same check reproduced the two single-lead rows above). It holds the same 57,843 windows as the V2 backbone (lead II is never blank, so the II + V2 filter equals the V2 filter), with its own random 80/20 train / validation draw: the totals per split and per source are equal, the assignment differs (39,179 of 57,843 windows, 68 %, share a split; Narval's count).

### Fine-tuning folds

| fold | dataset | split | records | windows | II windows (no masked) | V2 windows (no masked) | II episodes | V2 episodes | record IDs (masked: II / V2) |
|--:|---|---|--:|--:|--:|--:|--:|--:|---|
| 1 | `cv5_optB_leadV2_opt2p_fold01` | train | 10 | 78,797 | 63,201 | 62,892 | 42 | 20 | s20011 s20061 s20071 s20091 s20111 s20121 s20131 s20141 s20201 s20241 (masked: s20091 s20121 / s20121 s20241) |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | validation | 2 | 15,609 | 15,609 | 15,609 | 12 | 4 | s20101 s20231 |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | test | 3 | 23,754 | 23,754 | 15,959 | 10 | 8 | s20051 s20211 s20221 (masked: – / s20221) |
| 1 | `cv5_optB_leadV2_opt2p_fold01` | unused | 1 | 8,204 | 8,204 | 0 | 5 | 0 | s20081 (masked: – / s20081) |
| 2 | `cv5_optB_fold02` | train | 11 | 86,485 | 70,889 | 62,785 | 47 | 23 | s20051 s20061 s20071 s20091 s20111 s20121 s20131 s20201 s20211 s20221 s20241 (masked: s20091 s20121 / s20121 s20221 s20241) |
| 2 | `cv5_optB_fold02` | validation | 2 | 16,636 | 16,636 | 16,636 | 6 | 6 | s20011 s20231 |
| 2 | `cv5_optB_fold02` | test | 3 | 23,243 | 23,243 | 15,039 | 16 | 3 | s20081 s20101 s20141 (masked: – / s20081) |
| 3 | `cv5_optB_fold03` | train | 11 | 85,631 | 78,130 | 62,131 | 56 | 24 | s20051 s20071 s20081 s20101 s20111 s20121 s20131 s20201 s20211 s20221 s20231 (masked: s20121 / s20081 s20121 s20221) |
| 3 | `cv5_optB_fold03` | validation | 2 | 16,066 | 16,066 | 16,066 | 5 | 5 | s20011 s20141 |
| 3 | `cv5_optB_fold03` | test | 3 | 24,667 | 16,572 | 16,263 | 8 | 3 | s20061 s20091 s20241 (masked: s20091 / s20241) |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | train | 10 | 78,565 | 62,969 | 54,865 | 40 | 18 | s20011 s20051 s20091 s20101 s20121 s20141 s20201 s20211 s20221 s20241 (masked: s20091 s20121 / s20121 s20221 s20241) |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | validation | 2 | 15,751 | 15,751 | 15,751 | 7 | 5 | s20061 s20071 |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | test | 3 | 23,844 | 23,844 | 23,844 | 17 | 9 | s20111 s20131 s20231 |
| 4 | `cv5_optB_leadV2_opt2p_fold04` | unused | 1 | 8,204 | 8,204 | 0 | 5 | 0 | s20081 (masked: – / s20081) |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | train | 10 | 79,310 | 63,714 | 55,810 | 40 | 20 | s20051 s20061 s20081 s20091 s20101 s20111 s20121 s20211 s20221 s20231 (masked: s20091 s20121 / s20081 s20121 s20221) |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | validation | 2 | 15,295 | 15,295 | 15,295 | 9 | 3 | s20131 s20141 |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | test | 3 | 23,355 | 23,355 | 23,355 | 18 | 9 | s20011 s20071 s20201 |
| 5 | `cv5_optB_leadV2_opt2p_fold05` | unused | 1 | 8,404 | 8,404 | 0 | 2 | 0 | s20241 (masked: – / s20241) |

Checks: test records over the 5 folds = the 15 scored records, each once: yes; fold 1: episodes over splits = 69 II / 32 V2: yes; fold 1: 16 records over splits: yes; fold 2: episodes over splits = 69 II / 32 V2: yes; fold 2: 16 records over splits: yes; fold 3: episodes over splits = 69 II / 32 V2: yes; fold 3: 16 records over splits: yes; fold 4: episodes over splits = 69 II / 32 V2: yes; fold 4: 16 records over splits: yes; fold 5: episodes over splits = 69 II / 32 V2: yes; fold 5: 16 records over splits: yes; pre-training total rows in the cluster report = 58,942 (Unified v2): yes.
Note: the CSVs hold the windows the trainer used (step 6 reproduced every saved test MSE on them); the prediction grid of the trainer's H5 has 263 more 10 s segments over the 15 records (119,126 vs 118,863), which have no CSV row. The scored set A grid (T1) is a separate 10 s grid from sample 0.
<!-- /data -->
<!-- ceiling -->
## Gold-standard ceiling (baseline)

Generated 2026-11-22 01:22:12 by `Tasks/final/fin_step7_data_ceiling.py` (VM). The true window labels (set A: median `.16a` `ee` of each 10 s grid window, anchors 2500k + 2499) through the same FSM and scorer as T5, baseline condition (no noise, no gate), masked pairs excluded, union of II and V2. It is the score of a perfect window-level estimator under this FSM. Control: equals phase 1's oracle union (yes).

| pipeline | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | duration Se | duration +P | redundant |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gold-standard labels | 100 | 9 | 1 | 0.917 | 0.990 | 0.952 | 0.920 | 16.3 | 0.997 | 0.923 | 3 |
| NK-PR (T5) | 71.0 | 26.0 | 30.0 | 0.732 | 0.703 | 0.717 | 0.445 | 34.3 | 0.524 | 0.749 | 10.0 |
| NK-FP (T5) | 87.0 | 10.0 | 14.0 | 0.897 | 0.861 | 0.879 | 0.849 | 15.3 | 0.916 | 0.921 | 11.0 |
| L1 (T5) | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 0.552 ± 0.030 | 65.6 ± 6.6 | 0.738 ± 0.022 | 0.687 ± 0.028 | 18.0 ± 2.6 |
| L2 (T5) | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 0.546 ± 0.033 | 53.3 ± 13.8 | 0.694 ± 0.034 | 0.720 ± 0.051 | 20.3 ± 0.6 |

With the true labels as input, the 1 missed episode and 9 false alarms reflect differences between the FSM on 10 s window medians and the way the gold-standard episodes were built; their cause was not examined.
<!-- /ceiling -->

<!-- ceiling_inputs -->
### Ceiling note: how each system feeds the FSM (written by hand, 2026-11-22)

Checked in the code (`Tasks/final/fin_step5_nk.py`, `Tasks/C_neurokit/c13_fp_gap.segments`, `c20b_paper_rule.paper_beats`,
`ltst_pipeline/old_fsm_gs.run_fsm_on_record`). Every final system feeds the FSM **one value per 10 s window, never per beat**, on the same grid
(windows [2500k, 2500k + 2500)), so the ceiling applies to all four.

| | Fiducial-PR (NK-PR) | Fiducial-FP (NK-FP) | L1, L2 | Ceiling (gold-standard labels) |
|---|---|---|---|---|
| per-beat value | median of 3 samples (baseline) / 5 (artifact) at J+60 minus the median of the PR segment (P offset to R onset); rodrigues2021 delineation | 5-sample mean at R + b + 60 ms minus the 5-sample mean at R − a; rodrigues2021 R-peaks | – (one prediction per window) | `.16a` `ee` of each annotated beat |
| beat → window | J+60 sample | ST point (R + b + 60 ms) | – | J+60 sample |
| window value | median | median | model output | median (rounded to 4 decimals) |
| minimum beats | ≥ 5 usable beats (finite ST and baseline), else no value | ≥ 5 usable beats, else no value | – | ≥ 1 beat, else no value |
| FSM anchor | 2500k + 2500 | 2500k + 2500 | 2500k + 2499 | 2500k + 2499 |

- A window with no value is skipped by the FSM: its state is frozen and the skipped time does not count toward warm-up durations (the same for
  gated windows). The ceiling's label is the learned models' training target (phase 1's oracle control).
- What Ch4 should state for the two fiducial systems: (1) they need ≥ 5 usable detected beats per window, the ceiling ≥ 1 annotated beat; on the
  baseline signal, Fiducial-PR has a value on 80.7 % (II) / 75.0 % (V2) of labelled windows, Fiducial-FP on 100 % (T1); (2) they use detected beats
  (rodrigues2021), not annotated ones, and Fiducial-FP places beats by its ST point, not J+60; (3) their anchor is one sample (4 ms) later, which does
  not change the window a beat belongs to.
- Not measured: how much the ≥ 5-beat rule alone would change the ceiling (it would need a ceiling run with ≥ 5 annotated beats per window).
<!-- /ceiling_inputs -->
