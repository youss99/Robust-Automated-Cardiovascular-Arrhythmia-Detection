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
