# step3e results: end-to-end supervised baselines (VM, 2026-11-24)

The comparisons and metrics were fixed in `DECISIONS.md` (2026-11-24) before any step3e result existed.
- **Training:** Narval report `2026-11-24_step3e_training.md`.
- **Copy-back:** 450 of 450 files match `results/SHA256SUMS_step3e_copyback.txt` (list sha256 `14af1730…`). The 912 files of the existing final runs
  still match their list.
- **Pipelines:**
  - **L1supE2E / L2supE2E** = supervised, 0 head-only + 90 full-network epochs (the new runs);
  - **L1sup / L2sup** = their twins (20 + 70);
  - **L1 / L2** = pretrained.

  Thesis names: Learned-1L/2L (supervised-only, end-to-end), Learned-1L/2L (supervised-only), Learned-1L/2L (pretrained).

## Inference (step 6, `FIN_STEP6_SET=step3e`)

`fin_step6_infer.py` has a new switch. With it, only the 45 step3e runs are processed, with their own markers and check files (`*_step3e`);
the 90-run outputs are not touched. Conditions: baseline and artifact seed 42 (the conditions the fixed comparisons use). Checks:
- **G1:** predictions at each trainer's test rows reproduce the saved test MSE: 45 of 45 (worst 0.052 %); dataset CSV sha256 45 of 45.
- **G2:** window counts equal the published runs (119,234 grid windows, 119,126 training-grid windows, 15 records).
- **G3:** every checkpoint sha256 is in the step3e copy-back list: 45 of 45.

## Scoring (`fin_step7_step3e.py`)

The scorer uses `fin_step7_tier1.py`'s own functions: the alarm path, the gate (rodrigues2021, 6 dB), the scorer, the set A labels and the
regression metrics. The output is in `Tasks/final/runs/step7_step3e/`.
- **Controls, run first; all pass.** This path reproduces the stored tier-1 results of L1, L2, L1sup and L2sup exactly:
  - the T5 rows, every seed and condition;
  - the T1 rows;
  - the T7 rows L1 − L1sup and L2 − L2sup.
- **Independent check:** the per-record alarm components sum to every T5 row (54 of 54, 15 records each), and the T7 point values equal the
  seed-mean F1.

## T1: regression, baseline, set A (mean ± SD over S1–S3)

| lead | pipeline | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | R² |
|---|---|--:|--:|--:|--:|--:|--:|
| II | L1 | 0.0288 ± 0.0003 | 0.0033 ± 0.0010 | 0.0429 ± 0.0005 | 0.702 ± 0.016 | 0.928 ± 0.002 | 0.830 ± 0.004 |
| II | L1sup | 0.0366 ± 0.0007 | 0.0070 ± 0.0008 | 0.0573 ± 0.0014 | 0.603 ± 0.014 | 0.849 ± 0.013 | 0.696 ± 0.015 |
| II | L1supE2E | 0.0365 ± 0.0015 | 0.0068 ± 0.0028 | 0.0556 ± 0.0017 | 0.626 ± 0.011 | 0.858 ± 0.014 | 0.714 ± 0.018 |
| II | L2 | 0.0290 ± 0.0018 | 0.0020 ± 0.0023 | 0.0444 ± 0.0016 | 0.661 ± 0.011 | 0.931 ± 0.008 | 0.817 ± 0.013 |
| II | L2sup | 0.0358 ± 0.0009 | 0.0020 ± 0.0009 | 0.0585 ± 0.0034 | 0.567 ± 0.018 | 0.846 ± 0.028 | 0.684 ± 0.037 |
| II | L2supE2E | 0.0361 ± 0.0010 | 0.0034 ± 0.0017 | 0.0586 ± 0.0027 | 0.560 ± 0.014 | 0.848 ± 0.023 | 0.682 ± 0.029 |
| V2 | L1 | 0.0355 ± 0.0003 | 0.0077 ± 0.0006 | 0.0444 ± 0.0002 | 0.374 ± 0.012 | 0.502 ± 0.006 | 0.168 ± 0.006 |
| V2 | L1sup | 0.0331 ± 0.0003 | 0.0046 ± 0.0004 | 0.0425 ± 0.0004 | 0.405 ± 0.017 | 0.540 ± 0.001 | 0.238 ± 0.016 |
| V2 | L1supE2E | 0.0336 ± 0.0013 | 0.0050 ± 0.0021 | 0.0426 ± 0.0009 | 0.429 ± 0.028 | 0.550 ± 0.006 | 0.236 ± 0.031 |
| V2 | L2 | 0.0290 ± 0.0011 | 0.0020 ± 0.0007 | 0.0368 ± 0.0011 | 0.389 ± 0.037 | 0.659 ± 0.027 | 0.427 ± 0.036 |
| V2 | L2sup | 0.0349 ± 0.0006 | -0.0010 ± 0.0014 | 0.0441 ± 0.0007 | 0.336 ± 0.018 | 0.480 ± 0.021 | 0.181 ± 0.027 |
| V2 | L2supE2E | 0.0349 ± 0.0016 | 0.0004 ± 0.0035 | 0.0436 ± 0.0019 | 0.358 ± 0.035 | 0.498 ± 0.046 | 0.197 ± 0.068 |

## T5: alarms, union, FULL scope (clean = baseline, noisy = artifact seed 42)

| condition | pipeline | TP | FP | FN | precision | recall | F1 | false-alarm h |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| clean | L1 | 74.7 ± 0.6 | 37.7 ± 3.2 | 26.3 ± 0.6 | 0.665 ± 0.018 | 0.739 ± 0.006 | 0.700 ± 0.007 | 65.6 ± 6.6 |
| clean | L1sup | 65.7 ± 1.5 | 50.3 ± 3.1 | 35.3 ± 1.5 | 0.566 ± 0.019 | 0.650 ± 0.015 | 0.605 ± 0.016 | 59.9 ± 1.1 |
| clean | L1supE2E | 63.7 ± 4.0 | 45.0 ± 4.6 | 37.3 ± 4.0 | 0.586 ± 0.035 | 0.630 ± 0.040 | 0.607 ± 0.035 | 53.2 ± 9.7 |
| clean | L2 | 77.3 ± 1.5 | 29.7 ± 2.3 | 23.7 ± 1.5 | 0.723 ± 0.016 | 0.766 ± 0.015 | 0.744 ± 0.011 | 53.3 ± 13.8 |
| clean | L2sup | 65.3 ± 1.2 | 32.3 ± 4.5 | 35.7 ± 1.2 | 0.670 ± 0.031 | 0.647 ± 0.011 | 0.658 ± 0.017 | 48.7 ± 3.4 |
| clean | L2supE2E | 64.3 ± 2.5 | 30.3 ± 7.0 | 36.7 ± 2.5 | 0.682 ± 0.055 | 0.637 ± 0.025 | 0.658 ± 0.033 | 50.9 ± 5.7 |
| noisy | L1 | 73.0 ± 1.0 | 46.3 ± 2.5 | 28.0 ± 1.0 | 0.612 ± 0.010 | 0.723 ± 0.010 | 0.663 ± 0.002 | 49.1 ± 3.5 |
| noisy | L1sup | 62.3 ± 1.5 | 69.7 ± 2.3 | 38.7 ± 1.5 | 0.472 ± 0.009 | 0.617 ± 0.015 | 0.535 ± 0.010 | 40.7 ± 5.2 |
| noisy | L1supE2E | 61.7 ± 4.7 | 62.0 ± 5.0 | 39.3 ± 4.7 | 0.499 ± 0.030 | 0.611 ± 0.047 | 0.549 ± 0.035 | 38.9 ± 3.9 |
| noisy | L2 | 73.3 ± 2.1 | 66.7 ± 18.1 | 27.7 ± 2.1 | 0.530 ± 0.061 | 0.726 ± 0.021 | 0.610 ± 0.034 | 25.6 ± 11.4 |
| noisy | L2sup | 62.7 ± 1.5 | 50.0 ± 14.7 | 38.3 ± 1.5 | 0.562 ± 0.064 | 0.620 ± 0.015 | 0.588 ± 0.032 | 41.2 ± 2.7 |
| noisy | L2supE2E | 62.3 ± 3.2 | 45.7 ± 2.5 | 38.7 ± 3.2 | 0.577 ± 0.019 | 0.617 ± 0.032 | 0.596 ± 0.023 | 39.6 ± 3.7 |
| gated | L1 | 75.3 ± 2.3 | 24.0 ± 4.6 | 25.7 ± 2.3 | 0.760 ± 0.029 | 0.746 ± 0.023 | 0.752 ± 0.006 | 72.9 ± 5.5 |
| gated | L1sup | 67.0 ± 1.0 | 25.3 ± 4.5 | 34.0 ± 1.0 | 0.727 ± 0.034 | 0.663 ± 0.010 | 0.693 ± 0.014 | 69.4 ± 0.9 |
| gated | L1supE2E | 64.0 ± 4.4 | 21.7 ± 1.5 | 37.0 ± 4.4 | 0.747 ± 0.026 | 0.634 ± 0.043 | 0.685 ± 0.036 | 61.3 ± 4.5 |
| gated | L2 | 75.7 ± 1.2 | 20.0 ± 3.5 | 25.3 ± 1.2 | 0.791 ± 0.032 | 0.749 ± 0.011 | 0.770 ± 0.021 | 65.1 ± 5.6 |
| gated | L2sup | 64.7 ± 2.1 | 19.7 ± 4.0 | 36.3 ± 2.1 | 0.768 ± 0.032 | 0.640 ± 0.021 | 0.698 ± 0.009 | 58.0 ± 4.4 |
| gated | L2supE2E | 63.3 ± 4.0 | 17.0 ± 3.0 | 37.7 ± 4.0 | 0.789 ± 0.021 | 0.627 ± 0.040 | 0.698 ± 0.019 | 60.0 ± 4.8 |

## T7: paired bootstrap, gated, seed 42 (* = 95 % CI excludes 0)

| comparison | metric | A | B | A − B | 95 % CI |
|---|---|--:|--:|--:|---|
| L1supE2E - L1sup | dF1 (union) | 0.685 | 0.693 | -0.008 | [-0.035, +0.020] |
| L1supE2E - L1sup | dFP (union) | 21.7 | 25.3 | -3.7 | [-8.7, +0.3] |
| L1supE2E - L1sup | dMAE II (mV) | 0.0367 | 0.0368 | -0.0001 | [-0.0021, +0.0016] |
| L1supE2E - L1sup | dMAE V2 (mV) | 0.0338 | 0.0332 | +0.0006 | [-0.0011, +0.0033] |
| L2supE2E - L2sup | dF1 (union) | 0.698 | 0.698 | +0.000 | [-0.020, +0.020] |
| L2supE2E - L2sup | dFP (union) | 17.0 | 19.7 | -2.7 | [-7.0, +1.0] |
| L2supE2E - L2sup | dMAE II (mV) | 0.0366 | 0.0363 | +0.0003 | [-0.0014, +0.0023] |
| L2supE2E - L2sup | dMAE V2 (mV) | 0.0357 | 0.0357 | -0.0000 | [-0.0012, +0.0014] |
| L1 - L1supE2E | dF1 (union) | 0.752 | 0.685 | +0.067* | [+0.018, +0.130] |
| L1 - L1supE2E | dFP (union) | 24.0 | 21.7 | +2.3 | [-2.3, +7.0] |
| L1 - L1supE2E | dMAE II (mV) | 0.0288 | 0.0367 | -0.0079* | [-0.0125, -0.0040] |
| L1 - L1supE2E | dMAE V2 (mV) | 0.0359 | 0.0338 | +0.0021 | [-0.0025, +0.0085] |
| L2 - L2supE2E | dF1 (union) | 0.770 | 0.698 | +0.072* | [+0.001, +0.148] |
| L2 - L2supE2E | dFP (union) | 20.0 | 17.0 | +3.0 | [-4.7, +11.7] |
| L2 - L2supE2E | dMAE II (mV) | 0.0292 | 0.0366 | -0.0074* | [-0.0144, -0.0024] |
| L2 - L2supE2E | dMAE V2 (mV) | 0.0303 | 0.0357 | -0.0054* | [-0.0100, -0.0007] |
| L1 - L1sup | dF1 (union) | 0.752 | 0.693 | +0.059* | [+0.013, +0.117] |
| L1 - L1sup | dFP (union) | 24.0 | 25.3 | -1.3 | [-10.3, +6.0] |
| L1 - L1sup | dMAE II (mV) | 0.0288 | 0.0368 | -0.0080* | [-0.0141, -0.0031] |
| L1 - L1sup | dMAE V2 (mV) | 0.0359 | 0.0332 | +0.0027 | [-0.0030, +0.0100] |
| L2 - L2sup | dF1 (union) | 0.770 | 0.698 | +0.072* | [+0.010, +0.141] |
| L2 - L2sup | dFP (union) | 20.0 | 19.7 | +0.3 | [-7.3, +7.3] |
| L2 - L2sup | dMAE II (mV) | 0.0292 | 0.0363 | -0.0071* | [-0.0143, -0.0016] |
| L2 - L2sup | dMAE V2 (mV) | 0.0303 | 0.0357 | -0.0054* | [-0.0101, -0.0009] |

## Training: best epoch and validation loss (job_log.txt, 15 runs per row)

| pipeline | mode | best epoch median (range) | best valid loss median |
|---|---|---|--:|
| L1supE2E | lead_II | 36 (11–81) | 0.00089 |
| L1sup | lead_II | 53 (39–83) | 0.00083 |
| L1supE2E | lead_V2 | 59 (9–84) | 0.00116 |
| L1sup | lead_V2 | 62 (30–85) | 0.00120 |
| L2supE2E | two_lead | 57 (16–87) | 0.00079 |
| L2sup | two_lead | 73 (35–82) | 0.00080 |

## Reading (the comparisons fixed in DECISIONS.md)

- **(a) Schedule effect: removing the head-only phase changes nothing measurable.**
  - Gated, ΔF1 is −0.008 [−0.035, +0.020] for L1 and +0.000 [−0.020, +0.020] for L2.
  - ΔMAE is within ±0.0006 mV, with every CI including 0.
  - ΔFP is −3.7 for L1 and −2.7 for L2. Both lean towards fewer false alarms, but both CIs include 0.
  - Baseline and artifact F1 are also within one seed SD (baseline: 0.607 vs 0.605 for L1, 0.658 vs 0.658 for L2).
  - The end-to-end runs vary more between seeds (for example gated TP SD 4.4 vs 1.0 for L1) and select earlier checkpoints (median best epoch
    36 / 59 / 57 vs 53 / 62 / 73). Their best validation loss is similar (Narval report: median ratio 0.99).
- **(b) Pretraining effect, against both supervised arms: the same size.**
  - Gated ΔF1: L1 − L1supE2E +0.067 [+0.018, +0.130] next to L1 − L1sup +0.059 [+0.013, +0.117]; L2 − L2supE2E +0.072 [+0.001, +0.148]
    next to L2 − L2sup +0.072 [+0.010, +0.141]. All four CIs exclude 0.
  - ΔMAE II: −0.0079 and −0.0074 mV against the end-to-end arm, −0.0080 and −0.0071 against the twins; all four exclude 0.
  - ΔMAE V2: L2 −0.0054 mV against both (CI excludes 0); L1 +0.0021 / +0.0027, CI includes 0.
  - ΔFP: all four CIs include 0.
  - So the advantage of pretraining reported in T7 does not come from the supervised models' 20 head-only epochs on a random encoder. One CI
    lower bound is close to 0 (L2 − L2supE2E ΔF1, +0.001).
- **Known limits** (`DECISIONS.md`):
  - Comparison (a) also changes the number of full-network epochs (90 vs 70).
  - In comparison (b) the pretrained arm keeps layer decay 0.65, no weight decay on biases, its 20 head-only epochs and a single unseeded
    pre-training run per backbone.
  - The learning rate of 1e-3 was not tuned for training from scratch. The end-to-end runs reach their best validation loss early and then
    overfit (Narval report); selection on validation loss handles this as specified. Nothing was retrained or tuned after these results.
