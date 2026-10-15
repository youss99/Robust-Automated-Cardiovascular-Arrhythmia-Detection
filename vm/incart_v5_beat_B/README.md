# incart_v5_beat_B: INCART central-beat classifier (lead V5)

Classifies the beat at the centre of a 10 s lead-V5 window as **background**, **PAC** or **PVC**.
PatchECG Small backbone (`patch_ecg_unified_v5_fixed_best.pt`, pretrained on Unified-fixed: Chapman, CPSC 2018,
CPSC-extra, Georgia; no INCART) fine-tuned on INCART with the thesis recipe (Table A.2: 20 head + 70 full epochs,
lr 1e-3, layer decay 0.65, weight decay 0.05, batch 128). Run B: per-window RevIN normalization
(`instance`, batch-independent), ce loss, selected epoch 28 on mean PAC/PVC
validation AUPRC. Patient split: val [3, 6, 16, 24, 31], test [2, 10, 17, 22, 27]; all other INCART
patients train.

## Run on a patient
Needs: torch, numpy, pandas, scipy, einops, neurokit2. From this folder:
```bash
python -m src.incart_beat.predict_patient --model incart_v5_beat_B.pt \
    --csv /path/to/aligned_leads_8103_8123_first5h.csv --lead_col V5 --fs 250 --units mV \
    --out patient_beats_V5.csv            # add --beats beats.csv to use your own beat positions
```
Give the lead's **physical units** (`--units mV` for the Homer CSVs); conversion to uV (what the backbone was
pretrained on) and resampling to 100 Hz happen inside, once. Feed only lead V5 to this model.

Output: one row per beat: `sample`, `time_s`, `full_window`, `p_background`, `p_PAC`, `p_PVC`, `pred` (argmax).
Beats within 5 s of either end have no full window (NaN). Without `--beats`, beats are NeuroKit2 R-peaks on the
same lead; detector misses or double detections become missing or extra rows.

## How it did on INCART (window-level, natural class frequencies)
Test (patients 2, 10, 17, 22, 27):
| class | n_true | precision | recall | f1 | auprc | auroc |
|---|---|---|---|---|---|---|
| background | 27129 | 0.962 | 0.999 | 0.980 | 0.998 | 0.986 |
| PAC | 428 | 0.534 | 0.481 | 0.506 | 0.498 | 0.937 |
| PVC | 2631 | 0.985 | 0.610 | 0.753 | 0.948 | 0.993 |

Validation (patients 3, 6, 16, 24, 31):
| class | n_true | precision | recall | f1 | auprc | auroc |
|---|---|---|---|---|---|---|
| background | 29894 | 0.992 | 0.941 | 0.966 | 0.998 | 0.981 |
| PAC | 181 | 0.918 | 0.370 | 0.528 | 0.561 | 0.900 |
| PVC | 2161 | 0.523 | 0.925 | 0.668 | 0.914 | 0.985 |

## Read before evaluating against your annotations
- **Use the probabilities, not only `pred`.** The argmax operating point does not transfer between patients
  (validation patients 24/31 got too many PVC calls, test patient 27 too few: recall 0.26 with AUPRC 0.87). Sweep a
  threshold on `p_PAC` / `p_PVC` and report AUPRC alongside F1.
- **PAC rests on very few patients.** 97% of training PACs come from INCART patients 15 and 30; the test PAC
  result is essentially one patient (10: AUPRC 0.74, recall 0.43). Expect PAC to be the weak class on Homer.
- Domain shift: INCART is 30 min resting-ish Holter at 257 Hz; Homer is continuous monitoring at 250 Hz.

## Next steps
1. Score these per-beat predictions against your per-lead annotations (match beats within a tolerance, e.g.
   +/-150 ms), reporting AUPRC and a threshold sweep per class.
2. Calibrate per-class thresholds on INCART validation only, then re-check on test and Homer.
3. Same-recipe models exist for lead II and lead V5 (`vm/incart_ii_beat_B`, `vm/incart_v5_beat_B`): feed each
   model only its own lead (II -> II model; the chest lead, CM5 ~ V5, -> V5 model) and score against that lead's
   annotations.

Files: `incart_v5_beat_B.pt` (weights + metadata incl. the preprocessing contract), `src/` (only the modules inference
needs), `metrics/` (per-class, per-patient, confusion; `model_meta.json`).
