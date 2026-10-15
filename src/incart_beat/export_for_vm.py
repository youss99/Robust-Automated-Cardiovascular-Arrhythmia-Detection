"""
Build a small self-contained bundle (model + the code it needs + key metrics + README) for inference elsewhere.

  python -m src.incart_beat.export_for_vm --run_dir results/fine-tune/saved_models/incart/incart_ii_beat_B
Writes to vm/<run name>/ in the repo by default (pulled from the VM, see vm/VM_RUNBOOK.md).
"""
import argparse
import json
import os
import shutil

import pandas as pd
import torch

from src.incart_beat.prepare_incart import TEST_PATIENTS, VAL_PATIENTS

CODE = ['src/__init__.py', 'src/incart_beat/__init__.py', 'src/incart_beat/preprocessing.py',
        'src/incart_beat/model.py', 'src/incart_beat/predict_patient.py', 'src/core/constants/definitions.py',
        'src/core/models/vit_1d.py', 'src/core/models/layers/revin.py']
METRICS = ['val_per_class.csv', 'val_per_patient.csv', 'val_confusion.csv',
           'test_per_class.csv', 'test_per_patient.csv', 'test_confusion.csv']


def table(path):
    df = pd.read_csv(path)[['class', 'n_true', 'precision', 'recall', 'f1', 'auprc', 'auroc']]
    rows = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    rows += ['| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r) + ' |'
             for r in df.itertuples(index=False)]
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--out_dir', default=None, help='default: vm/<run name>')
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.run_dir))
    a.out_dir = a.out_dir or os.path.join('vm', name)
    os.makedirs(os.path.join(a.out_dir, 'metrics'), exist_ok=True)

    ck = torch.load(os.path.join(a.run_dir, 'best.pt'), map_location='cpu', weights_only=False)
    meta = ck['meta']
    torch.save({'model': ck['model'], 'meta': meta}, os.path.join(a.out_dir, f'{name}.pt'))  # no optimizer state
    for f in CODE:
        os.makedirs(os.path.join(a.out_dir, os.path.dirname(f)), exist_ok=True)
        shutil.copy2(f, os.path.join(a.out_dir, f))
    for f in METRICS:
        if os.path.exists(os.path.join(a.run_dir, f)):
            shutil.copy2(os.path.join(a.run_dir, f), os.path.join(a.out_dir, 'metrics', f))
    with open(os.path.join(a.out_dir, 'metrics', 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=str)

    lead = meta['preprocessing']['lead']
    test = os.path.join(a.run_dir, 'test_per_class.csv')
    readme = f"""# {name}: INCART central-beat classifier (lead {lead})

Classifies the beat at the centre of a 10 s lead-{lead} window as **background**, **PAC** or **PVC**.
PatchECG Small backbone (`{os.path.basename(meta['backbone'])}`, pretrained on Unified-fixed: Chapman, CPSC 2018,
CPSC-extra, Georgia; no INCART) fine-tuned on INCART with the thesis recipe (Table A.2: 20 head + 70 full epochs,
lr 1e-3, layer decay 0.65, weight decay 0.05, batch 128). Run {meta['run']}: per-window RevIN normalization
(`{meta['norm_stats']}`, batch-independent), {meta['loss']} loss, selected epoch {meta['epoch']} on mean PAC/PVC
validation AUPRC. Patient split: val {list(VAL_PATIENTS)}, test {list(TEST_PATIENTS)}; all other INCART
patients train.

## Run on a patient
Needs: torch, numpy, pandas, scipy, einops, neurokit2. From this folder:
```bash
python -m src.incart_beat.predict_patient --model {name}.pt \\
    --csv /path/to/aligned_leads_8103_8123_first5h.csv --lead_col {lead} --fs 250 --units mV \\
    --out patient_beats_{lead}.csv            # add --beats beats.csv to use your own beat positions
```
Give the lead's **physical units** (`--units mV` for the Homer CSVs); conversion to uV (what the backbone was
pretrained on) and resampling to 100 Hz happen inside, once. Feed only lead {lead} to this model.

Output: one row per beat: `sample`, `time_s`, `full_window`, `p_background`, `p_PAC`, `p_PVC`, `pred` (argmax).
Beats within 5 s of either end have no full window (NaN). Without `--beats`, beats are NeuroKit2 R-peaks on the
same lead; detector misses or double detections become missing or extra rows.

## How it did on INCART (window-level, natural class frequencies)
Test (patients {', '.join(map(str, TEST_PATIENTS))}):
{table(test) if os.path.exists(test) else '(not evaluated)'}

Validation (patients {', '.join(map(str, VAL_PATIENTS))}):
{table(os.path.join(a.run_dir, 'val_per_class.csv'))}

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

Files: `{name}.pt` (weights + metadata incl. the preprocessing contract), `src/` (only the modules inference
needs), `metrics/` (per-class, per-patient, confusion; `model_meta.json`).
"""
    with open(os.path.join(a.out_dir, 'README.md'), 'w') as f:
        f.write(readme)
    print(f'wrote {a.out_dir}')


if __name__ == '__main__':
    main()
