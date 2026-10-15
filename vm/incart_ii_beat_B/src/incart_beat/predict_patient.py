"""
Classify every beat of one patient lead as background / PAC / PVC with a trained central-beat model.

Beats come from --beats (a CSV with a `sample` column at --fs, or a `time_s` column) or, by default, from
NeuroKit2 R-peak detection on the same lead. Each beat is classified from the 10 s window centred on it, using the
shared preprocessing (units -> uV, resample to 100 Hz) recorded in the checkpoint. Beats within 5 s of either end
have no full window and get NaN probabilities.

  python -m src.incart_beat.predict_patient --model incart_ii_beat_B.pt \
      --csv aligned_leads_8103_8123_first5h.csv --lead_col II --fs 250 --units mV --out patient_beats_II.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.incart_beat import preprocessing as pp
from src.incart_beat.model import load_beat_classifier, predict_beats


def detect_beats(x, fs, method):
    import neurokit2 as nk
    cleaned = nk.ecg_clean(x, sampling_rate=fs, method='neurokit')
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method)
    return np.asarray(info['ECG_R_Peaks'], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='exported checkpoint (model + meta)')
    p.add_argument('--csv', required=True, help='one row per sample')
    p.add_argument('--lead_col', required=True, help='column name (or integer index with --no_header)')
    p.add_argument('--no_header', action='store_true')
    p.add_argument('--fs', type=float, required=True, help='sampling rate of the CSV (Homer: 250)')
    p.add_argument('--units', required=True, choices=['mV', 'uV', 'V'], help='physical units of the lead column')
    p.add_argument('--beats', default=None, help='optional CSV with `sample` (at --fs) or `time_s` per beat')
    p.add_argument('--rpeak_method', default='neurokit', help='nk.ecg_peaks method when --beats is not given')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    model, meta = load_beat_classifier(args.model, args.device)
    print(f"model: lead {meta['preprocessing']['lead']}, norm {meta['norm_stats']}, loss {meta['loss']}, "
          f"selected epoch {meta['epoch']}; input {meta['preprocessing']['input_units']} at "
          f"{meta['preprocessing']['target_fs']} Hz", flush=True)

    col = int(args.lead_col) if args.no_header else args.lead_col
    df = pd.read_csv(args.csv, header=None if args.no_header else 'infer', usecols=[col])
    x = df[col].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f'{np.sum(~np.isfinite(x))} non-finite samples in {args.lead_col}; clean them first')

    if args.beats:
        b = pd.read_csv(args.beats)
        beats = (b['sample'].to_numpy() if 'sample' in b else np.rint(b['time_s'].to_numpy() * args.fs)).astype(np.int64)
        src = f'{args.beats}'
    else:
        beats = detect_beats(x, args.fs, args.rpeak_method)
        src = f'NeuroKit2 ecg_peaks(method={args.rpeak_method})'
    print(f'{len(x)} samples ({len(x) / args.fs / 3600:.2f} h) at {args.fs} Hz; {len(beats)} beats from {src}',
          flush=True)

    out = predict_beats(model, x, args.fs, args.units, beats, device=args.device, batch_size=args.batch_size)
    out.insert(1, 'time_s', out['sample'] / args.fs)
    out.to_csv(args.out, index=False)
    counts = out['pred'].value_counts(dropna=False).to_dict()
    print(f'predicted classes (argmax): {counts}; wrote {args.out}')


if __name__ == '__main__':
    main()
