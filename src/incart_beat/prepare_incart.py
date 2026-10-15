"""
Build the cached INCART single-lead dataset (default lead II) for central-beat classification (CPU only, ~1 min).

Splits patients BEFORE any window is formed, then writes:
  signals_uV_100hz.npy   (n_records, 180000) float32 microvolts, the selected lead
  records.csv            record -> patient, split, header gain/baseline/units, row in signals array
  beats.csv              one row per reference annotation with ids, original symbol, label and inclusion reason
  split.json, label_mapping.json, exclusion_counts.csv, class_counts.csv, preprocessing_config.json, prep_report.txt

Usage: python -m src.incart_beat.prepare_incart --incart_dir <Datasets/incartdb> --out_dir <cache dir> [--lead V5]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import wfdb

from src.incart_beat import preprocessing as pp

VAL_PATIENTS = (3, 6, 16, 24, 31)
TEST_PATIENTS = (2, 10, 17, 22, 27)


def patient_split(pid):
    return 'val' if pid in VAL_PATIENTS else 'test' if pid in TEST_PATIENTS else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incart_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--lead', default=pp.LEAD, help='WFDB signal name, e.g. II or V5')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    def log(msg=''):
        print(msg, flush=True)
        report.append(msg)

    records = open(os.path.join(args.incart_dir, 'RECORDS')).read().split()

    # 1) Patient split from headers, before touching signals or windows
    rec_rows = []
    for row, rec in enumerate(records):
        hdr = wfdb.rdheader(os.path.join(args.incart_dir, rec))
        pid = int(re.search(r'patient\s+(\d+)', ' '.join(hdr.comments)).group(1))
        ch = hdr.sig_name.index(args.lead)
        rec_rows.append({'row': row, 'record': rec, 'patient': pid, 'split': patient_split(pid),
                         'fs': hdr.fs, 'n_samples': hdr.sig_len, 'lead_index': ch,
                         'adc_gain': hdr.adc_gain[ch], 'baseline': hdr.baseline[ch], 'units': hdr.units[ch]})
    df_rec = pd.DataFrame(rec_rows)
    for name, group in df_rec.groupby('patient'):
        assert group.split.nunique() == 1, f'patient {name} spans splits'
    split = {s: sorted(df_rec.loc[df_rec.split == s, 'patient'].unique().tolist()) for s in ('train', 'val', 'test')}
    assert not (set(split['train']) & set(split['val']) or set(split['train']) & set(split['test'])
                or set(split['val']) & set(split['test']))
    log(f"Patients: train {len(split['train'])}, val {split['val']}, test {split['test']}")
    log(f"Header units for lead {args.lead}: {df_rec.units.value_counts().to_dict()} "
        f"(INCART headers omit units; WFDB applies the spec default 'mV'); "
        f"gains {sorted(df_rec.adc_gain.unique())}; baselines {sorted(df_rec.baseline.unique())}")

    # 2) Signals: physical (once, by WFDB) -> uV -> 100 Hz
    n_out = None
    sigs, beat_rows = [], []
    for r in rec_rows:
        path = os.path.join(args.incart_dir, r['record'])
        phys, fs, units = pp.load_wfdb_lead(path, args.lead)
        # Guard against double ADC conversion: WFDB p_signal must equal (adc - baseline) / gain from the header
        dig = wfdb.rdrecord(path, channel_names=[args.lead], physical=False).d_signal[:, 0].astype(np.float64)
        assert np.allclose(phys, (dig - r['baseline']) / r['adc_gain'], atol=1e-9), r['record']
        x = pp.prepare_lead(phys, fs, units)
        n_out = len(x) if n_out is None else n_out
        assert len(x) == n_out, f"{r['record']}: unexpected length {len(x)}"
        sigs.append(x)

        ann = wfdb.rdann(path, 'atr')
        c100 = pp.map_samples(ann.sample, fs)
        _, full = pp.window_bounds(c100, len(x))
        for i, (s, sym, c, ok) in enumerate(zip(ann.sample, ann.symbol, c100, full)):
            if sym in pp.LABEL_MAP:
                reason = 'included' if ok else 'incomplete_window'
            elif sym in pp.EXCLUDED_BEAT_SYMBOLS:
                reason = 'excluded_symbol'
            else:
                reason = 'unmapped_symbol'
            beat_rows.append({'beat_id': f"{r['record']}_{i:06d}", 'record': r['record'], 'patient': r['patient'],
                              'split': r['split'], 'ann_index': i, 'sample_native': int(s), 'sample_100hz': int(c),
                              'symbol': sym, 'label': pp.LABEL_MAP.get(sym, -1), 'full_window': bool(ok),
                              'status': reason})
    signals = np.stack(sigs)
    df = pd.DataFrame(beat_rows)

    # 3) Report
    log(f'Signals: {signals.shape} float32 {pp.INPUT_UNITS} at {pp.TARGET_FS} Hz; '
        f'per-record std range {signals.std(1).min():.1f}-{signals.std(1).max():.1f} uV')
    excl = (df[df.status != 'included'].groupby(['status', 'symbol', 'split']).size()
            .unstack('split', fill_value=0).reindex(columns=['train', 'val', 'test'], fill_value=0))
    excl['total'] = excl.sum(1)
    log('\nNot used as centres (status x symbol x split):\n' + excl.to_string())
    inc = df[df.status == 'included']
    cls = (inc.assign(cls=inc.label.map(dict(enumerate(pp.CLASS_NAMES))))
           .groupby(['cls', 'split']).size().unstack('split', fill_value=0)
           .reindex(columns=['train', 'val', 'test'], fill_value=0).reindex(pp.CLASS_NAMES))
    log('\nIncluded centres by class x split:\n' + cls.to_string())
    for c in ('PAC', 'PVC'):
        k = pp.CLASS_NAMES.index(c)
        per = inc[inc.label == k].groupby(['split', 'patient']).size()
        log(f'\n{c} centres per patient:\n' + per.to_string())
    tr_pac = inc[(inc.split == 'train') & (inc.label == 1)].patient.value_counts()
    log(f'\nNOTE: {tr_pac.iloc[:2].sum()} of {tr_pac.sum()} training PAC centres '
        f'({100 * tr_pac.iloc[:2].sum() / tr_pac.sum():.1f}%) come from patients '
        f'{tr_pac.index[0]} and {tr_pac.index[1]}; PAC results reflect very few patients.')

    # 4) Write
    np.save(os.path.join(args.out_dir, 'signals_uV_100hz.npy'), signals)
    df_rec.to_csv(os.path.join(args.out_dir, 'records.csv'), index=False)
    df.to_csv(os.path.join(args.out_dir, 'beats.csv'), index=False)
    excl.to_csv(os.path.join(args.out_dir, 'exclusion_counts.csv'))
    cls.to_csv(os.path.join(args.out_dir, 'class_counts.csv'))
    with open(os.path.join(args.out_dir, 'split.json'), 'w') as f:
        json.dump({'unit': 'patient', 'val_patients': list(VAL_PATIENTS), 'test_patients': list(TEST_PATIENTS),
                   **{f'{k}_patients_all': v for k, v in split.items()},
                   'records': df_rec.groupby('split').record.apply(list).to_dict()}, f, indent=1)
    with open(os.path.join(args.out_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'class_names': pp.CLASS_NAMES, 'label_map': pp.LABEL_MAP,
                   'excluded_beat_symbols': list(pp.EXCLUDED_BEAT_SYMBOLS),
                   'other_symbols': 'unmapped -> excluded and counted by symbol'}, f, indent=1)
    with open(os.path.join(args.out_dir, 'preprocessing_config.json'), 'w') as f:
        json.dump(pp.preprocessing_config(args.lead), f, indent=1)
    with open(os.path.join(args.out_dir, 'prep_report.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    log(f'\nWrote cache to {args.out_dir}')


if __name__ == '__main__':
    main()
