"""
NeuroKit-centred INCART cache ("train like we deploy"): same signals, records and patient split as prepare_incart,
but every window centre is a NeuroKit2 R-peak from the patient-inference detector
(src.incart_beat.predict_patient.detect_beats, method 'neurokit') instead of the reference annotation position.
INCART's reference positions are algorithmic ("middle of the QRS from all 12 leads", never hand-corrected), and the
classifier only works within about +-20 ms of its training centre, so training on reference centres and deploying on
NeuroKit peaks mismatched the two (audit 2026-10-14).

Each NeuroKit peak is matched one-to-one to the nearest reference BEAT annotation within +-150 ms (nearest pairs first):
  matched, symbol in LABEL_MAP          -> that label; 'included' if the 10 s window fits, else 'incomplete_window'
  matched, excluded / unmapped symbol   -> 'excluded_symbol' / 'unmapped_symbol' (as prepare_incart)
  extra peak on an already-matched beat -> 'nk_duplicate' (double detection; not used for training)
  no reference beat within 150 ms       -> background (label 0, symbol 'N', match 'unmatched'): a spurious detection
                                           has to be called background at deployment (3-class scheme)
Reference beats with no NeuroKit peak within 150 ms can never be windows at deployment; they are absent from
beats.csv and counted per class and split in prep_report.txt (the R-peak ceiling).

Usage: python -m src.incart_beat.prepare_incart_nk --incart_dir <Datasets/incartdb> \
           --ref_cache <cache from prepare_incart, same lead> --out_dir <new cache>
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import wfdb

from src.incart_beat import preprocessing as pp
from src.incart_beat.predict_patient import detect_beats

BEAT_SYMBOLS = set('NLRBAaJSVrFejnE/fQ?')
TOL_S = 0.15


def match(nk, ref, tol):
    """One-to-one nearest-first matching of NeuroKit peaks to reference beats within tol samples.
    Returns ref index per peak (-1 = none) and a flag for peaks that had a candidate but lost it (duplicates)."""
    pairs = []
    for i, s in enumerate(nk):
        lo, hi = np.searchsorted(ref, s - tol), np.searchsorted(ref, s + tol, side='right')
        pairs += [(abs(int(ref[j]) - int(s)), i, j) for j in range(lo, hi)]
    pairs.sort()
    to_ref, taken, had = np.full(len(nk), -1), np.zeros(len(ref), bool), np.zeros(len(nk), bool)
    for _, i, j in pairs:
        had[i] = True
        if to_ref[i] < 0 and not taken[j]:
            to_ref[i], taken[j] = j, True
    return to_ref, had & (to_ref < 0), taken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incart_dir', required=True)
    parser.add_argument('--ref_cache', required=True, help='cache built by prepare_incart for the same lead')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    def log(msg=''):
        print(msg, flush=True)
        report.append(msg)

    cfg = json.load(open(os.path.join(args.ref_cache, 'preprocessing_config.json')))
    lead = cfg['lead']
    assert cfg == pp.preprocessing_config(lead), 'reference cache was built with a different preprocessing definition'
    df_rec = pd.read_csv(os.path.join(args.ref_cache, 'records.csv'))
    n_out = np.load(os.path.join(args.ref_cache, 'signals_uV_100hz.npy'), mmap_mode='r').shape[1]
    log(f'NeuroKit-centred INCART cache, lead {lead}; reference cache {args.ref_cache}; match tolerance +-{TOL_S * 1000:.0f} ms')

    rows, missed = [], []
    for r in df_rec.itertuples():
        path = os.path.join(args.incart_dir, r.record)
        phys, fs, units = pp.load_wfdb_lead(path, lead)
        nk = np.asarray(detect_beats(phys, fs, 'neurokit'), dtype=np.int64)
        ann = wfdb.rdann(path, 'atr')
        keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        ref_idx = np.flatnonzero(keep); ref = np.asarray(ann.sample)[keep]; sym = np.asarray(ann.symbol)[keep]
        to_ref, dup, taken = match(nk, ref, int(round(TOL_S * fs)))
        c100 = pp.map_samples(nk, fs)
        _, full = pp.window_bounds(c100, n_out)
        for i, (s, c, ok, j, d) in enumerate(zip(nk, c100, full, to_ref, dup)):
            if j >= 0:
                sy = sym[j]
                if sy in pp.LABEL_MAP:
                    reason = 'included' if ok else 'incomplete_window'
                elif sy in pp.EXCLUDED_BEAT_SYMBOLS:
                    reason = 'excluded_symbol'
                else:
                    reason = 'unmapped_symbol'
                m, lab, out_sym, ref_ann, off = 'matched', pp.LABEL_MAP.get(sy, -1), sy, int(ref_idx[j]), (int(s) - int(ref[j])) / fs * 1000
            elif d:
                m, lab, out_sym, ref_ann, off, reason = 'duplicate', -1, '', -1, np.nan, 'nk_duplicate'
            else:
                m, lab, out_sym, ref_ann, off = 'unmatched', 0, 'N', -1, np.nan
                reason = 'included' if ok else 'incomplete_window'
            rows.append({'beat_id': f'{r.record}_nk{i:06d}', 'record': r.record, 'patient': r.patient, 'split': r.split,
                         'ann_index': ref_ann, 'sample_native': int(s), 'sample_100hz': int(c), 'symbol': out_sym,
                         'label': lab, 'full_window': bool(ok), 'status': reason, 'match': m,
                         'offset_ms': round(off, 1) if np.isfinite(off) else np.nan})
        # reference beats of the three classes that a deployed model can never see (no NeuroKit peak within 150 ms)
        rc100 = pp.map_samples(ref, fs); _, rfull = pp.window_bounds(rc100, n_out)
        for j in np.flatnonzero(~taken & rfull):
            if sym[j] in pp.LABEL_MAP:
                missed.append({'record': r.record, 'patient': r.patient, 'split': r.split, 'symbol': sym[j],
                               'label': pp.LABEL_MAP[sym[j]]})
    df = pd.DataFrame(rows); mis = pd.DataFrame(missed, columns=['record', 'patient', 'split', 'symbol', 'label'])

    inc = df[df.status == 'included']
    cls = (inc.assign(cls=inc.label.map(dict(enumerate(pp.CLASS_NAMES)))).groupby(['cls', 'split']).size()
           .unstack('split', fill_value=0).reindex(columns=['train', 'val', 'test'], fill_value=0).reindex(pp.CLASS_NAMES))
    log('\nIncluded centres by class x split (background includes unmatched NeuroKit detections):\n' + cls.to_string())
    log('\nNeuroKit peaks by match status x split:\n' + df.groupby(['match', 'split']).size().unstack('split', fill_value=0)
        .reindex(columns=['train', 'val', 'test'], fill_value=0).to_string())
    off = df[df.match == 'matched'].assign(cls=lambda d: d.label.map(dict(enumerate(pp.CLASS_NAMES))))
    log('\nNeuroKit peak minus reference position (ms), matched beats, p10 / p50 / p90:')
    for c, g in off[off.label >= 0].groupby('cls'):
        log(f'  {c:10s} n={len(g):6d}  {g.offset_ms.quantile(.1):+.0f} / {g.offset_ms.median():+.0f} / {g.offset_ms.quantile(.9):+.0f}')
    ref_n = (pd.read_csv(os.path.join(args.ref_cache, 'beats.csv'), keep_default_na=False).query("status == 'included'")
             .groupby(['label', 'split']).size())
    log('\nR-peak ceiling: reference beats (full window) with no NeuroKit peak within 150 ms, by class x split:')
    for k, name in enumerate(pp.CLASS_NAMES):
        parts = []
        for s in ('train', 'val', 'test'):
            n_miss = int(((mis.label == k) & (mis.split == s)).sum()); n_ref = int(ref_n.get((k, s), 0))
            parts.append(f'{s} {n_miss}/{n_ref} ({100 * n_miss / max(n_ref, 1):.1f}%)')
        log(f'  {name:10s} ' + '   '.join(parts))
    tr_pac = inc[(inc.split == 'train') & (inc.label == 1)].patient.value_counts()
    log(f'\nNOTE: {tr_pac.iloc[:2].sum()} of {tr_pac.sum()} training PAC centres come from patients '
        f'{tr_pac.index[0]} and {tr_pac.index[1]}.')

    excl = (df[df.status != 'included'].groupby(['status', 'symbol', 'split']).size()
            .unstack('split', fill_value=0).reindex(columns=['train', 'val', 'test'], fill_value=0))
    for f in ('signals_uV_100hz.npy', 'records.csv', 'split.json', 'preprocessing_config.json'):
        shutil.copy2(os.path.join(args.ref_cache, f), os.path.join(args.out_dir, f))
    df.to_csv(os.path.join(args.out_dir, 'beats.csv'), index=False)
    mis.to_csv(os.path.join(args.out_dir, 'nk_missed_reference_beats.csv'), index=False)
    excl.to_csv(os.path.join(args.out_dir, 'exclusion_counts.csv'))
    cls.to_csv(os.path.join(args.out_dir, 'class_counts.csv'))
    with open(os.path.join(args.out_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'class_names': pp.CLASS_NAMES, 'label_map': pp.LABEL_MAP,
                   'excluded_beat_symbols': list(pp.EXCLUDED_BEAT_SYMBOLS),
                   'centres': 'NeuroKit2 R-peaks (detect_beats, neurokit), matched to reference beats within +-150 ms',
                   'unmatched_neurokit_peaks': 'background (label 0, symbol N)'}, f, indent=1)
    with open(os.path.join(args.out_dir, 'prep_report.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    log(f'\nWrote cache to {args.out_dir}')


if __name__ == '__main__':
    main()
