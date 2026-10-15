"""Checks a prepare_multi_nk cache before it is used for training (2026-10-15). Run from the repo root:
    .venv/bin/python vm/verify_multi_nk_cache.py <cache_dir> [--data_root "<DATASET dir>"] [--per_source 300]
1. train.py's WindowSource loads the cache (its own asserts: full windows, labels >= 0, train.py fine classes).
2. Every included window plus the jitter margin lies inside its own source-record segment of its row.
3. Splits are patient-disjoint.
4. With --data_root: random included beats per source are rebuilt from the raw WFDB record; window samples, NeuroKit
   centre position and reference symbol must match the cache exactly.
5. Amplitude / polarity per source (unit or lead-polarity errors show up here) and NeuroKit-centre accounting.
Writes VERIFY.txt into the cache folder; exit status 1 if any check fails.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import wfdb

sys.path.insert(0, os.getcwd())
from src.incart_beat import preprocessing as pp, train as T
from src.incart_beat import prepare_multi_nk as M

ap = argparse.ArgumentParser()
ap.add_argument('cache_dir'); ap.add_argument('--data_root', default=None); ap.add_argument('--per_source', type=int, default=300)
args = ap.parse_args()
C = args.cache_dir; out, fails = [], []


def log(msg=''):
    print(msg, flush=True); out.append(msg)


def check(name, ok, detail=''):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


beats = pd.read_csv(f'{C}/beats.csv', keep_default_na=False, low_memory=False)
segs = pd.read_csv(f'{C}/segments.csv'); recs = pd.read_csv(f'{C}/records.csv')
sig = np.load(f'{C}/signals_uV_100hz.npy', mmap_mode='r'); lm = json.load(open(f'{C}/label_mapping.json'))
margin = lm.get('window_margin_samples', 0)
family = json.load(open(f'{C}/preprocessing_config.json'))['lead']
inc = beats[beats.status == 'included'].copy()
inc['row'] = inc.record.map(dict(zip(recs.record, recs.row))).astype(int)
inc['sample_100hz'] = inc.sample_100hz.astype(int); inc['label'] = inc.label.astype(int)
log(f'Cache {C}: signals {sig.shape}, {len(beats)} NeuroKit centres, {len(inc)} included, margin {margin} samples')

# 1 ---- train.py loader
try:
    src = T.WindowSource(C, torch.device('cpu'))
    check('train.py WindowSource loads the cache', len(src.beats) == len(inc), f'{len(src.beats)} windows')
    del src
except Exception as e:
    check('train.py WindowSource loads the cache', False, repr(e))

# 2 ---- window + margin inside its own record segment
m = inc.merge(segs, on=['source_db', 'source_record', 'row'], how='left', suffixes=('', '_seg'))
lo = m.sample_100hz - pp.HALF_WINDOW - margin; hi = m.sample_100hz + pp.HALF_WINDOW + margin
inside = m.row_offset.notna() & (lo >= m.row_offset) & (hi <= m.row_offset + m.length)
check('windows (+margin) inside their own record segment', bool(inside.all()), f'{int((~inside).sum())} outside of {len(m)}')
# 3 ---- patient-disjoint splits
ns = beats.groupby('patient').split.nunique()
check('every patient in exactly one split', bool((ns == 1).all()), f'{int((ns > 1).sum())} patients in >1 split')
sp = json.load(open(f'{C}/split.json'))
sets = [set(sp[f'{s}_patients']) for s in ('train', 'val', 'test')]
check('split.json lists are disjoint', not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]))

# 4 ---- rebuild random windows from the raw records
if args.data_root:
    rng = np.random.default_rng(0); bad_sig = bad_pos = bad_sym = n_chk = 0
    for db, g in m.groupby('source_db'):
        lead = dict(M.SOURCES[family])[db]
        recs_db = rng.choice(g.source_record.unique(), size=min(12, g.source_record.nunique()), replace=False)
        for rec in recs_db:
            gg = g[g.source_record == rec]; gg = gg.iloc[rng.choice(len(gg), size=min(len(gg), max(1, args.per_source // 12)), replace=False)]
            path = os.path.join(args.data_root, db, rec)
            phys, fs, units = pp.load_wfdb_lead(path, lead); phys = np.where(np.isfinite(phys), phys, 0.0)
            x = pp.prepare_lead(phys, fs, units)
            ann = wfdb.rdann(path, 'atr'); keep = np.array([s in M.BEAT_SYMBOLS for s in ann.symbol])
            ref = np.asarray(ann.sample)[keep]; tol = int(round(M.TOL_S * fs))
            for b in gg.itertuples():
                c = int(pp.map_samples([int(b.sample_native)], fs)[0]); n_chk += 1
                bad_pos += int(b.row_offset + (c - b.rec_start) != b.sample_100hz)
                w_raw = x[c - pp.HALF_WINDOW: c + pp.HALF_WINDOW]
                w_cache = np.asarray(sig[b.row, b.sample_100hz - pp.HALF_WINDOW: b.sample_100hz + pp.HALF_WINDOW])
                bad_sig += int(w_raw.shape != w_cache.shape or not np.array_equal(w_raw, w_cache))
                if b.match == 'matched':
                    ai = int(b.ann_index); sy = ann.symbol[ai]
                    bad_sym += int(sy != b.source_symbol or M.label_of(sy, db) != b.label or abs(int(ann.sample[ai]) - int(b.sample_native)) > tol)
                else:                               # unmatched peak: background and no reference beat within tolerance
                    bad_sym += int(b.label != 0 or (len(ref) and np.min(np.abs(ref - int(b.sample_native))) <= tol))
    check('windows rebuilt from raw records equal the cache', bad_sig == 0, f'{bad_sig} of {n_chk} differ')
    check('centre positions consistent', bad_pos == 0, f'{bad_pos} of {n_chk}')
    check('labels consistent with the reference annotation', bad_sym == 0, f'{bad_sym} of {n_chk}')

# 5 ---- amplitude / polarity per source, and accounting
log('\nPer source: included windows, median |centre ±50 ms| peak (uV), fraction with positive peak at the centre:')
rng = np.random.default_rng(1)
for db, g in inc.groupby('source_db'):
    s = g.iloc[rng.choice(len(g), size=min(3000, len(g)), replace=False)]
    seg = np.stack([np.asarray(sig[r, c - 5: c + 6]) for r, c in zip(s.row, s.sample_100hz)])
    base = np.stack([np.median(np.asarray(sig[r, c - pp.HALF_WINDOW: c + pp.HALF_WINDOW])) for r, c in zip(s.row, s.sample_100hz)])
    dev = seg - base[:, None]; pk = dev[np.arange(len(dev)), np.argmax(np.abs(dev), axis=1)]
    log(f'  {db:9s} n={len(g):8d}  median |peak| {np.median(np.abs(pk)):7.0f} uV   positive {np.mean(pk > 0):.2f}')
log('\nNeuroKit centres by status x source:\n' + beats.groupby(['status', 'source_db']).size().unstack('source_db', fill_value=0).to_string())
log('\nIncluded by class x split:\n' + inc.groupby(['label', 'split']).size().unstack('split', fill_value=0)
    .rename(index=dict(enumerate(pp.CLASS_NAMES))).to_string())
log(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
open(f'{C}/VERIFY.txt', 'w').write('\n'.join(out) + '\n')
sys.exit(1 if fails else 0)
