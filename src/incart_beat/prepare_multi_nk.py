"""
Multi-database NeuroKit-centred beat cache for the central-beat classifier (2026-10-15).

Sources per lead family
  II : INCART (II), MIT-BIH Arrhythmia (MLII), SVDB (ECG1, ~modified II), CPSC 2021 (II)
  V5 : INCART (V5), European ST-T (V5)
Output: the same files train.py reads for prepare_incart / prepare_incart_nk caches (train.py runs unchanged):
  signals_uV_100hz.npy (rows x 180000 float32 uV), records.csv (record = row id, row), beats.csv, split.json, ...
plus segments.csv (which source record sits where in which row) and nk_missed_reference_beats.csv.

Storage: records last 8 s to 7 h, so signals are stored in 30-min rows. Records longer than a row are cut into chunks
that overlap by one window plus two margins; shorter records are packed back to back, only with records of the same
split. A beat is kept only if its 10 s window plus MARGIN (100 ms) on each side lies inside its own source record, so
a window never mixes recordings or reaches row padding, even with train.py --jitter_ms up to 100.

Centres and labels (as prepare_incart_nk): NeuroKit2 peaks (detect_beats 'neurokit', the patient-inference detector)
matched one-to-one to reference beats within +-150 ms. Background: N L R B e j n; PAC: A, and S in SVDB / European
ST-T only (their supraventricular-premature symbol; flagged pac_from_S; INCART and MIT-BIH keep S excluded, as
prepare_incart); PVC: V; excluded: a J E F Q / f ? r ! and non-beats. Unmatched NeuroKit peaks -> background, except
inside ventricular flutter / fibrillation episodes ('[' ... ']', '(VFL', '(VF'), which are excluded; double detections
excluded; reference beats NeuroKit misses are listed in nk_missed_reference_beats.csv (the R-peak ceiling). `symbol`
holds a train.py-compatible symbol (N/R/j/n/B/A/V); `source_symbol` keeps the original.

Patient-level splits (seed 0)
  INCART       as before: val 3 6 16 24 31, test 2 10 17 22 27
  MIT-BIH      DS1 -> train, DS2 -> test (de Chazal 2004); 202 dropped (same patient as 201); paced 102 104 107 217 out
  SVDB, CPSC   70 / 15 / 15 train / val / test patients, stratified by PAC and PVC burden; of 5,000 seeded draws the
               one whose val / test shares of PAC, PVC and all beats are closest to 15 % (heavy patients stay out of val)
  Eur. ST-T    85 / 15 train / val, no test; records grouped by (age, sex) so a person never straddles splits
Usage:  python -m src.incart_beat.prepare_multi_nk --family II --data_root "<DATASET dir>" --out_dir <cache> [--workers 6]
"""
import argparse
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb

from src.incart_beat import preprocessing as pp
from src.incart_beat.predict_patient import detect_beats

ROW = 180000                                        # 30 min at 100 Hz
MARGIN = 10                                         # 100 ms at 100 Hz kept free around every window (training jitter)
OVERLAP = pp.WINDOW_LEN + 2 * MARGIN                # chunks of long records overlap by one window plus margins
TOL_S = 0.15
BEAT_SYMBOLS = set('NLRBAaJSVrFejnE/fQ?!')
BACKGROUND = set('NLRBejn'); PVC = set('V')
PAC_FROM_S = {'svdb', 'edb'}                        # S is only PAC where it is the database's SVPB symbol
N_SPLIT_DRAWS = 5000
FINE = set(['N', 'R', 'j', 'n', 'B', 'A', 'V'])      # train.py FINE_CLASSES
MIT_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MIT_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]  # 202 dropped
INCART_VAL, INCART_TEST = (3, 6, 16, 24, 31), (2, 10, 17, 22, 27)
SOURCES = {'II': [('incartdb', 'II'), ('mitdb', 'MLII'), ('svdb', 'ECG1'), ('cpsc2021', 'II')],
           'V5': [('incartdb', 'V5'), ('edb', 'V5')]}


def label_of(sym, db):
    if sym in BACKGROUND:
        return 0
    if sym == 'A' or (sym == 'S' and db in PAC_FROM_S):
        return 1
    if sym in PVC:
        return 2
    return -1


def fits(c, n):
    """Whole window plus MARGIN on each side inside a signal of n samples (centres c at 100 Hz)."""
    c = np.asarray(c, dtype=np.int64)
    return (c - pp.HALF_WINDOW - MARGIN >= 0) & (c + pp.HALF_WINDOW + MARGIN <= n)


def flutter_episodes(ann):
    """Sample intervals of ventricular flutter / fibrillation: '[' ... ']' and '(VFL' / '(VF' rhythm episodes."""
    end = np.iinfo(np.int64).max
    iv, start = [], None
    rhythm = [(int(s), (a or '').strip('\x00').strip()) for s, y, a in zip(ann.sample, ann.symbol, ann.aux_note) if y == '+']
    for s, y in zip(ann.sample, ann.symbol):
        if y == '[':
            start = int(s)
        elif y == ']' and start is not None:
            iv.append((start, int(s))); start = None
    if start is not None:
        iv.append((start, end))
    for k, (s, a) in enumerate(rhythm):
        if a in ('(VFL', '(VF'):
            iv.append((s, rhythm[k + 1][0] if k + 1 < len(rhythm) else end))
    return iv


def train_symbol(sym, lab):
    if lab == 1:
        return 'A'
    if lab == 2:
        return 'V'
    return sym if sym in FINE else 'N'


def list_records(db, data_root, lead):
    """(record path relative to the db folder, patient id, extra) for every usable record that has the lead."""
    d = os.path.join(data_root, db)
    recs = [l.strip() for l in open(os.path.join(d, 'RECORDS')) if l.strip()]
    out = []
    for r in recs:
        if not os.path.exists(os.path.join(d, r + '.hea')) or not os.path.exists(os.path.join(d, r + '.atr')):
            continue
        h = wfdb.rdheader(os.path.join(d, r))
        if lead not in h.sig_name:
            continue
        if db == 'incartdb':
            pid = int(re.search(r'patient\s+(\d+)', ' '.join(h.comments)).group(1)); extra = None
        elif db == 'mitdb':
            if int(r) not in MIT_DS1 + MIT_DS2:
                continue
            pid = 201 if int(r) in (201, 202) else int(r); extra = 'DS1' if int(r) in MIT_DS1 else 'DS2'
        elif db == 'cpsc2021':
            pid = int(re.match(r'Training_set_I+/data_(\d+)_\d+', r).group(1)); extra = None
        elif db == 'edb':
            m = re.search(r'Age:\s*(\d+)\s+Sex:\s*([MF])', ' '.join(h.comments))
            pid = r; extra = f'{m.group(1)}{m.group(2)}' if m else r
        else:                                       # svdb: one record per person
            pid = r; extra = None
        out.append((r, f'{db}:{pid}', extra))
    return out


def process(args):
    """NeuroKit centres, matching and labels for one source record."""
    db, data_root, rec, lead, patient = args
    path = os.path.join(data_root, db, rec)
    phys, fs, units = pp.load_wfdb_lead(path, lead)
    bad = ~np.isfinite(phys)
    if bad.any():
        phys = np.where(bad, 0.0, phys)
    try:
        nk = np.asarray(detect_beats(phys, fs, 'neurokit'), dtype=np.int64)
    except Exception as e:                          # e.g. records of a few seconds: no beats, logged
        print(f'NeuroKit failed on {db}/{rec}: {e}', flush=True); nk = np.zeros(0, np.int64)
    x = pp.prepare_lead(phys, fs, units)
    ann = wfdb.rdann(path, 'atr')
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ref = np.asarray(ann.sample)[keep]; sym = np.asarray(ann.symbol)[keep]; ref_idx = np.flatnonzero(keep)
    in_ep = lambda t: np.any([(t >= a) & (t <= b) for a, b in flutter_episodes(ann)] or [np.zeros(len(t), bool)], axis=0)
    tol = int(round(TOL_S * fs)); pairs = []
    for i, s in enumerate(nk):
        lo, hi = np.searchsorted(ref, s - tol), np.searchsorted(ref, s + tol, side='right')
        pairs += [(abs(int(ref[j]) - int(s)), i, j) for j in range(lo, hi)]
    pairs.sort()
    to_ref, taken, had = np.full(len(nk), -1), np.zeros(len(ref), bool), np.zeros(len(nk), bool)
    for _, i, j in pairs:
        had[i] = True
        if to_ref[i] < 0 and not taken[j]:
            to_ref[i], taken[j] = j, True
    c100 = pp.map_samples(nk, fs); full = fits(c100, len(x)); flutter = in_ep(nk)
    rows = []
    for i, (s, c, ok, j, fl) in enumerate(zip(nk, c100, full, to_ref, flutter)):
        if fl:
            src_sym = sym[j] if j >= 0 else ''; lab, m, off, ai, status = -1, 'flutter_episode', np.nan, -1, 'excluded_episode'
        elif j >= 0:
            src_sym = sym[j]; lab = label_of(src_sym, db); m = 'matched'; off = (int(s) - int(ref[j])) / fs * 1000; ai = int(ref_idx[j])
            status = ('included' if ok else 'incomplete_window') if lab >= 0 else 'excluded_symbol'
        elif had[i]:
            src_sym, lab, m, off, ai, status = '', -1, 'duplicate', np.nan, -1, 'nk_duplicate'
        else:
            src_sym, lab, m, off, ai = 'N', 0, 'unmatched', np.nan, -1
            status = 'included' if ok else 'incomplete_window'
        rows.append(dict(source_db=db, source_record=rec, patient=patient, ann_index=ai, sample_native=int(s),
                         c_rec=int(c), source_symbol=src_sym, label=lab, symbol=train_symbol(src_sym, lab) if lab >= 0 else src_sym,
                         full_window=bool(ok), status=status, match=m, offset_ms=round(off, 1) if np.isfinite(off) else np.nan,
                         pac_from_S=lab == 1 and src_sym == 'S'))
    rfull = fits(pp.map_samples(ref, fs), len(x)) & ~in_ep(ref)
    missed = [dict(source_db=db, source_record=rec, patient=patient, source_symbol=sym[j], label=label_of(sym[j], db))
              for j in np.flatnonzero(~taken & rfull) if label_of(sym[j], db) >= 0]
    counts = Counter(label_of(s, db) for s in sym)
    return dict(db=db, rec=rec, patient=patient, fs=fs, n=len(x), x=x, beats=rows, missed=missed, nan_samples=int(bad.sum()),
                n_pac=counts.get(1, 0), n_pvc=counts.get(2, 0), n_ref=int(len(ref)))


def assign_splits(db, recs, results, seed=0):
    """Patient-level split per database (see module docstring)."""
    pat = sorted({p for _, p, _ in recs})
    if db == 'incartdb':
        return {p: ('val' if int(p.split(':')[1]) in INCART_VAL else 'test' if int(p.split(':')[1]) in INCART_TEST else 'train') for p in pat}
    if db == 'mitdb':
        return {p: ('train' if any(e == 'DS1' for _, q, e in recs if q == p) else 'test') for p in pat}
    rng = np.random.default_rng(seed)
    if db == 'edb':                                 # group records by (age, sex); a group stays in one split
        grp = {p: e for _, p, e in recs}; groups = sorted(set(grp.values())); rng.shuffle(groups)
        n_val = max(1, round(0.15 * len(groups))); val = set(groups[:n_val])
        return {p: ('val' if grp[p] in val else 'train') for p in pat}
    # 70 / 15 / 15 patients per stratum (PAC-rich / PVC-rich / both / neither). PAC and PVC are concentrated in a few
    # patients, so a single draw can put most PACs of a database into val; out of N_SPLIT_DRAWS seeded draws keep the
    # one whose val and test shares of PAC, PVC and all reference beats are closest to 15 % each.
    burden = {p: np.zeros(3) for p in pat}
    for r in results:
        burden[r['patient']] += (r['n_pac'], r['n_pvc'], r['n_ref'])
    B = np.stack([burden[p] for p in pat]); tot = np.maximum(B.sum(0), 1)
    strata = {}
    for i, p in enumerate(pat):
        strata.setdefault((B[i, 0] >= 10, B[i, 1] >= 10), []).append(i)
    best, best_cost = None, np.inf
    for t in range(N_SPLIT_DRAWS):
        rng = np.random.default_rng([seed, t]); assign = np.full(len(pat), 'train', dtype=object)
        for key in sorted(strata):
            idx = np.array(strata[key]); rng.shuffle(idx); n = len(idx)
            n_val = round(0.15 * n); n_test = round(0.15 * n)
            assign[idx[:n_val]] = 'val'; assign[idx[n_val:n_val + n_test]] = 'test'
        cost = sum(((B[assign == s].sum(0) / tot - 0.15) ** 2).sum() for s in ('val', 'test'))
        if cost < best_cost:
            best, best_cost = assign.copy(), cost
    return dict(zip(pat, best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=list(SOURCES), required=True)
    ap.add_argument('--data_root', required=True, help='folder holding incartdb/ mitdb/ svdb/ cpsc2021/ edb/')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    def log(msg=''):
        print(msg, flush=True); report.append(msg)

    log(f'Multi-database NeuroKit-centred cache, family {args.family}, sources {SOURCES[args.family]}')
    all_res, split_of = [], {}
    for db, lead in SOURCES[args.family]:
        recs = list_records(db, args.data_root, lead)
        with Pool(args.workers) as pool:
            res = pool.map(process, [(db, args.data_root, r, lead, p) for r, p, _ in recs], chunksize=1)
        split_of.update(assign_splits(db, recs, res, args.seed))
        all_res += res
        n_pat = len({p for _, p, _ in recs})
        who = (f'{n_pat} patients' if db != 'edb' else
               f'{n_pat} records counted as patients (split by {len({e for *_, e in recs})} age+sex groups)')
        log(f'{db} ({lead}): {len(recs)} records, {who}, '
            f'{sum(r["n"] for r in res) / 100 / 3600:.1f} h, NaN samples zeroed {sum(r["nan_samples"] for r in res)}')

    # ---- pack into 30-min rows: long records -> overlapping chunks; short records packed per (db, split)
    rows_sig, segs = [], []
    def new_row():
        rows_sig.append(np.zeros(ROW, np.float32)); return len(rows_sig) - 1
    open_row = {}
    for r in sorted(all_res, key=lambda r: (r['db'], split_of[r['patient']], -r['n'])):
        sp = split_of[r['patient']]; n = r['n']
        if n > ROW:
            s = 0
            while True:
                k = new_row(); ln = min(ROW, n - s); rows_sig[k][:ln] = r['x'][s:s + ln]
                segs.append(dict(row=k, row_offset=0, rec_start=s, length=ln, source_db=r['db'], source_record=r['rec'], patient=r['patient'], split=sp))
                if s + ln >= n:
                    break
                s += ROW - OVERLAP
        else:
            key = (r['db'], sp)
            if key not in open_row or open_row[key][1] + n > ROW:
                open_row[key] = (new_row(), 0)
            k, used = open_row[key]; rows_sig[k][used:used + n] = r['x']
            segs.append(dict(row=k, row_offset=used, rec_start=0, length=n, source_db=r['db'], source_record=r['rec'], patient=r['patient'], split=sp))
            open_row[key] = (k, used + n)
    segs = pd.DataFrame(segs)

    # ---- beats: place each beat in the chunk where its full window fits
    beat_rows, missed = [], []
    chunks = {k: list(g.sort_values('rec_start')[['rec_start', 'length', 'row', 'row_offset']].itertuples(index=False, name=None))
              for k, g in segs.groupby(['source_db', 'source_record'])}
    for r in all_res:
        sp = split_of[r['patient']]
        sg = chunks[(r['db'], r['rec'])]
        for i, b in enumerate(r['beats']):
            c = b['c_rec']; row, local = -1, -1
            if b['full_window']:
                for st, ln, rw, ro in sg:
                    if st <= c - pp.HALF_WINDOW - MARGIN and c + pp.HALF_WINDOW + MARGIN <= st + ln:
                        row, local = rw, ro + (c - st); break
            status = b['status'] if (row >= 0 or b['status'] != 'included') else 'incomplete_window'
            beat_rows.append(dict(beat_id=f"{r['db']}:{r['rec']}_nk{i:06d}", record=f'row{row:05d}' if row >= 0 else '',
                                  patient=r['patient'], split=sp, ann_index=b['ann_index'], sample_native=b['sample_native'],
                                  sample_100hz=int(local) if row >= 0 else -1, symbol=b['symbol'], label=b['label'],
                                  full_window=row >= 0, status=status, match=b['match'], offset_ms=b['offset_ms'],
                                  source_db=r['db'], source_record=r['rec'], source_symbol=b['source_symbol'], pac_from_S=b['pac_from_S']))
        missed += [{**m, 'split': sp} for m in r['missed']]
    df = pd.DataFrame(beat_rows); mis = pd.DataFrame(missed)
    signals = np.stack(rows_sig)
    recs_df = pd.DataFrame({'record': [f'row{k:05d}' for k in range(len(rows_sig))], 'row': range(len(rows_sig))})

    # ---- report
    inc = df[df.status == 'included']
    tab = inc.groupby(['source_db', 'split', 'label']).size().unstack('label', fill_value=0).rename(columns=dict(enumerate(pp.CLASS_NAMES)))
    log('\nIncluded centres (source x split x class):\n' + tab.to_string())
    pats = inc.groupby(['source_db', 'split']).patient.nunique().unstack('split', fill_value=0)
    log('\nPatients (source x split):\n' + pats.to_string())
    for c, k in (('PAC', 1), ('PVC', 2)):
        per = inc[(inc.label == k) & (inc.split == 'train')].patient.value_counts()
        log(f'\nTraining {c}: {per.sum()} centres from {(per >= 10).sum()} patients with >= 10 '
            f'(top two patients hold {100 * per.iloc[:2].sum() / max(per.sum(), 1):.0f}%)')
    log(f"PAC centres from S symbols (SVDB / European ST-T): {int(inc.pac_from_S.sum())}")
    log('\nNeuroKit peaks by match status (source x match):\n' +
        df.groupby(['source_db', 'match']).size().unstack('match', fill_value=0).to_string())
    um = df.groupby(['source_db', 'source_record']).match.agg(peaks='size', unmatched=lambda m: int((m == 'unmatched').sum()))
    um['frac'] = (um.unmatched / um.peaks).round(3); hi = um[(um.frac > 0.10) & (um.unmatched >= 20)]
    log(f'\nRecords with > 10% (and >= 20) unmatched NeuroKit peaks, i.e. spurious detections or unannotated stretches: '
        f'{len(hi)} of {len(um)}' + ('\n' + hi.sort_values('frac', ascending=False).head(20).to_string() if len(hi) else ''))
    if len(mis):
        log('\nR-peak ceiling: reference beats NeuroKit missed (source x class):\n' +
            mis.groupby(['source_db', 'label']).size().unstack('label', fill_value=0).rename(columns=dict(enumerate(pp.CLASS_NAMES))).to_string())
    log(f'\nSignals: {signals.shape} float32 ({signals.nbytes / 1e9:.2f} GB), {len(segs)} record segments')

    # ---- write
    np.save(os.path.join(args.out_dir, 'signals_uV_100hz.npy'), signals)
    recs_df.to_csv(os.path.join(args.out_dir, 'records.csv'), index=False)
    segs.to_csv(os.path.join(args.out_dir, 'segments.csv'), index=False)
    df.to_csv(os.path.join(args.out_dir, 'beats.csv'), index=False)
    mis.to_csv(os.path.join(args.out_dir, 'nk_missed_reference_beats.csv'), index=False)
    tab.to_csv(os.path.join(args.out_dir, 'class_counts.csv'))
    lead_cfg = 'II' if args.family == 'II' else 'V5'
    with open(os.path.join(args.out_dir, 'preprocessing_config.json'), 'w') as f:
        json.dump(pp.preprocessing_config(lead_cfg), f, indent=1)
    with open(os.path.join(args.out_dir, 'split.json'), 'w') as f:
        json.dump({'unit': 'patient', 'seed': args.seed, 'sources': SOURCES[args.family],
                   **{f'{s}_patients': sorted(p for p, v in split_of.items() if v == s) for s in ('train', 'val', 'test')}}, f, indent=1)
    with open(os.path.join(args.out_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'class_names': pp.CLASS_NAMES, 'background': sorted(BACKGROUND), 'PAC': ['A', 'S (only in ' + ', '.join(sorted(PAC_FROM_S)) + ')'],
                   'PVC': sorted(PVC), 'excluded': 'a J E F Q / f ? r ! and non-beat annotations; S outside ' + ', '.join(sorted(PAC_FROM_S)) +
                   "; NeuroKit peaks inside ventricular flutter / fibrillation episodes ('[' ... ']', '(VFL', '(VF')",
                   'S_as_PAC': 'flagged in beats.csv pac_from_S', 'window_margin_samples': MARGIN, 'row_samples': ROW,
                   'chunk_overlap_samples': OVERLAP,
                   'centres': 'NeuroKit2 R-peaks (detect_beats, neurokit) matched to reference beats within +-150 ms',
                   'unmatched_neurokit_peaks': 'background'}, f, indent=1)
    with open(os.path.join(args.out_dir, 'prep_report.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    log(f'\nWrote cache to {args.out_dir}')


if __name__ == '__main__':
    main()
