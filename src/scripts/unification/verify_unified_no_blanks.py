"""
Verify a rebuilt UnifiedDataset.h5 row by row: no blank (all-zero) signals, no NaN/inf, every row identical to its
source dataset matched by filename, and every label equal to the source's label for that filename. Exits 1 if any check fails. Loads whole datasets (about 12 GB): run as a CPU job.

  python src/scripts/unification/verify_unified_no_blanks.py --datasets ../Datasets
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SHOW = ['II', 'V2', 'V5']
FLAT_UV = 1.0  # peak-to-peak below this (in the stored ~uV units) counts as a flat lead


def load(path):
    with h5py.File(path, 'r') as f:
        return f['tracings'][:]


def filename_key(s):
    """Path from the dataset folder down, so /home/student/... prefixes cannot affect matching."""
    return s.split('/chapman/')[-1] if '/chapman/' in s else s.split('/training/')[-1]


def compare_to_source(unified, names, labels, rows, src_h5, src_csv, label):
    """Every unified row must exactly equal a source row with the same filename, and carry that row's label."""
    source = load(src_h5)
    src_df = pd.read_csv(src_csv, usecols=['filename', 'label'])
    lookup = {}
    for i, name in enumerate(src_df.filename.map(filename_key)):
        lookup.setdefault(name, []).append(i)
    src_labels = src_df.label.to_numpy()
    missing, mismatch, wrong_label = 0, 0, 0
    for r in rows:
        cands = lookup.get(names[r])
        if not cands:
            missing += 1
            continue
        matches = [c for c in cands if np.array_equal(unified[r], source[c])]
        if not matches:
            mismatch += 1
        elif labels[r] not in src_labels[matches]:
            wrong_label += 1
    print(f'{label}: {len(rows)} rows vs {os.path.basename(src_h5)} ({source.shape[0]} rows): no filename match '
          f'{missing}, signal differs {mismatch}, label differs {wrong_label}', flush=True)
    return missing == 0 and mismatch == 0 and wrong_label == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='../Datasets')
    ap.add_argument('--dataset', default='Unified_Dataset_No_PTB_Segment_v2')
    a = ap.parse_args()
    root = os.path.join(a.datasets, a.dataset)
    df = pd.read_csv(os.path.join(root, 'modified_dataset', 'modified_dataset.csv'), usecols=['filename', 'label'])
    x = load(os.path.join(root, 'UnifiedDataset.h5'))
    print(f'UnifiedDataset.h5 {x.shape} {x.dtype}; label CSV rows {len(df)}', flush=True)
    src = df.filename.str.extract(r'/(chapman|cpsc_2018_extra|cpsc_2018|georgia|ptb-xl|ptb|st_petersburg_incart)/')[0]
    ok = x.shape[0] == len(df) and src.notna().all()
    print(f'rows match CSV and every row has a known source: {ok}')

    zero = (x == 0).all(axis=2)
    flat = np.ptp(x, axis=2) < FLAT_UV
    nonfinite = ~np.isfinite(x).all(axis=(1, 2))
    rows = []
    for s in sorted(src.unique()):
        m = (src == s).to_numpy()
        rows.append({'source': s, 'rows': int(m.sum()), 'first': int(np.flatnonzero(m)[0]),
                     'last': int(np.flatnonzero(m)[-1]), 'all_12_blank': int(zero[m].all(1).sum()),
                     'any_lead_blank': int(zero[m].any(1).sum()), 'nonfinite': int(nonfinite[m].sum()),
                     **{f'{l}_blank': int(zero[m, LEADS.index(l)].sum()) for l in SHOW},
                     **{f'{l}_flat': int(flat[m, LEADS.index(l)].sum()) for l in SHOW}})
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print('\n' + table.to_string(index=False) + '\n')
    print('blank rows by lead (whole dataset): '
          + ' '.join(f'{l}={int(zero[:, k].sum())}' for k, l in enumerate(LEADS)))
    blank_all, n_nonfinite = int(zero.all(1).sum()), int(nonfinite.sum())
    ok &= blank_all == 0 and n_nonfinite == 0
    print(f'rows blank on all 12 leads: {blank_all}; rows with NaN/inf: {n_nonfinite}\n', flush=True)

    distinct = len({x[i].tobytes() for i in range(len(x))})
    print(f'distinct signals: {distinct} of {len(x)}', flush=True)
    names, labels = df.filename.map(filename_key).to_numpy(), df.label.to_numpy()
    is_chapman = (src == 'chapman').to_numpy()
    ok &= compare_to_source(x, names, labels, np.flatnonzero(~is_chapman),
                            os.path.join(a.datasets, 'cinc-2020', 'Cinc2020Dataset.h5'),
                            os.path.join(a.datasets, 'cinc-2020', 'modified_dataset', 'modified_dataset.csv'), 'CinC')
    ok &= compare_to_source(x, names, labels, np.flatnonzero(is_chapman),
                            os.path.join(a.datasets, 'chapman', 'ChapmanDataset.h5'),
                            os.path.join(a.datasets, 'chapman', 'modified_dataset', 'modified_dataset.csv'), 'Chapman')
    print('\nRESULT:', 'ALL CHECKS PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
