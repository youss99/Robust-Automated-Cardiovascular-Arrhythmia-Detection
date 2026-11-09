"""
Rebuild the Unified dataset so every row's signal belongs to the record named on that row.

In Unified_Dataset_No_PTB_Segment_fixed, rows 0-18436 are labelled Chapman but hold copies of the CinC signals of
rows 40507-58943. This takes each Chapman row from chapman/ChapmanDataset.h5 by filename and keeps the CinC rows,
which already match cinc-2020/Cinc2020Dataset.h5. Records blank on all 12 leads are dropped and listed.
Row order and labels are otherwise unchanged. Writes to a new folder and refuses to overwrite one.

  python src/scripts/unification/rebuild_unified_v2.py --datasets ../Datasets
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='../Datasets')
    ap.add_argument('--source', default='Unified_Dataset_No_PTB_Segment_fixed')
    ap.add_argument('--out', default='Unified_Dataset_No_PTB_Segment_v2')
    a = ap.parse_args()
    src_root, out_root = os.path.join(a.datasets, a.source), os.path.join(a.datasets, a.out)
    if os.path.exists(out_root):
        raise SystemExit(f'{out_root} already exists; remove it first if you mean to rebuild')

    df = pd.read_csv(os.path.join(src_root, 'modified_dataset', 'modified_dataset.csv'))
    with h5py.File(os.path.join(src_root, 'UnifiedDataset.h5'), 'r') as f:
        x = f['tracings'][:]
    assert len(df) == len(x), (len(df), x.shape)
    is_chapman = df.filename.str.contains('/chapman/').to_numpy()

    with h5py.File(os.path.join(a.datasets, 'chapman', 'ChapmanDataset.h5'), 'r') as f:
        chapman = f['tracings'][:]
    chapman_csv = pd.read_csv(os.path.join(a.datasets, 'chapman', 'modified_dataset', 'modified_dataset.csv'))
    key = lambda s: s.split('/chapman/')[-1]
    assert chapman_csv.filename.is_unique
    row_of = dict(zip(chapman_csv.filename.map(key), range(len(chapman_csv))))
    source_rows = df.filename[is_chapman].map(key).map(row_of)
    assert source_rows.notna().all(), 'a Chapman filename is missing from the source'
    changed = int((x[is_chapman] != chapman[source_rows.astype(int).to_numpy()]).any(axis=(1, 2)).sum())
    x[is_chapman] = chapman[source_rows.astype(int).to_numpy()]
    print(f'Chapman rows {int(is_chapman.sum())}; replaced with a different signal: {changed}', flush=True)

    blank = (x == 0).all(axis=(1, 2))
    dropped = df[blank]
    print(f'dropped {len(dropped)} rows blank on all 12 leads:\n{dropped.to_string()}', flush=True)
    keep = ~blank

    os.makedirs(os.path.join(out_root, 'modified_dataset'))
    df[keep].to_csv(os.path.join(out_root, 'modified_dataset', 'modified_dataset.csv'), index=False)
    dropped.to_csv(os.path.join(out_root, 'dropped_blank_records.csv'), index=False)
    with h5py.File(os.path.join(out_root, 'UnifiedDataset.h5'), 'w') as f:
        f.create_dataset('tracings', data=x[keep], dtype='float32')
    with open(os.path.join(out_root, 'README.txt'), 'w') as f:
        f.write(f'Built by src/scripts/unification/rebuild_unified_v2.py from {a.source}.\n'
                f'{a.source} rows 0-18436 were labelled Chapman but held copies of the CinC signals of rows '
                f'40507-58943.\nChapman rows now come from chapman/ChapmanDataset.h5 by filename ({changed} rows '
                f'changed); CinC rows are unchanged.\n{len(dropped)} rows blank on all 12 leads were dropped, see '
                f'dropped_blank_records.csv. Rows: {int(keep.sum())}.\nChecks: verify_unified_no_blanks.py.\n')
    print(f'wrote {out_root}: {int(keep.sum())} rows', flush=True)


if __name__ == '__main__':
    main()
