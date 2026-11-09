"""
Write a working copy of the Unified dataset for one lead set, without the rows where a selected lead is blank
(all zeros), so a pretraining job never sees a window with no ECG in a lead it trains on.

The filter lives here and not in the dataset class because the trainer saves its split to the CSV and later treats
each CSV row number as the H5 row: dropping rows in the loader would point later rows at the wrong signals. H5 and
CSV are written together, so row i of one is always row i of the other. All 12 leads are kept; the trainer selects.

  python src/scripts/unification/stage_unified_subset.py --source <Unified_v2> --leads II,V2 --out $SLURM_TMPDIR/unified
"""
import argparse
import os

import h5py
import pandas as pd

LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # EcgInterface.all_leads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--leads', required=True, help='comma-separated, e.g. II,V2')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    idx = [LEADS.index(l.strip()) for l in a.leads.split(',')]

    df = pd.read_csv(os.path.join(a.source, 'modified_dataset', 'modified_dataset.csv'), dtype=str)
    with h5py.File(os.path.join(a.source, 'UnifiedDataset.h5'), 'r') as f:
        x = f['tracings'][:]
    assert len(df) == len(x), (len(df), x.shape)

    blank = (x[:, idx] == 0).all(axis=2)            # [rows x selected leads]
    drop = blank.any(axis=1)
    print(f'{a.source}: {len(x)} rows; leads {a.leads}: blank per lead '
          f'{dict(zip(a.leads.split(","), blank.sum(axis=0).tolist()))}; removed {int(drop.sum())}, '
          f'kept {int((~drop).sum())}', flush=True)

    os.makedirs(os.path.join(a.out, 'modified_dataset'))
    df[~drop].to_csv(os.path.join(a.out, 'modified_dataset', 'modified_dataset.csv'), index=False)
    with h5py.File(os.path.join(a.out, 'UnifiedDataset.h5'), 'w') as f:
        f.create_dataset('tracings', data=x[~drop], dtype='float32')


if __name__ == '__main__':
    main()
