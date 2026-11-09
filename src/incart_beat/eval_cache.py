"""
Score saved central-beat models on one split of any prepared cache (e.g. multi-database models on the
reference-centred INCART validation windows). Validation only unless --allow_test is given.

  python -m src.incart_beat.eval_cache --cache_dir ../Datasets/incart_beat_leadV5_100hz \
      --ckpt results/fine-tune/saved_models/incart_multi_nk/incart_v5_beat_B/best.pt [more .pt ...]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.incart_beat.model import BeatClassifier
from src.incart_beat.train import WindowSource, class_metrics, eval_order, label_scheme, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--allow_test', action='store_true')
    ap.add_argument('--batch_size', type=int, default=128)
    a = ap.parse_args()
    if a.split == 'test' and not a.allow_test:
        raise SystemExit('refusing to score the test split without --allow_test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src = WindowSource(a.cache_dir, device)
    df, rows, starts, _ = src.split(a.split)
    print(f'{a.cache_dir} [{a.split}]: {len(df)} windows, {df.patient.nunique()} patients, '
          f'class counts {np.bincount(df.label, minlength=3).tolist()}', flush=True)
    out = []
    for ck_path in a.ckpt:
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        meta = ck['meta']
        classes, post = label_scheme(meta)
        if meta.get('norm_stats') == 'BN':
            print(f'WARNING {ck_path}: BN normalization is batch-dependent; scored with the fixed eval order')
        model = BeatClassifier(norm_stats=meta['norm_stats'], n_classes=len(classes))
        model.load_state_dict(ck['model'])
        model.to(device)
        probs = post(predict(model, src, rows, starts, a.batch_size, eval_order(len(df), 0)))
        pc = class_metrics(df.label.values, probs).set_index('class')
        out.append({'model': os.path.relpath(os.path.dirname(ck_path), 'results/fine-tune/saved_models'),
                    'epoch': meta.get('epoch'),
                    **{f'{c}_{m}': pc.loc[c, m] for c in ('PAC', 'PVC') for m in ('auprc', 'auroc', 'precision',
                                                                                  'recall', 'f1')}})
    res = pd.DataFrame(out)
    res.insert(2, 'mean_PAC_PVC_auprc', (res.PAC_auprc + res.PVC_auprc) / 2)
    pd.set_option('display.width', 250)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
