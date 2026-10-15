"""
Summarize runs A/B/C/D over training seeds (validation only): per-seed values, mean +/- sd per run, and paired
differences B-A (normalization), C-B (focal loss) and D-B (fine labels) within each seed (same init and example order per seed).

  python -m src.incart_beat.aggregate_seeds --seeds 200,201,202 [--lead II]
"""
import argparse
import json
import os

import pandas as pd

RES = 'results/fine-tune/saved_models/incart'
DEFAULT_SEED = 200


def run_row(lead, run, seed):
    d = os.path.join(RES, f"incart_{lead.lower()}_beat_{run}{'' if seed == DEFAULT_SEED else f'_s{seed}'}")
    if not os.path.exists(os.path.join(d, 'DONE')):
        return None
    h = pd.read_csv(os.path.join(d, 'history.csv'))
    full = h[h.phase == 'full']
    best = int(h.sel_mean_auprc_pac_pvc.idxmax())
    m = json.load(open(os.path.join(d, 'val_metrics.json')))
    bd = json.load(open(os.path.join(d, 'batch_dependence_after.json')))['conditions']
    return {'run': run, 'seed': seed, 'best_epoch': best,
            'sel_at_best': m['sel_mean_auprc_pac_pvc'], 'PAC_auprc': m['PAC_auprc'], 'PVC_auprc': m['PVC_auprc'],
            'PAC_auroc': m['PAC_auroc'], 'PVC_auroc': m['PVC_auroc'],
            'sel_median_full': full.sel_mean_auprc_pac_pvc.median(),
            'PAC_auprc_median_full': full.PAC_auprc.median(), 'PVC_auprc_median_full': full.PVC_auprc.median(),
            'bd_after_max_abs_diff': max(v['max_abs_prob_diff'] for v in bd.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='200,201,202')
    ap.add_argument('--lead', default='II')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    rows = [r for s in seeds for run in 'ABCD' if (r := run_row(a.lead, run, s)) is not None]
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ('run', 'seed')]
    pd.set_option('display.width', 200)
    print('Per seed (validation; test untouched):\n' + df.round(4).to_string(index=False))

    summ = df.groupby('run')[metrics].agg(['mean', 'std']).round(4)
    print('\nMean +/- sd over seeds:\n' + summ.T.to_string())

    w = df.pivot(index='seed', columns='run')
    diffs = []
    for name, (x, y) in {'B-A (normalization)': ('B', 'A'), 'C-B (focal loss)': ('C', 'B'),
                         'D-B (fine labels)': ('D', 'B')}.items():
        for m in ('sel_at_best', 'sel_median_full', 'PAC_auprc', 'PVC_auprc', 'PAC_auprc_median_full',
                  'PVC_auprc_median_full'):
            if x not in w[m] or y not in w[m]:
                continue
            d = (w[m][x] - w[m][y]).dropna()
            diffs.append({'comparison': name, 'metric': m, 'n_seeds': len(d), 'mean_diff': d.mean(), 'sd': d.std(),
                          'min': d.min(), 'max': d.max(), 'same_sign_all_seeds': bool((d > 0).all() or (d < 0).all())})
    dd = pd.DataFrame(diffs)
    print('\nPaired differences within seed:\n' + dd.round(4).to_string(index=False))

    out = os.path.join(RES, '_comparison')
    os.makedirs(out, exist_ok=True)
    tag = '' if a.lead == 'II' else f'_{a.lead.lower()}'
    df.to_csv(os.path.join(out, f'seed_runs{tag}.csv'), index=False)
    summ.to_csv(os.path.join(out, f'seed_summary{tag}.csv'))
    dd.to_csv(os.path.join(out, f'seed_paired_diffs{tag}.csv'), index=False)
    print(f'\nwrote {out}/seed_*{tag}.csv')


if __name__ == '__main__':
    main()
