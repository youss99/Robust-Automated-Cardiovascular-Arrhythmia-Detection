"""Average of the two multi-database runs (multi_nk + multi_nk_jit50) scored as one model, 'multi_ens' (2026-10-15).

Post-hoc check, decided after the two runs disagreed on individual patient cells (MULTI_ANALYSIS_2026-10-15.md): the
softmax probabilities are averaged, and thresholds come from the averaged validation probabilities exactly as for a
single model (vm/eval_multi_nk.py).
Writes: incart_multi_eval/test_per_source_ens.csv, incart_multi_eval/probs_<lead>.npz (validation / test probabilities
of both runs), ladder_beats/multi_ens_<lead>_P<p>.csv, and 'multi_ens|<lead>|<class>' entries in
ladder_beats/thresholds_multi.json. Then re-run /usr/bin/python3 v3_multi.py for the patient tables.
Run from the repo root with the repo venv (GPU):  .venv/bin/python vm/eval_multi_ensemble.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.getcwd())
from src.incart_beat import train as T
from src.incart_beat.model import load_beat_classifier

D = '/home/student/GIT/Project_1/Datasets'; R = 'results/fine-tune/saved_models'
MULTI = '/home/student/Desktop/Research Data/DATASET/beat_caches'
OUT = f'{R}/incart_multi_eval'; LB = '/home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14/ladder_beats'
RUNS = {'multi_nk': f'{R}/incart_multi_nk/incart_{{l}}_beat_B/best.pt', 'multi_nk_jit50': f'{R}/incart_multi_nk_jit50/incart_{{l}}_beat_B/best.pt'}
TARGETS = {'PVC': (0.85, 0.90, 0.95), 'PAC': (0.50, 0.70, 0.85)}
dev = torch.device('cuda')


def tau_at_se(s, target):                               # as vm/eval_multi_nk.py
    s = np.sort(s)[::-1]; return float(s[int(np.ceil(target * len(s))) - 1])


def counts(y, s, tau, n_missed, hours):                 # as vm/eval_multi_nk.py
    fire = s >= tau; tp = int((fire & y).sum()); fp = int((fire & ~y).sum()); n = int(y.sum()) + n_missed
    return dict(tau=tau, TP=tp, FP=fp, FN=n - tp, n=n, Se=tp / n if n else np.nan, PPV=tp / (tp + fp) if tp + fp else np.nan,
                F1=2 * tp / (tp + fp + n) if n else np.nan, FP_per_h=fp / hours if hours else np.nan)


TH = json.load(open(f'{LB}/thresholds_multi.json')); rows = []
for lead in ('II', 'V5'):
    P = {}
    for vs, cache in (('incart', f'{D}/incart_beat_lead{lead}_100hz_nk'), ('multi', f'{MULTI}/multi_{lead}_nk')):
        src = T.WindowSource(cache, dev); splits = ('val',) if vs == 'incart' else ('val', 'test')
        dfs = {s: src.split(s) for s in splits}
        for m, path in RUNS.items():
            model = load_beat_classifier(path.format(l=lead.lower()), dev)[0]
            for s in splits:
                df, r, st, _ = dfs[s]; P[(vs, s, m)] = T.predict(model, src, r, st, 512)
            del model; torch.cuda.empty_cache()
        for s in splits:
            P[(vs, s, 'y')] = dfs[s][0].label.to_numpy()
        if vs == 'multi':
            dft = dfs['test'][0]
        del src, dfs; torch.cuda.empty_cache()
    np.savez(f'{OUT}/probs_{lead}.npz', **{'|'.join(k): v for k, v in P.items()})
    # single-run thresholds must reproduce eval_multi_nk.py
    for m in RUNS:
        assert abs(tau_at_se(P[('multi', 'val', m)][P[('multi', 'val', 'y')] == 2, 2], 0.90) - TH[f'{m}|{lead}|PVC']['multi_se90']) < 1e-5
    ens = {k: (P[k[:2] + ('multi_nk',)] + P[k[:2] + ('multi_nk_jit50',)]) / 2 for k in [('incart', 'val'), ('multi', 'val'), ('multi', 'test')]}
    segs = pd.read_csv(f'{MULTI}/multi_{lead}_nk/segments.csv')
    rec_len = segs.assign(end=segs.rec_start + segs.length).groupby(['source_db', 'source_record', 'split']).end.max().reset_index()
    missed = pd.read_csv(f'{MULTI}/multi_{lead}_nk/nk_missed_reference_beats.csv')
    for cls, k in (('PVC', 2), ('PAC', 1)):
        th = {f'{vs}_se{round(t * 100)}': tau_at_se(ens[(vs, 'val')][P[(vs, 'val', 'y')] == k, k], t) for vs in ('incart', 'multi') for t in TARGETS[cls]}
        th['own_prefix'] = 'multi'; TH[f'multi_ens|{lead}|{cls}'] = th
        yt = P[('multi', 'test', 'y')] == k; s = ens[('multi', 'test')][:, k]
        for db, g in dft.groupby('source_db'):
            ix = g.index.to_numpy(); hrs = rec_len[(rec_len.source_db == db) & (rec_len.split == 'test')].end.sum() / 100 / 3600
            nm = int(((missed.source_db == db) & (missed.split == 'test') & (missed.label == k)).sum())
            base = dict(lead=lead, model='multi_ens', cls=cls, source_db=db, test_hours=round(hrs, 2), n_detected=int(yt[ix].sum()), nk_missed=nm,
                        AUPRC=average_precision_score(yt[ix], s[ix]) if yt[ix].any() else np.nan)
            for rule in ('incart_se90', 'multi_se90', 'incart_se85', 'incart_se50', 'multi_se50'):
                if rule in th:
                    rows.append({**base, 'rule': rule, **counts(yt[ix], s[ix], th[rule], nm, hrs)})
    print(f'{lead}: ensemble validation / test done', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/test_per_source_ens.csv', index=False)
json.dump(TH, open(f'{LB}/thresholds_multi.json', 'w'), indent=1)
for lead in ('II', 'CM5'):
    for p in (21, 27, 36):
        a = pd.read_csv(f'{LB}/multi_nk_{lead}_P{p}.csv'); b = pd.read_csv(f'{LB}/multi_nk_jit50_{lead}_P{p}.csv')
        assert len(a) == len(b) and np.array_equal(a.time_s.to_numpy(), b.time_s.to_numpy())
        a[['p_PAC', 'p_PVC']] = (a[['p_PAC', 'p_PVC']].to_numpy() + b[['p_PAC', 'p_PVC']].to_numpy()) / 2
        a.to_csv(f'{LB}/multi_ens_{lead}_P{p}.csv', index=False)
print('wrote ensemble patient probabilities, thresholds and test rows')
