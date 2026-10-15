"""Evaluate the INCART beat-model centring ladder (2026-10-15) the way the models are deployed.

Models (per lead II / V5): cluster_B (vm/ bundle), vm_ref, vm_nk, vm_nk_jit50 (vm/run_incart_nk_ladder.sh).
Part A, INCART: every model scored on the NeuroKit-centred validation and test windows (prepare_incart_nk caches) and,
  for the reproduction check, on the reference-centred ones. Thresholds come from the NeuroKit-centred VALIDATION split:
  sensitivity 0.85 (PVC) / 0.50 (PAC) on detected beats, and best F1. Test metrics are system-level: reference beats
  NeuroKit never detected count as misses.  -> results/fine-tune/saved_models/incart_vm_logs/incart_ladder_eval.csv
Part B, patients P21 / P27 / P36: per-beat probabilities of every model on the same NeuroKit beats the earlier runs used
  -> GIT/3_patients_pvc_apb/v3_eval_2026-10-14/ladder_beats/<model>_<lead>_P<p>.csv, plus thresholds.json.
Run from the repo root with the repo venv (GPU):  .venv/bin/python vm/eval_incart_ladder.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.getcwd())
from src.incart_beat import preprocessing as pp, train as T
from src.incart_beat.model import load_beat_classifier, predict_beats

D = '/home/student/GIT/Project_1/Datasets'; R = 'results/fine-tune/saved_models'
OUT_A = f'{R}/incart_vm_logs/incart_ladder_eval.csv'
OUT_B = '/home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14/ladder_beats'
MODELS = {'cluster_B': 'vm/incart_{l}_beat_B/incart_{l}_beat_B.pt', 'vm_ref': f'{R}/incart_vm/incart_{{l}}_beat_B/best.pt',
          'vm_nk': f'{R}/incart_vm_nk/incart_{{l}}_beat_B/best.pt', 'vm_nk_jit50': f'{R}/incart_vm_nk_jit50/incart_{{l}}_beat_B/best.pt'}
TARGET = {'PVC': 0.85, 'PAC': 0.50}
dev = torch.device('cuda')
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--models', default=','.join(MODELS)); _ap.add_argument('--tag', default='')
_args = _ap.parse_args()
MODELS = {k: v for k, v in MODELS.items() if k in _args.models.split(',')}
OUT_A = OUT_A.replace('.csv', f'{_args.tag}.csv')


def tau_at_se(s, target):
    s = np.sort(s)[::-1]; return float(s[int(np.ceil(target * len(s))) - 1])


def best_f1_tau(y, s, n_pos):
    o = np.argsort(-s, kind='stable'); s, y = s[o], y[o]; tp = np.cumsum(y); fp = np.cumsum(~y)
    last = np.r_[s[1:] != s[:-1], True]; f = np.where(last, 2 * tp / (tp + fp + n_pos), -1); return float(s[int(np.argmax(f))])


def scores(y, s, tau, n_missed):
    fire = s >= tau; tp = int((fire & y).sum()); fp = int((fire & ~y).sum()); n = int(y.sum()) + n_missed
    return dict(tau=tau, TP=tp, FP=fp, n=n, Se=tp / n if n else np.nan, PPV=tp / (tp + fp) if tp + fp else np.nan,
                F1=2 * tp / (tp + fp + n) if n else np.nan)


# ------------------------------------------------------------------ Part A: INCART
rows, thresholds = [], {}
for lead in ('II', 'V5'):
    l = lead.lower()
    caches = {'nk': f'{D}/incart_beat_lead{lead}_100hz_nk', 'ref': f'{D}/incart_beat_lead{lead}_100hz'}
    srcs = {k: T.WindowSource(c, dev) for k, c in caches.items()}
    missed = pd.read_csv(f"{caches['nk']}/nk_missed_reference_beats.csv")
    for mname, path in MODELS.items():
        path = path.format(l=l)
        if not os.path.exists(path):
            print('missing', path); continue
        model = load_beat_classifier(path, dev)[0]
        for centre, src in srcs.items():
            P = {}
            for split in ('val', 'test'):
                df, r_, s_, y_ = src.split(split)
                P[split] = (df.label.values, T.predict(model, src, r_, s_, 512))
            for cls, k in (('PVC', 2), ('PAC', 1)):
                yv, pv = P['val'][0] == k, P['val'][1][:, k]
                miss_v = int(((missed.split == 'val') & (missed.label == k)).sum()) if centre == 'nk' else 0
                miss_t = int(((missed.split == 'test') & (missed.label == k)).sum()) if centre == 'nk' else 0
                taus = {f'val_se{round(TARGET[cls] * 100)}': tau_at_se(pv[yv], TARGET[cls]), 'val_bestF1': best_f1_tau(yv, pv, int(yv.sum()) + miss_v)}
                if centre == 'nk':
                    thresholds[f'{mname}|{lead}|{cls}'] = taus
                yt, pt = P['test'][0] == k, P['test'][1][:, k]
                for rule, tau in taus.items():
                    rows.append(dict(lead=lead, model=mname, centres=centre, cls=cls, rule=rule, AUPRC_test=average_precision_score(yt, pt),
                                     AUPRC_val=average_precision_score(yv, pv), **scores(yt, pt, tau, miss_t)))
        print(f'{lead} {mname}: done', flush=True)
        del model; torch.cuda.empty_cache()
    del srcs; torch.cuda.empty_cache()
A = pd.DataFrame(rows); A.to_csv(OUT_A, index=False)
os.makedirs(OUT_B, exist_ok=True)
json.dump(thresholds, open(f'{OUT_B}/thresholds.json', 'w'), indent=1)

# ------------------------------------------------------------------ Part B: patients (same NeuroKit beats as before)
EDF = {21: ('/home/student/GIT/Patient 21/aligned_leads_8103_8123.edf', {'II': 1, 'CM5': 0}),
       27: ('/home/student/Downloads/aligned_leads_8075_8080_edited_edited_edited_edited_edited_edited_edited_edited_edited_edited_edited.edf', {'II': 0, 'CM5': 1}),
       36: ('/home/student/GIT/3_patients_pvc_apb/patient 36/source_bundle_2026-10-14/aligned_leads_8108_8120.edf', {'II': 0, 'CM5': 1})}


def reader(path, ch):
    h = open(path, 'rb').read(256); ns = int(h[252:256]); hb = 256 + 256 * ns; b = open(path, 'rb').read(hb)[256:]
    fld = lambda off, w: [b[off * ns + i * w: off * ns + (i + 1) * w].decode().strip() for i in range(ns)]
    pmin = [float(v) for v in fld(104, 8)]; pmax = [float(v) for v in fld(112, 8)]; dmin = [float(v) for v in fld(120, 8)]
    dmax = [float(v) for v in fld(128, 8)]; spr = [int(v) for v in fld(216, 8)]; nrec = int(h[236:244])
    off = sum(spr[:ch]); g = (pmax[ch] - pmin[ch]) / (dmax[ch] - dmin[ch])
    mm = np.memmap(path, dtype='<i2', mode='r', offset=hb, shape=(nrec, sum(spr)))
    def read(t0, t1):
        r0 = max(0, int(np.floor(t0 / 0.1))); r1 = min(nrec, int(np.ceil(t1 / 0.1)))
        x = (np.asarray(mm[r0:r1, off:off + spr[ch]], np.float64).reshape(-1) - dmin[ch]) * g + pmin[ch]
        return np.where(np.isfinite(x), x, 0.0), r0 * 0.1
    return read


def patient_beats(p, lead):
    if p == 36:
        b = pd.read_csv('/home/student/GIT/3_patients_pvc_apb/patient 36/p36_beats.csv'); return b[b.lead == lead][['time_s', 'span']]
    f = (f'/home/student/GIT/incart_ii_beat_B/patient_eval/beats_P{p}.csv' if lead == 'II' else
         f'/home/student/GIT/incart_v5_beat_B/patient_eval/beats_CM5_P{p}.csv')
    return pd.read_csv(f).dropna(subset=['p_PVC'])[['time_s', 'span']]


for lead, ml in (('II', 'ii'), ('CM5', 'v5')):
    for mname, path in MODELS.items():
        path = path.format(l=ml)
        if not os.path.exists(path):
            continue
        model = load_beat_classifier(path, dev)[0]
        for p, (edf, chans) in EDF.items():
            read = reader(edf, chans[lead]); B = patient_beats(p, lead).sort_values('time_s').reset_index(drop=True); out = []
            for sp, g in B.groupby('span'):
                x, t0 = read(g.time_s.min() - 30, g.time_s.max() + 30)
                o = predict_beats(model, x, 250, 'mV', np.rint((g.time_s.to_numpy() - t0) * 250).astype(np.int64), device=dev, batch_size=512)
                out.append(pd.DataFrame({'time_s': g.time_s.to_numpy(), 'span': sp, 'p_PAC': o.p_PAC.to_numpy(), 'p_PVC': o.p_PVC.to_numpy()}))
            pd.concat(out).to_csv(f'{OUT_B}/{mname}_{lead}_P{p}.csv', index=False)
        print(f'patients {lead} {mname}: done', flush=True)
        del model; torch.cuda.empty_cache()
pd.set_option('display.width', 220)
print(A[A.centres == 'nk'].pivot_table(index=['lead', 'cls', 'model'], columns='rule', values=['F1', 'Se', 'PPV', 'AUPRC_test'], aggfunc='first').round(3).to_string())
