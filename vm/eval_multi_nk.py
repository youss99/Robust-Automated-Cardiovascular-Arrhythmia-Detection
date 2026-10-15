"""Evaluate the multi-database NeuroKit-centred beat models (cluster runs, 2026-10-15) next to the INCART-only ladder.

Models per lead (II / V5): cluster_B, vm_nk, vm_nk_jit50 (INCART only) and multi_nk, multi_nk_jit50 (multi-database
caches from src/incart_beat/prepare_multi_nk.py, recipe B, --steps_per_epoch 2000, jitter 0 / 50 ms).
Part A  validation: INCART NeuroKit-centred val (prepare_incart_nk caches) and multi-database val
        (beat_caches/multi_{II,V5}_nk); thresholds at sensitivity targets on each. Cross-check: the multi models'
        validation probabilities must equal the cluster's val_predictions.csv.
Part B  multi-database TEST split per source database, system level (reference beats NeuroKit missed count as misses):
        AUPRC, and Se / PPV / F1 / FP per hour at validation thresholds; also per test patient.
Part C  patients P21 / P27 / P36: per-beat probabilities on the same NeuroKit beats as every earlier run.
Outputs: results/fine-tune/saved_models/incart_multi_eval/{test_per_source.csv, test_per_patient.csv, val_check.txt}
         GIT/3_patients_pvc_apb/v3_eval_2026-10-14/ladder_beats/{multi_nk,multi_nk_jit50}_{II,CM5}_P{21,27,36}.csv
         and ladder_beats/thresholds_multi.json
Run from the repo root with the repo venv (GPU):  .venv/bin/python vm/eval_multi_nk.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.getcwd())
from src.incart_beat import train as T
from src.incart_beat.model import load_beat_classifier, predict_beats

D = '/home/student/GIT/Project_1/Datasets'; R = 'results/fine-tune/saved_models'
MULTI = '/home/student/Desktop/Research Data/DATASET/beat_caches'
OUT = f'{R}/incart_multi_eval'; LB = '/home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14/ladder_beats'
MODELS = {'cluster_B': 'vm/incart_{l}_beat_B/incart_{l}_beat_B.pt',
          'vm_nk': f'{R}/incart_vm_nk/incart_{{l}}_beat_B/best.pt', 'vm_nk_jit50': f'{R}/incart_vm_nk_jit50/incart_{{l}}_beat_B/best.pt',
          'multi_nk': f'{R}/incart_multi_nk/incart_{{l}}_beat_B/best.pt', 'multi_nk_jit50': f'{R}/incart_multi_nk_jit50/incart_{{l}}_beat_B/best.pt'}
NEW = ('multi_nk', 'multi_nk_jit50')
TARGETS = {'PVC': (0.85, 0.90, 0.95), 'PAC': (0.50, 0.70, 0.85)}
dev = torch.device('cuda'); os.makedirs(OUT, exist_ok=True)


def tau_at_se(s, target):
    s = np.sort(s)[::-1]; return float(s[int(np.ceil(target * len(s))) - 1])


def counts(y, s, tau, n_missed, hours):
    fire = s >= tau; tp = int((fire & y).sum()); fp = int((fire & ~y).sum()); n = int(y.sum()) + n_missed
    return dict(tau=tau, TP=tp, FP=fp, FN=n - tp, n=n, Se=tp / n if n else np.nan, PPV=tp / (tp + fp) if tp + fp else np.nan,
                F1=2 * tp / (tp + fp + n) if n else np.nan, FP_per_h=fp / hours if hours else np.nan)


def load(m, lead):
    return load_beat_classifier(MODELS[m].format(l=lead.lower()), dev)


thresholds, rows_src, rows_pat, check = {}, [], [], []
for lead in ('II', 'V5'):
    # ---------------------------------------------------------------- Part A: validation thresholds
    probs = {}
    src = T.WindowSource(f'{D}/incart_beat_lead{lead}_100hz_nk', dev); dfv, rv, sv, _ = src.split('val')
    for m in MODELS:
        model = load(m, lead)[0]; probs[(m, 'incart')] = (dfv.label.to_numpy(), T.predict(model, src, rv, sv, 512))
        del model; torch.cuda.empty_cache()
    del src; torch.cuda.empty_cache()
    src = T.WindowSource(f'{MULTI}/multi_{lead}_nk', dev)
    dfm, rm, sm, _ = src.split('val'); dft, rt, st, _ = src.split('test')
    segs = pd.read_csv(f'{MULTI}/multi_{lead}_nk/segments.csv')
    rec_len = segs.assign(end=segs.rec_start + segs.length).groupby(['source_db', 'source_record', 'patient', 'split']).end.max().reset_index()
    missed = pd.read_csv(f'{MULTI}/multi_{lead}_nk/nk_missed_reference_beats.csv')
    for m in MODELS:
        model, meta = load(m, lead)
        pm = T.predict(model, src, rm, sm, 512); probs[(m, 'multi')] = (dfm.label.to_numpy(), pm)
        pt = T.predict(model, src, rt, st, 512)
        del model; torch.cuda.empty_cache()
        if m in NEW:                                    # cross-check against the cluster's own validation predictions
            cp = pd.read_csv(f"{os.path.dirname(MODELS[m].format(l=lead.lower()))}/val_predictions.csv", usecols=['beat_id', 'p_PAC', 'p_PVC'])
            j = dfm[['beat_id']].assign(p_PAC=pm[:, 1], p_PVC=pm[:, 2]).merge(cp, on='beat_id', suffixes=('', '_cl'))
            d = max(np.abs(j.p_PAC - j.p_PAC_cl).max(), np.abs(j.p_PVC - j.p_PVC_cl).max())
            check.append(f'{lead} {m}: best epoch {meta["epoch"]}, {len(j)}/{len(dfm)} val windows matched, max |VM - cluster| prob {d:.2e}')
            print(check[-1], flush=True)
        for cls, k in (('PVC', 2), ('PAC', 1)):
            th = {}
            for vs in ('incart', 'multi'):
                y, p = probs[(m, vs)]
                for t in TARGETS[cls]:
                    th[f'{vs}_se{round(t * 100)}'] = tau_at_se(p[y == k, k], t)
            th['own_prefix'] = 'multi' if m in NEW else 'incart'
            thresholds[f'{m}|{lead}|{cls}'] = th
            # ------------------------------------------------------------ Part B: test per source / patient
            yt = dft.label.to_numpy() == k; s = pt[:, k]
            for db, g in dft.groupby('source_db'):
                ix = g.index.to_numpy(); hrs = rec_len[(rec_len.source_db == db) & (rec_len.split == 'test')].end.sum() / 100 / 3600
                nm = int(((missed.source_db == db) & (missed.split == 'test') & (missed.label == k)).sum())
                base = dict(lead=lead, model=m, cls=cls, source_db=db, test_hours=round(hrs, 2), n_detected=int(yt[ix].sum()), nk_missed=nm,
                            AUPRC=average_precision_score(yt[ix], s[ix]) if yt[ix].any() else np.nan)
                for rule in ('incart_se90', 'multi_se90', 'incart_se85', 'incart_se50', 'multi_se50'):
                    if rule in th:
                        rows_src.append({**base, 'rule': rule, **counts(yt[ix], s[ix], th[rule], nm, hrs)})
                rule = 'multi_se90' if cls == 'PVC' else 'multi_se50'
                for pat, gp in g.groupby('patient'):
                    ip = gp.index.to_numpy(); hp = rec_len[rec_len.patient == pat].end.sum() / 100 / 3600
                    nmp = int(((missed.patient == pat) & (missed.label == k)).sum())
                    rows_pat.append(dict(lead=lead, model=m, cls=cls, source_db=db, patient=pat, rule=rule, **counts(yt[ip], s[ip], th[rule], nmp, hp)))
        print(f'{lead} {m}: validation and test done', flush=True)
    del src; torch.cuda.empty_cache()
pd.DataFrame(rows_src).to_csv(f'{OUT}/test_per_source.csv', index=False)
pd.DataFrame(rows_pat).to_csv(f'{OUT}/test_per_patient.csv', index=False)
open(f'{OUT}/val_check.txt', 'w').write('\n'.join(check) + '\n')
json.dump(thresholds, open(f'{LB}/thresholds_multi.json', 'w'), indent=1)

# ---------------------------------------------------------------- Part C: patients (same NeuroKit beats as before)
EDF = {21: ('/home/student/GIT/Patient 21/aligned_leads_8103_8123.edf', {'II': 1, 'CM5': 0}),
       27: ('/home/student/Downloads/aligned_leads_8075_8080_edited_edited_edited_edited_edited_edited_edited_edited_edited_edited_edited.edf', {'II': 0, 'CM5': 1}),
       36: ('/home/student/GIT/3_patients_pvc_apb/patient 36/source_bundle_2026-10-14/aligned_leads_8108_8120.edf', {'II': 0, 'CM5': 1})}


def reader(path, ch):                                   # as vm/eval_incart_ladder.py
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


def patient_beats(p, lead):                             # as vm/eval_incart_ladder.py
    if p == 36:
        b = pd.read_csv('/home/student/GIT/3_patients_pvc_apb/patient 36/p36_beats.csv'); return b[b.lead == lead][['time_s', 'span']]
    f = (f'/home/student/GIT/incart_ii_beat_B/patient_eval/beats_P{p}.csv' if lead == 'II' else
         f'/home/student/GIT/incart_v5_beat_B/patient_eval/beats_CM5_P{p}.csv')
    return pd.read_csv(f).dropna(subset=['p_PVC'])[['time_s', 'span']]


for lead, ml in (('II', 'II'), ('CM5', 'V5')):
    for m in NEW:
        model = load(m, ml)[0]
        for p, (edf, chans) in EDF.items():
            read = reader(edf, chans[lead]); B = patient_beats(p, lead).sort_values('time_s').reset_index(drop=True); out = []
            for sp, g in B.groupby('span'):
                x, t0 = read(g.time_s.min() - 30, g.time_s.max() + 30)
                o = predict_beats(model, x, 250, 'mV', np.rint((g.time_s.to_numpy() - t0) * 250).astype(np.int64), device=dev, batch_size=512)
                out.append(pd.DataFrame({'time_s': g.time_s.to_numpy(), 'span': sp, 'p_PAC': o.p_PAC.to_numpy(), 'p_PVC': o.p_PVC.to_numpy()}))
            pd.concat(out).to_csv(f'{LB}/{m}_{lead}_P{p}.csv', index=False)
        print(f'patients {lead} {m}: done', flush=True)
        del model; torch.cuda.empty_cache()
print('\n'.join(check))
