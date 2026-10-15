"""
Independent audit of the INCART central-beat pipeline (CPU only). Re-derives results by separate code paths:

  1. data    : raw WFDB -> uV -> resample_poly(100/257) re-implemented here and compared with the cache; labels,
               split and annotation mapping re-checked against the specification (not against pp constants).
  2. timing  : QRS-energy peak near each annotation at 257 Hz vs at 100 Hz -> catches resampling/index drift.
  3. metrics : AUPRC (own average-precision code), AUROC (rank formula), precision/recall/F1/confusion recomputed
               from the saved per-beat predictions; selected epoch, head reload and recipe checked against history.
  4. paths   : saved B models re-run through the patient-inference path (raw record -> predict_beats) must
               reproduce the training-time validation probabilities.
  5. bundles : vm/ bundles byte-identical to the repo code and best.pt weights, and runnable from their own folder.

Exit code 1 if any hard check FAILs (WARN does not fail). Run from the repo root:
  python -m src.incart_beat.checks.audit --homer_csv <10-min Homer CSV>
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sgn
from scipy.stats import rankdata

PROJ = os.path.abspath('..')
DATASETS = os.path.join(PROJ, 'Datasets')
RES = 'results/fine-tune/saved_models/incart'
# Specification (from the task brief), written out independently of preprocessing.py
SPEC_MAP = {'N': 0, 'R': 0, 'B': 0, 'j': 0, 'n': 0, 'A': 1, 'V': 2}
SPEC_EXCLUDED = {'F', 'Q', 'S'}
SPEC_VAL, SPEC_TEST = {3, 6, 16, 24, 31}, {2, 10, 17, 22, 27}
RECIPE = {'head_epochs': 20, 'full_epochs': 70, 'lr': 1e-3, 'weight_decay': 0.05, 'layer_decay': 0.65,
          'batch_size': 128}
RESULTS = []


def check(name, ok, detail='', hard=True):
    tag = 'PASS' if ok else ('FAIL' if hard else 'WARN')
    print(f'[{tag}] {name}: {detail}', flush=True)
    RESULTS.append((tag, name, detail))


def section(title):
    print(f'\n===== {title}', flush=True)


# ------------------------------------------------------------------------------------------------ 1-2 data, timing
def qrs_peak_offsets(x, fs, centres, half_s=0.15):
    """Offset (s) of the 5-20 Hz energy maximum within +/-half_s of each centre (zero-phase filter)."""
    b, a = sgn.butter(2, [5, 20], btype='band', fs=fs)
    e = np.abs(sgn.filtfilt(b, a, x))
    h = int(round(half_s * fs))
    ok = (centres - h >= 0) & (centres + h < len(e))
    idx = centres[ok, None] + np.arange(-h, h + 1)[None, :]
    off = np.full(len(centres), np.nan)
    off[ok] = (np.argmax(e[idx], axis=1) - h) / fs
    return off


def audit_data(lead):
    section(f'1-2. Data, labels, split, timing (lead {lead})')
    cache = os.path.join(DATASETS, f'incart_beat_lead{lead}_100hz')
    signals = np.load(os.path.join(cache, 'signals_uV_100hz.npy'))
    recs = pd.read_csv(os.path.join(cache, 'records.csv'))
    beats = pd.read_csv(os.path.join(cache, 'beats.csv'), keep_default_na=False)

    # split
    p2s = recs.groupby('patient').split.agg(set)
    check('each patient in exactly one split', (p2s.map(len) == 1).all())
    got = {s: set(recs.patient[recs.split == s]) for s in ('train', 'val', 'test')}
    check('val/test patients match the brief', got['val'] == SPEC_VAL and got['test'] == SPEC_TEST,
          f"val {sorted(got['val'])} test {sorted(got['test'])}")
    check('train = all other patients', got['train'] == set(recs.patient) - SPEC_VAL - SPEC_TEST and
          len(got['train']) == 22, f'{len(got["train"])} train patients')
    check('beats.csv split consistent with records', (beats.split == beats.record.map(dict(zip(recs.record,
                                                                                             recs.split)))).all())

    sig_err, n_ann_bad, lab_bad, map_bad, offs = 0.0, 0, 0, 0, []
    for r in recs.itertuples():
        path = os.path.join(DATASETS, 'incartdb', r.record)
        x, fields = wfdb.rdsamp(path, channel_names=[lead])
        fs = fields['fs']
        assert fields['units'][0] == 'mV', fields['units']
        x_uv = x[:, 0].astype(np.float64) * 1000.0                       # mV -> uV, once
        y = sgn.resample_poly(x_uv, 100, 257).astype(np.float32)          # independent of pp.resample_ratio
        sig_err = max(sig_err, float(np.abs(y - signals[r.row]).max()))

        ann = wfdb.rdann(path, 'atr')
        b = beats[beats.record == r.record].sort_values('ann_index')
        n_ann_bad += int(len(b) != len(ann.sample) or list(b.symbol) != list(ann.symbol)
                         or not np.array_equal(b.sample_native.values, ann.sample))
        # 257 is prime and odd, so s*100/257 is never exactly x.5: floor(+0.5) and np.rint cannot disagree
        c100 = np.floor(ann.sample * 100 / 257 + 0.5).astype(int)
        map_bad += int(np.sum(c100 != b.sample_100hz.values))
        full = (c100 - 500 >= 0) & (c100 + 500 <= len(y))
        exp_status = np.where([s in SPEC_MAP for s in ann.symbol], np.where(full, 'included', 'incomplete_window'),
                              np.where([s in SPEC_EXCLUDED for s in ann.symbol], 'excluded_symbol',
                                       'unmapped_symbol'))
        exp_label = np.array([SPEC_MAP.get(s, -1) for s in ann.symbol])
        lab_bad += int(np.sum(exp_status != b.status.values) + np.sum(exp_label != b.label.values))

        inc = (b.status == 'included').values
        o_nat = qrs_peak_offsets(x_uv, fs, ann.sample[inc])
        o_100 = qrs_peak_offsets(signals[r.row].astype(np.float64), 100, b.sample_100hz.values[inc])
        pos = ann.sample[inc] / len(x_uv)
        offs.append(pd.DataFrame({'record': r.record, 'symbol': np.asarray(ann.symbol)[inc], 'pos': pos,
                                  'off_native_ms': o_nat * 1e3, 'off_100_ms': o_100 * 1e3}))

    check('cached signals == independent WFDB->uV->resample_poly', sig_err < 0.05,
          f'max |diff| {sig_err:.4f} uV over {len(recs)} records')
    check('annotation samples/symbols == raw .atr', n_ann_bad == 0, f'{n_ann_bad} records differ')
    check('100 Hz annotation mapping == round(s*100/257)', map_bad == 0, f'{map_bad} beats differ')
    check('labels/status match the brief (A=PAC, V=PVC, N/R/B/j/n=bg, F/Q/S excluded)', lab_bad == 0,
          f'{lab_bad} mismatches')

    inc = beats[beats.status == 'included']
    cls = inc.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(cls.rename(columns={0: 'bg', 1: 'PAC', 2: 'PVC'}).to_string())
    tr_pac = inc[(inc.split == 'train') & (inc.label == 1)].patient.value_counts()
    check('PAC concentration claim (top-2 train patients)', True,
          f'{tr_pac.iloc[:2].sum()}/{tr_pac.sum()} = {100 * tr_pac.iloc[:2].sum() / tr_pac.sum():.1f}% '
          f'(patients {list(tr_pac.index[:2])})', hard=False)

    o = pd.concat(offs).dropna()
    d = o.off_100_ms - o.off_native_ms
    within = float((d.abs() <= 15).mean())
    early, late = d[o.pos < 0.1].median(), d[o.pos > 0.9].median()
    print(f'INCART annotation vs QRS-energy peak at 257 Hz (ms): median {o.off_native_ms.median():.1f}, '
          f'IQR [{o.off_native_ms.quantile(.25):.1f}, {o.off_native_ms.quantile(.75):.1f}]; by symbol: '
          + ', '.join(f'{s} {g.off_native_ms.median():.1f}' for s, g in o.groupby('symbol') if len(g) > 50))
    check('no systematic timing shift from resampling (peak offset 100 Hz vs 257 Hz)', abs(d.median()) <= 5,
          f'median diff {d.median():.1f} ms over {len(d)} beats')
    check('per-beat timing agreement (noise allowed)', within >= 0.9,
          f'{100 * within:.1f}% of beats within +/-15 ms', hard=False)
    check('no timing drift across 30 min records', abs(early - late) <= 5,
          f'median diff first 10% {early:.1f} ms vs last 10% {late:.1f} ms')
    return beats


# ------------------------------------------------------------------------------------------------ 3 metrics
def ap_score(y, s):
    o = np.argsort(-s, kind='mergesort')
    y, s = y[o], s[o]
    thr = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[thr]
    prec, rec = tp / (thr + 1), tp / y.sum()
    return float(np.sum((rec - np.r_[0, rec[:-1]]) * prec))


def auroc(y, s):
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def audit_run(lead, run, split, beats, log_text):
    d = os.path.join(RES, f'incart_{lead.lower()}_beat_{run}')
    p = pd.read_csv(os.path.join(d, f'{split}_predictions.csv'), keep_default_na=False)
    exp = beats[(beats.status == 'included') & (beats.split == split)]
    check(f'{lead}-{run} {split}: predictions cover exactly the {split} windows',
          set(p.beat_id) == set(exp.beat_id) and len(p) == len(exp), f'{len(p)} rows')
    m = p.merge(exp[['beat_id', 'label', 'symbol', 'patient']], on='beat_id', suffixes=('', '_exp'))
    check(f'{lead}-{run} {split}: labels/patients match cache', (m.label == m.label_exp).all() and
          (m.patient == m.patient_exp).all() and (m.symbol == m.symbol_exp).all())
    P = p[['p_background', 'p_PAC', 'p_PVC']].to_numpy(np.float64)
    y, pred = p.label.to_numpy(), P.argmax(1)
    check(f'{lead}-{run} {split}: saved pred == argmax(probs)', (pred == p.pred.to_numpy()).all())

    pc = pd.read_csv(os.path.join(d, f'{split}_per_class.csv')).set_index('class')
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, pred), 1)
    saved_cm = pd.read_csv(os.path.join(d, f'{split}_confusion.csv'), index_col=0).to_numpy()
    check(f'{lead}-{run} {split}: confusion matrix recomputed', np.array_equal(cm, saved_cm), str(cm.tolist()))
    worst = 0.0
    for k, c in enumerate(['background', 'PAC', 'PVC']):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        mine = {'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'f1': 2 * tp / (2 * tp + fp + fn),
                'auprc': ap_score((y == k).astype(int), P[:, k]), 'auroc': auroc((y == k).astype(int), P[:, k])}
        for q in mine:  # NaN on both sides (undefined metric) counts as agreement, NaN on one side does not
            a_, b_ = float(mine[q]), float(pc.loc[c, q])
            worst = max(worst, 0.0 if (np.isnan(a_) and np.isnan(b_)) else (abs(a_ - b_) if not (np.isnan(a_) or
                                                                                              np.isnan(b_)) else np.inf))
    check(f'{lead}-{run} {split}: precision/recall/F1/AUPRC/AUROC recomputed', worst < 1e-4,
          f'max |diff| {worst:.2e} (CSV rounding of probabilities allowed)')

    if split == 'val':
        cfg = json.load(open(os.path.join(d, 'config.json')))
        tc = cfg['train_config']
        check(f'{lead}-{run}: thesis recipe settings', all(abs(tc[k] - v) < 1e-12 for k, v in RECIPE.items()),
              ', '.join(f'{k}={tc[k]}' for k in RECIPE))
        h = pd.read_csv(os.path.join(d, 'history.csv'))
        check(f'{lead}-{run}: 90 epochs, 20 head then 70 full', len(h) == 90 and (h.phase[:20] == 'head').all()
              and (h.phase[20:] == 'full').all())
        best_ep = int(h.sel_mean_auprc_pac_pvc.idxmax())
        sel_log = re.search(rf'\[{run}\] selected epoch (\d+)', log_text)
        check(f'{lead}-{run}: selected epoch = argmax of validation history',
              sel_log is not None and int(sel_log.group(1)) == best_ep, f'history argmax {best_ep}')
        vm = json.load(open(os.path.join(d, 'val_metrics.json')))
        check(f'{lead}-{run}: reported val metric == history at selected epoch',
              abs(vm['sel_mean_auprc_pac_pvc'] - h.sel_mean_auprc_pac_pvc[best_ep]) < 1e-4,
              f"{vm['sel_mean_auprc_pac_pvc']:.4f} vs {h.sel_mean_auprc_pac_pvc[best_ep]:.4f}")
        head_best = int(h.sel_mean_auprc_pac_pvc[:20].idxmax())
        rl = re.search(rf'\[{run}\] full fine-tuning starts from best head epoch (\d+)', log_text)
        check(f'{lead}-{run}: full FT restarted from best head epoch', rl is not None and
              int(rl.group(1)) == head_best, f'history head argmax {head_best}')
        exp_bb = f'patch_ecg_unified_{lead.lower()}_fixed'
        check(f'{lead}-{run}: backbone is {exp_bb}', exp_bb in cfg['backbone'], cfg['backbone'])
    return p


def error_breakdown(p, title):
    fp = p[(p.label == 0) & (p.pred == 2)]
    fpac = p[(p.label == 0) & (p.pred == 1)]
    print(f'{title}: background->PVC errors by original symbol {fp.symbol.value_counts().to_dict()}, '
          f'by patient {fp.patient.value_counts().to_dict()}; background->PAC by symbol '
          f'{fpac.symbol.value_counts().to_dict()}')


# ------------------------------------------------------------------------------------------------ 4 paths
def audit_inference_path(lead, beats):
    import torch
    from src.incart_beat import preprocessing as pp
    from src.incart_beat.model import load_beat_classifier, predict_beats
    d = os.path.join(RES, f'incart_{lead.lower()}_beat_B')
    model, meta = load_beat_classifier(os.path.join(d, 'best.pt'), 'cpu')
    p = pd.read_csv(os.path.join(d, 'val_predictions.csv'), keep_default_na=False)
    worst, n = 0.0, 0
    for rec in ('I06', 'I35'):  # patient 3 (PACs) and patient 16 (PVCs)
        x, fs, units = pp.load_wfdb_lead(os.path.join(DATASETS, 'incartdb', rec), lead)
        b = beats[(beats.record == rec) & (beats.status == 'included')]
        out = predict_beats(model, x, fs, units, b.sample_native.values, device='cpu')
        ref = p.set_index('beat_id').loc[b.beat_id, ['p_background', 'p_PAC', 'p_PVC']].to_numpy()
        worst = max(worst, float(np.abs(out[['p_background', 'p_PAC', 'p_PVC']].to_numpy() - ref).max()))
        n += len(b)
    check(f'{lead}-B: patient-inference path reproduces training-time val probabilities', worst < 1e-4,
          f'{n} beats from I06/I35, max |diff| {worst:.2e} (CPU vs GPU)')


# ------------------------------------------------------------------------------------------------ 5 bundles
def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def audit_bundle(lead, homer_csv, col):
    import torch
    name = f'incart_{lead.lower()}_beat_B'
    bdir = os.path.join('vm', name)
    files = [os.path.relpath(os.path.join(r, f), bdir) for r, _, fs_ in os.walk(os.path.join(bdir, 'src'))
             for f in fs_ if f.endswith('.py')]
    same = all(sha(os.path.join(bdir, f)) == sha(f) for f in files)
    check(f'bundle {name}: {len(files)} code files identical to repo', same and len(files) == 8)
    a = torch.load(os.path.join(bdir, f'{name}.pt'), map_location='cpu', weights_only=False)
    b = torch.load(os.path.join(RES, name, 'best.pt'), map_location='cpu', weights_only=False)
    check(f'bundle {name}: weights identical to best.pt, no optimizer state',
          a['model'].keys() == b['model'].keys() and all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
          and 'opt' not in a and a['meta']['epoch'] == b['meta']['epoch'])
    out = os.path.join(os.environ.get('SLURM_TMPDIR', '/tmp'), f'{name}_homer.csv')
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}  # keep the bundle free of __pycache__
    cmd = [sys.executable, '-c', 'import src.incart_beat.model as m; print(m.__file__)']
    where = subprocess.run(cmd, cwd=bdir, capture_output=True, text=True, env=env).stdout.strip()
    r = subprocess.run([sys.executable, '-m', 'src.incart_beat.predict_patient', '--model', f'{name}.pt',
                        '--csv', homer_csv, '--lead_col', col, '--fs', '250', '--units', 'mV', '--out', out,
                        '--device', 'cpu'], cwd=bdir, capture_output=True, text=True, env=env)
    print(r.stdout.strip(), r.stderr.strip()[-2000:], sep='\n')
    ok = r.returncode == 0 and os.path.exists(out)
    if ok:
        o = pd.read_csv(out)
        ok = len(o) > 0 and o.full_window.any() and o.loc[o.full_window, ['p_background', 'p_PAC', 'p_PVC']].notna().all().all()
        detail = f'{len(o)} beats, {int(o.full_window.sum())} classified, preds {o.pred.value_counts().to_dict()}'
    else:
        detail = f'exit {r.returncode}'
    check(f'bundle {name}: runs from its own folder on 10 min of Homer ({col})', ok and bdir in where,
          f'{detail}; code from {where}')


def guarded(name, fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # record and continue
        import traceback
        traceback.print_exc()
        check(f'{name} completed without error', False, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--homer_csv', required=True)
    a = ap.parse_args()
    logs = {'II': open('logs/incart_beat-5512726.out').read(), 'V5': open('logs/incart_beat_v5-5517540.out').read()}

    section('0. Pretraining data excludes INCART')
    csv = open(os.path.join(DATASETS, 'Unified_Dataset_No_PTB_Segment_fixed/modified_dataset/modified_dataset.csv')).read()
    hits = len(re.findall(r'incart|petersburg|/I\d{4}\.mat', csv, flags=re.I))
    check('no INCART records in Unified-fixed label CSV', hits == 0, f'{hits} hits')

    for lead, runs in (('II', ('A', 'B', 'C')), ('V5', ('B',))):
        beats = guarded(f'data audit {lead}', audit_data, lead)
        if beats is None:
            continue
        section(f'3. Metrics recomputed (lead {lead})')
        for run in runs:
            p = guarded(f'{lead}-{run} val audit', audit_run, lead, run, 'val', beats, logs[lead])
            if p is not None:
                error_breakdown(p, f'{lead}-{run} val')
        cfgs = [json.load(open(os.path.join(RES, f'incart_{lead.lower()}_beat_{r}', 'config.json'))) for r in runs]
        check(f'{lead}: identical initial weights across runs', len({c['init_hash'] for c in cfgs}) == 1,
              cfgs[0]['init_hash'])
        p = guarded(f'{lead}-B test audit', audit_run, lead, 'B', 'test', beats, logs[lead])
        if p is not None:
            error_breakdown(p, f'{lead}-B test')
        section(f'4. Training path vs patient-inference path (lead {lead})')
        guarded(f'{lead} inference-path audit', audit_inference_path, lead, beats)

    section('5. VM bundles')
    guarded('bundle II', audit_bundle, 'II', a.homer_csv, 'II')
    guarded('bundle V5', audit_bundle, 'V5', a.homer_csv, 'V2')

    n_fail = sum(t == 'FAIL' for t, *_ in RESULTS)
    n_warn = sum(t == 'WARN' for t, *_ in RESULTS)
    print(f'\n===== SUMMARY: {len(RESULTS)} checks, {n_fail} FAIL, {n_warn} WARN', flush=True)
    for t, n, dtl in RESULTS:
        if t != 'PASS':
            print(f'  {t}: {n} -- {dtl}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
