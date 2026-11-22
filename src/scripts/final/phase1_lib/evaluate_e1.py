#!/usr/bin/env python3
"""E1 evaluation (2026-10-29): NeuroKit-centred, class-balanced ST regression models vs E0 (published) and Test 1.

Known inconsistencies, accepted for this run (see memory e1_dataset_build): E1 II/V2 start from the
patch_ecg_unified_{ii,v2}_fixed backbones, E0 / Test 1 from patch_ecg_unified_{ii,v2}_only; the "chronic-offset"
subset is undefined, so no with/without split is reported.

infer  (GPU) every NeuroKit beat of every held-out record gets a prediction from the fold model that held it out, for
       each seed (s200, s300, s400). Window = [beat - 1250, beat + 1250) at 250 Hz, resampled per window with the
       inference pipeline's own function (as training, ST_ARM step 5). clean: NeuroKit peaks from the E1 build cache;
       noisy: NSTDB 'em', seed 42 (paper / Test 1 contamination), NeuroKit re-detected on the contaminated lead (same
       5-min chunked detector). V5: the 4 leave-one-record-out models, clean only.
       Gate: predictions on each record's trainer test rows (10 s grid) reproduce the trainer's own test MSE.
score  per beat (clean): bias, SD, MAE, % |error| > 0.1 mV, 95th percentile of |error|, by lead, DMY-only and label
       class, on the usable matched beats of the 26 lead-records (MANUAL_OMIT out) and the 4 V5 records.
       E0 = published grid models (beat gets its 10 s grid window), Test 1 = beat-centred models (beat gets the window
       it lies in); all three arms for context, from the Test 1 prediction files.
       alarms: the paper's FSM (old_fsm_gs.run_fsm_on_record) + MANUAL_OMIT + score_alarms per_record_components,
       101-episode GS, conditions clean / noisy / noisy + 6 dB SQI gate (Test 1's exact gate implementation).
       Traces: cad10     = E0's cadence: the beat nearest the centre of each 10 s grid window, anchored at that beat
               cad10_end = same predictions anchored at the grid window end (E0's time stamps)
               hop2      = the beat nearest each point of a 2 s grid (within +-1 s), anchored at the beat
       Modes lead_II, lead_V2 (scopes FULL and own lead) and union (II alarms on II + V2 alarms on V2), per seed.
       Paired record-clustered bootstrap (10,000, seed 20260805) for E1 - E0 and E1 - Test 1 (Unified-pretrained arm).
Usage: $VENV/bin/python evaluate_e1.py infer --condition clean|noisy ; ... score --condition clean|noisy|noisy_sqi
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("/home/student/GIT/4_ltst_beatcenter_test1/e1_eval_2026-10-29")  # vendored: its data files stay in the original folder; was: HERE = Path(__file__).resolve().parent
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/1_ltst_paper_cmbec48/results/neurokit/_code")      # nk_core (correct location)
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/1_ltst_paper_cmbec48/results/alarms/_code")         # score_alarms
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(1, "/home/student/GIT/4_ltst_beatcenter_test1/beatcenter_test1/_code")
import evaluate_test1 as T1                                                               # noqa: E402

REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SAVED = REPO / "results/fine-tune/saved_models/patient-segment-regression"
LTST = Path("/home/student/Desktop/Research Data/DATASET NK/ltst_records")
BUILD = Path("/home/student/GIT/4_ltst_beatcenter_test1/e1_build/cache")
B16 = Path("/home/student/GIT/4_ltst_beatcenter_test1/split_design_2026-10-15/beats_16a")
T1D = Path("/home/student/GIT/4_ltst_beatcenter_test1/beatcenter_test1")
PAPER = Path("/home/student/GIT/1_ltst_paper_cmbec48")
GS_PATH = str(PAPER / "transfer_bundle/gs_per_beat.csv"); DUR_CSV = str(PAPER / "results/neurokit/_sweep/record_durations.csv")
FS, WIN, HALF = 250, 2500, 1250
SEEDS = (200, 300, 400)
LEADS = {"lead_II": 0, "lead_V2": 1}
FOLDS = T1.FOLDS; REC2FOLD = T1.REC2FOLD
V5 = {"s20571": 0, "s20581": 0, "s30671": 1, "s30681": 2}
PRED = HERE / "predictions"


def ckpt(lead: str, fold: int, seed: int) -> Path:
    tag = "unified_ii_fixed" if lead == "lead_II" else "unified_v2_fixed"
    name = f"ltstdb_e1_{lead}_regression_pretrained_{tag}_fold{fold:02d}_s{seed}_thesis_small"
    return SAVED / name / f"{name}_best.pt"


def ckpt_v5(rec: str) -> Path:
    name = f"ltstdb_e1_lead_V5_regression_pretrained_unified_v5_fixed_fold_test_{rec}_s200_thesis_small"
    return SAVED / name / f"{name}_best.pt"


def nk_detect(x: np.ndarray) -> np.ndarray:
    """The E1 build's detector (build_e1.py S1): NeuroKit clean + peaks in 5-min chunks with 10 s padding."""
    import neurokit2 as nk
    x = np.where(np.isfinite(x), x, 0.0); step, pad, peaks = 300 * FS, 10 * FS, []
    for s in range(0, len(x), step):
        a, b = max(0, s - pad), min(len(x), s + step + pad)
        cl = nk.ecg_clean(x[a:b], sampling_rate=FS, method="neurokit")
        p = np.asarray(nk.ecg_peaks(cl, sampling_rate=FS, method="neurokit")[1]["ECG_R_Peaks"], dtype=np.int64) + a
        peaks.append(p[(p >= s) & (p < min(s + step, len(x)))])
    return np.unique(np.concatenate(peaks))


# ----------------------------------------------------------------------------- inference
def infer(condition: str) -> None:
    import torch
    import wfdb
    from ltst_pipeline.ml_inference import load_patchecg_model, _resample_to_model_fs, create_patch, PATCH_LEN, DataAugmentation
    dev = "cuda"; PRED.mkdir(exist_ok=True); t0 = time.time()

    @torch.no_grad()
    def forward(model, revin, sig1: np.ndarray, centres: np.ndarray) -> np.ndarray:
        out = np.full(len(centres), np.nan, np.float32); ok = (centres - HALF >= 0) & (centres + HALF <= len(sig1))
        idx = np.flatnonzero(ok); off = np.arange(WIN)
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]; seg = sig1[centres[j, None] - HALF + off][..., None]           # (n, 2500, 1)
            seg = np.where(np.isfinite(seg), seg, 0.0).astype(np.float32)
            xb = torch.from_numpy(np.ascontiguousarray(_resample_to_model_fs(seg, FS))).to(dev)  # (n, 1, 1000)
            xb = revin(xb, "norm")
            pr = model(create_patch(xb, patch_len=PATCH_LEN, stride=PATCH_LEN), mask=None, inverted_mask=None, indices=None,
                       augmentation=DataAugmentation.none).detach().cpu().numpy()
            out[j] = pr.reshape(len(j), -1)[:, 0]
        return out

    rows = []
    for fold, recs in FOLDS.items():
        sigs = {}
        for rec in recs:
            p = T1.load_signal(rec, condition)                                                   # (N, >=2), noisy = paper noise
            for lead, ch in LEADS.items():
                x = p[:, ch].astype(np.float64)
                pk = np.load(BUILD / "nk" / f"{rec}_ch{ch}.npy") if condition == "clean" else nk_detect(x)
                sigs[(rec, lead)] = (x, pk)
        for lead in LEADS:
            for seed in SEEDS:
                model, revin, _ = load_patchecg_model(str(ckpt(lead, fold, seed)), device=dev, lead_mode="lead_II")
                for rec in recs:
                    x, pk = sigs[(rec, lead)]
                    rows.append(pd.DataFrame(dict(lead=lead, seed=seed, record=rec, fold=fold, nk_sample=pk, pred_mV=forward(model, revin, x, pk))))
                del model; torch.cuda.empty_cache()
        print(f"[{time.time() - t0:7.0f}s] {condition} fold {fold} done", flush=True)
    if condition == "clean":
        for rec, ch in V5.items():
            x = wfdb.rdrecord(str(LTST / rec / rec), channels=[ch]).p_signal[:, 0].astype(np.float64)
            pk = np.load(BUILD / "nk" / f"{rec}_ch{ch}.npy")
            model, revin, _ = load_patchecg_model(str(ckpt_v5(rec)), device=dev, lead_mode="lead_II")
            rows.append(pd.DataFrame(dict(lead="lead_V5", seed=200, record=rec, fold=0, nk_sample=pk, pred_mV=forward(model, revin, x, pk))))
            del model; torch.cuda.empty_cache()
        print(f"[{time.time() - t0:7.0f}s] V5 done", flush=True)
    D = pd.concat(rows, ignore_index=True); D.to_pickle(PRED / f"e1_every_beat_{condition}.pkl")
    print(f"wrote {len(D):,} predictions", flush=True)
    if condition == "clean":
        gate_trainer(D)


def gate_trainer(D: pd.DataFrame) -> None:
    """Our predictions at the trainer's own test rows must reproduce the trainer's test MSE (same windows, same model)."""
    out = []
    for lead in list(LEADS) + ["lead_V5"]:
        for d in sorted(SAVED.glob(f"ltstdb_e1_{lead}_*")):
            name = d.name
            m = pd.read_csv(next(d.glob("_test_metrics_*small.csv")))
            if lead == "lead_V5":
                recs = [name.split("fold_test_")[1].split("_s")[0]]; seed = 200
            else:
                fold = int(name.split("_fold")[1][:2]); recs = FOLDS[fold]; seed = int(name.split(f"fold{fold:02d}_s")[1].split("_")[0])
            ys, ps = [], []
            for rec in recs:
                sel = pd.read_csv(BUILD.parent / "cache" / "select" / f"{lead}_{rec}.csv.gz"); sel = sel[sel.role == "test"]
                P = D[(D.lead == lead) & (D.seed == seed) & (D.record == rec)].set_index("nk_sample").pred_mV
                ys.append(sel.ee_mV.to_numpy()); ps.append(P.reindex(sel.nk_sample).to_numpy())
            y, p = np.concatenate(ys), np.concatenate(ps)
            col = next(c for c in m.columns if c.lower() in ("test_mse", "mse"))
            out.append(dict(run=name, n=len(y), mse_here=float(np.nanmean((p - y) ** 2)), mse_trainer=float(m[col].iloc[0])))
    G = pd.DataFrame(out); G["rel_diff"] = (G.mse_here - G.mse_trainer).abs() / G.mse_trainer
    G.to_csv(HERE / "gate_trainer_mse.csv", index=False)
    print(f"TRAINER GATE: {int((G.rel_diff < 0.02).sum())}/{len(G)} runs reproduce the trainer's test MSE within 2 %", flush=True)


# ----------------------------------------------------------------------------- per-beat metrics
def labels(lead: str, rec: str) -> pd.DataFrame:
    ev = pd.read_csv(BUILD / "select" / f"eval_beats_{lead}_{rec}.csv.gz")
    ev = ev[ev.matched & ev.usable].copy()
    ch = LEADS.get(lead, V5.get(rec)); z = np.load(B16 / f"{rec}.npz")
    ev["DMY"] = z[f"ST{ch}_DMY"][ev.idx16a.astype(int)].astype(bool)
    return ev[["nk_sample", "ee_mV", "cls", "DMY"]]


def err_stats(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    if not len(e):
        return dict(n=0)
    sd = e.std(ddof=1) if len(e) > 1 else np.nan
    return dict(n=len(e), bias_uV=1000 * e.mean(), SD_uV=1000 * sd, MAE_uV=1000 * np.abs(e).mean(), pct_gt100uV=100 * np.mean(np.abs(e) > 0.1),
                P95_abs_uV=1000 * np.percentile(np.abs(e), 95), LoA_lo_uV=1000 * (e.mean() - 1.96 * sd), LoA_hi_uV=1000 * (e.mean() + 1.96 * sd))


def per_beat() -> pd.DataFrame:
    from ltst_pipeline.old_fsm_gs import MANUAL_OMIT
    D = pd.read_pickle(PRED / "e1_every_beat_clean.pkl"); rows = []
    paper = {"lead_II": "II", "lead_V2": "V2"}
    for lead in list(LEADS) + ["lead_V5"]:
        recs = list(V5) if lead == "lead_V5" else [r for r in REC2FOLD if (r, paper[lead]) not in MANUAL_OMIT]
        L = pd.concat([labels(lead, r).assign(record=r) for r in recs], ignore_index=True)
        sources = {}
        for seed in sorted(D[D.lead == lead].seed.unique()):
            P = D[(D.lead == lead) & (D.seed == seed)][["record", "nk_sample", "pred_mV"]]
            sources[f"E1 s{seed}"] = L.merge(P, on=["record", "nk_sample"], how="left").pred_mV.to_numpy()
        if lead != "lead_V5":
            col = "pred_II_mV" if lead == "lead_II" else "pred_V2_mV"
            for mset, scheme, label in (("published", "grid", "E0"), ("test1", "beat", "Test 1")):
                for arm in T1.ARMS:
                    P = pd.read_csv(T1D / "predictions" / f"{mset}__{arm}__{lead}__{scheme}.csv.gz")
                    v = []
                    for rec, g in L.groupby("record", sort=False):
                        q = P[P.record == rec].sort_values("anchor_samp"); a = q.anchor_samp.to_numpy(); pv = q[col].to_numpy()
                        s = g.nk_sample.to_numpy()
                        if scheme == "grid":                              # window k covers [k*2500, k*2500+2500), anchor = end
                            k = s // WIN; m = dict(zip((a - (WIN - 1)) // WIN, pv)); v.append(pd.Series([m.get(x, np.nan) for x in k], index=g.index))
                        else:                                             # the beat-centred window that contains the beat
                            j = np.clip(np.searchsorted(a, s), 1, len(a) - 1); j = np.where(np.abs(a[j] - s) < np.abs(a[j - 1] - s), j, j - 1)
                            v.append(pd.Series(np.where(np.abs(a[j] - s) < HALF, pv[j], np.nan), index=g.index))
                    sources[f"{label} {arm}"] = pd.concat(v).reindex(L.index).to_numpy()
        y = L.ee_mV.to_numpy()
        for src, p in sources.items():
            for subset, m in [("all", np.ones(len(L), bool)), ("DMY only", L.DMY.to_numpy()), ("non-DMY", ~L.DMY.to_numpy())] + \
                             [(f"class {c}", (L.cls == c).to_numpy()) for c in ("depression", "normal", "elevation", "severe_elevation")]:
                rows.append(dict(lead=lead, source=src, subset=subset, coverage=float(np.isfinite(p[m]).mean()) if m.any() else np.nan, **err_stats((p - y)[m])))
    R = pd.DataFrame(rows); R.to_csv(HERE / "per_beat_metrics.csv", index=False); return R


# ----------------------------------------------------------------------------- alarms
def traces(D: pd.DataFrame, lead: str, seed: int) -> dict:
    """{scheme: DataFrame(record, anchor_samp, pred_II_mV, pred_V2_mV)} for one lead model and seed."""
    import wfdb
    col = "pred_II_mV" if lead == "lead_II" else "pred_V2_mV"; other = "pred_V2_mV" if lead == "lead_II" else "pred_II_mV"; out = {"cad10": [], "cad10_end": [], "hop2": []}
    for rec, g in D[(D.lead == lead) & (D.seed == seed)].groupby("record"):
        g = g[np.isfinite(g.pred_mV)].sort_values("nk_sample"); s = g.nk_sample.to_numpy(); pv = g.pred_mV.to_numpy()
        n = wfdb.rdheader(str(LTST / rec / rec)).sig_len
        def nearest(points, tol):
            j = np.clip(np.searchsorted(s, points), 1, len(s) - 1); j = np.where(np.abs(s[j] - points) < np.abs(s[j - 1] - points), j, j - 1)
            ok = np.abs(s[j] - points) <= tol; return j, ok
        starts = np.arange(n // WIN) * WIN; j, ok = nearest(starts + HALF, HALF)
        out["cad10"].append(pd.DataFrame({"record": rec, "anchor_samp": s[j[ok]], col: pv[j[ok]], other: np.nan}))
        out["cad10_end"].append(pd.DataFrame({"record": rec, "anchor_samp": starts[ok] + WIN - 1, col: pv[j[ok]], other: np.nan}))
        j, ok = nearest(np.arange(0, n, 2 * FS), FS); jj = np.unique(j[ok])
        out["hop2"].append(pd.DataFrame({"record": rec, "anchor_samp": s[jj], col: pv[jj], other: np.nan}))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def score(condition: str, threshold: float = 6.0) -> None:
    import nk_core as K
    from score_alarms import per_record_components, metrics_from_components, gs_scope
    tag = condition if condition != "noisy_sqi" else f"noisy_sqi_{threshold:g}dB"
    D = pd.read_pickle(PRED / f"e1_every_beat_{'clean' if condition == 'clean' else 'noisy'}.pkl")
    gs = K.load_gs(GS_PATH); durations = {r.record: float(r.duration_s) for r in pd.read_csv(DUR_CSV).itertuples()}; mask = set(K.MANUAL_OMIT)
    ctx = T1.sqi_context(condition)
    if ctx is not None:
        views = ctx.pop("_views"); ctx = {r: (T1._Cols(views[(r, 0)], views[(r, 1)]), ctx[r][1]) for r in REC2FOLD}
    alarms, t0 = {}, time.time()
    for seed in SEEDS:
        tr = {lead: traces(D, lead, seed) for lead in LEADS}
        for sc in ("cad10", "cad10_end", "hop2"):
            for lead in LEADS:
                alarms[(seed, lead, sc)] = K.apply_mask(T1.alarms_from_predictions(tr[lead][sc], ctx, threshold), mask)
            alarms[(seed, "union", sc)] = pd.concat([alarms[(seed, "lead_II", sc)][lambda d: d.lead == "II"],
                                                     alarms[(seed, "lead_V2", sc)][lambda d: d.lead == "V2"]], ignore_index=True)
        print(f"[{time.time() - t0:6.0f}s] {tag}: FSM done for seed {seed}", flush=True)
    adir = HERE / "alarms" / tag; adir.mkdir(parents=True, exist_ok=True)
    for (seed, m, sc), df in alarms.items():
        df.to_csv(adir / f"e1_s{seed}__{m}__{sc}.csv", index=False)
    comps, rows = {}, []
    for (seed, m, sc), df in alarms.items():
        for scope in T1.SCOPES[m]:
            c = per_record_components(gs_scope(gs, scope), K.scope_filter(df, scope), durations, scope, mask, K.RECORDS_15)
            comps[(f"E1 s{seed}", m, sc, scope)] = c
            rows.append(dict(condition=tag, model=f"E1 s{seed}", mode=m, scheme=sc, scope=scope, **metrics_from_components(c)))
    # comparators: E0 (published grid) and Test 1 (beat), Unified-pretrained ("Chapman") arm, re-scored here from Test 1's alarm files
    gate = []
    M1 = pd.read_csv(T1D / f"metrics_{tag}.csv")
    for mset, sc, label in (("published", "grid", "E0"), ("test1", "beat", "Test 1")):
        for m in ("lead_II", "lead_V2", "union"):
            df = pd.read_csv(T1D / "alarms" / tag / f"{mset}__Chapman-pretrained__{m}__{sc}.csv")
            for scope in T1.SCOPES[m]:
                c = per_record_components(gs_scope(gs, scope), K.scope_filter(df, scope), durations, scope, mask, K.RECORDS_15)
                comps[(label, m, sc, scope)] = c; r = metrics_from_components(c)
                rows.append(dict(condition=tag, model=label, mode=m, scheme=sc, scope=scope, **r))
                ref = M1[(M1.model_set == mset) & (M1.arm == "Chapman-pretrained") & (M1["mode"] == m) & (M1.scheme == sc) & (M1.scope == scope)].iloc[0]
                gate.append(all(int(ref[k]) == r[k] for k in ("TP", "FP", "FN", "Ignored")))
    print(f"SCORER GATE [{tag}]: {sum(gate)}/{len(gate)} E0 / Test 1 rows re-scored here equal Test 1's metrics", flush=True)
    M = pd.DataFrame(rows); M.to_csv(HERE / f"alarm_metrics_{tag}.csv", index=False)
    # paired record-clustered bootstrap: E1 (each seed, cad10) minus E0 / Test 1
    rng = np.random.default_rng(20260805); recs = K.RECORDS_15; idx = rng.integers(0, len(recs), size=(10_000, len(recs))); brow = []
    def f1s(arr, cols):
        TP, FP, FN = (arr[..., cols.index(k)] for k in ("TP", "FP", "FN")); tTP, tFP, tFN = (arr[..., cols.index(k)] for k in ("tTP", "tFP", "tFN"))
        return (np.where(2 * TP + FP + FN > 0, 2 * TP / np.maximum(2 * TP + FP + FN, 1e-12), 0),
                np.where(2 * tTP + tFP + tFN > 0, 2 * tTP / np.maximum(2 * tTP + tFP + tFN, 1e-12), 0), TP, FP)
    for seed in SEEDS:
        for ref, rsc in (("E0", "grid"), ("Test 1", "beat")):
            for m in ("lead_II", "lead_V2", "union"):
                for scope in T1.SCOPES[m]:
                    ca = comps[(f"E1 s{seed}", m, "cad10", scope)].set_index("record").loc[recs]; cb = comps[(ref, m, rsc, scope)].set_index("record").loc[recs]
                    cols = list(ca.columns)
                    ea, ta, pa, fa = f1s(ca.to_numpy()[idx].astype(float).sum(axis=1), cols); eb, tb, pb, fb = f1s(cb.to_numpy()[idx].astype(float).sum(axis=1), cols)
                    a0, b0 = metrics_from_components(ca.reset_index()), metrics_from_components(cb.reset_index())
                    for met, d, pt in (("episode_F1", ea - eb, a0["F1"] - b0["F1"]), ("time_F1", ta - tb, a0["time_F1"] - b0["time_F1"]),
                                       ("TP", pa - pb, a0["TP"] - b0["TP"]), ("FP", fa - fb, a0["FP"] - b0["FP"])):
                        lo, hi = np.percentile(d, [2.5, 97.5])
                        brow.append(dict(condition=tag, contrast=f"E1 s{seed} (cad10) - {ref}", mode=m, scope=scope, metric=met, delta=pt, ci_lo=lo, ci_hi=hi,
                                         significant=bool(lo > 0 or hi < 0)))
    pd.DataFrame(brow).to_csv(HERE / f"bootstrap_{tag}.csv", index=False)
    print(f"wrote alarm_metrics_{tag}.csv, bootstrap_{tag}.csv", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["infer", "score", "perbeat"])
    ap.add_argument("--condition", choices=["clean", "noisy", "noisy_sqi"], default="clean"); ap.add_argument("--threshold", type=float, default=6.0)
    a = ap.parse_args()
    if a.stage == "infer":
        infer(a.condition)
    elif a.stage == "perbeat":
        per_beat()
    else:
        score(a.condition, a.threshold)
