#!/usr/bin/env python3
"""Test 1-NK evaluation (2026-10-30): Test 1's segments re-centred on each lead's NeuroKit R-peak, Unified arm.

The trained runs use the patch_ecg_unified_{ii,v2}_fixed backbones (edited on Nibi; Test 1 used *_only), so:
  Test 1-NK vs Test 1 = centring + backbone change;  Test 1-NK vs E1 = window selection only (both NeuroKit, _fixed).

infer  every NeuroKit beat of every held-out record, each lead by its own fold model (as E1, single seed). clean: NeuroKit
       peaks from the E1 build cache; noisy: paper noise (NSTDB em, seed 42) and the NeuroKit peaks E1's evaluation
       detected on the contaminated signal. Chronological batches of 1024 (RevIN BN).
       Gate: predictions at the trainer's own test rows (Test 1 test segments at their NeuroKit centres) reproduce each
       run's test MSE.
score  per beat (clean) as E1 (`../e1_eval_2026-10-29/per_beat_metrics.csv` for the comparators); alarms with the paper's
       FSM / scorer, conditions clean / noisy / noisy + 6 dB gate, traces:
         greedy10  Test 1's deployment rule on NeuroKit beats (next centre >= 10 s after the previous), anchor = beat
         cad10     beat nearest each E0 10 s window centre, anchor = beat (E1's main row)
         hop2      beat nearest each point of a 2 s grid
       comparators: E0 (grid), Test 1 (.atr, as reported), Test 1 at its training centres (ann_sample), E1 (3 seeds).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("/home/student/GIT/4_ltst_beatcenter_test1/test1nk_eval_2026-10-30")  # vendored: its data files stay in the original folder; was: HERE = Path(__file__).resolve().parent
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/4_ltst_beatcenter_test1/e1_eval_2026-10-29")
import evaluate_e1 as E                                                                   # noqa: E402
T1 = E.T1
NAME = "ltstdb_beatcenter_test1nk_cv5_optB_{lead}_regression_pretrained_unified_{tag}_fixed_fold{fold:02d}_thesis_small"
TAG = {"lead_II": "ii", "lead_V2": "v2"}
ANN = Path("/home/student/GIT/4_ltst_beatcenter_test1/test1_ann_centres_2026-10-29")
DS = E.REPO / "custom_datasets/ltstdb_regression_beatcenter_100hz_test1nk_nocompress"


def ckpt(lead, fold):
    n = NAME.format(lead=lead, tag=TAG[lead], fold=fold); return E.SAVED / n / f"{n}_best.pt"


def infer(condition):
    import torch
    from ltst_pipeline.ml_inference import load_patchecg_model, _resample_to_model_fs, create_patch, PATCH_LEN, DataAugmentation
    dev = "cuda"; t0 = time.time(); rows = []
    N = None if condition == "clean" else pd.read_pickle(E.PRED / "e1_every_beat_noisy.pkl").query("seed == 200")[["lead", "record", "nk_sample"]]

    @torch.no_grad()
    def run(model, revin, x, pk):
        out = np.full(len(pk), np.nan, np.float32); ok = (pk - E.HALF >= 0) & (pk + E.HALF <= len(x)); idx = np.flatnonzero(ok); off = np.arange(E.WIN)
        for i in range(0, len(idx), 1024):
            j = idx[i:i + 1024]; seg = x[pk[j, None] - E.HALF + off][..., None]
            xb = revin(torch.from_numpy(np.ascontiguousarray(_resample_to_model_fs(np.where(np.isfinite(seg), seg, 0.0).astype(np.float32), E.FS))).to(dev), "norm")
            out[j] = model(create_patch(xb, patch_len=PATCH_LEN, stride=PATCH_LEN), mask=None, inverted_mask=None, indices=None,
                           augmentation=DataAugmentation.none).detach().cpu().numpy().reshape(len(j), -1)[:, 0]
        return out

    for fold, recs in E.FOLDS.items():
        sig = {rec: T1.load_signal(rec, condition) for rec in recs}
        for lead, ch in E.LEADS.items():
            model, revin, _ = load_patchecg_model(str(ckpt(lead, fold)), device=dev, lead_mode=lead)
            for rec in recs:
                pk = np.load(E.BUILD / "nk" / f"{rec}_ch{ch}.npy") if N is None else N[(N.lead == lead) & (N.record == rec)].nk_sample.to_numpy()
                rows.append(pd.DataFrame(dict(lead=lead, seed=200, record=rec, fold=fold, nk_sample=pk, pred_mV=run(model, revin, sig[rec][:, ch].astype(np.float64), pk))))
            del model; torch.cuda.empty_cache()
        print(f"[{time.time() - t0:5.0f}s] {condition} fold {fold}", flush=True)
    D = pd.concat(rows, ignore_index=True); D.to_pickle(HERE / f"t1nk_every_beat_{condition}.pkl")
    if condition == "clean":
        C = pd.read_csv(DS / "centres.csv"); meta = pd.read_csv(DS / "modified_dataset/modified_dataset.csv"); out = []
        for lead, ch in E.LEADS.items():
            for fold, recs in E.FOLDS.items():
                m = pd.read_csv(next((ckpt(lead, fold).parent).glob("_test_metrics_*small.csv")))
                c = C[(C.lead == ch) & C.patient_id.isin(recs) & C.nk_matched]
                y = meta.loc[c.row.to_numpy(), f"target_lead{ch}_mV"].to_numpy()
                P = D[(D.lead == lead)].set_index(["record", "nk_sample"]).pred_mV
                p = P.reindex(pd.MultiIndex.from_arrays([c.patient_id, c.centre_sample])).to_numpy()
                out.append(dict(lead=lead, fold=fold, n=len(y), mse_here=float(np.nanmean((p - y) ** 2)), mse_trainer=float(m.test_MSE.iloc[0])))
        G = pd.DataFrame(out); G["rel_diff"] = (G.mse_here - G.mse_trainer).abs() / G.mse_trainer; G.to_csv(HERE / "gate_trainer_mse.csv", index=False)
        print(f"TRAINER GATE: {int((G.rel_diff < 0.05).sum())}/{len(G)} within 5 % (matched rows only)\n{G.round(5).to_string(index=False)}", flush=True)


def per_beat():
    from ltst_pipeline.old_fsm_gs import MANUAL_OMIT
    D = pd.read_pickle(HERE / "t1nk_every_beat_clean.pkl"); rows = []
    for lead in E.LEADS:
        paper = lead.split("_")[1]; recs = [r for r in E.REC2FOLD if (r, paper) not in MANUAL_OMIT]
        L = pd.concat([E.labels(lead, r).assign(record=r) for r in recs], ignore_index=True)
        p = L.merge(D[D.lead == lead][["record", "nk_sample", "pred_mV"]], on=["record", "nk_sample"], how="left").pred_mV.to_numpy(); y = L.ee_mV.to_numpy()
        for subset, m in [("all", np.ones(len(L), bool)), ("DMY only", L.DMY.to_numpy()), ("non-DMY", ~L.DMY.to_numpy())] + \
                         [(f"class {c}", (L.cls == c).to_numpy()) for c in ("depression", "normal", "elevation", "severe_elevation")]:
            rows.append(dict(lead=lead, source="Test 1-NK", subset=subset, coverage=float(np.isfinite(p[m]).mean()) if m.any() else np.nan, **E.err_stats((p - y)[m])))
    R = pd.concat([pd.DataFrame(rows), pd.read_csv(E.HERE / "per_beat_metrics.csv").query("lead != 'lead_V5'")], ignore_index=True)
    R.to_csv(HERE / "per_beat_metrics.csv", index=False)


def greedy(D, lead):
    col, other = ("pred_II_mV", "pred_V2_mV") if lead == "lead_II" else ("pred_V2_mV", "pred_II_mV"); out = []
    for rec, g in D[D.lead == lead].groupby("record"):
        g = g[np.isfinite(g.pred_mV)].sort_values("nk_sample"); keep, last = [], None
        for i, s in enumerate(g.nk_sample.to_numpy()):
            if last is None or s >= last + E.WIN:
                keep.append(i); last = s
        h = g.iloc[keep]; out.append(pd.DataFrame({"record": rec, "anchor_samp": h.nk_sample.to_numpy(), col: h.pred_mV.to_numpy(), other: np.nan}))
    return pd.concat(out, ignore_index=True)


def score(condition, threshold=6.0):
    import nk_core as K
    from score_alarms import per_record_components, metrics_from_components, gs_scope
    tag = condition if condition != "noisy_sqi" else f"noisy_sqi_{threshold:g}dB"
    D = pd.read_pickle(HERE / f"t1nk_every_beat_{'clean' if condition == 'clean' else 'noisy'}.pkl")
    gs = K.load_gs(E.GS_PATH); durations = {r.record: float(r.duration_s) for r in pd.read_csv(E.DUR_CSV).itertuples()}; mask = set(K.MANUAL_OMIT)
    ctx = T1.sqi_context(condition)
    if ctx is not None:
        views = ctx.pop("_views"); ctx = {r: (T1._Cols(views[(r, 0)], views[(r, 1)]), ctx[r][1]) for r in E.REC2FOLD}
    rows, al = [], {}
    tr = {lead: {**E.traces(D, lead, 200), "greedy10": greedy(D, lead)} for lead in E.LEADS}
    for sc in ("greedy10", "cad10", "hop2"):
        for lead in E.LEADS:
            al[(lead, sc)] = K.apply_mask(T1.alarms_from_predictions(tr[lead][sc], ctx, threshold), mask)
        al[("union", sc)] = pd.concat([al[("lead_II", sc)].query("lead == 'II'"), al[("lead_V2", sc)].query("lead == 'V2'")], ignore_index=True)
        print(f"[{tag}] {sc} FSM done", flush=True)
    adir = HERE / "alarms" / tag; adir.mkdir(parents=True, exist_ok=True)
    for (m, sc), df in al.items():
        df.to_csv(adir / f"t1nk__{m}__{sc}.csv", index=False)
        for scope in T1.SCOPES[m]:
            c = per_record_components(gs_scope(gs, scope), K.scope_filter(df, scope), durations, scope, mask, K.RECORDS_15)
            rows.append(dict(condition=tag, model="Test 1-NK", mode=m, scheme=sc, scope=scope, **metrics_from_components(c)))
    keep = ["TP", "FP", "FN", "Ignored", "F1", "time_F1", "sensitivity", "PPV_time", "IoU_whole_record"]
    M1 = pd.read_csv(E.T1D / f"metrics_{tag}.csv").query("arm == 'Chapman-pretrained'")
    for mset, sc, lab in (("published", "grid", "E0"), ("test1", "beat", "Test 1 (.atr, as reported)")):
        for r in M1[(M1.model_set == mset) & (M1.scheme == sc) & M1["mode"].isin(["lead_II", "lead_V2", "union"])].itertuples(index=False):
            rows.append(dict(condition=tag, model=lab, mode=r.mode, scheme=sc, scope=r.scope, **{k: getattr(r, k) for k in keep}))
    A = pd.read_csv(ANN / f"metrics_{tag}.csv").query("model == 'Test 1, ann_sample centres' and arm == 'Chapman-pretrained'")
    for r in A[A["mode"].isin(["lead_II", "lead_V2", "union"])].itertuples(index=False):
        rows.append(dict(condition=tag, model="Test 1 (training centres)", mode=r.mode, scheme="ann", scope=r.scope, **{k: getattr(r, k) for k in keep}))
    E1 = pd.read_csv(E.HERE / f"alarm_metrics_{tag}.csv").query("model.str.startswith('E1')", engine="python")
    for r in E1[E1.scheme.isin(["cad10", "hop2"])].itertuples(index=False):
        rows.append(dict(condition=tag, model=r.model, mode=r.mode, scheme=r.scheme, scope=r.scope, **{k: getattr(r, k) for k in keep}))
    pd.DataFrame(rows).to_csv(HERE / f"alarm_metrics_{tag}.csv", index=False); print(f"wrote alarm_metrics_{tag}.csv", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]; cond = sys.argv[2] if len(sys.argv) > 2 else "clean"
    {"infer": lambda: infer(cond), "perbeat": per_beat, "score": lambda: score(cond)}[st]()
