#!/usr/bin/env python3
"""Phase 1 evaluation on the VM (README_phase1_runs.md, "Predictions to save" and "Scoring (every run)").

Nothing published is written to. Output: Tasks/phase1/runs/<RUN>/ and Tasks/phase1/runs/_scores/.

Stages (repo .venv, GPU for infer):
  infer R1|R1b|R2|R2b|R3   out-of-fold predictions, then the MSE gate: predictions at the trainer's own test rows must reproduce the run's saved test MSE within 1 %.
                           R1 = the published checkpoints through this code, a control: it must reproduce the published prediction files.
  combine                  R2_R2b = R2 with R2b's V2 folds 1, 4, 5 swapped in, R3's comparison run.
  regression               clean, out-of-fold, per lead, with and without masked pairs: MAE, bias, RMSE, slope (OLS of prediction on label, as C section 9), r, p100, R2; also on DMY beats.
  alarms <tag>             PROSE FSM + published scorer. tag: clean | noisy | noisy_sqi_6dB (.atr gate) | noisy_sqi_6dB_rodrigues2021 | noisy_sqi_6dB_neurokit (control on the published predictions only).
  controls                 this code against the published numbers.

Prediction files, every one with the window's first sample:
  grid runs   <RUN>/predictions[_noisy]/<RUN>__<arm>__<mode>__grid.csv.gz : record, fold, anchor_samp (window end, the FSM's time stamp), window_start_samp, pred_II_mV, pred_V2_mV. Published columns plus window_start_samp.
              <RUN>/predictions_training_grid/<RUN>__<arm>__<mode>.csv.gz : record, seg_idx, samp_end_raw, fold, window_start_samp, pred_II_mV, pred_V2_mV. The trainer's windows, clean, for regression.
  R3          R3/predictions[_noisy]/R3__pretrained__<lead>__every_beat.csv.gz : record, fold, lead, peak_samp (window centre, frozen detector), window_start_samp, pred_mV. Every detector beat.
              The alarm traces (>= 10 s rule, 10 s cadence, 2 s hop) are selected from these at scoring time by Test 1-NK's own functions, so the FSM, gate and scoring rerun from the files alone.
Reused unchanged: ltst_pipeline.ml_inference (load_patchecg_model, predict_st_per_segment), evaluate_test1 (load_signal, alarms_from_predictions, sqi_context, SCOPES), evaluate_e1.traces, evaluate_t1nk.greedy, score_alarms."""
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/4_ltst_beatcenter_test1/test1nk_eval_2026-10-30")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Tasks/C_neurokit")
import c6_noisy_fold as C6  # noqa: E402,F401  (reads the local NSTDB em instead of fetching it)
import evaluate_t1nk as V  # noqa: E402
import nk_core as K  # noqa: E402
from score_alarms import per_record_components, metrics_from_components, gs_scope  # noqa: E402

E, T1 = V.E, V.T1
FS, WIN, HALF, TOL = 250, 2500, 1250, 25
REPO = T1.REPO
SAVE = REPO / "results/fine-tune/saved_models/patient-segment-regression"
PUB = REPO / "results/fine-tune/saved_models/patient-segment-regression_2"
ROOT = Path("/home/student/GIT/Tasks/phase1/runs")
SC = ROOT / "_scores"
FOLD_DIR = REPO / "phase1_folds"
V2DS = REPO / "custom_datasets/ltstdb_regression_v2_beatcentred_rodrigues2021"
DS = Path("/home/student/Desktop/Research Data/DATASET NK")
PEAKS = DS / "C_neurokit/c14_detectors/peaks"
DELIN = DS / "Delineation Data"
OFFSETS = "/home/student/GIT/results/regression/C_training_grid_offsets.csv"
PUB_TG = Path("/home/student/GIT/results/regression/predictions_training_grid")
TRUTH = REPO / "custom_datasets/ltstdb_regression_median_100hz_nocompress/modified_dataset/modified_dataset.csv"
DMY = DS / "C_neurokit/c5_iso/c5_dmy_per_beat.csv"
DET = "rodrigues2021"
MODES = ("lead_II", "lead_V2", "two_lead")
SINGLE = ("lead_II", "lead_V2")
ARMS = ("pretrained", "supervised")
ALL = (1, 2, 3, 4, 5)
OPT2P = (1, 4, 5)
CH = {"lead_II": 0, "lead_V2": 1}
OUT_LEADS = {"lead_II": ("II",), "lead_V2": ("V2",), "two_lead": ("II", "V2")}
COL = {"II": "pred_II_mV", "V2": "pred_V2_mV"}
TARGET = {"II": "target_lead0_mV", "V2": "target_lead1_mV"}
PUB_TAG = {"lead_II": "pretrained_chapman_ii_only", "lead_V2": "pretrained_chapman_v2_only", "two_lead": "pretrained_chapman_ii_v2"}
PUB_ARM = {"pretrained": "Chapman-pretrained", "supervised": "supervised"}
JOBS = {"R1": [(m, a, ALL) for m in MODES for a in ARMS], "R1b": [(m, a, ALL) for m in MODES for a in ARMS], "R2": [(m, "pretrained", ALL) for m in MODES],
        "R2b": [("lead_V2", "pretrained", OPT2P)], "R3": [(m, "pretrained", ALL) for m in SINGLE]}
FOLDS145 = [r for k in OPT2P for r in T1.FOLDS[k]]
THRESHOLD = 6.0


def run_dir(run):
    return ROOT / ("R1_control" if run == "R1" else run)


def log(*a):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")
    print(*a, flush=True)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def name_of(run, mode, arm, k):
    if run == "R1":
        return PUB, f"ltstdb_median_cv5_optB_{mode}_regression_{PUB_TAG[mode] if arm == 'pretrained' else 'supervised'}_fold{k:02d}_thesis_small"
    return SAVE, f"{run}_{mode}_{arm}_fold{k:02d}"


def ckpt(run, mode, arm, k):
    root, n = name_of(run, mode, arm, k)
    return root / n / f"{n}_best.pt"


def saved_mse(run, mode, arm, k):
    root, n = name_of(run, mode, arm, k)
    return float(pd.read_csv(root / n / f"_test_metrics_{n}.csv").test_MSE.iloc[0])


def fold_csv(run, mode, k):
    if run == "R3":
        return V2DS / f"{'leadV2_opt2p_fold' if mode == 'lead_V2' and k in OPT2P else 'fold'}{k:02d}/modified_dataset/modified_dataset.csv"
    return FOLD_DIR / f"{'cv5_optB_leadV2_opt2p_fold' if run == 'R2b' else 'cv5_optB_fold'}{k:02d}/modified_dataset/modified_dataset.csv"


def grid_file(run, arm, mode, cond):
    if run == "R1_published":
        return T1.pred_path("published", PUB_ARM[arm], mode, "grid", cond)
    d = "R1_control" if run == "R1" else run
    return ROOT / d / ("predictions" if cond == "clean" else "predictions_noisy") / f"{d}__{arm}__{mode}__grid.csv.gz"


def tg_file(run, arm, mode):
    if run == "R1_published":
        return PUB_TG / f"{mode}_{'pre' if arm == 'pretrained' else 'sup'}.csv.gz"
    d = "R1_control" if run == "R1" else run
    return ROOT / d / "predictions_training_grid" / f"{d}__{arm}__{mode}.csv.gz"


def beat_file(lead, cond):
    return ROOT / "R3" / ("predictions" if cond == "clean" else "predictions_noisy") / f"R3__pretrained__{lead}__every_beat.csv.gz"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ inference
def infer_grid(run):
    from ltst_pipeline.ml_inference import load_patchecg_model, predict_st_per_segment
    offs = pd.read_csv(OFFSETS).set_index("record").offset_samples
    store, failed, prov, t0 = {}, [], {}, time.time()

    def frame(rec, k, samp, st):
        n = len(samp)
        return pd.DataFrame({"record": rec, "fold": k, "anchor_samp": samp, "window_start_samp": samp - (WIN - 1), "pred_II_mV": st.get("II", np.full(n, np.nan, np.float32)), "pred_V2_mV": st.get("V2", np.full(n, np.nan, np.float32))})

    for k in sorted({f for _, _, fs in JOBS[run] for f in fs}):
        recs = T1.FOLDS[k]
        sig = {(r, c): T1.load_signal(r, c)[:, [0, 1]].astype(np.float32) for r in recs for c in ("clean", "noisy")}
        for mode, arm, fs in JOBS[run]:
            if k not in fs or (mode, arm) in failed:
                continue
            try:
                p = ckpt(run, mode, arm, k)
                prov[f"{mode}/{arm}/fold{k:02d}"] = {"checkpoint": str(p), "sha256": sha(p), "git_saved_mse": saved_mse(run, mode, arm, k)}
                model, revin, device = load_patchecg_model(str(p), device="cuda", lead_mode=mode)
                for r in recs:
                    for c in ("clean", "noisy"):
                        samp, st = predict_st_per_segment(sig[r, c], FS, model, revin, device, lead_order=("II", "V2"), batch_size=1024, lead_mode=mode)
                        store.setdefault((mode, arm, c), []).append(frame(r, k, samp, st))
                    o = int(offs[r])
                    n_seg = (len(sig[r, "clean"]) - o) // WIN
                    samp, st = predict_st_per_segment(np.ascontiguousarray(sig[r, "clean"][o:o + n_seg * WIN]), FS, model, revin, device, lead_order=("II", "V2"), batch_size=1024, lead_mode=mode)
                    f = frame(r, k, samp + o, st).rename(columns={"anchor_samp": "samp_end_raw"})
                    f.insert(1, "seg_idx", np.arange(len(f)))
                    store.setdefault((mode, arm, "tg"), []).append(f)
                del model
                log(run, f"{mode} {arm} fold{k}", f"{time.time() - t0:.0f}s")
            except Exception:  # noqa: BLE001
                failed.append((mode, arm))
                log(run, "FAILED", mode, arm, k, traceback.format_exc().replace("\n", " | "))
    for (mode, arm, c), parts in store.items():
        if (mode, arm) in failed:
            continue
        df = pd.concat(parts, ignore_index=True)
        if c == "tg":
            write(df.sort_values(["record", "seg_idx"]), tg_file(run, arm, mode))
        else:
            write(df.sort_values(["record", "anchor_samp"]), grid_file(run, arm, mode, c))
    (run_dir(run) / "provenance.json").write_text(json.dumps({"run": run, "failed": failed, "checkpoints": prov, "batch_size": 1024, "noise": "NSTDB em seed 42, T1.load_signal", "code": __file__}, indent=1))
    return failed


def infer_r3():
    import torch
    from ltst_pipeline.ml_inference import load_patchecg_model, _resample_to_model_fs, create_patch, PATCH_LEN, DataAugmentation
    dev, t0, rows, prov, nonfin = "cuda", time.time(), [], {}, {}

    @torch.no_grad()
    def run_model(model, revin, x, pk):  # evaluate_t1nk.infer's run(), unchanged
        out = np.full(len(pk), np.nan, np.float32)
        ok = (pk - HALF >= 0) & (pk + HALF <= len(x))
        idx = np.flatnonzero(ok)
        off = np.arange(WIN)
        for i in range(0, len(idx), 1024):
            j = idx[i:i + 1024]
            seg = x[pk[j, None] - HALF + off][..., None]
            xb = revin(torch.from_numpy(np.ascontiguousarray(_resample_to_model_fs(np.where(np.isfinite(seg), seg, 0.0).astype(np.float32), FS))).to(dev), "norm")
            out[j] = model(create_patch(xb, patch_len=PATCH_LEN, stride=PATCH_LEN), mask=None, inverted_mask=None, indices=None, augmentation=DataAugmentation.none).detach().cpu().numpy().reshape(len(j), -1)[:, 0]
        return out

    for cond in ("clean", "noisy"):
        rows = []
        for k in ALL:
            sig = {r: T1.load_signal(r, cond) for r in T1.FOLDS[k]}
            for lead in SINGLE:
                ch = CH[lead]
                p = ckpt("R3", lead, "pretrained", k)
                prov[f"{lead}/fold{k:02d}"] = {"checkpoint": str(p), "sha256": sha(p), "saved_mse": saved_mse("R3", lead, "pretrained", k)}
                model, revin, _ = load_patchecg_model(str(p), device=dev, lead_mode=lead)
                for r in T1.FOLDS[k]:
                    pk = np.unique(np.load(PEAKS / f"{cond}_{DET}_{r}_lead{ch}.npy").astype(np.int64))
                    x = sig[r][:, ch].astype(np.float64)
                    nonfin[f"{cond}/{r}/{ch}"] = int((~np.isfinite(x)).sum())
                    rows.append(pd.DataFrame(dict(record=r, fold=k, lead=lead, peak_samp=pk, window_start_samp=pk - HALF, pred_mV=run_model(model, revin, x, pk))))
                del model
                torch.cuda.empty_cache()
            log("R3", cond, f"fold{k}", f"{time.time() - t0:.0f}s")
        d = pd.concat(rows, ignore_index=True)
        for lead in SINGLE:
            write(d[d.lead == lead].sort_values(["record", "peak_samp"]), beat_file(lead, cond))
    (run_dir("R3") / "provenance.json").write_text(json.dumps({"run": "R3", "checkpoints": prov, "detector": DET, "peaks": str(PEAKS), "nonfinite_samples": nonfin, "batch_size": 1024, "noise": "NSTDB em seed 42, T1.load_signal; peaks detected on the noisy signal", "code": __file__}, indent=1))


def gate(run):
    rows = []
    if run == "R3":
        C = pd.read_csv(V2DS / "centres.csv")
        for lead in SINGLE:
            ch = CH[lead]
            P = pd.read_csv(beat_file(lead, "clean"), usecols=["record", "peak_samp", "pred_mV"]).set_index(["record", "peak_samp"]).pred_mV
            for k in ALL:
                meta = pd.read_csv(fold_csv("R3", lead, k), usecols=["strat_fold_annotated", f"target_lead{ch}_mV"])
                te = np.flatnonzero(meta.strat_fold_annotated.to_numpy() == 2)
                c = C[(C.lead == ch) & C.row.isin(te)].sort_values("row")
                assert len(c) == len(te), (lead, k, len(c), len(te))
                p = P.reindex(pd.MultiIndex.from_arrays([c.patient_id, c.centre_sample])).to_numpy()
                y = meta[f"target_lead{ch}_mV"].to_numpy()[c.row.to_numpy()]
                rows.append(dict(run=run, mode=lead, arm="pretrained", fold=k, n=len(te), n_missing=int(np.isnan(p).sum()), mse_here=float(np.nanmean((p - y) ** 2)), mse_saved=saved_mse(run, lead, "pretrained", k)))
    else:
        for mode, arm, fs in JOBS[run]:
            tg = pd.read_csv(tg_file(run, arm, mode))
            leads = OUT_LEADS[mode]
            for k in fs:
                md = pd.read_csv(fold_csv(run, mode, k), usecols=["patient_id", "segment_id", "strat_fold_annotated", *TARGET.values()])
                te = md[md.strat_fold_annotated == 2].copy()
                te["seg_idx"] = te.segment_id.str.slice(3).astype(int)
                m = te.merge(tg, left_on=["patient_id", "seg_idx"], right_on=["record", "seg_idx"], how="inner")
                assert len(m) == len(te), (run, mode, arm, k, len(m), len(te))
                p = m[[COL[x] for x in leads]].to_numpy(float)
                y = m[[TARGET[x] for x in leads]].to_numpy(float)
                rows.append(dict(run=run, mode=mode, arm=arm, fold=k, n=len(te), n_missing=int(np.isnan(p).sum()), mse_here=float(((p - y) ** 2).mean()), mse_saved=saved_mse(run, mode, arm, k)))
    G = pd.DataFrame(rows)
    G["rel_diff"] = (G.mse_here - G.mse_saved).abs() / G.mse_saved
    G["pass"] = (G.rel_diff <= 0.01) & (G.n_missing == 0)
    write(G, run_dir(run) / "gate_trainer_mse.csv")
    log(run, "MSE gate", f"{int(G['pass'].sum())}/{len(G)} within 1 %", f"worst {G.rel_diff.max():.5f}")
    return bool(G["pass"].all())


def control_predictions():
    """The published checkpoints through this code must equal the published prediction files."""
    rows = []
    for mode, arm, _ in JOBS["R1"]:
        for cond in ("clean", "noisy"):
            a, b = pd.read_csv(grid_file("R1", arm, mode, cond)), pd.read_csv(grid_file("R1_published", arm, mode, cond))
            m = a.merge(b, on=["record", "anchor_samp"], suffixes=("", "_pub"))
            d = np.nanmax(np.abs(np.concatenate([(m[c] - m[c + "_pub"]).to_numpy(float) for c in COL.values() if m[c].notna().any()])))
            rows.append(dict(mode=mode, arm=arm, grid=f"alarm {cond}", n_here=len(a), n_published=len(b), n_joined=len(m), max_abs_diff=float(d)))
        a, b = pd.read_csv(tg_file("R1", arm, mode)), pd.read_csv(tg_file("R1_published", arm, mode))
        m = a.merge(b, on=["record", "seg_idx", "samp_end_raw"], suffixes=("", "_pub"))
        d = np.nanmax(np.abs(np.concatenate([(m[c] - m[c + "_pub"]).to_numpy(float) for c in COL.values() if m[c].notna().any()])))
        rows.append(dict(mode=mode, arm=arm, grid="training", n_here=len(a), n_published=len(b), n_joined=len(m), max_abs_diff=float(d)))
    R = pd.DataFrame(rows)
    R["pass"] = (R.max_abs_diff < 1e-4) & (R.n_here == R.n_published) & (R.n_here == R.n_joined)
    write(R, run_dir("R1") / "control_vs_published_predictions.csv")
    log("R1", "predictions vs published", f"{int(R['pass'].sum())}/{len(R)} pass", f"worst {R.max_abs_diff.max():.2e}")
    return bool(R["pass"].all())


def combine():
    for lead in SINGLE:
        for cond in ("clean", "noisy", "tg"):
            def read(run):
                return pd.read_csv(tg_file(run, "pretrained", lead) if cond == "tg" else grid_file(run, "pretrained", lead, cond))
            a = read("R2")
            if lead == "lead_V2":
                b = read("R2b")
                assert sorted(b.fold.unique()) == list(OPT2P)
                a = pd.concat([a[~a.fold.isin(OPT2P)], b], ignore_index=True)
            key = "seg_idx" if cond == "tg" else "anchor_samp"
            assert a.record.nunique() == 15 and not a.duplicated(["record", key]).any()
            write(a.sort_values(["record", key]), tg_file("R2_R2b", "pretrained", lead) if cond == "tg" else grid_file("R2_R2b", "pretrained", lead, cond))
    log("combine", "R2_R2b written")


# ------------------------------------------------------------------------------------------ regression
def reg(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return dict(n=int(len(y)))
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(n=int(len(y)), MAE=float(np.abs(e).mean()), bias=float(e.mean()), RMSE=float(np.sqrt((e ** 2).mean())), slope=float(np.polyfit(y, p, 1)[0]), r=float(np.corrcoef(y, p)[0, 1]), p100_pct=float(100 * (np.abs(e) > 0.1).mean()), R2=1 - float((e ** 2).sum()) / sst if sst > 0 else np.nan)


GRID_RUNS = [("R1", "R1_published", MODES, ARMS), ("R1b", "R1b", MODES, ARMS), ("R2", "R2", MODES, ("pretrained",)), ("R2b", "R2b", ("lead_V2",), ("pretrained",)), ("R2_R2b", "R2_R2b", SINGLE, ("pretrained",))]


def dmy_table():
    d = pd.read_csv(DMY, usecols=["record", "lead", "sample_16a", "dmy_ee_mV"])
    d = d[d.record.isin(T1.REC2FOLD)].reset_index(drop=True)
    d["lead"] = d.lead.map({"ML2": "II", "MV2": "V2"})
    return d


def regression():
    truth = pd.read_csv(TRUTH, usecols=["patient_id", "segment_id", *TARGET.values()]).rename(columns={"patient_id": "record"})
    truth = truth[truth.record.isin(K.RECORDS_15)].copy()
    truth["seg_idx"] = truth.segment_id.str.slice(3).astype(int)
    D = dmy_table()
    rows, per_fold, per_rec, beats = [], [], [], []

    def add(run, arm, mode, lead, kind, df):
        """df: record, y, p. One row per records set x masked subset, plus per-fold and per-record rows."""
        if run == "R2b":
            df = df[df.record.isin(FOLDS145)]
        df = df.assign(fold=df.record.map(T1.REC2FOLD), masked=[(r, lead) in K.MANUAL_OMIT for r in df.record])
        tag = dict(run=run, arm=arm, mode=mode, lead=lead, kind=kind)
        sets = [("folds145" if run == "R2b" else "all", df)]
        if run == "R2" and mode == "lead_V2":  # the like-for-like set for R2b
            sets.append(("folds145", df[df.record.isin(FOLDS145)]))
        for recset, g in sets:
            rows.append({**tag, "records": recset, "subset": "all", **reg(g.y, g.p)})
            rows.append({**tag, "records": recset, "subset": "no_masked", **reg(g.y[~g.masked], g.p[~g.masked])})
        for sub, g in (("all", df), ("no_masked", df[~df.masked])):
            for k, q in g.groupby("fold"):
                per_fold.append({**tag, "fold": k, "subset": sub, **reg(q.y, q.p)})
        for r, q in df.groupby("record"):
            per_rec.append({**tag, "record": r, "masked": bool(q.masked.iloc[0]), **reg(q.y, q.p)})

    for run, src, modes, arms in GRID_RUNS:
        for arm in arms:
            for mode in modes:
                tg = pd.read_csv(tg_file(src, arm, mode))
                m = tg.merge(truth, on=["record", "seg_idx"], how="inner")
                ag = pd.read_csv(grid_file(src, arm, mode, "clean"))
                for lead in OUT_LEADS[mode]:
                    add(run, arm, mode, lead, "windows", pd.DataFrame(dict(record=m.record, y=m[TARGET[lead]], p=m[COL[lead]])))
                    q = D[D.lead == lead].copy()
                    q["anchor_samp"] = (q.sample_16a // WIN) * WIN + WIN - 1
                    q = q.merge(ag[["record", "anchor_samp", COL[lead]]], on=["record", "anchor_samp"], how="left").rename(columns={COL[lead]: "p", "dmy_ee_mV": "y"})
                    add(run, arm, mode, lead, "DMY", q[["record", "y", "p"]])
                    beats.append(q[["record", "lead", "sample_16a", "y", "p"]].assign(run=run, arm=arm, mode=mode))
        log("regression", run)
    C = pd.read_csv(V2DS / "centres.csv")
    meta = pd.read_csv(V2DS / "modified_dataset/modified_dataset.csv", usecols=list(TARGET.values()))
    for lead in SINGLE:
        L, ch = lead.split("_")[1], CH[lead]
        P = pd.read_csv(beat_file(lead, "clean"))
        c = C[(C.lead == ch) & C.patient_id.isin(K.RECORDS_15)]
        q = c.merge(P[["record", "peak_samp", "pred_mV"]], left_on=["patient_id", "centre_sample"], right_on=["record", "peak_samp"], how="left")
        add("R3", "pretrained", lead, L, "windows", pd.DataFrame(dict(record=q.patient_id, y=meta[TARGET[L]].to_numpy()[q.row.to_numpy()], p=q.pred_mV)))
        d, out = D[D.lead == L].copy(), []
        for r, g in d.groupby("record"):  # as c7_learned_dmy.test1nk: nearest detector beat, +/-100 ms
            p = P[P.record == r].sort_values("peak_samp")
            pk, pv = p.peak_samp.to_numpy(), p.pred_mV.to_numpy()
            s = g.sample_16a.to_numpy()
            i = np.clip(np.searchsorted(pk, s), 1, len(pk) - 1)
            j = np.where(np.abs(pk[i] - s) < np.abs(pk[i - 1] - s), i, i - 1)
            out.append(g.assign(p=np.where(np.abs(pk[j] - s) <= TOL, pv[j], np.nan)))
        q = pd.concat(out, ignore_index=True).rename(columns={"dmy_ee_mV": "y"})
        add("R3", "pretrained", lead, L, "DMY", q[["record", "y", "p"]])
        beats.append(q[["record", "lead", "sample_16a", "y", "p"]].assign(run="R3", arm="pretrained", mode=lead))
    B = pd.concat(beats, ignore_index=True)
    for lead in SINGLE:  # R3's DMY comparison on the beats both runs have
        L = lead.split("_")[1]
        a = B[(B.run == "R3") & (B.lead == L)].set_index(["record", "sample_16a"])
        b = B[(B.run == "R2_R2b") & (B.lead == L)].set_index(["record", "sample_16a"])
        common = a.index[a.p.notna()].intersection(b.index[b.p.notna()])
        for run, x in (("R3", a), ("R2_R2b", b)):
            g = x.loc[common].reset_index()
            add(run, "pretrained", lead, L, "DMY_common", g[["record", "y", "p"]])
    SC.mkdir(parents=True, exist_ok=True)
    write(pd.DataFrame(rows), SC / "regression.csv")
    write(pd.DataFrame(per_fold), SC / "regression_per_fold.csv")
    write(pd.DataFrame(per_rec), SC / "regression_per_record.csv")
    write(B, SC / "dmy_per_beat.csv.gz")
    log("regression", "done", len(rows), "rows")


def regression_t1nk():
    """Extra comparison, added at the user's request on 2026-11-09: Test 1-NK beside R3. It is not the README's comparison run: it differs from R3 in detector, backbone, V2 validation and window selection.
    Windows: Test 1-NK's own, only those centred on a NeuroKit peak (evaluate_t1nk's nk_matched rows). DMY: nearest NeuroKit beat within 100 ms, as C section 9. DMY_common3: the beats R2_R2b, R3 and Test 1-NK all have. Writes _scores/regression_t1nk.csv."""
    P = pd.read_pickle(V.HERE / "t1nk_every_beat_clean.pkl").query("seed == 200")
    C = pd.read_csv(V.DS / "centres.csv")
    meta = pd.read_csv(V.DS / "modified_dataset/modified_dataset.csv", usecols=list(TARGET.values()))
    D = dmy_table()
    B = pd.read_csv(SC / "dmy_per_beat.csv.gz")
    rows = []

    def add(run, mode, lead, kind, record, y, p):
        df = pd.DataFrame(dict(record=np.asarray(record), y=np.asarray(y, float), p=np.asarray(p, float)))
        m = np.array([(r, lead) in K.MANUAL_OMIT for r in df.record])
        tag = dict(run=run, arm="pretrained", mode=mode, lead=lead, kind=kind, records="all")
        rows.append({**tag, "subset": "all", **reg(df.y, df.p)})
        rows.append({**tag, "subset": "no_masked", **reg(df.y[~m], df.p[~m])})

    for lead in SINGLE:
        L, ch = lead.split("_")[1], CH[lead]
        Q = P[P.lead == lead]
        c = C[(C.lead == ch) & C.patient_id.isin(K.RECORDS_15) & C.nk_matched]
        q = c.merge(Q[["record", "nk_sample", "pred_mV"]], left_on=["patient_id", "centre_sample"], right_on=["record", "nk_sample"], how="left")
        add("T1NK", lead, L, "windows", q.patient_id, meta[TARGET[L]].to_numpy()[q.row.to_numpy()], q.pred_mV)
        out = []
        for r, g in D[D.lead == L].groupby("record"):
            x = Q[Q.record == r].sort_values("nk_sample")
            pk, pv = x.nk_sample.to_numpy(), x.pred_mV.to_numpy()
            s = g.sample_16a.to_numpy()
            i = np.clip(np.searchsorted(pk, s), 1, len(pk) - 1)
            j = np.where(np.abs(pk[i] - s) < np.abs(pk[i - 1] - s), i, i - 1)
            out.append(g.assign(p=np.where(np.abs(pk[j] - s) <= TOL, pv[j], np.nan)))
        d = pd.concat(out, ignore_index=True).rename(columns={"dmy_ee_mV": "y"})
        add("T1NK", lead, L, "DMY", d.record, d.y, d.p)
        a = B[(B.run == "R3") & (B.lead == L)].set_index(["record", "sample_16a"])
        b = B[(B.run == "R2_R2b") & (B.lead == L)].set_index(["record", "sample_16a"])
        t = d.set_index(["record", "sample_16a"])
        common = a.index[a.p.notna()].intersection(b.index[b.p.notna()]).intersection(t.index[t.p.notna()])
        for run, x in (("R3", a), ("R2_R2b", b), ("T1NK", t)):
            g = x.loc[common].reset_index()
            add(run, lead, L, "DMY_common3", g.record, g.y, g.p)
    write(pd.DataFrame(rows), SC / "regression_t1nk.csv")
    log("regression_t1nk", "done", len(rows), "rows")


# ------------------------------------------------------------------------------------------ alarms
def r3_traces(D):
    """{scheme: {lead: frame}} from every-beat rows with columns lead, seed, record, nk_sample, pred_mV: Test 1-NK's functions, unchanged."""
    tr = {lead: {**E.traces(D, lead, 200), "greedy10": V.greedy(D, lead)} for lead in SINGLE}
    return {sc: {lead: tr[lead][sc] for lead in SINGLE} for sc in ("greedy10", "cad10", "hop2")}


def variants(tag):
    cond = "clean" if tag == "clean" else "noisy"
    out = []
    if tag.startswith("t1nk_gate_"):  # extra (user, 2026-11-09): Test 1-NK's stored predictions, gated with detector peaks
        T = pd.read_pickle(V.HERE / "t1nk_every_beat_noisy.pkl").query("seed == 200")
        for sc, by in r3_traces(T).items():
            out += [("T1NK_control", "pretrained", lead, sc, by[lead]) for lead in SINGLE]
        return out
    grid = GRID_RUNS[:1] if tag == "noisy_sqi_6dB_neurokit" else GRID_RUNS
    for run, src, modes, arms in grid:
        for arm in arms:
            for mode in modes:
                out.append((run, arm, mode, "grid", pd.read_csv(grid_file(src, arm, mode, cond))))
    if tag == "noisy_sqi_6dB_neurokit":
        return out
    D = pd.concat([pd.read_csv(beat_file(lead, cond)) for lead in SINGLE], ignore_index=True).rename(columns={"peak_samp": "nk_sample"}).assign(seed=200)
    for sc, by in r3_traces(D).items():
        out += [("R3", "pretrained", lead, sc, by[lead]) for lead in SINGLE]
    if tag in ("clean", "noisy", "noisy_sqi_6dB"):  # control: Test 1-NK's stored predictions through this path
        T = pd.read_pickle(V.HERE / f"t1nk_every_beat_{cond}.pkl").query("seed == 200")
        for sc, by in r3_traces(T).items():
            out += [("T1NK_control", "pretrained", lead, sc, by[lead]) for lead in SINGLE]
    return out


def det_peaks(det, rec, ch):
    return np.sort(pd.read_csv(DELIN / f"delin_noise_nofilter_{det}" / f"lead_{ch}" / f"waves_{rec}_lead{ch}.csv", usecols=["r_peak"]).r_peak.to_numpy().astype(int))


def alarms(tag):
    gs = K.load_gs(T1.GS_PATH)
    durations = {r.record: float(r.duration_s) for r in pd.read_csv(T1.DUR_CSV).itertuples()}
    mask = set(K.MANUAL_OMIT)
    source = {"clean": None, "noisy": None, "noisy_sqi_6dB": "atr", "noisy_sqi_6dB_rodrigues2021": "rodrigues2021", "noisy_sqi_6dB_neurokit": "neurokit", "t1nk_gate_neurokit": "neurokit", "t1nk_gate_rodrigues2021": "rodrigues2021"}[tag]
    ctx = None
    if source:  # one process, one peak source: the SQI memo in sqi_context keys on (record, lead, beat), so sources must not share it
        c = T1.sqi_context("noisy_sqi")
        views = c.pop("_views")
        cols = {r: T1._Cols(views[(r, 0)], views[(r, 1)]) for r in T1.REC2FOLD}
        ctx = {r: (cols[r], c[r][1]) for r in T1.REC2FOLD} if source == "atr" else {ch: {r: (cols[r], det_peaks(source, r, ch)) for r in T1.REC2FOLD} for ch in (0, 1)}

    def fsm(df):
        if source in (None, "atr"):
            return T1.alarms_from_predictions(df, ctx, THRESHOLD)
        parts = []
        for ch, (col, other) in enumerate((("pred_II_mV", "pred_V2_mV"), ("pred_V2_mV", "pred_II_mV"))):
            if df[col].notna().any():
                parts.append(T1.alarms_from_predictions(df.assign(**{other: np.nan}), ctx[ch], THRESHOLD))
        return pd.concat(parts, ignore_index=True)

    al, t0 = {}, time.time()
    for run, arm, mode, sc, df in variants(tag):
        al[(run, arm, mode, sc)] = K.apply_mask(fsm(df), mask)
        log("alarms", tag, run, arm, mode, sc, f"{time.time() - t0:.0f}s")
    for (run, arm, mode, sc) in list(al):
        if mode == "lead_II" and (run, arm, "lead_V2", sc) in al:
            al[(run, arm, "union", sc)] = pd.concat([al[(run, arm, "lead_II", sc)].query("lead == 'II'"), al[(run, arm, "lead_V2", sc)].query("lead == 'V2'")], ignore_index=True)
    rows, comps = [], []
    for (run, arm, mode, sc), df in al.items():
        write(df, ROOT / run / "alarms" / tag / f"{run}__{arm}__{mode}__{sc}.csv")
        records = FOLDS145 if run == "R2b" else K.RECORDS_15
        for scope in T1.SCOPES[mode]:
            c = per_record_components(gs_scope(gs, scope), K.scope_filter(df, scope), durations, scope, mask, records)
            key = dict(condition=tag, run=run, arm=arm, mode=mode, scheme=sc, scope=scope)
            rows.append({**key, **metrics_from_components(c), "false_hours": c.tFP.sum() / 3600, "alarm_hours": c.pred_dur.sum() / 3600, "n_records": len(records)})
            comps.append(c.assign(**key))
    SC.mkdir(parents=True, exist_ok=True)
    write(pd.DataFrame(rows), SC / f"alarms_{tag}.csv")
    write(pd.concat(comps, ignore_index=True), SC / f"alarm_components_{tag}.csv")
    (SC / f"DONE_alarms_{tag}").write_text(time.strftime("%F %T") + "\n")
    log("alarms", tag, "done", f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------ controls
def controls():
    lines, ok_all = [], True

    def out(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    def same(a, b, on, cols, tol=1e-9):
        m = a.merge(b, on=on, suffixes=("", "_ref"))
        bad = [c for c in cols if not np.allclose(m[c].to_numpy(float), m[c + "_ref"].to_numpy(float), atol=tol, rtol=0, equal_nan=True)]
        return len(m), bad

    p = run_dir("R1") / "control_vs_published_predictions.csv"
    if p.exists():
        R = pd.read_csv(p)
        out("published checkpoints through this code equal the published prediction files", R["pass"].all(), f"{len(R)} files, worst |diff| {R.max_abs_diff.max():.2e} mV")
    for run in ("R1", "R1b", "R2", "R2b", "R3"):
        p = run_dir(run) / "gate_trainer_mse.csv"
        if p.exists():
            G = pd.read_csv(p)
            out(f"{run}: predictions at the trainer's test rows reproduce the saved test MSE within 1 %", G["pass"].all(), f"{int(G['pass'].sum())}/{len(G)}, worst {G.rel_diff.max():.4%}")
    cols = ["TP", "FP", "FN", "Ignored", "n_predicted_alarms", "F1", "time_F1", "IoU_whole_record", "sensitivity", "PPV_time"]
    for tag in ("clean", "noisy", "noisy_sqi_6dB"):
        if not (SC / f"alarms_{tag}.csv").exists():
            out(f"alarm scores for {tag}", False, "file missing")
            continue
        A = pd.read_csv(SC / f"alarms_{tag}.csv")
        ref = pd.read_csv(f"/home/student/GIT/results/beatcenter_test1/metrics_{tag}.csv").query("model_set == 'published' and scheme == 'grid' and arm != 'PTBXL-pretrained'")
        mine = A[A.run == "R1"].assign(arm=lambda d: d.arm.map(PUB_ARM))
        n, bad = same(mine, ref, ["arm", "mode", "scope"], cols)
        out(f"published predictions through this scorer equal metrics_{tag}.csv", n == len(ref) and not bad, f"{n}/{len(ref)} rows, differing: {bad}")
        ref = pd.read_csv(V.HERE / f"alarm_metrics_{tag}.csv")
        ref = ref[(ref.model == "Test 1-NK") & ref.scheme.isin(["greedy10", "cad10", "hop2"])]
        mine = A[A.run == "T1NK_control"]
        n, bad = same(mine, ref, ["mode", "scheme", "scope"], cols)
        out(f"Test 1-NK's stored predictions through R3's path equal alarm_metrics_{tag}.csv", n == len(ref) and not bad, f"{n}/{len(ref)} rows, differing: {bad}")
    p = SC / "alarms_noisy_sqi_6dB_neurokit.csv"
    if p.exists():
        A = pd.read_csv(p).query("run == 'R1'").assign(arm=lambda d: d.arm.map(PUB_ARM))
        A = A[((A["mode"] == "union") & (A.scope == "FULL")) | ((A["mode"] == "lead_II") & (A.scope == "own_II")) | ((A["mode"] == "lead_V2") & (A.scope == "own_V2"))].rename(columns={"IoU_whole_record": "IoU"})
        ref = pd.read_csv(DS / "C_neurokit/c8_regate/results_regate_6dB.csv").query("rpeaks == 'nk'")
        n, bad = same(A, ref, ["arm", "mode"], ["TP", "FP", "FN", "IoU"], tol=1e-9)
        out("the detector-gate path with NeuroKit peaks equals c8_regate (section 10b)", n == len(ref) and not bad, f"{n}/{len(ref)} rows, differing: {bad}")
    R = pd.read_csv(SC / "regression.csv")
    b1 = pd.read_csv("/home/student/GIT/results/regression/B1_regression_metrics.csv").query("aggregation == 'pooled' and lead_output in ['II', 'V2']")
    b1 = b1.assign(mode=b1.lead_mode, arm=b1.arm.map({"supervised": "supervised", "Unified-pretrained": "pretrained"}), lead=b1.lead_output).rename(columns={"MAE_mV": "MAE", "RMSE_mV": "RMSE"})
    mine = R[(R.run == "R1") & (R.kind == "windows") & (R.subset == "all")]
    n, bad = same(mine, b1, ["mode", "arm", "lead"], ["n", "R2", "MAE", "RMSE"], tol=1e-6)
    out("published training-grid predictions through this code equal B1_regression_metrics.csv", n == len(b1) and not bad, f"{n}/{len(b1)} rows, differing: {bad}")
    c7 = pd.read_csv(DS / "C_neurokit/c7_learned_dmy/c7_dmy_learned_vs_nk.csv")
    B = pd.read_csv(SC / "dmy_per_beat.csv.gz")
    B = B[(B.run == "R1") & B["mode"].isin(SINGLE)]
    worst, n = 0.0, 0
    for arm, name in (("supervised", "E0_supervised_grid"), ("pretrained", "E0_Chapman-pretrained_grid")):
        for lead, lab in (("II", "ML2"), ("V2", "MV2")):
            a = B[(B.arm == arm) & (B.lead == lead)].merge(c7[c7.lead == lab][["record", "sample_16a", name]], on=["record", "sample_16a"])
            ok = a.p.notna() & a[name].notna()
            n += int(ok.sum())
            worst = max(worst, float((a.p - a[name])[ok].abs().max()))
    out("DMY grid lookup equals C section 9's per-beat predictions", worst < 1e-3 and n > 0, f"{n} beat-predictions, worst |diff| {worst:.2e} mV")
    SC.mkdir(parents=True, exist_ok=True)
    (SC / "controls.txt").write_text("\n".join(lines) + f"\n{'ALL PASSED' if ok_all else 'SOME FAILED'}\n")
    print("\n".join(lines), flush=True)
    return ok_all


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "infer":
        run = a[1]
        if run == "R3":
            infer_r3()
        else:
            failed = infer_grid(run)
            if failed:
                sys.exit(f"inference failed for {failed}")
        ok = gate(run)
        if run == "R1":
            ok &= control_predictions()
            if not ok:
                sys.exit("control failed")
        elif not ok:
            log(run, "GATE FAILED: see gate_trainer_mse.csv")
            (run_dir(run) / "GATE_FAILED").write_text("see gate_trainer_mse.csv\n")
    elif a[0] == "combine":
        combine()
    elif a[0] == "regression":
        regression()
    elif a[0] == "regression_t1nk":
        regression_t1nk()
    elif a[0] == "alarms":
        alarms(a[1])
    elif a[0] == "controls":
        sys.exit(0 if controls() else "controls failed")
    else:
        sys.exit(__doc__)
