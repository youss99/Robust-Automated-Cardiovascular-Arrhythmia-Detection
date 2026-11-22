#!/usr/bin/env python3
"""R-peak finalists through the real delineation: offset, DMY comparison, real alarms. rodrigues2021 and khamis2016, with neurokit as the control.

Input: Delineation Data/delin_{raw,noise}_nofilter_<detector> (c17_delineate.py; neurokit = the stored delineation).
Per beat, as C §7, §8 and §10: ST = 5-sample mean at the delineated J+60, baseline = 5-sample mean at R-peak minus an offset, ST deviation = ST - baseline (c6_noisy_fold.per_beat).

Offsets (ms before that detector's R-peak):
  rpk64   fixed 64 ms (C §8)
  rpkfold the stored method: median DMY ISO position over the training records of the record's fold, the same for every detector (C §10)
  rpkdet  re-derived for each detector and lead: median, over the DMY beats of the training records of the record's fold, of the DMY ISO point
          measured from that detector's clean R-peak nearest the beat (within 100 ms). This allows for where each detector puts the R-peak.
Folds as c6_noisy_fold.FOLDS. s20121 is in no fold: its beats are written for the DMY step only, with offsets from all fold records, and it is never scored.

Alarms, through the real FSM and the paper's scorer: clean, noisy (NSTDB em, seed 42), noisy + SQI gate 6 dB (gate on the contaminated signal after a
0.67 Hz high-pass, 30 s window, 5 s cadence) with that detector's own noisy R-peaks. 10 s grid and median of at least 5 usable beats, as C §10.
DMY comparison: c4_dmy_vs_nk.py with that detector's clean delineation (window median over each DMY beat's NL/NR beats).

Controls, which stop the run if they fail:
  neurokit per-beat values must equal the stored ones: clean st5mean_rpk64, st5mean_rpkfold (c5_iso); noisy noisy_st5mean_rpk64, noisy_st5mean_rpkfold (c6_noisy).
  neurokit gated rpk64 and rpkfold alarms must equal the stored gated_nk/sqi6.csv; clean rpkfold must equal c5_iso's alarms.csv.
  neurokit DMY values must equal Tasks/V2_diagnostics/out/dmy_nk_rpkfold.csv.

Stages: offsets | alarms <workers> | dmy | score | all <workers>. Detached and relaunch-safe (finished records and stages are skipped).
Output: DATASET NK/C_neurokit/c20_finalists/ (offsets.csv, st_beats/, per_record/, dmy_per_beat_<det>.csv, results_*.csv, controls.txt, DONE)."""
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TASK = Path("/home/student/GIT/Tasks/C_neurokit")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(TASK))
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
import c6_noisy_fold as C6  # noqa: E402  (also patches wfdb to read the local NSTDB em)
import c13_fp_gap as C13  # noqa: E402
import c4_dmy_vs_nk as C4  # noqa: E402
from c14_detectors_noisy import run_records  # noqa: E402
from ltst_pipeline import old_fsm_gs as F  # noqa: E402
from ltst_pipeline.filters import highpass_filter  # noqa: E402

DS = Path("/home/student/Desktop/Research Data/DATASET NK")
CN = DS / "C_neurokit"
OUT = CN / "c20_finalists"
PER = OUT / "per_record"
STB = OUT / "st_beats"
DEL = DS / "Delineation Data"
DETS = ("neurokit", "rodrigues2021", "khamis2016")
CFGS = ("rpk64", "rpkfold", "rpkdet")
FS, THR = 250, 6.0
COLS = ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_point", "J_plus_60ms"]
FOLD_OF = {r: k for k, v in C6.FOLDS.items() for r in v}
TEST = sorted(FOLD_OF)


def delin(det, cond, rec, lead, cols=COLS):
    return pd.read_csv(DEL / f"delin_{'raw' if cond == 'clean' else 'noise'}_nofilter_{det}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=cols)


# ------------------------------------------------------------------ offsets
def offsets():
    t = OUT / "offsets.csv"
    if t.exists():
        return pd.read_csv(t)
    rows = []
    for rec in sorted(set(C6.RECORDS)):
        a16 = C4.load_16a(rec)
        for lead in (0, 1):
            L = a16[(a16.lead_idx == lead) & a16.DMY].sort_values("sample")
            if L.empty:
                continue
            s16 = L["sample"].to_numpy()
            for det in DETS:
                R = np.sort(delin(det, "clean", rec, lead, ["r_peak"]).r_peak.to_numpy().astype(int))
                i = np.clip(np.searchsorted(R, s16), 1, len(R) - 1)
                near = np.where(np.abs(R[i] - s16) < np.abs(R[i - 1] - s16), R[i], R[i - 1])
                ok = np.abs(near - s16) <= C4.TOL
                # ISO sample = fiducial - ISO_ms, fiducial = 16a sample - 16 ms; distance before the detector's R-peak, in ms
                iso_before_r = (near - (s16 - C4.FID_SHIFT - L.ISO_ms.to_numpy() / 4.0)) * 4.0
                rows += [dict(record=rec, lead=lead, detector=det, iso_before_r_ms=v, dmy_ISO_ms=q) for v, q in zip(iso_before_r[ok], L.ISO_ms.to_numpy()[ok])]
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "offsets_per_dmy_beat.csv", index=False)
    stored = pd.read_csv(CN / "c6_noisy/fold_offsets.csv", index_col=0).iloc[:, 0].to_dict()
    out = []
    for fold, test in C6.FOLDS.items():
        train = [r for r in TEST if r not in test]
        for rec in test:
            for det in DETS:
                for lead in (0, 1):
                    q = B[(B.detector == det) & (B.lead == lead) & B.record.isin(train)]
                    out.append(dict(record=rec, fold=fold, detector=det, lead=lead, rpk64=64.0, rpkfold=float(stored[rec]), rpkdet=float(q.iso_before_r_ms.median()), n_dmy_train=len(q)))
    for det in DETS:  # s20121 is in no fold: written only so the DMY step has every record, never scored
        for lead in (0, 1):
            q = B[(B.detector == det) & (B.lead == lead) & B.record.isin(TEST)]
            out.append(dict(record="s20121", fold="none", detector=det, lead=lead, rpk64=64.0, rpkfold=float(stored["s20121"]), rpkdet=float(q.iso_before_r_ms.median()), n_dmy_train=len(q)))
    O = pd.DataFrame(out)
    # the stored method, recomputed here: must equal c6's fold offsets
    d = pd.read_csv(CN / "c5_iso/c5_dmy_per_beat.csv")
    for fold, test in C6.FOLDS.items():
        med = float(d[d.record.isin(TEST) & ~d.record.isin(test)].dmy_ISO_ms.median())
        assert all(abs(stored[r] - med) < 1e-9 for r in test), fold
    O.to_csv(t, index=False)
    return O


# ------------------------------------------------------------------ alarms
def job(task):
    rec = task[0]
    target = PER / f"alarms_{rec}.csv"
    if target.exists():
        return rec
    O = pd.read_csv(OUT / "offsets.csv")
    clean, noisy = C6.noisy_signal(rec)
    hp = highpass_filter(noisy, fs=FS)
    rows, ctl = [], []

    def keep(alarms, det, cfg, cond):
        rows.extend(dict(detector=det, config=cfg, condition=cond, record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, onset_actual_s=a.onset_actual / FS, end_actual_s=a.end_actual / FS, criterion=a.crit) for a in alarms)

    for lead in (0, 1):
        start, end = C13.grid(rec, lead)
        ecg = np.ascontiguousarray(hp[:, lead])
        for det in DETS:
            dc, dn = delin(det, "clean", rec, lead), delin(det, "noisy", rec, lead)
            gate_peaks = np.sort(dn.r_peak.to_numpy().astype(int))
            memo = {}
            o = O[(O.record == rec) & (O.detector == det) & (O.lead == lead)].iloc[0]
            for cfg in CFGS:
                off = float(o[cfg])
                st_c, iso_c = C6.per_beat(clean[:, lead], dc, "rpk", off)
                st_n, iso_n = C6.per_beat(noisy[:, lead], dn, "rpk", off)
                C6.write(STB / det, cfg, rec, lead, dc, st_c, iso_c)
                if det == "neurokit" and cfg in ("rpk64", "rpkfold"):
                    for cond, v, ref in (("clean", st_c - iso_c, CN / f"c5_iso/st_beats/st5mean_{cfg}/{rec}/lead_{lead}/waves_{rec}_lead{lead}_st5mean_{cfg}.csv"), ("noisy", st_n - iso_n, CN / f"c6_noisy/st_beats/noisy_st5mean_{cfg}/{rec}/lead_{lead}/waves_{rec}_lead{lead}_noisy_st5mean_{cfg}.csv")):
                        r = pd.read_csv(ref, usecols=["st_deviation_mV"]).st_deviation_mV.to_numpy()
                        ctl.append(dict(record=rec, lead=lead, config=cfg, condition=cond, same=bool(np.allclose(np.nan_to_num(r, nan=9e9), np.nan_to_num(np.round(v, 6), nan=9e9), atol=1e-9))))
                sc = C13.segments(dc.J_plus_60ms.to_numpy(float), np.round(st_c - iso_c, 6), start)
                sn = C13.segments(dn.J_plus_60ms.to_numpy(float), np.round(st_n - iso_n, 6), start)
                with contextlib.redirect_stdout(io.StringIO()):
                    keep(F.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=end.tolist(), st_arr=sc.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=False), det, cfg, "clean")
                    keep(F.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=False), det, cfg, "noisy")
                for s in end:
                    if int(s) not in memo:
                        memo[int(s)] = F.compute_sqi_last30s(r_peaks=gate_peaks, sig_filt=ecg, current_r_peak=int(s), pre_sec=0.35, post_sec=0.35, window_sec=30.0, fs=float(FS), min_beats=3)
                orig = F.compute_sqi_last30s
                F.compute_sqi_last30s = lambda r_peaks, sig_filt, current_r_peak, pre_sec, post_sec, window_sec, fs, min_beats: memo[int(current_r_peak)]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        keep(F.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=ecg, r_peaks=gate_peaks, fs=FS, use_sqi=True, sqi_threshold_db=THR, sqi_every_s=5.0, sqi_window_s=30.0), det, cfg, "gated6")
                finally:
                    F.compute_sqi_last30s = orig
    pd.DataFrame(ctl).to_csv(PER / f"control_{rec}.csv", index=False)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["detector", "config", "condition", "record", "lead_tag", "lead_paper", "onset_actual_s", "end_actual_s", "criterion"]).to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(rec, "done", flush=True)
    return rec


# ------------------------------------------------------------------ DMY
def dmy():
    ref = pd.read_csv("/home/student/GIT/Tasks/V2_diagnostics/out/dmy_nk_rpkfold.csv")
    lines = []
    for det in DETS:
        t = OUT / f"dmy_per_beat_{det}.csv"
        if not t.exists():
            C4.DELIN = DEL / f"delin_raw_nofilter_{det}"
            sys.argv = ["c4", "--st_root", str(STB / det), "--configs", *CFGS, "--out_csv", str(t)]
            with contextlib.redirect_stdout(io.StringIO()):
                C4.main()
        if det == "neurokit":
            d = pd.read_csv(t)
            m = d.merge(ref, on=["record", "lead", "sample_16a"], suffixes=("", "_ref"))
            for cfg in ("rpk64", "rpkfold"):
                a, b = m[f"{cfg}_median_mV"].to_numpy(), m[f"st5mean_{cfg}_median_mV"].to_numpy()
                same = len(m) == len(ref) == len(d) and np.allclose(np.nan_to_num(a, nan=9e9), np.nan_to_num(b, nan=9e9), atol=1e-9)
                lines.append(f"DMY, neurokit {cfg}: {len(d)} DMY beats here, {len(ref)} stored, identical: {same}")
    return lines


# ------------------------------------------------------------------ score
def score(dmy_lines):
    pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/1_ltst_paper_cmbec48/results/alarms/_code")
    from c3_score_paper import K, GS_PATH, DURATIONS
    from score_alarms import per_record_components, gs_scope
    gs = K.load_gs(str(GS_PATH))
    dur = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}
    mask = set(K.MANUAL_OMIT)
    A = pd.concat([pd.read_csv(f, dtype={"lead_paper": str}) for f in sorted(PER.glob("alarms_s2*.csv"))], ignore_index=True)
    lines = list(dmy_lines)
    C = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("control_s2*.csv"))], ignore_index=True)
    lines.append(f"per-beat values, neurokit against stored: {int(C.same.sum())} of {len(C)} record-lead-config-condition files identical")
    cols = ["record", "lead_paper", "onset_actual_s", "end_actual_s"]
    for cfg, cond, stored in (("rpk64", "gated6", CN / "c6_noisy/runs_seg10s_min5/noisy_st5mean_rpk64/gated_nk/sqi6.csv"), ("rpkfold", "gated6", CN / "c6_noisy/runs_seg10s_min5/noisy_st5mean_rpkfold/gated_nk/sqi6.csv"), ("rpkfold", "clean", CN / "c5_iso/runs_seg10s_min5/st5mean_rpkfold/alarms.csv"), ("rpk64", "clean", CN / "c5_iso/runs_seg10s_min5/st5mean_rpk64/alarms.csv")):
        a = A[(A.detector == "neurokit") & (A.config == cfg) & (A.condition == cond)]
        b = pd.read_csv(stored, dtype={"lead_paper": str})
        a, b = [x[x.record.isin(TEST) & [(r, l) not in mask for r, l in zip(x.record, x.lead_paper)]][cols].sort_values(cols).round(3).reset_index(drop=True) for x in (a, b)]
        lines.append(f"alarms, neurokit {cfg} {cond} against stored: {len(a)} here, {len(b)} stored, identical: {a.equals(b)}")
    pooled, per = [], []
    TMP = OUT / "tmp_alarms.csv"
    for (det, cfg, cond), g in A.groupby(["detector", "config", "condition"]):
        g.drop(columns=["detector", "config", "condition"]).to_csv(TMP, index=False)
        raw = K.load_alarms(str(TMP))
        for mode, scope in {"union": "FULL", "lead_II": "own_II", "lead_V2": "own_V2"}.items():
            df = raw if mode == "union" else raw[raw["lead"] == mode.split("_")[1]]
            pred = K.scope_filter(K.apply_mask(df.reset_index(drop=True), mask), scope)
            c = per_record_components(gs_scope(gs, scope), pred, dur, scope, mask, K.RECORDS_15)
            tp, fp, fn = int(c.TP.sum()), int(c.FP.sum()), int(c.FN.sum())
            d = c.tTP.sum() + c.tFP.sum() + c.tFN.sum()
            pooled.append(dict(detector=det, config=cfg, condition=cond, mode=mode, TP=tp, FP=fp, FN=fn, ep_F1=2 * tp / (2 * tp + fp + fn), IoU=c.tTP.sum() / d, false_hours=c.tFP.sum() / 3600, alarm_hours=c.pred_dur.sum() / 3600))
            for x in c.itertuples():
                dd = x.tTP + x.tFP + x.tFN
                per.append(dict(detector=det, config=cfg, condition=cond, mode=mode, record=x.record, TP=x.TP, FP=x.FP, FN=x.FN, IoU=x.tTP / dd if dd else np.nan, false_hours=x.tFP / 3600))
    TMP.unlink()
    P = pd.DataFrame(pooled)
    P.to_csv(OUT / "results_alarms.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "results_alarms_per_record.csv", index=False)
    # the stored paper scores for the neurokit controls
    S = pd.read_csv(CN / "c6_noisy/results_noisy_paper_scoring.csv")
    for cfg in ("rpk64", "rpkfold"):
        s = S[(S.config == f"noisy_st5mean_{cfg}") & (S.min_beats == 5) & (S.sqi_db == 6) & S.condition.str.contains("nk")]
        for mode in ("union", "lead_II", "lead_V2"):
            x, y = P[(P.detector == "neurokit") & (P.config == cfg) & (P.condition == "gated6") & (P["mode"] == mode)].iloc[0], s[s["mode"] == mode].iloc[0]
            lines.append(f"score, neurokit {cfg} gated {mode}: {x.TP}/{x.FP}/{x.FN} here, {int(y.TP)}/{int(y.FP)}/{int(y.FN)} stored, identical: {(x.TP, x.FP, x.FN) == (int(y.TP), int(y.FP), int(y.FN)) and abs(x.IoU - y.IoU) < 1e-9}")
    # DMY agreement, per detector, config and lead, with and without masked records
    rows = []
    MASKED = {"ML2": {"s20091", "s20121"}, "MV2": {"s20081", "s20121", "s20221", "s20241"}}
    for det in DETS:
        d = pd.read_csv(OUT / f"dmy_per_beat_{det}.csv")
        for cfg in CFGS:
            for lead in ("ML2", "MV2"):
                for sub in ("all", "no masked"):
                    q = d[(d.lead == lead) & d.record.isin(TEST)]
                    if sub != "all":
                        q = q[~q.record.isin(MASKED[lead])]
                    x, y = q[f"{cfg}_median_mV"], q.dmy_ee_mV
                    m = x.notna()
                    e = (x - y)[m]
                    rows.append(dict(detector=det, config=cfg, lead=lead, subset=sub, dmy_beats=len(q), usable=int(m.sum()), bias=e.mean(), MAE=e.abs().mean(), RMSE=float(np.sqrt((e ** 2).mean())), share_err_over_0p1=100 * float((e.abs() > 0.1).mean()), r=float(np.corrcoef(x[m], y[m])[0, 1]), slope=float(np.polyfit(y[m], x[m], 1)[0])))
    pd.DataFrame(rows).to_csv(OUT / "results_dmy.csv", index=False)
    (OUT / "controls.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    ok = all(x.endswith("True") for x in lines if "identical:" in x) and int(C.same.sum()) == len(C)
    return ok


if __name__ == "__main__":
    stage = sys.argv[1]
    for d in (OUT, PER, STB):
        d.mkdir(parents=True, exist_ok=True)
    if stage in ("offsets", "all"):
        offsets()
    if stage in ("alarms", "all"):
        failed = run_records(job, [(r,) for r in TEST + ["s20121"]], int(sys.argv[2]), need_gb=6.0)
        if failed:
            (OUT / "FAILED").write_text(f"failed records: {failed}\n")
            sys.exit(f"failed records: {failed}")
    if stage in ("dmy", "score", "all"):
        ok = score(dmy())
        if not ok:
            (OUT / "CONTROL_FAILED").write_text((OUT / "controls.txt").read_text())
            sys.exit("a control failed: see controls.txt")
        (OUT / "FAILED").unlink(missing_ok=True)
        (OUT / "DONE").write_text("done\n")
        print("DONE", flush=True)
