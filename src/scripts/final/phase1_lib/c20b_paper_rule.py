#!/usr/bin/env python3
"""The published paper's NeuroKit rule for the R-peak finalists, through the same pipeline as c20_finalists.py.

Paper rule ("PR median", C §6 ref_stored, §10 old method):
  ST       median at J+60: 3 samples on clean signal (st3med), 5 samples on noisy (st5med), as stored by the paper
  baseline median of the PR segment [P offset, R onset), only where P offset < R onset < R-peak < J+60
  10 s fixed-grid segments, median, at least 5 usable beats; paper FSM; gate 6 dB with each detector's own noisy R-peaks.
Detectors: neurokit (control), rodrigues2021, khamis2016, each with its own delineation (Delineation Data).

Controls, which stop the run if they fail:
  neurokit per-beat values must equal the stored rebuilt ones: clean c5_iso st3med_prmed, noisy c6_noisy noisy_st5med_prmed.
  neurokit clean and gated alarms must equal the stored c5_iso st3med_prmed/alarms.csv and c6_noisy noisy_st5med_prmed/gated_nk/sqi6.csv.
DMY: c4_dmy_vs_nk.py on the clean per-beat values, as in c20.
Usage: python3 c20b_paper_rule.py <workers>   (detached). Output: c20_finalists/paper_rule/ (per_record/, st_beats/, dmy_per_beat_<det>.csv, results_*.csv, controls.txt, DONE)."""
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(Path(__file__).parent))
import c20_finalists as B  # noqa: E402
from c5_iso_offsets import window_stat  # noqa: E402

OUT = B.OUT / "paper_rule"
PER = OUT / "per_record"
STB = OUT / "st_beats"
CN = B.CN


def paper_beats(x, df, half):
    """st{3,5}med_prmed, as c6_noisy_fold.per_beat 'prmed' but with the ST window half-width given (1 = 3 samples, 2 = 5)."""
    rp, p_off, r_on, j60 = (df[c].to_numpy(dtype=float) for c in ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_plus_60ms"])
    j_ok = ~np.isnan(j60) & (rp < j60)
    st = window_stat(x, np.where(j_ok, j60, np.nan), half, np.median)
    iso = np.full(len(df), np.nan)
    ok = ~np.isnan(p_off) & ~np.isnan(r_on) & j_ok & (p_off >= 0) & (r_on < len(x)) & (p_off < r_on) & (r_on < rp)
    for i in np.where(ok)[0]:
        iso[i] = np.median(x[int(p_off[i]):int(r_on[i])])
    return st, iso


def job(task):
    rec = task[0]
    target = PER / f"alarms_{rec}.csv"
    if target.exists():
        return rec
    clean, noisy = B.C6.noisy_signal(rec)
    hp = B.highpass_filter(noisy, fs=B.FS)
    rows, ctl = [], []

    def keep(alarms, det, cond):
        rows.extend(dict(detector=det, config="paper", condition=cond, record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, onset_actual_s=a.onset_actual / B.FS, end_actual_s=a.end_actual / B.FS, criterion=a.crit) for a in alarms)

    for lead in (0, 1):
        start, end = B.C13.grid(rec, lead)
        ecg = np.ascontiguousarray(hp[:, lead])
        for det in B.DETS:
            dc, dn = B.delin(det, "clean", rec, lead), B.delin(det, "noisy", rec, lead)
            gate_peaks = np.sort(dn.r_peak.to_numpy().astype(int))
            st_c, iso_c = paper_beats(clean[:, lead], dc, 1)
            st_n, iso_n = paper_beats(noisy[:, lead], dn, 2)
            B.C6.write(STB / det, "paper", rec, lead, dc, st_c, iso_c)
            if det == "neurokit":
                for cond, v, ref in (("clean", st_c - iso_c, CN / f"c5_iso/st_beats/st3med_prmed/{rec}/lead_{lead}/waves_{rec}_lead{lead}_st3med_prmed.csv"), ("noisy", st_n - iso_n, CN / f"c6_noisy/st_beats/noisy_st5med_prmed/{rec}/lead_{lead}/waves_{rec}_lead{lead}_noisy_st5med_prmed.csv")):
                    r = pd.read_csv(ref, usecols=["st_deviation_mV"]).st_deviation_mV.to_numpy()
                    ctl.append(dict(record=rec, lead=lead, condition=cond, same=bool(np.allclose(np.nan_to_num(r, nan=9e9), np.nan_to_num(np.round(v, 6), nan=9e9), atol=1e-9))))
            sc = B.C13.segments(dc.J_plus_60ms.to_numpy(float), np.round(st_c - iso_c, 6), start)
            sn = B.C13.segments(dn.J_plus_60ms.to_numpy(float), np.round(st_n - iso_n, 6), start)
            with contextlib.redirect_stdout(io.StringIO()):
                keep(B.F.run_fsm_on_record(rec_id=rec, lead_tag=B.C13.TAG[lead], samp_arr=end.tolist(), st_arr=sc.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=B.FS, use_sqi=False), det, "clean")
                keep(B.F.run_fsm_on_record(rec_id=rec, lead_tag=B.C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=B.FS, use_sqi=False), det, "noisy")
                keep(B.F.run_fsm_on_record(rec_id=rec, lead_tag=B.C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=ecg, r_peaks=gate_peaks, fs=B.FS, use_sqi=True, sqi_threshold_db=B.THR, sqi_every_s=5.0, sqi_window_s=30.0), det, "gated6")
    pd.DataFrame(ctl).to_csv(PER / f"control_{rec}.csv", index=False)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["detector", "config", "condition", "record", "lead_tag", "lead_paper", "onset_actual_s", "end_actual_s", "criterion"]).to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(rec, "done", flush=True)
    return rec


def main():
    for d in (OUT, PER, STB):
        d.mkdir(parents=True, exist_ok=True)
    failed = B.run_records(job, [(r,) for r in B.TEST + ["s20121"]], int(sys.argv[1]), need_gb=6.0)
    if failed:
        (OUT / "FAILED").write_text(f"failed records: {failed}\n")
        sys.exit(f"failed records: {failed}")
    # point c20's DMY and scoring code at this folder; the controls are this rule's own
    B.OUT, B.PER, B.STB, B.CFGS = OUT, PER, STB, ("paper",)
    for det in B.DETS:
        t = OUT / f"dmy_per_beat_{det}.csv"
        if not t.exists():
            B.C4.DELIN = B.DEL / f"delin_raw_nofilter_{det}"
            sys.argv = ["c4", "--st_root", str(STB / det), "--configs", "paper", "--out_csv", str(t)]
            with contextlib.redirect_stdout(io.StringIO()):
                B.C4.main()
    lines = []
    # DMY control: neurokit paper rule against c5's stored DMY values for st3med_prmed
    ref = pd.read_csv(CN / "c5_iso/c5_dmy_per_beat.csv", usecols=["record", "lead", "sample_16a", "st3med_prmed_median_mV"])
    d = pd.read_csv(OUT / "dmy_per_beat_neurokit.csv").merge(ref, on=["record", "lead", "sample_16a"])
    lines.append(f"DMY, neurokit paper rule against c5 st3med_prmed: {len(d)} DMY beats, identical: {np.allclose(np.nan_to_num(d.paper_median_mV, nan=9e9), np.nan_to_num(d.st3med_prmed_median_mV, nan=9e9), atol=1e-9)}")
    A = pd.concat([pd.read_csv(f, dtype={"lead_paper": str}) for f in sorted(PER.glob("alarms_s2*.csv"))], ignore_index=True)
    pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/1_ltst_paper_cmbec48/results/alarms/_code")
    from c3_score_paper import K
    mask = set(K.MANUAL_OMIT)
    cols = ["record", "lead_paper", "onset_actual_s", "end_actual_s"]
    for cond, stored in (("clean", CN / "c5_iso/runs_seg10s_min5/st3med_prmed/alarms.csv"), ("gated6", CN / "c6_noisy/runs_seg10s_min5/noisy_st5med_prmed/gated_nk/sqi6.csv")):
        a = A[(A.detector == "neurokit") & (A.condition == cond)]
        b = pd.read_csv(stored, dtype={"lead_paper": str})
        a, b = [x[x.record.isin(B.TEST) & [(r, l) not in mask for r, l in zip(x.record, x.lead_paper)]][cols].sort_values(cols).round(3).reset_index(drop=True) for x in (a, b)]
        lines.append(f"alarms, neurokit paper rule {cond} against stored: {len(a)} here, {len(b)} stored, identical: {a.equals(b)}")
    C = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("control_s2*.csv"))], ignore_index=True)
    lines.append(f"per-beat values, neurokit paper rule against stored: {int(C.same.sum())} of {len(C)} record-lead-condition files identical")
    # scoring and DMY tables
    from c3_score_paper import GS_PATH, DURATIONS
    from score_alarms import per_record_components, gs_scope
    gs = K.load_gs(str(GS_PATH))
    dur = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}
    pooled, per = [], []
    TMP = OUT / "tmp_alarms.csv"
    for (det, cond), g in A.groupby(["detector", "condition"]):
        g.drop(columns=["detector", "config", "condition"]).to_csv(TMP, index=False)
        raw = K.load_alarms(str(TMP))
        for mode, scope in {"union": "FULL", "lead_II": "own_II", "lead_V2": "own_V2"}.items():
            df = raw if mode == "union" else raw[raw["lead"] == mode.split("_")[1]]
            c = per_record_components(gs_scope(gs, scope), K.scope_filter(K.apply_mask(df.reset_index(drop=True), mask), scope), dur, scope, mask, K.RECORDS_15)
            tp, fp, fn = int(c.TP.sum()), int(c.FP.sum()), int(c.FN.sum())
            dd = c.tTP.sum() + c.tFP.sum() + c.tFN.sum()
            pooled.append(dict(detector=det, config="paper", condition=cond, mode=mode, TP=tp, FP=fp, FN=fn, ep_F1=2 * tp / (2 * tp + fp + fn), IoU=c.tTP.sum() / dd, false_hours=c.tFP.sum() / 3600, alarm_hours=c.pred_dur.sum() / 3600))
            for x in c.itertuples():
                q = x.tTP + x.tFP + x.tFN
                per.append(dict(detector=det, config="paper", condition=cond, mode=mode, record=x.record, TP=x.TP, FP=x.FP, FN=x.FN, IoU=x.tTP / q if q else np.nan, false_hours=x.tFP / 3600))
    TMP.unlink()
    pd.DataFrame(pooled).to_csv(OUT / "results_alarms.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "results_alarms_per_record.csv", index=False)
    MASKED = {"ML2": {"s20091", "s20121"}, "MV2": {"s20081", "s20121", "s20221", "s20241"}}
    rows = []
    for det in B.DETS:
        dm = pd.read_csv(OUT / f"dmy_per_beat_{det}.csv")
        for lead in ("ML2", "MV2"):
            for sub in ("all", "no masked"):
                q = dm[(dm.lead == lead) & dm.record.isin(B.TEST)]
                if sub != "all":
                    q = q[~q.record.isin(MASKED[lead])]
                x, y = q.paper_median_mV, q.dmy_ee_mV
                m = x.notna()
                e = (x - y)[m]
                rows.append(dict(detector=det, config="paper", lead=lead, subset=sub, dmy_beats=len(q), usable=int(m.sum()), bias=e.mean(), MAE=e.abs().mean(), RMSE=float(np.sqrt((e ** 2).mean())), share_err_over_0p1=100 * float((e.abs() > 0.1).mean()), r=float(np.corrcoef(x[m], y[m])[0, 1]), slope=float(np.polyfit(y[m], x[m], 1)[0])))
    pd.DataFrame(rows).to_csv(OUT / "results_dmy.csv", index=False)
    (OUT / "controls.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not all(x.endswith("True") for x in lines if "identical:" in x) or int(C.same.sum()) != len(C):
        (OUT / "CONTROL_FAILED").write_text("\n".join(lines) + "\n")
        sys.exit("a control failed")
    (OUT / "DONE").write_text("done\n")


if __name__ == "__main__":
    main()
