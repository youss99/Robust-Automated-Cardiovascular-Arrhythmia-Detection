#!/usr/bin/env python3
"""Which R-peak detector holds up in the noise bursts? Clean and noisy raw LTST signal, scored against .atr, then a no-delineation alarm estimate.

Detectors: nk.ecg_peaks(x, sampling_rate=250, method=m) on the raw lead, no cleaning, once per record, as the stored pipeline does.
           CONTROL: method "neurokit" must reproduce the stored R-peaks of delin_{raw,noise}_nofilter_neurokit.

Detection (stage score), per record-lead, inside and outside noise bursts:
  accurate  : within 12 ms of the nearest .atr beat, after removing that detector's median offset on the clean signal (record-lead)
  displaced : 12-148 ms off        spurious : no .atr beat within 148 ms        missed : .atr beat with no detected R-peak within 148 ms
  TP/FP/FN  : one-to-one match with .atr within 12 samples (the +-50 ms of exports_ltst_rpeaks)
  noise part: median of n(R + gap) - n(R - 64 ms), n = noisy - clean, gap = the record-lead's median J+60 - R-peak on the stored clean delineation

Alarm estimate (stage alarms), NOT the real pipeline: no delineation is re-run, so J+60 is a fixed point after the R-peak.
  ST = 5-sample mean at R + gap, baseline = 5-sample mean at R - 64 ms, 10 s segments, min 5 usable beats, real FSM, paper scorer.
  variant aligned: R is first shifted by (detector's median offset - neurokit's offset at its true position, within 20 ms of .atr), both from the clean signal, so all detectors
                   share the same two measuring points and only beat-to-beat R-peak errors differ. It uses .atr, so it is a diagnostic, not deployable.
  variant as_is  : R used as detected.
  Conditions: clean; noisy ungated; noisy gated 6 dB (paper's SQI gate, with that detector's own noisy R-peaks).
  CONTROL: neurokit in this estimate against the real pipeline's rows (R-peak - 64 ms: clean, noisy, gated 6 dB).

Output: c14_detectors/ (peaks/, per_record/, detection_by_record.csv, detection_pooled.csv, control.json, alarm_results.csv, alarms_*.csv)
Usage : python3 c14_detectors_noisy.py detect|score|alarms <workers> <method> [<method> ...]   (run detached through c14_run.sh)
"""
import contextlib
import io
import json
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

TASK = Path("/home/student/GIT/Tasks/C_neurokit")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(TASK))
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
import c6_noisy_fold as C6  # noqa: E402  (also patches wfdb to read the local NSTDB em)
import c13_fp_gap as C13  # noqa: E402
from c5_iso_offsets import ms, window_stat  # noqa: E402
from ltst_pipeline import old_fsm_gs as F  # noqa: E402
from ltst_pipeline.filters import highpass_filter  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact as contam  # noqa: E402

DS = C13.DS
CN = DS / "C_neurokit"
OUT = Path(os.environ.get("C14_OUT", CN / "c14_detectors"))  # c16_run.sh gives each extra detector its own folder; the defaults are the verified §10c run
PEAKS = Path(os.environ.get("C14_PEAKS", OUT / "peaks"))
START_WAIT = int(os.environ.get("C14_START_WAIT", 45))
PER = OUT / "per_record"
RECS, FS = C13.RECS, C13.FS
LN = {0: "ML2", 1: "MV2"}
PAPER = {0: "II", 1: "V2"}
OMIT = {("s20091", "II"), ("s20121", "II"), ("s20081", "V2"), ("s20121", "V2"), ("s20221", "V2"), ("s20241", "V2")}
ACC, TRUE, TOL = 3, 37, 12  # samples: 12 ms, 148 ms, the 48 ms that +-50 ms rounds to at 250 Hz


def pk(cond, m, rec, lead):
    return PEAKS / f"{cond}_{m}_{rec}_lead{lead}.npy"


def detect_record(a):
    import neurokit2 as nk
    rec, methods = a
    clean, noisy = C6.noisy_signal(rec)
    for cond, X in (("clean", clean), ("noisy", noisy)):
        for lead in (0, 1):
            x = np.ascontiguousarray(X[:, lead])
            for m in methods:
                p = pk(cond, m, rec, lead)
                if p.exists():
                    continue
                t0 = time.time()
                try:
                    # "<method>_cleaned": nk.ecg_clean first, for finding the R-peak only (the path behind Test 1-NK and E1); every measurement stays on the raw signal
                    cleaned = m.endswith("_cleaned")
                    _, info = nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=FS, method="neurokit") if cleaned else x, sampling_rate=FS, method=m[:-len("_cleaned")] if cleaned else m)
                    r = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
                except Exception as e:  # noqa: BLE001
                    print(f"FAILED {cond} {m} {rec} lead{lead}: {type(e).__name__}: {e}", flush=True)
                    continue
                tmp = str(p) + ".tmp.npy"
                np.save(tmp, r)
                os.replace(tmp, p)
                print(f"{cond} {m} {rec} lead{lead}: {len(r)} peaks, {time.time() - t0:.0f} s", flush=True)
    return rec


def nearest_off(det, ref):
    i = np.clip(np.searchsorted(ref, det), 1, len(ref) - 1)
    return det - np.where(np.abs(ref[i] - det) < np.abs(ref[i - 1] - det), ref[i], ref[i - 1])


def match(ref, det, tol):
    """One-to-one: each .atr beat takes the nearest unused R-peak within tol."""
    used, got = np.zeros(len(det), bool), np.zeros(len(ref), bool)
    j = 0
    for i, r in enumerate(ref):
        while j < len(det) and det[j] < r - tol:
            j += 1
        k, best, bd = j, -1, tol + 1
        while k < len(det) and det[k] <= r + tol:
            if not used[k] and abs(det[k] - r) < bd:
                best, bd = k, abs(det[k] - r)
            k += 1
        if best >= 0:
            used[best], got[i] = True, True
    return got, used


def merge_iv(iv):
    out = []
    for a, b in sorted((a, b) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def setup(rec):
    clean, noisy = C6.noisy_signal(rec)
    _, log = contam(np.zeros((clean.shape[0], 2)), seed=42, fs=FS)
    mask = np.zeros(clean.shape[0] + 1, bool)
    for b in log:
        mask[b["start_sample"]: b["end_sample"]] = True
    atr = np.sort(F.get_rpeaks_from_atr(wfdb.rdann(str(DS / "ltst_records" / rec / rec), "atr"))).astype(np.int64)
    return clean, noisy, mask, atr


def true_position(rec, lead, atr):
    """NeuroKit's offset from .atr and its J+60 distance, from the clean beats nearest the .atr beat (within 20 ms), so a flip record uses its true cluster and not its most common one.
    Falls back to every beat when fewer than 10 % of beats are that near, as on a lead whose R-peak sits consistently away from the .atr sample."""
    w = pd.read_csv(DS / "Delineation Data/delin_raw_nofilter_neurokit" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=["r_peak", "J_plus_60ms"]).dropna()
    r = w.r_peak.to_numpy().astype(np.int64)
    off = nearest_off(r, atr)
    near = np.abs(off) <= 5
    keep = near if near.mean() >= 0.10 else np.abs(off) <= 25
    return float(np.median(off[keep])), int(np.median((w.J_plus_60ms.to_numpy() - r)[keep]))


def gap_of(rec, lead, atr):
    return true_position(rec, lead, atr)[1]


def score_record(a):
    rec, methods = a
    clean, noisy, mask, atr = setup(rec)
    rows, ctl = [], []
    for lead in (0, 1):
        n = noisy[:, lead] - clean[:, lead]
        g = gap_of(rec, lead, atr)
        for m in methods:
            if not (pk("clean", m, rec, lead).exists() and pk("noisy", m, rec, lead).exists()):
                continue
            Pc = np.load(pk("clean", m, rec, lead))
            off_c = nearest_off(Pc, atr)
            med = float(np.median(off_c[np.abs(off_c) <= 25]))
            for cond in ("clean", "noisy"):
                P = np.load(pk(cond, m, rec, lead))
                off = nearest_off(P, atr)
                cls = np.where(np.abs(off) > TRUE, "spurious", np.where(np.abs(off - med) <= ACC, "accurate", "displaced"))
                got_tol, used_tol = match(atr, P, TOL)
                near = np.abs(nearest_off(atr, P)) <= TRUE
                noise = window_stat(n, P.astype(float) + g, 2, np.mean) - window_stat(n, P.astype(float) - ms(64), 2, np.mean) if cond == "noisy" else np.full(len(P), np.nan)
                for region, pm, am in (("burst", mask[P], mask[atr]), ("outside", ~mask[P], ~mask[atr])):
                    r = dict(record=rec, lead=LN[lead], method=m, condition=cond, region=region, in_scope=rec != "s20121" and (rec, PAPER[lead]) not in OMIT, median_offset=med, atr_beats=int(am.sum()), detected=int(pm.sum()),
                             accurate=int((cls[pm] == "accurate").sum()), displaced=int((cls[pm] == "displaced").sum()), spurious=int((cls[pm] == "spurious").sum()), within20ms_atr=int((np.abs(off[pm]) <= 5).sum()), missed_148ms=int((~near[am]).sum()), TP=int(got_tol[am].sum()), FN=int((~got_tol[am]).sum()), FP=int((~used_tol[pm]).sum()))
                    if cond == "noisy":
                        for c in ("accurate", "displaced", "spurious"):
                            v = noise[pm][cls[pm] == c]
                            v = v[np.isfinite(v)]
                            r[f"noise_sum_{c}"], r[f"noise_n_{c}"] = float(v.sum()), len(v)
                        v = noise[pm]
                        r["noise_median_all"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                    rows.append(r)
            if m == "neurokit":
                for cond, d in (("clean", "delin_raw_nofilter_neurokit"), ("noisy", "delin_noise_nofilter_neurokit")):
                    st = pd.read_csv(DS / "Delineation Data" / d / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=["r_peak"]).r_peak.to_numpy().astype(np.int64)
                    P = np.load(pk(cond, m, rec, lead))
                    ctl.append(dict(record=rec, lead=LN[lead], condition=cond, stored=len(st), here=len(P), identical=bool(np.array_equal(np.sort(st), np.sort(P)))))
    json.dump(dict(rows=rows, control=ctl), open(PER / f"score_{rec}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))


def values(x, R, g):
    jc = R + g
    return jc, np.round(window_stat(x, jc, 2, np.mean) - window_stat(x, R - ms(64), 2, np.mean), 6)


def alarm_rows(rec, lead, seg_end, st, method, variant, cond, ecg=None, r_peaks=None):
    with contextlib.redirect_stdout(io.StringIO()):
        if ecg is None:
            out = F.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=seg_end.tolist(), st_arr=st.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=False)
        else:
            out = F.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=seg_end.tolist(), st_arr=st.tolist(), ecg_sig_for_sqi=ecg, r_peaks=r_peaks, fs=FS, use_sqi=True, sqi_threshold_db=6.0, sqi_every_s=5.0, sqi_window_s=30.0)
    return [dict(method=method, variant=variant, condition=cond, record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, onset_actual_s=a.onset_actual / FS, end_actual_s=a.end_actual / FS, criterion=a.crit) for a in out]


def alarm_record(a):
    rec, methods = a
    todo = [m for m in methods if not (PER / f"alarms_{m}_{rec}.csv").exists()]
    if not todo:
        return rec
    clean, noisy, _, atr = setup(rec)
    hp = highpass_filter(noisy, fs=FS)
    for m in todo:
        rows = []
        for lead in (0, 1):
            if not (pk("clean", m, rec, lead).exists() and pk("noisy", m, rec, lead).exists()):
                continue
            target, g = true_position(rec, lead, atr)
            start, end = C13.grid(rec, lead)
            Pc, Pn = np.load(pk("clean", m, rec, lead)).astype(float), np.load(pk("noisy", m, rec, lead)).astype(float)
            off_m = nearest_off(Pc.astype(np.int64), atr)
            delta = float(np.median(off_m[np.abs(off_m) <= 25]) - target)
            ecg = np.ascontiguousarray(hp[:, lead])
            rp = np.sort(Pn.astype(int))
            memo = {}
            for s in end:
                memo[int(s)] = F.compute_sqi_last30s(r_peaks=rp, sig_filt=ecg, current_r_peak=int(s), pre_sec=0.35, post_sec=0.35, window_sec=30.0, fs=float(FS), min_beats=3)
            orig = F.compute_sqi_last30s
            F.compute_sqi_last30s = lambda r_peaks, sig_filt, current_r_peak, pre_sec, post_sec, window_sec, fs, min_beats: memo[int(current_r_peak)]
            try:
                for variant, d in (("aligned", delta), ("as_is", 0.0)):
                    jc, dev = values(np.ascontiguousarray(clean[:, lead]), Pc - d, g)
                    rows += alarm_rows(rec, lead, end, C13.segments(jc, dev, start), m, variant, "clean")
                    jn, dvn = values(np.ascontiguousarray(noisy[:, lead]), Pn - d, g)
                    sn = C13.segments(jn, dvn, start)
                    rows += alarm_rows(rec, lead, end, sn, m, variant, "noisy")
                    rows += alarm_rows(rec, lead, end, sn, m, variant, "gated6", ecg=ecg, r_peaks=rp)
            finally:
                F.compute_sqi_last30s = orig
        pd.DataFrame(rows).to_csv(PER / f"alarms_{m}_{rec}.csv", index=False)
        print(rec, m, "alarms done", flush=True)
    return rec


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def run_records(fn, tasks, workers, need_gb=7.0, retries=3):
    """One process per record. A new one starts only when enough memory is free (the machine has no swap); one that dies is retried, so a killed worker cannot hang the run."""
    pending, running, tries, failed = list(tasks), {}, {t[0]: 0 for t in tasks}, []
    while pending or running:
        for rec, (p, t) in list(running.items()):
            if not p.is_alive():
                p.join()
                del running[rec]
                if p.exitcode != 0:
                    tries[rec] += 1
                    print(f"{rec}: worker exited with {p.exitcode}, try {tries[rec]} of {retries}", flush=True)
                    (pending.append(t) if tries[rec] <= retries else failed.append(rec))
                else:
                    print(rec, "done", flush=True)
        started = False
        if pending and len(running) < workers and (avail_gb() >= need_gb or not running):
            t = pending.pop(0)
            p = Process(target=fn, args=(t,))
            p.start()
            running[t[0]] = (p, t)
            started = True
        time.sleep(START_WAIT if started else 5)  # let a new worker reach its working size before judging memory again
    if failed:
        print("FAILED records:", failed, flush=True)
    return failed


def main():
    stage, workers, methods = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    for d in (OUT, PEAKS, PER):
        d.mkdir(parents=True, exist_ok=True)
    tasks = [(r, methods) for r in RECS]
    if stage == "detect":
        run_records(detect_record, tasks, workers)
    elif stage == "score":
        run_records(score_record, tasks, workers)
        res = [json.load(open(PER / f"score_{r}.json")) for r in RECS if (PER / f"score_{r}.json").exists()]
        D = pd.DataFrame([x for r in res for x in r["rows"] if x["method"] in methods])
        D.to_csv(OUT / "detection_by_record.csv", index=False)
        ctl = pd.DataFrame([x for r in res for x in r["control"]])
        if len(ctl):  # only a run that includes neurokit has this control
            json.dump(dict(neurokit_identical=int(ctl.identical.sum()), of=len(ctl), records=len(res), mismatches=ctl[~ctl.identical].to_dict("records")), open(OUT / "control.json", "w"), indent=1)
            print("CONTROL neurokit vs stored R-peaks:", int(ctl.identical.sum()), "of", len(ctl), "identical", flush=True)
        for scope, X in (("in_scope", D[D.in_scope]), ("all", D)):
            g = X.groupby(["lead", "method", "condition", "region"]).sum(numeric_only=True).reset_index()
            g["pct_accurate"], g["pct_displaced"], g["pct_spurious"] = 100 * g.accurate / g.detected, 100 * g.displaced / g.detected, 100 * g.spurious / g.detected
            g["pct_missed"] = 100 * g.missed_148ms / g.atr_beats
            g["precision"], g["recall"] = g.TP / (g.TP + g.FP), g.TP / (g.TP + g.FN)
            g["F1"] = 2 * g.TP / (2 * g.TP + g.FP + g.FN)
            for c in ("accurate", "displaced", "spurious"):
                if f"noise_sum_{c}" in g:
                    g[f"mean_noise_{c}"] = g[f"noise_sum_{c}"] / g[f"noise_n_{c}"]
            g.to_csv(OUT / f"detection_pooled_{scope}.csv", index=False)
            if scope == "in_scope":
                pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
                print(g[["lead", "method", "condition", "region", "atr_beats", "detected", "TP", "FP", "FN", "F1", "pct_accurate", "pct_displaced", "pct_spurious", "pct_missed", "mean_noise_accurate", "mean_noise_displaced", "mean_noise_spurious"]].round(3).to_string(index=False), flush=True)
    elif stage == "alarms":
        run_records(alarm_record, tasks, workers)
        from c3_score_paper import score as paper_score, K, GS_PATH, DURATIONS
        gs = K.load_gs(str(GS_PATH))
        dur = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}
        A = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("alarms_*_s2*.csv")) if f.stat().st_size > 5], ignore_index=True)
        out = []
        for (m, v, c), g in A.groupby(["method", "variant", "condition"]):
            p = OUT / f"alarms_{m}_{v}_{c}.csv"
            g.to_csv(p, index=False)
            out += [dict(method=m, variant=v, condition=c, mode=r["mode"], n_alarms=r["n_alarms"], TP=r["TP"], FP=r["FP"], FN=r["FN"], ep_F1=r["ep_F1"], dur_Se=r["dur_Se"], dur_PPV=r["dur_PPV"], IoU=r["IoU"]) for r in paper_score(p, gs, dur, set(K.MANUAL_OMIT))]
        R = pd.DataFrame(out)
        def false_hours(p, leads):
            a = pd.read_csv(p).assign(lead=lambda d: d.lead_paper.astype(str))
            tot = 0.0
            for rec in RECS:
                for l in leads:
                    if rec == "s20121" or (rec, l) in OMIT:
                        continue
                    x, g_ = a[(a.record == rec) & (a.lead == l)], gs[(gs.record == rec) & (gs.lead == l)]
                    ux, ug = merge_iv(zip(x.onset_actual_s, x.end_actual_s)), merge_iv(zip(g_.onset_actual_s, g_.end_actual_s))
                    tot += sum(b - a_ for a_, b in ux) - sum(max(0.0, min(b, d) - max(a_, c)) for a_, b in ux for c, d in ug)
            return tot / 3600
        LEADS = {"union": ("II", "V2"), "lead_II": ("II",), "lead_V2": ("V2",)}
        R["false_hours"] = [false_hours(OUT / f"alarms_{r.method}_{r.variant}_{r.condition}.csv", LEADS[r.mode]) for r in R.itertuples()]
        R.to_csv(OUT / "alarm_results.csv", index=False)
        pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300)
        print(R.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
