#!/usr/bin/env python3
"""Why does the old PR-median method have fewer noisy false alarms than R-peak - 64 ms (43 vs 109, ungated, min 5)?

Part 1, swap test. The two configurations differ in three things:
    ST measure : 5-sample median (PR-median) vs 5-sample mean (R-peak - 64 ms)
    baseline   : PR-segment median vs 5-sample mean at R-peak - 64 ms
    beats      : those passing the PR gate vs every beat with a J point
Each is swapped on its own, from the stored per-beat files, through the real FSM (old_fsm_gs.run_fsm_on_record, no SQI) and
the paper scorer. CONTROL: the two unswapped rows must reproduce the stored alarm files exactly, or nothing else counts.
Part 2, offset test. The PR-median baseline sits above R-peak - 64 ms by a per-record offset; apply or remove that offset.
Part 3, occupancy. For each configuration: is the FSM already in an alarm when a noise burst starts, and do free bursts raise one?
Part 4, baseline window. R-peak - 64 ms with a median and with wider windows, from the noisy signal.

Output: c13_fp_gap/ (swap_results.csv, occupancy.csv, burst_st.csv, results.json, alarms_<config>.csv, DONE).
Usage : setsid nohup python3 -u c13_fp_gap.py > <out>/run.log 2>&1 < /dev/null &
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

TASK = Path("/home/student/GIT/Tasks/C_neurokit")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(TASK))
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
import c6_noisy_fold as C6  # noqa: E402  (also patches wfdb to read the local NSTDB em)
from c3_score_paper import score as paper_score, K, GS_PATH, DURATIONS  # noqa: E402
from c5_iso_offsets import ms, window_stat  # noqa: E402
from ltst_pipeline import old_fsm_gs as F  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact as contam  # noqa: E402

DS = Path("/home/student/Desktop/Research Data/DATASET NK")
CN = DS / "C_neurokit"
C5D, C6D = CN / "c5_iso", CN / "c6_noisy"
OUT = CN / "c13_fp_gap"
FS, SEG = 250, 2500
MIN_BEATS = 5
RECS = ["s20011", "s20051", "s20061", "s20071", "s20081", "s20091", "s20101", "s20111", "s20121", "s20131", "s20141", "s20201", "s20211", "s20221", "s20231", "s20241"]
TAG = {0: "ST0", 1: "ST1"}
PAPER = {0: "II", 1: "V2"}
OMIT = set(K.MANUAL_OMIT)
GS = K.load_gs(str(GS_PATH))
DUR = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}


def beat_file(root, cfg, rec, lead, cols):
    return pd.read_csv(next((root / "st_beats" / cfg / rec / f"lead_{lead}").glob("*.csv")), usecols=cols)


def grid(rec, lead):
    """The stored 10 s grid (segment_end_sample of each row)."""
    s = pd.read_csv(next((C6D / "runs_seg10s_min5/noisy_st5mean_rpk64/seg10s_median" / rec / f"lead_{lead}").glob("*.csv")), usecols=["segment_start_sample", "segment_end_sample"])
    return s.segment_start_sample.to_numpy().astype(int), s.segment_end_sample.to_numpy().astype(int)


def segments(j60, val, start):
    """Median of the usable beats in each grid segment, by J+60 sample; NaN below MIN_BEATS."""
    ok = np.isfinite(val) & np.isfinite(j60)
    idx = np.searchsorted(start, j60[ok].astype(np.int64), side="right") - 1
    out = np.full(len(start), np.nan)
    v = val[ok]
    order = np.argsort(idx, kind="stable")
    idx, v = idx[order], v[order]
    for k in np.unique(idx):
        x = v[np.searchsorted(idx, k, "left"): np.searchsorted(idx, k, "right")]
        if len(x) >= MIN_BEATS:
            out[k] = np.median(x)
    return out


def fsm(rec, lead, end, st):
    with contextlib.redirect_stdout(io.StringIO()):
        return F.run_fsm_on_record(rec_id=rec, lead_tag=TAG[lead], samp_arr=end.tolist(), st_arr=st.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=False)


def run_config(name, value_fn):
    """value_fn(rec, lead) -> per-beat ST deviation array. Returns the alarm table and the segment values."""
    rows, segs = [], {}
    for rec in RECS:
        for lead in (0, 1):
            start, end = grid(rec, lead)
            st = segments(BEATS[(rec, lead)]["j60"], value_fn(rec, lead), start)
            segs[(rec, lead)] = st
            for a in fsm(rec, lead, end, st):
                rows.append(dict(record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, onset_breach_s=a.onset_breach / FS, onset_actual_s=a.onset_actual / FS, end_actual_s=a.end_actual / FS, criterion=a.crit))
    A = pd.DataFrame(rows, columns=["record", "lead_tag", "lead_paper", "onset_breach_s", "onset_actual_s", "end_actual_s", "criterion"])
    p = OUT / f"alarms_{name}.csv"
    A.to_csv(p, index=False)
    return A, segs, p


def scored(p, name, **extra):
    out = []
    for r in paper_score(p, GS, DUR, OMIT):
        out.append(dict(config=name, mode=r["mode"], n_alarms=r["n_alarms"], TP=r["TP"], FP=r["FP"], FN=r["FN"], ep_F1=r["ep_F1"], dur_Se=r["dur_Se"], dur_PPV=r["dur_PPV"], IoU=r["IoU"], **extra))
    return out


def same_alarms(A, stored):
    k = ["record", "lead_paper", "onset_actual_s", "end_actual_s"]
    a = A.assign(lead_paper=A.lead_paper.astype(str))[k].sort_values(k).reset_index(drop=True)
    b = pd.read_csv(stored).assign(lead_paper=lambda d: d.lead_paper.astype(str))[k].sort_values(k).reset_index(drop=True)
    return len(a) == len(b) and (a.record == b.record).all() and (a.lead_paper == b.lead_paper).all() and np.allclose(a.onset_actual_s, b.onset_actual_s) and np.allclose(a.end_actual_s, b.end_actual_s), len(a), len(b)


def bursts():
    B = {}
    for r in RECS:
        h = wfdb.rdheader(str(DS / "ltst_records" / r / r))
        _, log = contam(np.zeros((h.sig_len, 2)), seed=42, fs=FS)
        B[r] = [(b["start_sample"] / FS, b["end_sample"] / FS) for b in log]
    return B


def main():
    global BEATS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DONE").unlink(missing_ok=True)
    R = {}
    BEATS = {}
    cols = ["r_peak", "J_plus_60ms", "st_value_mV", "iso_baseline_mV", "st_deviation_mV"]
    for rec in RECS:
        for lead in (0, 1):
            a = beat_file(C6D, "noisy_st5med_prmed", rec, lead, cols)
            b = beat_file(C6D, "noisy_st5mean_rpk64", rec, lead, cols)
            assert np.array_equal(a.r_peak.to_numpy(), b.r_peak.to_numpy()), (rec, lead)
            BEATS[(rec, lead)] = dict(j60=b.J_plus_60ms.to_numpy(dtype=float), st_med=a.st_value_mV.to_numpy(), iso_p=a.iso_baseline_mV.to_numpy(), dev_p=a.st_deviation_mV.to_numpy(), st_mean=b.st_value_mV.to_numpy(), iso_r=b.iso_baseline_mV.to_numpy(), dev_r=b.st_deviation_mV.to_numpy(), rp=b.r_peak.to_numpy())
        print(rec, "loaded", flush=True)
    rd = lambda x: np.round(x, 6)  # noqa: E731  (the rebuilt files round the ST deviation to 1e-6 mV)
    mask = lambda b: np.where(np.isfinite(b["iso_p"]), 1.0, np.nan)  # noqa: E731  (beats passing the PR gate)
    B_ = BEATS
    CONFIGS = [
        ("R64 (control)", lambda r, l: B_[(r, l)]["dev_r"]),
        ("PRmed (control)", lambda r, l: B_[(r, l)]["dev_p"]),
        ("R64, ST median", lambda r, l: rd(B_[(r, l)]["st_med"] - B_[(r, l)]["iso_r"])),
        ("R64, PR-gate beats", lambda r, l: B_[(r, l)]["dev_r"] * mask(B_[(r, l)])),
        ("R64, ST median, PR-gate beats", lambda r, l: rd(B_[(r, l)]["st_med"] - B_[(r, l)]["iso_r"]) * mask(B_[(r, l)])),
        ("PRmed, ST mean", lambda r, l: rd(B_[(r, l)]["st_mean"] - B_[(r, l)]["iso_p"])),
    ]
    rows, ALARMS, SEGS = [], {}, {}
    for name, fn in CONFIGS:
        A, s, p = run_config(name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), fn)
        ALARMS[name], SEGS[name] = A, s
        rows += scored(p, name)
        print(name, "done", len(A), "alarms", flush=True)
    ok1 = same_alarms(ALARMS["R64 (control)"], C6D / "runs_seg10s_min5/noisy_st5mean_rpk64/alarms.csv")
    ok2 = same_alarms(ALARMS["PRmed (control)"], C6D / "runs_seg10s_min5/noisy_st5med_prmed/alarms.csv")
    R["control_R64"], R["control_PRmed"] = [bool(ok1[0]), ok1[1], ok1[2]], [bool(ok2[0]), ok2[1], ok2[2]]
    print("CONTROL", R["control_R64"], R["control_PRmed"], flush=True)
    # ------------------------------------------------------------------ offset test
    # per record-lead offset between the two baselines, from the clean files so the noise does not set it
    off = {}
    for rec in RECS:
        for lead in (0, 1):
            cp = beat_file(C5D, "st3med_prmed", rec, lead, ["iso_baseline_mV"]).iso_baseline_mV.to_numpy()
            cr = beat_file(C5D, "st5mean_rpk64", rec, lead, ["iso_baseline_mV"]).iso_baseline_mV.to_numpy()
            d = cp - cr
            off[(rec, lead)] = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
    R["clean_baseline_offset_mV"] = {PAPER[l]: dict(median_over_records=float(np.median([off[(r, l)] for r in RECS])), min=float(min(off[(r, l)] for r in RECS)), max=float(max(off[(r, l)] for r in RECS))) for l in (0, 1)}
    for name, fn in [("R64 shifted down by the PR offset", lambda r, l: rd(B_[(r, l)]["dev_r"] - off[(r, l)])),
                     ("PRmed with the PR offset removed", lambda r, l: rd(B_[(r, l)]["dev_p"] + off[(r, l)]))]:
        A, s, p = run_config(name.replace(" ", "_").replace(",", ""), fn)
        ALARMS[name], SEGS[name] = A, s
        rows += scored(p, name)
        print(name, "done", len(A), "alarms", flush=True)
    # ------------------------------------------------------------------ baseline window: median and wider windows at R-peak - 64 ms
    win = {}
    for rec in RECS:
        _, noisy = C6.noisy_signal(rec)
        for lead in (0, 1):
            b = B_[(rec, lead)]
            x = np.ascontiguousarray(noisy[:, lead])
            c = b["rp"].astype(float) - ms(64)
            win[(rec, lead)] = {"5-sample median": window_stat(x, c, 2, np.median), "13-sample median": window_stat(x, c, 6, np.median), "13-sample mean": window_stat(x, c, 6, np.mean), "25-sample median": window_stat(x, c, 12, np.median)}
        print(rec, "baseline windows done", flush=True)
    for w in ("5-sample median", "13-sample median", "13-sample mean", "25-sample median"):
        name = f"R64, baseline {w}"
        A, s, p = run_config(name.replace(" ", "_").replace(",", ""), lambda r, l, w=w: rd(B_[(r, l)]["st_mean"] - win[(r, l)][w]))
        ALARMS[name], SEGS[name] = A, s
        rows += scored(p, name)
        print(name, "done", len(A), "alarms", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "swap_results.csv", index=False)
    # ------------------------------------------------------------------ occupancy and burst behaviour, per lead, masked scope
    BUR = bursts()
    occ, bst = [], []
    fp_of = {}
    for name in ("R64 (control)", "PRmed (control)", "R64 shifted down by the PR offset", "PRmed with the PR offset removed", "R64, PR-gate beats", "R64, ST median, PR-gate beats"):
        A = ALARMS[name].assign(lead=lambda d: d.lead_paper.astype(str))
        for lead, lp in PAPER.items():
            recs = [r for r in RECS if (r, lp) not in OMIT and r != "s20121"]
            n = started_inside = free = free_new = free_new_fp = 0
            time_in, time_all = 0.0, 0.0
            vals = []
            run_neg = run_pos2 = 0
            for rec in recs:
                a = A[(A.record == rec) & (A.lead == lp)]
                g = GS[(GS.record == rec) & (GS.lead == lp)]
                iv = list(zip(a.onset_actual_s, a.end_actual_s))
                time_in += sum(max(0.0, e - s) for s, e in iv)
                time_all += DUR[rec]
                st = SEGS[name][(rec, lead)]
                _, end = grid(rec, lead)
                t = end / FS
                for s_b, e_b in BUR[rec]:
                    n += 1
                    if any(s <= s_b <= e for s, e in iv):
                        started_inside += 1
                        continue
                    free += 1
                    # a new alarm whose warm-up began inside this burst
                    new = a[(a.onset_breach_s >= s_b) & (a.onset_breach_s <= e_b)]
                    if len(new):
                        free_new += 1
                        if any(len(g[(g.onset_actual_s < r.end_actual_s) & (g.end_actual_s > r.onset_actual_s)]) == 0 for r in new.itertuples()):
                            free_new_fp += 1
                    m = (t > s_b) & (t <= e_b)
                    v = st[m]
                    vals.append(v)
                    fin = v[np.isfinite(v)]

                    def longest(c):
                        best = cur = 0
                        for x in c:
                            cur = cur + 1 if x else 0
                            best = max(best, cur)
                        return best
                    run_neg += longest(fin <= -0.10) >= 6
                    run_pos2 += longest(fin >= 0.20) >= 6
            v = np.concatenate(vals) if vals else np.array([])
            v = v[np.isfinite(v)]
            occ.append(dict(config=name, lead=lp, bursts=n, started_inside_alarm=started_inside, free_bursts=free, free_with_new_alarm=free_new, free_with_new_FP=free_new_fp, alarm_time_pct=100 * time_in / time_all, free_run_ge60s_neg=int(run_neg), free_run_ge60s_pos020=int(run_pos2)))
            bst.append(dict(config=name, lead=lp, segments_in_free_bursts=len(v), pct_le_m010=float(100 * np.mean(v <= -0.10)), pct_ge_p010=float(100 * np.mean(v >= 0.10)), pct_ge_p020=float(100 * np.mean(v >= 0.20)), pct_abs_le_005=float(100 * np.mean(np.abs(v) <= 0.05)), median=float(np.median(v))))
        print(name, "occupancy done", flush=True)
    pd.DataFrame(occ).to_csv(OUT / "occupancy.csv", index=False)
    pd.DataFrame(bst).to_csv(OUT / "burst_st.csv", index=False)
    json.dump(R, open(OUT / "results.json", "w"), indent=1)
    (OUT / "DONE").write_text("done\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
