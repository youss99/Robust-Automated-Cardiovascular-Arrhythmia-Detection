"""Noisy / gated / fold-faithful evaluation of the fixed-offset ISO NeuroKit baseline.

Stages (run by c6_run.sh):
  build  : per-beat ST deviation (ST = 5-sample mean at J+60; ISO = 5-sample mean at R-peak - offset)
           noisy (delin_noise_nofilter_neurokit on the NSTDB-em-contaminated signal, seed 42):
             noisy_st5mean_rpk64, noisy_st5mean_rpkfold, noisy_st5med_prmed (stored noisy measurement; must
             reproduce ST features Data/st_noise_pr_only_nofilter exactly)
           clean: st5mean_rpkfold  (-> c5_iso/st_beats)
           rpkfold = offset from the DMY ISO median of the TRAINING records of the record's fold (64 or 68 ms)
  gated  : paper's SQI gate (gen_sqi_sweep.py): SQI on the contaminated signal after 0.67 Hz HP, 30 s window,
           5 s cadence, thresholds -20..+25 dB, FSM unchanged; on each noisy run's 10 s segments.
           R-peaks for the SQI: "nk"  = NeuroKit R-peaks detected on the noisy lead (deployable; primary)
                                "atr" = LTST .atr beats, as in the paper -- reproduction check only
                                        (noisy_st5med_prmed, min 5 must give the paper's union F1 0.677)
  score  : paper scorer (union / lead_II / lead_V2) for every gated alarm file

Usage: python3 c6_noisy_fold.py build | gated nk | gated atr | score
       FULL_SWEEP=1 python3 c6_noisy_fold.py gated nk     (all thresholds -20..+25 dB; default only 5 and 6 dB)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Tasks/C_neurokit")
from ltst_pipeline import old_fsm_gs as F  # noqa: E402
from ltst_pipeline.filters import highpass_filter  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact  # noqa: E402
from c5_iso_offsets import window_stat, ms  # noqa: E402

ROOT = Path("/home/student/Desktop/Research Data/DATASET NK")
DELIN_NOISY = ROOT / "Delineation Data" / "delin_noise_nofilter_neurokit"
DELIN_CLEAN = ROOT / "Delineation Data" / "delin_raw_nofilter_neurokit"
STORED_NOISY = ROOT / "ST features Data" / "st_noise_pr_only_nofilter"
C5 = ROOT / "C_neurokit" / "c5_iso"
OUT = ROOT / "C_neurokit" / "c6_noisy"
EM_LOCAL = "/home/student/Desktop/Research Data/DATASET/nstdb/mit-bih-noise-stress-test-database-1.0.0/em"
FS = 250
RECORDS = ["s20011", "s20051", "s20061", "s20071", "s20081", "s20091", "s20101", "s20111",
           "s20121", "s20131", "s20141", "s20201", "s20211", "s20221", "s20231", "s20241"]
FOLDS = {"fold01": ["s20051", "s20211", "s20221"], "fold02": ["s20081", "s20101", "s20141"],
         "fold03": ["s20061", "s20091", "s20241"], "fold04": ["s20111", "s20131", "s20231"],
         "fold05": ["s20011", "s20071", "s20201"]}
THRESHOLDS = list(range(-20, 26))   # full sweep: FULL_SWEEP=1 python3 c6_noisy_fold.py gated nk
QUICK_THRESHOLDS = [5, 6]           # default: 5 dB = paper's NK and learned gate, 6 dB = Test 1 evaluations
COLS = ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_point", "J_plus_60ms"]

_rdrecord = wfdb.rdrecord


def _rdrecord_local_em(name, *a, **k):
    """noise.py fetches NSTDB 'em' from PhysioNet; read the verified-identical local copy instead."""
    if name == "em" and k.get("pn_dir") == "nstdb":
        k.pop("pn_dir")
        return _rdrecord(EM_LOCAL, *a, **k)
    return _rdrecord(name, *a, **k)


wfdb.rdrecord = _rdrecord_local_em


def fold_offsets():
    """Per test record: median DMY ISO offset (ms) over the training records of its fold."""
    d = pd.read_csv(C5 / "c5_dmy_per_beat.csv")
    off = {}
    for test in FOLDS.values():
        med = float(d[d.record.isin(sum(FOLDS.values(), [])) & ~d.record.isin(test)].dmy_ISO_ms.median())
        off.update({r: med for r in test})
    off["s20121"] = float(d[d.record.isin(sum(FOLDS.values(), []))].dmy_ISO_ms.median())  # not in any fold; excluded at scoring
    return off


def noisy_signal(rec):
    clean = wfdb.rdrecord(str(ROOT / "ltst_records" / rec / rec)).p_signal
    noisy, _ = contaminate_ecg_with_motion_artifact(clean, fs=FS)
    return clean, noisy


def per_beat(x, df, iso_kind, off_ms=None):
    rp, p_off, r_on, j60 = (df[c].to_numpy(dtype=float) for c in ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_plus_60ms"])
    j_ok = ~np.isnan(j60) & (rp < j60)
    jc = np.where(j_ok, j60, np.nan)
    if iso_kind == "prmed":
        st = window_stat(x, jc, 2, np.median)
        iso = np.full(len(df), np.nan)
        ok = (~np.isnan(p_off) & ~np.isnan(r_on) & j_ok & (p_off >= 0) & (r_on < len(x)) & (p_off < r_on) & (r_on < rp))
        for i in np.where(ok)[0]:
            iso[i] = np.median(x[int(p_off[i]):int(r_on[i])])
    else:
        st = window_stat(x, jc, 2, np.mean)
        iso = window_stat(x, rp - ms(off_ms), 2, np.mean)
    return st, iso


def write(root, cfg, rec, lead, df, st, iso):
    out = df[["r_peak", "J_point", "J_plus_60ms"]].copy()
    out["st_value_mV"] = st
    out["iso_baseline_mV"] = iso
    out["st_deviation_mV"] = np.round(st - iso, 6)
    d = root / cfg / rec / f"lead_{lead}"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"waves_{rec}_lead{lead}_{cfg}.csv", index=False)


def build():
    off = fold_offsets()
    pd.Series(off, name="iso_offset_ms").to_csv(OUT / "fold_offsets.csv")
    for rec in RECORDS:
        clean, noisy = noisy_signal(rec)
        for lead in (0, 1):
            dn = pd.read_csv(DELIN_NOISY / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=COLS)
            xn = noisy[:, lead]
            st, iso = per_beat(xn, dn, "prmed")
            write(OUT / "st_beats", "noisy_st5med_prmed", rec, lead, dn, st, iso)
            ref = pd.read_csv(next((STORED_NOISY / rec / f"lead_{lead}").glob("*.csv")), usecols=["st_deviation_mV"]).st_deviation_mV.to_numpy()
            same = np.allclose(np.nan_to_num(ref, nan=9e9), np.nan_to_num(st - iso, nan=9e9), atol=1e-9)
            print(rec, lead, "noisy_st5med_prmed == stored noisy:", same, flush=True)
            for cfg, o in (("noisy_st5mean_rpk64", 64.0), ("noisy_st5mean_rpkfold", off[rec])):
                st, iso = per_beat(xn, dn, "rpk", o)
                write(OUT / "st_beats", cfg, rec, lead, dn, st, iso)
            dc = pd.read_csv(DELIN_CLEAN / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=COLS)
            st, iso = per_beat(clean[:, lead], dc, "rpk", off[rec])
            write(C5 / "st_beats", "st5mean_rpkfold", rec, lead, dc, st, iso)
    (OUT / "BUILT").touch()


def gated(source="nk"):
    thresholds = THRESHOLDS if os.environ.get("FULL_SWEEP") else QUICK_THRESHOLDS
    runs = sorted(p for p in OUT.glob("runs_seg10s_min*/*") if (p / "seg10s_median").is_dir())
    if source == "atr":
        runs = [r for r in runs if r.name == "noisy_st5med_prmed" and r.parent.name == "runs_seg10s_min5"]
    acc = {(r, t): [] for r in runs for t in thresholds}
    for rec in RECORDS:
        base = str(ROOT / "ltst_records" / rec / rec)
        atr_peaks = F.get_rpeaks_from_atr(wfdb.rdann(base, "atr"))
        _, noisy = noisy_signal(rec)
        sqi_sig = highpass_filter(noisy, fs=FS)
        for lead in (0, 1):
            tag = f"ST{lead}"
            ecg = sqi_sig[:, lead]
            if source == "atr":
                r_peaks = atr_peaks
            else:
                r_peaks = np.sort(pd.read_csv(DELIN_NOISY / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv",
                                              usecols=["r_peak"]).r_peak.to_numpy().astype(int))
            memo = {}
            for run in runs:
                seg = next((run / "seg10s_median" / rec / f"lead_{lead}").glob("*.csv"))
                df = pd.read_csv(seg)
                strategy = seg.stem.split(f"_lead{lead}_", 1)[1]
                samp = pd.to_numeric(df["segment_end_sample"], errors="coerce")
                st = pd.to_numeric(df["st_deviation_agg_mV"], errors="coerce")
                v = samp.notna()
                samp_arr, st_arr = samp[v].astype(int).tolist(), st[v].tolist()
                for s in samp_arr:
                    if s not in memo:
                        memo[s] = F.compute_sqi_last30s(r_peaks=r_peaks, sig_filt=ecg, current_r_peak=int(s),
                                                        pre_sec=0.35, post_sec=0.35, window_sec=30.0,
                                                        fs=float(FS), min_beats=3)
                orig = F.compute_sqi_last30s
                F.compute_sqi_last30s = lambda r_peaks, sig_filt, current_r_peak, pre_sec, post_sec, window_sec, fs, min_beats: memo[int(current_r_peak)]
                try:
                    for t in thresholds:
                        for a in F.run_fsm_on_record(rec_id=rec, lead_tag=tag, samp_arr=samp_arr, st_arr=st_arr,
                                                     ecg_sig_for_sqi=ecg, r_peaks=r_peaks, fs=FS, use_sqi=True,
                                                     sqi_threshold_db=float(t), sqi_every_s=5.0, sqi_window_s=30.0):
                            acc[(run, t)].append({
                                "record": a.record, "lead_tag": a.lead_tag, "lead_paper": a.lead_paper, "strategy": strategy,
                                "onset_breach_sample": a.onset_breach, "onset_expected_sample": a.onset_expected,
                                "onset_actual_sample": a.onset_actual, "end_expected_sample": a.end_expected,
                                "end_actual_sample": a.end_actual, "onset_breach_s": a.onset_breach / FS,
                                "onset_expected_s": a.onset_expected / FS, "onset_actual_s": a.onset_actual / FS,
                                "end_expected_s": a.end_expected / FS, "end_actual_s": a.end_actual / FS,
                                "duration_expected_s": (a.end_expected - a.onset_expected) / FS,
                                "duration_actual_s": (a.end_actual - a.onset_actual) / FS,
                                "warmup_duration_s": (a.onset_actual - a.onset_breach) / FS, "criterion": a.crit})
                finally:
                    F.compute_sqi_last30s = orig
        print(rec, "gated done", flush=True)
    for (run, t), rows in acc.items():
        d = run / f"gated_{source}"
        d.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(d / f"sqi{t}.csv", index=False)
    (OUT / f"GATED_{source}").touch()


def score():
    from c3_score_paper import score as paper_score, K, GS_PATH, DURATIONS
    gs = K.load_gs(str(GS_PATH))
    durations = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}
    mask = set(K.MANUAL_OMIT)
    rows = []
    for run in sorted(OUT.glob("runs_seg10s_min*/*")):
        m = int(run.parent.name.split("min")[1])
        if (run / "alarms.csv").exists():
            rows += [{"config": run.name, "min_beats": m, "condition": "noisy", "sqi_db": np.nan, **r}
                     for r in paper_score(run / "alarms.csv", gs, durations, mask)]
        for source in ("nk", "atr"):
            for t in THRESHOLDS:
                f = run / f"gated_{source}" / f"sqi{t}.csv"
                if f.exists():
                    rows += [{"config": run.name, "min_beats": m, "condition": f"noisy+SQI ({source} R-peaks)", "sqi_db": t, **r}
                             for r in paper_score(f, gs, durations, mask)]
    pd.DataFrame(rows).to_csv(OUT / "results_noisy_paper_scoring.csv", index=False)
    print("wrote", OUT / "results_noisy_paper_scoring.csv", len(rows))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"build": build, "gated": gated, "score": score}[sys.argv[1]](*sys.argv[2:])
