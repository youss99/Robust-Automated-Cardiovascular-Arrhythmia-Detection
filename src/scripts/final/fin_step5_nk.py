#!/usr/bin/env python3
"""Final runs, VM step 5 (docs/instructions_final_results.md, DECISIONS.md): the NeuroKit pipelines, and the SQI gate every pipeline uses.

NK-PR     NeuroKit paper rule on rodrigues2021 delineation: ST = median at J+60 (3 samples clean, 5 noisy) minus the PR-segment median
          (c20b_paper_rule.paper_beats, unchanged); 10 s segments = median of >= 5 usable beats by J+60 (c13_fp_gap.segments); anchor 2500k + 2500.
NK-FP     fixed point on rodrigues2021 R-peaks (phase 1's M2-R code): ST = 5-sample mean at R + b + 60 ms minus the 5-sample mean at R - a;
          segments by the ST point; same anchor. a, b per fold and lead from the DMY beats of the fold's 13 non-test records, DMY beats on
          masked pairs excluded (DECISIONS.md 2026-11-20), nearest clean rodrigues2021 R-peak within 100 ms of the .16a sample,
          a = median(R - (fiducial - ISO)), b = median((fiducial + JP) - R), fiducial = .16a sample - 16 ms (p1_main_noisy.m2r_fit's procedure).
NK-FP-nk  NK-FP on NeuroKit's own R-peaks (a, b refit the same way on NeuroKit's clean peaks). Clean and seed 42.
NK-PR-pub NK-PR exactly as published (M1): the stored published per-beat values (NeuroKit delineation), clean and seed 42.
Conditions: clean, noisy 42, 43, 44 (NSTDB em, contaminate_ecg_with_motion_artifact(seed)); R-peaks from the step-4 cache.
Gate table (used by step 7 for every pipeline): the paper's SQI (0.67 Hz high-pass of the noisy signal, 30 s look-back, 0.35 s / 0.35 s,
min 3 beats; old_fsm_gs.compute_sqi_last30s) from the cached rodrigues2021 noisy peaks, at both anchors (learned grid 2500k + 2499,
NeuroKit grid 2500k + 2500), seeds 42-44; plus the same SQI from .atr peaks at seed 42 (NK-PR-pub, ablation A4). Burst fraction per window.
Controls:
  fit   the fit code on phase 1's M2-R record set (the 11 training records, masked pairs in, NeuroKit peaks) reproduces m2r_offsets.csv.
  NK-FP the NK-FP code with NeuroKit peaks and m2r_offsets.csv reproduces the stored M2-R values window by window (clean, seed 42).
  NK-PR rodrigues2021 NK-PR alarms (clean, noisy, gated 6 dB with rodrigues2021 peaks) through this code equal c20b_paper_rule's stored lists.
Stages: fit | values <workers> <conds, e.g. clean,noisy42> | summarise.   Output: Tasks/final/runs/step5/.
Usage: setsid nohup <repo .venv>/bin/python -u -W ignore fin_step5_nk.py <stage> ... > logs/step5_<stage>.log 2>&1 < /dev/null &"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1_lib"))  # vendored phase-1 code (phase1_lib/MANIFEST.tsv)
import p1_eval as P  # noqa: E402
import c6_noisy_fold as C6  # noqa: E402  (local NSTDB em)
import c13_fp_gap as C13  # noqa: E402
import c20_finalists as B  # noqa: E402
import c20b_paper_rule as PR  # noqa: E402
from c14_detectors_noisy import run_records  # noqa: E402
import ltst_pipeline.old_fsm_gs as G  # noqa: E402
from ltst_pipeline.filters import highpass_filter  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact  # noqa: E402

K, T1 = P.K, P.T1
FS, WIN, THR = 250, 2500, 6.0
OUT = Path("/home/student/GIT/Tasks/final/runs/step5")
PARTS = OUT / "parts"
DS = Path("/home/student/Desktop/Research Data/DATASET NK")
DEL = DS / "Delineation Data"
CACHE = DS / "final" / "rpeaks_rodrigues2021"
CN = DS / "C_neurokit"
M1_NOISY = CN / "c6_noisy/st_beats/noisy_st5med_prmed/{rec}/lead_{lead}/waves_{rec}_lead{lead}_noisy_st5med_prmed.csv"
M1_CLEAN = CN / "c20_finalists/paper_rule/st_beats/neurokit/paper/{rec}/lead_{lead}/waves_{rec}_lead{lead}_paper.csv"
M2R_DIR = Path("/home/student/GIT/Tasks/phase1/runs/_scores/main_noisy")
SEED = {"clean": None, "noisy42": 42, "noisy43": 43, "noisy44": 44}
TAG = {"clean": "raw", "noisy42": "noise", "noisy43": "noise43", "noisy44": "noise44"}
LEADS = {0: "II", 1: "V2"}
RECS16 = sorted(B.TEST + ["s20121"])
RECS = list(K.RECORDS_15)
MASKED = {(r, LEADS[l]) for r in RECS16 for l in (0, 1) if (r, LEADS[l]) in K.MANUAL_OMIT}
MAN = {k: pd.read_csv(P.REPO / f"ptbxl_models_v2/cv5_optB_fold_manifests/fold{k:02d}.csv") for k in range(1, 6)}
NONTEST = {f"fold{k:02d}": sorted(m[m.split != "test"].patient_id) for k, m in MAN.items()}


def delin(det, cond, rec, lead, cols=B.COLS):
    return pd.read_csv(DEL / f"delin_{TAG[cond]}_nofilter_{det}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=cols)


def peaks(det, cond, rec, lead):
    if det == "rodrigues2021":
        return np.load(CACHE / cond / f"{rec}_lead{lead}.npy").astype(np.int64)
    return np.sort(delin(det, cond, rec, lead, ["r_peak"]).r_peak.to_numpy().astype(np.int64))


def log(*a):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")
    print(*a, flush=True)


# ------------------------------------------------------------------ fit
def dmy_ab(det):
    """One row per DMY beat with a clean R-peak of det within 100 ms: a_ms, b_ms (p1_main_noisy.m2r_fit's formulas)."""
    C4 = B.C4
    rows = []
    for rec in RECS16:
        a16 = C4.load_16a(rec)
        for lead in (0, 1):
            L = a16[(a16.lead_idx == lead) & a16.DMY].sort_values("sample")
            if L.empty:
                continue
            s16 = L["sample"].to_numpy()
            R = peaks(det, "clean", rec, lead)
            i = np.clip(np.searchsorted(R, s16), 1, len(R) - 1)
            near = np.where(np.abs(R[i] - s16) < np.abs(R[i - 1] - s16), R[i], R[i - 1])
            ok = np.abs(near - s16) <= C4.TOL
            fid = s16 - C4.FID_SHIFT
            a_ms = (near - (fid - L.ISO_ms.to_numpy() / 4.0)) * 4.0
            b_ms = ((fid + L.JP_ms.to_numpy() / 4.0) - near) * 4.0
            rows += [dict(detector=det, record=rec, lead=LEADS[lead], masked=(rec, LEADS[lead]) in MASKED, a_ms=x, b_ms=y) for x, y in zip(a_ms[ok], b_ms[ok])]
    return pd.DataFrame(rows)


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, ok = [], True
    D = pd.concat([dmy_ab("rodrigues2021"), dmy_ab("neurokit")], ignore_index=True)
    D.to_csv(OUT / "ab_per_dmy_beat.csv.gz", index=False)
    # control: phase 1's M2-R fit (NeuroKit peaks, the 11 training records, masked pairs in)
    ref = pd.read_csv(M2R_DIR / "m2r_offsets.csv")
    bad = []
    for r in ref.itertuples():
        train = r.train_records.split()
        q = D[(D.detector == "neurokit") & (D.lead == r.lead) & D.record.isin(train)]
        if not (np.isclose(q.a_ms.median(), r.a_ms) and np.isclose(q.b_ms.median(), r.b_ms) and len(q) == r.n_dmy_train):
            bad.append((r.fold, r.lead, q.a_ms.median(), r.a_ms, q.b_ms.median(), r.b_ms, len(q), r.n_dmy_train))
    ok &= not bad
    lines.append(f"{'PASS' if not bad else 'FAIL'}  fit control: phase 1's M2-R fit (NeuroKit, 11 training records, masked in) reproduces m2r_offsets.csv: 10 fold-leads, differing {bad}")
    rows = []
    for det in ("rodrigues2021", "neurokit"):
        for fold, recs in NONTEST.items():
            for lead in ("II", "V2"):
                q_all = D[(D.detector == det) & (D.lead == lead) & D.record.isin(recs)]
                q = q_all[~q_all.masked]
                rows.append(dict(detector=det, fold=fold, lead=lead, a_ms=float(q.a_ms.median()), b_ms=float(q.b_ms.median()), a_samples=C6.ms(float(q.a_ms.median())),
                                 b_samples=C6.ms(float(q.b_ms.median())), n_dmy=len(q), n_dmy_masked_excluded=len(q_all) - len(q), n_records=q.record.nunique(),
                                 records=" ".join(sorted(q.record.unique())), n_dmy_s20121=int((q.record == "s20121").sum()), n_dmy_s20121_before_exclusion=int((q_all.record == "s20121").sum())))
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "nkfp_ab.csv", index=False, lineterminator="\n")
    s21 = int(F.n_dmy_s20121.sum())
    ok &= s21 == 0
    lines.append(f"{'PASS' if s21 == 0 else 'FAIL'}  s20121 contributes no DMY beat to any NK-FP fit: {s21} after exclusion "
                 f"({int(F[F.detector == 'rodrigues2021'].n_dmy_s20121_before_exclusion.max())} per fold-lead before; s20121 is masked on both leads)")
    (OUT / "FIT.txt").write_text("\n".join(lines) + "\n\n" + F.drop(columns="records").to_string(index=False) + "\n")
    log("fit", *lines)
    if not ok:
        (OUT / "FAILED_fit").write_text("see FIT.txt\n")
        sys.exit("fit control failed")
    (OUT / "DONE_fit").write_text(time.strftime("%F %T") + "\n")


# ------------------------------------------------------------------ values (one process per record and condition set)
def fixed_point(xs, rp, a_s, b_s, start):
    point = rp + b_s + 15  # 60 ms = 15 samples
    iso = C6.window_stat(xs, rp - a_s, 2, np.mean)
    stv = C6.window_stat(xs, point, 2, np.mean)
    return C13.segments(point.astype(float), np.round(stv - iso, 6), start)


def job(task):
    key, rec, conds, n_grid = task
    todo = [c for c in conds if not (PARTS / f"{rec}_{c}.csv.gz").exists()]
    if not todo:
        return key
    import wfdb
    t0 = time.time()
    raw = wfdb.rdrecord(str(T1.RECORDS_DIR / rec / rec)).p_signal
    ab = pd.read_csv(OUT / "nkfp_ab.csv")
    ab_ctl = pd.read_csv(M2R_DIR / "m2r_offsets.csv")
    fold = B.FOLD_OF[rec]
    atr = None
    for cond in todo:
        seed = SEED[cond]
        x, logb = (raw, []) if seed is None else contaminate_ecg_with_motion_artifact(raw, seed=seed, fs=FS)
        hp = None if seed is None else highpass_filter(x, fs=FS)
        parts = []
        for lead in (0, 1):
            start, end = C13.grid(rec, lead)
            n = max(int(n_grid), len(start))
            k = np.arange(n)

            def pad(v):
                return np.concatenate([np.asarray(v, float), np.full(n - len(v), np.nan)])
            xs = x[:, lead]
            col = {}
            d = delin("rodrigues2021", cond, rec, lead)
            st, iso = PR.paper_beats(xs, d, 1 if seed is None else 2)
            col["NKPR"] = pad(C13.segments(d.J_plus_60ms.to_numpy(float), np.round(st - iso, 6), start))
            rp = peaks("rodrigues2021", cond, rec, lead)
            if not np.array_equal(np.unique(rp), np.unique(d.r_peak.to_numpy().astype(np.int64))):
                raise RuntimeError(f"{rec} {lead} {cond}: cached peaks differ from the delineation's")
            o = ab[(ab.detector == "rodrigues2021") & (ab.fold == fold) & (ab.lead == LEADS[lead])].iloc[0]
            col["NKFP"] = pad(fixed_point(xs, rp.astype(float), int(o.a_samples), int(o.b_samples), start))
            if cond in ("clean", "noisy42"):
                rn = peaks("neurokit", cond, rec, lead).astype(float)
                o = ab[(ab.detector == "neurokit") & (ab.fold == fold) & (ab.lead == LEADS[lead])].iloc[0]
                col["NKFPnk"] = pad(fixed_point(xs, rn, int(o.a_samples), int(o.b_samples), start))
                oc = ab_ctl[(ab_ctl.fold == fold) & (ab_ctl.lead == LEADS[lead])].iloc[0]
                col["M2Rctl"] = pad(fixed_point(xs, rn, int(oc.a_samples), int(oc.b_samples), start))
                w = pd.read_csv(str(M1_CLEAN if cond == "clean" else M1_NOISY).format(rec=rec, lead=lead), usecols=["J_plus_60ms", "st_deviation_mV"])
                col["NKPRpub"] = pad(C13.segments(w.J_plus_60ms.to_numpy(float), w.st_deviation_mV.to_numpy(float), start))
            ag, an = k * WIN + WIN - 1, k * WIN + WIN
            if seed is not None:
                ecg = np.ascontiguousarray(hp[:, lead])

                def sqi(a, pk):
                    return G.compute_sqi_last30s(r_peaks=pk, sig_filt=ecg, current_r_peak=int(a), pre_sec=0.35, post_sec=0.35, window_sec=30.0, fs=float(FS), min_beats=3)
                col["sqi_grid"] = [sqi(a, rp) for a in ag]
                col["sqi_nk"] = [sqi(a, rp) for a in an]
                if seed == 42:
                    if atr is None:
                        atr = np.asarray(G.get_rpeaks_from_atr(wfdb.rdann(str(T1.RECORDS_DIR / rec / rec), "atr")))
                    col["sqi_grid_atr"] = [sqi(a, atr) for a in ag]
                    col["sqi_nk_atr"] = [sqi(a, atr) for a in an]
                ov = np.zeros(n)
                for b in logb:
                    s, e = int(b["start_sample"]), int(b["end_sample"])
                    ov += np.clip(np.minimum(k * WIN + WIN, e) - np.maximum(k * WIN, s), 0, None)
                col["burst_frac"] = ov / WIN
            parts.append(pd.DataFrame(dict(record=rec, lead=LEADS[lead], k=k, anchor_grid=ag, anchor_nk=an, nk_window=k < len(start), **col)))
        tmp = PARTS / f".{rec}_{cond}.tmp.csv.gz"
        pd.concat(parts, ignore_index=True).to_csv(tmp, index=False)
        os.replace(tmp, PARTS / f"{rec}_{cond}.csv.gz")
        log("values", rec, cond, f"{time.time() - t0:.0f}s")
    return key


def values(workers, conds):
    PARTS.mkdir(parents=True, exist_ok=True)
    if not (OUT / "DONE_fit").exists():
        sys.exit("run the fit stage first")
    n_grid = pd.read_csv(P.grid_file("R1_published", "pretrained", "two_lead", "clean"), usecols=["record"]).record.value_counts()
    t0 = time.time()
    failed = run_records(job, [(f"{r}:{','.join(conds)}", r, conds, int(n_grid[r])) for r in RECS], workers, need_gb=3.0)
    log("values stage", ",".join(conds), f"{time.time() - t0:.0f}s wall", f"failed {failed}")
    if failed:
        (OUT / f"FAILED_values_{'_'.join(conds)}").write_text(f"{failed}\n")
        sys.exit(f"failed: {failed}")
    (OUT / f"DONE_values_{'_'.join(conds)}").write_text(f"{time.strftime('%F %T')} {time.time() - t0:.0f}s wall\n")


# ------------------------------------------------------------------ summarise: controls and combined tables
def fsm(rec, lead, samp, st, gated, memo=None):
    if gated:
        G.compute_sqi_last30s = lambda **kw: memo[int(kw["current_r_peak"])]  # the precomputed gate: same function, same arguments
    with contextlib.redirect_stdout(io.StringIO()):
        al = G.run_fsm_on_record(rec_id=rec, lead_tag=C13.TAG[lead], samp_arr=[int(s) for s in samp], st_arr=[float(v) for v in st],
                                 ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=gated,
                                 sqi_threshold_db=THR, sqi_every_s=5.0, sqi_window_s=30.0)
    return [(str(a.record), str(a.lead_tag), str(a.lead_paper), round(a.onset_actual / FS, 6), round(a.end_actual / FS, 6), str(a.crit)) for a in al]


def summarise():
    lines, ok = [], True

    def out(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"{'PASS' if good else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    V = {c: pd.concat([pd.read_csv(PARTS / f"{r}_{c}.csv.gz") for r in RECS], ignore_index=True) for c in SEED}
    # control: NK-FP code with NeuroKit peaks and phase 1's offsets reproduces the stored M2-R values
    for c, ref_col in (("clean", "M2R_clean"), ("noisy42", "M2R_noisy")):
        ref = pd.concat([pd.read_csv(M2R_DIR / "per_record" / f"m2r_{r}.csv.gz", usecols=["record", "lead", "k", ref_col]) for r in RECS], ignore_index=True)
        ref["lead"] = ref.lead.map(LEADS)
        m = V[c].merge(ref, on=["record", "lead", "k"])
        good = len(m) == len(ref) and np.allclose(np.nan_to_num(m.M2Rctl, nan=9e9), np.nan_to_num(m[ref_col], nan=9e9), atol=1e-9)
        out(f"NK-FP code with NeuroKit peaks + phase 1's offsets equals the stored M2-R values ({c})", good, f"{len(m)}/{len(ref)} windows")
    # control: NK-PR rodrigues2021 alarms vs c20b's stored lists (clean, noisy, gated6)
    stored = pd.concat([pd.read_csv(PR.PER / f"alarms_{r}.csv") for r in RECS], ignore_index=True).query("detector == 'rodrigues2021'")
    for cond, vc, gated in (("clean", "clean", False), ("noisy", "noisy42", False), ("gated6", "noisy42", True)):
        mine = []
        for (rec, L), g in V[vc][V[vc].nk_window].groupby(["record", "lead"]):
            g = g.sort_values("k")
            lead = 0 if L == "II" else 1
            mine += fsm(rec, lead, g.anchor_nk, g.NKPR, gated, dict(zip(g.anchor_nk.astype(int), g.sqi_nk)) if gated else None)
        s = stored[stored.condition == cond]
        ref = sorted(zip(s.record.astype(str), s.lead_tag.astype(str), s.lead_paper.astype(str), s.onset_actual_s.round(6), s.end_actual_s.round(6), s.criterion.astype(str)))
        out(f"NK-PR rodrigues2021 alarm list equals c20b_paper_rule's ({cond})", sorted(mine) == ref, f"{len(mine)} here, {len(ref)} stored")
    # combined tables and coverage
    cov = []
    for c, df in V.items():
        df.to_csv(OUT / f"nk_values_{c}.csv.gz", index=False)
        q = df[df.nk_window & ~np.array([(r, l) in MASKED for r, l in zip(df.record, df.lead)], bool)]
        for col in ("NKPR", "NKFP", "NKFPnk", "NKPRpub"):
            if col in q:
                for L, g in q.groupby("lead"):
                    cov.append(dict(condition=c, pipeline=col, lead=L, windows=len(g), with_value=int(g[col].notna().sum()), coverage_pct=100 * g[col].notna().mean()))
    pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False)
    (OUT / "CONTROLS.txt").write_text("\n".join(lines) + ("\nALL PASSED\n" if ok else "\nFAILED\n"))
    log("summarise", *lines)
    (OUT / ("DONE_step5" if ok else "FAILED_step5")).write_text(time.strftime("%F %T") + "\n")
    return ok


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "fit":
        fit()
    elif st == "values":
        values(int(sys.argv[2]), sys.argv[3].split(","))
    elif st == "summarise":
        sys.exit(0 if summarise() else 1)
