#!/usr/bin/env python3
"""Final runs, VM step 4 checks (docs/instructions_final_results.md, step 4), after fin_step4_seed44.py.

A  R-peak cache: rodrigues2021 peaks per record, lead and signal (clean, noisy 42, 43, 44), taken from the delineation files' r_peak column
   (which the delineation scripts check equals their input peaks). Written once; every pipeline reads it.
B  Same setup for the reused clean / seed 42 / seed 43 outputs: rodrigues2021 re-detected here (nk.ecg_peaks on the raw lead, the clean
   p_signal or contaminate_ecg_with_motion_artifact(seed)) must equal the stored peaks in all 32 record-leads of every condition.
C  Spot-check against .atr, per condition and lead: TP/FP/FN one-to-one within 12 samples (the +-50 ms of the screening), % spurious
   (no .atr beat within 148 ms), % missed; whole record and inside noise bursts.
D  The freeze rule within each fold (c21b_step4_report.choose, unchanged: clean DMY MAE, p100, worst record-lead MAE per lead, margin
   0.005 mV / 1 point; tie-break = noise spurious rate, seeds 42 and 43), on that fold's 13 non-test records, offsets fitted on those records
   as the published per-fold check fits them. Control first: the same code on the training records only must reproduce the published
   per-fold lines (c21_step4/report/rule.txt) and the stored clean_trainfold DMY values.
Output: DATASET NK/final/rpeaks_rodrigues2021/<cond>/<rec>_lead<k>.npy + MANIFEST.csv; Tasks/final/runs/step4_checks/ (tables, CHECKS.txt,
DONE_step4_checks or FAILED_step4_checks).
Usage: setsid nohup python3 -u -W ignore fin_step4_checks.py <workers> > logs/step4_checks.log 2>&1 < /dev/null &"""
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402

sys.path.insert(0, "/home/student/GIT/Tasks/C_neurokit")
import c21b_step4_report as R  # noqa: E402
import neurokit2 as nk  # noqa: E402

B, C14, contam = R.B, R.C14, R.contam
DS = B.DS
DEL = DS / "Delineation Data"
CACHE = DS / "final" / "rpeaks_rodrigues2021"
OUT = Path("/home/student/GIT/Tasks/final/runs/step4_checks")
PER = OUT / "per_record"
DET, FS = "rodrigues2021", 250
CONDS = {"clean": None, "noisy42": 42, "noisy43": 43, "noisy44": 44}
TAG = {"clean": "raw", "noisy42": "noise", "noisy43": "noise43", "noisy44": "noise44"}
RECS16 = R.RECS16
MAN = {k: pd.read_csv(R.REPO / f"ptbxl_models_v2/cv5_optB_fold_manifests/fold{k:02d}.csv") for k in range(1, 6)}
NONTEST = {k: sorted(m[m.split != "test"].patient_id) for k, m in MAN.items()}
TRAIN = R.TRAIN
LN = {0: "II", 1: "V2"}


def delin(cond, rec, lead, det=DET):
    return DEL / f"delin_{TAG[cond]}_nofilter_{det}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv"


def offsets(records):
    O = pd.read_csv(B.OUT / "offsets_per_dmy_beat.csv")
    return {(k, d, l): float(O[(O.detector == d) & (O.lead == l) & O.record.isin(records[k])].iso_before_r_ms.median()) for k in records for d in R.DETS for l in (0, 1)}


def job(task):
    rec = task[0]
    target = PER / f"{rec}.done"
    if target.exists():
        return rec
    raw = wfdb.rdrecord(str(DS / "ltst_records" / rec / rec)).p_signal
    atr = np.sort(B.F.get_rpeaks_from_atr(wfdb.rdann(str(DS / "ltst_records" / rec / rec), "atr"))).astype(np.int64)
    n = raw.shape[0]
    same, spot, dmy = [], [], []
    for cond, seed in CONDS.items():
        x = raw if seed is None else contam(raw, seed=seed, fs=FS)[0]
        burst = R.burst_mask(n, seed) if seed is not None else None
        for lead in (0, 1):
            stored = np.sort(pd.read_csv(delin(cond, rec, lead), usecols=["r_peak"]).r_peak.to_numpy().astype(np.int64))
            _, info = nk.ecg_peaks(np.ascontiguousarray(x[:, lead]).astype(float), sampling_rate=FS, method=DET)
            here = np.unique(np.asarray(info["ECG_R_Peaks"], dtype=np.int64))
            same.append(dict(record=rec, lead=LN[lead], condition=cond, n_stored=len(stored), n_redetected=len(here), identical=bool(np.array_equal(stored, here))))
            p = CACHE / cond / f"{rec}_lead{lead}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(p) + f".tmp{os.getpid()}.npy"
            np.save(tmp, stored)
            os.replace(tmp, p)
            off = C14.nearest_off(stored, atr)
            aoff = C14.nearest_off(atr, stored)
            for scope, dm, am in (("record", np.ones(len(stored), bool), np.ones(len(atr), bool)),) + ((("burst", burst[stored], burst[atr]),) if burst is not None else ()):
                spot.append(dict(record=rec, lead=LN[lead], condition=cond, scope=scope, n_atr=int(am.sum()), n_detected=int(dm.sum()),
                                 TP=int((np.abs(aoff[am]) <= C14.TOL).sum()), FN=int((np.abs(aoff[am]) > C14.TOL).sum()), FP=int((np.abs(off[dm]) > C14.TOL).sum()),
                                 spurious=int((np.abs(off[dm]) > C14.TRUE).sum()), missed=int((np.abs(aoff[am]) > C14.TRUE).sum())))
    # D: DMY values on clean, offsets fitted per fold on the training records (control) and on the non-test records
    a16 = B.C4.load_16a(rec)
    for name, recs, FO in (("train", TRAIN, offsets(TRAIN)), ("nontest", NONTEST, offsets(NONTEST))):
        for lead in (0, 1):
            L = a16[a16.lead_idx == lead].sort_values("sample").reset_index(drop=True)
            if not L.DMY.any():
                continue
            for det in R.DETS:
                w = pd.read_csv(delin("clean", rec, lead, det), usecols=B.COLS)
                for k in recs:
                    if rec in recs[k]:
                        dmy += [dict(record=rec, lead=LN[lead], detector=det, subset=name, fold=k, offset_ms=FO[(k, det, lead)], sample=s, ee=e, value=v)
                                for s, e, v in R.dmy_values(raw[:, lead], w, FO[(k, det, lead)], L)]
    pd.DataFrame(same).to_csv(PER / f"same_{rec}.csv", index=False)
    pd.DataFrame(spot).to_csv(PER / f"spot_{rec}.csv", index=False)
    pd.DataFrame(dmy).to_csv(PER / f"dmy_{rec}.csv", index=False)
    target.write_text("done\n")
    print(rec, "done", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarise():
    lines, ok = [], True

    def out(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"{'PASS' if good else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    S = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("same_s2*.csv"))], ignore_index=True)
    S.to_csv(OUT / "same_setup.csv", index=False)
    for cond in CONDS:
        q = S[S.condition == cond]
        out(f"B {cond}: re-detected rodrigues2021 peaks equal the stored ones", len(q) == 32 and q.identical.all(), f"{int(q.identical.sum())}/{len(q)} record-leads")
    man = pd.DataFrame([dict(condition=c, record=r, lead=k, n_peaks=len(np.load(CACHE / c / f"{r}_lead{k}.npy")), sha256=sha(CACHE / c / f"{r}_lead{k}.npy"),
                             source=str(delin(c, r, k))) for c in CONDS for r in RECS16 for k in (0, 1)])
    man.to_csv(CACHE / "MANIFEST.csv", index=False, lineterminator="\n")
    out("A cache complete", len(man) == 128, f"{len(man)} files (4 conditions x 16 records x 2 leads) in {CACHE}")
    P = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("spot_s2*.csv"))], ignore_index=True)
    P.to_csv(OUT / "atr_spotcheck_per_record.csv", index=False)
    T = P.groupby(["condition", "lead", "scope"])[["n_atr", "n_detected", "TP", "FP", "FN", "spurious", "missed"]].sum().reset_index()
    T["sensitivity"], T["PPV"] = T.TP / (T.TP + T.FN), (T.n_detected - T.FP) / T.n_detected
    T["F1"] = 2 * T.sensitivity * T.PPV / (T.sensitivity + T.PPV)
    T["pct_spurious"], T["pct_missed"] = 100 * T.spurious / T.n_detected, 100 * T.missed / T.n_atr
    T.to_csv(OUT / "atr_spotcheck_summary.csv", index=False)
    # D
    D = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("dmy_s2*.csv"))], ignore_index=True)
    D["err"] = D.value - D.ee
    D.to_csv(OUT / "dmy_fold_values.csv.gz", index=False)
    ref = pd.concat([pd.read_csv(f) for f in sorted(R.PER.glob("dmy_s2*.csv"))], ignore_index=True).query("condition == 'clean_trainfold'")
    m = D[D.subset == "train"].merge(ref, on=["record", "lead", "detector", "fold", "sample"], suffixes=("", "_ref"))
    out("D control: clean DMY values with training-record fold offsets equal the stored clean_trainfold values",
        len(m) == len(ref) == len(D[D.subset == "train"]) and np.allclose(np.nan_to_num(m.value, nan=9e9), np.nan_to_num(m.value_ref, nan=9e9), atol=2e-6), f"{len(m)}/{len(ref)} beat-values")
    SP = pd.read_csv(R.OUT / "spurious_per_record.csv")

    def stats(q):
        u = q.dropna(subset=["value"])
        return dict(dmy_beats=len(q), usable=len(u), MAE=float(u.err.abs().mean()), p100=float(100 * (u.err.abs() > 0.1).mean()))

    def fold_rule(subset, recs):
        res, tabs = [], []
        for k in recs:
            F_ = D[(D.subset == subset) & (D.fold == k)]
            tab = {}
            for det in R.DETS:
                for lead in ("II", "V2"):
                    q2 = F_[(F_.detector == det) & (F_.lead == lead) & ~F_.record.isin(R.MASKED[lead])]
                    per = {r: stats(g) for r, g in q2.groupby("record")}
                    wr = max(per, key=lambda r: per[r]["MAE"])
                    tab[(det, lead)] = {**stats(q2), "worst_record_MAE": per[wr]["MAE"]}
                    tabs.append(dict(subset=subset, fold=k, detector=det, lead=lead, n_records=len(per), **tab[(det, lead)], worst_record=wr))
            tr = SP[SP.record.isin(recs[k]) & SP.in_scope]
            sp = {d: 100 * tr[tr.detector == d].spurious_in_burst.sum() / tr[tr.detector == d].detected_in_burst.sum() for d in R.DETS}
            label = f"Fold {k}, training records only, no masked" if subset == "train" else f"Fold {k}, non-test records ({len(recs[k])}), no masked"
            res.append(choose(tab, sp, label)[0])
        return res, tabs

    choose = _choose()
    res_tr, _ = fold_rule("train", TRAIN)
    pub = [l for l in (R.OUT / "rule.txt").read_text().splitlines() if l.startswith("Fold ")]
    out("D control: the rule on the training records reproduces the published per-fold lines", res_tr == pub, "" if res_tr == pub else f"here {res_tr} vs published {pub}")
    res_nt, tabs = fold_rule("nontest", NONTEST)
    pd.DataFrame(tabs).to_csv(OUT / "fold_rule_table.csv", index=False)
    (OUT / "fold_rule.txt").write_text("\n".join(res_nt) + "\n")
    wins = all(r.endswith("choice rodrigues2021") or "picks rodrigues2021" in r for r in res_nt)
    lines.append(("RESULT  rodrigues2021 is chosen in every fold on the non-test records" if wins else "RESULT  rodrigues2021 is NOT chosen in every fold on the non-test records (reported, nothing changed: plan frozen)"))
    lines += ["        " + r for r in res_nt]
    return ok, lines, T


def _choose():
    """c21b_step4_report.summarise's inner choose(), unchanged (it is a closure there, so it is copied here verbatim)."""
    DETS, MARGIN_MAE, MARGIN_P100 = R.DETS, R.MARGIN_MAE, R.MARGIN_P100

    def choose(tab, sp, label):
        better = {}
        for a in DETS:
            for b in DETS:
                if a == b:
                    continue
                wins, losses = [], []
                for lead in ("II", "V2"):
                    x, y = tab[(a, lead)], tab[(b, lead)]
                    for name, m, marg in (("MAE", "MAE", MARGIN_MAE), ("p100", "p100", MARGIN_P100), ("worst record-lead MAE", "worst_record_MAE", MARGIN_MAE)):
                        d = x[m] - y[m]
                        if d < -marg:
                            wins.append(f"{lead} {name}")
                        elif d > marg:
                            losses.append(f"{lead} {name}")
                better[(a, b)] = (wins, losses)
        beaten = {d: [o for o in DETS if o != d and better[(o, d)][0] and not better[(o, d)][1]] for d in DETS}
        left = [d for d in DETS if not beaten[d]]
        res = f"{label}: not beaten beyond the margin: {left}"
        if len(left) > 1 and sp is not None:
            order = sorted(left, key=lambda d: sp[d])
            res += f"; tie-break on noise spurious rate picks {order[0]} ({', '.join(f'{d} {sp[d]:.1f} %' for d in order)})"
        elif len(left) == 1:
            res += f"; choice {left[0]}"
        return res, better
    return choose


if __name__ == "__main__":
    workers = int(sys.argv[1])
    for d in (OUT, PER, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    for m in ("DONE_step4_checks", "FAILED_step4_checks"):
        (OUT / m).unlink(missing_ok=True)
    missing = [str(delin(c, r, k)) for c in CONDS for r in RECS16 for k in (0, 1) if not delin(c, r, k).exists()]
    if missing:
        (OUT / "FAILED_step4_checks").write_text(f"missing delineation files: {missing}\n")
        sys.exit(f"missing {len(missing)} delineation files")
    failed = C14.run_records(job, [(r,) for r in RECS16], workers, need_gb=6.0)
    if failed:
        (OUT / "FAILED_step4_checks").write_text(f"failed: {failed}\n")
        sys.exit(f"failed: {failed}")
    ok, lines, T = summarise()
    pd.set_option("display.width", 200)
    lines += ["", T.round(4).to_string(index=False)]
    (OUT / "CHECKS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    (OUT / ("DONE_step4_checks" if ok else "FAILED_step4_checks")).write_text(time.strftime("%F %T") + "\n")
