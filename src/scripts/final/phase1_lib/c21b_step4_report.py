#!/usr/bin/env python3
"""Phase 1 step 4, part 2 (README_phase1_runs.md): the numbers for the R-peak detector freeze. rodrigues2021, khamis2016, neurokit, each with its own fold offsets.

Per detector and lead:
  DMY error against the .16a DMY ee: MAE, p100 (share of |error| > 0.1 mV), bias, RMSE, usable beats, worst record-lead and its MAE.
     clean = the decision (user, 2026-11-08); noisy seeds 42 and 43 are extra rows. With and without masked pairs. Window median over each DMY beat's NL/NR beats, as C section 5.
  spurious R-peak rate inside noise bursts, seeds 42 and 43: share of detected peaks in a burst further than 148 ms from any .atr beat (C section 10c's definition).
  alarms (reported, never used to choose): clean, noisy and gated 6 dB for seeds 42 (from c20) and 43 (here), own noisy R-peaks for the gate.
  which records hold khamis2016's clean V2 false alarms.
Rule, fixed in advance: primary = DMY MAE and p100 per lead, including the worst record-lead, on clean signal. Tie = MAE within 0.005 mV and p100 within 1 point (margin set after §12 was seen). Tie-break = lower spurious rate in noise.
Also: the same choice made within each fold, on that fold's training records only (manifest split = train), with offsets fitted on those records.

Controls: clean own-offset DMY values must equal c20's dmy_per_beat files; spurious rates must reproduce the stored screening figures; the alarm code at seed 42 must reproduce c20's alarms on two records.
Usage: python3 c21b_step4_report.py control | run <workers> | summarise. Output: C_neurokit/c21_step4/report/."""
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(Path(__file__).parent))
import c20_finalists as B  # noqa: E402
import c14_detectors_noisy as C14  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact as contam  # noqa: E402

OUT = B.CN / "c21_step4/report"
PER = OUT / "per_record"
PEAKS43 = B.CN / "c21_step4/peaks"
REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
FS = 250
DETS = B.DETS
RECS16 = sorted(B.TEST + ["s20121"])
TRAIN = {k: list(pd.read_csv(REPO / f"ptbxl_models_v2/cv5_optB_fold_manifests/fold{k:02d}.csv").query("split == 'train'").patient_id) for k in range(1, 6)}
MASKED = {"II": {"s20091", "s20121"}, "V2": {"s20081", "s20121", "s20221", "s20241"}}
LN = {0: "II", 1: "V2"}
MARGIN_MAE, MARGIN_P100 = 0.005, 1.0


def delin_path(det, cond, rec, lead):
    tag = {"clean": "raw", "noisy42": "noise", "noisy43": "noise43"}[cond]
    return B.DEL / f"delin_{tag}_nofilter_{det}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv"


def dmy_values(x, w, off, L):
    """One value per DMY beat: median of the usable per-beat ST deviations over its NL/NR window, matching each window beat to the nearest R-peak within 100 ms."""
    st, iso = B.C6.per_beat(x, w, "rpk", off)
    dev = np.round(st - iso, 6)
    R = w.r_peak.to_numpy().astype(np.int64)
    s16 = L["sample"].to_numpy()
    i = np.clip(np.searchsorted(R, s16), 1, len(R) - 1)
    near = np.where(np.abs(R[i] - s16) < np.abs(R[i - 1] - s16), i, i - 1)
    idx = np.where(np.abs(s16 - R[near]) <= B.C4.TOL, near, -1)
    out = []
    for k in np.where(L.DMY.to_numpy())[0]:
        r = L.iloc[k]
        win = idx[max(0, k - int(r.NL)): min(len(L) - 1, k + int(r.NR)) + 1]
        v = dev[win[win >= 0]]
        v = v[~np.isnan(v)]
        out.append((int(r["sample"]), r.ee_uV / 1000.0, float(np.median(v)) if len(v) else np.nan))
    return out


def burst_mask(n, seed):
    _, log = contam(np.zeros((n, 2)), seed=seed, fs=FS)
    m = np.zeros(n + 1, bool)
    for b in log:
        m[b["start_sample"]: b["end_sample"]] = True
    return m


def peaks_file(det, seed, rec, lead):
    if seed == 43:
        return PEAKS43 / f"noisy43_{det}_{rec}_lead{lead}.npy"
    for base in (B.CN / "c14_detectors/peaks", B.CN / "c16_all_detectors/peaks"):
        q = base / f"noisy_{det}_{rec}_lead{lead}.npy"
        if q.exists():
            return q
    raise FileNotFoundError(f"{det} {rec} {lead}")


def alarms(rec, seed, noisy, clean_rows=False):
    """rpkdet alarms for one noise seed, as c20_finalists.job."""
    O = pd.read_csv(B.OUT / "offsets.csv")
    hp = B.highpass_filter(noisy, fs=FS)
    rows = []
    cond_tag = {"noisy42": "noisy", "noisy43": "noisy"}
    for lead in (0, 1):
        start, end = B.C13.grid(rec, lead)
        ecg = np.ascontiguousarray(hp[:, lead])
        for det in DETS:
            p = delin_path(det, f"noisy{seed}", rec, lead)
            if not p.exists():
                continue
            dn = pd.read_csv(p, usecols=B.COLS)
            gate_peaks = np.sort(dn.r_peak.to_numpy().astype(int))
            off = float(O[(O.record == rec) & (O.detector == det) & (O.lead == lead)].iloc[0].rpkdet)
            st, iso = B.C6.per_beat(noisy[:, lead], dn, "rpk", off)
            sn = B.C13.segments(dn.J_plus_60ms.to_numpy(float), np.round(st - iso, 6), start)
            keep = lambda al, c: rows.extend(dict(detector=det, seed=seed, condition=c, record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, onset_actual_s=a.onset_actual / FS, end_actual_s=a.end_actual / FS, criterion=a.crit) for a in al)
            with contextlib.redirect_stdout(io.StringIO()):
                keep(B.F.run_fsm_on_record(rec_id=rec, lead_tag=B.C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=None, r_peaks=None, fs=FS, use_sqi=False), "noisy")
                keep(B.F.run_fsm_on_record(rec_id=rec, lead_tag=B.C13.TAG[lead], samp_arr=end.tolist(), st_arr=sn.tolist(), ecg_sig_for_sqi=ecg, r_peaks=gate_peaks, fs=FS, use_sqi=True, sqi_threshold_db=B.THR, sqi_every_s=5.0, sqi_window_s=30.0), "gated6")
    return pd.DataFrame(rows, columns=["detector", "seed", "condition", "record", "lead_tag", "lead_paper", "onset_actual_s", "end_actual_s", "criterion"])


def fold_offsets():
    B_ = pd.read_csv(B.OUT / "offsets_per_dmy_beat.csv")
    return {(k, d, l): float(B_[(B_.detector == d) & (B_.lead == l) & B_.record.isin(tr)].iso_before_r_ms.median()) for k, tr in TRAIN.items() for d in DETS for l in (0, 1)}


def job(task):
    rec = task[0]
    target = PER / f"dmy_{rec}.csv"
    if target.exists():
        return rec
    O = pd.read_csv(B.OUT / "offsets.csv")
    FO = fold_offsets()
    clean, noisy42 = B.C6.noisy_signal(rec)
    noisy43, _ = contam(clean, seed=43, fs=FS)
    a16 = B.C4.load_16a(rec)
    atr = np.sort(B.F.get_rpeaks_from_atr(wfdb.rdann(str(B.DS / "ltst_records" / rec / rec), "atr"))).astype(np.int64)
    n = clean.shape[0]
    masks = {42: burst_mask(n, 42), 43: burst_mask(n, 43)}
    sig = {"clean": clean, "noisy42": noisy42, "noisy43": noisy43}
    rows, spur = [], []
    for lead in (0, 1):
        L = a16[a16.lead_idx == lead].sort_values("sample").reset_index(drop=True)
        for det in DETS:
            off = float(O[(O.record == rec) & (O.detector == det) & (O.lead == lead)].iloc[0].rpkdet)
            for cond, X in sig.items():
                p = delin_path(det, cond, rec, lead)
                if not p.exists() or not L.DMY.any():
                    continue
                w = pd.read_csv(p, usecols=B.COLS)
                rows += [dict(record=rec, lead=LN[lead], detector=det, condition=cond, fold=0, sample=s, ee=e, value=v) for s, e, v in dmy_values(X[:, lead], w, off, L)]
                if cond == "clean":
                    for k in TRAIN:
                        if rec in TRAIN[k]:
                            rows += [dict(record=rec, lead=LN[lead], detector=det, condition="clean_trainfold", fold=k, sample=s, ee=e, value=v) for s, e, v in dmy_values(X[:, lead], w, FO[(k, det, lead)], L)]
            for seed in (42, 43):
                q = peaks_file(det, seed, rec, lead)
                if not q.exists():
                    continue
                P = np.load(q).astype(np.int64)
                far = np.abs(C14.nearest_off(P, atr)) > C14.TRUE
                inb = masks[seed][P]
                spur.append(dict(record=rec, lead=LN[lead], detector=det, seed=seed, in_scope=rec != "s20121" and rec not in MASKED[LN[lead]], detected_in_burst=int(inb.sum()), spurious_in_burst=int((inb & far).sum())))
    al = alarms(rec, 43, noisy43)
    pd.DataFrame(spur).to_csv(PER / f"spurious_{rec}.csv", index=False)
    al.to_csv(PER / f"alarms43_{rec}.csv", index=False)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(rec, "done", flush=True)
    return rec


def control():
    """Alarm code at seed 42 against c20 on two records, and the screening spurious rates."""
    lines = []
    cols = ["record", "lead_paper", "onset_actual_s", "end_actual_s"]
    for rec in ("s20011", "s20051"):
        _, noisy = B.C6.noisy_signal(rec)
        a = alarms(rec, 42, noisy)
        b = pd.read_csv(B.PER / f"alarms_{rec}.csv", dtype={"lead_paper": str})
        for det in DETS:
            for c in ("noisy", "gated6"):
                x = a[(a.detector == det) & (a.condition == c)][cols].astype({"lead_paper": str}).sort_values(cols).round(3).reset_index(drop=True)
                y = b[(b.detector == det) & (b.config == "rpkdet") & (b.condition == c)][cols].sort_values(cols).round(3).reset_index(drop=True)
                lines.append(f"alarms at seed 42, {rec} {det} {c}: {len(x)} here, {len(y)} in c20, identical: {x.equals(y)}")
    (OUT / "control_alarms.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return all(x.endswith("True") for x in lines)


def summarise():
    D = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("dmy_s2*.csv"))], ignore_index=True)
    D = D[D.record.isin(RECS16)]
    D["err"] = D.value - D.ee
    lines = []
    # control: clean own-offset values against c20
    for det in DETS:
        c = pd.read_csv(B.OUT / f"dmy_per_beat_{det}.csv")
        c["lead"] = c.lead.map({"ML2": "II", "MV2": "V2"})
        m = D[(D.detector == det) & (D.condition == "clean")].merge(c, left_on=["record", "lead", "sample"], right_on=["record", "lead", "sample_16a"])
        same = len(m) == len(c) and np.allclose(np.nan_to_num(m.value, nan=9e9), np.nan_to_num(m.rpkdet_median_mV, nan=9e9), atol=2e-6)
        lines.append(f"clean DMY values, {det}: {len(m)} of {len(c)} beats matched to c20, identical: {same}")

    def stats(q):
        u = q.dropna(subset=["value"])
        return dict(dmy_beats=len(q), usable=len(u), MAE=float(u.err.abs().mean()), p100=float(100 * (u.err.abs() > 0.1).mean()), bias=float(u.err.mean()), RMSE=float(np.sqrt((u.err ** 2).mean())))

    rows, worst = [], []
    for cond in ("clean", "noisy42", "noisy43"):
        for det in DETS:
            for lead in ("II", "V2"):
                for sub in ("no masked", "all"):
                    q = D[(D.condition == cond) & (D.detector == det) & (D.lead == lead) & D.record.isin(B.TEST)]
                    if sub != "all":
                        q = q[~q.record.isin(MASKED[lead])]
                    if q.empty:
                        continue
                    per = {r: stats(g) for r, g in q.groupby("record")}
                    wr = max(per, key=lambda r: per[r]["MAE"])
                    rows.append(dict(condition=cond, detector=det, lead=lead, subset=sub, **stats(q), worst_record=wr, worst_record_MAE=per[wr]["MAE"], worst_record_p100=per[wr]["p100"]))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "dmy_summary.csv", index=False)
    SP = pd.concat([pd.read_csv(f) for f in sorted(PER.glob("spurious_s2*.csv"))], ignore_index=True)
    SP.to_csv(OUT / "spurious_per_record.csv", index=False)
    q = SP[SP.in_scope]
    T = q.groupby(["detector", "lead", "seed"])[["detected_in_burst", "spurious_in_burst"]].sum().reset_index()
    T["spurious_pct"] = 100 * T.spurious_in_burst / T.detected_in_burst
    T.to_csv(OUT / "spurious_summary.csv", index=False)
    # rule
    out = []

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

    for sub in ("no masked", "all"):
        tab = {(r.detector, r.lead): r._asdict() for r in S[(S.condition == "clean") & (S.subset == sub)].itertuples()}
        sp = {d: 100 * q[q.detector == d].spurious_in_burst.sum() / q[q.detector == d].detected_in_burst.sum() for d in DETS}
        res, better = choose(tab, sp, f"Clean DMY, {sub}")
        out.append(res)
        for (a, b), (w, l) in better.items():
            out.append(f"   {a} vs {b}: better on {w or 'nothing'}; worse on {l or 'nothing'}")
    for k in TRAIN:
        F_ = D[(D.condition == "clean_trainfold") & (D.fold == k)]
        tab = {}
        for det in DETS:
            for lead in ("II", "V2"):
                q2 = F_[(F_.detector == det) & (F_.lead == lead) & ~F_.record.isin(MASKED[lead])]
                per = {r: stats(g) for r, g in q2.groupby("record")}
                wr = max(per, key=lambda r: per[r]["MAE"])
                tab[(det, lead)] = {**stats(q2), "worst_record_MAE": per[wr]["MAE"]}
        tr = SP[SP.record.isin(TRAIN[k]) & SP.in_scope]
        sp = {d: 100 * tr[tr.detector == d].spurious_in_burst.sum() / tr[tr.detector == d].detected_in_burst.sum() for d in DETS}
        out.append(choose(tab, sp, f"Fold {k}, training records only, no masked")[0])
    (OUT / "rule.txt").write_text("\n".join(out) + "\n")
    (OUT / "controls.txt").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print("\n".join(lines))
    print(S.round(4).to_string(index=False))
    print(T.round(2).to_string(index=False))
    print("\n".join(out), flush=True)
    return all(x.endswith("True") for x in lines)


if __name__ == "__main__":
    stage = sys.argv[1]
    for d in (OUT, PER):
        d.mkdir(parents=True, exist_ok=True)
    if stage == "control":
        sys.exit(0 if control() else "alarm control failed")
    if stage == "run":
        failed = C14.run_records(job, [(r,) for r in RECS16], int(sys.argv[2]), need_gb=6.0)
        if failed:
            (OUT / "FAILED_part2").write_text(f"failed: {failed}\n")
            sys.exit(f"failed: {failed}")
    if stage in ("run", "summarise"):
        ok = summarise()
        (OUT / ("DONE_part2" if ok else "CONTROL_FAILED")).write_text("done\n" if ok else "see controls.txt\n")
        print("DONE" if ok else "CONTROL FAILED", flush=True)
