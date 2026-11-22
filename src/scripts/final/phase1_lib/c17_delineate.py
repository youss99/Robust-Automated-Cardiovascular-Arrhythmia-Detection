#!/usr/bin/env python3
"""Real delineation for the R-peak finalists, the same way as the stored delin_{raw,noise}_nofilter_neurokit.

Same function and settings: ltst_pipeline.delineation.delineate_long_ecg_chunked, dwt, W 20 s, H 20 s, M 3 s, raw signal with no filter; noisy = NSTDB em, seed 42 (c6_noisy_fold.noisy_signal).
One difference, in where the R-peaks come from: they are read from the saved screening files (c14/c16 peaks), which hold the output of nk.ecg_peaks on these same signals. This guarantees the screening's R-peaks and saves
repeating slow detection (about 18 min per lead-condition for khamis2016).

Control, before anything else is accepted: neurokit through this script on one clean and one noisy file must equal the stored delineation in every column. If it does not, nothing further starts.
Each file: one process, written to a temporary name, then checked (its R-peaks must equal the input R-peaks) and renamed. A failure is logged and retried, and never stops other files. Finished files are skipped on a re-launch.
Output: Delineation Data/delin_{raw,noise}_nofilter_<method>/lead_<k>/waves_<rec>_lead<k>.csv; log C_neurokit/c17_delineation/ (STATUS.tsv, CONTROL.txt, DONE_all).
Usage: setsid nohup python3 -u -W ignore c17_delineate.py --workers 8 --methods rodrigues2021 khamis2016 > <log> 2>&1 < /dev/null &

--formatpeaks_fix (added 2026-11-07, approved by the user for the 2 files that crashed): NeuroKit's ecg_delineate sets an index past the end of the
window to NaN only when it is the last in its list (ecg_delineate.py:192-194, the check at 193). One elsewhere in the list crashes signal_formatpeaks, whose output this
pipeline does not use (delineation.py keeps only the waves dict). The fix drops such indices in that formatting step only; the waves are unchanged.
Any file that finished without the fix had no such index, so the fix cannot change it. To show that, the neurokit control is re-run with the fix into
c17_delineation/control_formatfix and must equal the stored delineation before the missing files start. Marker: c17_delineation/DONE_formatfix.
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Tasks/C_neurokit")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
import c6_noisy_fold as C6  # noqa: E402  (reads the local NSTDB em)
import neurokit2 as nk  # noqa: E402
from ltst_pipeline import delineation as DL  # noqa: E402

DS = Path("/home/student/Desktop/Research Data/DATASET NK")
CN, DEL = DS / "C_neurokit", DS / "Delineation Data"
OUT = CN / "c17_delineation"
RECS = "s20011 s20051 s20061 s20071 s20081 s20091 s20101 s20111 s20121 s20131 s20141 s20201 s20211 s20221 s20231 s20241".split()
PEAKS = {"neurokit": CN / "c14_detectors/peaks", "rodrigues2021": CN / "c14_detectors/peaks", "manikandan2012": CN / "c14_detectors/peaks"}
COND = {"clean": "raw", "noisy": "noise"}
FS, TIMEOUT = 250, 3 * 3600


def peak_file(m, cond, rec, lead):
    return PEAKS.get(m, CN / "c16_all_detectors/peaks") / f"{cond}_{m}_{rec}_lead{lead}.npy"


def out_file(m, cond, rec, lead, root=DEL):
    return root / f"delin_{COND[cond]}_nofilter_{m}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv"


def patch_formatpeaks():
    ED = sys.modules["neurokit2.ecg.ecg_delineate"]  # the package name itself resolves to the function
    orig = ED.signal_formatpeaks

    def safe(info, desired_length, *a, **k):
        return orig({key: [v for v in vals if v < desired_length] for key, vals in info.items()}, desired_length, *a, **k)
    ED.signal_formatpeaks = safe


def delineate(x, peaks, rec, lead, path):
    class Shim:  # the R-peaks are already known; everything else is the unmodified function
        ecg_delineate = staticmethod(nk.ecg_delineate)
        ecg_peaks = staticmethod(lambda s, sampling_rate, method: (None, {"ECG_R_Peaks": peaks}))
    DL.nk = Shim
    return DL.delineate_long_ecg_chunked(x, fs=FS, W_s=20.0, H_s=20.0, M_s=3.0, method="dwt", rpeak_method="(saved)", out_csv=str(path), record_name=rec, lead_id=lead)


def log_row(*a):
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")


def job(t):
    m, cond, rec, lead, path = t
    t0 = time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        clean, noisy = C6.noisy_signal(rec)
        x = np.ascontiguousarray((clean if cond == "clean" else noisy)[:, lead]).astype(float)
        del clean, noisy
        peaks = np.unique(np.load(peak_file(m, cond, rec, lead)).astype(int))
        delineate(x, peaks, rec, lead, tmp)
        w = pd.read_csv(tmp, usecols=["r_peak", "J_plus_60ms"])
        if not np.array_equal(np.sort(w.r_peak.to_numpy().astype(int)), peaks):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"R-peaks in the output differ from the input: {len(w)} vs {len(peaks)}")
        os.replace(tmp, path)
        log_row(m, cond, rec, lead, "done", round(time.time() - t0), len(w), round(100 * w.J_plus_60ms.isna().mean(), 2))
    except Exception:  # noqa: BLE001
        print(f"FAILED {m} {cond} {rec} lead{lead}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def run(tasks, workers, need_gb=6.0, wait=30, retries=2, after=None):
    """after(label) runs when a task finishes; returning False cancels everything still waiting."""
    pending, running, tries, failed = list(tasks), {}, {}, []
    while pending or running:
        for key, (p, t, t0) in list(running.items()):
            if p.is_alive() and time.time() - t0 > TIMEOUT:
                print(f"{key}: over the time limit, stopping it", flush=True)
                p.kill()
            if not p.is_alive():
                p.join()
                del running[key]
                if p.exitcode != 0:
                    tries[key] = tries.get(key, 0) + 1
                    print(f"{key}: exit {p.exitcode}, try {tries[key]} of {retries + 1}", flush=True)
                    if tries[key] <= retries:
                        pending.append(t)
                    else:
                        failed.append(key)
                        log_row(*t[:4], "FAILED after retries", "", "", "")
                if after and not after(key):
                    pending.clear()
        started = False
        if pending and len(running) < workers and (avail_gb() >= need_gb or not running):
            t = pending.pop(0)
            key = "/".join(map(str, t[:4]))
            p = Process(target=job, args=(t,))
            p.start()
            running[key] = (p, t, time.time())
            started = True
        time.sleep(wait if started else 5)
    return failed


def control_check(root):
    """Every column of the control files against the stored neurokit delineation."""
    lines, ok = [], True
    for cond, rec, lead in CONTROL:
        a = pd.read_csv(out_file("neurokit", cond, rec, lead, root))
        b = pd.read_csv(out_file("neurokit", cond, rec, lead))
        same = list(a.columns) == list(b.columns) and len(a) == len(b) and all(a[c].astype(str).equals(b[c].astype(str)) if a[c].dtype == object else np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True) for c in a.columns)
        lines.append(f"{cond} {rec} lead{lead}: {len(a)} beats, {'identical to the stored delineation' if same else 'DIFFERENT from the stored delineation'}")
        ok &= same
    return ok, lines


CONTROL = [("clean", "s20081", 0), ("noisy", "s20051", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", nargs="+", default=["rodrigues2021", "khamis2016"])
    ap.add_argument("--formatpeaks_fix", action="store_true", help="drop past-the-end indices in NeuroKit's formatting step; re-runs the control first")
    ap.add_argument("--smoke", action="store_true", help="1 h of noisy s20011 lead 0 per method, written to a scratch folder")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        scratch = OUT / "smoke"
        scratch.mkdir(exist_ok=True)
        _, noisy = C6.noisy_signal("s20011")
        x = np.ascontiguousarray(noisy[:, 0]).astype(float)[:3600 * FS]
        for m in a.methods + ["neurokit"]:
            t0 = time.time()
            p = np.unique(np.load(peak_file(m, "noisy", "s20011", 0)).astype(int))
            p = p[p < len(x)]
            q = scratch / f"{m}.csv"
            delineate(x, p, "s20011", 0, q)
            w = pd.read_csv(q)
            print(f"{m}: {len(p)} R-peaks in, {len(w)} rows out, J+60 missing {100 * w.J_plus_60ms.isna().mean():.1f} %, {time.time() - t0:.0f} s", flush=True)
        return
    if a.formatpeaks_fix:
        patch_formatpeaks()
        (OUT / "DONE_formatfix").unlink(missing_ok=True)
        root = OUT / "control_formatfix"
        ctl = [("neurokit", c, r, l, out_file("neurokit", c, r, l, root)) for c, r, l in CONTROL if not out_file("neurokit", c, r, l, root).exists()]
        failed = run(ctl, a.workers)
        ok, lines = control_check(root) if not failed else (False, [f"control files failed: {failed}"])
        (OUT / "CONTROL_formatfix.txt").write_text("\n".join(lines) + ("\nPASSED\n" if ok else "\nFAILED: nothing further was started\n"))
        print("\n".join(lines), flush=True)
        failed = []
        if ok:
            for m in a.methods:
                for c in ("clean", "noisy"):
                    for r in RECS:
                        for l in (0, 1):
                            p = out_file(m, c, r, l)
                            if not p.exists():
                                for t in p.parent.glob(p.name + ".tmp*"):
                                    t.unlink()
            tasks = [(m, c, r, l, out_file(m, c, r, l)) for m in a.methods for c in ("clean", "noisy") for r in RECS for l in (0, 1) if not out_file(m, c, r, l).exists()]
            print("missing files:", [t[:4] for t in tasks], flush=True)
            failed = run(tasks, a.workers)
        (OUT / "DONE_formatfix").write_text(f"{time.strftime('%F %T')} control {'passed' if ok else 'FAILED'}; failed files: {failed}\n")
        print("DONE, failed files:", failed, flush=True)
        return
    (OUT / "DONE_all").unlink(missing_ok=True)
    root = OUT / "control"
    tasks = [("neurokit", c, r, l, out_file("neurokit", c, r, l, root)) for c, r, l in CONTROL if not out_file("neurokit", c, r, l, root).exists()]
    for m in a.methods:
        tasks += [(m, c, r, l, out_file(m, c, r, l)) for c in ("clean", "noisy") for r in RECS for l in (0, 1) if not out_file(m, c, r, l).exists()]
    state = {"checked": False, "ok": True}

    def after(_key):
        if not state["checked"] and all(out_file("neurokit", c, r, l, root).exists() for c, r, l in CONTROL):
            state["checked"] = True
            state["ok"], lines = control_check(root)
            (OUT / "CONTROL.txt").write_text("\n".join(lines) + ("\nPASSED\n" if state["ok"] else "\nFAILED: nothing further was started\n"))
            print("\n".join(lines), flush=True)
        return state["ok"]

    if not tasks and all(out_file("neurokit", c, r, l, root).exists() for c, r, l in CONTROL):
        after("")
    failed = run(tasks, a.workers, after=after)
    (OUT / "DONE_all").write_text(f"{time.strftime('%F %T')} control {'passed' if state['ok'] else 'FAILED'}; failed files: {failed}\n")
    print("DONE, failed files:", failed, flush=True)


if __name__ == "__main__":
    main()
