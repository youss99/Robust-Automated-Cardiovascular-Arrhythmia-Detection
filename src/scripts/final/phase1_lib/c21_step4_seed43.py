#!/usr/bin/env python3
"""Phase 1 step 4, part 1 (README_phase1_runs.md): noise seed 43 through R-peak detection and delineation, for rodrigues2021, khamis2016 and neurokit.

Same protocol as seed 42: NSTDB em through ltst_pipeline.noise.contaminate_ecg_with_motion_artifact with seed=43, R-peaks from nk.ecg_peaks on the raw noisy lead
(no cleaning, once per record), delineation through c17_delineate.delineate (delineate_long_ecg_chunked, dwt, W 20 s, H 20 s, M 3 s, no filter), with the --formatpeaks_fix of c17.

Controls before anything else starts; a failure stops the run:
  neurokit, seed 42, through this script on one file must equal the stored delin_noise_nofilter_neurokit in every column.
  rodrigues2021 and khamis2016, seed 42, R-peaks only, on one record-lead each, must equal the stored screening peak files.
Each job: one process, written to a temporary name, R-peaks in the output checked against the input, then renamed. Finished files are skipped on a re-launch. A failed job is retried and never stops the others.
Output: Delineation Data/delin_noise43_nofilter_<detector>/lead_<k>/waves_<rec>_lead<k>.csv; C_neurokit/c21_step4/ (peaks/, control/, CONTROL.txt, STATUS.tsv, DONE_part1 or FAILED_part1).
Usage: C14_START_WAIT=15 setsid nohup python3 -u -W ignore c21_step4_seed43.py <workers> > <log> 2>&1 < /dev/null &"""
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(Path(__file__).parent))
import c17_delineate as C17  # noqa: E402  (also imports c6_noisy_fold, which reads the local NSTDB em)
import neurokit2 as nk  # noqa: E402
from c14_detectors_noisy import run_records  # noqa: E402
from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact as contam  # noqa: E402

DS = C17.DS
OUT = C17.CN / "c21_step4"
PEAKS = OUT / "peaks"
DETS = ("rodrigues2021", "khamis2016", "neurokit")
SEED, FS = 43, 250


def out_file(det, seed, rec, lead, root=C17.DEL):
    return root / f"delin_noise{'' if seed == 42 else seed}_nofilter_{det}" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv"


def peak_file(det, seed, rec, lead):
    return PEAKS / f"noisy{seed}_{det}_{rec}_lead{lead}.npy"


def signal(rec, lead, seed):
    clean = wfdb.rdrecord(str(DS / "ltst_records" / rec / rec)).p_signal
    noisy, _ = contam(clean, seed=seed, fs=FS)
    return np.ascontiguousarray(noisy[:, lead]).astype(float)


def peaks_for(det, seed, rec, lead, x):
    p = peak_file(det, seed, rec, lead)
    if p.exists():
        return np.load(p)
    _, info = nk.ecg_peaks(x, sampling_rate=FS, method=det)
    r = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    tmp = str(p) + f".tmp{os.getpid()}.npy"
    np.save(tmp, r)
    os.replace(tmp, p)
    return r


def log_row(*a):
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")


def job(t):
    key, det, seed, rec, lead, path = t
    t0 = time.time()
    try:
        x = signal(rec, lead, seed)
        r = peaks_for(det, seed, rec, lead, x)
        if path is None:  # R-peaks only
            log_row(det, seed, rec, lead, "peaks done", round(time.time() - t0), len(r), "")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        peaks = np.unique(r.astype(int))
        C17.delineate(x, peaks, rec, lead, tmp)
        w = pd.read_csv(tmp, usecols=["r_peak", "J_plus_60ms"])
        if not np.array_equal(np.sort(w.r_peak.to_numpy().astype(int)), peaks):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"R-peaks in the output differ from the input: {len(w)} vs {len(peaks)}")
        os.replace(tmp, path)
        log_row(det, seed, rec, lead, "done", round(time.time() - t0), len(w), round(100 * w.J_plus_60ms.isna().mean(), 2))
    except Exception:  # noqa: BLE001
        print(f"FAILED {key}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)


def stored_peak_file(det, rec, lead):
    for base in (C17.CN / "c14_detectors/peaks", C17.CN / "c16_all_detectors/peaks"):
        q = base / f"noisy_{det}_{rec}_lead{lead}.npy"
        if q.exists():
            return q
    raise FileNotFoundError(f"{det} {rec} {lead}")


def control(workers):
    root = OUT / "control"
    ctl_delin = ("neurokit", 42, "s20051", 1)
    ctl_peaks = (("rodrigues2021", 42, "s20201", 1), ("khamis2016", 42, "s20061", 0))
    p = out_file(*ctl_delin[:2], *ctl_delin[2:], root=root)
    tasks = [] if p.exists() else [("ctl/" + "/".join(map(str, ctl_delin)), *ctl_delin, p)]
    tasks += [("ctl/" + "/".join(map(str, c)), *c, None) for c in ctl_peaks if not peak_file(*c).exists()]
    failed = run_records(job, tasks, workers, need_gb=6.0)
    lines = []
    ok = not failed
    if failed:
        lines.append(f"control jobs failed: {failed}")
    else:
        a, b = pd.read_csv(p), pd.read_csv(out_file("neurokit", 42, "s20051", 1))
        same = list(a.columns) == list(b.columns) and len(a) == len(b) and all(a[c].astype(str).equals(b[c].astype(str)) if a[c].dtype == object else np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True) for c in a.columns)
        lines.append(f"neurokit seed 42 s20051 lead1 delineation: {len(a)} beats, identical to the stored delineation: {same}")
        ok &= same
        for det, seed, rec, lead in ctl_peaks:
            mine, stored = np.load(peak_file(det, seed, rec, lead)), np.load(stored_peak_file(det, rec, lead))
            same = np.array_equal(np.sort(mine), np.sort(stored))
            lines.append(f"{det} seed 42 {rec} lead{lead} R-peaks: {len(mine)} here, {len(stored)} stored, identical: {same}")
            ok &= same
    (OUT / "CONTROL.txt").write_text("\n".join(lines) + ("\nPASSED\n" if ok else "\nFAILED: nothing further was started\n"))
    print("\n".join(lines), flush=True)
    return ok


if __name__ == "__main__":
    workers = int(sys.argv[1])
    for d in (OUT, PEAKS):
        d.mkdir(parents=True, exist_ok=True)
    C17.patch_formatpeaks()
    for marker in ("DONE_part1", "FAILED_part1"):
        (OUT / marker).unlink(missing_ok=True)
    if not control(workers):
        (OUT / "FAILED_part1").write_text("control failed\n")
        sys.exit("control failed")
    tasks = [("/".join((det, str(SEED), rec, str(lead))), det, SEED, rec, lead, out_file(det, SEED, rec, lead)) for det in DETS for rec in C17.RECS for lead in (0, 1) if not out_file(det, SEED, rec, lead).exists()]
    print(len(tasks), "files to do", flush=True)
    failed = run_records(job, tasks, workers, need_gb=6.0)
    if failed:
        (OUT / "FAILED_part1").write_text(f"failed: {failed}\n")
        sys.exit(f"failed: {failed}")
    (OUT / "DONE_part1").write_text(f"{time.strftime('%F %T')} done\n")
    print("DONE", flush=True)
