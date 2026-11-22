#!/usr/bin/env python3
"""Final runs, VM step 4 (docs/instructions_final_results.md): noise seed 44 through rodrigues2021 R-peak detection and delineation.

Same protocol as seeds 42 and 43, through the seed-43 script's own functions (C_neurokit/c21_step4_seed43.py, unchanged):
NSTDB em via ltst_pipeline.noise.contaminate_ecg_with_motion_artifact(seed=44), R-peaks from nk.ecg_peaks(method="rodrigues2021") on the raw
noisy lead, delineation via c17_delineate.delineate (dwt, W 20 s, H 20 s, M 3 s, no filter, --formatpeaks_fix).

Control first (a failure stops everything): seed 43, rodrigues2021, s20201 lead 1, through this script must reproduce the stored seed-43
R-peaks and the stored delin_noise43_nofilter_rodrigues2021 file in every column.
Output: Delineation Data/delin_noise44_nofilter_rodrigues2021/lead_<k>/waves_<rec>_lead<k>.csv; peaks and logs in DATASET NK/final/step4/
(peaks/, control/, CONTROL.txt, STATUS.tsv, DONE_seed44 or FAILED_seed44). Finished files are skipped on a re-launch.
Usage: setsid nohup python3 -u -W ignore fin_step4_seed44.py <workers> > <log> 2>&1 < /dev/null &"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1_lib"))  # vendored phase-1 code (phase1_lib/MANIFEST.tsv)
import c21_step4_seed43 as C21  # noqa: E402

C17 = C21.C17
OUT = C17.DS / "final" / "step4"
C21.OUT, C21.PEAKS = OUT, OUT / "peaks"  # the module's functions read these globals: logs and peak files go here, not into c21_step4
DET, SEED = "rodrigues2021", 44
CTL = (DET, 43, "s20201", 1)


def same_frame(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return all(a[c].astype(str).equals(b[c].astype(str)) if a[c].dtype == object else np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True) for c in a.columns)


def control(workers):
    det, seed, rec, lead = CTL
    p = C21.out_file(det, seed, rec, lead, root=OUT / "control")
    tasks = [] if p.exists() else [("ctl/" + "/".join(map(str, CTL)), det, seed, rec, lead, p)]
    failed = C21.run_records(C21.job, tasks, workers, need_gb=6.0)
    if failed:
        lines, ok = [f"control job failed: {failed}"], False
    else:
        mine_pk = np.load(C21.peak_file(det, seed, rec, lead))
        stored_pk = np.load(C17.CN / "c21_step4" / "peaks" / f"noisy{seed}_{det}_{rec}_lead{lead}.npy")
        a, b = pd.read_csv(p), pd.read_csv(C21.out_file(det, seed, rec, lead))
        ok_pk, ok_del = np.array_equal(np.sort(mine_pk), np.sort(stored_pk)), same_frame(a, b)
        lines = [f"{det} seed {seed} {rec} lead{lead} R-peaks: {len(mine_pk)} here, {len(stored_pk)} stored, identical: {ok_pk}",
                 f"{det} seed {seed} {rec} lead{lead} delineation: {len(a)} beats here, {len(b)} stored, identical in every column: {ok_del}"]
        ok = ok_pk and ok_del
    (OUT / "CONTROL.txt").write_text("\n".join(lines) + ("\nPASSED\n" if ok else "\nFAILED: nothing further was started\n"))
    print("\n".join(lines), flush=True)
    return ok


if __name__ == "__main__":
    workers = int(sys.argv[1])
    for d in (OUT, C21.PEAKS):
        d.mkdir(parents=True, exist_ok=True)
    C17.patch_formatpeaks()
    for m in ("DONE_seed44", "FAILED_seed44"):
        (OUT / m).unlink(missing_ok=True)
    if not control(workers):
        (OUT / "FAILED_seed44").write_text("control failed\n")
        sys.exit("control failed")
    tasks = [("/".join((DET, str(SEED), r, str(k))), DET, SEED, r, k, C21.out_file(DET, SEED, r, k)) for r in C17.RECS for k in (0, 1) if not C21.out_file(DET, SEED, r, k).exists()]
    print(len(tasks), "files to do", flush=True)
    failed = C21.run_records(C21.job, tasks, workers, need_gb=6.0)
    if failed:
        (OUT / "FAILED_seed44").write_text(f"failed: {failed}\n")
        sys.exit(f"failed: {failed}")
    (OUT / "DONE_seed44").write_text(f"{time.strftime('%F %T')} done\n")
    print("DONE", flush=True)
