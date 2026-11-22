#!/usr/bin/env python3
"""Final runs, step 7a (docs/instructions_final_results.md, step 7): re-score R1 with its published setup before anything else is scored.

R1 = the published CMBEC predictions (patient-segment-regression_2 checkpoints, 10 s grid), NSTDB em seed 42, the paper's .atr SQI gate at 6 dB,
PROSE FSM + published scorer, through the scoring code the final tables use (Tasks/phase1/eval/p1_eval.alarms, unchanged; only its output
folder is redirected here and its variant list is restricted to R1).
Pass rule: every count equals PAPER_RESULTS_TWO_MODELS.md exactly, every rate equals it at the paper's 3 decimals, and every row equals the
full-precision published metrics_<condition>.csv (all modes and scopes). A failure writes FAILED_7a and stops; nothing is adjusted.
Output: Tasks/final/runs/7a_r1_rescore/ (alarms/, _scores/, CHECK_7a.txt, compare_paper.csv, DONE_7a or FAILED_7a).
Usage: setsid nohup <repo .venv>/bin/python -u fin_7a_r1_rescore.py > logs/7a.log 2>&1 < /dev/null &"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/student/GIT/Tasks/phase1/eval")
import p1_eval as P  # noqa: E402

OUT = Path("/home/student/GIT/Tasks/final/runs/7a_r1_rescore")
P.ROOT, P.SC = OUT, OUT / "_scores"  # alarms() writes ROOT/<run>/alarms/<tag>/ and SC/alarms_<tag>.csv: never the phase-1 folders
TAGS = ("clean", "noisy", "noisy_sqi_6dB")
PAPER_MD = "/home/student/GIT/1_ltst_paper_cmbec48/results/PAPER_RESULTS_TWO_MODELS.md"
E5 = "/home/student/GIT/1_ltst_paper_cmbec48/results/alarms/E5_paper_results_table.csv"
HOURS = 331.2242  # paper: 15 scored records

# PAPER_RESULTS_TWO_MODELS.md, "Alarm — C1_alarm_metrics_pooled.csv" table (FULL scope, 101 episodes), and section 1 for union-pre's duration columns.
PAPER = [  # (arm, mode, tag, TP, FP, FN, Ign, P, R, F1, FP/24h/pt)
    ("pretrained", "union", "clean", 75, 40, 26, 21, 0.652, 0.743, 0.694, 2.90),
    ("pretrained", "union", "noisy", 73, 57, 28, 99, 0.562, 0.723, 0.632, 4.13),
    ("pretrained", "union", "noisy_sqi_6dB", 74, 28, 27, 16, (0.725, 0.726), 0.733, 0.729, 2.03),  # section 1 says 0.725, the C1 table 0.726 (74/102 = 0.72549)
    ("pretrained", "two_lead", "clean", 72, 41, 29, 25, 0.637, 0.713, 0.673, 2.97),
    ("pretrained", "two_lead", "noisy", 70, 48, 31, 64, 0.593, 0.693, 0.639, 3.48),
    ("pretrained", "two_lead", "noisy_sqi_6dB", 70, 32, 31, 22, 0.686, 0.693, 0.690, 2.32),
]
PAPER_DUR = [  # section 1 headline table: (tag, duration sensitivity, duration PPV, IoU), union-pre
    ("clean", 0.762, 0.647, 0.538), ("noisy", 0.644, 0.674, 0.491), ("noisy_sqi_6dB", 0.749, 0.618, 0.512)]
PAPER_SUP = [("clean", 22), ("noisy", 122), ("noisy_sqi_6dB", 13)]  # section 2b/§10: union-sup Ignored 22 -> 122 -> 13


def r1_only(tag):
    cond = "clean" if tag == "clean" else "noisy"
    return [("R1", arm, mode, "grid", pd.read_csv(P.grid_file("R1_published", arm, mode, cond))) for arm in P.ARMS for mode in P.MODES]


P.variants = r1_only


def r3(x):
    return float(f"{x:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for m in ("DONE_7a", "FAILED_7a"):
        (OUT / m).unlink(missing_ok=True)
    for tag in TAGS:
        if not (P.SC / f"DONE_alarms_{tag}").exists():
            P.alarms(tag)
    A = pd.concat([pd.read_csv(P.SC / f"alarms_{t}.csv") for t in TAGS], ignore_index=True)
    full = A[(A.run == "R1") & (A.scope == "FULL")].set_index(["arm", "mode", "condition"])
    lines, rows, ok = [], [], True

    def chk(name, mine, ref, exact):
        nonlocal ok
        refs = ref if isinstance(ref, tuple) else (ref,)
        good = any((mine == r) if exact else (r3(mine) == r) for r in refs)
        ok &= good
        rows.append({"check": name, "rescored": mine, "paper": "/".join(map(str, refs)), "pass": good})

    for arm, mode, tag, tp, fp, fn, ign, p, r, f1, fph in PAPER:
        x = full.loc[(arm, mode, tag)]
        for k, v in (("TP", tp), ("FP", fp), ("FN", fn), ("Ignored", ign)):
            chk(f"{arm} {mode} {tag} {k}", int(x[k]), v, True)
        for k, v in (("precision", p), ("recall", r), ("F1", f1)):
            chk(f"{arm} {mode} {tag} {k}", float(x[k]), v, False)
        chk(f"{arm} {mode} {tag} FP/24h/pt", round(float(x.FP) * 24 / HOURS, 2), fph, True)
    for tag, ds, dp, iou in PAPER_DUR:
        x = full.loc[("pretrained", "union", tag)]
        for k, v in (("sensitivity", ds), ("PPV_time", dp), ("IoU_whole_record", iou)):
            chk(f"pretrained union {tag} {k}", float(x[k]), v, False)
    for tag, ign in PAPER_SUP:
        chk(f"supervised union {tag} Ignored", int(full.loc[("supervised", "union", tag)].Ignored), ign, True)
    # E5 table at full precision (union rows of both arms)
    e5 = pd.read_csv(E5)
    e5 = e5[e5.system.str.startswith("Learned")]
    cmap = {"episode_F1": "F1", "duration_sensitivity": "sensitivity", "duration_PPV": "PPV_time", "IoU_whole_record": "IoU_whole_record", "fragmentation_Ignored": "Ignored"}
    for e in e5.itertuples():
        arm = "supervised" if "supervised" in e.system else "pretrained"
        x = full.loc[(arm, "union", e.condition)]
        for ec, mc in cmap.items():
            good = bool(np.isclose(float(x[mc]), float(getattr(e, ec)), rtol=0, atol=1e-9))
            ok &= good
            rows.append({"check": f"E5 {arm} union {e.condition} {mc}", "rescored": float(x[mc]), "paper": float(getattr(e, ec)), "pass": good})
    pd.DataFrame(rows).to_csv(OUT / "compare_paper.csv", index=False, lineterminator="\n")
    n_ok = sum(r["pass"] for r in rows)
    lines.append(f"{'PASS' if n_ok == len(rows) else 'FAIL'}  PAPER_RESULTS_TWO_MODELS.md (C1 table, section 1, section 2b) and E5_paper_results_table.csv: {n_ok}/{len(rows)} values equal")
    # every published metrics row at full precision
    cols = ["TP", "FP", "FN", "Ignored", "n_predicted_alarms", "F1", "time_F1", "IoU_whole_record", "sensitivity", "PPV_time"]
    for tag in TAGS:
        ref = pd.read_csv(f"/home/student/GIT/results/beatcenter_test1/metrics_{tag}.csv").query("model_set == 'published' and scheme == 'grid' and arm != 'PTBXL-pretrained'")
        mine = A[(A.run == "R1") & (A.condition == tag)].assign(arm=lambda d: d.arm.map(P.PUB_ARM))
        m = mine.merge(ref, on=["arm", "mode", "scope"], suffixes=("", "_ref"))
        bad = [c for c in cols if not np.allclose(m[c].to_numpy(float), m[c + "_ref"].to_numpy(float), atol=1e-9, rtol=0, equal_nan=True)]
        good = len(m) == len(ref) == len(mine) and not bad
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'}  metrics_{tag}.csv (published, full precision): {len(m)}/{len(ref)} rows, differing columns: {bad}")
    lines.append("ALL PASSED" if ok else "FAILED: stop and report; nothing downstream was scored")
    (OUT / "CHECK_7a.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    (OUT / ("DONE_7a" if ok else "FAILED_7a")).write_text(time.strftime("%F %T") + "\n")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
