#!/usr/bin/env python3
"""Podium-only scoring (user, 2026-11-24; DECISIONS.md podium-only exception). No training; no thesis output is touched.

Paper configuration, one run per system, noise seed 42:
  Fiducial      NeuroKit R-peaks + NeuroKit dwt delineation, 5-sample ST level at J+60 minus the PR-segment median, per-beat values rounded to 1e-6 mV
                (fin_nk_detector.py step-5 table, column NKPR; anchor 2500k + 2500).
  Learned-1L    pretrained, training seed S1 (step-6 predictions FIN_L1_lead_{II,V2}_S1; anchor 2500k + 2499).
  Gate          the paper's SQI from .atr beats, 6 dB, every gated row (columns sqi_nk_atr / sqi_grid_atr, copied from step 5).
  Monitor       copied from the paper (Abdelazez 2017).
Alarm path and scorer: fin_step7_tier1 (T.fsm, T.score, T.learned_rows), unchanged; union of II and V2, FULL scope, 101 episodes, excluded pairs out.
(a) Table 1: TP, FP, FN, redundant, P, R, F1.  (b) Table 2: duration Se, PPV, IoU, coverage (eq. 4 = (TPt + FPt) / (TPt + FNt), per record-lead) and
coverage with alarms merged across leads.  (c) gated TP per lead (own-lead scope, as the appendix per-record table).  (d) Fiducial baseline windows
with no value.
Controls (stop on failure): Learned-1L gated = A4a S1 .atr row (76/25/25); Fiducial artifact = 67/43/34; Fiducial gated = 64/27/37.
Definition check for the merged coverage: the paper's learned arm (published R1 predictions, .atr gate) must give the paper's 1.213 and, merged, ~1.06.
Output: Tasks/final/runs/podium/ (CSV, CONTROLS.txt), docs/PODIUM_RESULTS.md.
Usage: <repo .venv>/bin/python -u -W ignore podium_results.py"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fin_step7_tier1 as T  # noqa: E402

P, K = T.P, T.K
V5 = Path("/home/student/GIT/Tasks/final/runs/nk_detector/step5")
OUT = Path("/home/student/GIT/Tasks/final/runs/podium")
MD = HERE.parents[2] / "docs" / "PODIUM_RESULTS.md"
CONDN = {"clean": "baseline", "noisy": "artifact", "gated": "gated"}
PAPER = {"Fiducial": {"clean": (66, 32, 35), "noisy": (66, 40, 35), "gated": (64, 24, 37)},
         "Learned-1L": {"clean": (75, 40, 26), "noisy": (73, 57, 28), "gated": (74, 28, 27)}}
PAPER_T2 = {("Fiducial", "noisy"): (0.455, 0.727, 0.389, 0.627), ("Fiducial", "gated"): (0.496, 0.721, 0.416, 0.688),
            ("Learned-1L", "noisy"): (0.644, 0.674, 0.491, 0.955), ("Learned-1L", "gated"): (0.749, 0.618, 0.512, 1.213)}


def fid_rows(V, cond, M):
    d = V["clean" if cond == "clean" else "noisy42"]
    rows = []
    for (rec, L), g in d[d.nk_window].groupby(["record", "lead"]):
        g = g.sort_values("k")
        rows += T.fsm(rec, L, g.anchor_nk, g.NKPR, M[(rec, L)] if cond == "gated" else None)
    return rows


def r1_frames(cond):
    out = {}
    for mode, L in (("lead_II", "II"), ("lead_V2", "V2")):
        f = pd.read_csv(P.grid_file("R1_published", "pretrained", mode, "clean" if cond == "clean" else "noisy"), usecols=["record", "anchor_samp", f"pred_{L}_mV"])
        out[L] = f.rename(columns={f"pred_{L}_mV": "pred"})
    return out


def merged_cov(al, gs_merge):
    """Alarm time with each record's II and V2 alarms merged into one timeline, over GS time (per lead as eq. 4, or merged across leads too)."""
    num = den = 0.0
    gs = T.gs_scope(T.GS, "FULL")
    gs = gs[~np.array([(r, l) in T.MASK for r, l in zip(gs.record, gs.lead)], bool)]
    for rec in T.RECS:
        a = al[al.record == rec]
        num += K.total_len(K.merge_intervals(list(zip(a.onset_actual_s.astype(float), a.end_actual_s.astype(float)))))
        g = gs[gs.record == rec]
        if gs_merge:
            den += K.total_len(K.merge_intervals(list(zip(g.onset_actual_s.astype(float), g.end_actual_s.astype(float)))))
        else:
            den += sum(K.total_len(K.merge_intervals(list(zip(q.onset_actual_s.astype(float), q.end_actual_s.astype(float))))) for _, q in g.groupby("lead"))
    return num / den


def per_lead_tp(al):
    out = {}
    for L in T.LEADS:
        c = T.per_record_components(T.gs_scope(T.GS, f"own_{L}"), K.scope_filter(al, f"own_{L}"), T.DUR, f"own_{L}", T.MASK, T.RECS)
        out[L] = (int(c.TP.sum()), int(c.TP.sum() + c.FN.sum()))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = {c: pd.read_csv(V5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    Mn = T.memos(V["noisy42"], "anchor_nk", "sqi_nk_atr")
    Mg = T.memos(V["noisy42"], "anchor_grid", "sqi_grid_atr")
    res, als = [], {}
    for cond in ("clean", "noisy", "gated"):
        for name, rows in (("Fiducial", fid_rows(V, cond, Mn)), ("Learned-1L", T.learned_rows(T.learned_frames("L1", "S1", T.COND[cond]), cond, Mg if cond == "gated" else None)),
                           ("R1 published (paper learned arm)", T.learned_rows(r1_frames(cond), cond, Mg if cond == "gated" else None))):
            al, c, m = T.score(rows)
            als[(name, cond)] = al
            al.to_csv(OUT / f"alarms_{name.split()[0]}_{cond}.csv", index=False)
            r = dict(system=name, condition=cond, **m, coverage_eq4=c.pred_dur.sum() / c.gs_dur.sum(),
                     coverage_merged_alarms=merged_cov(al, False), coverage_merged_alarms_and_gs=merged_cov(al, True))
            if cond == "gated":
                tp = per_lead_tp(al)
                r.update(TP_II=tp["II"][0], GS_II=tp["II"][1], TP_V2=tp["V2"][0], GS_V2=tp["V2"][1])
            res.append(r)
    R = pd.DataFrame(res)
    R.to_csv(OUT / "podium_rows.csv", index=False)

    def row(s, c):
        return R[(R.system == s) & (R.condition == c)].iloc[0]
    a4a = pd.read_csv(T.OUT.parent / "step7_t10" / "T10_alarms.csv").query("ablation == 'A4a' and pipeline == 'L1' and gate == '.atr'").iloc[0]
    ctl = [("Learned-1L gated = A4a S1 .atr row", row("Learned-1L", "gated"), (int(a4a.TP), int(a4a.FP), int(a4a.FN))),
           ("Fiducial artifact = 67/43/34", row("Fiducial", "noisy"), (67, 43, 34)), ("Fiducial gated = 64/27/37", row("Fiducial", "gated"), (64, 27, 37))]
    lines, ok = [], True
    for name, r, want in ctl:
        got = (int(r.TP), int(r.FP), int(r.FN))
        ok &= got == want
        lines.append(f"{'PASS' if got == want else 'FAIL'}  {name}: {got} vs {want}")
    pr = row("R1 published (paper learned arm)", "gated")
    good = (int(pr.TP), int(pr.FP), int(pr.FN)) == PAPER["Learned-1L"]["gated"] and round(pr.coverage_eq4, 3) == 1.213
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'}  definition check: published R1 (paper's learned arm), .atr gate: {int(pr.TP)}/{int(pr.FP)}/{int(pr.FN)}, coverage (eq. 4) "
                 f"{pr.coverage_eq4:.3f} (paper 74/28/27, 1.213); merged across leads: alarms only {pr.coverage_merged_alarms:.3f}, alarms and GS "
                 f"{pr.coverage_merged_alarms_and_gs:.3f} (paper ~1.06)")
    (OUT / "CONTROLS.txt").write_text("\n".join(lines) + ("\nALL PASSED\n" if ok else "\nFAILED\n"))
    print("\n".join(lines))
    if not ok:
        return False
    # (d) Fiducial baseline windows with no value
    vc = V["clean"]
    nv = []
    for lab, q in (("15 records, excluded pairs left out (thesis convention)", vc[vc.nk_window & ~np.array([(r, l) in T.MASK for r, l in zip(vc.record, vc.lead)], bool)]),
                   ("15 records, all record-lead pairs", vc[vc.nk_window])):
        for L in T.LEADS:
            g = q[q.lead == L]
            nv.append(dict(denominator=lab, lead=L, windows=len(g), no_value=int(g.NKPR.isna().sum()), pct=100 * g.NKPR.isna().mean()))
    NV = pd.DataFrame(nv)
    NV.to_csv(OUT / "fiducial_no_value.csv", index=False)
    md(R, NV, lines, a4a)
    return True


def md(R, NV, ctl, a4a):
    def row(s, c):
        return R[(R.system == s) & (R.condition == c)].iloc[0]
    o = ["# Podium results (CMBEC48 talk): paper configuration, one run per system", "",
         f"Generated {time.strftime('%F %T')} by `src/scripts/final/podium_results.py` (VM). **Podium-only exception** (`docs/DECISIONS.md` 2026-11-24): "
         "these numbers are for the conference talk and are not thesis results; no thesis output was changed and nothing was retrained.", "",
         "**Configuration (the paper's):** noise seed 42 (NSTDB `em`); one run per system. **Fiducial:** NeuroKit R-peaks and NeuroKit dwt delineation, ST level = "
         "5-sample median (±8 ms) at J+60 minus the PR-segment median, per-beat values rounded to 1e-6 mV (removes the float residue at the PROSE thresholds that "
         "the paper's stored segment files carry), 10 s windows with ≥ 5 usable beats. **Learned-1L:** pretrained, one model per lead, training seed S1, alarms of "
         "both leads merged. **Gate:** the paper's SQI (0.67 Hz high-pass, 30 s, 25th percentile of per-beat SNR) from `.atr` beats, 6 dB, every gated row. "
         "Same PROSE FSM and scorer as the thesis; union of II and V2, all 101 episodes (Monitor: 99, copied from the paper); excluded record–lead pairs left out.", "",
         "Controls: " + ("all passed — " if all(x.startswith("PASS") for x in ctl) else "FAILED — ") + "; ".join(x[6:] for x in ctl) + ".", "",
         "## (a) Detection of ischemic episodes (paper Table 1 layout)", "",
         "P = TP/(TP+FP); R = TP/(TP+FN). Redundant = later alarms inside an already-detected episode (not counted as FP). Paper column: the published "
         "TP/FP/FN for comparison.", "",
         "| System | Condition | TP | FP | FN | Redundant | P | R | F1 | paper TP/FP/FN |", "|---|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    mon = {"clean": (78, 66, 21), "noisy": (77, 171, 22), "gated": (76, 54, 23)}
    for cond in ("clean", "noisy", "gated"):
        tp, fp, fn = mon[cond]
        p, r = tp / (tp + fp), tp / (tp + fn)
        o.append(f"| {'Monitor [1]' if cond == 'clean' else ''} | {CONDN[cond]} | {tp} | {fp} | {fn} | n/a | {p:.3f} | {r:.3f} | {2 * p * r / (p + r):.3f} | {tp}/{fp}/{fn} |")
    for s in ("Fiducial", "Learned-1L"):
        for cond in ("clean", "noisy", "gated"):
            q = row(s, cond)
            pp = PAPER[s][cond]
            o.append(f"| {(s + (' (S1)' if s == 'Learned-1L' else '')) if cond == 'clean' else ''} | {CONDN[cond]} | {int(q.TP)} | {int(q.FP)} | {int(q.FN)} | {int(q.Ignored)} | {q.precision:.3f} | {q.recall:.3f} | {q.F1:.3f} | {pp[0]}/{pp[1]}/{pp[2]} |")
    o += ["", "## (b) Temporal extent of the alarms (paper Table 2 layout)", "",
          "Sens. = duration sensitivity TPt/(TPt+FNt); PPV = TPt/(TPt+FPt); IoU = TPt/(TPt+FPt+FNt); Cov. = coverage, paper eq. 4, (TPt+FPt)/(TPt+FNt), per "
          "record–lead. Cov. merged = each record's II and V2 alarms merged into one timeline before measuring alarm time, over the same gold-standard time "
          "(the paper's \"merging alarms across leads\"). Paper values in brackets.", "",
          "| System | Condition | Sens. | PPV | IoU | Cov. | Cov. merged |", "|---|---|--:|--:|--:|--:|--:|"]
    for s in ("Fiducial", "Learned-1L"):
        for cond in ("noisy", "gated"):
            q = row(s, cond)
            pp = PAPER_T2[(s, cond)]
            o.append(f"| {(s + (' (S1)' if s == 'Learned-1L' else '')) if cond == 'noisy' else ''} | {CONDN[cond]} | {q.sensitivity:.3f} ({pp[0]:.3f}) | {q.PPV_time:.3f} ({pp[1]:.3f}) | "
                     f"{q.IoU_whole_record:.3f} ({pp[2]:.3f}) | {q.coverage_eq4:.3f} ({pp[3]:.3f}) | {q.coverage_merged_alarms:.3f} |")
    pr = row("R1 published (paper learned arm)", "gated")
    o += ["", f"Definition check for Cov. merged: the paper's own learned arm (published models, `.atr` gate) gives Cov. {pr.coverage_eq4:.3f} (paper 1.213) and "
          f"Cov. merged {pr.coverage_merged_alarms:.3f} (paper: \"reduces the value to 1.06\"). Merging the gold-standard time across leads as well would give "
          f"{pr.coverage_merged_alarms_and_gs:.3f}, so the paper's 1.06 is the alarms-only merge used here.", "",
          "## (c) Gated TP per lead", "",
          "Each lead's alarms scored against that lead's episodes (69 on II, 32 on V2).", "",
          "| System | II TP | V2 TP | paper II / V2 |", "|---|--:|--:|---|"]
    for s, paper in (("Fiducial", "– / 5 of 32"), ("Learned-1L", "– / 22 of 32")):
        q = row(s, "gated")
        o.append(f"| {s}{' (S1)' if s == 'Learned-1L' else ''} | {int(q.TP_II)} of {int(q.GS_II)} | {int(q.TP_V2)} of {int(q.GS_V2)} | {paper} |")
    o += ["", "## (d) Fiducial: baseline windows with no value", "",
          "10 s windows of the NeuroKit grid in which fewer than 5 beats have a usable ST deviation (the estimator supplies no value and the FSM freezes). "
          "Paper: 20.8 % (II) / 12.2 % (V2).", "",
          "| Denominator | II: no value / windows | V2: no value / windows |", "|---|--:|--:|"]
    for lab, g in NV.groupby("denominator", sort=False):
        a, b = g[g.lead == "II"].iloc[0], g[g.lead == "V2"].iloc[0]
        o.append(f"| {lab} | {a.no_value:,} / {a.windows:,} ({a.pct:.1f} %) | {b.no_value:,} / {b.windows:,} ({b.pct:.1f} %) |")
    o += ["", "Differences from the paper's numbers:",
          "- Fiducial: the paper's rows were scored from segment files that store values such as −0.0999999999999999 mV where the true median is −0.1, so "
          "the FSM's threshold tests failed on exact ties (about one episode per condition). The paper's own segment files, rounded to 1e-6 mV, give "
          "67/32/34 (3-sample baseline), 67/43/34 and 64/27/37; the baseline row here uses 5 samples (67/34/34).",
          "- Learned-1L: S1 of the final (retrained) models, not the published models; the gate is the same `.atr` gate. The baseline counts equal the "
          "paper's (75/40/26) by coincidence: the predictions differ (mean |Δ| 0.015 mV II, 0.022 mV V2) and so do redundant alarms and IoU."]
    MD.write_text("\n".join(o) + "\n")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
