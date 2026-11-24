#!/usr/bin/env python3
"""step3e scoring (DECISIONS.md 2026-11-24: comparisons and metrics fixed before any step3e result).

Pipelines: L1supE2E (lead II model on II + V2 model on V2, union) and L2supE2E (two-lead model), supervised, 0 head-only + 90 full epochs,
S1-S3; next to their twins L1sup / L2sup (20 + 70) and the pretrained L1 / L2. Everything goes through fin_step7_tier1's own functions
(alarm path, gate, scorer, set A labels, regression metrics); only the pairs in the bootstrap are new.
  T1  clean, set A windows, per lead: MAE, bias, RMSE, slope, r, p100, R2 (mean +- SD over S1-S3).
  T5  union, FULL scope, clean / noisy (seed 42) / gated (seed 42, 6 dB, rodrigues2021): TP, FP, FN, precision, recall, F1, false-alarm hours.
  T7  gated, seed 42, record-clustered paired bootstrap (5,000 resamples, seed 20260805, training seeds averaged inside each resample):
      (a) L1supE2E - L1sup, L2supE2E - L2sup; (b) L1 - L1supE2E, L2 - L2supE2E, next to L1 - L1sup, L2 - L2sup. dF1, dFP, dMAE II, dMAE V2.
  Training: best epoch and best validation loss per run (job_log.txt).
Controls first (stop the run): this script's path reproduces the stored tier-1 results of the existing pipelines exactly: T5 rows of L1, L2,
L1sup, L2sup (every seed and condition), their T1 rows, and the T7 rows L1 - L1sup and L2 - L2sup.
Output: Tasks/final/runs/step7_step3e/ (CSVs, SUMMARY.md, CONTROL.txt, DONE_step3e). RESULTS_FINAL.md and docs/thesis_tables are not written.
Usage: <repo .venv>/bin/python -u -W ignore fin_step7_step3e.py"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fin_step7_tier1 as T  # noqa: E402

OUT = Path("/home/student/GIT/Tasks/final/runs/step7_step3e")
STORED = Path("/home/student/GIT/Tasks/final/runs/step7")
SAVE = Path(__file__).resolve().parents[3] / "results/fine-tune/saved_models/patient-segment-regression"
E2E = {"L1supE2E": [("lead_II", "II"), ("lead_V2", "V2")], "L2supE2E": [("two_lead", "II"), ("two_lead", "V2")]}
OLD = ("L1", "L2", "L1sup", "L2sup")
PAIRS_CTL = [("L1", "L1sup"), ("L2", "L2sup")]
PAIRS = [("L1supE2E", "L1sup"), ("L2supE2E", "L2sup"), ("L1", "L1supE2E"), ("L2", "L2supE2E"), ("L1", "L1sup"), ("L2", "L2sup")]
T.OUT = OUT
T.LEARNED.update(E2E)


def t7(C, Fg, pairs):
    """fin_step7_tier1.t7 with the pairs as an argument (same code otherwise)."""
    idx15 = np.random.default_rng(T.BSEED).integers(0, len(T.RECS), size=(T.NB, len(T.RECS)))
    cg = C[C.condition == "gated"]

    def acomp(pipe):
        return [cg[(cg.pipeline == pipe) & (cg.seed == s)].set_index("record").loc[T.RECS][["TP", "FP", "FN"]].to_numpy(float) for s in T.SEEDS]

    def f1(s):
        return 2 * s[..., 0] / (2 * s[..., 0] + s[..., 1] + s[..., 2])

    rows = []
    for a, b in pairs:
        A, B_ = acomp(a), acomp(b)
        full = {"F1": (np.mean([f1(x.sum(0)) for x in A]), np.mean([f1(x.sum(0)) for x in B_])), "FP": (np.mean([x.sum(0)[1] for x in A]), np.mean([x.sum(0)[1] for x in B_]))}
        bootA = {"F1": np.mean([f1(x[idx15].sum(1)) for x in A], 0), "FP": np.mean([x[idx15].sum(1)[:, 1] for x in A], 0)}
        bootB = {"F1": np.mean([f1(x[idx15].sum(1)) for x in B_], 0), "FP": np.mean([x[idx15].sum(1)[:, 1] for x in B_], 0)}
        for m in ("F1", "FP"):
            d = bootA[m] - bootB[m]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(dict(comparison=f"{a} - {b}", metric=f"d{m} (union)", value_a=float(full[m][0]), value_b=float(full[m][1]), delta=float(full[m][0] - full[m][1]), ci_lo=float(lo), ci_hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0), n_records=len(T.RECS)))
        for L in T.LEADS:
            recs = [r for r in T.RECS if (r, L) not in T.MASK]
            idx = np.random.default_rng(T.BSEED).integers(0, len(recs), size=(T.NB, len(recs)))
            SA = [T.reg_sums(Fg[(a, s, L)], recs) for s in T.SEEDS]
            SB = [T.reg_sums(Fg[(b, s, L)], recs) for s in T.SEEDS]
            fa = np.mean([x.sum(0)[1] / x.sum(0)[0] for x in SA])
            fb = np.mean([x.sum(0)[1] / x.sum(0)[0] for x in SB])
            ba = np.mean([x[idx].sum(1)[:, 1] / x[idx].sum(1)[:, 0] for x in SA], 0)
            bb = np.mean([x[idx].sum(1)[:, 1] / x[idx].sum(1)[:, 0] for x in SB], 0)
            lo, hi = np.nanpercentile(ba - bb, [2.5, 97.5])
            rows.append(dict(comparison=f"{a} - {b}", metric=f"dMAE {L} (mV)", value_a=float(fa), value_b=float(fb), delta=float(fa - fb), ci_lo=float(lo), ci_hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0), n_records=len(recs)))
    return pd.DataFrame(rows)


def alarms(V, pipes):
    Mg = T.memos(V["noisy42"], "anchor_grid", "sqi_grid")
    res, comps = [], []
    for cond in ("clean", "noisy", "gated"):
        for pipe in pipes:
            for s in T.SEEDS:
                al, c, met = T.score(T.learned_rows(T.learned_frames(pipe, s, T.COND[cond]), cond, Mg if cond == "gated" else None))
                al.to_csv(OUT / "alarms" / f"{pipe}_{s}_{cond}.csv", index=False)
                res.append(dict(pipeline=pipe, seed=s, condition=cond, **met))
                comps.append(c.assign(pipeline=pipe, seed=s, condition=cond))
                T.log("alarms", pipe, s, cond, met["TP"], met["FP"], met["FN"])
    return pd.DataFrame(res), pd.concat(comps, ignore_index=True)


def same(a, b, keys, cols):
    m = a.merge(b, on=keys, suffixes=("", "_ref"))
    return len(m) == len(a) == len(b) and all(np.allclose(m[c].astype(float), m[c + "_ref"].astype(float), rtol=0, atol=1e-12, equal_nan=True) for c in cols)


def training():
    rows = []
    for pipe, twin in (("L1supE2E", "L1sup"), ("L2supE2E", "L2sup")):
        for mode, _ in dict.fromkeys(E2E[pipe]):
            for s in T.SEEDS:
                for k in range(1, 6):
                    for p in (pipe, twin):
                        rid = f"FIN_{p}_{mode}_{s}_fold{k:02d}"
                        txt = (SAVE / rid / "job_log.txt").read_text()
                        rows.append(dict(pipeline=p, mode=mode, seed=s, fold=k, best_epoch=int(re.search(r"best_row_epoch: (\d+)", txt).group(1)),
                                         best_valid_loss=float(re.search(r"best_valid_loss: (\S+)", txt).group(1))))
    return pd.DataFrame(rows)


def pm(x, d):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}"


def main():
    (OUT / "alarms").mkdir(parents=True, exist_ok=True)
    for m in ("DONE_step3e", "FAILED_step3e"):
        (OUT / m).unlink(missing_ok=True)
    for need in (T.S6 / "DONE_step6_step3e", T.S6 / "DONE_step6", T.S5 / "DONE_step5", STORED / "DONE_tier1"):
        if not need.exists():
            sys.exit(f"missing {need}")
    V = {c: pd.read_csv(T.S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    Lb = T.labels()
    # ---- controls: the existing pipelines through this path equal the stored tier-1 results
    R_old, C_old = alarms(V, OLD)
    R_ref = pd.read_csv(STORED / "T5_alarms_per_seed.csv").query("pipeline in @OLD")
    mcols = [c for c in R_old.columns if c not in ("pipeline", "seed", "condition")]
    ctl = [("T5 rows of L1, L2, L1sup, L2sup (every seed and condition) equal the stored tier-1 rows", same(R_old, R_ref, ["pipeline", "seed", "condition"], mcols))]
    F = T.reg_frames(V, Lb, "clean")
    T1t = T.t1(F)
    T1_ref = pd.read_csv(STORED / "T1_regression_per_seed.csv").query("pipeline in @OLD")
    rcols = [c for c in T1_ref.columns if c not in ("pipeline", "seed", "lead")]
    ctl.append(("T1 rows of L1, L2, L1sup, L2sup equal the stored tier-1 rows", same(T1t[T1t.pipeline.isin(OLD)], T1_ref, ["pipeline", "seed", "lead"], rcols)))
    Fg = T.reg_frames(V, Lb, "gated", gated=True)
    R_new, C_new = alarms(V, tuple(E2E))
    C = pd.concat([C_old, C_new], ignore_index=True)
    T7c = t7(C, Fg, PAIRS_CTL)
    T7_ref = pd.read_csv(STORED / "T7_paired_bootstrap.csv")
    T7_ref = T7_ref[T7_ref.comparison.isin([f"{a} - {b}" for a, b in PAIRS_CTL])]
    ctl.append(("T7 rows L1 - L1sup and L2 - L2sup equal the stored tier-1 rows", same(T7c, T7_ref, ["comparison", "metric"], ["value_a", "value_b", "delta", "ci_lo", "ci_hi"])))
    lines = [f"{'PASS' if g else 'FAIL'}  {n}" for n, g in ctl]
    (OUT / "CONTROL.txt").write_text("\n".join(lines) + "\n")
    T.log("control", *lines)
    if not all(g for _, g in ctl):
        (OUT / "FAILED_step3e").write_text("control failed\n")
        return False
    # ---- step3e results
    R = pd.concat([R_old, R_new], ignore_index=True)
    R.to_csv(OUT / "T5_alarms_per_seed.csv", index=False)
    C.to_csv(OUT / "alarm_components_per_record.csv", index=False)
    T1t.to_csv(OUT / "T1_regression_per_seed.csv", index=False)
    T7t = t7(C, Fg, PAIRS)
    T7t.to_csv(OUT / "T7_paired_bootstrap.csv", index=False)
    TR = training()
    TR.to_csv(OUT / "training_best_epochs.csv", index=False)
    order = ["L1", "L1sup", "L1supE2E", "L2", "L2sup", "L2supE2E"]
    md = ["# step3e scoring", "", f"{time.strftime('%F %T')}. Controls: " + "; ".join(lines), "", "## T1 (clean, set A)", "",
          "| lead | pipeline | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | R² |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for L in T.LEADS:
        for p in order:
            q = T1t[(T1t.pipeline == p) & (T1t.lead == L)]
            md.append(f"| {L} | {p} | {pm(q.MAE, 4)} | {pm(q.bias, 4)} | {pm(q.RMSE, 4)} | {pm(q.slope, 3)} | {pm(q.r, 3)} | {pm(q.R2, 3)} |")
    md += ["", "## T5 (union, FULL)", "", "| condition | pipeline | TP | FP | FN | precision | recall | F1 | false-alarm h |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for cond in ("clean", "noisy", "gated"):
        for p in order:
            q = R[(R.pipeline == p) & (R.condition == cond)]
            md.append(f"| {cond} | {p} | {pm(q.TP, 1)} | {pm(q.FP, 1)} | {pm(q.FN, 1)} | {pm(q.precision, 3)} | {pm(q.recall, 3)} | {pm(q.F1, 3)} | {pm(q.false_hours, 1)} |")
    md += ["", "## T7 (gated, seed 42; * = 95 % CI excludes 0)", "", "| comparison | metric | A | B | A − B | 95 % CI |", "|---|---|--:|--:|--:|---|"]
    for r in T7t.itertuples():
        d = 4 if "MAE" in r.metric else (1 if "FP" in r.metric else 3)
        md.append(f"| {r.comparison} | {r.metric} | {r.value_a:.{d}f} | {r.value_b:.{d}f} | {r.delta:+.{d}f}{'*' if r.excludes_zero else ''} | [{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}] |")
    md += ["", "## Training (job_log.txt)", "", "| pipeline | mode | best epoch median (range) | best valid loss median |", "|---|---|---|--:|"]
    for (p, m), q in TR.groupby(["pipeline", "mode"], sort=False):
        md.append(f"| {p} | {m} | {int(q.best_epoch.median())} ({q.best_epoch.min()}–{q.best_epoch.max()}) | {q.best_valid_loss.median():.5f} |")
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (OUT / "DONE_step3e").write_text(time.strftime("%F %T") + "\n")
    T.log("done")
    print("\n".join(md))
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
