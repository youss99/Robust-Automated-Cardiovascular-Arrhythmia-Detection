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
Output: Tasks/final/runs/step7_step3e/ (CSVs, SUMMARY.md, CONTROL.txt, DONE_step3e). RESULTS_FINAL.md is not written.
Stage tex (user, 2026-11-24): docs/thesis_tables/T11a_supervised_schedule.tex (per system) and T11b_supervised_schedule_paired.tex (bootstrap),
from the saved CSVs only (no recomputation), with the thesis names written directly (relabel leaves them unchanged).
Usage: <repo .venv>/bin/python -u -W ignore fin_step7_step3e.py [tex]"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fin_step7_tier1 as T  # noqa: E402

OUT = Path("/home/student/GIT/Tasks/final/runs/step7_step3e")
STORED = Path("/home/student/GIT/Tasks/final/runs/_archive_2026-11-24_supervised_20plus70/step7")  # tier 1 as it was when step3e was scored (L1sup / L2sup rows)
SAVE = Path(__file__).resolve().parents[3] / "results/fine-tune/saved_models/patient-segment-regression"
E2E = {"L1supE2E": [("lead_II", "II"), ("lead_V2", "V2")], "L2supE2E": [("two_lead", "II"), ("two_lead", "V2")]}
OLD = ("L1", "L2", "L1sup", "L2sup")
PAIRS_CTL = [("L1", "L1sup"), ("L2", "L2sup")]
PAIRS = [("L1supE2E", "L1sup"), ("L2supE2E", "L2sup"), ("L1", "L1supE2E"), ("L2", "L2supE2E"), ("L1", "L1sup"), ("L2", "L2sup")]
T.OUT = OUT
T.LEARNED.update(E2E)
T.LEARNED.update({"L1sup": [("lead_II", "II"), ("lead_V2", "V2")], "L2sup": [("two_lead", "II"), ("two_lead", "V2")]})  # archived arm (DECISIONS.md 2026-11-24)


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


# ------------------------------------------------------------------ LaTeX (stage tex)
NAME = {"L1": "Learned-1L (pretrained)", "L1sup": "Learned-1L (supervised-only)", "L1supE2E": "Learned-1L (supervised-only, end-to-end)",
        "L2": "Learned-2L (pretrained)", "L2sup": "Learned-2L (supervised-only)", "L2supE2E": "Learned-2L (supervised-only, end-to-end)"}
SCHED = {"L1": "20 + 70", "L1sup": "20 + 70", "L1supE2E": "0 + 90", "L2": "20 + 70", "L2sup": "20 + 70", "L2supE2E": "0 + 90"}


def tpm(x, d):
    x = np.asarray(x, float)
    f = (lambda v: f"{v:.{d}f}".replace("-", "$-$"))
    return f"{f(x.mean())} $\\pm$ {x.std(ddof=1):.{d}f}"


def tex_write(name, caption, label, colspec, header, body, note):
    lines = [f"% {name}: generated {time.strftime('%F %T')} by src/scripts/final/fin_step7_step3e.py (VM). Needs \\usepackage{{booktabs}}.",
             "\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += ["\\midrule" if row == "MIDRULE" else row + " \\\\" for row in body]
    lines += ["\\bottomrule", "\\end{tabular}", "", f"\\par\\smallskip\\footnotesize {note}", "\\end{table}", ""]
    (T.TEX / f"{name}.tex").write_text("\n".join(lines))


def tex():
    if not (OUT / "DONE_step3e").exists():
        sys.exit("run the scoring first")
    T1t, R = pd.read_csv(OUT / "T1_regression_per_seed.csv"), pd.read_csv(OUT / "T5_alarms_per_seed.csv")
    T7t, TR = pd.read_csv(OUT / "T7_paired_bootstrap.csv"), pd.read_csv(OUT / "training_best_epochs.csv")
    body = []
    for i, group in enumerate((("L1", "L1sup", "L1supE2E"), ("L2", "L2sup", "L2supE2E"))):
        if i:
            body.append("MIDRULE")
        for p in group:
            mae = [T1t[(T1t.pipeline == p) & (T1t.lead == L)].MAE for L in T.LEADS]
            f1 = [R[(R.pipeline == p) & (R.condition == c)].F1 for c in ("clean", "noisy", "gated")]
            fp = R[(R.pipeline == p) & (R.condition == "gated")].FP
            modes = [m for m, _ in dict.fromkeys(T.LEARNED[p])]
            ep = [int(re.search(r"best_row_epoch: (\d+)", (SAVE / f"FIN_{p}_{m}_{s}_fold{k:02d}" / "job_log.txt").read_text()).group(1))
                  for m in modes for s in T.SEEDS for k in range(1, 6)]
            body.append(f"{NAME[p]} & {SCHED[p]} & {tpm(mae[0], 4)} & {tpm(mae[1], 4)} & {tpm(f1[0], 3)} & {tpm(f1[1], 3)} & {tpm(f1[2], 3)} & {tpm(fp, 1)} & "
                        f"{int(np.median(ep))}")
    tex_write("T11a_supervised_schedule", "Supervised-only training schedule: head-only phase vs end-to-end (T11a).", "tab:t11a_schedule",
              "lcrrrrrrr", "System & Epochs & MAE II (mV) & MAE V2 (mV) & F1 baseline & F1 artifact & F1 gated & FP gated & Best epoch", body,
              "Epochs: head-only (frozen encoder) + full-network fine-tuning epochs. Supervised-only, end-to-end: the same runs as supervised-only (data, folds, "
              "seeds, initial weights, optimizer) except 0 + 90 epochs. MAE: baseline condition, all 10\\,s windows (as Table T1). F1 and FP: union of lead II and V2 "
              "alarms, 101 ischemic episodes in 15 records (as Table T5); artifact: NSTDB em, seed 42; gated: SQI from rodrigues2021 R-peaks, 6\\,dB. "
              "Mean $\\pm$ SD over three training seeds. Best epoch: epoch of the lowest validation loss (0--89, head-only epochs included), median over the runs (single-lead systems: 30, lead II and V2 models; two-lead systems: 15).")
    body = []
    for t in T7t.itertuples():
        d = 4 if "MAE" in t.metric else 3 if "F1" in t.metric else 1
        f = (lambda v, s="": f"{v:{s}.{d}f}".replace("-", "$-$"))
        m = t.metric.replace("dF1", "$\\Delta$F1").replace("dFP", "$\\Delta$FP").replace("dMAE", "$\\Delta$MAE")
        a, b = t.comparison.split(" - ")
        body.append(f"{NAME[a]} $-$ {NAME[b]} & {m} & {f(t.value_a)} & {f(t.value_b)} & {f(t.delta, '+')}{'$^{*}$' if t.excludes_zero else ''} & [{f(t.ci_lo, '+')}, {f(t.ci_hi, '+')}]")
    tex_write("T11b_supervised_schedule_paired", "Paired differences for the training-schedule comparison, gated, noise seed 42 (T11b).", "tab:t11b_schedule_paired",
              "llrrrl", "Comparison & Metric & A & B & A $-$ B & 95\\,\\% CI", body,
              f"Record-clustered paired bootstrap, {T.NB:,} resamples; training seeds averaged inside each resample (as Table T7; the last two comparisons "
              "repeat Table T7). $\\Delta$MAE: windows the gate keeps, predictions in the artifact condition against baseline labels, excluded record--lead pairs "
              "not scored. $^{*}$: 95\\,\\% CI excludes zero.")
    print("wrote", T.TEX / "T11a_supervised_schedule.tex", T.TEX / "T11b_supervised_schedule_paired.tex")
    return True


if __name__ == "__main__":
    sys.exit(0 if (tex() if sys.argv[1:] == ["tex"] else main()) else 1)
