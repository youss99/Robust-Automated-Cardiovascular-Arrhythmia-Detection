#!/usr/bin/env python3
"""Final runs, step 7, remaining tables (user, 2026-11-22: T2, T3, T4, T6 in that order; T8, T9 skipped). One table per call.

Same labels, predictions, NeuroKit values, gate tables, alarm path and scorer as fin_step7_tier1.py (imported, unchanged).
Pipelines L1, L2 (mean +- SD over S1-S3 unless stated), NK-PR (NK-FP dropped, DECISIONS.md 2026-11-24); masked pairs excluded; 15 records.
T2  clean, DMY beats: label = DMY ee; value = the 10 s window containing the beat (k = sample_16a // 2500; C section 9, phase 1's set B).
    T1's metrics + coverage (share of DMY beats with a value).
T3  clean, set A windows, by |true ST| band: < 0.05, 0.05-0.10, 0.10-0.20, >= 0.20 mV (lower bound inclusive). n, MAE, bias (p - y), and the
    shrinkage toward zero = mean of -sign(y) (p - y) (positive = the estimate is closer to 0 than the label).
T4  noisy (NSTDB em seed 42) against the clean labels, set A, ungated (all windows) and gated (SQI >= 6 dB from rodrigues2021 peaks at the learned
    grid anchor, the same gate for every pipeline, as T7). MAE overall and inside noise bursts (windows entirely inside a burst of the seed-42
    contamination log, burst_frac = 1), with n.
T6  gated alarms, noise seeds 42, 43, 44 (each seed with its own noisy signal, predictions and gate): NK-PR, NK-FP, L1 (S1), L2 (S1).
    Mean +- SD over the three noise seeds of TP, FP, FN, F1, false-alarm hours. Check: seed 42 equals the tier-1 gated rows.
Output: Tasks/final/runs/step7_rest/, docs/RESULTS_FINAL.md (block per table), docs/thesis_tables/T{2,3,4,6}_*.tex.
Usage: <repo .venv>/bin/python -u -W ignore fin_step7_rest.py T2|T3|T4|T6"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other final drivers
import fin_step7_tier1 as T  # noqa: E402
import fin_step7_t10_fig1 as X  # noqa: E402  (upsert, tex_write)

P = T.P
OUT = Path("/home/student/GIT/Tasks/final/runs/step7_rest")
LEADS, MASK, WIN, SEEDS = T.LEADS, T.MASK, T.WIN, T.SEEDS
PIPES = ("L1", "L2", "NK-PR")  # NK-FP dropped from the thesis (DECISIONS.md 2026-11-24)
BANDS = [("< 0.05", 0.0, 0.05), ("0.05–0.10", 0.05, 0.10), ("0.10–0.20", 0.10, 0.20), ("≥ 0.20", 0.20, np.inf)]
X.tex_write.__globals__["time"] = time
SCRIPT = "src/scripts/final/fin_step7_rest.py"


def tex_write(name, caption, label, colspec, header, body, note):
    X.tex_write(name, caption, label, colspec, header, body, note)
    f = T.TEX / f"{name}.tex"
    f.write_text(f.read_text().replace("fin_step7_t10_fig1.py", "fin_step7_rest.py", 1))


def seeds(pipe):
    return SEEDS if pipe in ("L1", "L2") else ("-",)


def pmx(q, col, d):
    return T.pm(q[col], d)


def tpmx(q, col, d):
    return T.tpm(q[col], d)


def head(title, text):
    return [title, "", f"Generated {time.strftime('%F %T')} by `{SCRIPT}` (VM). " + text, ""]


# ------------------------------------------------------------------ T2
def t2():
    D = P.dmy_table()
    D = D[~np.array([(r, l) in MASK for r, l in zip(D.record, D.lead)], bool)].copy()
    D["k"] = D.sample_16a // WIN
    Vc = pd.read_csv(T.S5 / "nk_values_clean.csv.gz", usecols=["record", "lead", "k", "nk_window", "NKPR", "NKFP"])
    rows = []
    for L in LEADS:
        q = D[D.lead == L][["record", "k", "dmy_ee_mV"]].rename(columns={"dmy_ee_mV": "y"})
        for pipe in PIPES:
            for s in seeds(pipe):
                if pipe in T.NK:
                    v = Vc[(Vc.lead == L) & Vc.nk_window][["record", "k", T.NK[pipe]]].rename(columns={T.NK[pipe]: "p"})
                else:
                    mode = dict((l, m) for m, l in T.LEARNED[pipe])[L]
                    v = pd.read_csv(T.S6 / "predictions_clean" / f"FIN_{pipe}_{mode}_{s}__grid.csv.gz", usecols=["record", "anchor_samp", f"pred_{L}_mV"])
                    v["k"] = (v.anchor_samp - (WIN - 1)) // WIN
                    v = v[["record", "k", f"pred_{L}_mV"]].rename(columns={f"pred_{L}_mV": "p"})
                m = q.merge(v, on=["record", "k"], how="left", validate="m:1")
                rows.append(dict(pipeline=pipe, seed=s, lead=L, beats=len(m), with_value=int(m.p.notna().sum()), coverage_pct=100 * m.p.notna().mean(), **P.reg(m.y, m.p)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "T2_regression_dmy_per_seed.csv", index=False)
    out = head("## T2. Regression, DMY beats (clean)", "DMY beats (label = the beat's expert ST deviation); estimate = the 10 s window containing the beat (C §9, phase 1's set B); "
               "masked pairs excluded; T1's metrics; coverage = share of DMY beats with a value. L1, L2: mean ± SD over S1–S3.")
    out += ["| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for pipe in PIPES:
            q = R[(R.pipeline == pipe) & (R.lead == L)]
            out.append(f"| {L} | {pipe} | {pmx(q, 'coverage_pct', 1)} | {pmx(q, 'n', 0)} | {pmx(q, 'MAE', 4)} | {pmx(q, 'bias', 4)} | {pmx(q, 'RMSE', 4)} | {pmx(q, 'slope', 3)} | {pmx(q, 'r', 3)} | {pmx(q, 'p100_pct', 1)} | {pmx(q, 'R2', 3)} |")
            body.append(f"{L} & {pipe} & {tpmx(q, 'coverage_pct', 1)} & {tpmx(q, 'MAE', 4)} & {tpmx(q, 'bias', 4)} & {tpmx(q, 'RMSE', 4)} & {tpmx(q, 'slope', 3)} & {tpmx(q, 'r', 3)} & {tpmx(q, 'p100_pct', 1)} & {tpmx(q, 'R2', 3)}")
    out.append(f"\nDMY beats per lead (masked pairs out): II {int(R[R.lead == 'II'].beats.iloc[0]):,}, V2 {int(R[R.lead == 'V2'].beats.iloc[0]):,}.")
    X.upsert("t2", "\n".join(out))
    tex_write("T2_regression_dmy", "ST-deviation regression on DMY beats, clean signal (T2).", "tab:t2_regression_dmy",
              "llrrrrrrrr", "Lead & Pipeline & Coverage (\\%) & MAE (mV) & Bias (mV) & RMSE (mV) & Slope & $r$ & p100 (\\%) & $R^2$", body,
              "Label: expert ST deviation of each DMY beat; estimate: the 10\\,s window containing the beat. Masked record--lead pairs excluded; 15 records. "
              "Slope: OLS of estimate on label. p100: share with $|$error$|$ > 0.1\\,mV. L1, L2: mean $\\pm$ SD over three training seeds.")
    return R


# ------------------------------------------------------------------ T3
def t3():
    V = {c: pd.read_csv(T.S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    F = T.reg_frames(V, T.labels(), "clean")
    rows = []
    for (pipe, s, L), f in F.items():
        if pipe not in PIPES:
            continue
        q = f[f.y.notna()]
        ay = q.y.abs()
        for name, lo, hi in BANDS:
            b = q[(ay >= lo) & (ay < hi)]
            g = b[b.p.notna()]
            e = g.p - g.y
            rows.append(dict(pipeline=pipe, seed=s, lead=L, band=name, windows=len(b), pct_of_lead=100 * len(b) / len(q), n=len(g),
                             MAE=float(e.abs().mean()) if len(g) else np.nan, bias=float(e.mean()) if len(g) else np.nan,
                             shrink=float((-np.sign(g.y) * e).mean()) if len(g) else np.nan))
    R = pd.DataFrame(rows)
    # partition check: bands cover every labelled window exactly once
    chk = R.groupby(["pipeline", "seed", "lead"]).windows.sum().reset_index()
    tot = {L: int(next(f for (p, s, l), f in F.items() if l == L).y.notna().sum()) for L in LEADS}
    ok = all(int(r.windows) == tot[r.lead] for r in chk.itertuples())
    R.to_csv(OUT / "T3_bands_per_seed.csv", index=False)
    out = head("## T3. Error by true ST band (clean)", "Set A windows (as T1), masked pairs excluded, banded by |label| (lower bound inclusive). "
               "Bias = estimate − label; shrinkage = mean of −sign(label) × (estimate − label), positive when the estimate is closer to 0 than the label. "
               f"Bands partition the labelled windows: {'yes' if ok else 'NO'}. L1, L2: mean ± SD over S1–S3.")
    out += ["| lead | \\|ST\\| band (mV) | windows (% of lead) | pipeline | n with value | MAE (mV) | bias (mV) | shrinkage (mV) |", "|---|---|--:|---|--:|--:|--:|--:|"]
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for name, _, _ in BANDS:
            for j, pipe in enumerate(PIPES):
                q = R[(R.pipeline == pipe) & (R.lead == L) & (R.band == name)]
                w = q.iloc[0]
                lab = f"{int(w.windows):,} ({w.pct_of_lead:.1f} %)" if j == 0 else ""
                out.append(f"| {L if j == 0 else ''} | {name if j == 0 else ''} | {lab} | {pipe} | {pmx(q, 'n', 0)} | {pmx(q, 'MAE', 4)} | {pmx(q, 'bias', 4)} | {pmx(q, 'shrink', 4)} |")
                tname = name.replace("< ", "$<$\\,").replace("≥ ", "$\\geq$\\,").replace("–", "--")
                body.append(f"{L if j == 0 else ''} & {tname if j == 0 else ''} & {(f'{int(w.windows):,}'.replace(',', '{,}') + f' ({w.pct_of_lead:.1f})') if j == 0 else ''} & {pipe} & {tpmx(q, 'MAE', 4)} & {tpmx(q, 'bias', 4)} & {tpmx(q, 'shrink', 4)}")
    X.upsert("t3", "\n".join(out))
    tex_write("T3_error_by_band", "ST-deviation error by true ST band, clean signal (T3).", "tab:t3_bands",
              "llrlrrr", "Lead & $|$ST$|$ (mV) & Windows (\\%) & Pipeline & MAE (mV) & Bias (mV) & Shrinkage (mV)", body,
              "10\\,s windows banded by the absolute expert ST deviation (lower bound inclusive); masked pairs excluded. Bias: estimate $-$ label. "
              "Shrinkage: mean of $-$sign(label)\\,$\\times$\\,(estimate $-$ label), positive when the estimate is closer to zero than the label. "
              "NK-PR has no value on some windows (T1 coverage). L1, L2: mean $\\pm$ SD over three training seeds.")
    return R, ok


# ------------------------------------------------------------------ T4
def t4():
    V = {c: pd.read_csv(T.S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    F = T.reg_frames(V, T.labels(), "noisy")
    bf = V["noisy42"][["record", "lead", "k", "burst_frac"]]
    rows = []
    for (pipe, s, L), f in F.items():
        if pipe not in PIPES:
            continue
        f = f.merge(bf[bf.lead == L].drop(columns="lead"), on=["record", "k"], how="left", validate="1:1")
        q = f[f.y.notna()]
        burst = q.burst_frac.fillna(0) >= 1.0
        for gname, gm in (("ungated", np.ones(len(q), bool)), ("gated", q.gate.to_numpy(bool))):
            for scope, sm in (("all", np.ones(len(q), bool)), ("burst", burst.to_numpy(bool))):
                sel = q[gm & sm]
                g = sel[sel.p.notna()]
                rows.append(dict(pipeline=pipe, seed=s, lead=L, gate=gname, scope=scope, windows=len(sel), n=len(g), coverage_pct=100 * len(g) / max(len(sel), 1),
                                 MAE=float((g.p - g.y).abs().mean()) if len(g) else np.nan, bias=float((g.p - g.y).mean()) if len(g) else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "T4_noise_per_seed.csv", index=False)
    out = head("## T4. Noise robustness (NSTDB `em`, seed 42)", "Noisy estimates against the clean labels, set A windows, masked pairs excluded. Gated = windows the 6 dB SQI gate "
               "(rodrigues2021 peaks, at the learned grid anchor) keeps, the same windows for every pipeline (as T7). Bursts = windows entirely inside a burst of the "
               "seed-42 contamination log. n = windows with a value. L1, L2: mean ± SD over S1–S3.")
    out += ["| lead | pipeline | ungated MAE (mV) | n | ungated burst MAE (mV) | n | gated MAE (mV) | n | gated burst MAE (mV) | n |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for pipe in PIPES:
            cells, tcells = [], []
            for gname in ("ungated", "gated"):
                for scope in ("all", "burst"):
                    q = R[(R.pipeline == pipe) & (R.lead == L) & (R.gate == gname) & (R.scope == scope)]
                    cells += [pmx(q, "MAE", 4), f"{int(q.n.mean()):,}"]
                    tcells += [tpmx(q, "MAE", 4), f"{int(q.n.mean()):,}".replace(",", "{,}")]
            out.append(f"| {L} | {pipe} | " + " | ".join(cells) + " |")
            body.append(f"{L} & {pipe} & " + " & ".join(tcells))
    w = {(L, g, sc): int(R[(R.pipeline == 'L1') & (R.seed == 'S1') & (R.lead == L) & (R.gate == g) & (R.scope == sc)].windows.iloc[0]) for L in LEADS for g in ("ungated", "gated") for sc in ("all", "burst")}
    out.append("\nLabelled windows: " + "; ".join(f"{L} {w[(L, 'ungated', 'all')]:,} (bursts {w[(L, 'ungated', 'burst')]:,}), gate keeps {w[(L, 'gated', 'all')]:,} (bursts {w[(L, 'gated', 'burst')]:,})" for L in LEADS) + ".")
    X.upsert("t4", "\n".join(out))
    tex_write("T4_noise_robustness", "ST-deviation error on the noisy signal, ungated and gated (T4).", "tab:t4_noise",
              "llrrrrrrrr", "Lead & Pipeline & \\multicolumn{2}{c}{Ungated} & \\multicolumn{2}{c}{Ungated, bursts} & \\multicolumn{2}{c}{Gated} & \\multicolumn{2}{c}{Gated, bursts} \\\\\n"
              " & & MAE (mV) & $n$ & MAE (mV) & $n$ & MAE (mV) & $n$ & MAE (mV) & $n$", body,
              "NSTDB em noise, seed 42; noisy estimates against clean labels on 10\\,s windows; masked pairs excluded. Gated: windows kept by the 6\\,dB SQI gate "
              "(rodrigues2021 R-peaks), the same windows for every pipeline. Bursts: windows entirely inside a noise burst. $n$: windows with a value. "
              "L1, L2: mean $\\pm$ SD over three training seeds ($n$: mean).")
    return R


# ------------------------------------------------------------------ T6
def t6():
    R5 = pd.read_csv(T.OUT / "T5_alarms_per_seed.csv")
    rows = []
    for ns in (42, 43, 44):
        V = pd.read_csv(T.S5 / f"nk_values_noisy{ns}.csv.gz")
        Mg, Mn = T.memos(V, "anchor_grid", "sqi_grid"), T.memos(V, "anchor_nk", "sqi_nk")
        for pipe in PIPES:
            if pipe in T.NK:
                r = []
                for (rec, L), g in V[V.nk_window].groupby(["record", "lead"]):
                    g = g.sort_values("k")
                    r += T.fsm(rec, L, g.anchor_nk, g[T.NK[pipe]], Mn[(rec, L)])
                s = "-"
            else:
                s = "S1"
                r = T.learned_rows(T.learned_frames(pipe, s, f"noisy{ns}"), "gated", Mg)
            al, c, met = T.score(r)
            al.to_csv(OUT / "alarms" / f"{pipe}_{s}_gated_noise{ns}.csv", index=False)
            c.assign(pipeline=pipe, seed=s, noise_seed=ns).to_csv(OUT / "alarms" / f"components_{pipe}_{s}_gated_noise{ns}.csv", index=False)
            rows.append(dict(pipeline=pipe, seed=s, noise_seed=ns, **met))
            print("T6", pipe, ns, met["TP"], met["FP"], met["FN"], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "T6_alarms_per_noise_seed.csv", index=False)
    chk = []
    for pipe in PIPES:
        s = "S1" if pipe in ("L1", "L2") else "-"
        a = R[(R.pipeline == pipe) & (R.noise_seed == 42)].iloc[0]
        b = R5[(R5.pipeline == pipe) & (R5.seed == s) & (R5.condition == "gated")].iloc[0]
        same = all(int(a[k]) == int(b[k]) for k in ("TP", "FP", "FN", "Ignored"))
        chk.append(f"{'PASS' if same else 'FAIL'}  T6 seed 42 equals tier-1 gated {pipe} {s}")
    ok = all(c.startswith("PASS") for c in chk)
    out = head("## T6. Alarm robustness across noise seeds (gated)", "NSTDB `em` seeds 42, 43, 44, each with its own noisy signal, predictions, NeuroKit values and SQI gate "
               "(rodrigues2021 peaks, 6 dB). " + 'Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record–lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN).' + " Mean ± SD (ddof 1) over the three noise seeds; L1, L2: training seed S1. "
               f"Check: seed 42 equals the tier-1 gated rows ({sum(c.startswith('PASS') for c in chk)} / {len(chk)}).")
    out += ["| pipeline | TP | FP | FN | F1 | per seed 42 / 43 / 44: TP / FP / FN |", "|---|--:|--:|--:|--:|---|"]
    body = []
    for pipe in PIPES:
        q = R[R.pipeline == pipe].sort_values("noise_seed")
        lab = pipe + (" (S1)" if pipe in ("L1", "L2") else "")
        per = "; ".join(f"{int(r.TP)}/{int(r.FP)}/{int(r.FN)}" for r in q.itertuples())
        out.append(f"| {lab} | {pmx(q, 'TP', 1)} | {pmx(q, 'FP', 1)} | {pmx(q, 'FN', 1)} | {pmx(q, 'F1', 3)} | {per} |")
        body.append(f"{lab} & {tpmx(q, 'TP', 1)} & {tpmx(q, 'FP', 1)} & {tpmx(q, 'FN', 1)} & {tpmx(q, 'F1', 3)}")
    X.upsert("t6", "\n".join(out))
    tex_write("T6_alarm_robustness", "ST-deviation episode alarms across three noise realisations, gated (T6).", "tab:t6_robustness",
              "lrrrr", "Pipeline & TP & FP & FN & F1", body,
              "NSTDB em noise, seeds 42, 43 and 44, each gated with its own SQI (rodrigues2021 R-peaks, 6\\,dB). Alarms of both leads, each scored against the gold-standard episodes of its own lead; counts pooled over the 26 evaluated record--lead pairs. Precision (P) = TP/(TP+FP); recall (R) = TP/(TP+FN). 101 episodes, 15 records. "
              "Mean $\\pm$ SD over the three noise seeds; L1, L2: training seed S1.")
    return R, chk, ok


def main(which):
    (OUT / "alarms").mkdir(parents=True, exist_ok=True)
    (OUT / f"DONE_{which}").unlink(missing_ok=True)
    ok = True
    if which == "T2":
        t2()
    elif which == "T3":
        ok = t3()[1]
    elif which == "T4":
        t4()
    elif which == "T6":
        _, chk, ok = t6()
        (OUT / "T6_CHECKS.txt").write_text("\n".join(chk) + "\n")
        print("\n".join(chk))
    (OUT / (f"DONE_{which}" if ok else f"FAILED_{which}")).write_text(time.strftime("%F %T") + "\n")
    print(which, "done" if ok else "FAILED", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
