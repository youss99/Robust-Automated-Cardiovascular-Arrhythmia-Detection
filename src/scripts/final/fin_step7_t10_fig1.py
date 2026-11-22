#!/usr/bin/env python3
"""Final runs, step 7 after tier 1 (DECISIONS.md 2026-11-21: then T10 and Fig. 1): ablations A1-A4 (T10) and the Bland-Altman figure (Fig. 1).

Same alarm path, scorer, labels and gate tables as fin_step7_tier1.py (imported, unchanged). Noise seed 42.
A1  L1 vs L1-sup, L2 vs L2-sup: training seeds S1-S3 (tier-1 per-seed files, no recomputation).
A2  L1 vs L1-old = the published CMBEC R1 predictions (pretrained, lead II + V2 models, union), gate from rodrigues2021 peaks.
    L1-rebuilt: not run (step 2b was not submitted).
A3  NK-FP vs NK-FP-nk (NeuroKit's own peaks, refit on them; gate rodrigues2021); NK-PR vs NK-PR-pub (published M1 per-beat values; gate .atr).
A4  L1 (S1), L2 (S1), NK-FP: (a) .atr peaks for the gate instead of rodrigues2021, 6 dB; (b) SQI threshold sweep -20 to +25 dB, 1 dB steps,
    rodrigues2021 gate. Gated MAE = noisy predictions vs clean labels on the set A windows the gate keeps (gate at the learned grid anchor).
T1 metrics (clean, set A, masked pairs out) and T5 metrics (union, FULL) per row.
Fig. 1  Bland-Altman, clean, DMY beats (label = DMY ee; value = the 10 s window containing the beat, k = sample_16a // 2500, as phase 1's set B
    and C section 9), L1 (S1), L2 (S1), NK-FP; masked pairs out. Per lead: bias, 95 % limits of agreement (bias +- 1.96 SD), OLS slope of the
    difference on the mean.
Controls first (stop the run): published M1 (NKPRpub) and M2-R (M2Rctl, NeuroKit peaks + phase 1's offsets) through this alarm path reproduce
phase 1's stored FULL counts (main_noisy / main_noisy_atr); the 6 dB point of the sweep equals the tier-1 gated rows.
Output: Tasks/final/runs/step7_t10/, docs/RESULTS_FINAL.md (t10 and fig1 blocks), docs/thesis_tables/T10_*.tex, docs/thesis_figures/*.pdf.
Usage: setsid nohup <repo .venv>/bin/python -u -W ignore fin_step7_t10_fig1.py > logs/step7_t10.log 2>&1 < /dev/null &"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other final drivers
import fin_step7_tier1 as T  # noqa: E402  (patches the SQI call exactly as tier 1 did)

P, K = T.P, T.K
OUT = Path("/home/student/GIT/Tasks/final/runs/step7_t10")
TIER1 = T.OUT
FIG = T.DOCS / "thesis_figures"
LEADS, MASK, WIN = T.LEADS, T.MASK, T.WIN
SWEEP = list(range(-20, 26))
COLORS = {"L1": "#2a78d6", "L2": "#eb6834", "NK-FP": "#1baf7a"}  # dataviz default categorical slots 1-3, fixed order
DISPLAY = {"L1": "Learned-1L", "L2": "Learned-2L", "NK-FP": "Fiducial-FP"}  # figure labels (DECISIONS.md 2026-11-22 terminology)
MARK = {"L1": ("o", "-"), "L2": ("s", "--"), "NK-FP": ("^", ":")}
PHASE1 = {("M1", "clean"): (67, 32, 34, 13), ("M1", "noisy"): (67, 43, 34, 15), ("M1", "gated_atr"): (64, 27, 37, 7),
          ("M2R", "clean"): (86, 23, 15, 11), ("M2R", "noisy"): (85, 94, 16, 42)}  # runs/_scores/main_noisy{,_atr}/alarms.csv, FULL


def log(*a):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")
    print(*a, flush=True)


def nk_rows(d, col, memo):
    rows = []
    for (rec, L), g in d[d.nk_window].groupby(["record", "lead"]):
        g = g.sort_values("k")
        rows += T.fsm(rec, L, g.anchor_nk, g[col], None if memo is None else memo[(rec, L)])
    return rows


def r1_frames(cond):
    out = {}
    for mode, L in (("lead_II", "II"), ("lead_V2", "V2")):
        f = pd.read_csv(P.grid_file("R1_published", "pretrained", mode, "clean" if cond == "clean" else "noisy"), usecols=["record", "anchor_samp", f"pred_{L}_mV"])
        out[L] = f.rename(columns={f"pred_{L}_mV": "pred"})
    return out


def count(met):
    return (int(met["TP"]), int(met["FP"]), int(met["FN"]), int(met["Ignored"]))


def controls(V, M):
    lines, ok = [], True
    for (pipe, cond), want in PHASE1.items():
        col = "NKPRpub" if pipe == "M1" else "M2Rctl"
        d = V["clean"] if cond == "clean" else V["noisy42"]
        _, _, met = T.score(nk_rows(d, col, M["nk_atr"] if cond == "gated_atr" else None))
        got = count(met)
        ok &= got == want
        lines.append(f"{'PASS' if got == want else 'FAIL'}  control phase 1 {pipe} {cond} through this alarm path: {got} vs stored {want}")
    return ok, lines


# ------------------------------------------------------------------ regression helpers
def reg_frame(V, Lb, L, cond, src):
    """set A windows for lead L, masked out; src: ('learned', pipe, seed) | ('R1',) | ('nk', col). Adds the rodrigues and .atr gates."""
    base = Lb[["record", "k", f"oracle_{L}_mV"]].rename(columns={f"oracle_{L}_mV": "y"})
    base = base[base.record.isin(T.RECS) & ~base.record.map(lambda r: (r, L) in MASK)]
    g = V["noisy42"][V["noisy42"].lead == L][["record", "k", "sqi_grid", "sqi_grid_atr"]]
    base = base.merge(g, on=["record", "k"], how="left")
    if src[0] == "nk":
        v = V["clean" if cond == "clean" else "noisy42"]
        v = v[(v.lead == L) & v.nk_window][["record", "k", src[1]]].rename(columns={src[1]: "p"})
    else:
        if src[0] == "R1":
            mode = "lead_II" if L == "II" else "lead_V2"
            f = pd.read_csv(P.grid_file("R1_published", "pretrained", mode, "clean" if cond == "clean" else "noisy"), usecols=["record", "anchor_samp", f"pred_{L}_mV"])
        else:
            mode = dict((l, m) for m, l in T.LEARNED[src[1]])[L]
            f = pd.read_csv(T.S6 / f"predictions_{T.COND[cond]}" / f"FIN_{src[1]}_{mode}_{src[2]}__grid.csv.gz", usecols=["record", "anchor_samp", f"pred_{L}_mV"])
        f["k"] = (f.anchor_samp - (WIN - 1)) // WIN
        v = f[["record", "k", f"pred_{L}_mV"]].rename(columns={f"pred_{L}_mV": "p"})
    return base.merge(v, on=["record", "k"], how="left", validate="1:1")


def t1_row(f):
    q = f[f.y.notna()]
    return dict(windows=len(q), with_value=int(q.p.notna().sum()), coverage_pct=100 * q.p.notna().mean(), **P.reg(q.y, q.p))


def gated_mae(f, col):
    q = f[f.y.notna()]
    keep = np.isfinite(q[col]) & (q[col] >= T.THR)
    k = q[keep & q.p.notna()]
    return dict(kept_pct=100 * keep.mean(), n_kept=int(len(k)), MAE_gated=float((k.p - k.y).abs().mean()))


# ------------------------------------------------------------------ T10
def t10(V, Lb, M):
    R5 = pd.read_csv(TIER1 / "T5_alarms_per_seed.csv")
    R1t = pd.read_csv(TIER1 / "T1_regression_per_seed.csv")
    alarms, reg, comps = [], [], []

    def add_alarm(abl, name, seed, cond, rows, gate="rodrigues2021"):
        al, c, met = T.score(rows)
        al.to_csv(OUT / "alarms" / f"{abl}_{name}_{seed}_{cond}_{gate}.csv", index=False)
        alarms.append(dict(ablation=abl, pipeline=name, seed=seed, condition=cond, gate=gate, **met))
        comps.append(c.assign(ablation=abl, pipeline=name, seed=seed, condition=cond, gate=gate))
        log("alarms", abl, name, seed, cond, gate, met["TP"], met["FP"], met["FN"])

    def copy_tier1(abl, pipe, seeds):
        for _, r in R5[(R5.pipeline == pipe) & R5.seed.isin(seeds)].iterrows():
            alarms.append(dict(ablation=abl, **{k: r[k] for k in r.index if k not in ("pipeline", "seed", "condition")}, pipeline=pipe, seed=r.seed, condition=r.condition, gate="rodrigues2021", source="tier 1"))
        for _, r in R1t[(R1t.pipeline == pipe) & R1t.seed.isin(seeds)].iterrows():
            reg.append(dict(ablation=abl, **r.to_dict(), source="tier 1"))

    # A1
    for pipe in ("L1", "L1sup", "L2", "L2sup"):
        copy_tier1("A1", pipe, list(T.SEEDS))
    # A2: L1 (S1-S3, and S1 alone) vs L1-old (published R1)
    copy_tier1("A2", "L1", list(T.SEEDS))
    for cond in ("clean", "noisy", "gated"):
        add_alarm("A2", "L1-old", "published", cond, T.learned_rows(r1_frames(cond), cond, M["grid_rod"] if cond == "gated" else None))
    for L in LEADS:
        reg.append(dict(ablation="A2", pipeline="L1-old", seed="published", lead=L, **t1_row(reg_frame(V, Lb, L, "clean", ("R1",)))))
    # A3
    copy_tier1("A3", "NK-FP", ["-"])
    copy_tier1("A3", "NK-PR", ["-"])
    for name, col, gk, gname in (("NK-FP-nk", "NKFPnk", "nk_rod", "rodrigues2021"), ("NK-PR-pub", "NKPRpub", "nk_atr", ".atr")):
        for cond in ("clean", "noisy", "gated"):
            d = V["clean"] if cond == "clean" else V["noisy42"]
            add_alarm("A3", name, "-", cond, nk_rows(d, col, M[gk] if cond == "gated" else None), gate=gname)
        for L in LEADS:
            reg.append(dict(ablation="A3", pipeline=name, seed="-", lead=L, **t1_row(reg_frame(V, Lb, L, "clean", ("nk", col)))))
    # A4a: .atr gate vs rodrigues2021 gate, gated only; gated MAE on kept windows for both gates
    gm = []
    for pipe, seed in (("L1", "S1"), ("L2", "S1"), ("NK-FP", "-")):
        for _, r in R5[(R5.pipeline == pipe) & (R5.seed == seed) & (R5.condition == "gated")].iterrows():
            alarms.append(dict(ablation="A4a", **{k: r[k] for k in r.index if k not in ("pipeline", "seed", "condition")}, pipeline=pipe, seed=seed, condition="gated", gate="rodrigues2021", source="tier 1"))
        if pipe == "NK-FP":
            add_alarm("A4a", pipe, seed, "gated", nk_rows(V["noisy42"], "NKFP", M["nk_atr"]), gate=".atr")
        else:
            add_alarm("A4a", pipe, seed, "gated", T.learned_rows(T.learned_frames(pipe, seed, "noisy42"), "gated", M["grid_atr"]), gate=".atr")
        for L in LEADS:
            f = reg_frame(V, Lb, L, "noisy", ("nk", "NKFP") if pipe == "NK-FP" else ("learned", pipe, seed))
            for gname, col in (("rodrigues2021", "sqi_grid"), (".atr", "sqi_grid_atr")):
                gm.append(dict(pipeline=pipe, seed=seed, lead=L, gate=gname, **gated_mae(f, col)))
    # A4b: threshold sweep, rodrigues2021 gate
    sweep = []
    for thr in SWEEP:
        T.THR = float(thr)
        for pipe, seed in (("L1", "S1"), ("L2", "S1"), ("NK-FP", "-")):
            rows = nk_rows(V["noisy42"], "NKFP", M["nk_rod"]) if pipe == "NK-FP" else T.learned_rows(T.learned_frames(pipe, seed, "noisy42"), "gated", M["grid_rod"])
            _, c, met = T.score(rows)
            rec = dict(threshold_db=thr, pipeline=pipe, seed=seed, **met)
            for L in LEADS:
                f = reg_frame(V, Lb, L, "noisy", ("nk", "NKFP") if pipe == "NK-FP" else ("learned", pipe, seed))
                gmm = gated_mae(f, "sqi_grid")
                rec[f"kept_pct_{L}"], rec[f"MAE_gated_{L}"] = gmm["kept_pct"], gmm["MAE_gated"]
            sweep.append(rec)
        log("sweep", thr)
    T.THR = 6.0
    A, Rg, GM, S = pd.DataFrame(alarms), pd.DataFrame(reg), pd.DataFrame(gm), pd.DataFrame(sweep)
    A.to_csv(OUT / "T10_alarms.csv", index=False)
    Rg.to_csv(OUT / "T10_regression.csv", index=False)
    GM.to_csv(OUT / "T10_A4a_gated_mae.csv", index=False)
    S.to_csv(OUT / "T10_A4b_sweep.csv", index=False)
    pd.concat(comps, ignore_index=True).to_csv(OUT / "T10_alarm_components_per_record.csv", index=False)
    # consistency: 6 dB sweep point = tier-1 gated rows; A4a GM at rodrigues = tier-1 T7 values are not the same set (S1 only), so not compared
    chk = []
    for pipe, seed in (("L1", "S1"), ("L2", "S1"), ("NK-FP", "-")):
        a = S[(S.threshold_db == 6) & (S.pipeline == pipe)].iloc[0]
        b = R5[(R5.pipeline == pipe) & (R5.seed == seed) & (R5.condition == "gated")].iloc[0]
        same = all(int(a[k]) == int(b[k]) for k in ("TP", "FP", "FN", "Ignored"))
        chk.append(f"{'PASS' if same else 'FAIL'}  sweep at 6 dB equals tier-1 gated {pipe} {seed}: {(int(a.TP), int(a.FP), int(a.FN), int(a.Ignored))} vs {(int(b.TP), int(b.FP), int(b.FN), int(b.Ignored))}")
    return A, Rg, GM, S, chk


# ------------------------------------------------------------------ Fig. 1
def fig1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = P.dmy_table()
    D = D[~np.array([(r, l) in MASK for r, l in zip(D.record, D.lead)], bool)].copy()
    D["k"] = D.sample_16a // WIN
    Vc = pd.read_csv(T.S5 / "nk_values_clean.csv.gz", usecols=["record", "lead", "k", "nk_window", "NKFP"])
    vals = {}
    for pipe in ("L1", "L2", "NK-FP"):
        parts = []
        for L in LEADS:
            q = D[D.lead == L][["record", "lead", "sample_16a", "k", "dmy_ee_mV"]]
            if pipe == "NK-FP":
                v = Vc[(Vc.lead == L) & Vc.nk_window][["record", "k", "NKFP"]].rename(columns={"NKFP": "p"})
            else:
                mode = dict((l, m) for m, l in T.LEARNED[pipe])[L]
                v = pd.read_csv(T.S6 / "predictions_clean" / f"FIN_{pipe}_{mode}_S1__grid.csv.gz", usecols=["record", "anchor_samp", f"pred_{L}_mV"])
                v["k"] = (v.anchor_samp - (WIN - 1)) // WIN
                v = v[["record", "k", f"pred_{L}_mV"]].rename(columns={f"pred_{L}_mV": "p"})
            parts.append(q.merge(v, on=["record", "k"], how="left", validate="m:1"))
        vals[pipe] = pd.concat(parts, ignore_index=True).rename(columns={"dmy_ee_mV": "y"})
    rows, beats = [], []
    for pipe, d in vals.items():
        beats.append(d.assign(pipeline=pipe, seed="S1" if pipe != "NK-FP" else "-"))
        for L in LEADS:
            q = d[(d.lead == L) & d.y.notna() & d.p.notna()]
            m, df = (q.p + q.y) / 2, q.p - q.y
            sd = df.std(ddof=1)
            rows.append(dict(pipeline=pipe, lead=L, n=len(q), n_dmy=int((d.lead == L).sum()), bias=df.mean(), sd=sd, loa_lo=df.mean() - 1.96 * sd, loa_hi=df.mean() + 1.96 * sd,
                             slope_diff_on_mean=float(np.polyfit(m, df, 1)[0]), MAE=float(df.abs().mean())))
    St = pd.DataFrame(rows)
    St.to_csv(OUT / "Fig1_bland_altman_stats.csv", index=False)
    pd.concat(beats, ignore_index=True).to_csv(OUT / "Fig1_bland_altman_beats.csv.gz", index=False)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    ink, muted = "#262624", "#8a8981"
    for i, L in enumerate(LEADS):
        for j, pipe in enumerate(("L1", "L2", "NK-FP")):
            a = ax[i, j]
            d = vals[pipe]
            q = d[(d.lead == L) & d.y.notna() & d.p.notna()]
            m, df = (q.p + q.y) / 2, q.p - q.y
            s = St[(St.pipeline == pipe) & (St.lead == L)].iloc[0]
            a.scatter(m, df, s=4, alpha=0.35, color=COLORS[pipe], linewidths=0, rasterized=True)
            a.axhline(0, color=muted, lw=0.6)
            a.axhline(s.bias, color=ink, lw=1.0)
            for v in (s.loa_lo, s.loa_hi):
                a.axhline(v, color=ink, lw=0.8, ls="--")
            xs = np.linspace(m.min(), m.max(), 2)
            b1, b0 = np.polyfit(m, df, 1)
            a.plot(xs, b0 + b1 * xs, color=ink, lw=0.8, ls=":")
            a.text(0.03, 0.97, f"bias {s.bias:+.3f} mV\nLoA [{s.loa_lo:+.3f}, {s.loa_hi:+.3f}]\nslope {s.slope_diff_on_mean:+.2f}   n = {int(s.n)}",
                   transform=a.transAxes, va="top", ha="left", fontsize=6.5, color=ink)
            if i == 0:
                a.set_title(DISPLAY[pipe] + (" (S1)" if pipe != "NK-FP" else ""), fontsize=8.5, color=ink)
            if j == 0:
                a.set_ylabel(f"Lead {L}\nestimate − expert (mV)")
            if i == 1:
                a.set_xlabel("mean of estimate and expert (mV)")
            a.grid(True, color="#e6e5e0", lw=0.4)
            a.set_axisbelow(True)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "Fig1_bland_altman.pdf")
    fig.savefig(OUT / "Fig1_bland_altman.png", dpi=150)
    plt.close(fig)
    return St


def fig_sweep(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    ink = "#262624"
    for k, (col, lab) in enumerate((("F1", "F1 (union)"), ("FP", "false-positive episodes"), ("kept_pct_II", "lead-II windows kept (%)"))):
        a = ax[k]
        for pipe in ("L1", "L2", "NK-FP"):
            q = S[S.pipeline == pipe].sort_values("threshold_db")
            mk, ls = MARK[pipe]
            a.plot(q.threshold_db, q[col], color=COLORS[pipe], lw=1.5, ls=ls, marker=mk, ms=2.5, label=DISPLAY[pipe] + (" (S1)" if pipe != "NK-FP" else ""))
        a.axvline(6, color=ink, lw=0.6, ls="--")
        a.set_xlabel("SQI threshold (dB)")
        a.set_title(lab, fontsize=8.5, color=ink)
        a.grid(True, color="#e6e5e0", lw=0.4)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "FigA4b_gate_threshold_sweep.pdf")
    fig.savefig(OUT / "FigA4b_gate_threshold_sweep.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ RESULTS_FINAL.md and .tex
def upsert(tag, block):
    md = T.MD.read_text()
    a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    block = a + "\n" + block.rstrip("\n") + "\n" + b + "\n"
    if a in md:
        x, rest = md.split(a, 1)
        md = x + block + rest.split(b, 1)[1].lstrip("\n")
    else:
        md = md.rstrip("\n") + "\n\n" + block
    T.MD.write_text(md)


ORDER = [("A1", "L1"), ("A1", "L1sup"), ("A1", "L2"), ("A1", "L2sup"), ("A2", "L1"), ("A2", "L1-old"),
         ("A3", "NK-FP"), ("A3", "NK-FP-nk"), ("A3", "NK-PR"), ("A3", "NK-PR-pub")]
LABEL = {"L1sup": "L1-sup", "L2sup": "L2-sup"}


def seeds_of(q):
    s = sorted(set(q.seed.astype(str)))
    return "S1–S3" if s == ["S1", "S2", "S3"] else ", ".join(s)


def md_t10(A, Rg, GM, S, ctl, chk):
    pm = T.pm
    out = ["## T10. Ablations A1–A4", "",
           f"Generated {time.strftime('%F %T')} by `src/scripts/final/fin_step7_t10_fig1.py` (VM). Noise seed 42. Same alarm path, scorer, labels and gate tables as tier 1; "
           "rows marked S1–S3 are mean ± SD over the training seeds (A1, and L1 in A2), others one run. Each ablation row sits under the row it is compared with. "
           "L1-rebuilt (A2, optional) was not run: step 2b was not submitted.", "",
           "Controls: " + ("published M1 and M2-R through this alarm path reproduce phase 1's stored counts (5 / 5); the 6 dB point of the sweep equals tier 1 (3 / 3)."
                           if all(l.startswith("PASS") for l in ctl + chk) else "FAILED (see CONTROL.txt)"), "",
           "### T10a. Regression (T1 metrics, clean)", "",
           "| ablation | lead | pipeline | seeds | coverage % | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |", "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for L in LEADS:
        for abl, pipe in ORDER:
            q = Rg[(Rg.ablation == abl) & (Rg.pipeline == pipe) & (Rg.lead == L)]
            out.append(f"| {abl} | {L} | {LABEL.get(pipe, pipe)} | {seeds_of(q)} | {pm(q.coverage_pct, 1)} | {pm(q.MAE, 4)} | {pm(q.bias, 4)} | {pm(q.RMSE, 4)} | {pm(q.slope)} | {pm(q.r)} | {pm(q.p100_pct, 1)} | {pm(q.R2)} |")
    out += ["", "### T10b. Alarms (T5 metrics, union, FULL)", "",
            "Gate: rodrigues2021 peaks, 6 dB, except NK-PR-pub (`.atr`, as published) and the A4a `.atr` rows.", "",
            "| ablation | condition | pipeline | seeds | gate | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | redundant |", "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for abl, pipe in ORDER:
        for cond in ("clean", "noisy", "gated"):
            q = A[(A.ablation == abl) & (A.pipeline == pipe) & (A.condition == cond)]
            gate = "–" if cond != "gated" else q.gate.iloc[0]
            out.append(f"| {abl} | {cond} | {LABEL.get(pipe, pipe)} | {seeds_of(q)} | {gate} | {pm(q.TP, 1)} | {pm(q.FP, 1)} | {pm(q.FN, 1)} | {pm(q.precision)} | {pm(q.recall)} | {pm(q.F1)} | {pm(q.IoU_whole_record)} | {pm(q.false_hours, 1)} | {pm(q.Ignored, 1)} |")
    out += ["", "### T10c. A4a: gate peaks, `.atr` vs rodrigues2021 (gated, 6 dB)", "",
            "Gated MAE: noisy predictions against the clean labels on the set A windows each gate keeps (gate at the learned grid anchor), masked pairs out.", "",
            "| pipeline | gate | TP | FP | FN | F1 | false-alarm h | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for pipe in ("L1", "L2", "NK-FP"):
        for gate in ("rodrigues2021", ".atr"):
            q = A[(A.ablation == "A4a") & (A.pipeline == pipe) & (A.gate == gate)].iloc[0]
            g = {L: GM[(GM.pipeline == pipe) & (GM.gate == gate) & (GM.lead == L)].iloc[0] for L in LEADS}
            out.append(f"| {pipe}{' (S1)' if pipe != 'NK-FP' else ''} | {gate} | {int(q.TP)} | {int(q.FP)} | {int(q.FN)} | {q.F1:.3f} | {q.false_hours:.1f} | {g['II'].kept_pct:.1f} | {g['II'].MAE_gated:.4f} | {g['V2'].kept_pct:.1f} | {g['V2'].MAE_gated:.4f} |")
    out += ["", "### T10d. A4b: SQI threshold sweep (rodrigues2021 gate)", "",
            "Every 5 dB shown; all 46 thresholds (−20 to +25 dB, 1 dB steps) in `Tasks/final/runs/step7_t10/T10_A4b_sweep.csv` and `docs/thesis_figures/FigA4b_gate_threshold_sweep.pdf`.", "",
            "| threshold (dB) | pipeline | TP | FP | FN | F1 | II kept % | II MAE (mV) | V2 kept % | V2 MAE (mV) |", "|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for thr in [t for t in SWEEP if t % 5 == 0 or t == 6]:
        for pipe in ("L1", "L2", "NK-FP"):
            q = S[(S.threshold_db == thr) & (S.pipeline == pipe)].iloc[0]
            out.append(f"| {thr:+d}{' (main)' if thr == 6 else ''} | {pipe}{' (S1)' if pipe != 'NK-FP' else ''} | {int(q.TP)} | {int(q.FP)} | {int(q.FN)} | {q.F1:.3f} | {q.kept_pct_II:.1f} | {q.MAE_gated_II:.4f} | {q.kept_pct_V2:.1f} | {q.MAE_gated_V2:.4f} |")
    upsert("t10", "\n".join(out))


def md_fig1(St):
    out = ["## Fig. 1. Bland–Altman (clean, DMY beats)", "",
           f"Generated {time.strftime('%F %T')} by `src/scripts/final/fin_step7_t10_fig1.py` (VM). Figure: `docs/thesis_figures/Fig1_bland_altman.pdf`. "
           "Label = the DMY beat's expert ST deviation; estimate = the 10 s window containing the beat (as phase 1's set B and C §9); masked pairs out. "
           "LoA = bias ± 1.96 SD of the difference; slope = OLS of the difference on the mean (negative = compression of large deviations).", "",
           "| lead | pipeline | n beats | bias (mV) | SD (mV) | 95 % LoA (mV) | slope (diff on mean) | MAE (mV) |", "|---|---|--:|--:|--:|---|--:|--:|"]
    for L in LEADS:
        for pipe in ("L1", "L2", "NK-FP"):
            s = St[(St.pipeline == pipe) & (St.lead == L)].iloc[0]
            out.append(f"| {L} | {pipe}{' (S1)' if pipe != 'NK-FP' else ''} | {int(s.n)} | {s.bias:+.4f} | {s.sd:.4f} | [{s.loa_lo:+.4f}, {s.loa_hi:+.4f}] | {s.slope_diff_on_mean:+.3f} | {s.MAE:.4f} |")
    upsert("fig1", "\n".join(out))


def tex_write(name, caption, label, colspec, header, body, note):
    T.TEX.mkdir(parents=True, exist_ok=True)
    lines = [f"% {name}: generated {time.strftime('%F %T')} by src/scripts/final/fin_step7_t10_fig1.py (VM). Needs \\usepackage{{booktabs}}.",
             "\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for row in body:
        lines.append("\\midrule" if row == "MIDRULE" else row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", f"\\par\\smallskip\\footnotesize {note}", "\\end{table}", ""]
    (T.TEX / f"{name}.tex").write_text("\n".join(lines))


def tex_all(A, Rg, GM, S, St):
    tpm = T.tpm
    neg = (lambda v, d: f"{v:+.{d}f}".replace("-", "$-$"))
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for abl, pipe in ORDER:
            q = Rg[(Rg.ablation == abl) & (Rg.pipeline == pipe) & (Rg.lead == L)]
            body.append(f"{abl} & {L} & {LABEL.get(pipe, pipe)} & {seeds_of(q).replace('–', '--')} & {tpm(q.coverage_pct, 1)} & {tpm(q.MAE, 4)} & {tpm(q.bias, 4)} & {tpm(q.RMSE, 4)} & {tpm(q.slope)} & {tpm(q.r)} & {tpm(q.p100_pct, 1)} & {tpm(q.R2)}")
    tex_write("T10_ablations_regression", "Ablations A1--A3: ST-deviation regression on all 10\\,s windows, clean signal (T10a).", "tab:t10_regression",
              "llllrrrrrrrr", "Abl. & Lead & Pipeline & Seeds & Cov.\\ (\\%) & MAE (mV) & Bias (mV) & RMSE (mV) & Slope & $r$ & p100 (\\%) & $R^2$", body,
              "Each ablation row follows the row it is compared with. A1: supervised, no pre-trained backbone. A2: L1-old = published CMBEC single-lead models. "
              "A3: NK-FP-nk = fixed point on NeuroKit's own R-peaks; NK-PR-pub = published paper-rule values. Masked record--lead pairs excluded; S1--S3: mean $\\pm$ SD over training seeds.")
    body = []
    for abl, pipe in ORDER:
        for cond in ("clean", "noisy", "gated"):
            q = A[(A.ablation == abl) & (A.pipeline == pipe) & (A.condition == cond)]
            gate = "--" if cond != "gated" else ("rod." if q.gate.iloc[0] == "rodrigues2021" else "\\texttt{.atr}")
            body.append(f"{abl} & {cond.capitalize()} & {LABEL.get(pipe, pipe)} & {seeds_of(q).replace('–', '--')} & {gate} & {tpm(q.TP, 1)} & {tpm(q.FP, 1)} & {tpm(q.FN, 1)} & {tpm(q.precision)} & {tpm(q.recall)} & {tpm(q.F1)} & {tpm(q.false_hours, 1)} & {tpm(q.Ignored, 1)}")
        if (abl, pipe) != ORDER[-1]:
            body.append("MIDRULE")
    tex_write("T10_ablations_alarms", "Ablations A1--A3: ST-deviation episode alarms (T10b).", "tab:t10_alarms",
              "lllllrrrrrrrr", "Abl. & Condition & Pipeline & Seeds & Gate & TP & FP & FN & Precision & Recall & F1 & FA (h) & Redundant", body,
              "Union of lead II and V2 alarms, 101 episodes, 15 records; noisy: NSTDB em, seed 42; gate 6\\,dB on rodrigues2021 R-peaks (rod.) or annotated beats (\\texttt{.atr}). "
              "S1--S3: mean $\\pm$ SD over training seeds.")
    body = []
    for pipe in ("L1", "L2", "NK-FP"):
        for gate in ("rodrigues2021", ".atr"):
            q = A[(A.ablation == "A4a") & (A.pipeline == pipe) & (A.gate == gate)].iloc[0]
            g = {L: GM[(GM.pipeline == pipe) & (GM.gate == gate) & (GM.lead == L)].iloc[0] for L in LEADS}
            gtex = "rodrigues2021" if gate != ".atr" else "\\texttt{.atr}"
            body.append(f"{pipe}{' (S1)' if pipe != 'NK-FP' else ''} & {gtex} & {int(q.TP)} & {int(q.FP)} & {int(q.FN)} & {q.F1:.3f} & {q.false_hours:.1f} & {g['II'].kept_pct:.1f} & {g['II'].MAE_gated:.4f} & {g['V2'].kept_pct:.1f} & {g['V2'].MAE_gated:.4f}")
    tex_write("T10_A4a_gate_peaks", "Ablation A4a: R-peaks for the SQI gate, annotated beats vs rodrigues2021 (gated, 6\\,dB; T10c).", "tab:t10_a4a",
              "llrrrrrrrrr", "Pipeline & Gate peaks & TP & FP & FN & F1 & FA (h) & II kept (\\%) & II MAE (mV) & V2 kept (\\%) & V2 MAE (mV)", body,
              "Gated MAE: noisy predictions against clean labels on the 10\\,s windows the gate keeps; masked pairs excluded.")
    body = []
    for thr in [t for t in SWEEP if t % 5 == 0 or t == 6]:
        for n, pipe in enumerate(("L1", "L2", "NK-FP")):
            q = S[(S.threshold_db == thr) & (S.pipeline == pipe)].iloc[0]
            body.append(f"{neg(thr, 0) if n == 0 else ''} & {pipe}{' (S1)' if pipe != 'NK-FP' else ''} & {int(q.TP)} & {int(q.FP)} & {int(q.FN)} & {q.F1:.3f} & {q.kept_pct_II:.1f} & {q.MAE_gated_II:.4f} & {q.kept_pct_V2:.1f} & {q.MAE_gated_V2:.4f}")
        if thr != 25:
            body.append("MIDRULE")
    tex_write("T10_A4b_threshold_sweep", "Ablation A4b: SQI threshold sweep, rodrigues2021 gate (T10d).", "tab:t10_a4b",
              "rlrrrrrrrr", "Thr.\\ (dB) & Pipeline & TP & FP & FN & F1 & II kept (\\%) & II MAE (mV) & V2 kept (\\%) & V2 MAE (mV)", body,
              "Every 5\\,dB and the main threshold (6\\,dB); 1\\,dB steps from $-$20 to +25\\,dB in the companion figure. NSTDB em, seed 42.")
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for pipe in ("L1", "L2", "NK-FP"):
            s = St[(St.pipeline == pipe) & (St.lead == L)].iloc[0]
            body.append(f"{L} & {pipe}{' (S1)' if pipe != 'NK-FP' else ''} & {int(s.n)} & {neg(s.bias, 4)} & {s.sd:.4f} & [{neg(s.loa_lo, 4)}, {neg(s.loa_hi, 4)}] & {neg(s.slope_diff_on_mean, 3)}")
    tex_write("Fig1_bland_altman_stats", "Bland--Altman agreement with expert ST deviation on DMY beats, clean signal (Fig.~1).", "tab:fig1_ba",
              "llrrrlr", "Lead & Pipeline & $n$ & Bias (mV) & SD (mV) & 95\\,\\% LoA (mV) & Slope", body,
              "Estimate: the 10\\,s window containing the beat. LoA: bias $\\pm$ 1.96\\,SD. Slope: OLS of the difference on the mean. Masked record--lead pairs excluded.")


def main():
    (OUT / "alarms").mkdir(parents=True, exist_ok=True)
    for m in ("DONE_t10", "FAILED_t10"):
        (OUT / m).unlink(missing_ok=True)
    if not (TIER1 / "DONE_tier1").exists():
        sys.exit("tier 1 not done")
    V = {c: pd.read_csv(T.S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    M = {"grid_rod": T.memos(V["noisy42"], "anchor_grid", "sqi_grid"), "grid_atr": T.memos(V["noisy42"], "anchor_grid", "sqi_grid_atr"),
         "nk_rod": T.memos(V["noisy42"], "anchor_nk", "sqi_nk"), "nk_atr": T.memos(V["noisy42"], "anchor_nk", "sqi_nk_atr")}
    ok, ctl = controls(V, M)
    log("control", *ctl)
    (OUT / "CONTROL.txt").write_text("\n".join(ctl) + "\n")
    if not ok:
        (OUT / "FAILED_t10").write_text("control failed\n")
        return False
    Lb = T.labels()
    A, Rg, GM, S, chk = t10(V, Lb, M)
    (OUT / "CONTROL.txt").write_text("\n".join(ctl + chk) + "\n")
    log("check", *chk)
    if not all(l.startswith("PASS") for l in chk):
        (OUT / "FAILED_t10").write_text("sweep check failed\n")
        return False
    St = fig1()
    fig_sweep(S)
    md_t10(A, Rg, GM, S, ctl, chk)
    md_fig1(St)
    tex_all(A, Rg, GM, S, St)
    # the script that made the figures is this file, src/scripts/final/fin_step7_t10_fig1.py (no copy next to the PDFs)
    (OUT / "DONE_t10").write_text(time.strftime("%F %T") + "\n")
    log("done")
    return True


def figures():
    """Redraw both figures from the saved outputs (no alarms are rerun)."""
    fig1()
    fig_sweep(pd.read_csv(OUT / "T10_A4b_sweep.csv"))
    return True


if __name__ == "__main__":
    sys.exit(0 if (figures() if sys.argv[1:] == ["figures"] else main()) else 1)
