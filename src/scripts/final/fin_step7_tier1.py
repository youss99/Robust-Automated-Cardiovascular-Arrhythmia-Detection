#!/usr/bin/env python3
"""Final runs, step 7 tier 1 (DECISIONS.md 2026-11-21: T1, T5, T7 first): MON, NK-PR, NK-FP, L1, L2 -> RESULTS_FINAL.md (tier-1 sections).

Inputs: step 6 predictions (Tasks/final/runs/step6), step 5 NeuroKit values and the gate table (Tasks/final/runs/step5), the published per-window
oracle labels (Tasks/phase1/runs/_scores/oracle/oracle_grid.csv.gz: label = median .16a ee of the window's beats, the 10 s grid).
Alarms: PROSE FSM (old_fsm_gs.run_fsm_on_record, cadence 5 s, 30 s window, 6 dB) on each pipeline's own anchors (learned 2500k + 2499, NeuroKit
2500k + 2500); gated = the paper's SQI from rodrigues2021 noisy peaks (seed 42), read from the step-5 gate table through the same function
call (phase 1's precomputed-gate path). Masked pairs removed (nk_core.apply_mask). Scored with score_alarms against the 101-episode GS, 15 records.
L1 = lead II model on II + V2 model on V2 (union); L2 = the two-lead model on both leads.
T1  clean, set A windows, per lead, masked pairs out: n, MAE, bias, RMSE, slope (OLS of prediction on label), r, p100, R2; coverage. Learned rows:
    mean +- SD (ddof 1) over S1-S3, plus each seed.
T5  clean / noisy (seed 42) / gated (seed 42): TP, FP, FN, precision, recall, F1, IoU, false-alarm hours, duration Se and +P, redundant alarms
    (Ignored). Learned: mean +- SD over S1-S3. MON copied from the paper (gated TP 76, FP 54, FN 23; 99 episodes).
T7  gated, seed 42: L2-L1, L2-NK-FP, L1-NK-FP, L2-NK-PR, L1-L1sup, L2-L2sup. Record-clustered paired bootstrap, 5,000 resamples (seed 20260805),
    training seeds averaged inside each resample. dF1 and dFP (union alarms, FULL scope); dMAE per lead on set A windows the gate keeps
    (gate at the learned anchor, the same gate for both pipelines; each pipeline on its own windows with a value; masked pairs out).
Control first (stops the run): the published R1 predictions through this file's alarm path with the step-5 .atr SQI reproduce the paper's
union-pre and two_lead-pre TP/FP/FN/Ignored for clean, noisy and gated.
Output: Tasks/final/runs/step7/ (per-record CSVs, tables) and RESULTS_FINAL.md in the repo's docs/ (tier-1 block between markers).
Usage: setsid nohup <repo .venv>/bin/python -u -W ignore fin_step7_tier1.py > logs/step7_tier1.log 2>&1 < /dev/null &"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1_lib"))  # vendored phase-1 code (phase1_lib/MANIFEST.tsv)
import p1_eval as P  # noqa: E402
import ltst_pipeline.old_fsm_gs as G  # noqa: E402
from score_alarms import per_record_components, metrics_from_components, gs_scope  # noqa: E402

K, T1 = P.K, P.T1
FS, WIN, THR = 250, 2500, 6.0
S6 = Path("/home/student/GIT/Tasks/final/runs/step6")
S5 = Path("/home/student/GIT/Tasks/final/runs/step5")
OUT = Path("/home/student/GIT/Tasks/final/runs/step7")
DOCS = Path(__file__).resolve().parents[3] / "docs"  # the docs of the repo this script lives in (data paths stay in P)
MD = DOCS / "RESULTS_FINAL.md"
ORACLE = P.SC / "oracle" / "oracle_grid.csv.gz"
RECS = list(K.RECORDS_15)
MASK = set(K.MANUAL_OMIT)
SEEDS = ("S1", "S2", "S3")
LEADS = ("II", "V2")
LIDX = {"II": 0, "V2": 1}
TAG = {"II": "ST0", "V2": "ST1"}
NB, BSEED = 5000, 20260805
COND = {"clean": "clean", "noisy": "noisy42", "gated": "noisy42"}
LEARNED = {"L1": [("lead_II", "II"), ("lead_V2", "V2")], "L2": [("two_lead", "II"), ("two_lead", "V2")],
           "L1sup": [("lead_II", "II"), ("lead_V2", "V2")], "L2sup": [("two_lead", "II"), ("two_lead", "V2")]}
NK = {"NK-PR": "NKPR", "NK-FP": "NKFP"}
MON = {"clean": (78, 66, 21), "noisy": (77, 171, 22), "gated": (76, 54, 23)}  # PAPER_RESULTS_TWO_MODELS.md section 1 (Abdelazez 2017; 99 episodes)
GS = K.load_gs(T1.GS_PATH)
DUR = {r.record: float(r.duration_s) for r in pd.read_csv(T1.DUR_CSV).itertuples()}
_MEMO = {}
G.compute_sqi_last30s = lambda **kw: _MEMO[int(kw["current_r_peak"])]  # the precomputed gate: same function, same arguments


def log(*a):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")
    print(*a, flush=True)


def fsm(rec, lead, samp, st, memo):
    global _MEMO
    st = np.asarray(st, float)
    if not np.isfinite(st).any():
        return []
    _MEMO = memo or {}
    with contextlib.redirect_stdout(io.StringIO()):
        al = G.run_fsm_on_record(rec_id=rec, lead_tag=TAG[lead], samp_arr=[int(s) for s in samp], st_arr=[float(v) for v in st], ecg_sig_for_sqi=None,
                                 r_peaks=None, fs=FS, use_sqi=memo is not None, sqi_threshold_db=THR, sqi_every_s=5.0, sqi_window_s=30.0)
    return [dict(record=a.record, lead_tag=a.lead_tag, lead_paper=a.lead_paper, lead=a.lead_paper, onset_actual_s=a.onset_actual / FS, end_actual_s=a.end_actual / FS, criterion=a.crit) for a in al]


def score(rows):
    al = K.apply_mask(pd.DataFrame(rows, columns=["record", "lead_tag", "lead_paper", "lead", "onset_actual_s", "end_actual_s", "criterion"]), MASK)
    c = per_record_components(gs_scope(GS, "FULL"), K.scope_filter(al, "FULL"), DUR, "FULL", MASK, RECS)
    return al, c, {**metrics_from_components(c), "false_hours": c.tFP.sum() / 3600}


def gate_table(cond):
    V = pd.read_csv(S5 / f"nk_values_{cond}.csv.gz")
    return V


def memos(V, anchor, col):
    return {(r, L): dict(zip(g[anchor].astype(int), g[col])) for (r, L), g in V.groupby(["record", "lead"])}


def learned_rows(df_by_lead, cond, M):
    """df_by_lead: {lead: frame with record, anchor_samp, pred}; M: memo dict or None."""
    rows = []
    for L, d in df_by_lead.items():
        for rec, g in d.groupby("record"):
            if (rec, L) in MASK:
                continue
            g = g.sort_values("anchor_samp")
            rows += fsm(rec, L, g.anchor_samp, g.pred, None if M is None else M[(rec, L)])
    return rows


def learned_frames(pipe, s, cond):
    out = {}
    for mode, L in LEARNED[pipe]:
        f = pd.read_csv(S6 / f"predictions_{cond}" / f"FIN_{pipe}_{mode}_{s}__grid.csv.gz", usecols=["record", "anchor_samp", f"pred_{L}_mV"])
        out[L] = f.rename(columns={f"pred_{L}_mV": "pred"})
    return out


def control(Vg):
    """Published R1 predictions, this alarm path, the step-5 .atr SQI: must reproduce the paper."""
    M = memos(Vg, "anchor_grid", "sqi_grid_atr")
    paper = {("pretrained", "union"): {"clean": (75, 40, 26, 21), "noisy": (73, 57, 28, 99), "gated": (74, 28, 27, 16)},
             ("pretrained", "two_lead"): {"clean": (72, 41, 29, 25), "noisy": (70, 48, 31, 64), "gated": (70, 32, 31, 22)}}
    lines, ok = [], True
    for (arm, mode), exp in paper.items():
        for cond, want in exp.items():
            pc = "clean" if cond == "clean" else "noisy"
            frames = {}
            for m, L in ((("lead_II", "II"), ("lead_V2", "V2")) if mode == "union" else (("two_lead", "II"), ("two_lead", "V2"))):
                f = pd.read_csv(P.grid_file("R1_published", arm, m, pc), usecols=["record", "anchor_samp", f"pred_{L}_mV"])
                frames[L] = f.rename(columns={f"pred_{L}_mV": "pred"})
            _, _, met = score(learned_rows(frames, cond, M if cond == "gated" else None))
            got = (int(met["TP"]), int(met["FP"]), int(met["FN"]), int(met["Ignored"]))
            ok &= got == want
            lines.append(f"{'PASS' if got == want else 'FAIL'}  control R1 {arm} {mode} {cond} (.atr gate from the step-5 table): {got} vs paper {want}")
    return ok, lines


# ------------------------------------------------------------------ T5 and per-record alarm components
def alarms_all(V):
    Mg = memos(V["noisy42"], "anchor_grid", "sqi_grid")
    Mn = memos(V["noisy42"], "anchor_nk", "sqi_nk")
    res, comps = [], []
    for cond in ("clean", "noisy", "gated"):
        for pipe in LEARNED:
            for s in SEEDS:
                al, c, met = score(learned_rows(learned_frames(pipe, s, COND[cond]), cond, Mg if cond == "gated" else None))
                al.to_csv(OUT / "alarms" / f"{pipe}_{s}_{cond}.csv", index=False)
                res.append(dict(pipeline=pipe, seed=s, condition=cond, **met))
                comps.append(c.assign(pipeline=pipe, seed=s, condition=cond))
                log("alarms", pipe, s, cond, met["TP"], met["FP"], met["FN"])
        for name, col in NK.items():
            d = V[COND[cond]]
            rows = []
            for (rec, L), g in d[d.nk_window].groupby(["record", "lead"]):
                g = g.sort_values("k")
                rows += fsm(rec, L, g.anchor_nk, g[col], Mn[(rec, L)] if cond == "gated" else None)
            al, c, met = score(rows)
            al.to_csv(OUT / "alarms" / f"{name}_{cond}.csv", index=False)
            res.append(dict(pipeline=name, seed="-", condition=cond, **met))
            comps.append(c.assign(pipeline=name, seed="-", condition=cond))
            log("alarms", name, cond, met["TP"], met["FP"], met["FN"])
    R = pd.DataFrame(res)
    C = pd.concat(comps, ignore_index=True)
    R.to_csv(OUT / "T5_alarms_per_seed.csv", index=False)
    C.to_csv(OUT / "alarm_components_per_record.csv", index=False)
    return R, C


# ------------------------------------------------------------------ T1 and per-record regression
def labels():
    O = pd.read_csv(ORACLE).query("condition == 'clean'")
    O["k"] = O.window_start_samp // WIN
    return O[["record", "k", "oracle_II_mV", "oracle_V2_mV"]]


def reg_frames(V, Lb, cond, gated=False):
    """{(pipeline, seed, lead): frame record, k, y, p, gate} on set A windows, masked pairs out."""
    out = {}
    g = V["noisy42"][["record", "lead", "k", "sqi_grid"]]
    for L in LEADS:
        base = Lb[["record", "k", f"oracle_{L}_mV"]].rename(columns={f"oracle_{L}_mV": "y"})
        base = base[base.record.isin(RECS) & ~base.record.map(lambda r: (r, L) in MASK)]
        base = base.merge(g[g.lead == L].drop(columns="lead"), on=["record", "k"], how="left")
        base["gate"] = np.isfinite(base.sqi_grid) & (base.sqi_grid >= THR)
        for pipe in LEARNED:
            for s in SEEDS:
                mode = dict((l, m) for m, l in LEARNED[pipe])[L]
                f = pd.read_csv(S6 / f"predictions_{COND[cond]}" / f"FIN_{pipe}_{mode}_{s}__grid.csv.gz", usecols=["record", "anchor_samp", f"pred_{L}_mV"])
                f["k"] = (f.anchor_samp - (WIN - 1)) // WIN
                out[(pipe, s, L)] = base.merge(f[["record", "k", f"pred_{L}_mV"]].rename(columns={f"pred_{L}_mV": "p"}), on=["record", "k"], how="left", validate="1:1")
        for name, col in NK.items():
            v = V[COND[cond]]
            v = v[(v.lead == L) & v.nk_window][["record", "k", col]].rename(columns={col: "p"})
            out[(name, "-", L)] = base.merge(v, on=["record", "k"], how="left", validate="1:1")
    if gated:
        out = {k: f[f.gate] for k, f in out.items()}
    return out


def t1(F):
    rows = []
    for (pipe, s, L), f in F.items():
        q = f[f.y.notna()]
        rows.append(dict(pipeline=pipe, seed=s, lead=L, windows=len(q), with_value=int(q.p.notna().sum()), coverage_pct=100 * q.p.notna().mean(), **P.reg(q.y, q.p)))
    return pd.DataFrame(rows)


def reg_sums(f, recs):
    rows = []
    for r in recs:
        q = f[(f.record == r) & f.y.notna() & f.p.notna()]
        y, p = q.y.to_numpy(float), q.p.to_numpy(float)
        e = p - y
        rows.append([len(y), np.abs(e).sum()])
    return np.array(rows, float)


# ------------------------------------------------------------------ T7
def t7(C, Fg):
    pairs = [("L2", "L1"), ("L2", "NK-FP"), ("L1", "NK-FP"), ("L2", "NK-PR"), ("L1", "L1sup"), ("L2", "L2sup")]
    idx15 = np.random.default_rng(BSEED).integers(0, len(RECS), size=(NB, len(RECS)))
    cg = C[C.condition == "gated"]

    def acomp(pipe):
        seeds = SEEDS if pipe in LEARNED else ("-",)
        return [cg[(cg.pipeline == pipe) & (cg.seed == s)].set_index("record").loc[RECS][["TP", "FP", "FN"]].to_numpy(float) for s in seeds]

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
            rows.append(dict(comparison=f"{a} - {b}", metric=f"d{m} (union)", value_a=float(full[m][0]), value_b=float(full[m][1]), delta=float(full[m][0] - full[m][1]), ci_lo=float(lo), ci_hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0), n_records=len(RECS)))
        for L in LEADS:
            recs = [r for r in RECS if (r, L) not in MASK]
            idx = np.random.default_rng(BSEED).integers(0, len(recs), size=(NB, len(recs)))

            def mae_sets(pipe):
                seeds = SEEDS if pipe in LEARNED else ("-",)
                return [reg_sums(Fg[(pipe, s, L)], recs) for s in seeds]
            SA, SB = mae_sets(a), mae_sets(b)
            fa = np.mean([x.sum(0)[1] / x.sum(0)[0] for x in SA])
            fb = np.mean([x.sum(0)[1] / x.sum(0)[0] for x in SB])
            ba = np.mean([x[idx].sum(1)[:, 1] / x[idx].sum(1)[:, 0] for x in SA], 0)
            bb = np.mean([x[idx].sum(1)[:, 1] / x[idx].sum(1)[:, 0] for x in SB], 0)
            lo, hi = np.nanpercentile(ba - bb, [2.5, 97.5])
            rows.append(dict(comparison=f"{a} - {b}", metric=f"dMAE {L} (mV)", value_a=float(fa), value_b=float(fb), delta=float(fa - fb), ci_lo=float(lo), ci_hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0), n_records=len(recs)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T7_paired_bootstrap.csv", index=False)
    return T


# ------------------------------------------------------------------ RESULTS_FINAL.md
def pm(x, d=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}"


def md_tier1(T1t, R, T7t, ctl_lines):
    out = ["<!-- tier1 -->", "## Tier 1 (T1, T5, T7)", "",
           f"Generated {time.strftime('%F %T')} by `src/scripts/final/fin_step7_tier1.py` (VM). 15 records, 101 ST-deviation episodes (MON: 99, as published). "
           "Learned rows: mean ± SD (ddof 1) over training seeds S1–S3; per-seed values in `Tasks/final/runs/step7/`. Noise: NSTDB `em`, seed 42. "
           "Gate: the paper's SQI from rodrigues2021 peaks on the noisy signal, 6 dB.", "",
           "Control (published R1 predictions through this alarm path, `.atr` gate): " + ("all 6 conditions reproduce the paper." if all(l.startswith("PASS") for l in ctl_lines) else "FAILED"), ""]
    out += ["### T1. Regression, all windows (clean)", "",
            "Set A: the 10 s grid windows, label = median `.16a` `ee` of the window's beats; masked pairs excluded; each pipeline on its own windows with a value (coverage = share of labelled windows with a value). "
            "Slope = OLS of prediction on label.", "",
            "| lead | pipeline | coverage % | n | MAE (mV) | bias (mV) | RMSE (mV) | slope | r | p100 % | R² |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for L in LEADS:
        for pipe in ("L1", "L2", "NK-PR", "NK-FP"):
            q = T1t[(T1t.pipeline == pipe) & (T1t.lead == L)]
            out.append(f"| {L} | {pipe} | {pm(q.coverage_pct, 1)} | {pm(q.n, 0)} | {pm(q.MAE, 4)} | {pm(q.bias, 4)} | {pm(q.RMSE, 4)} | {pm(q.slope)} | {pm(q.r)} | {pm(q.p100_pct, 1)} | {pm(q.R2)} |")
    out += ["", "### T5. Alarms, main table", "",
            "Union of II and V2 alarms, FULL scope. Duration Se / +P = time sensitivity / time PPV; redundant = Ignored (extra alarms inside an already-detected episode); "
            "false-alarm hours = time in false alarms. MON copied from the paper (episode counts only).", "",
            "| condition | pipeline | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | duration Se | duration +P | redundant |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for cond in ("clean", "noisy", "gated"):
        tp, fp, fn = MON[cond]
        p_, r_ = tp / (tp + fp), tp / (tp + fn)
        out.append(f"| {cond} | MON | {tp} | {fp} | {fn} | {p_:.3f} | {r_:.3f} | {2 * p_ * r_ / (p_ + r_):.3f} | – | – | – | – | n/a |")
        for pipe in ("NK-PR", "NK-FP", "L1", "L2"):
            q = R[(R.pipeline == pipe) & (R.condition == cond)]
            out.append(f"| {cond} | {pipe} | {pm(q.TP, 1)} | {pm(q.FP, 1)} | {pm(q.FN, 1)} | {pm(q.precision)} | {pm(q.recall)} | {pm(q.F1)} | {pm(q.IoU_whole_record)} | {pm(q.false_hours, 1)} | {pm(q.sensitivity)} | {pm(q.PPV_time)} | {pm(q.Ignored, 1)} |")
    out += ["", "### T7. Paired differences (gated, seed 42)", "",
            f"Record-clustered paired bootstrap, {NB:,} resamples (seed {BSEED}); training seeds averaged inside each resample. dMAE: set A windows the gate keeps "
            "(gate at the learned anchor, same for both), noisy predictions against the clean labels, masked pairs out, each pipeline on its own windows. * = 95 % CI excludes 0.", "",
            "| comparison | metric | A | B | A − B | 95 % CI |", "|---|---|--:|--:|--:|---|"]
    for t in T7t.itertuples():
        d = 4 if "MAE" in t.metric else 3 if "F1" in t.metric else 1
        out.append(f"| {t.comparison} | {t.metric} | {t.value_a:.{d}f} | {t.value_b:.{d}f} | {t.delta:+.{d}f}{' *' if t.excludes_zero else ''} | [{t.ci_lo:+.{d}f}, {t.ci_hi:+.{d}f}] |")
    out += ["<!-- /tier1 -->"]
    block = "\n".join(out) + "\n"
    head = "# Final thesis results\n\nFrozen setup: `docs/instructions_final_results.md` and `docs/DECISIONS.md`. Provenance: `Tasks/final/runs/step6/provenance.csv` (VM). Run log: `docs/reports/2026-11-21_vm_steps4_7.md`.\n\n"
    old = MD.read_text() if MD.exists() else head
    if "<!-- tier1 -->" in old:
        a, rest = old.split("<!-- tier1 -->", 1)
        tail = rest.split("<!-- /tier1 -->", 1)[1].lstrip("\n")
        old = a + block + ("\n" + tail if tail else "")  # keep one blank line before the next block
    else:
        old = old.rstrip("\n") + "\n\n" + block
    MD.write_text(old)


TEX = DOCS / "thesis_tables"


def tpm(x, d=3):
    x = np.asarray(x, float)
    f = (lambda v: f"{v:.{d}f}".replace("-", "$-$"))
    return f"{f(x.mean())} $\\pm$ {x.std(ddof=1):.{d}f}" if len(x) > 1 else f(x[0])


def tex_write(name, caption, label, colspec, header, body, note):
    """One standalone booktabs table per file: \\input it; needs \\usepackage{booktabs}."""
    TEX.mkdir(parents=True, exist_ok=True)
    lines = [f"% {name}: generated {time.strftime('%F %T')} by src/scripts/final/fin_step7_tier1.py (VM). Needs \\usepackage{{booktabs}}.",
             "\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for row in body:
        lines.append("\\midrule" if row == "MIDRULE" else row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", f"\\par\\smallskip\\footnotesize {note}", "\\end{table}", ""]
    (TEX / f"{name}.tex").write_text("\n".join(lines))


def tex_tier1(T1t, R, T7t):
    body = []
    for i, L in enumerate(LEADS):
        if i:
            body.append("MIDRULE")
        for pipe in ("L1", "L2", "NK-PR", "NK-FP"):
            q = T1t[(T1t.pipeline == pipe) & (T1t.lead == L)]
            body.append(f"{L} & {pipe} & {tpm(q.coverage_pct, 1)} & {tpm(q.MAE, 4)} & {tpm(q.bias, 4)} & {tpm(q.RMSE, 4)} & {tpm(q.slope)} & {tpm(q.r)} & {tpm(q.p100_pct, 1)} & {tpm(q.R2)}")
    tex_write("T1_regression_clean", "ST-deviation regression on all 10\\,s windows, clean signal (T1).", "tab:t1_regression",
              "llrrrrrrrr", "Lead & Pipeline & Coverage (\\%) & MAE (mV) & Bias (mV) & RMSE (mV) & Slope & $r$ & p100 (\\%) & $R^2$", body,
              "10\\,s grid windows, label = median expert ST deviation of the window's beats; masked record--lead pairs excluded; 15 records. "
              "Each pipeline on its own windows with a value. Slope: OLS of prediction on label. p100: share of windows with $|$error$|$ > 0.1\\,mV. "
              "L1, L2: mean $\\pm$ SD over three training seeds.")
    body = []
    for i, cond in enumerate(("clean", "noisy", "gated")):
        if i:
            body.append("MIDRULE")
        tp, fp, fn = MON[cond]
        p_, r_ = tp / (tp + fp), tp / (tp + fn)
        body.append(f"{cond.capitalize()} & MON & {tp} & {fp} & {fn} & {p_:.3f} & {r_:.3f} & {2 * p_ * r_ / (p_ + r_):.3f} & -- & -- & -- & -- & n/a")
        for pipe in ("NK-PR", "NK-FP", "L1", "L2"):
            q = R[(R.pipeline == pipe) & (R.condition == cond)]
            body.append(f" & {pipe} & {tpm(q.TP, 1)} & {tpm(q.FP, 1)} & {tpm(q.FN, 1)} & {tpm(q.precision)} & {tpm(q.recall)} & {tpm(q.F1)} & {tpm(q.IoU_whole_record)} & "
                        f"{tpm(q.false_hours, 1)} & {tpm(q.sensitivity)} & {tpm(q.PPV_time)} & {tpm(q.Ignored, 1)}")
    tex_write("T5_alarms_main", "ST-deviation episode alarms: clean, noisy and gated (T5).", "tab:t5_alarms",
              "llrrrrrrrrrrr", "Condition & Pipeline & TP & FP & FN & Precision & Recall & F1 & IoU & FA (h) & Dur.\\ Se & Dur.\\ +P & Redundant", body,
              "Union of lead II and V2 alarms, PROSE FSM, 101 ST-deviation episodes in 15 records (MON: 99 episodes, copied from Abdelazez et al.; episode counts only). "
              "Noisy: NSTDB em, seed 42; gated: SQI from rodrigues2021 R-peaks, 6\\,dB. FA: false-alarm hours. Dur.\\ Se / +P: duration sensitivity / positive predictivity. "
              "Redundant: extra alarms inside an already-detected episode. L1, L2: mean $\\pm$ SD over three training seeds.")
    body = []
    for t in T7t.itertuples():
        d = 4 if "MAE" in t.metric else 3 if "F1" in t.metric else 1
        f = (lambda v, s="": f"{v:{s}.{d}f}".replace("-", "$-$"))
        m = t.metric.replace("dF1", "$\\Delta$F1").replace("dFP", "$\\Delta$FP").replace("dMAE", "$\\Delta$MAE")
        body.append(f"{t.comparison.replace(' - ', ' $-$ ')} & {m} & {f(t.value_a)} & {f(t.value_b)} & {f(t.delta, '+')}{'$^{*}$' if t.excludes_zero else ''} & [{f(t.ci_lo, '+')}, {f(t.ci_hi, '+')}]")
    tex_write("T7_paired_differences", "Paired differences between pipelines, gated, noise seed 42 (T7).", "tab:t7_paired",
              "llrrrl", "Comparison & Metric & A & B & A $-$ B & 95\\,\\% CI", body,
              f"Record-clustered paired bootstrap, {NB:,} resamples; training seeds averaged inside each resample. "
              "$\\Delta$MAE: windows the gate keeps, noisy predictions against clean labels, masked pairs excluded. $^{*}$: 95\\,\\% CI excludes zero.")


def main():
    (OUT / "alarms").mkdir(parents=True, exist_ok=True)
    for m in ("DONE_tier1", "FAILED_tier1"):
        (OUT / m).unlink(missing_ok=True)
    for need in (S6 / "DONE_step6", S5 / "DONE_step5"):
        if not need.exists():
            sys.exit(f"missing {need}")
    V = {c: pd.read_csv(S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    ok, ctl = control(V["noisy42"])
    (OUT / "CONTROL.txt").write_text("\n".join(ctl) + "\n")
    log("control", *ctl)
    if not ok:
        (OUT / "FAILED_tier1").write_text("control failed\n")
        return False
    R, C = alarms_all(V)
    Lb = labels()
    F = reg_frames(V, Lb, "clean")
    T1t = t1(F)
    T1t.to_csv(OUT / "T1_regression_per_seed.csv", index=False)
    pd.concat([f.assign(pipeline=k[0], seed=k[1], lead=k[2]) for k, f in F.items()]).groupby(["pipeline", "seed", "lead", "record"]).apply(
        lambda q: pd.Series(dict(n=int((q.y.notna() & q.p.notna()).sum()), MAE=float((q.p - q.y).abs().mean())))).reset_index().to_csv(OUT / "T1_regression_per_record.csv", index=False)
    Fg = reg_frames(V, Lb, "gated", gated=True)
    T7t = t7(C, Fg)
    md_tier1(T1t, R, T7t, ctl)
    tex_tier1(T1t, R, T7t)
    (OUT / "DONE_tier1").write_text(time.strftime("%F %T") + "\n")
    log("done")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
