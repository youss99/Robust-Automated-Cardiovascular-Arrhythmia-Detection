#!/usr/bin/env python3
"""Final runs, step 7 appendix tables (user, 2026-11-22). One stage per call.

hyperparams  docs/thesis_tables/appendix_hyperparams.tex: pre-training and fine-tuning settings read from params.txt, one column per model type
             (pre-training: single-lead backbones II and V2, two-lead backbone; fine-tuning: L1, L2, L1-sup, L2-sup). Every fine-tuning run's
             params.txt is parsed (90); a value is printed only if it is the same in every run of that type (seed: the per-type set). The two-lead
             backbone's params.txt is not on the VM: its text, copied from Narval on 2026-11-22 (read-only check), is in
             docs/reports/2026-11-22_narval_ii_v2_rebuilt_params.txt; it must differ from ii_lossfix only in the lead, run name and paths.
per_record   docs/thesis_tables/appendix_per_record.tex: per record and lead, gated (artifact seed 42, SQI 6 dB from rodrigues2021 peaks), the
             GS episodes and TP / FP / FN of Fiducial-FP (NK-FP), L1 (S1) and L2 (S1), each lead scored on its own episodes (own-lead scope:
             that lead's alarms against that lead's episodes; the scorer's own_II / own_V2), plus the MAE on the windows the gate keeps
             (artifact predictions against baseline labels, as T7). Alarm lists: tier 1's stored gated files. Masked pairs shown as masked.
             Checks: the same alarm files scored in the FULL scope reproduce tier 1's per-record components; TP + FN = the lead's GS
             episodes in every row; MAE pooled over records equals T7's A / B values.
Output: Tasks/final/runs/step7_appendix/ (CSV, CHECKS.txt) and the two .tex files.
pr_loss      docs/thesis_tables/appendix_fiducial_pr_loss.tex: Fiducial-PR beat and window loss (see pr_loss()).
Usage: <repo .venv>/bin/python -u -W ignore fin_step7_appendix.py hyperparams|per_record|pr_loss"""
import ast
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other final drivers
import fin_step7_tier1 as T  # noqa: E402
import fin_step7_t10_fig1 as X  # noqa: E402

P, K = T.P, T.K
REPO = P.REPO
OUT = Path("/home/student/GIT/Tasks/final/runs/step7_appendix")
SCRIPT = "src/scripts/final/fin_step7_appendix.py"
FT = REPO / "results/fine-tune/saved_models/patient-segment-regression"
BB = REPO / "results/pre-train/saved_models/unified"


def tex_write(name, caption, label, colspec, header, body, note, size="\\small"):
    X.tex_write(name, caption, label, colspec, header, body, note)
    f = T.TEX / f"{name}.tex"
    s = f.read_text().replace("src/scripts/final/fin_step7_t10_fig1.py", SCRIPT, 1).replace("fin_step7_t10_fig1.py", SCRIPT, 1)
    f.write_text(s.replace("\\small", size, 1))


def parse(p):
    s = re.sub(r"<([\w.]+): [^>]*>", r"'\1'", Path(p).read_text().strip())
    return {k.arg: ast.literal_eval(k.value) for k in ast.parse(s, mode="eval").body.keywords}


# ------------------------------------------------------------------ hyperparams
ROWS = [  # (section, label, key in pre-training params, key in fine-tuning params, formatter)
    ("Architecture", "Encoder", "model", "model", None),
    ("Architecture", "Transformer layers / heads", ("n_layers", "n_heads"), ("n_layers", "n_heads"), None),
    ("Architecture", "Model dim.\\ / feed-forward dim.", ("d_model", "d_ff"), ("d_model", "d_ff"), None),
    ("Architecture", "Patch length / stride (samples)", ("patch_len", "stride"), ("patch_len", "stride"), None),
    ("Architecture", "Dropout / head dropout", ("dropout", "head_dropout"), ("dropout", "head_dropout"), None),
    ("Architecture", "Shared patch embedding", "shared_embedding", "shared_embedding", None),
    ("Architecture", "Normalisation (RevIN mode) / scaler", ("revin_mode", "scaler"), ("revin_mode", "scaler"), None),
    ("Architecture", "Head", "head_type", ("head_type", "class_token"), None),
    ("Input", "Leads", "custom_lead_selection", "custom_lead_selection", None),
    ("Input", "Context points", "context_points", "context_points", None),
    ("Objective", "Mask ratio", "mask_ratio", None, None),
    ("Objective", "Noise masking / zero-noise mixing", ("noise_masking", "zero_noise_mixing"), None, None),
    ("Objective", "Loss", None, "regression_loss", None),
    ("Objective", "Data augmentation", "data_augmentation", "data_augmentation", None),
    ("Optimisation", "Optimiser", "opt", "opt", None),
    ("Optimisation", "Betas / eps", ("opt_betas", "opt_eps"), ("opt_betas", "opt_eps"), None),
    ("Optimisation", "Learning rate", "lr", "lr", None),
    ("Optimisation", "Layer-wise LR decay", "layer_decay", "layer_decay", None),
    ("Optimisation", "Weight decay", "weight_decay", "weight_decay", None),
    ("Optimisation", "Scheduler", "scheduler", "scheduler", None),
    ("Optimisation", "Warm-up epochs / warm-up LR / min.\\ LR", ("warmup_epochs", "warmup_lr", "min_lr"), ("warmup_epochs", "warmup_lr", "min_lr"), None),
    ("Optimisation", "Batch size", "batch_size", "batch_size", None),
    ("Optimisation", "Epochs", "n_epochs_pretrain", ("n_epochs_finetune_head", "n_epochs_finetune"), None),
    ("Optimisation", "SAM", "sam", "sam", None),
    ("Optimisation", "Model selection", None, "model_selection_metric", None),
    ("Initialisation", "Backbone", None, "pretrained_model_path", None),
    ("Initialisation", "Seed", None, "seed", None),
]


def fmt(v):
    if isinstance(v, str):
        v = v.split(".")[-1] if re.match(r"^[A-Z]\w*\.\w+$", v) else v
        return v.replace("_", "\\_")
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return str(v)


def value(runs, key):
    """runs: list of param dicts; key: str, tuple or None. One string if constant across runs, else the per-run set."""
    if key is None:
        return "--"
    keys = key if isinstance(key, tuple) else (key,)
    vals = {" / ".join(fmt(r.get(k)) for k in keys) for r in runs}
    if len(vals) == 1:
        return vals.pop()
    return None


def hyperparams():
    pre = {"II": parse(BB / "patch_ecg_unified_ii_lossfix/params.txt"), "V2": parse(BB / "patch_ecg_unified_v2_lossfix/params.txt")}
    diff = sorted(k for k in pre["II"] if pre["II"][k] != pre["V2"].get(k))
    two = parse(T.DOCS / "reports/2026-11-22_narval_ii_v2_rebuilt_params.txt")
    diff2 = sorted(k for k in set(two) | set(pre["II"]) if two.get(k) != pre["II"].get(k))
    assert set(two) == set(pre["II"]) and diff2 == ["custom_lead_selection", "model_name", "root_path", "save_path"], diff2
    ft = {}
    for p in sorted(FT.glob("FIN_*/params.txt")):
        typ = re.match(r"FIN_(L1sup|L2sup|L1|L2)_", p.parent.name).group(1)
        ft.setdefault(typ, []).append(parse(p))
    assert {t: len(v) for t, v in ft.items()} == {"L1": 30, "L2": 15, "L1sup": 30, "L2sup": 15}, {t: len(v) for t, v in ft.items()}
    cols = ["pre_single", "pre_two", "L1", "L2", "L1sup", "L2sup"]
    table, lines = [], [f"pre-training II vs V2 backbone params.txt differ only in: {diff}",
                        f"PASS  two-lead backbone (Narval copy) vs ii_lossfix: same 77-key set, differ only in {diff2}"]
    for sec, lab, kp, kf, _ in ROWS:
        row = dict(section=sec, parameter=lab)
        if kp is None:
            row["pre_single"] = "--"
        elif kp == "custom_lead_selection":
            row["pre_single"] = "II or V2 (one backbone each)"
        else:
            v = value([pre["II"], pre["V2"]], kp)
            assert v is not None, (lab, kp)
            row["pre_single"] = v
        row["pre_two"] = "II, V2" if kp == "custom_lead_selection" else ("--" if kp is None else value([two], kp))
        for t in ("L1", "L2", "L1sup", "L2sup"):
            runs = ft[t]
            if kf == "seed":
                s = sorted({r["seed"] for r in runs})
                row[t] = f"{s[0]}--{s[-1]} (3 seeds $\\times$ 5 folds)"
                lines.append(f"{t} seeds: {s}")
            elif kf == "pretrained_model_path":
                b = {Path(r["pretrained_model_path"]).parent.name if r["pretrained_model_path"] else "None" for r in runs}
                row[t] = ", ".join(sorted(x.replace("patch_ecg_unified_", "").replace("_", "\\_") for x in b)) if b != {"None"} else "none (random init.)"
            elif kf == "custom_lead_selection":
                row[t] = ", ".join(sorted({r["custom_lead_selection"] for r in runs})).replace(",", ", ").replace(",  ", ", ")
                row[t] = "II or V2 (one model each)" if t in ("L1", "L1sup") else "II, V2"
            else:
                v = value(runs, kf)
                if v is None:
                    raise SystemExit(f"{t}: {lab} not constant")
                row[t] = v
        table.append(row)
    H = pd.DataFrame(table)
    OUT.mkdir(parents=True, exist_ok=True)
    H.to_csv(OUT / "appendix_hyperparams.csv", index=False)
    ft_same = all(H.loc[i, "L1"] == H.loc[i, t] for i in H.index for t in ("L2", "L1sup", "L2sup") if H.loc[i, "parameter"] not in ("Leads", "Backbone", "Seed"))
    lines.append(f"{'PASS' if ft_same else 'INFO'}  fine-tuning settings identical across L1, L2, L1-sup, L2-sup except leads, backbone and seed: {ft_same}")
    body, sec0 = [], None
    for r in H.itertuples():
        if r.section != sec0:
            if sec0 is not None:
                body.append("MIDRULE")
            body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{r.section}}}}}")
            sec0 = r.section
        body.append(f"{r.parameter} & {r.pre_single} & {r.pre_two} & {r.L1} & {r.L2} & {r.L1sup} & {r.L2sup}")
    tex_write("appendix_hyperparams", "Pre-training and fine-tuning settings (from each run's \\texttt{params.txt}).", "tab:appendix_hyperparams",
              "p{3.6cm}p{2.1cm}p{1.4cm}p{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}",
              "Parameter & \\multicolumn{2}{c}{Pre-training (Unified v2)} & \\multicolumn{4}{c}{Fine-tuning (LTST)} \\\\\n & Single-lead & Two-lead & L1 & L2 & L1-sup & L2-sup",
              body,
              "Values read from \\texttt{params.txt}: the two single-lead backbones (\\texttt{ii\\_lossfix}, \\texttt{v2\\_lossfix}; identical except the lead) and all 90 "
              "fine-tuning runs; a value is shown when it is the same in every run of the column; --: not shown for that stage. Fine-tuning epochs: head only, then full network. "
              "Seeds: S1 = 200, S2 = 300, S3 = 400, plus fold $-$ 1. Two-lead backbone (\\texttt{ii\\_v2\\_rebuilt}): \\texttt{params.txt} read on the cluster; it differs from the single-lead backbones "
              "only in the leads. It was trained on 2026-11-06, before the pre-training loss fix (commit b8f688d); the fix does not change "
              "pre-training with two or more leads, and the backbone check (step 0b) passed. Windows: Table~\\ref{tab:t0_pretraining}.", size="\\scriptsize")
    (OUT / "HYPERPARAMS_CHECKS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return True


# ------------------------------------------------------------------ per_record
SYS = [("Fiducial-FP", "NK-FP", "-", "NK-FP_gated.csv"), ("L1", "L1", "S1", "L1_S1_gated.csv"), ("L2", "L2", "S1", "L2_S1_gated.csv")]


def per_record():
    A = T.OUT / "alarms"
    C = pd.read_csv(T.OUT / "alarm_components_per_record.csv").query("condition == 'gated'")
    T7 = pd.read_csv(T.OUT / "T7_paired_bootstrap.csv")
    V = {c: pd.read_csv(T.S5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    Fg = T.reg_frames(V, T.labels(), "gated", gated=True)
    gs = T.GS
    rows, lines, ok = [], [], True
    for name, key, seed, f in SYS:
        al = pd.read_csv(A / f)
        # control: FULL scope from the stored file equals tier 1's per-record components
        c_full = T.per_record_components(T.gs_scope(gs, "FULL"), K.scope_filter(al, "FULL"), T.DUR, "FULL", T.MASK, T.RECS)
        ref = C[(C.pipeline == key) & (C.seed == seed)].set_index("record").loc[T.RECS]
        same = np.array_equal(c_full.set_index("record").loc[T.RECS][["TP", "FP", "FN", "Ignored"]].to_numpy(), ref[["TP", "FP", "FN", "Ignored"]].to_numpy())
        ok &= same
        lines.append(f"{'PASS' if same else 'FAIL'}  {name}: stored gated alarms, FULL scope, equal tier 1's per-record components")
        for L in T.LEADS:
            c = T.per_record_components(T.gs_scope(gs, f"own_{L}"), K.scope_filter(al, f"own_{L}"), T.DUR, f"own_{L}", T.MASK, T.RECS).set_index("record")
            fr = Fg[(key, seed, L)]
            for rec in T.RECS:
                q = fr[(fr.record == rec) & fr.y.notna() & fr.p.notna()]
                rows.append(dict(system=name, record=rec, lead=L, masked=(rec, L) in T.MASK, gs_episodes=int(((gs.record == rec) & (gs.lead == L)).sum()),
                                 TP=int(c.loc[rec, "TP"]), FP=int(c.loc[rec, "FP"]), FN=int(c.loc[rec, "FN"]), n_windows=len(q),
                                 MAE=float((q.p - q.y).abs().mean()) if len(q) else np.nan, abs_err_sum=float((q.p - q.y).abs().sum())))
    R = pd.DataFrame(rows)
    um = R[~R.masked]
    tpfn = bool(((um.TP + um.FN) == um.gs_episodes).all())
    ok &= tpfn
    lines.append(f"{'PASS' if tpfn else 'FAIL'}  TP + FN = the lead's GS episodes in every unmasked row")
    nomask = bool((R[R.masked][["TP", "FP", "FN", "n_windows"]] == 0).all().all())
    ok &= nomask
    lines.append(f"{'PASS' if nomask else 'FAIL'}  masked rows have no alarms, episodes or windows")
    for name, key, seed, _ in SYS:
        for L in T.LEADS:
            g = R[(R.system == name) & (R.lead == L)]
            pooled = g.abs_err_sum.sum() / g.n_windows.sum()
            t7 = T7[T7.metric == f"dMAE {L} (mV)"]
            tv = t7[t7.comparison.str.startswith(key + " -")].value_a.iloc[0] if (t7.comparison.str.startswith(key + " -")).any() else t7[t7.comparison.str.endswith("- " + key)].value_b.iloc[0]
            same = key == "NK-FP" or True
            # T7 averages the three training seeds for L1 / L2; S1 alone is compared with T10c (A4a), which is S1
            if key == "NK-FP":
                good = np.isclose(pooled, tv, atol=5e-5)
                lines.append(f"{'PASS' if good else 'FAIL'}  {name} {L}: pooled gated MAE {pooled:.4f} equals T7 {tv:.4f}")
                ok &= good
    GM = pd.read_csv(X.OUT / "T10_A4a_gated_mae.csv").query("gate == 'rodrigues2021'")
    for name, key, seed, _ in SYS[1:]:
        for L in T.LEADS:
            g = R[(R.system == name) & (R.lead == L)]
            pooled = g.abs_err_sum.sum() / g.n_windows.sum()
            ref = GM[(GM.pipeline == key) & (GM.lead == L)].MAE_gated.iloc[0]
            good = np.isclose(pooled, ref, atol=5e-5)
            ok &= good
            lines.append(f"{'PASS' if good else 'FAIL'}  {name} (S1) {L}: pooled gated MAE {pooled:.4f} equals T10c (S1) {ref:.4f}")
    R5 = pd.read_csv(T.OUT / "T5_alarms_per_seed.csv").query("condition == 'gated'")
    for name, key, seed, _ in SYS:
        g = R[(R.system == name) & ~R.masked]
        u = R5[(R5.pipeline == key) & (R5.seed == seed)].iloc[0]
        got, want = (int(g.TP.sum()), int(g.FP.sum()), int(g.FN.sum())), (int(u.TP), int(u.FP), int(u.FN))
        ok &= got == want
        lines.append(f"{'PASS' if got == want else 'FAIL'}  {name}: per-lead totals {got} equal the gated union totals {want} (the table note says so)")
    OUT.mkdir(parents=True, exist_ok=True)
    R.drop(columns="abs_err_sum").to_csv(OUT / "appendix_per_record.csv", index=False)
    (OUT / "PER_RECORD_CHECKS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    body = []
    for i, L in enumerate(T.LEADS):
        if i:
            body.append("MIDRULE")
        for rec in T.RECS:
            g = R[(R.record == rec) & (R.lead == L)].set_index("system")
            if g.masked.iloc[0]:
                body.append(f"{rec} & {L} & \\multicolumn{{13}}{{l}}{{\\textit{{masked (excluded from scoring)}}}}")
                continue
            cells = [rec, L, str(int(g.gs_episodes.iloc[0]))]
            for name, *_ in SYS:
                r = g.loc[name]
                cells += [str(r.TP), str(r.FP), str(r.FN), f"{r.MAE:.3f}" if np.isfinite(r.MAE) else "--"]
            body.append(" & ".join(cells))
    body.append("MIDRULE")
    tot = ["Total", "", str(int(R[~R.masked].groupby(['record', 'lead']).gs_episodes.first().sum()))]
    for name, *_ in SYS:
        g = R[(R.system == name) & ~R.masked]
        tot += [str(int(g.TP.sum())), str(int(g.FP.sum())), str(int(g.FN.sum())), "--"]
    body.append(" & ".join(tot))
    tex_write("appendix_per_record", "Per-record results, gated (artifact condition, noise seed 42, SQI gate 6\\,dB).", "tab:appendix_per_record",
              "llr" + "rrrr" * 3,
              "Record & Lead & GS & \\multicolumn{4}{c}{Fiducial-FP} & \\multicolumn{4}{c}{L1 (S1)} & \\multicolumn{4}{c}{L2 (S1)} \\\\\n"
              "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\\cmidrule(lr){12-15}\n & & & TP & FP & FN & MAE & TP & FP & FN & MAE & TP & FP & FN & MAE",
              body,
              "GS: gold-standard ischemic episodes on that lead (periods meeting the PROSE criteria on the annotated .16a ST deviation). TP/FP/FN: that lead's "
              "alarms scored against that lead's GS episodes (per-lead scope; the main tables score the union of both leads; here the per-lead totals equal the union totals, gated, S1). "
              "MAE (mV): windows kept by the SQI gate, estimates in the artifact condition against baseline labels. Masked record--lead pairs are excluded "
              "from all scoring.", size="\\scriptsize")
    return ok


# ------------------------------------------------------------------ pr_loss
def pr_loss():
    """Fiducial-PR (NK-PR) beat and window loss, from the stored rodrigues2021 delineations (baseline and artifact seed 42), 15 records,
    masked pairs out. Beat tests as c20b_paper_rule.paper_beats (lines 39 and 42): J+60 found and after R; P offset and R onset found,
    P offset >= 0, P offset < R onset < R. Windows as c13_fp_gap.segments: beat -> 10 s window by its J+60 sample, value when >= 5 usable beats.
    Controls: windows with a value equal step 5's coverage.csv (NKPR, every condition and lead); on s20011 (both leads, baseline) the usable
    test equals the beats where paper_beats on the real signal gives a finite ST and baseline."""
    import c13_fp_gap as C13
    import c20_finalists as B
    import c20b_paper_rule as PR
    import wfdb
    cov = pd.read_csv(T.S5 / "coverage.csv").query("pipeline == 'NKPR'")
    rows, lines, ok = [], [], True
    for cond, tag in (("clean", "raw"), ("noisy42", "noise")):
        for rec in T.RECS:
            for lead, L in ((0, "II"), (1, "V2")):
                if (rec, L) in T.MASK:
                    continue
                w = pd.read_csv(B.DEL / f"delin_{tag}_nofilter_rodrigues2021" / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv",
                                usecols=["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_plus_60ms"])
                rp, po, ro, j = (w[c].to_numpy(float) for c in ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_plus_60ms"])
                j_ok = ~np.isnan(j) & (rp < j)
                no_po, no_ro = j_ok & np.isnan(po), j_ok & ~np.isnan(po) & np.isnan(ro)
                pres = j_ok & ~np.isnan(po) & ~np.isnan(ro)
                order_bad = pres & ~((po >= 0) & (po < ro) & (ro < rp))
                use = pres & ~order_bad
                if cond == "clean" and rec == "s20011":
                    x = wfdb.rdrecord(str(T.T1.RECORDS_DIR / rec / rec)).p_signal[:, lead]
                    d = pd.read_csv(B.DEL / f"delin_raw_nofilter_rodrigues2021/lead_{lead}/waves_{rec}_lead{lead}.csv")
                    st, iso = PR.paper_beats(x, d, 1)
                    same = np.array_equal(np.isfinite(st) & np.isfinite(iso), use)
                    ok &= same
                    lines.append(f"{'PASS' if same else 'FAIL'}  s20011 {L} baseline: usable test equals paper_beats on the signal ({int(use.sum())} beats)")
                start, _ = C13.grid(rec, lead)
                idx = np.searchsorted(start, j[use].astype(np.int64), side="right") - 1
                cnt = np.bincount(idx[idx >= 0], minlength=len(start))[:len(start)]
                med_pr = float(np.median((ro - po)[use]) * 4) if use.any() else np.nan
                rows.append(dict(condition=cond, record=rec, lead=L, r_peaks=len(w), no_J60=int((~j_ok).sum()), no_P_offset=int(no_po.sum()), no_R_onset=int(no_ro.sum()),
                                 P_offset_not_before_R_onset=int((order_bad & (po >= ro)).sum()), other_order=int((order_bad & (po < ro)).sum()), usable=int(use.sum()),
                                 median_PR_ms=med_pr, windows=len(start), windows_value=int((cnt >= 5).sum()), windows_1_4=int(((cnt > 0) & (cnt < 5)).sum()),
                                 windows_0=int((cnt == 0).sum())))
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "fiducial_pr_loss_per_record.csv", index=False)
    G = R.groupby(["condition", "lead"]).sum(numeric_only=True)
    for (c, L), g in G.iterrows():
        ref = cov[(cov.condition == c) & (cov.lead == L)].iloc[0]
        same = int(g.windows_value) == int(ref.with_value) and int(g.windows) == int(ref.windows)
        ok &= same
        lines.append(f"{'PASS' if same else 'FAIL'}  {c} {L}: windows with a value {int(g.windows_value)} of {int(g.windows)} equal step 5 coverage {int(ref.with_value)} of {int(ref.windows)}")
        parts = int(g.no_J60 + g.no_P_offset + g.no_R_onset + g.P_offset_not_before_R_onset + g.other_order + g.usable)
        same = parts == int(g.r_peaks)
        ok &= same
        lines.append(f"{'PASS' if same else 'FAIL'}  {c} {L}: beat categories add up to the R-peaks ({parts} vs {int(g.r_peaks)})")
    (OUT / "PR_LOSS_CHECKS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    worst = R[R.condition == "clean"].assign(pct=lambda d: 100 * d.windows_value / d.windows).sort_values("pct").head(3)
    cols = [("clean", "II"), ("noisy42", "II"), ("clean", "V2"), ("noisy42", "V2")]

    def cell(c, L, num, den="r_peaks"):
        g = G.loc[(c, L)]
        return f"{int(g[num]):,} ({100 * g[num] / g[den]:.1f})".replace(",", "{,}")
    body = ["\\multicolumn{5}{l}{\\textit{Beats (detected R-peaks)}}",
            "R-peaks detected & " + " & ".join(f"{int(G.loc[k].r_peaks):,}".replace(",", "{,}") for k in cols)]
    for lab, key in (("No J+60 (R offset not found)", "no_J60"), ("P offset not found", "no_P_offset"), ("R onset not found", "no_R_onset"),
                     ("P offset not before R onset", "P_offset_not_before_R_onset"), ("\\textbf{Usable}", "usable")):
        body.append(f"{lab} & " + " & ".join(cell(c, L, key) for c, L in cols))
    body.append("Median PR-segment length, usable beats (ms) & " + " & ".join(f"{R[(R.condition == c) & (R.lead == L)].median_PR_ms.median():.0f}" for c, L in cols))
    body += ["MIDRULE", "\\multicolumn{5}{l}{\\textit{10\\,s windows}}",
             "Windows & " + " & ".join(f"{int(G.loc[k].windows):,}".replace(",", "{,}") for k in cols)]
    for lab, key in (("\\textbf{With a value} ($\\geq$\\,5 usable beats)", "windows_value"), ("1--4 usable beats (no value)", "windows_1_4"), ("No usable beat", "windows_0")):
        body.append(f"{lab} & " + " & ".join(cell(c, L, key, "windows") for c, L in cols))
    tex_write("appendix_fiducial_pr_loss", "Fiducial-PR: beats and 10\\,s windows lost at each step (rodrigues2021 R-peaks, NeuroKit DWT delineation).",
              "tab:appendix_fiducial_pr_loss", "lrrrr",
              "& \\multicolumn{2}{c}{Lead II} & \\multicolumn{2}{c}{Lead V2} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Baseline & Artifact & Baseline & Artifact",
              body,
              "Counts with percentages in brackets (beats: of detected R-peaks; windows: of all 10\\,s windows). No beat failed only on R onset $\\geq$ R (other order checks: 0). 15 records, masked record--lead pairs excluded; "
              "artifact: NSTDB em, seed 42. Each beat is counted once, at the first test it fails, in the order listed. A usable beat has J+60 after R, "
              "and a P offset before the R onset before R; its ST deviation is the median at J+60 minus the median of the PR segment. A window has a value "
              "when at least 5 usable beats have their J+60 inside it. Lowest share of windows with a value, baseline: "
              + "; ".join(f"{r.record} {r.lead} {r.pct:.0f}\\,\\%" for r in worst.itertuples()) + ".", size="\\small")
    return ok


if __name__ == "__main__":
    st = sys.argv[1]
    ok = {"hyperparams": hyperparams, "per_record": per_record, "pr_loss": pr_loss}[st]()
    (OUT / (f"DONE_{st}" if ok else f"FAILED_{st}")).write_text(time.strftime("%F %T") + "\n")
    sys.exit(0 if ok else 1)
